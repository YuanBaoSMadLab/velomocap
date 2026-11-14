"""Export SAM 3D Body as a chain of small ONNX graphs instead of one big one.

Why it is split
---------------
MHR (`mhr_model.pt`) contains a `for` loop over `pmi` that mutates
`global_skel_state` in place with `aten::index_copy_`. The traced `prim::Loop`
therefore carries no values, and torch's ONNX `prim_loop` lowering asserts
(`len(node_outputs)=1 != n_outputs=0`). That is a shape of the graph, not a
missing operator, so no symbolic can rescue it — every export variant fails the
same way.

MHR *on its own* converts cleanly (`tools/onnx/mhr_export.py`), so the way out is
to keep it out of the decoder graph and drive the chain from Rust:

    backbone -> init -> step0 -> pose -> [mhr] -> post
                       step1 -> pose -> [mhr] -> post
                          ...
                       step5 -> pose        (final body token)

Every graph is batch-1: SAM is a per-frame stage and fixing the batch removes
all of the dynamic-`expand` corner cases.

Files written to `--out`
------------------------
  sam3d_backbone.onnx  image, ray_cond -> embeddings, image_seq, augment_seq
  sam3d_init.onnx      bbox/camera/keypoints -> tokens, token_augment
  sam3d_step{i}.onnx   tokens, token_augment, image_seq, augment_seq -> ...
  sam3d_pose.onnx      tokens -> normed, pred, pred_cam
  sam3d_mhr.onnx       identity, model_params, face -> verts, skel_state
  sam3d_post.onnx      tokens, verts, skel_state, pred_cam, cameras -> tokens...
  sam3d_split.json     token layout + image size, for the Rust side
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

IMAGE_SIZE = 512


# ---------------------------------------------------------------------------
# model loading
# ---------------------------------------------------------------------------
def load_model(repo: str, ckpt: str, mhr: str, config: str):
    """Build `SAM3DBody` without pulling in the training stack."""
    sys.path.insert(0, repo)
    torch.set_grad_enabled(False)

    if "sam_3d_body" not in sys.modules:
        package = types.ModuleType("sam_3d_body")
        package.__path__ = [os.path.join(repo, "sam_3d_body")]
        sys.modules["sam_3d_body"] = package

    import sam3d_export as S
    from sam_3d_body.models.meta_arch import SAM3DBody
    from sam_3d_body.utils.checkpoint import load_state_dict
    from sam_3d_body.utils.config import get_config

    S.patch_mhr_utils_for_export()
    S.patch_camera_encoder_for_export()
    S.patch_roma_for_export()
    S.register_onnx_symbolics()

    cfg = get_config(config)
    cfg.defrost()
    cfg.MODEL.MHR_HEAD.MHR_MODEL_PATH = mhr
    cfg.freeze()

    real = torch.cuda.is_available
    torch.cuda.is_available = lambda: False
    try:
        model = SAM3DBody(cfg)
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        state = state.get("state_dict", state)
        load_state_dict(model, state, strict=False)
    finally:
        torch.cuda.is_available = real
    model.eval()
    model = model.float()
    model.backbone_dtype = torch.float32
    del state
    return model


class Parts(nn.Module):
    """A module holding only the given attributes of `model`, by dotted path.

    Registering the whole model would drag `head_pose.mhr` — 664 MB of scripted
    graph — into every exported file.
    """

    def __init__(self, model, paths=()):
        super().__init__()
        for path in paths:
            value = model
            for name in path.split("."):
                value = getattr(value, name)
            setattr(self, path.replace(".", "__"), value)


# ---------------------------------------------------------------------------
# maths mirrored from the reference implementation
# ---------------------------------------------------------------------------
def condition_vector(bbox_center, bbox_scale, cam_int):
    """CLIFF conditioning with `USE_INTRIN_CENTER=True`."""
    cx, cy = bbox_center[:, 0:1], bbox_center[:, 1:2]
    b = bbox_scale[:, 0:1]
    focal = cam_int[:, 0, 0:1]
    cond = torch.cat(
        [cx - cam_int[:, 0, 2:3], cy - cam_int[:, 1, 2:3], b], dim=-1
    )
    return torch.cat([cond[:, 0:2] / focal, cond[:, 2:3] / focal], dim=-1)


def prompt_embeddings(prompt_encoder, keypoints):
    """Export-friendly `PromptEncoder._embed_keypoints` (dummy prompt)."""
    coords = keypoints[:, :, :2]
    labels = keypoints[:, :, -1]

    pe = prompt_encoder.pe_layer._pe_encoding(coords.to(torch.float))
    invalid = (labels == -2).unsqueeze(-1)
    not_a_point = (labels == -1).unsqueeze(-1)
    valid = (~invalid & ~not_a_point).unsqueeze(-1)

    out = torch.where(
        invalid,
        prompt_encoder.invalid_point_embed.weight.view(1, 1, -1).expand_as(pe),
        pe,
    )
    out = torch.where(
        not_a_point,
        prompt_encoder.not_a_point_embed.weight.view(1, 1, -1).expand_as(pe),
        out,
    )
    joint = prompt_encoder.point_embeddings[0].weight.view(1, 1, -1).expand_as(pe)
    return torch.where(valid, out + joint, out)


def centered_affine(bbox_center, bbox_scale, image_size=IMAGE_SIZE):
    """Full-to-crop affine for an unrotated box, centred on the crop.

    Crop index `u` maps back to `x = (u - a02) / a00`, i.e. the box centre lands
    on crop index `image_size / 2`. This mirrors `get_warp_matrix` with rotation
    0 and is the mapping `velomocap-infer` implements in `Sam3dInput::affine`.
    """
    cx, cy = float(bbox_center[0]), float(bbox_center[1])
    sx = image_size / float(bbox_scale[0])
    sy = image_size / float(bbox_scale[1])
    half = image_size * 0.5
    return torch.tensor(
        [[sx, 0.0, half - cx * sx], [0.0, sy, half - cy * sy], [0.0, 0.0, 1.0]]
    )[None]


def ray_condition(affine_trans, cam_int, image_size=IMAGE_SIZE):
    """`SAM3DBody.get_ray_condition` on the CPU.

    The reference method ends in `.cuda()`, which makes it unusable as a build
    step; the arithmetic below is otherwise identical.
    """
    grid = torch.stack(
        torch.meshgrid(
            torch.arange(image_size), torch.arange(image_size), indexing="xy"
        ),
        dim=2,
    )
    affine = affine_trans[0]
    scale = torch.tensor([affine[0, 0], affine[1, 1]])
    offset = torch.tensor([affine[0, 2], affine[1, 2]])
    grid = grid / scale - offset / scale
    intrinsics = cam_int[0]
    grid = grid - torch.tensor([intrinsics[0, 2], intrinsics[1, 2]])
    grid = grid / torch.tensor([intrinsics[0, 0], intrinsics[1, 1]])
    return grid.permute(2, 0, 1)[None]


def full_to_crop(kps2d, affine_trans, img_size):
    homogeneous = torch.cat([kps2d, torch.ones_like(kps2d[:, :, -1:])], dim=-1)
    out = homogeneous @ affine_trans.mT
    return out[..., :2] / img_size.unsqueeze(1) - 0.5


def project_keypoints(kps3d, pred_cam, bbox_center, bbox_size, cam_int,
                      default_scale_factor=1.0):
    """`PerspectiveHead.perspective_projection`, without the export traps.

    The reference negates the camera axes in place (`pred_cam[..., [0, 2]] *= -1`)
    and then calls `tensor.reshape(batch_size, -1, 2)`. Both lowerings emit a node
    with an empty input — the reshape one because `batch_size` is a traced
    `aten::size`, so torch has to build the target shape at run time and resolve
    the `-1` with an `Equal`/`Where` cluster whose first operand gets dropped.
    Neither is needed: negating out of place and skipping the no-op reshapes
    gives the same numbers with a graph that loads.
    """
    cam = torch.stack([-pred_cam[:, 0], pred_cam[:, 1], -pred_cam[:, 2]], dim=-1)
    s, tx, ty = cam[:, 0], cam[:, 1], cam[:, 2]

    # depth ~= f / s, with f in NDC space (Zolly sec. 3.1).
    bs = bbox_size * s * default_scale_factor + 1e-8
    focal = cam_int[:, 0, 0]
    tz = 2 * focal / bs
    cx = 2 * (bbox_center[:, 0] - cam_int[:, 0, 2]) / bs
    cy = 2 * (bbox_center[:, 1] - cam_int[:, 1, 2]) / bs
    pred_cam_t = torch.stack([tx + cx, ty + cy, tz], dim=-1)

    j3d_cam = kps3d + pred_cam_t.unsqueeze(1)
    projected = j3d_cam / j3d_cam[:, :, -1].unsqueeze(-1)
    j2d = torch.einsum("bij,bkj->bki", cam_int, projected)[:, :, :2]
    return j2d, j3d_cam[:, :, 2]


def params_from_pred(head, pred):
    """`MHRHead.forward` up to (not including) the scripted MHR call."""
    import roma

    from sam_3d_body.models.modules.mhr_utils import (
        compact_cont_to_model_params_body,
        compact_cont_to_model_params_hand,
        mhr_param_hand_mask,
    )

    from sam_3d_body.models.modules.geometry_utils import rot6d_to_rotmat

    global_rot_6d = pred[:, :6]
    global_rot = roma.rotmat_to_euler("ZYX", rot6d_to_rotmat(global_rot_6d))
    global_trans = torch.zeros_like(global_rot)

    count = 6
    pose_cont = pred[:, count : count + head.body_cont_dim]
    count += head.body_cont_dim
    pose_euler = compact_cont_to_model_params_body(pose_cont)
    pose_euler = torch.where(
        mhr_param_hand_mask, torch.zeros_like(pose_euler), pose_euler
    )
    pose_euler = torch.cat(
        [pose_euler[:, :-3], torch.zeros_like(pose_euler[:, -3:])], dim=1
    )

    shape = pred[:, count : count + head.num_shape_comps]
    count += head.num_shape_comps
    scale = pred[:, count : count + head.num_scale_comps]
    count += head.num_scale_comps
    hand = pred[:, count : count + head.num_hand_comps * 2]
    count += head.num_hand_comps * 2
    face = pred[:, count : count + head.num_face_comps] * 0

    # `replace_hands_in_pose`, out of place.
    left, right = torch.split(hand, [head.num_hand_pose_comps] * 2, dim=1)
    left_model = compact_cont_to_model_params_hand(
        head.hand_pose_mean[None] + torch.einsum("da,ab->db", left, head.hand_pose_comps)
    )
    right_model = compact_cont_to_model_params_hand(
        head.hand_pose_mean[None] + torch.einsum("da,ab->db", right, head.hand_pose_comps)
    )

    body = pose_euler[..., :130]
    full = torch.cat([global_trans * 10, global_rot, body], dim=1)  # 136
    full = full.scatter(
        1, head.hand_joint_idxs_left[None].expand(full.shape[0], -1), left_model
    )
    full = full.scatter(
        1, head.hand_joint_idxs_right[None].expand(full.shape[0], -1), right_model
    )

    scales = head.scale_mean[None] + scale @ head.scale_comps
    model_params = torch.cat([full, scales], dim=1)  # 204
    return shape, model_params, face


def keypoints_from_mhr(mapping, verts, skel_state):
    """MHR outputs -> the 70 body keypoints the decoder refines against."""
    coords, _, _ = torch.split(skel_state, [3, 4, 1], dim=2)
    verts = verts / 100
    coords = coords / 100
    vert_joints = torch.cat([verts, coords], dim=1)
    kps = (
        (mapping @ vert_joints.permute(1, 0, 2).flatten(1, 2))
        .reshape(-1, vert_joints.shape[0], 3)
        .permute(1, 0, 2)
    )
    kps = kps[:, :70]
    return torch.cat([kps[..., :1], -kps[..., 1:]], dim=-1)


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------
class BackboneStage(nn.Module):
    """Backbone + the mask prompt + the camera encoder + the dense PE.

    `forward_pose_branch` adds `mask_embeddings` to the backbone features before
    the camera encoder. With no mask supplied, `_get_mask_prompt` returns the
    learned `no_mask_embed` vector broadcast over the patch grid — not
    `mask_downscaling(zeros)`, which is the `mask_score > 0` branch. Leaving it
    out made every person short and far (1.06-1.54 m at 7.5-9.7 m instead of
    1.64-1.73 m at 3.5-5.4 m).
    """

    def __init__(self, model):
        super().__init__()
        self.p = Parts(
            model, ["backbone", "ray_cond_emb", "prompt_encoder", "image_mean", "image_std"]
        )
        self.register_buffer(
            "mask_bias",
            model.prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1).clone(),
        )

    def forward(self, image, ray_cond):
        p = self.p
        x = (image - p.image_mean) / p.image_std
        embeddings = p.backbone(x)
        if isinstance(embeddings, tuple):
            embeddings = embeddings[-1]
        embeddings = embeddings + self.mask_bias
        embeddings = p.ray_cond_emb(embeddings.type(x.dtype), ray_cond)
        augment = p.prompt_encoder.get_dense_pe(embeddings.shape[-2:])
        return (
            embeddings,
            embeddings.flatten(2).permute(0, 2, 1),
            augment.flatten(2).permute(0, 2, 1),
        )


class InitStage(nn.Module):
    """Everything that happens once, before the decoder loop."""

    def __init__(self, model):
        super().__init__()
        self.p = Parts(
            model,
            [
                "init_pose",
                "init_camera",
                "init_to_token_mhr",
                "prev_to_token_mhr",
                "prompt_encoder",
                "prompt_to_token",
                "keypoint_embedding",
                "hand_box_embedding",
                "keypoint3d_embedding",
            ],
        )

    def forward(self, bbox_center, bbox_scale, cam_int, keypoints):
        p = self.p
        batch = bbox_center.shape[0]

        condition = condition_vector(bbox_center, bbox_scale, cam_int)
        seed = torch.cat(
            [
                p.init_pose.weight.expand(batch, -1),
                p.init_camera.weight.expand(batch, -1),
            ],
            dim=-1,
        ).unsqueeze(1)

        init_input = torch.cat([condition.view(batch, 1, -1), seed], dim=-1)
        token_embeddings = p.init_to_token_mhr(init_input).view(batch, 1, -1)
        prev_embeddings = p.prev_to_token_mhr(seed).view(batch, 1, -1)

        prompt = prompt_embeddings(p.prompt_encoder, keypoints)
        prompt = prompt.reshape(prompt.shape[0], -1)
        prompt = p.prompt_to_token(prompt).unsqueeze(1)

        tokens = torch.cat([token_embeddings, prev_embeddings, prompt], dim=1)
        augment = torch.cat(
            [torch.zeros_like(token_embeddings), prev_embeddings, prompt], dim=1
        )

        def append(tensor, weight):
            extra = weight[None].repeat(batch, 1, 1)
            out = torch.cat([tensor, extra], dim=1)
            return out, torch.zeros_like(extra)

        tokens, pad = append(tokens, p.hand_box_embedding.weight)
        augment = torch.cat([augment, pad], dim=1)

        kps_start = tokens.shape[1]
        tokens, pad = append(tokens, p.keypoint_embedding.weight)
        augment = torch.cat([augment, pad], dim=1)
        kps_end = tokens.shape[1]

        kps3d_start = tokens.shape[1]
        tokens, pad = append(tokens, p.keypoint3d_embedding.weight)
        augment = torch.cat([augment, pad], dim=1)

        self.layout = {
            "hand_det_start": 3,
            "kps_start": int(kps_start),
            "kps_end": int(kps_end),
            "kps3d_start": int(kps3d_start),
            "tokens": int(tokens.shape[1]),
        }
        return tokens, augment


class StepStage(nn.Module):
    def __init__(self, model, index):
        super().__init__()
        self.layer = model.decoder.layers[index]

    def forward(self, tokens, token_augment, image_seq, augment_seq):
        tokens, image_seq = self.layer(
            tokens, image_seq, token_augment, augment_seq, None
        )
        return tokens, token_augment, image_seq


class PoseStage(nn.Module):
    """`norm_final`, the pose MLP and the camera head — and nothing else.

    The 519-value `pred` vector is *not* decoded here. Decoding it
    (`compact_cont_to_model_params_body`, `batchXYZfrom6D`, the hand PCA blend,
    the scale basis, `roma`'s matrix→Euler ZYX) is pure scalar maths over one
    vector, and every attempt to trace it leaves a node whose first input is the
    empty string, so the graph cannot be loaded. `velomocap-infer` owns that
    layout instead (`sam3d_params.rs`); the graph stops at `pred`.
    """

    def __init__(self, model):
        super().__init__()
        self.p = Parts(
            model,
            [
                "decoder.norm_final",
                "init_pose",
                "init_camera",
                "head_camera",
                "head_pose.proj",
            ],
        )

    def forward(self, tokens):
        p = self.p
        normed = p.decoder__norm_final(tokens)
        pose_token = normed[:, 0]

        pred = p.head_pose__proj(pose_token) + p.init_pose.weight
        pred_cam = p.head_camera(pose_token, p.init_camera.weight)
        return normed, pred, pred_cam


class PostStage(nn.Module):
    """MHR vertices -> keypoints -> the two token refinements."""

    def __init__(self, model, layout, image_size=IMAGE_SIZE):
        super().__init__()
        self.p = Parts(
            model,
            [
                "head_pose.keypoint_mapping",
                "keypoint_posemb_linear",
                "keypoint_feat_linear",
                "keypoint3d_posemb_linear",
            ],
        )
        self.layout = layout
        self.image_size = image_size

    def forward(
        self,
        tokens,
        token_augment,
        embeddings,
        verts,
        skel_state,
        pred_cam,
        affine_trans,
        bbox_center,
        bbox_scale,
        ori_img_size,
        cam_int,
    ):
        p = self.p
        start = self.layout["kps_start"]
        end = self.layout["kps_end"]
        start3d = self.layout["kps3d_start"]
        end3d = start3d + (end - start)

        kps3d = keypoints_from_mhr(p.head_pose__keypoint_mapping, verts, skel_state)

        kps2d, kps2d_depth = project_keypoints(
            kps3d,
            pred_cam,
            bbox_center,
            bbox_scale[:, 0],
            cam_int,
        )

        crop = full_to_crop(
            kps2d,
            affine_trans,
            torch.full((1, 1), float(self.image_size), dtype=tokens.dtype),
        )

        inside = (crop + 0.5)[:, :, :]
        invalid = (
            (inside[:, :, 0] < 0)
            | (inside[:, :, 0] > 1)
            | (inside[:, :, 1] < 0)
            | (inside[:, :, 1] > 1)
            | (kps2d_depth < 1e-5)
        )
        valid = (~invalid).to(tokens.dtype).unsqueeze(-1)

        token_augment = torch.cat(
            [
                token_augment[:, :start],
                p.keypoint_posemb_linear(crop) * valid,
                token_augment[:, end:],
            ],
            dim=1,
        )

        sampled = F.grid_sample(
            embeddings,
            crop[:, :, None, :] * 2,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampled = sampled.squeeze(3).permute(0, 2, 1) * valid
        tokens = torch.cat(
            [
                tokens[:, :start],
                tokens[:, start:end] + p.keypoint_feat_linear(sampled),
                tokens[:, end:],
            ],
            dim=1,
        )

        pelvis = (kps3d[:, [9]] + kps3d[:, [10]]) / 2
        token_augment = torch.cat(
            [
                token_augment[:, :start3d],
                p.keypoint3d_posemb_linear(kps3d - pelvis),
                token_augment[:, end3d:],
            ],
            dim=1,
        )
        return tokens, token_augment


# ---------------------------------------------------------------------------
# export + verify
# ---------------------------------------------------------------------------
def export_stage(module, inputs, input_names, output_names, path, opset):
    started = time.time()
    module = module.eval()
    traced = torch.jit.freeze(torch.jit.trace(module, inputs, check_trace=False))
    try:
        torch.onnx.export(
            traced,
            inputs,
            path,
            input_names=input_names,
            output_names=output_names,
            opset_version=opset,
            do_constant_folding=True,
            external_data=True,
            dynamo=False,
        )
    except Exception:  # noqa: BLE001
        print(f"      FAILED {os.path.basename(path)}", flush=True)
        traceback.print_exc()
        raise
    size = os.path.getsize(path) / 1e6
    print(
        f"      {os.path.basename(path):26s} {size:8.2f} MB ({time.time() - started:.1f}s)",
        flush=True,
    )
    # `torch.onnx.export` happily emits a graph that cannot be loaded (a node
    # input left as an empty string). `verify` opens every graph with ONNX
    # Runtime, which is the check that actually matters; the ONNX checker is
    # skipped here because it cannot resolve the per-tensor external files.
    return {"file": os.path.basename(path), "inputs": input_names, "outputs": output_names}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=r"E:\RGBMocap\export\sam-3d-body")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--mhr", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument(
        "--mhr-onnx",
        default=None,
        help="reuse this MHR graph instead of exporting one (tools/onnx/mhr_export.py)",
    )
    args = parser.parse_args()

    print("[1/5] loading model ...", flush=True)
    model = load_model(args.repo, args.ckpt, args.mhr, args.config)
    os.makedirs(args.out, exist_ok=True)

    print("[2/5] building stages + reference chain ...", flush=True)
    torch.manual_seed(0)
    bbox_center = torch.tensor([[640.0, 400.0]])
    bbox_scale = torch.tensor([[200.0, 300.0]]) * 1.25
    cam_int = torch.tensor(
        [[[1100.0, 0.0, 640.0], [0.0, 1100.0, 360.0], [0.0, 0.0, 1.0]]]
    )
    cam = {
        "affine_trans": centered_affine(bbox_center[0], bbox_scale[0]),
        "bbox_center": bbox_center,
        "bbox_scale": bbox_scale,
        "ori_img_size": torch.tensor([[1280.0, 720.0]]),
        "cam_int": cam_int,
    }
    image = torch.rand(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    ray_cond = ray_condition(cam["affine_trans"], cam_int)
    keypoints = torch.zeros(1, 1, 3)
    keypoints[:, :, -1] = -2

    backbone = BackboneStage(model).eval()
    init = InitStage(model).eval()
    pose = PoseStage(model).eval()
    steps = [StepStage(model, i).eval() for i in range(len(model.decoder.layers))]

    backbone_io = (image, ray_cond)
    backbone_out = backbone(*backbone_io)
    embeddings, image_seq, augment_seq = backbone_out

    init_io = (cam["bbox_center"], cam["bbox_scale"], cam["cam_int"], keypoints)
    tokens, token_augment = init(*init_io)
    layout = init.layout
    print("      token layout:", layout, flush=True)

    post = PostStage(model, layout).eval()

    step_io, step_out = [], []
    post_io = None
    pred_cases = []
    for index, step in enumerate(steps):
        current = (tokens, token_augment, image_seq, augment_seq)
        tokens, token_augment, image_seq = step(*current)
        step_io.append(current)
        step_out.append((tokens, token_augment, image_seq))
        if index == len(steps) - 1:
            break

        pose_in = (tokens,)
        normed, pred, pred_cam = pose(*pose_in)
        pred_cases.append(pred.detach())
        identity, model_params, face = params_from_pred(model.head_pose, pred)
        verts, skel_state = model.head_pose.mhr(identity, model_params, face)

        post_in = (
            tokens,
            token_augment,
            embeddings,
            verts,
            skel_state,
            pred_cam,
            cam["affine_trans"],
            cam["bbox_center"],
            cam["bbox_scale"],
            cam["ori_img_size"],
            cam["cam_int"],
        )
        if post_io is None:
            post_io = post_in
        tokens, token_augment = post(*post_in)

    pose_io = (tokens,)
    normed, pred, pred_cam = pose(*pose_io)
    pred_cases.append(pred.detach())
    identity, model_params, face = params_from_pred(model.head_pose, pred)
    body_token = normed[:, 0]
    print("      body_token", tuple(body_token.shape), flush=True)

    print("[3/5] dumping parameter tables for the Rust decoder ...", flush=True)
    dump_params(args.out, model.head_pose, pred_cases)
    check_projection(model, post_io)
    verts_final, skel_state_final = model.head_pose.mhr(identity, model_params, face)
    dump_chain(
        args.out, cam, image, ray_cond, body_token, pred, pred_cam,
        identity, model_params, verts_final, skel_state_final,
    )

    print("[3/5] exporting ...", flush=True)
    manifest = {"layout": layout, "image_size": IMAGE_SIZE, "stages": {}}
    manifest["stages"]["backbone"] = export_stage(
        backbone, backbone_io, ["image", "ray_cond"],
        ["embeddings", "image_seq", "augment_seq"],
        os.path.join(args.out, "sam3d_backbone.onnx"), args.opset,
    )
    manifest["stages"]["init"] = export_stage(
        init, init_io,
        ["bbox_center", "bbox_scale", "cam_int", "keypoints"],
        ["tokens", "token_augment"],
        os.path.join(args.out, "sam3d_init.onnx"), args.opset,
    )
    for index, step in enumerate(steps):
        manifest["stages"][f"step{index}"] = export_stage(
            step, step_io[index],
            ["tokens", "token_augment", "image_seq", "augment_seq"],
            ["tokens_next", "token_augment_next", "image_seq_next"],
            os.path.join(args.out, f"sam3d_step{index}.onnx"), args.opset,
        )
    manifest["stages"]["pose"] = export_stage(
        pose, pose_io, ["tokens"],
        ["normed", "pred", "pred_cam"],
        os.path.join(args.out, "sam3d_pose.onnx"), args.opset,
    )
    if args.mhr_onnx:
        # The scripted MHR graph is the same either way, and it is 693 MB: reuse
        # the one `mhr_export.py` already verified instead of writing a second.
        if not os.path.isfile(args.mhr_onnx):
            raise SystemExit(f"no MHR graph at {args.mhr_onnx}")
        mhr_file = os.path.basename(args.mhr_onnx)
        # `audit` and `pack_all` expect every graph of the chain to live in
        # `--out`; copy the reused MHR graph there unless it already is.
        target = os.path.join(args.out, mhr_file)
        if os.path.abspath(args.mhr_onnx) != os.path.abspath(target):
            import shutil

            shutil.copyfile(args.mhr_onnx, target)
        manifest["stages"]["mhr"] = {
            "file": mhr_file,
            "inputs": ["identity", "model_params", "face"],
            "outputs": ["verts", "skel_state"],
        }
        print(f"      reusing MHR graph {mhr_file}", flush=True)
    else:
        mhr_file = "sam3d_mhr.onnx"
        manifest["stages"]["mhr"] = export_stage(
            _MhrStage(model), (identity, model_params, face),
            ["identity", "model_params", "face"], ["verts", "skel_state"],
            os.path.join(args.out, mhr_file), args.opset,
        )
    manifest["stages"]["post"] = export_stage(
        post, post_io,
        [
            "tokens", "token_augment", "embeddings", "verts", "skel_state",
            "pred_cam", "affine_trans", "bbox_center", "bbox_scale",
            "ori_img_size", "cam_int",
        ],
        ["tokens_next", "token_augment_next"],
        os.path.join(args.out, "sam3d_post.onnx"), args.opset,
    )

    with open(os.path.join(args.out, "sam3d_split.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    stage_files = [
        "sam3d_backbone.onnx",
        "sam3d_init.onnx",
        *[f"sam3d_step{i}.onnx" for i in range(len(steps))],
        "sam3d_pose.onnx",
        mhr_file,
        "sam3d_post.onnx",
    ]
    print("[4/5] saved manifest", flush=True)
    clean = audit(args.out, stage_files)

    print("[5/5] replaying through ONNX Runtime ...", flush=True)
    if not clean:
        print("PARITY SKIPPED (a graph cannot be loaded)", flush=True)
        return 1
    ok = verify(args.out, backbone_io, init_io, step_io, post_io,
                cam, keypoints, body_token, model.head_pose, mhr_file)
    print("PARITY", "OK" if ok else "MISMATCH", flush=True)
    print("      packing external data ...", flush=True)
    pack_all(args.out, stage_files)
    return 0 if ok else 1


def project_reference(model, post_io):
    """The reference projection, for the self-check below."""
    _, _, _, verts, skel_state, pred_cam, _, bbox_center, bbox_scale, _, cam_int = post_io
    kps3d = keypoints_from_mhr(model.head_pose.keypoint_mapping, verts, skel_state)
    return model.head_camera.perspective_projection(
        kps3d, pred_cam, bbox_center, bbox_scale[:, 0], None, cam_int,
        use_intrin_center=True,
    ), kps3d


def check_projection(model, post_io):
    """`project_keypoints` must agree with the reference, value for value."""
    want, kps3d = project_reference(model, post_io)
    got2d, got_depth = project_keypoints(
        kps3d, post_io[5], post_io[7], post_io[8][:, 0], post_io[10]
    )
    diff2d = (got2d - want["pred_keypoints_2d"]).abs().max().item()
    diff_depth = (got_depth - want["pred_keypoints_2d_depth"]).abs().max().item()
    print(
        f"      projection check: 2d max abs {diff2d:.3e}  depth {diff_depth:.3e}",
        flush=True,
    )
    if diff2d > 1e-4 or diff_depth > 1e-4:
        raise AssertionError("project_keypoints disagrees with the reference")


def _optional_inputs(model):
    """A predicate for "this node input may legally be the empty string".

    ONNX uses an empty string to omit an *optional* node input, so an empty input
    is only a defect when the schema says the parameter is required. MHR's graph
    relies on this for the upper bound of four `Clip` nodes.
    """
    import onnx

    version = next(
        (entry.version for entry in model.opset_import if entry.domain in ("", "ai.onnx")),
        17,
    )
    cache = {}

    def is_optional(op_type, index):
        key = (op_type, index)
        if key not in cache:
            try:
                params = onnx.defs.get_schema(op_type, version, "").inputs
                options = onnx.defs.OpSchema.FormalParameterOption
                if index < len(params):
                    allowed = params[index].option != options.Single
                else:
                    allowed = bool(params) and params[-1].option == options.Variadic
            except Exception:  # noqa: BLE001 - unknown op: assume required
                allowed = False
            cache[key] = allowed
        return cache[key]

    return is_optional


def audit(out_dir, names):
    """Report nodes whose *required* input is the empty string."""
    import onnx

    clean = True
    for name in names:
        model = onnx.load(os.path.join(out_dir, name), load_external_data=False)
        is_optional = _optional_inputs(model)
        broken, optional = [], 0
        for node in model.graph.node:
            for index, item in enumerate(node.input):
                if item != "":
                    continue
                if is_optional(node.op_type, index):
                    optional += 1
                else:
                    broken.append((node.op_type, node.name, index))
        if broken:
            clean = False
            print(f"      {name:26s} {len(broken)} required input(s) missing {broken[:4]}", flush=True)
        elif optional:
            print(f"      {name:26s} clean ({optional} omitted optional input(s))", flush=True)
    if clean:
        print(f"      all {len(names)} graphs loadable", flush=True)
    return clean


class _MhrStage(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.mhr = model.head_pose.mhr
        for param in self.mhr.parameters():
            param.requires_grad_(False)

    def forward(self, identity, model_params, face):
        return self.mhr(identity, model_params, face)


def verify(out_dir, backbone_io, init_io, step_io, post_io, cam, keypoints,
           expected_token, head, mhr_file="sam3d_mhr.onnx"):
    """Replay the whole chain in ONNX Runtime and compare the final token.

    `pred` comes out of the pose graph and is decoded here by the reference
    Python implementation; `velomocap-infer` decodes it in Rust instead, and the
    two agree to the tolerance checked by `sam3d_params_ref.json`.
    """
    import numpy as np
    import onnxruntime as ort

    def session(name):
        return ort.InferenceSession(
            os.path.join(out_dir, name), providers=["CPUExecutionProvider"]
        )

    def run(name, feeds):
        """Feed by position, so torch's renaming of duplicate names is moot."""
        sess = session(name)
        ordered = [
            feeds[item.name] if item.name in feeds else feeds[item.name.rsplit(".", 1)[0]]
            for item in sess.get_inputs()
        ]
        return sess.run(None, {
            item.name: (value.numpy() if hasattr(value, "numpy") else value)
            for item, value in zip(sess.get_inputs(), ordered)
        })

    def decode(pred):
        return params_from_pred(head, torch.from_numpy(pred))

    embeddings, image_seq, augment_seq = run(
        "sam3d_backbone.onnx", {"image": backbone_io[0], "ray_cond": backbone_io[1]}
    )
    tokens, token_augment = run(
        "sam3d_init.onnx",
        {
            "bbox_center": init_io[0],
            "bbox_scale": init_io[1],
            "cam_int": init_io[2],
            "keypoints": init_io[3],
        },
    )

    for index in range(len(step_io)):
        tokens, token_augment, image_seq = run(
            f"sam3d_step{index}.onnx",
            {
                "tokens": tokens,
                "token_augment": token_augment,
                "image_seq": image_seq,
                "augment_seq": augment_seq,
            },
        )
        if index == len(step_io) - 1:
            break
        normed, pred, pred_cam = run("sam3d_pose.onnx", {"tokens": tokens})
        identity, model_params, face = decode(pred)
        identity = identity.numpy()
        model_params = model_params.numpy()
        face = face.numpy()
        verts, skel_state = run(
            mhr_file,
            {"identity": identity, "model_params": model_params, "face": face},
        )
        tokens, token_augment = run(
            "sam3d_post.onnx",
            {
                "tokens": tokens,
                "token_augment": token_augment,
                "embeddings": embeddings,
                "verts": verts,
                "skel_state": skel_state,
                "pred_cam": pred_cam,
                "affine_trans": cam["affine_trans"].numpy(),
                "bbox_center": cam["bbox_center"].numpy(),
                "bbox_scale": cam["bbox_scale"].numpy(),
                "ori_img_size": cam["ori_img_size"].numpy(),
                "cam_int": cam["cam_int"].numpy(),
            },
        )

    normed = run("sam3d_pose.onnx", {"tokens": tokens})[0]
    got = normed[:, 0]
    want = expected_token.numpy()
    diff = np.abs(got - want)
    scale = float(np.abs(want).max())
    print(
        f"      body_token: max abs {diff.max():.3e} rel {diff.max()/scale:.3e} "
        f"mean {diff.mean():.3e}",
        flush=True,
    )
    return diff.max() / scale < 2e-3


def dump_params(out_dir, head, pred_cases):
    """Write the MHR parameter tables plus a reference fixture for Rust.

    `velomocap-infer` reads `sam3d_params.json` + `sam3d_params.bin` and turns the
    519-value `pred` into `identity`/`model_params`/`face` itself; the fixture in
    `sam3d_params_ref.json` pins that decoder to this reference, value for value.
    """
    import numpy as np

    tensors = {
        "hand_pose_mean": head.hand_pose_mean.detach().float().cpu(),
        "hand_pose_comps": head.hand_pose_comps.detach().float().cpu(),
        "scale_mean": head.scale_mean.detach().float().cpu(),
        "scale_comps": head.scale_comps.detach().float().cpu(),
    }
    blob = bytearray()
    table = {}
    for name, value in tensors.items():
        # The shape stays in the metadata; the blob is flat little-endian f32.
        array = value.reshape(-1).numpy().astype("<f4")
        table[name] = {
            "offset": len(blob) // 4,
            "len": int(array.size),
            "shape": [int(d) for d in value.shape],
        }
        blob += array.tobytes()

    meta = {
        "npose": int(head.npose),
        "body_cont_dim": int(head.body_cont_dim),
        "num_shape_comps": int(head.num_shape_comps),
        "num_scale_comps": int(head.num_scale_comps),
        "num_hand_comps": int(head.num_hand_comps),
        "num_hand_pose_comps": int(head.num_hand_pose_comps),
        "num_face_comps": int(head.num_face_comps),
        "hand_joint_idxs_left": [int(v) for v in head.hand_joint_idxs_left],
        "hand_joint_idxs_right": [int(v) for v in head.hand_joint_idxs_right],
        "tensors": table,
    }
    with open(os.path.join(out_dir, "sam3d_params.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    with open(os.path.join(out_dir, "sam3d_params.bin"), "wb") as fh:
        fh.write(bytes(blob))

    # Two synthetic predictions on top of the real ones, so the fixture covers
    # the hand PCA blend and the scale basis with values the decoder has never
    # seen from the network.
    generator = torch.Generator().manual_seed(20261113)
    cases = list(pred_cases) + [
        torch.randn(1, head.npose, generator=generator) * 0.7 for _ in range(2)
    ]

    records = []
    for index, pred in enumerate(cases):
        identity, model_params, face = params_from_pred(head, pred)
        records.append(
            {
                "name": f"case{index}",
                "pred": [float(v) for v in pred.flatten()],
                "identity": [float(v) for v in identity.flatten()],
                "model_params": [float(v) for v in model_params.flatten()],
                "face": [float(v) for v in face.flatten()],
            }
        )
    with open(
        os.path.join(out_dir, "sam3d_params_ref.json"), "w", encoding="utf-8"
    ) as fh:
        json.dump({"cases": records}, fh)

    params = [abs(v) for record in records for v in record["model_params"]]
    print(
        f"      sam3d_params.bin {len(blob) / 1e6:.4f} MB, {len(records)} fixture "
        f"cases, |model_params| max {max(params):.3f}",
        flush=True,
    )


def _external_locations(onnx, path):
    """Every `location` any tensor in the graph points at."""
    model = onnx.load(path, load_external_data=False)
    external = onnx.TensorProto.EXTERNAL
    tensors = list(model.graph.initializer)
    for node in model.graph.node:
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.TENSOR:
                tensors.append(attribute.t)
    return {
        entry.value
        for tensor in tensors
        if tensor.data_location == external
        for entry in tensor.external_data
        if entry.key == "location"
    }


def dump_chain(out_dir, cam, image, ray_cond, body_token, pred, pred_cam,
               identity, model_params, verts, skel_state):
    """Dump one full reference frame, so the Rust chain can be compared to it.

    The tensors are written as a flat little-endian `f32` blob with a JSON index,
    because the image and the rays alone are 9 MB of numbers. `crop` is the same
    image the backbone saw, in `[H, W, C]` order, which is the layout
    `Sam3dInput::crop` uses.
    """
    crop = image[0].permute(1, 2, 0).contiguous().reshape(-1)
    tensors = {
        "crop": crop,
        "ray": ray_cond.reshape(-1),
        "body_token": body_token.reshape(-1),
        "pred": pred.reshape(-1),
        "pred_cam": pred_cam.reshape(-1),
        "identity": identity.reshape(-1),
        "model_params": model_params.reshape(-1),
        "verts": verts.reshape(-1),
        "skel_state": skel_state.reshape(-1),
    }
    blob = bytearray()
    table = {}
    for name, value in tensors.items():
        array = value.detach().float().cpu().numpy().astype("<f4")
        table[name] = {"offset": len(blob) // 4, "len": int(array.size)}
        blob += array.tobytes()

    meta = {
        "image_size": IMAGE_SIZE,
        "layout": "flat <f4, one block per tensor",
        "tensors": table,
        "bbox_center": [float(v) for v in cam["bbox_center"][0]],
        "bbox_scale": [float(v) for v in cam["bbox_scale"][0]],
        "cam_int": [[float(v) for v in row] for row in cam["cam_int"][0]],
        "affine": [[float(v) for v in row] for row in cam["affine_trans"][0]],
    }
    with open(os.path.join(out_dir, "sam3d_chain_ref.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    with open(os.path.join(out_dir, "sam3d_chain_ref.bin"), "wb") as fh:
        fh.write(bytes(blob))
    print(
        f"      sam3d_chain_ref.bin {len(blob) / 1e6:.2f} MB "
        f"(crop, ray, tokens, parameters, mesh)",
        flush=True,
    )


def pack_all(out_dir, names):
    """Collapse torch's one-file-per-tensor output into graph + one sidecar.

    `torch.jit.freeze` turns every weight into a `Constant` node attribute, and
    torch writes each of those to its own file (`_Constant_42_attr__value`). With
    `convert_attribute=False` they stay inline in the graph protobuf, which blows
    the 2 GB protobuf limit on the backbone (`EncodeError: Failed to serialize
    proto`). Converting attributes too puts them in the sidecar with the rest.

    Packing is not idempotent: ONNX's external-data writer appends, so re-packing
    an already packed graph doubles the sidecar. The location check and the
    unlink below keep a second run from doing that.
    """
    import onnx

    for name in names:
        path = os.path.join(out_dir, name)
        stem = os.path.splitext(name)[0]
        sidecar_name = f"{stem}.weights.bin"
        sidecar = os.path.join(out_dir, sidecar_name)
        locations = _external_locations(onnx, path)
        if locations == {sidecar_name}:
            print(f"      {name:26s} already packed", flush=True)
            continue
        if not locations:
            # A single self-contained file (MHR): nothing to fold.
            print(f"      {name:26s} already self-contained", flush=True)
            continue
        model = onnx.load(path, load_external_data=True)
        if os.path.exists(sidecar):
            os.remove(sidecar)
        onnx.save_model(
            model,
            path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=sidecar_name,
            size_threshold=1024,
            convert_attribute=True,
        )
        removed = 0
        for leaf in locations:
            candidate = os.path.join(out_dir, leaf)
            if leaf != sidecar_name and os.path.isfile(candidate):
                os.remove(candidate)
                removed += 1
        graph_mb = os.path.getsize(path) / 1e6
        weights_mb = os.path.getsize(sidecar) / 1e6 if os.path.exists(sidecar) else 0.0
        print(
            f"      {name:26s} {graph_mb:8.2f} + {weights_mb:8.2f} MB "
            f"({removed} loose tensor files removed)",
            flush=True,
        )


if __name__ == "__main__":
    raise SystemExit(main())
