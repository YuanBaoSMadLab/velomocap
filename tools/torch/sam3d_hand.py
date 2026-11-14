"""The SAM 3D Body *hand* branch as one `nn.Module`.

Mirrors `SAM3DBody.forward_decoder_hand` exactly, including everything
`MHRHead.forward` does for a head built with `enable_hand_model=True`:

* the head predicts a **wrist-centric** pose, so `mhr_forward` first lifts the
  global rotation/translation into body space with `local_to_world_wrist`,
  `right_wrist_coords` and `root_coords`;
* every non-hand model parameter is then zeroed (`nonhand_param_idxs`), because
  only the right hand is predicted — the left hand is a flipped image plus a
  flipped box, and fusion picks whichever pass survives;
* `mhr_forward(return_keypoints=True)` zeroes the keypoints outside the right
  hand (21..42) before the 2D projection.

Two other things are hand-specific and were missing from the first export:

* the hand camera head has `DEFAULT_SCALE_FACTOR_HAND = 10` (config
  `MODEL.CAMERA_HEAD.DEFAULT_SCALE_FACTOR_HAND`), which enters
  `perspective_projection` as a multiplier on the box size;
* the keypoint tokens are refined with `keypoint_*_hand` layers, the 2D prompt
  uses `keypoint_posemb_linear_hand` and the 3D one `keypoint3d_posemb_linear_hand`.

The layout is fixed by the head widths (3 init + 2 hand boxes + 70 keypoints +
70 keypoints 3D = 145 tokens), so it is declared here instead of being probed by
running the decoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import roma

from sam3d_body import prompt_embeddings_labeled  # noqa: E402
from sam3d_split import (  # noqa: E402
    IMAGE_SIZE,
    Parts,
    condition_vector,
    full_to_crop,
    keypoints_from_mhr,
    prompt_embeddings,
    project_keypoints,
)

# `MODEL.CAMERA_HEAD.DEFAULT_SCALE_FACTOR_HAND` in the shipped config.
HAND_SCALE_FACTOR = 10.0

# 3 init/prev/prompt + 2 hand boxes + 70 2D + 70 3D keypoints.
HAND_LAYOUT = {
    "hand_det_start": 3,
    "kps_start": 5,
    "kps_end": 75,
    "kps3d_start": 75,
    "tokens": 145,
}


def hand_keypoints(mapping, verts, skel_state):
    """`MHRHead.mhr_forward(..., return_keypoints=True)` for the hand head.

    `keypoints_from_mhr` is the body flavour (70 keypoints, camera axes). The
    hand head additionally zeroes every keypoint outside 21..42, because the
    wrist-centric model only knows the right hand.
    """
    kps = keypoints_from_mhr(mapping, verts, skel_state)
    return kps[:, 21:42]


def hand_keypoints_full(mapping, verts, skel_state):
    """Same, but kept at 70 entries with the non-hand ones zeroed."""
    kps = keypoints_from_mhr(mapping, verts, skel_state)
    keep = torch.zeros_like(kps)
    keep[:, 21:42] = 1.0
    return kps * keep


def params_from_pred_hand(head, pred):
    """`MHRHead.forward` up to (not including) the scripted MHR call, hand head."""
    from sam_3d_body.models.modules.geometry_utils import rot6d_to_rotmat
    from sam_3d_body.models.modules.mhr_utils import (
        compact_cont_to_model_params_body,
        compact_cont_to_model_params_hand,
        mhr_param_hand_mask,
    )

    global_rot_6d = pred[:, :6]
    global_rot_ori = roma.rotmat_to_euler("ZYX", rot6d_to_rotmat(global_rot_6d))
    global_trans_ori = torch.zeros_like(global_rot_ori)

    # `enable_hand_model`: transfer the wrist-centric prediction to the body.
    global_rot = roma.rotmat_to_euler(
        "xyz",
        roma.euler_to_rotmat("xyz", global_rot_ori) @ head.local_to_world_wrist,
    )
    global_trans = (
        -(
            roma.euler_to_rotmat("xyz", global_rot)
            @ (head.right_wrist_coords - head.root_coords)
            + head.root_coords
        )
        + global_trans_ori
    )

    count = 6
    pose_cont = pred[:, count : count + head.body_cont_dim]
    count += head.body_cont_dim
    pose_euler = compact_cont_to_model_params_body(pose_cont)
    # `mhr_param_hand_mask` is a module-level CPU tensor; on CUDA it has to move.
    pose_euler = torch.where(
        mhr_param_hand_mask.to(pose_euler.device), torch.zeros_like(pose_euler), pose_euler
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
    # `mhr_forward`: only the right hand survives.
    model_params = model_params.index_fill(
        1, head.nonhand_param_idxs, 0.0
    )
    return shape, model_params, face


class HandBackbone(nn.Module):
    """Backbone + the hand camera encoder + the hand positional embedding.

    `forward_pose_branch` adds the *mask* prompt to the backbone features
    (`image_embeddings + mask_embeddings`) whenever `PROMPT_ENCODER.MASK_PROMPT`
    is `v1`, which it is here. `_get_mask_prompt` then branches on `mask_score`:
    with no mask supplied it returns `no_mask_embeddings`, i.e. the learned
    `no_mask_embed` vector broadcast over the patch grid — *not*
    `mask_downscaling(zeros)` (that is the `mask_score > 0` branch). Inference
    never supplies a mask, so the term is that constant; it is folded in as a
    buffer.
    """

    def __init__(self, model, mask_bias=None, dtype=None):
        super().__init__()
        self.p = Parts(
            model,
            [
                "backbone",
                "ray_cond_emb_hand",
                "hand_pe_layer",
                "image_mean",
                "image_std",
            ],
        )
        self.dtype = dtype
        if dtype is not None:
            self.p.backbone.to(dtype)
        if mask_bias is None:
            mask_bias = model.prompt_encoder.no_mask_embed.weight.reshape(
                1, -1, 1, 1
            ).clone()
        self.register_buffer("mask_bias", mask_bias)

    def forward(self, image, ray_cond):
        p = self.p
        x = (image - p.image_mean) / p.image_std
        embeddings = p.backbone(x if self.dtype is None else x.type(self.dtype))
        if isinstance(embeddings, tuple):
            embeddings = embeddings[-1]
        embeddings = embeddings + self.mask_bias
        embeddings = p.ray_cond_emb_hand(embeddings.type(x.dtype), ray_cond)
        augment = p.hand_pe_layer(embeddings.shape[-2:]).unsqueeze(0)
        return (
            embeddings,
            embeddings.flatten(2).permute(0, 2, 1),
            augment.flatten(2).permute(0, 2, 1),
        )


class HandInit(nn.Module):
    """The 145 tokens, before the first decoder round."""

    def __init__(self, model):
        super().__init__()
        self.p = Parts(
            model,
            [
                "init_pose_hand",
                "init_camera_hand",
                "init_to_token_mhr_hand",
                "prev_to_token_mhr_hand",
                "prompt_encoder",
                "prompt_to_token",
                "hand_box_embedding",
                "keypoint_embedding_hand",
                "keypoint3d_embedding_hand",
            ],
        )
        self.layout = dict(HAND_LAYOUT)
        self.register_buffer(
            "joint_weight",
            torch.cat([e.weight for e in model.prompt_encoder.point_embeddings], dim=0),
        )

    def forward(self, bbox_center, bbox_scale, cam_int, keypoints):
        p = self.p
        batch = bbox_center.shape[0]

        condition = condition_vector(bbox_center, bbox_scale, cam_int)
        estimate = torch.cat(
            [
                p.init_pose_hand.weight.expand(batch, -1),
                p.init_camera_hand.weight.expand(batch, -1),
            ],
            dim=-1,
        ).unsqueeze(1)

        init_input = torch.cat([condition.view(batch, 1, -1), estimate], dim=-1)
        token = p.init_to_token_mhr_hand(init_input).view(batch, 1, -1)
        previous = p.prev_to_token_mhr_hand(estimate).view(batch, 1, -1)

        prompt = prompt_embeddings_labeled(
            p.prompt_encoder, keypoints, self.joint_weight
        )
        # Per-token Linear, so it applies to (B, N, 1280) directly; the hand
        # batch is 2 (both hands in one forward).
        prompt = p.prompt_to_token(prompt)

        tokens = torch.cat([token, previous, prompt], dim=1)
        augment = torch.cat(
            [torch.zeros_like(token), previous, prompt], dim=1
        )

        def append(tensor, weight):
            extra = weight[None].repeat(batch, 1, 1)
            return torch.cat([tensor, extra], dim=1), torch.zeros_like(extra)

        tokens, pad = append(tokens, p.hand_box_embedding.weight)
        augment = torch.cat([augment, pad], dim=1)
        tokens, pad = append(tokens, p.keypoint_embedding_hand.weight)
        augment = torch.cat([augment, pad], dim=1)
        tokens, pad = append(tokens, p.keypoint3d_embedding_hand.weight)
        augment = torch.cat([augment, pad], dim=1)
        return tokens, augment


class HandStep(nn.Module):
    def __init__(self, model, index):
        super().__init__()
        self.layer = model.decoder_hand.layers[index]

    def forward(self, tokens, token_augment, image_seq, augment_seq):
        tokens, image_seq = self.layer(
            tokens, image_seq, token_augment, augment_seq, None
        )
        return tokens, token_augment, image_seq


class HandPose(nn.Module):
    """`norm_final` + `head_pose_hand(pose_token, init_pose_hand)` + the camera head."""

    def __init__(self, model):
        super().__init__()
        self.p = Parts(
            model,
            [
                "decoder_hand.norm_final",
                "init_pose_hand",
                "head_pose_hand.proj",
                "head_camera_hand",
                "init_camera_hand",
            ],
        )

    def forward(self, tokens):
        p = self.p
        normed = p.decoder_hand__norm_final(tokens)
        pose_token = normed[:, 0]
        pred = p.head_pose_hand__proj(pose_token) + p.init_pose_hand.weight
        pred_cam = p.head_camera_hand(pose_token, p.init_camera_hand.weight)
        return normed, pred, pred_cam


class HandPost(nn.Module):
    """The two keypoint token updates, hand flavour."""

    def __init__(self, model, layout=None, image_size=IMAGE_SIZE,
                 scale_factor=HAND_SCALE_FACTOR):
        super().__init__()
        self.p = Parts(
            model,
            [
                "head_pose_hand.keypoint_mapping",
                "keypoint_posemb_linear_hand",
                "keypoint_feat_linear_hand",
                "keypoint3d_posemb_linear_hand",
            ],
        )
        self.layout = dict(layout or HAND_LAYOUT)
        self.image_size = image_size
        self.scale_factor = scale_factor

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
        cam_int,
    ):
        p = self.p
        start = self.layout["kps_start"]
        end = self.layout["kps_end"]
        start3d = self.layout["kps3d_start"]
        end3d = start3d + (end - start)

        # `forward_decoder_hand` projects `pred_keypoints_3d`, i.e. the 70
        # keypoints with everything outside the right hand zeroed.
        kps3d = hand_keypoints_full(
            p.head_pose_hand__keypoint_mapping, verts, skel_state
        )
        kps2d, kps2d_depth = project_keypoints(
            kps3d,
            pred_cam,
            bbox_center,
            bbox_scale[:, 0],
            cam_int,
            default_scale_factor=self.scale_factor,
        )
        img_size = torch.full(
            (1, 1), float(self.image_size), dtype=tokens.dtype, device=tokens.device
        )
        crop = full_to_crop(kps2d, affine_trans, img_size)
        inside = crop + 0.5
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
                p.keypoint_posemb_linear_hand(crop) * valid,
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
                tokens[:, start:end] + p.keypoint_feat_linear_hand(sampled),
                tokens[:, end:],
            ],
            dim=1,
        )

        pelvis = (kps3d[:, [9]] + kps3d[:, [10]]) / 2
        token_augment = torch.cat(
            [
                token_augment[:, :start3d],
                p.keypoint3d_posemb_linear_hand(kps3d - pelvis),
                token_augment[:, end3d:],
            ],
            dim=1,
        )
        return tokens, token_augment


class HandBranch(nn.Module):
    """Crop in, hand pose and hand mesh out — the whole `forward_decoder_hand`.

    The first output is the *whole* normalised token block (`[1, 145, 1024]`),
    not just token 0: the app's feature extractor hooks the decoder and reads
    `output[0][0, 0, :]`, so both the caller and the checkers want to slice it
    themselves.
    """

    def __init__(self, model, image_size=IMAGE_SIZE, mask_bias=None, dtype=None):
        super().__init__()
        self.backbone = HandBackbone(model, mask_bias, dtype)
        self.init = HandInit(model)
        self.layout = dict(HAND_LAYOUT)
        self.pose = HandPose(model)
        self.steps = nn.ModuleList(
            [HandStep(model, i) for i in range(len(model.decoder_hand.layers))]
        )
        self.post = HandPost(model, self.layout)
        self.mhr = model.head_pose.mhr
        self.head = model.head_pose_hand
        self.image_size = image_size
        self.register_buffer(
            "grid",
            torch.stack(
                torch.meshgrid(
                    torch.arange(image_size).float(),
                    torch.arange(image_size).float(),
                    indexing="xy",
                ),
                dim=2,
            ),
        )

    def affine(self, bbox_center, bbox_scale):
        """Full-to-crop affine, `get_warp_matrix` with rotation 0."""
        cx = bbox_center[:, 0:1]
        cy = bbox_center[:, 1:2]
        sx = self.image_size / bbox_scale[:, 0:1]
        sy = self.image_size / bbox_scale[:, 1:2]
        half = self.image_size * 0.5
        zero = torch.zeros_like(sx)
        row0 = torch.cat([sx, zero, half - cx * sx], dim=1)
        row1 = torch.cat([zero, sy, half - cy * sy], dim=1)
        row2 = torch.cat([zero, zero, zero + 1.0], dim=1)
        return torch.stack([row0, row1, row2], dim=1)

    def rays(self, affine, cam_int):
        """`get_ray_condition`, out of place."""
        scale = torch.cat([affine[:, 0:1, 0], affine[:, 1:2, 1]], dim=1)
        offset = torch.cat([affine[:, 0:1, 2], affine[:, 1:2, 2]], dim=1)
        g = self.grid[None] / scale[:, None, None, :] - offset[:, None, None, :] / scale[
            :, None, None, :
        ]
        centre = torch.cat([cam_int[:, 0:1, 2], cam_int[:, 1:2, 2]], dim=1)
        focal = torch.cat([cam_int[:, 0:1, 0], cam_int[:, 1:2, 1]], dim=1)
        g = (g - centre[:, None, None, :]) / focal[:, None, None, :]
        return g.permute(0, 3, 1, 2)

    def forward(self, image, bbox_center, bbox_scale, cam_int, keypoints, ori_img_size):
        affine = self.affine(bbox_center, bbox_scale)
        ray = self.rays(affine, cam_int)

        embeddings, image_seq, augment_seq = self.backbone(image, ray)
        tokens, token_augment = self.init(bbox_center, bbox_scale, cam_int, keypoints)

        for index, step in enumerate(self.steps):
            tokens, token_augment, image_seq = step(
                tokens, token_augment, image_seq, augment_seq
            )
            if index == len(self.steps) - 1:
                break
            _, pred, pred_cam = self.pose(tokens)
            shape, model_params, face = params_from_pred_hand(self.head, pred)
            verts, skel_state = self.mhr(shape, model_params, face)
            tokens, token_augment = self.post(
                tokens,
                token_augment,
                embeddings,
                verts,
                skel_state,
                pred_cam,
                affine,
                bbox_center,
                bbox_scale,
                cam_int,
            )

        normed, pred, pred_cam = self.pose(tokens)
        shape, model_params, face = params_from_pred_hand(self.head, pred)
        verts, skel_state = self.mhr(shape, model_params, face)

        # `MHRHead.forward`: camera system difference, and the mesh is metres.
        sign = torch.tensor([1.0, -1.0, -1.0], dtype=verts.dtype, device=verts.device)
        verts_signed = verts / 100 * sign
        joint_rots = roma.unitquat_to_rotmat(skel_state[:, :, 3:7])
        j3d = hand_keypoints_full(self.head.keypoint_mapping, verts, skel_state)
        kps2d, kps2d_depth = project_keypoints(
            j3d,
            pred_cam,
            bbox_center,
            bbox_scale[:, 0],
            cam_int,
            default_scale_factor=self.post.scale_factor,
        )
        return (
            normed,
            pred,
            pred_cam,
            verts_signed,
            j3d,
            kps2d,
            kps2d_depth,
            model_params,
            joint_rots,
        )
