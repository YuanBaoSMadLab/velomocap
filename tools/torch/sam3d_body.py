"""The SAM 3D Body *body* branch as one `nn.Module`, with the full output dict.

`sam3d_split`'s stages were written for the ONNX export: they stop at the token
and the mesh, and the backbone leaves out the mask prompt that
`forward_pose_branch` adds (`image_embeddings + mask_embeddings`) — with a zero
mask that term is a per-channel constant, so the ONNX chain and the first
TorchScript body module were both missing it. This module reproduces
`forward_step(batch, "body")` end to end, including

* the mask bias,
* the 2D projection (`camera_project`),
* the joint rotations (`roma.unitquat_to_rotmat` of the skeleton state),
* the model parameters MHR actually consumed,
* the optional `prev_estimate` of `run_keypoint_prompt`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import roma

from sam3d_split import (
    IMAGE_SIZE,
    Parts,
    condition_vector,
    full_to_crop,
    keypoints_from_mhr,
    params_from_pred,
    prompt_embeddings,
    project_keypoints,
)

BODY_LAYOUT = {
    "hand_det_start": 3,
    "kps_start": 5,
    "kps_end": 75,
    "kps3d_start": 75,
    "tokens": 145,
}


def params_from_pred_body(head, pred):
    """`params_from_pred`, with the CPU-pinned hand mask moved to `pred`."""
    import sam3d_split

    from sam_3d_body.models.modules import mhr_utils

    mask = mhr_utils.mhr_param_hand_mask
    try:
        mhr_utils.mhr_param_hand_mask = mask.to(pred.device)
        return params_from_pred(head, pred)
    finally:
        mhr_utils.mhr_param_hand_mask = mask


def project_keypoints_cam(
    kps3d, pred_cam, bbox_center, bbox_size, cam_int, default_scale_factor=1.0
):
    """`PerspectiveHead.perspective_projection`, keeping the camera translation.

    `project_keypoints` in `sam3d_split` mirrors this but drops `pred_cam_t` and
    `focal_length`, which `run_inference` needs for its final 2D projection.
    """
    cam = torch.stack([-pred_cam[:, 0], pred_cam[:, 1], -pred_cam[:, 2]], dim=-1)
    s, tx, ty = cam[:, 0], cam[:, 1], cam[:, 2]
    bs = bbox_size * s * default_scale_factor + 1e-8
    focal = cam_int[:, 0, 0]
    tz = 2 * focal / bs
    cx = 2 * (bbox_center[:, 0] - cam_int[:, 0, 2]) / bs
    cy = 2 * (bbox_center[:, 1] - cam_int[:, 1, 2]) / bs
    pred_cam_t = torch.stack([tx + cx, ty + cy, tz], dim=-1)
    j3d_cam = kps3d + pred_cam_t.unsqueeze(1)
    j2d = torch.einsum(
        "bij,bkj->bki", cam_int, j3d_cam / j3d_cam[:, :, -1].unsqueeze(-1)
    )[:, :, :2]
    return j2d, j3d_cam[:, :, 2], pred_cam_t, focal


class BodyBackbone(nn.Module):
    """Backbone + mask prompt + camera encoder + dense positional embedding.

    `dtype` is the precision the ViT runs at. The shipped config has
    `TRAIN.USE_FP16: true, FP16_TYPE: bfloat16`, and `_initialze_model` converts
    exactly this module to bfloat16 (`convert_to_fp16`) while everything else
    stays float32 — so `bfloat16` is the regime the app actually runs. It is
    also ~2x faster and halves the traced file.
    """

    def __init__(self, model, mask_bias=None, dtype=None):
        super().__init__()
        self.p = Parts(
            model, ["backbone", "ray_cond_emb", "prompt_encoder", "image_mean", "image_std"]
        )
        self.dtype = dtype
        if dtype is not None:
            self.p.backbone.to(dtype)
        if mask_bias is None:
            # `_get_mask_prompt` with no mask supplied returns the learned
            # `no_mask_embed` vector, not `mask_downscaling(zeros)`.
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
        embeddings = p.ray_cond_emb(embeddings.type(x.dtype), ray_cond)
        augment = p.prompt_encoder.get_dense_pe(embeddings.shape[-2:])
        return (
            embeddings,
            embeddings.flatten(2).permute(0, 2, 1),
            augment.flatten(2).permute(0, 2, 1),
        )


def prompt_embeddings_labeled(prompt_encoder, keypoints, joint_weight):
    """`PromptEncoder._embed_keypoints`, with the per-label joint embedding.

    `sam3d_split.prompt_embeddings` hardcodes `point_embeddings[0]`, which is
    right for the first pass (every label is -2, so no joint embedding is
    added) but wrong for the keypoint prompt, whose labels are real joint
    indices (41, 62, 8, 7). `joint_weight` is the stacked weights of
    `prompt_encoder.point_embeddings` (a `ModuleList`, one `Embedding` each).
    """
    coords = keypoints[:, :, :2]
    labels = keypoints[:, :, -1]

    pe = prompt_encoder.pe_layer._pe_encoding(coords.to(torch.float))
    invalid = (labels == -2).unsqueeze(-1)
    not_a_point = (labels == -1).unsqueeze(-1)
    # `sam3d_split.prompt_embeddings` unsqueezes a second time here, which makes
    # the result (B, N, 1, C). It still works for the dummy prompt because the
    # caller flattens it, but the real prompt is (B, N, C) in the reference.
    valid = ~invalid & ~not_a_point

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
    index = labels.clamp(min=0).long()
    joint = torch.nn.functional.embedding(index, joint_weight)
    return torch.where(valid, out + joint, out)


class BodyInit(nn.Module):
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
        self.layout = dict(BODY_LAYOUT)
        self.register_buffer(
            "joint_weight",
            torch.cat([e.weight for e in model.prompt_encoder.point_embeddings], dim=0),
        )

    def forward(self, bbox_center, bbox_scale, cam_int, keypoints, prev_estimate=None):
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
        tokens = p.init_to_token_mhr(init_input).view(batch, 1, -1)

        if prev_estimate is None:
            prev_estimate = seed
        previous = p.prev_to_token_mhr(prev_estimate).view(batch, 1, -1)

        prompt = prompt_embeddings_labeled(
            p.prompt_encoder, keypoints, self.joint_weight
        )
        # `prompt_to_token` is a per-token Linear, so it applies to (B, N, 1280)
        # directly — N is 1 for the dummy prompt and 4 for the keypoint prompt.
        prompt = p.prompt_to_token(prompt)

        tokens = torch.cat([tokens, previous, prompt], dim=1)
        augment = torch.cat([torch.zeros_like(tokens[:, :1]), previous, prompt], dim=1)
        prefix = 2 + int(prompt.shape[1])

        def append(tensor, weight):
            extra = weight[None].repeat(batch, 1, 1)
            return torch.cat([tensor, extra], dim=1), torch.zeros_like(extra)

        tokens, pad = append(tokens, p.hand_box_embedding.weight)
        augment = torch.cat([augment, pad], dim=1)
        tokens, pad = append(tokens, p.keypoint_embedding.weight)
        augment = torch.cat([augment, pad], dim=1)
        tokens, pad = append(tokens, p.keypoint3d_embedding.weight)
        augment = torch.cat([augment, pad], dim=1)
        return tokens, augment, prefix


class BodyStep(nn.Module):
    def __init__(self, model, index):
        super().__init__()
        self.layer = model.decoder.layers[index]

    def forward(self, tokens, token_augment, image_seq, augment_seq):
        tokens, image_seq = self.layer(
            tokens, image_seq, token_augment, augment_seq, None
        )
        return tokens, token_augment, image_seq


class BodyPose(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.p = Parts(
            model,
            [
                "decoder.norm_final",
                "init_pose",
                "head_pose.proj",
                "head_camera",
                "init_camera",
            ],
        )

    def forward(self, tokens):
        p = self.p
        normed = p.decoder__norm_final(tokens)
        pose_token = normed[:, 0]
        pred = p.head_pose__proj(pose_token) + p.init_pose.weight
        pred_cam = p.head_camera(pose_token, p.init_camera.weight)
        return normed, pred, pred_cam


class BodyPost(nn.Module):
    """MHR vertices -> keypoints -> the two token refinements."""

    def __init__(self, model, layout=None, image_size=IMAGE_SIZE):
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
        self.layout = dict(layout or BODY_LAYOUT)
        self.image_size = image_size
        self.keypoint_count = 70

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
        prefix,
    ):
        p = self.p
        start = prefix + 2
        end = start + self.keypoint_count
        start3d = end
        end3d = start3d + self.keypoint_count

        kps3d = keypoints_from_mhr(p.head_pose__keypoint_mapping, verts, skel_state)
        kps2d, kps2d_depth = project_keypoints(
            kps3d, pred_cam, bbox_center, bbox_scale[:, 0], cam_int
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


class BodyPass(nn.Module):
    """`forward_decoder` + `camera_project` for one 512 crop, batch 1."""

    def __init__(self, model, mask_bias=None, dtype=None):
        super().__init__()
        self.backbone = BodyBackbone(model, mask_bias, dtype)
        self.init = BodyInit(model)
        self.layout = dict(BODY_LAYOUT)
        self.pose = BodyPose(model)
        self.steps = nn.ModuleList(
            [BodyStep(model, i) for i in range(len(model.decoder.layers))]
        )
        self.post = BodyPost(model, self.layout)
        self.mhr = model.head_pose.mhr
        self.head = model.head_pose

    def affine(self, bbox_center, bbox_scale):
        cx = bbox_center[:, 0:1]
        cy = bbox_center[:, 1:2]
        sx = IMAGE_SIZE / bbox_scale[:, 0:1]
        sy = IMAGE_SIZE / bbox_scale[:, 1:2]
        half = IMAGE_SIZE * 0.5
        zero = torch.zeros_like(sx)
        row0 = torch.cat([sx, zero, half - cx * sx], dim=1)
        row1 = torch.cat([zero, sy, half - cy * sy], dim=1)
        row2 = torch.cat([zero, zero, zero + 1.0], dim=1)
        return torch.stack([row0, row1, row2], dim=1)

    def run(self, image, bbox_center, bbox_scale, cam_int, keypoints, prev_estimate=None):
        """Everything up to and including the last pose prediction + MHR."""
        affine = self.affine(bbox_center, bbox_scale)
        ray = _rays(affine, cam_int)
        embeddings, image_seq, augment_seq = self.backbone(image, ray)
        return self.run_decoder(
            embeddings,
            image_seq,
            augment_seq,
            bbox_center,
            bbox_scale,
            cam_int,
            keypoints,
            prev_estimate,
        )

    def run_decoder(
        self,
        embeddings,
        image_seq,
        augment_seq,
        bbox_center,
        bbox_scale,
        cam_int,
        keypoints,
        prev_estimate=None,
    ):
        """The decoder on *given* backbone features.

        `run_keypoint_prompt` reuses `output["image_embeddings"]` from the first
        body pass — the backbone runs once per crop, not once per pass. The
        features here are already ray-conditioned, which is the same thing the
        reference gets by conditioning the stored pre-ray features inside
        `forward_decoder`.
        """
        affine = self.affine(bbox_center, bbox_scale)
        tokens, token_augment, prefix = self.init(
            bbox_center, bbox_scale, cam_int, keypoints, prev_estimate
        )

        for index, step in enumerate(self.steps):
            tokens, token_augment, image_seq = step(
                tokens, token_augment, image_seq, augment_seq
            )
            if index == len(self.steps) - 1:
                break
            _, pred, pred_cam = self.pose(tokens)
            shape, model_params, face = params_from_pred_body(self.head, pred)
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
                prefix,
            )

        normed, pred, pred_cam = self.pose(tokens)
        return (
            normed,
            pred,
            pred_cam,
            embeddings,
            image_seq,
            augment_seq,
            tokens,
            token_augment,
        )


def _rays(affine, cam_int, image_size=IMAGE_SIZE):
    grid = torch.stack(
        torch.meshgrid(
            torch.arange(image_size, dtype=affine.dtype, device=affine.device),
            torch.arange(image_size, dtype=affine.dtype, device=affine.device),
            indexing="xy",
        ),
        dim=2,
    )
    scale = torch.cat([affine[:, 0:1, 0], affine[:, 1:2, 1]], dim=1)
    offset = torch.cat([affine[:, 0:1, 2], affine[:, 1:2, 2]], dim=1)
    g = grid[None] / scale[:, None, None, :] - offset[:, None, None, :] / scale[
        :, None, None, :
    ]
    centre = torch.cat([cam_int[:, 0:1, 2], cam_int[:, 1:2, 2]], dim=1)
    focal = torch.cat([cam_int[:, 0:1, 0], cam_int[:, 1:2, 1]], dim=1)
    g = (g - centre[:, None, None, :]) / focal[:, None, None, :]
    return g.permute(0, 3, 1, 2)
