"""The whole SAM 3D Body inference as one `nn.Module`.

This is `SAM3DBody.run_inference(inference_type="full")` — the four passes the
app drives through `Sam3dTokenExtractor`:

1. body decoder on the body crop -> pose, mesh, and the two hand-detection
   tokens (`bbox_embed` / `hand_cls_embed`);
2. hand boxes -> left crop (image flipped, box x-flipped) and right crop, each
   run through the hand decoder;
3. four fusion criteria (wrist angle, box size, hand 2D inside the box, wrist
   2D distance) build a four-point keypoint prompt, and the body decoder runs
   again with it;
4. the hand pose and wrist rotations are written back, `fix_wrist_euler` clamps
   them, and `mhr_forward` rebuilds the final mesh.

Crops are produced inside the graph with `grid_sample` from a fixed 512x512
base grid, so the module can take a frame at its native resolution: the
`cv2.warpAffine` the reference performs is the same bilinear sample under
`align_corners=False`, and the affine is the one `get_warp_matrix` builds.

Inputs (batch 1):

    frame        [1, 3, H, W]   RGB in [0, 1], the *original* frame
    bbox_center  [1, 2]         person box centre, full-image pixels
    bbox_scale   [1, 2]         person box size, already padded (x1.25)
    cam_int      [1, 3, 3]
    frame_size   [1, 2]         (width, height)

Outputs:

    verts        [1, 18439, 3]  metres, camera axes (as `MHRHead.forward`)
    keypoints_3d [1, 70, 3]     same axes
    keypoints_2d [1, 70, 2]     full-image pixels
    pred_cam     [1, 3]
    pred_cam_t   [1, 3]
    focal_length [1]
    model_params [1, 204]
    joint_rots   [1, 127, 3, 3]
    hand_valid   [1, 2]         1.0 where the hand pass survived the criteria
    body_token   [1, 1024]      token 0 of the first body decoder call
    left_token   [1, 1024]      token 0 of the left-hand decoder call
    right_token  [1, 1024]      token 0 of the right-hand decoder call
    skel_state   [1, 127, 8]    raw MHR skeleton state of the final mesh
    pred         [1, 519]       the prediction that mesh was decoded from

The last three are what `flowhmr` stacks into its 3072-wide per-frame feature;
they are the same rows `Sam3dTokenExtractor` hooks, so the TorchScript path can
replace the Python extractor without touching the motion model.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import roma

from sam3d_split import IMAGE_SIZE, Parts, keypoints_from_mhr
from sam3d_body import BodyPass, params_from_pred_body, project_keypoints_cam
from sam3d_hand import HAND_SCALE_FACTOR, HandBranch, hand_keypoints_full

# `run_inference` builds the hand prompt from these limits.
THRESH_WRIST_ANGLE = 1.4
HAND_BOX_SIZE_THRESH = 64.0
HAND_KPS2D_THRESH = 0.5
HAND_WRIST_KPS2D_THRESH = 0.25
# `HAND_BOX_SIZE` -> `prepare_batch(padding=0.9)` -> `fix_aspect_ratio(0.75)` ->
# `fix_aspect_ratio(1.0)`: the crop covers 1.2x the detected box.
HAND_PADDING = 1.2


def pred_parts(head, pred):
    """The pieces of the 519-value `pred` that `run_inference` manipulates."""
    from sam_3d_body.models.modules.mhr_utils import (
        compact_cont_to_model_params_body,
        mhr_param_hand_mask,
    )

    from sam_3d_body.models.modules.geometry_utils import rot6d_to_rotmat

    count = 6
    global_rot = roma.rotmat_to_euler("ZYX", rot6d_to_rotmat(pred[:, :6]))
    pose_cont = pred[:, count : count + head.body_cont_dim]
    count += head.body_cont_dim
    pose_euler = compact_cont_to_model_params_body(pose_cont)
    pose_euler = torch.where(
        mhr_param_hand_mask.to(pose_euler.device),
        torch.zeros_like(pose_euler),
        pose_euler,
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
    return global_rot, pose_euler, shape, scale, hand, face


def rotation_angle_difference(a, b):
    rel = torch.matmul(a, b.transpose(-2, -1))
    trace = rel[..., 0, 0] + rel[..., 1, 1] + rel[..., 2, 2]
    return torch.acos(torch.clamp((trace - 1) / 2, -1.0, 1.0))


def fix_wrist_euler(wrist_xzy, limits_x=(-2.2, 1.0), limits_z=(-2.2, 1.5),
                    limits_y=(-1.2, 1.5)):
    x, z, y = wrist_xzy[..., 0], wrist_xzy[..., 1], wrist_xzy[..., 2]
    x_alt = torch.atan2(torch.sin(x + torch.pi), torch.cos(x + torch.pi))
    z_alt = torch.atan2(torch.sin(-(z + torch.pi)), torch.cos(-(z + torch.pi)))
    y_alt = torch.atan2(torch.sin(y + torch.pi), torch.cos(y + torch.pi))

    def violation(value, limits):
        below = torch.clamp(limits[0] - value, min=0.0)
        above = torch.clamp(value - limits[1], min=0.0)
        return below**2 + above**2

    orig = (
        violation(x, limits_x) + violation(z, limits_z) + violation(y, limits_y)
    )
    alt = (
        violation(x_alt, limits_x)
        + violation(z_alt, limits_z)
        + violation(y_alt, limits_y)
    )
    use_alt = alt < orig
    stacked = torch.stack([x_alt, z_alt, y_alt], dim=-1)
    return torch.where(use_alt.unsqueeze(-1), stacked, wrist_xzy)


class Sam3dFull(nn.Module):
    def __init__(self, model, mask_bias=None, backbone_dtype=None, hands=True):
        super().__init__()
        # The shipped config converts the ViT to bfloat16 and leaves everything
        # else float32, so `backbone_dtype=torch.bfloat16` is the precision the
        # app runs at. It is also ~2x faster and roughly halves the traced file.
        if backbone_dtype is not None:
            model.backbone_dtype = backbone_dtype
        # `hands=False` is `run_inference(inference_type="body")`: one body pass,
        # no hand crops and no fusion. VeloMocap uses it for the frames whose
        # hand features come from a neighbour and are interpolated in
        # (`mocap/flow_hand_sampling.py`), which is the reference's default for
        # video: hands every fourth frame, body every frame.
        self.hands = hands
        self.body = BodyPass(model, mask_bias, backbone_dtype)
        # A body-only graph never touches the hand branch, so keeping it out of
        # the module keeps its 1.6 GB of weights out of the traced file too.
        self.hand = HandBranch(model, mask_bias=mask_bias, dtype=backbone_dtype) if hands else None
        self.bbox_embed = model.bbox_embed
        self.hand_cls_embed = model.hand_cls_embed
        self.head = model.head_pose
        self.hand_head = model.head_pose_hand
        self.mhr = model.head_pose.mhr

        for name in (
            "hand_pose_mean",
            "hand_pose_comps",
            "scale_mean",
            "scale_comps",
            "joint_rotation",
            "hand_joint_idxs_left",
            "hand_joint_idxs_right",
        ):
            self.register_buffer(name, getattr(model.head_pose, name).detach().clone())

        self.register_buffer(
            "base",
            torch.stack(
                torch.meshgrid(
                    torch.arange(IMAGE_SIZE).float(),
                    torch.arange(IMAGE_SIZE).float(),
                    indexing="xy",
                ),
                dim=2,
            ),
        )
        head = self.head
        self.shape_start = 6 + head.body_cont_dim
        self.shape_len = head.num_shape_comps
        self.scale_start = self.shape_start + self.shape_len
        self.scale_len = head.num_scale_comps
        self.hand_start = self.scale_start + self.scale_len
        self.hand_len = head.num_hand_comps * 2

    # -- geometry ---------------------------------------------------------
    def _crop(self, frame, affine, size):
        """`cv2.warpAffine` for a scale+translate affine, as one grid sample.

        The reference warps a **uint8** image, so the sample is rounded back to
        8 bits before it reaches the backbone; the rounding below reproduces
        that (cv2's fixed-point weights differ by well under one level).
        """
        # One grid per item in the batch: `affine` is [B, 3, 3] and `size` [B, 2],
        # so a batch of frames is cropped against its own boxes in one
        # `grid_sample`, exactly as `B` separate warpAffines would.
        u = self.base[..., 0].unsqueeze(0)
        v = self.base[..., 1].unsqueeze(0)
        sx = (u - affine[:, 0, 2 : 3, None]) / affine[:, 0, 0 : 1, None]
        sy = (v - affine[:, 1, 2 : 3, None]) / affine[:, 1, 1 : 2, None]
        gx = (2.0 * (sx + 0.5)) / size[:, 0:1, None] - 1.0
        gy = (2.0 * (sy + 0.5)) / size[:, 1:2, None] - 1.0
        grid = torch.stack([gx, gy], dim=-1)
        crop = F.grid_sample(
            frame, grid, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        return torch.round(crop * 255.0) / 255.0

    def _hand_affine(self, centre, size):
        """`get_warp_matrix` for a square hand box, full image -> 512 crop."""
        scale = size[:, 0:1]
        sx = IMAGE_SIZE / scale
        zero = torch.zeros_like(sx)
        row0 = torch.cat([sx, zero, IMAGE_SIZE * 0.5 - centre[:, 0:1] * sx], dim=1)
        row1 = torch.cat([zero, sx, IMAGE_SIZE * 0.5 - centre[:, 1:2] * sx], dim=1)
        row2 = torch.cat([zero, zero, zero + 1.0], dim=1)
        return torch.stack([row0, row1, row2], dim=1)

    # -- MHR --------------------------------------------------------------
    def _hand_model_params(self, hand):
        from sam_3d_body.models.modules.mhr_utils import (
            compact_cont_to_model_params_hand,
        )

        left, right = torch.split(hand, [hand.shape[1] // 2] * 2, dim=1)
        left_model = compact_cont_to_model_params_hand(
            self.hand_pose_mean[None]
            + torch.einsum("da,ab->db", left, self.hand_pose_comps)
        )
        right_model = compact_cont_to_model_params_hand(
            self.hand_pose_mean[None]
            + torch.einsum("da,ab->db", right, self.hand_pose_comps)
        )
        return left_model, right_model

    def mhr_forward(self, global_rot, body_pose, hand, scale, shape, face):
        """`MHRHead.mhr_forward` with `global_trans` zeroed, as the fusion calls it."""
        left, right = self._hand_model_params(hand)
        full = torch.cat(
            [torch.zeros_like(global_rot), global_rot, body_pose[:, :130]], dim=1
        )
        full = full.scatter(
            1, self.hand_joint_idxs_left[None].expand(full.shape[0], -1), left
        )
        full = full.scatter(
            1, self.hand_joint_idxs_right[None].expand(full.shape[0], -1), right
        )
        scales = self.scale_mean[None] + scale @ self.scale_comps
        model_params = torch.cat([full, scales], dim=1)
        verts, skel_state = self.mhr(shape, model_params, face)
        joint_rots = roma.unitquat_to_rotmat(skel_state[:, :, 3:7])
        j3d = keypoints_from_mhr(self.head.keypoint_mapping, verts, skel_state)
        return verts, skel_state, model_params, joint_rots, j3d

    # -- forward ----------------------------------------------------------
    def forward(self, frame, bbox_center, bbox_scale, cam_int, frame_size):
        count = frame.shape[0]
        dummy = torch.zeros(count, 1, 3, dtype=frame.dtype, device=frame.device)
        dummy[:, :, -1] = -2
        affine = self.body.affine(bbox_center, bbox_scale)
        crop = self._crop(frame, affine, frame_size)

        normed, pred, pred_cam, _emb, _iseq, _aseq, _tok, _taug = self.body.run(
            crop, bbox_center, bbox_scale, cam_int, dummy
        )
        global_rot, body_pose, shape, scale, hand, face = pred_parts(self.head, pred)
        _shape, model_params, _face = params_from_pred_body(self.head, pred)
        verts0, skel0 = self.mhr(_shape, model_params, _face)
        j3d0 = keypoints_from_mhr(self.head.keypoint_mapping, verts0, skel0)
        j2d0, depth0, cam_t0, focal0 = project_keypoints_cam(
            j3d0, pred_cam, bbox_center, bbox_scale[:, 0], cam_int
        )
        joint_rots0 = roma.unitquat_to_rotmat(skel0[:, :, 3:7])
        j2d_crop0 = self._full_to_crop(j2d0, affine)

        if not self.hands:
            # `run_inference(inference_type="body")`: the first body pass is the
            # answer, and the hand slots stay empty. The caller fills them from
            # the nearest sampled frame, exactly like `interpolate_hands`.
            sign = torch.tensor(
                [1.0, -1.0, -1.0], dtype=verts0.dtype, device=verts0.device
            )
            empty_token = torch.zeros(
                count, 1024, dtype=normed.dtype, device=normed.device
            )
            hand_valid = torch.zeros(count, 2, dtype=pred.dtype, device=pred.device)
            return (
                verts0 / 100 * sign,
                j3d0,
                j2d0,
                pred_cam,
                cam_t0,
                focal0,
                model_params,
                joint_rots0,
                hand_valid,
                normed[:, 0],
                empty_token,
                empty_token,
                skel0,
                pred,
            )

        # -- hand boxes ---------------------------------------------------
        box_tokens = normed[:, 3:5]
        hand_box = self.bbox_embed(box_tokens).sigmoid()
        hand_logits = self.hand_cls_embed(box_tokens)
        boxes = hand_box * IMAGE_SIZE
        box_scale = boxes[..., 2:].amax(dim=-1, keepdim=True).expand(-1, -1, 2)
        # `affine` is per item, the boxes are two per item: keep the leading dim.
        offset = torch.stack([affine[:, 0, 2], affine[:, 1, 2]], dim=1).unsqueeze(1)
        box_centre = (boxes[..., :2] - offset) / affine[:, 0, 0:1, None]
        box_scale = box_scale / affine[:, 0, 0:1, None]
        xyxy = torch.cat(
            [box_centre - box_scale * 0.5, box_centre + box_scale * 0.5], dim=-1
        )

        width = frame_size[:, 0].unsqueeze(-1).unsqueeze(-1)
        left_xyxy = torch.cat(
            [
                width - xyxy[:, 0:1, 2:3] - 1.0,
                xyxy[:, 0:1, 1:2],
                width - xyxy[:, 0:1, 0:1] - 1.0,
                xyxy[:, 0:1, 3:4],
            ],
            dim=-1,
        )
        right_xyxy = xyxy[:, 1:2]

        flipped = torch.flip(frame, dims=[3])

        left_centre = (left_xyxy[..., 0:2] + left_xyxy[..., 2:4]) * 0.5
        left_size = (left_xyxy[..., 2:4] - left_xyxy[..., 0:2]) * HAND_PADDING
        right_centre = (right_xyxy[..., 0:2] + right_xyxy[..., 2:4]) * 0.5
        right_size = (right_xyxy[..., 2:4] - right_xyxy[..., 0:2]) * HAND_PADDING

        left_affine = self._hand_affine(left_centre[:, 0], left_size[:, 0])
        right_affine = self._hand_affine(right_centre[:, 0], right_size[:, 0])
        # Both hands of every item go through one 2B forward. The crops are
        # independent, so this is the same arithmetic as 2B batch-1 calls with a
        # single ViT pass instead of 2B — worth ~a third of the runtime.
        hand_crop = torch.cat(
            [
                self._crop(flipped, left_affine, frame_size),
                self._crop(frame, right_affine, frame_size),
            ],
            dim=0,
        )
        hand_center = torch.cat([left_centre[:, 0], right_centre[:, 0]], dim=0)
        hand_size = torch.cat([left_size[:, 0], right_size[:, 0]], dim=0)
        hand_prompt = dummy.repeat(2, 1, 1)
        hand_frame = frame_size.repeat(2, 1)
        cam_batch = cam_int.repeat(2, 1, 1)
        hand_out = self.hand(
            hand_crop, hand_center, hand_size, cam_batch, hand_prompt, hand_frame
        )
        (
            lnormed, lpred, lcam_t, lverts, lj3d, lj2d, ldepth, lmp, lrots
        ) = [value[:count] for value in hand_out]
        (
            rnormed, rpred, rcam_t, rverts, rj3d, rj2d, rdepth, rmp, rrots
        ) = [value[count:] for value in hand_out]

        # -- unflip the left hand ----------------------------------------
        sm = self.scale_mean
        sc = self.scale_comps
        left_pred = self._unflip_scale(lpred, sm, sc)
        left_rots = self._unflip_rots(lrots)
        # `run_inference` does `lhand.hand[:, :54] = lhand.hand[:, 54:]`, so the
        # left slot ends up holding the flipped image's right-hand prediction.
        left_hand_comps = lpred[
            ..., self.hand_start + 54 : self.hand_start + 108
        ]

        # -- fusion criteria ----------------------------------------------
        lowarm = torch.tensor([76, 40], device=frame.device)
        wrist = torch.tensor([77, 41], device=frame.device)
        lowarm_rots = joint_rots0[:, lowarm]
        wrist_zero = lowarm_rots @ self.joint_rotation[wrist]
        ori_local = roma.euler_to_rotmat(
            "XZY", body_pose[:, [41, 43, 42, 31, 33, 32]].unflatten(1, (2, 3))
        )
        pred_global = torch.stack([left_rots[:, 78], rrots[:, 42]], dim=1)
        fused = torch.einsum("kabc,kabd->kadc", pred_global, wrist_zero)
        angle = rotation_angle_difference(ori_local, fused)
        valid_angle = angle < THRESH_WRIST_ANGLE

        box_ok = torch.stack(
            [
                (left_size.flatten(0, 1) > HAND_BOX_SIZE_THRESH).all(dim=1),
                (right_size.flatten(0, 1) > HAND_BOX_SIZE_THRESH).all(dim=1),
            ],
            dim=1,
        )
        kps_ok = torch.stack(
            [
                self._crop_max(lj2d, left_affine) < HAND_KPS2D_THRESH,
                self._crop_max(rj2d, right_affine) < HAND_KPS2D_THRESH,
            ],
            dim=1,
        )
        right_wrist = rj2d[:, [41]]
        left_wrist = lj2d[:, [41]]
        left_wrist = torch.cat(
            [width - left_wrist[..., 0:1] - 1.0, left_wrist[..., 1:2]], dim=-1
        )
        body_right = j2d0[:, [41]]
        body_left = j2d0[:, [62]]
        right_dist = (right_wrist - body_right).flatten(0, 1).norm(dim=-1) / (
            left_size.flatten(0, 1)[:, 0]
        )
        left_dist = (left_wrist - body_left).flatten(0, 1).norm(dim=-1) / (
            right_size.flatten(0, 1)[:, 0]
        )
        wrist_ok = torch.stack(
            [left_dist < HAND_WRIST_KPS2D_THRESH, right_dist < HAND_WRIST_KPS2D_THRESH],
            dim=1,
        )
        hand_valid = valid_angle & box_ok & kps_ok & wrist_ok

        # -- keypoint prompt + second body pass ---------------------------
        right_crop_kps = self._full_to_crop(right_wrist, affine)
        left_crop_kps = self._full_to_crop(left_wrist, affine)
        right_elbow = self._full_to_crop(j2d0[:, [8]], affine)
        left_elbow = self._full_to_crop(j2d0[:, [7]], affine)
        prompt = torch.cat(
            [right_crop_kps, left_crop_kps, right_elbow, left_elbow], dim=1
        )
        labels = torch.zeros(count, 4, 1, dtype=prompt.dtype, device=prompt.device)
        labels[:, 0, 0] = 41
        labels[:, 1, 0] = 62
        labels[:, 2, 0] = 8
        labels[:, 3, 0] = 7
        valid_key = (
            (prompt > -0.5) & (prompt < 0.5)
        ).all(dim=2) & hand_valid[:, [1, 0, 1, 0]].bool()
        # `where` instead of the reference's index assignment keeps the graph
        # shape-static; an invalid point becomes the dummy prompt (0, 0, -2).
        coords = torch.clamp(prompt + 0.5, min=0.0, max=1.0)
        coords = torch.where(valid_key.unsqueeze(-1), coords, torch.zeros_like(coords))
        labels = torch.where(valid_key.unsqueeze(-1), labels, -2.0 * torch.ones_like(labels))
        prompt = torch.cat([coords, labels], dim=-1)

        if os.environ.get("VM_DEBUG"):
            print("dbg prompt", tuple(prompt.shape), "valid", valid_key.tolist(),
                  flush=True)
            print("dbg angle", angle.tolist(), "box", box_ok.tolist(),
                  "kps", kps_ok.tolist(), "wrist", wrist_ok.tolist(), flush=True)

        prev_estimate = torch.cat(
            [pred[:, :266], shape, scale, hand, face, pred_cam], dim=1
        ).unsqueeze(1)
        # The reference reuses the first pass's backbone features here
        # (`run_keypoint_prompt` reads `output["image_embeddings"]`), so only the
        # decoder runs again — not the ViT.
        _n2, pred2, pred_cam2, _e2, _i2, _a2, _t2, _ta2 = self.body.run_decoder(
            _emb, _iseq, _aseq, bbox_center, bbox_scale, cam_int, prompt, prev_estimate
        )
        # `run_inference` only runs the prompt pass when at least one keypoint
        # survived the criteria; otherwise the first pass stands. Both are
        # computed, so the choice is one elementwise `where`.
        # Per item, not over the whole batch: one person's surviving prompt must
        # not switch another person's second pass on.
        use_prompt = valid_key.any(dim=1, keepdim=True)
        pred2 = torch.where(use_prompt, pred2, pred)
        pred_cam2 = torch.where(use_prompt, pred_cam2, pred_cam)
        global_rot2, body_pose2, shape2, scale2, hand2, face2 = pred_parts(
            self.head, pred2
        )
        _s2, model_params2, _f2 = params_from_pred_body(self.head, pred2)
        verts2, skel2 = self.mhr(_s2, model_params2, _f2)
        j3d2 = keypoints_from_mhr(self.head.keypoint_mapping, verts2, skel2)
        j2d2, depth2, cam_t2, focal2 = project_keypoints_cam(
            j3d2, pred_cam2, bbox_center, bbox_scale[:, 0], cam_int
        )
        joint_rots2 = roma.unitquat_to_rotmat(skel2[:, :, 3:7])

        # -- fuse hands ---------------------------------------------------
        updated_hand = torch.cat(
            [
                left_hand_comps,
                rpred[..., self.hand_start + 54 : self.hand_start + 108],
            ],
            dim=-1,
        )
        updated_scale = scale2.clone()
        updated_scale = self._set_col(updated_scale, 9, left_pred[:, self.scale_start + 9])
        updated_scale = self._set_col(updated_scale, 8, rpred[:, self.scale_start + 8])
        shared_scale = (
            left_pred[:, self.scale_start + 18 : self.scale_start + self.scale_len]
            + rpred[:, self.scale_start + 18 : self.scale_start + self.scale_len]
        ) * 0.5
        updated_scale = torch.cat(
            [updated_scale[:, :18], shared_scale], dim=1
        )
        updated_shape = shape2.clone()
        shared_shape = (
            left_pred[:, self.shape_start + 40 : self.shape_start + self.shape_len]
            + rpred[:, self.shape_start + 40 : self.shape_start + self.shape_len]
        ) * 0.5
        updated_shape = torch.cat([updated_shape[:, :40], shared_shape], dim=1)

        _v, _s, _mp, ik_rots, _j = self.mhr_forward(
            global_rot2, body_pose2, updated_hand, updated_scale, updated_shape, face2
        )
        lowarm_rots2 = ik_rots[:, lowarm]
        wrist_zero2 = lowarm_rots2 @ self.joint_rotation[wrist]
        pred_global2 = torch.stack([left_rots[:, 78], rrots[:, 42]], dim=1)
        fused2 = torch.einsum("kabc,kabd->kadc", pred_global2, wrist_zero2)
        wrist_xzy = fix_wrist_euler(roma.rotmat_to_euler("XZY", fused2))
        angle2 = rotation_angle_difference(ori_local, fused2)
        keep = (angle2 < THRESH_WRIST_ANGLE) & hand_valid
        keep_e = keep.unsqueeze(-1)

        body_pair = body_pose2[:, [41, 43, 42, 31, 33, 32]].unflatten(1, (2, 3))
        body_new = torch.where(keep_e, wrist_xzy, body_pair).flatten(1, 2)
        body_pose3 = self._set_cols(
            body_pose2, [41, 43, 42, 31, 33, 32], body_new
        )
        hand_pair = hand2.unflatten(1, (2, 54))
        hand_new = torch.where(
            keep_e, updated_hand.unflatten(1, (2, 54)), hand_pair
        ).flatten(1, 2)
        scale_pair = torch.stack(
            [scale2[:, 9], scale2[:, 8]], dim=1
        )
        updated_pair = torch.stack(
            [updated_scale[:, 9], updated_scale[:, 8]], dim=1
        )
        masked_pair = torch.where(keep, updated_pair, scale_pair)
        scale3 = self._set_col(scale2, 9, masked_pair[:, 0])
        scale3 = self._set_col(scale3, 8, masked_pair[:, 1])
        count = keep.float().sum(dim=1, keepdim=True)
        blended_scale = (
            left_pred[:, self.scale_start + 18 : self.scale_start + self.scale_len]
            * keep[:, [0]]
            + rpred[:, self.scale_start + 18 : self.scale_start + self.scale_len]
            * keep[:, [1]]
        ) / (count + 1e-8)
        scale3 = torch.cat(
            [
                scale3[:, :18],
                torch.where(count > 0, blended_scale, scale3[:, 18:]),
            ],
            dim=1,
        )
        blended_shape = (
            left_pred[:, self.shape_start + 40 : self.shape_start + self.shape_len]
            * keep[:, [0]]
            + rpred[:, self.shape_start + 40 : self.shape_start + self.shape_len]
            * keep[:, [1]]
        ) / (count + 1e-8)
        shape3 = torch.cat(
            [
                shape2[:, :40],
                torch.where(count > 0, blended_shape, shape2[:, 40:]),
            ],
            dim=1,
        )

        verts3, skel3, model_params3, joint_rots3, _j3 = self.mhr_forward(
            global_rot2, body_pose3, hand_new, scale3, shape3, face2
        )
        sign = torch.tensor([1.0, -1.0, -1.0], dtype=verts3.dtype, device=verts3.device)
        verts = verts3 / 100 * sign
        j3d = _j3
        # Final projection: `run_inference` uses the *image centre*, not cam_int.
        proj = j3d + cam_t2[:, None, :]
        size_x = frame_size[:, 0:1]
        size_y = frame_size[:, 1:2]
        x = proj[..., 0] * focal2[:, None] + size_x * 0.5 * proj[..., 2]
        y = proj[..., 1] * focal2[:, None] + size_y * 0.5 * proj[..., 2]
        j2d = torch.stack([x / proj[..., 2], y / proj[..., 2]], dim=-1)
        return (
            verts,
            j3d,
            j2d,
            pred_cam2,
            cam_t2,
            focal2,
            model_params3,
            joint_rots3,
            hand_valid.float(),
            normed[:, 0],
            lnormed[:, 0],
            rnormed[:, 0],
            # Raw MHR skeleton state (127 x 8, centimetres, MHR axes) and the
            # 519-value prediction the final mesh was decoded from: the same
            # fields `velomocap_infer::Sam3dFrame` carries, so the Rust side can
            # treat this module and the ONNX chain interchangeably.
            skel3,
            pred2,
        )

    # -- small tensor helpers --------------------------------------------
    def _full_to_crop(self, kps2d, affine):
        ones = torch.ones_like(kps2d[..., :1])
        hom = torch.cat([kps2d, ones], dim=-1)
        out = hom @ affine.mT
        return out[..., :2] / IMAGE_SIZE - 0.5

    def _crop_max(self, j2d, affine):
        crop = self._full_to_crop(j2d, affine)
        return crop.abs().amax(dim=(1, 2))

    def _unflip_scale(self, pred, mean, comps):
        """`run_inference` swaps the right-hand scale onto the left slot."""
        value = self._get_col(pred, self.scale_start + 9)
        right_mean = mean[8]
        left_mean = mean[9]
        right_std = comps[8, 8]
        left_std = comps[9, 9]
        flipped = (
            (right_mean + right_std * self._get_col(pred, self.scale_start + 8))
            - left_mean
        ) / left_std
        return self._set_col(pred, self.scale_start + 9, flipped)

    def _unflip_rots(self, rots):
        """`joint_global_rots[:, 78] = joint_global_rots[:, 42]` with y/z negated."""
        mirrored = rots[:, 42]
        sign = torch.tensor([1.0, -1.0, -1.0], dtype=rots.dtype, device=rots.device)
        mirrored = mirrored * sign[None, :, None]
        return self._put_row(rots, 78, mirrored)

    @staticmethod
    def _get_col(tensor, index):
        return tensor[:, index]

    @staticmethod
    def _set_col(tensor, index, value):
        cols = [tensor[:, i : i + 1] for i in range(tensor.shape[1])]
        cols[index] = value.unsqueeze(-1) if value.dim() == 1 else value
        return torch.cat(cols, dim=1)

    @staticmethod
    def _set_cols(tensor, indices, values):
        for slot, index in enumerate(indices):
            tensor = Sam3dFull._set_col(tensor, index, values[:, slot])
        return tensor

    @staticmethod
    def _put_row(tensor, index, value):
        rows = [tensor[:, i : i + 1] for i in range(tensor.shape[1])]
        rows[index] = value.unsqueeze(1)
        return torch.cat(rows, dim=1)
