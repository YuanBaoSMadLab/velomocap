"""Export SAM-3D-Body to ONNX for FlowHMR feature extraction.

FlowHMR only consumes the 3×1024 conditioning tokens SAM-3D-Body produces per
frame (body + left hand + right hand). This script exports two graphs:

  sam3d_body.onnx
      crop image [1,3,512,512], ray_cond [1,2,512,512], affine_trans [1,3,3],
      bbox_center [1,2], bbox_scale [1,2], ori_img_size [1,2], cam_int [1,3,3],
      keypoints [1,1,3]
      -> body_token [1,1024], hand_box [1,2,4]

  sam3d_hand.onnx
      same inputs (the crop is a hand crop) -> hand_token [1,1024]

Everything the reference does in `forward_decoder` is reproduced here explicitly
so the traced graph has no Python-side state.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGE_SIZE = 512
DECODER_DIM = 1024
NUM_KEYPOINTS = 70
NUM_HAND_TOKENS = 2


def _ray_downsample(rays: torch.Tensor, scale: float) -> torch.Tensor:
    """Exportable replacement for `F.interpolate(..., antialias=True)`.

    ONNX has no lowering for `aten::_upsample_bilinear2d_aa`. For the integer
    downsampling factor used here (512 → 32, i.e. exactly one patch) an average
    pool is the box-filter counterpart of the antialiased bilinear kernel, and
    the ray field is smooth enough that the two agree to well under a percent.
    """
    factor = int(round(1.0 / scale))
    if factor >= 1 and abs(1.0 / factor - scale) < 1e-6:
        return F.avg_pool2d(rays, kernel_size=factor, stride=factor)
    return F.interpolate(rays, scale_factor=scale, mode="bilinear", align_corners=False)


def patch_camera_encoder_for_export():
    """Route `CameraEncoder.forward` through the exportable downsampler."""
    import sam_3d_body.models.modules.camera_embed as camera_embed

    encoder_cls = camera_embed.CameraEncoder
    original_forward = encoder_cls.forward

    def forward(self, img_embeddings, rays):
        b, d, h, w = img_embeddings.shape
        scale = 1 / self.patch_size
        rays = _ray_downsample(rays, scale)
        rays = rays.permute(0, 2, 3, 1).contiguous()
        rays = torch.cat([rays, torch.ones_like(rays[..., :1])], dim=-1)
        emb = self.camera(pos=rays.reshape(b, -1, 3))
        emb = emb.reshape(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        z = torch.cat([img_embeddings, emb], dim=1)
        return self.norm(self.conv(z))

    encoder_cls.forward = forward
    return original_forward


def patch_roma_for_export():
    """Replace the roma conversions MHR uses with traceable equivalents.

    Upstream `roma` relies on `x.diagonal(...)`, `flatten(0, end_dim-1)` and
    in-place masked assignment, none of which survive ONNX export without a
    known rank. These rewrites keep the same maths with fully static shapes:

    * `rotmat_to_unitquat` — static indexing of the 4×4 candidate matrix.
    * `unitquat_to_rotmat` — the standard XYZW expansion.
    * `unitquat_to_euler` — scipy's algorithm with `where` instead of in-place
      scatter, using Python-int axis indices from the convention string.
    * `rotmat_to_euler` / `euler_to_rotmat` — routed through the above.
    """
    import math

    import roma

    def rotmat_to_unitquat(R):
        (a11, a12, a13), (a21, a22, a23), (a31, a32, a33) = (
            row.unbind(-1) for row in R.transpose(-1, -2).unbind(-2)
        )
        x = torch.stack(
            [
                1 + a11 - a22 - a33, a12 + a21, a13 + a31, a23 - a32,
                a21 + a12, 1 + a22 - a33 - a11, a23 + a32, a31 - a13,
                a31 + a13, a32 + a23, 1 + a33 - a11 - a22, a12 - a21,
                a23 - a32, a31 - a13, a12 - a21, 1 + a11 + a22 + a33,
            ],
            dim=-1,
        ).unflatten(-1, (4, 4))
        diag = torch.stack(
            [x[..., 0, 0], x[..., 1, 1], x[..., 2, 2], x[..., 3, 3]], dim=-1
        )
        i = diag.argmax(dim=-1)
        x_i = torch.gather(x, -2, i[..., None, None].expand(*i.shape, 1, 4))[..., 0, :]
        return x_i / torch.norm(x_i, dim=-1, keepdim=True)

    def unitquat_to_rotmat(q):
        # q is XYZW.
        i, j, k, r = q.unbind(-1)
        two_s = 2.0 / (q * q).sum(-1).clamp_min(1e-12)
        rows = [
            torch.stack(
                [
                    1 - two_s * (j * j + k * k),
                    two_s * (i * j - k * r),
                    two_s * (i * k + j * r),
                ],
                dim=-1,
            ),
            torch.stack(
                [
                    two_s * (i * j + k * r),
                    1 - two_s * (i * i + k * k),
                    two_s * (j * k - i * r),
                ],
                dim=-1,
            ),
            torch.stack(
                [
                    two_s * (i * k - j * r),
                    two_s * (j * k + i * r),
                    1 - two_s * (i * i + j * j),
                ],
                dim=-1,
            ),
        ]
        return torch.stack(rows, dim=-2)

    def _basis_index(axis: str) -> int:
        return {"x": 0, "y": 1, "z": 2}[axis]

    def unitquat_to_euler(convention, quat, as_tuple=False, degrees=False, epsilon=1e-7):
        assert len(convention) == 3
        pi = math.pi
        half_pi = math.pi / 2

        extrinsic = convention.islower()
        if not extrinsic:
            convention = convention.lower()[::-1]

        batch_shape = quat.shape[:-1]
        q = quat.reshape(-1, 4)

        i = _basis_index(convention[0])
        j = _basis_index(convention[1])
        k = _basis_index(convention[2])
        assert i != j and j != k
        symmetric = i == k
        if symmetric:
            k = 3 - i - j
        sign = (i - j) * (j - k) * (k - i) // 2

        if symmetric:
            a = q[:, 3]
            b = q[:, i]
            c = q[:, j]
            d = q[:, k] * sign
        else:
            a = q[:, 3] - q[:, j]
            b = q[:, i] + q[:, k] * sign
            c = q[:, j] + q[:, 3]
            d = q[:, k] * sign - q[:, i]

        angle_first, angle_third = (0, 2) if extrinsic else (2, 0)

        second = 2 * torch.atan2(torch.hypot(c, d), torch.hypot(a, b))
        case1 = second.abs() <= epsilon
        case2 = (second - pi).abs() <= epsilon
        case12 = case1 | case2

        half_sum = torch.atan2(b, a)
        half_diff = torch.atan2(d, c)

        angles = [None, second, None]
        angles[angle_first] = half_sum - half_diff
        angles[angle_third] = half_sum + half_diff

        zero = torch.zeros_like(second)
        angles[2] = torch.where(case12, zero, angles[2])
        angles[0] = torch.where(case1, 2 * half_sum, angles[0])
        angles[0] = torch.where(
            case2, 2 * (-1 if extrinsic else 1) * half_diff, angles[0]
        )

        if not symmetric:
            angles[angle_third] = angles[angle_third] * sign
            angles[1] = angles[1] - half_pi

        out = []
        for angle in angles:
            angle = torch.where(angle < -pi, angle + 2 * pi, angle)
            angle = torch.where(angle > pi, angle - 2 * pi, angle)
            if degrees:
                angle = torch.rad2deg(angle)
            out.append(angle.reshape(*batch_shape))

        return tuple(out) if as_tuple else torch.stack(out, dim=-1)

    def rotmat_to_euler(convention, rotmat, as_tuple=False, degrees=False, epsilon=1e-7):
        return unitquat_to_euler(
            convention,
            rotmat_to_unitquat(rotmat),
            as_tuple=as_tuple,
            degrees=degrees,
            epsilon=epsilon,
        )

    def _axis_matrix(axis: int, angle):
        one = torch.ones_like(angle)
        zero = torch.zeros_like(angle)
        cos, sin = torch.cos(angle), torch.sin(angle)
        if axis == 0:
            rows = [
                torch.stack([one, zero, zero], -1),
                torch.stack([zero, cos, -sin], -1),
                torch.stack([zero, sin, cos], -1),
            ]
        elif axis == 1:
            rows = [
                torch.stack([cos, zero, sin], -1),
                torch.stack([zero, one, zero], -1),
                torch.stack([-sin, zero, cos], -1),
            ]
        else:
            rows = [
                torch.stack([cos, -sin, zero], -1),
                torch.stack([sin, cos, zero], -1),
                torch.stack([zero, zero, one], -1),
            ]
        return torch.stack(rows, -2)

    def euler_to_rotmat(convention, angles, degrees=False, **kwargs):
        assert len(convention) == 3
        if degrees:
            angles = torch.deg2rad(angles)
        extrinsic = convention.islower()
        axes = [_basis_index(c.lower()) for c in convention]
        a = [angles[..., idx] for idx in range(3)]

        matrices = [_axis_matrix(axes[idx], a[idx]) for idx in range(3)]
        if extrinsic:
            # Extrinsic x1,x2,x3 composes as R3 @ R2 @ R1.
            return matrices[2] @ matrices[1] @ matrices[0]
        return matrices[0] @ matrices[1] @ matrices[2]

    roma.rotmat_to_unitquat = rotmat_to_unitquat
    roma.unitquat_to_rotmat = unitquat_to_rotmat
    roma.unitquat_to_euler = unitquat_to_euler
    roma.rotmat_to_euler = rotmat_to_euler
    roma.euler_to_rotmat = euler_to_rotmat
    roma.euler_to_unitquat = lambda convention, angles, **kw: rotmat_to_unitquat(
        euler_to_rotmat(convention, angles, **kw)
    )
    roma.euler_to_rotmat.__name__ = "euler_to_rotmat"


def _bmask(count: int, flag: bool) -> torch.Tensor:
    """`torch.ones(count).bool() * flag` without the Tensor×bool multiply.

    ONNX has no lowering for `aten::mul(Tensor, bool)`, and the constant fold
    that produces the mask is trivial anyway.
    """
    value = bool(flag)
    return torch.ones(count, dtype=torch.bool) if value else torch.zeros(count, dtype=torch.bool)


def register_onnx_symbolics():
    """Fill the gaps in torch's ONNX symbolic table for MHR.

    The set lives next to this script so it can be exercised against
    `mhr_model.pt` alone, which is seconds instead of minutes per iteration.
    Shape inference is additionally made non-fatal: a node it cannot type is
    still valid ONNX, and the alternative is losing the whole export.
    """
    import mhr_symbolics
    from onnx_export_safe import patch_safe_shape_inference

    mhr_symbolics.register()
    patch_safe_shape_inference()


def patch_mhr_utils_for_export():
    """Make `mhr_utils` traceable.

    Two things in `compact_cont_to_model_params_hand` /
    `compact_model_params_to_cont_hand` block export:

    * `hand_dofs_in_order` is a tensor, so `k in [3]` becomes a data-dependent
      guard and `torch.ones(2 * k)` a tensor-shaped constant.
    * `torch.ones(k).bool() * (k in [3])` produces `aten::mul(Tensor, bool)`,
      which has no ONNX lowering.

    Both are pure constants. Rewriting the module source with a plain Python
    list and `torch.full` gives bit-identical results and exports cleanly.
    """
    import inspect
    import re

    import sam_3d_body.models.heads.mhr_head as mhr_head
    import sam_3d_body.models.modules.mhr_utils as mhr_utils

    def rewrite(fn):
        source = inspect.getsource(fn)
        source = source.replace(
            "torch.tensor([3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 2, 3, 1, 1])",
            "[3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 2, 3, 1, 1]",
        )
        source = re.sub(
            r"torch\.ones\(([^)]*)\)\.bool\(\) \* \((k in \[[^\]]*\])\)",
            r"_bmask(\1, \2)",
            source,
        )
        # Module-level functions are already at column zero, so the source
        # compiles as-is.
        namespace = dict(mhr_utils.__dict__)
        namespace["_bmask"] = _bmask
        exec(compile(source, "<patched>", "exec"), namespace)  # noqa: S102
        return namespace[fn.__name__]

    for name in (
        "compact_cont_to_model_params_hand",
        "compact_model_params_to_cont_hand",
    ):
        patched = rewrite(getattr(mhr_utils, name))
        setattr(mhr_utils, name, patched)
        if hasattr(mhr_head, name):
            setattr(mhr_head, name, patched)

    # Even with the constant folding above, the source still selects with
    # boolean masks (`hand_cont[..., mask]`) and assigns back the same way.
    # torch lowers a byte mask to `aten::index`, which becomes a `ScatterND`
    # with an index the runtime rejects, and warns that the graph will be
    # wrong. Both masks are constants, so integer index tensors say the same
    # thing and export as `Gather` / `ScatterElements`.
    hand_cont_to_model = _hand_cont_to_model_params(mhr_utils.batchXYZfrom6D)
    setattr(mhr_utils, "compact_cont_to_model_params_hand", hand_cont_to_model)
    setattr(mhr_head, "compact_cont_to_model_params_hand", hand_cont_to_model)

    _patch_body_param_builder(mhr_utils, mhr_head)
    _patch_functional_normalize()


def _patch_functional_normalize():
    """`F.normalize` without its internal `expand_as`.

    torch implements it as `input / norm(input).clamp_min(eps).expand_as(input)`.
    The `expand_as` lowering builds its target shape by merging two shape
    vectors, and the emitted `Concat` loses its first input — the exported graph
    then fails to load with "input 0 ... has an empty string". Dividing by the
    `keepdim=True` norm broadcasts to the same result without any shape
    arithmetic.
    """
    import torch.nn.functional as F

    if getattr(F.normalize, "_velomocap_patch", False):
        return
    original = F.normalize

    def normalize(input, p=2.0, dim=1, eps=1e-12, out=None):  # noqa: A002
        if out is not None:
            return original(input, p, dim, eps, out)
        denom = input.norm(p, dim, keepdim=True).clamp_min(eps)
        return input / denom

    normalize._velomocap_patch = True
    F.normalize = normalize


def _patch_body_param_builder(mhr_utils, mhr_head):
    """`compact_cont_to_model_params_body` with static reshape targets.

    Two constructs in the original export to a broken graph — ONNX Runtime
    refuses to load the result with "input 0 ... has an empty string":

    * `unflatten(-1, (-1, 6))` and `unflatten(-1, (-1, 2))`. A `-1` inside an
      `unflatten` makes torch build the target shape from `aten::size`
      arithmetic, and the emitted `Concat` loses its first input. The counts are
      fixed by the layout (23 three-DoF and 58 one-DoF body joints), so naming
      them removes the arithmetic entirely.
    * `torch.zeros(*body_pose_cont.shape[:-1], 133)` — same size-list problem.
      Slicing the input says the same thing and exports as
      `ConstantOfShape(Shape(x))`.

    The result is checked against the original before it is installed.
    """
    import inspect

    source = inspect.getsource(mhr_utils.compact_cont_to_model_params_body)
    patched_source = source.replace(
        "torch.zeros(*body_pose_cont.shape[:-1], 133).to(body_pose_cont)",
        "torch.zeros_like(body_pose_cont[..., :133])",
    ).replace(
        "body_cont_3dofs.unflatten(-1, (-1, 6))",
        "body_cont_3dofs.unflatten(-1, (_BODY_3DOF, 6))",
    ).replace(
        "body_cont_1dofs.unflatten(-1, (-1, 2))",
        "body_cont_1dofs.unflatten(-1, (_BODY_1DOF, 2))",
    )
    if patched_source == source:
        raise RuntimeError(
            "compact_cont_to_model_params_body changed upstream; update the patch"
        )

    namespace = dict(mhr_utils.__dict__)
    namespace["_BODY_3DOF"] = 23
    namespace["_BODY_1DOF"] = 58
    exec(compile(patched_source, "<patched>", "exec"), namespace)  # noqa: S102
    builder = namespace["compact_cont_to_model_params_body"]

    generated = torch.Generator().manual_seed(11)
    sample = torch.randn(3, 260, generator=generated)
    expected = mhr_utils.compact_cont_to_model_params_body(sample)
    actual = builder(sample)
    if not torch.equal(expected, actual):
        raise RuntimeError(
            "compact_cont_to_model_params_body patch changed the result; "
            f"max diff {(expected - actual).abs().max().item():.3e}"
        )

    setattr(mhr_utils, "compact_cont_to_model_params_body", builder)
    setattr(mhr_head, "compact_cont_to_model_params_body", builder)


def _hand_cont_to_model_params(batch_xyz_from_6d):
    """`compact_cont_to_model_params_hand` with integer indices.

    Same maths, same order, no boolean masks: the 3-DoF joints contribute six
    continuous values that become one 6D rotation, the rest contribute sin/cos
    pairs that become one or two angles.
    """
    dofs = [3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 2, 3, 1, 1]
    cont_threedofs, cont_onedofs, out_threedofs, out_onedofs = [], [], [], []
    offset_cont = 0
    offset_out = 0
    for dof in dofs:
        if dof == 3:
            cont_threedofs += list(range(offset_cont, offset_cont + 6))
            out_threedofs += list(range(offset_out, offset_out + 3))
        else:
            cont_onedofs += list(range(offset_cont, offset_cont + 2 * dof))
            out_onedofs += list(range(offset_out, offset_out + dof))
        offset_cont += 2 * dof
        offset_out += dof

    idx_cont_3 = torch.tensor(cont_threedofs, dtype=torch.long)
    idx_cont_1 = torch.tensor(cont_onedofs, dtype=torch.long)
    idx_out_3 = torch.tensor(out_threedofs, dtype=torch.long)
    idx_out_1 = torch.tensor(out_onedofs, dtype=torch.long)

    def compact(hand_cont):
        if hand_cont.shape[-1] != 54:
            raise AssertionError("hand_cont must have 54 columns")
        # Keep the index tables on the input's device: the same patch is used by
        # the TorchScript path, which can run on CUDA.
        device = hand_cont.device
        idx_3 = idx_cont_3.to(device)
        idx_1 = idx_cont_1.to(device)
        # Explicit counts: a `-1` in `unflatten` makes torch derive the target
        # shape from `aten::size` arithmetic, which exports to a graph ONNX
        # Runtime will not load.
        threedofs = hand_cont.index_select(-1, idx_3).unflatten(
            -1, (len(out_threedofs) // 3, 6)
        )
        three = batch_xyz_from_6d(threedofs).flatten(-2, -1)
        onedofs = hand_cont.index_select(-1, idx_1).unflatten(
            -1, (len(out_onedofs), 2)
        )
        one = torch.atan2(onedofs[..., -2], onedofs[..., -1])

        # Derive the destination from a slice instead of building it from a
        # list of dynamic sizes: `torch.zeros(*x.shape[:-1], 27)` lowers through
        # `aten::zeros`, whose size list exports with a missing input.
        out = torch.zeros_like(hand_cont[..., :27])
        batch = hand_cont.shape[:-1]
        out = out.scatter(-1, idx_out_3.to(device).expand(*batch, -1), three)
        out = out.scatter(-1, idx_out_1.to(device).expand(*batch, -1), one)
        return out

    return compact


class BodyParts(nn.Module):
    """The submodules the export actually uses.

    `SAM3DBody` is a LightningModule, and `torch.jit.trace` refuses to descend
    into it (its `trainer` property raises). Registering the individual modules
    on a plain container keeps the traced tree clean while still letting the
    scripted MHR submodule take part in the trace.
    """

    def __init__(self, model):
        super().__init__()
        for name in (
            "backbone",
            "head_pose",
            "init_pose",
            "head_camera",
            "init_camera",
            "init_to_token_mhr",
            "prev_to_token_mhr",
            "prompt_encoder",
            "prompt_to_token",
            "decoder",
            "ray_cond_emb",
            "keypoint_embedding",
            "hand_box_embedding",
            "bbox_embed",
            "keypoint_posemb_linear",
            "keypoint_feat_linear",
            "keypoint3d_embedding",
            "keypoint3d_posemb_linear",
        ):
            setattr(self, name, getattr(model, name))

        for name in (
            "head_pose_hand",
            "init_pose_hand",
            "head_camera_hand",
            "init_camera_hand",
            "init_to_token_mhr_hand",
            "prev_to_token_mhr_hand",
            "decoder_hand",
            "hand_pe_layer",
            "ray_cond_emb_hand",
            "keypoint_embedding_hand",
            "keypoint_posemb_linear_hand",
            "keypoint_feat_linear_hand",
            "keypoint3d_embedding_hand",
            "keypoint3d_posemb_linear_hand",
        ):
            setattr(self, name, getattr(model, name))

        self.register_buffer("image_mean", model.image_mean.clone())
        self.register_buffer("image_std", model.image_std.clone())


class SAM3DBodyExport(nn.Module):
    """`forward_decoder` (+ `forward_decoder_hand`) with explicit tensor inputs."""

    def __init__(self, model, *, hand: bool, hand_scale_factor: float = 1.0):
        super().__init__()
        self.model = BodyParts(model)
        self.hand = hand
        self.hand_scale_factor = hand_scale_factor
        self.num_layers = len(model.decoder_hand.layers if hand else model.decoder.layers)

    # -- helpers ---------------------------------------------------------
    def _condition(self, bbox_center, bbox_scale, ori_img_size, cam_int):
        """CLIFF condition with `USE_INTRIN_CENTER=True`."""
        cx, cy = bbox_center[:, 0:1], bbox_center[:, 1:2]
        img_w, img_h = ori_img_size[:, 0:1], ori_img_size[:, 1:2]
        b = bbox_scale[:, 0:1]
        focal = cam_int[:, 0, 0:1]
        full_cx = cam_int[:, 0, 2:3]
        full_cy = cam_int[:, 1, 2:3]
        cond = torch.cat([cx - full_cx, cy - full_cy, b], dim=-1)
        cond = torch.cat(
            [cond[:, 0:2] / focal, cond[:, 2:3] / focal], dim=-1
        )
        return cond

    def _full_to_crop(self, kps2d, affine_trans, img_size):
        homogeneous = torch.cat([kps2d, torch.ones_like(kps2d[:, :, -1:])], dim=-1)
        out = homogeneous @ affine_trans.mT
        return out[..., :2] / img_size.unsqueeze(1) - 0.5

    def _perspective_project(self, kps3d, pred_cam, bbox_center, bbox_scale, ori_img_size, cam_int):
        head = self.model.head_pose_hand if self.hand else self.model.head_pose
        cam_head = self.model.head_camera_hand if self.hand else self.model.head_camera
        return cam_head.perspective_projection(
            kps3d,
            pred_cam,
            bbox_center,
            bbox_scale,
            ori_img_size,
            cam_int,
            use_intrin_center=True,
        )

    def _prompt_embeddings(self, keypoints):
        """Export-friendly `PromptEncoder._embed_keypoints`.

        The reference version asserts on tensor values and scatters into the
        embedding with boolean masks, neither of which survives tracing. FlowHMR
        always sends the single "invalid" dummy prompt, so a `where` over the
        three label cases reproduces it exactly without any data-dependent
        control flow.
        """
        enc = self.model.prompt_encoder
        coords = keypoints[:, :, :2]
        labels = keypoints[:, :, -1]

        pe = enc.pe_layer._pe_encoding(coords.to(torch.float))
        invalid = (labels == -2).unsqueeze(-1)
        not_a_point = (labels == -1).unsqueeze(-1)
        valid = (~invalid & ~not_a_point).unsqueeze(-1)

        point_embedding = torch.where(
            invalid,
            enc.invalid_point_embed.weight.view(1, 1, -1).expand_as(pe),
            pe,
        )
        point_embedding = torch.where(
            not_a_point,
            enc.not_a_point_embed.weight.view(1, 1, -1).expand_as(pe),
            point_embedding,
        )
        # Label 0..69 selects a joint embedding; the dummy prompt never uses it.
        joint = enc.point_embeddings[0].weight.view(1, 1, -1).expand_as(pe)
        point_embedding = torch.where(valid, point_embedding + joint, point_embedding)
        return point_embedding

    # -- one decoder pass ------------------------------------------------
    def _decode(
        self,
        image_embeddings,
        image_augment,
        condition_info,
        keypoints,
        affine_trans,
        bbox_center,
        bbox_scale,
        ori_img_size,
        cam_int,
        ray_cond,
    ):
        m = self.model
        decoder = m.decoder_hand if self.hand else m.decoder
        head_pose = m.head_pose_hand if self.hand else m.head_pose
        head_camera = m.head_camera_hand if self.hand else m.head_camera
        init_pose = m.init_pose_hand if self.hand else m.init_pose
        init_camera = m.init_camera_hand if self.hand else m.init_camera
        init_to_token = m.init_to_token_mhr_hand if self.hand else m.init_to_token_mhr
        prev_to_token = m.prev_to_token_mhr_hand if self.hand else m.prev_to_token_mhr
        posemb_linear = m.keypoint_posemb_linear_hand if self.hand else m.keypoint_posemb_linear
        feat_linear = m.keypoint_feat_linear_hand if self.hand else m.keypoint_feat_linear
        kps3d_posemb = (
            m.keypoint3d_posemb_linear_hand if self.hand else m.keypoint3d_posemb_linear
        )
        kps_embedding = m.keypoint_embedding_hand if self.hand else m.keypoint_embedding
        kps3d_embedding = m.keypoint3d_embedding_hand if self.hand else m.keypoint3d_embedding

        batch_size = image_embeddings.shape[0]

        prev_pose = init_pose.weight.expand(batch_size, -1)
        prev_camera = init_camera.weight.expand(batch_size, -1)
        init_estimate = torch.cat([prev_pose, prev_camera], dim=-1).unsqueeze(1)

        init_input = torch.cat([condition_info.view(batch_size, 1, -1), init_estimate], dim=-1)
        token_embeddings = init_to_token(init_input).view(batch_size, 1, -1)

        prev_embeddings = prev_to_token(init_estimate).view(batch_size, 1, -1)

        image_embeddings = (
            m.ray_cond_emb_hand(image_embeddings, ray_cond)
            if self.hand
            else m.ray_cond_emb(image_embeddings, ray_cond)
        )

        prompt_embeddings = self._prompt_embeddings(keypoints)
        # `_embed_keypoints` returns one embedding per prompt token; squeeze the
        # singleton prompt axis so it lines up with the pose tokens.
        prompt_embeddings = prompt_embeddings.reshape(prompt_embeddings.shape[0], -1)
        prompt_embeddings = m.prompt_to_token(prompt_embeddings).unsqueeze(1)

        base = torch.cat([token_embeddings, prev_embeddings, prompt_embeddings], dim=1)
        head_augment = torch.cat(
            [
                torch.zeros_like(token_embeddings),
                prev_embeddings,
                prompt_embeddings,
            ],
            dim=1,
        )

        def zeros_like_tokens(count, reference):
            return torch.zeros(
                (batch_size, count, reference.shape[-1]),
                dtype=reference.dtype,
                device=reference.device,
            )

        # Hand-detection tokens (body decoder only).
        hand_det_start = base.shape[1]
        if not self.hand:
            hand_tokens = m.hand_box_embedding.weight[None].repeat(batch_size, 1, 1)
            base = torch.cat([base, hand_tokens], dim=1)
            head_augment = torch.cat(
                [head_augment, zeros_like_tokens(hand_tokens.shape[1], base)], dim=1
            )

        # Keypoint tokens.
        kps_start = base.shape[1]
        kps_tokens = kps_embedding.weight[None].repeat(batch_size, 1, 1)
        base = torch.cat([base, kps_tokens], dim=1)
        head_augment = torch.cat(
            [head_augment, zeros_like_tokens(kps_tokens.shape[1], base)], dim=1
        )

        # 3-D keypoint tokens.
        kps3d_start = base.shape[1]
        kps3d_tokens = kps3d_embedding.weight[None].repeat(batch_size, 1, 1)
        base = torch.cat([base, kps3d_tokens], dim=1)
        head_augment = torch.cat(
            [head_augment, zeros_like_tokens(kps3d_tokens.shape[1], base)], dim=1
        )

        token_embeddings, token_augment = base, head_augment
        kps_end = kps_start + kps_tokens.shape[1]
        kps3d_end = kps3d_start + kps3d_tokens.shape[1]

        def pose_output_of(tokens):
            pose_token = tokens[:, 0]
            pose_output = head_pose(pose_token, prev_pose)
            pred_cam = head_camera(pose_token, prev_camera)
            pose_output["pred_cam"] = pred_cam
            cam_out = self._perspective_project(
                pose_output["pred_keypoints_3d"],
                pred_cam,
                bbox_center,
                bbox_scale[:, 0],
                ori_img_size,
                cam_int,
            )
            pose_output.update(cam_out)
            pose_output["pred_keypoints_2d_cropped"] = self._full_to_crop(
                pose_output["pred_keypoints_2d"], affine_trans, img_size_t
            )
            return pose_output

        img_size_t = torch.full(
            (batch_size, 1), float(IMAGE_SIZE), dtype=image_embeddings.dtype
        )

        def keypoint_update(tokens, augment, pose_output, layer_idx):
            if layer_idx == self.num_layers - 1:
                return tokens, augment

            kps2d = pose_output["pred_keypoints_2d_cropped"]
            kps_depth = pose_output["pred_keypoints_2d_depth"]
            kps01 = kps2d + 0.5
            invalid = (
                (kps01[:, :, 0] < 0)
                | (kps01[:, :, 0] > 1)
                | (kps01[:, :, 1] < 0)
                | (kps01[:, :, 1] > 1)
                | (kps_depth[:, :] < 1e-5)
            )
            valid = (~invalid).to(tokens.dtype).unsqueeze(-1)

            augment = torch.cat(
                [
                    augment[:, :kps_start],
                    posemb_linear(kps2d) * valid,
                    augment[:, kps_end:],
                ],
                dim=1,
            )

            sample_points = kps2d * 2
            sampled = F.grid_sample(
                image_embeddings,
                sample_points[:, :, None, :],
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            ).squeeze(3).permute(0, 2, 1)
            sampled = sampled * valid
            tokens = torch.cat(
                [
                    tokens[:, :kps_start],
                    tokens[:, kps_start : kps_end] + feat_linear(sampled),
                    tokens[:, kps_end:],
                ],
                dim=1,
            )
            return tokens, augment

        def keypoint3d_update(tokens, augment, pose_output):
            kps3d = pose_output["pred_keypoints_3d"]
            pelvis = (kps3d[:, [9]] + kps3d[:, [10]]) / 2
            kps3d = kps3d - pelvis
            augment = torch.cat(
                [
                    augment[:, :kps3d_start],
                    kps3d_posemb(kps3d),
                    augment[:, kps3d_end:],
                ],
                dim=1,
            )
            return tokens, augment

        image_seq = image_embeddings.flatten(2).permute(0, 2, 1)
        augment_seq = image_augment.flatten(2).permute(0, 2, 1)

        for layer_idx, layer in enumerate(decoder.layers):
            token_embeddings, image_seq = layer(
                token_embeddings, image_seq, token_augment, augment_seq, None
            )
            if layer_idx < self.num_layers - 1:
                pose_output = pose_output_of(decoder.norm_final(token_embeddings))
                token_embeddings, token_augment = keypoint_update(
                    token_embeddings, token_augment, pose_output, layer_idx
                )
                token_embeddings, token_augment = keypoint3d_update(
                    token_embeddings, token_augment, pose_output
                )

        final = decoder.norm_final(token_embeddings)
        pose_output = pose_output_of(final)
        token = final[:, 0]

        if self.hand:
            return token
        # Hand boxes come from the dedicated hand-detection tokens.
        hand_det_tokens = final[:, hand_det_start : hand_det_start + NUM_HAND_TOKENS]
        hand_box = m.bbox_embed(hand_det_tokens).sigmoid()
        return token, hand_box

    def forward(
        self,
        image,
        ray_cond,
        affine_trans,
        bbox_center,
        bbox_scale,
        ori_img_size,
        cam_int,
        keypoints,
    ):
        m = self.model

        x = (image - m.image_mean) / m.image_std
        embeddings = m.backbone(x)
        if isinstance(embeddings, tuple):
            embeddings = embeddings[-1]
        embeddings = embeddings.type(x.dtype)

        augment = m.prompt_encoder.get_dense_pe(embeddings.shape[-2:])
        if self.hand:
            augment = m.hand_pe_layer(embeddings.shape[-2:]).unsqueeze(0)

        condition = self._condition(bbox_center, bbox_scale, ori_img_size, cam_int)
        return self._decode(
            embeddings,
            augment,
            condition,
            keypoints,
            affine_trans,
            bbox_center,
            bbox_scale,
            ori_img_size,
            cam_int,
            ray_cond,
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=r"E:\RGBMocap\export\sam-3d-body")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--mhr", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--which", choices=["body", "hand", "both"], default="both")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--dynamo", action="store_true")
    ap.add_argument("--skip-ort", action="store_true")
    args = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, args.repo)
    torch.set_grad_enabled(False)

    # Import the package without running its `__init__`, which pulls in the
    # estimator and with it the whole training stack (webdataset, hydra,
    # detectron2, ...). The export only needs the model definition.
    import types

    if "sam_3d_body" not in sys.modules:
        pkg = types.ModuleType("sam_3d_body")
        pkg.__path__ = [os.path.join(args.repo, "sam_3d_body")]
        sys.modules["sam_3d_body"] = pkg

    from sam_3d_body.models.meta_arch import SAM3DBody
    from sam_3d_body.utils.config import get_config
    from sam_3d_body.utils.checkpoint import load_state_dict

    patch_mhr_utils_for_export()
    patch_camera_encoder_for_export()
    patch_roma_for_export()
    register_onnx_symbolics()

    print("[1/4] loading config + model ...", flush=True)
    cfg = get_config(args.config)
    cfg.defrost()
    cfg.MODEL.MHR_HEAD.MHR_MODEL_PATH = args.mhr
    cfg.freeze()

    # The MHR TorchScript submodule picks its device from `torch.cuda.is_available()`
    # at construction time; hide CUDA so the whole export stays on the CPU.
    real_is_available = torch.cuda.is_available
    torch.cuda.is_available = lambda: False
    try:
        model = SAM3DBody(cfg)
        state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        state = state.get("state_dict", state)
        load_state_dict(model, state, strict=False)
    finally:
        torch.cuda.is_available = real_is_available
    model.eval()
    # The released config runs the backbone in bfloat16. ONNX Runtime has only
    # partial bf16 coverage, and the fp32 path is numerically tighter, so the
    # exported graphs are float32 throughout.
    model = model.float()
    model.backbone_dtype = torch.float32
    print(f"      {sum(p.numel() for p in model.parameters()) / 1e9:.3f} B parameters")
    del state

    torch.manual_seed(0)
    image = torch.rand(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    ray_cond = (torch.rand(1, 2, IMAGE_SIZE, IMAGE_SIZE) - 0.5) * 0.1
    scale = torch.tensor([[200.0, 300.0]])
    affine_trans = torch.tensor(
        [[[512.0 / 200.0, 0.0, -256.0 * 512.0 / 200.0],
          [0.0, 512.0 / 300.0, -256.0 * 512.0 / 300.0],
          [0.0, 0.0, 1.0]]]
    )
    bbox_center = torch.tensor([[640.0, 400.0]])
    bbox_scale = scale * 1.25
    ori_img_size = torch.tensor([[1280.0, 720.0]])
    focal = 1100.0
    cam_int = torch.tensor(
        [[[focal, 0.0, 640.0], [0.0, focal, 360.0], [0.0, 0.0, 1.0]]]
    )
    keypoints = torch.zeros(1, 1, 3)
    keypoints[:, :, -1] = -2

    inputs = (image, ray_cond, affine_trans, bbox_center, bbox_scale, ori_img_size, cam_int, keypoints)
    names = [
        "image",
        "ray_cond",
        "affine_trans",
        "bbox_center",
        "bbox_scale",
        "ori_img_size",
        "cam_int",
        "keypoints",
    ]
    dynamic_axes = {name: {0: "batch"} for name in names}

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)

    jobs = []
    if args.which in ("body", "both"):
        jobs.append(("body", args.out))
    if args.which in ("hand", "both"):
        stem, ext = os.path.splitext(args.out)
        jobs.append(("hand", f"{stem}_hand{ext}"))

    for kind, path in jobs:
        hand = kind == "hand"
        print(f"[2/4] reference forward ({kind}) ...", flush=True)
        wrapper = SAM3DBodyExport(model, hand=hand).eval()
        t0 = time.time()
        ref = wrapper(*inputs)
        refs = ref if isinstance(ref, tuple) else (ref,)
        print(
            f"      done in {time.time()-t0:.1f}s "
            + ", ".join(str(tuple(r.shape)) for r in refs)
        )

        out_names = ["hand_box", "body_token"] if not hand else ["hand_token"]
        if not hand:
            out_names = ["body_token", "hand_box"]

        print(f"[3/4] exporting -> {path}", flush=True)
        t0 = time.time()
        # `torch.onnx.export` re-traces internally in a way that loses the
        # scripted MHR submodule, so trace explicitly first and hand the
        # resulting ScriptModule to the exporter. Freezing then inlines the
        # scripted submodule and restores the rank information the ONNX shape
        # inference needs for `aten::dim` / `aten::split`.
        traced = torch.jit.trace(wrapper, inputs, check_trace=False)
        traced = torch.jit.freeze(traced)
        torch.onnx.export(
            traced,
            inputs,
            path,
            input_names=names,
            output_names=out_names,
            dynamic_axes=dynamic_axes,
            opset_version=args.opset,
            do_constant_folding=True,
            external_data=True,
            dynamo=False,
        )
        print(f"      exported in {time.time()-t0:.1f}s")

        meta = {
            "kind": kind,
            "image_size": IMAGE_SIZE,
            "decoder_dim": DECODER_DIM,
            "num_keypoints": NUM_KEYPOINTS,
            "opset": args.opset,
            "inputs": names,
            "outputs": out_names,
            "image_mean": [0.485, 0.456, 0.406],
            "image_std": [0.229, 0.224, 0.225],
        }
        with open(path + ".meta.json", "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)

        for name, tensor in zip(names, inputs):
            np.save(os.path.join(out_dir, f"sam3d_{kind}_{name}.npy"), tensor.numpy())
        for name, tensor in zip(out_names, refs):
            np.save(os.path.join(out_dir, f"sam3d_{kind}_{name}.npy"), tensor.numpy())

        if not args.skip_ort:
            print("[4/4] verifying with onnxruntime ...", flush=True)
            try:
                import onnxruntime as rt
            except ImportError:
                print("      onnxruntime not installed, skipping")
                continue
            sess = rt.InferenceSession(path, providers=["CPUExecutionProvider"])
            feeds = {name: tensor.numpy() for name, tensor in zip(names, inputs)}
            got = sess.run(out_names, feeds)
            for name, expected, actual in zip(out_names, refs, got):
                d = np.abs(actual - expected.numpy())
                print(
                    f"      {name}: max abs diff={d.max():.3e} mean={d.mean():.3e}"
                )
                if d.max() > 2e-2:
                    raise SystemExit(f"{name} parity failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
