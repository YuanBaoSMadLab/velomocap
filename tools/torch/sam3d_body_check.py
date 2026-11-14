"""Check `sam3d_body.BodyPass` against the original `forward_step(..., "body")`.

Everything runs in one process on CUDA. This is the test the first body export
never had: it only compared the traced module against *itself*, so the missing
mask-prompt term (`no_mask_embed`, see `sam3d_body.BodyBackbone`) went
unnoticed.

    python tools/torch/sam3d_body_check.py --ckpt ... --mhr ... --config ...
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "onnx"))
sys.path.insert(0, HERE)

import roma  # noqa: E402

from sam3d_split import (  # noqa: E402
    keypoints_from_mhr,
    load_model,
    project_keypoints,
)
from sam3d_body import BodyPass, params_from_pred_body  # noqa: E402


def diff(name, mine, theirs):
    if mine is None or theirs is None:
        print(f"  {name:24s} missing")
        return None
    mine = mine.detach().float()
    theirs = theirs.detach().float()
    if tuple(mine.shape) != tuple(theirs.shape):
        print(f"  {name:24s} shape {tuple(mine.shape)} vs {tuple(theirs.shape)}")
        return None
    delta = (mine - theirs).abs()
    print(
        f"  {name:24s} max abs {delta.max().item():.3e}  "
        f"mean {delta.mean().item():.3e}  "
        f">1e-3 {float((delta > 1e-3).float().mean()) * 100:.2f}%  "
        f"(|ref| max {theirs.abs().max().item():.3f})"
    )
    return delta.max().item()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=r"E:\RGBMocap\export\sam-3d-body")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--mhr", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    print("[1/4] loading model ...", flush=True)
    model = load_model(args.repo, args.ckpt, args.mhr, args.config).cuda()
    device = torch.device("cuda")

    from sam_3d_body.data.transforms import (
        Compose,
        GetBBoxCenterScale,
        TopdownAffine,
        VisionTransformWrapper,
    )
    from sam_3d_body.data.utils.prepare_batch import prepare_batch
    from sam_3d_body.utils import recursive_to
    from torchvision.transforms import ToTensor

    transform_body = Compose(
        [
            GetBBoxCenterScale(padding=1.25),
            TopdownAffine(input_size=model.cfg.MODEL.IMAGE_SIZE, use_udp=False),
            VisionTransformWrapper(ToTensor()),
        ]
    )

    rng = np.random.default_rng(1)
    image = (rng.random((720, 1280, 3)) * 255).astype(np.uint8)
    box = np.array([[420.0, 90.0, 860.0, 700.0]], dtype=np.float32)
    cam_int = torch.tensor(
        [[[1400.0, 0.0, 640.0], [0.0, 1400.0, 360.0], [0.0, 0.0, 1.0]]]
    ).cuda()
    batch = prepare_batch(image, transform_body, box, cam_int=cam_int)
    batch = recursive_to(batch, "cuda")
    model._initialize_batch(batch)

    captured = {}
    rays = {}

    def hook(module, inputs, output):
        captured.setdefault("tokens", inputs[0].detach())
        captured.setdefault("image", inputs[1].detach())
        captured.setdefault("augment", inputs[2].detach())
        captured.setdefault("image_augment", inputs[3].detach())
        return None

    def ray_hook(module, inputs, output):
        rays.setdefault("ray_in", inputs[0].detach())
        rays.setdefault("ray_cond", inputs[1].detach())
        return None

    handle = model.decoder.register_forward_hook(hook)
    ray_handle = model.ray_cond_emb.register_forward_hook(ray_hook)
    print("[2/4] eager body pass ...", flush=True)
    with torch.no_grad():
        output = model.forward_step(batch, decoder_type="body")
    handle.remove()
    ray_handle.remove()
    ref = output["mhr"]

    crop = batch["img"][0, 0]
    bbox_center = batch["bbox_center"][0, 0]
    bbox_scale = batch["bbox_scale"][0, 0]
    keypoints = torch.zeros(1, 1, 3, device=device)
    keypoints[:, :, -1] = -2

    print("      bbox_center", bbox_center.tolist(), flush=True)
    print("      bbox_scale ", bbox_scale.tolist(), flush=True)
    print("      affine     ", batch["affine_trans"][0, 0].tolist(), flush=True)
    print("      ray_in     ", tuple(rays["ray_in"].shape), flush=True)
    print("      hand_box   ", ref["hand_box"][0].tolist(), flush=True)

    with torch.no_grad():
        raw = model.backbone(
            ((crop[None] - model.image_mean) / model.image_std).type(
                model.backbone_dtype
            )
        )
        if isinstance(raw, tuple):
            raw = raw[-1]

    print("[3/4] building the module ...", flush=True)
    body = BodyPass(model).eval().cuda()
    with torch.no_grad():
        normed, pred, pred_cam, emb, iseq, aseq, tokens, taug = body.run(
            crop[None], bbox_center[None], bbox_scale[None], cam_int, keypoints
        )
        shape, model_params, face = params_from_pred_body(body.head, pred)
        verts_raw, skel_state = body.mhr(shape, model_params, face)
        j3d = keypoints_from_mhr(body.head.keypoint_mapping, verts_raw, skel_state)
        j2d, depth = project_keypoints(
            j3d, pred_cam, bbox_center[None], bbox_scale[None][:, 0], cam_int
        )
        joint_rots = roma.unitquat_to_rotmat(skel_state[:, :, 3:7])
        sign = torch.tensor([1.0, -1.0, -1.0], device=device)
        verts = verts_raw / 100 * sign

    print("[4/4] comparing ...", flush=True)
    mask_bias = model.prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1)
    diff("ray_in vs raw+mask", rays["ray_in"], raw + mask_bias)
    diff("embeddings", emb, captured["image"])
    diff("augment_seq", aseq, captured["image_augment"].flatten(2).permute(0, 2, 1))
    diff("out.pred[:266]", pred[:, :266], ref["pred_pose_raw"])
    diff("out.pred_cam", pred_cam, ref["pred_cam"])
    diff("out.verts", verts, ref["pred_vertices"])
    diff("out.keypoints_3d", j3d, ref["pred_keypoints_3d"])
    diff("out.keypoints_2d", j2d, ref["pred_keypoints_2d"])
    diff("out.depth", depth, ref["pred_keypoints_2d_depth"])
    diff("out.model_params", model_params, ref["mhr_model_params"])
    diff("out.joint_rots", joint_rots, ref["joint_global_rots"])
    diff("out.hand", pred[:, 314:422], ref["hand"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
