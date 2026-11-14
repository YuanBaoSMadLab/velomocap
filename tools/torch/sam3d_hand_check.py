"""Check `sam3d_hand.HandBranch` against the original `forward_decoder_hand`.

Everything runs in one process on CUDA: the eager `forward_step(batch, "hand")`
with hooks on the mask prompt and the camera encoder, then the module, then a
stage-by-stage diff. Keeping it in-process matters — the exported module has to
reproduce the reference *given the same weights on the same device*, and a
cached reference from another process only adds noise to that comparison.

    python tools/torch/sam3d_hand_check.py --ckpt ... --mhr ... --config ...
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

from sam3d_split import load_model  # noqa: E402
from sam3d_hand import HandBranch  # noqa: E402


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
    value = delta.max().item()
    print(
        f"  {name:24s} max abs {value:.3e}  mean {delta.mean().item():.3e}  "
        f">1e-3 {float((delta > 1e-3).float().mean()) * 100:.2f}%  "
        f"(|ref| max {theirs.abs().max().item():.3f})"
    )
    return value


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

    transform_hand = Compose(
        [
            GetBBoxCenterScale(padding=0.9),
            TopdownAffine(input_size=model.cfg.MODEL.IMAGE_SIZE, use_udp=False),
            VisionTransformWrapper(ToTensor()),
        ]
    )

    rng = np.random.default_rng(0)
    image = (rng.random((480, 640, 3)) * 255).astype(np.uint8)
    box = np.array([[300.0, 180.0, 380.0, 260.0]], dtype=np.float32)
    focal = float(np.hypot(640, 480))
    cam_int = torch.tensor(
        [[[focal, 0.0, 320.0], [0.0, focal, 240.0], [0.0, 0.0, 1.0]]], device=device
    )
    batch = prepare_batch(image, transform_hand, box, cam_int=cam_int)
    batch = recursive_to(batch, "cuda")
    model._initialize_batch(batch)

    captured = {}
    masks = {}

    def hook(module, inputs, output):
        captured.setdefault("tokens", inputs[0].detach())
        captured.setdefault("image", inputs[1].detach())
        captured.setdefault("augment", inputs[2].detach())
        captured.setdefault("image_augment", inputs[3].detach())
        return None

    def mask_hook(module, inputs, output):
        masks.setdefault("input", inputs[0])
        masks.setdefault("output", output.detach())
        return None

    def ray_hook(module, inputs, output):
        masks.setdefault("ray_in", inputs[0].detach())
        masks.setdefault("ray_cond", inputs[1].detach())
        return None

    handle = model.decoder_hand.register_forward_hook(hook)
    mask_handle = model.prompt_encoder.mask_downscaling.register_forward_hook(mask_hook)
    ray_handle = model.ray_cond_emb_hand.register_forward_hook(ray_hook)
    print("[2/4] eager hand pass ...", flush=True)
    with torch.no_grad():
        output = model.forward_step(batch, decoder_type="hand")
    handle.remove()
    mask_handle.remove()
    ray_handle.remove()
    ref = output["mhr_hand"]

    crop = batch["img"][0, 0]
    bbox_center = batch["bbox_center"][0, 0]
    bbox_scale = batch["bbox_scale"][0, 0]
    ori = batch["ori_img_size"][0, 0]
    keypoints = torch.zeros(1, 1, 3, device=device)
    keypoints[:, :, -1] = -2

    print("      bbox_center", bbox_center.tolist(), flush=True)
    print("      bbox_scale ", bbox_scale.tolist(), flush=True)
    print("      affine     ", batch["affine_trans"][0, 0].tolist(), flush=True)
    print("      mask input ", tuple(masks["input"].shape), flush=True)
    print("      ray_in     ", tuple(masks["ray_in"].shape), flush=True)

    # --- the same input the eager pass used, recomputed here ---------------
    x = (crop[None] - model.image_mean) / model.image_std
    with torch.no_grad():
        raw = model.backbone(x.type(model.backbone_dtype))
        if isinstance(raw, tuple):
            raw = raw[-1]
        mask_emb = model.prompt_encoder.mask_downscaling(
            torch.zeros(1, 1, 512, 512, device=device)
        )

    print("[3/4] building the module ...", flush=True)
    branch = HandBranch(model).eval().cuda()
    inputs = (
        crop[None],
        bbox_center[None],
        bbox_scale[None],
        cam_int,
        keypoints,
        ori[None],
    )
    with torch.no_grad():
        mine_ray = branch.rays(branch.affine(inputs[1], inputs[2]), inputs[3])
        embeddings, image_seq, augment_seq = branch.backbone(inputs[0], mine_ray)
        tokens, token_augment = branch.init(*inputs[1:5])
        got = branch(*inputs)

    print("[4/4] comparing ...", flush=True)
    print("      mask downscale", tuple(mask_emb.shape), flush=True)
    diff("mask hook vs fresh", masks["output"], mask_emb)
    diff("ray_in vs raw+mask", masks["ray_in"], raw + mask_emb)
    diff("raw vs backbone stage", raw, branch.backbone.p.backbone(
        (inputs[0] - branch.backbone.p.image_mean) / branch.backbone.p.image_std))
    diff("ray", mine_ray, masks["ray_cond"])
    diff("embeddings", embeddings, captured["image"])
    diff("image_seq", image_seq, captured["image"].flatten(2).permute(0, 2, 1))
    diff(
        "augment_seq",
        augment_seq,
        captured["image_augment"].flatten(2).permute(0, 2, 1),
    )
    diff("init.tokens", tokens, captured["tokens"])
    diff("init.augment", token_augment, captured["augment"])

    normed, pred, pred_cam, verts, j3d, kps2d, depth, model_params, joint_rots = got
    # The app's feature row is token 0 of the decoder output; the hook captures
    # the same block, before `forward_decoder` slices out `[:, 3:5]`.
    diff("out.token0", normed[:, 0], captured["tokens"][:, 0])
    diff("out.pred[:266]", pred[:, :266], ref["pred_pose_raw"])
    diff("out.pred_cam", pred_cam, ref["pred_cam"])
    diff("out.verts", verts, ref["pred_vertices"])
    diff("out.keypoints_3d", j3d, ref["pred_keypoints_3d"])
    diff("out.keypoints_2d", kps2d, ref["pred_keypoints_2d"])
    diff("out.depth", depth, ref["pred_keypoints_2d_depth"])
    diff("out.model_params", model_params, ref["mhr_model_params"])
    diff("out.joint_rots", joint_rots, ref["joint_global_rots"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
