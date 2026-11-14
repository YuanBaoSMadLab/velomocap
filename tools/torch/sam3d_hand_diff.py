"""Find where the hand module diverges from the original.

Compares, stage by stage, what `forward_decoder_hand` feeds into `decoder_hand`
(backbone embeddings, token layout, prompt, conditioning) against the
TorchScript module's own pieces. The first stage that disagrees is the bug.

    python tools/torch/sam3d_hand_diff.py --ckpt ... --mhr ... --config ... --ref ...
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
from sam3d_hand_jit import HandBackbone, HandInit, HandBranch  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=r"E:\RGBMocap\export\sam-3d-body")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--mhr", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--ref", required=True)
    args = parser.parse_args()

    model = load_model(args.repo, args.ckpt, args.mhr, args.config).cuda()
    reference = torch.load(args.ref, map_location="cpu")
    device = torch.device("cuda")

    crop = reference["crop"][None].to(device)
    bbox_center = reference["bbox_center"][None].to(device)
    bbox_scale = reference["bbox_scale"][None].to(device)
    cam_int = reference["cam_int"][None].to(device)
    keypoints = reference["keypoints"][None, None].to(device)
    ori = reference["ori"][None].to(device)

    # --- eager: what the original feeds into the hand decoder -------------
    captured = {}

    def hook(module, inputs, output):
        captured.setdefault("tokens", inputs[0].detach())
        captured.setdefault("augment", inputs[2].detach())
        captured.setdefault("image", inputs[1].detach())
        captured.setdefault("image_augment", inputs[3].detach())
        return None

    handle = model.decoder_hand.register_forward_hook(hook)

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
    batch = prepare_batch(image, transform_hand, box, cam_int=cam_int.cpu())
    batch = recursive_to(batch, "cuda")
    model._initialize_batch(batch)
    model.hand_batch_idx = [0]
    model.body_batch_idx = []
    with torch.no_grad():
        model.forward_pose_branch(batch)
    handle.remove()

    eager_tokens = captured["tokens"]
    eager_image = captured["image"]
    eager_augment = captured["augment"]
    eager_image_augment = captured["image_augment"]

    # --- mine -------------------------------------------------------------
    branch = HandBranch(model).eval().cuda()
    with torch.no_grad():
        affine = branch.affine(bbox_center, bbox_scale)
        ray = branch.rays(affine, cam_int)
        embeddings, image_seq, augment_seq = branch.backbone(crop, ray)
        tokens, augment = branch.init(bbox_center, bbox_scale, cam_int, keypoints)

    def compare(name, mine, theirs):
        if mine.shape != theirs.shape:
            print(f"{name:16s} shape {tuple(mine.shape)} vs {tuple(theirs.shape)}")
            return
        diff = (mine - theirs).abs().max().item()
        print(f"{name:16s} max abs {diff:.3e}")

    compare("image_seq", image_seq, eager_image)
    compare("image_augment", augment_seq, eager_image_augment)
    compare("tokens", tokens, eager_tokens)
    compare("augment", augment, eager_augment)

    # The reference's affine vs mine.
    ref_affine = batch["affine_trans"][0, 0]
    print("affine diff", (affine[0] - ref_affine).abs().max().item())
    ref_center = batch["bbox_center"][0, 0]
    print("center diff", (bbox_center[0] - ref_center).abs().max().item())
    ref_scale = batch["bbox_scale"][0, 0]
    print("scale  diff", (bbox_scale[0] - ref_scale).abs().max().item())

    print("reference hand token first 4", eager_tokens[0, 0, :4].tolist())
    with torch.no_grad():
        mine_token = branch(*(
            crop, bbox_center, bbox_scale, cam_int, keypoints, ori
        ))[0]
    print("module    hand token first 4", mine_token[0, :4].tolist())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
