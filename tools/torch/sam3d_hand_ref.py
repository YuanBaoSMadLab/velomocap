"""Dump the hand-branch indices and one eager reference frame.

STATUS: the indices and head dimensions dump correctly, but the eager forward
still fails inside `get_ray_condition` with a device mismatch (`_initialize_batch`
ties to move the batch to CUDA while the model is on the CPU). Fix that before
trusting any reference tensor from here: rebuild the batch the way
`prepare_batch` does and keep the model on the same device as the tensors.

Hands need `keypoint_embedding_idxs_hand` / `keypoint3d_embedding_idxs_hand`
(which of the 70 keypoints get a token) and the same for the body branch, plus
the head dimensions. It also runs the *original* `forward_decoder_hand` on a
synthetic hand crop and writes the tensors the TorchScript module has to match.

    python tools/torch/sam3d_hand_ref.py --ckpt ... --mhr ... --config ... --out DIR
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "onnx"))

from sam3d_split import load_model  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=r"E:\RGBMocap\export\sam-3d-body")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--mhr", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    model = load_model(args.repo, args.ckpt, args.mhr, args.config).cuda()
    device = torch.device('cuda')
    cfg = model.cfg
    torch.manual_seed(0)

    info = {
        "image_size": list(cfg.MODEL.IMAGE_SIZE),
        "keypoint_embedding_idxs": list(map(int, model.keypoint_embedding_idxs)),
        "keypoint3d_embedding_idxs": list(map(int, model.keypoint3d_embedding_idxs)),
        "keypoint_embedding_idxs_hand": list(map(int, model.keypoint_embedding_idxs_hand)),
        "keypoint3d_embedding_idxs_hand": list(map(int, model.keypoint3d_embedding_idxs_hand)),
        "pelvis_idx": list(map(int, model.pelvis_idx)),
        "hand_body_cont_dim": int(model.head_pose_hand.body_cont_dim),
        "hand_num_shape_comps": int(model.head_pose_hand.num_shape_comps),
        "hand_num_scale_comps": int(model.head_pose_hand.num_scale_comps),
        "hand_num_hand_comps": int(model.head_pose_hand.num_hand_comps),
        "hand_num_face_comps": int(model.head_pose_hand.num_face_comps),
        "hand_npose": int(model.head_pose_hand.npose),
        "body_cont_dim": int(model.head_pose.body_cont_dim),
        "body_num_shape_comps": int(model.head_pose.num_shape_comps),
        "body_num_scale_comps": int(model.head_pose.num_scale_comps),
        "body_num_hand_comps": int(model.head_pose.num_hand_comps),
        "body_num_face_comps": int(model.head_pose.num_face_comps),
    }
    print(json.dumps(info, indent=2))
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "hand_ref.json"), "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=2)

    # ---- one eager hand frame -------------------------------------------
    from sam_3d_body.data.transforms import (
        Compose,
        GetBBoxCenterScale,
        TopdownAffine,
        VisionTransformWrapper,
    )
    from torchvision.transforms import ToTensor

    transform_hand = Compose(
        [
            GetBBoxCenterScale(padding=0.9),
            TopdownAffine(input_size=cfg.MODEL.IMAGE_SIZE, use_udp=False),
            VisionTransformWrapper(ToTensor()),
        ]
    )

    rng = np.random.default_rng(0)
    image = (rng.random((480, 640, 3)) * 255).astype(np.uint8)
    box = np.array([[300.0, 180.0, 380.0, 260.0]], dtype=np.float32)
    focal = float(np.hypot(640, 480))
    cam_int = torch.tensor(
        [[[focal, 0.0, 320.0], [0.0, focal, 240.0], [0.0, 0.0, 1.0]]],
        device=device,
    )

    data = transform_hand(
        {
            "img": image,
            "bbox": box[0],
            "bbox_format": "xyxy",
            "mask": np.zeros((480, 640, 1), np.uint8),
            "mask_score": np.array(0.0, np.float32),
        }
    )
    # Build the batch with the original helper instead of hand-rolling it: the
    # model reads `mask`, `mask_score`, `img_size`, `person_valid` and the
    # collated shapes, and guessing them is what made this harness fail.
    from sam_3d_body.data.utils.prepare_batch import prepare_batch
    from sam_3d_body.utils import recursive_to

    batch = prepare_batch(image, transform_hand, box, cam_int=cam_int)
    batch = recursive_to(batch, "cuda")

    # `forward_pose_branch` builds this dummy prompt itself; it is recorded here
    # so the TorchScript module can be fed exactly the same one.
    keypoints = torch.zeros(1, 1, 3, device=device)
    keypoints[:, :, -1] = -2
    model._initialize_batch(batch)
    captured = {}

    def hook(module, inputs, output):
        # A forward hook that returns anything *replaces* the module output, and
        # `setdefault` returns the stored value — which silently turns the
        # decoder's `(tokens, poses)` tuple into a single tensor.
        captured.setdefault("tokens", output[0].detach())
        return None

    hook = model.decoder_hand.register_forward_hook(hook)
    model.hand_batch_idx = [0]
    model.body_batch_idx = []
    with torch.no_grad():
        output = model.forward_pose_branch(batch)
    hook.remove()

    hand = output["mhr_hand"]
    tokens = captured["tokens"]
    os.makedirs(args.out, exist_ok=True)
    torch.save(
        {
            "crop": batch["img"][0, 0].detach().cpu(),
            "bbox_center": batch["bbox_center"][0, 0].detach().cpu(),
            "bbox_scale": batch["bbox_scale"][0, 0].detach().cpu(),
            "affine": batch["affine_trans"][0, 0].detach().cpu(),
            "cam_int": cam_int[0].detach().cpu(),
            "ori": batch["ori_img_size"][0, 0].detach().cpu(),
            "keypoints": keypoints[0, 0].detach().cpu(),
            "tokens": tokens[0],
            "verts": hand["pred_vertices"][0] if "pred_vertices" in hand else None,
            "pred_cam": hand.get("pred_cam")[0] if "pred_cam" in hand else None,
        },
        os.path.join(args.out, "hand_ref.pt"),
    )
    print("hand token shape", tuple(tokens.shape))
    print("first 4", tokens[0, 0, :4].tolist())
    print("mhr_hand keys", sorted(hand.keys()))
    with open(os.path.join(args.out, "hand_ref.json"), "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
