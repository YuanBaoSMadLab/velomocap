"""Print the attribute layout the full (body + hands) wrapper needs.

The body-only export was easy because everything it touched was already known.
Hands add a second decoder with its own token embeddings, its own pose head, its
own keypoint tokens and a fusion stage, so the exact names and dimensions matter.

    python tools/torch/sam3d_hand_probe.py --ckpt ... --mhr ... --config ...
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "onnx"))

from sam3d_split import load_model  # noqa: E402


def shape(value):
    if isinstance(value, torch.Tensor):
        return tuple(value.shape)
    if hasattr(value, "weight"):
        return tuple(value.weight.shape)
    return type(value).__name__


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=r"E:\RGBMocap\export\sam-3d-body")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--mhr", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    model = load_model(args.repo, args.ckpt, args.mhr, args.config)
    cfg = model.cfg

    print("IMAGE_SIZE", cfg.MODEL.IMAGE_SIZE)
    print("DECODER flags", {k: v for k, v in cfg.MODEL.DECODER.items() if isinstance(v, (bool, int, str))})
    print("BACKBONE type", cfg.MODEL.BACKBONE.TYPE)
    print()

    names = [
        "init_pose_hand", "init_camera_hand", "init_to_token_mhr_hand",
        "prev_to_token_mhr_hand", "prompt_to_token", "prompt_encoder",
        "hand_box_embedding", "keypoint_embedding_hand", "keypoint3d_embedding_hand",
        "hand_pe_layer", "ray_cond_emb_hand", "keypoint_posemb_linear_hand",
        "keypoint_feat_linear_hand", "keypoint3d_posemb_linear_hand",
        "keypoint_embedding_idxs_hand", "keypoint3d_embedding_idxs_hand",
        "pelvis_idx", "thresh_wrist_angle",
    ]
    for name in names:
        value = getattr(model, name, "<missing>")
        print(f"{name:32s} {shape(value) if not isinstance(value, str) else value}")

    print()
    print("decoder_hand layers", len(model.decoder_hand.layers))
    print("decoder_hand norm_final", shape(model.decoder_hand.norm_final))
    print("decoder_hand do_interm_preds", model.decoder_hand.do_interm_preds)
    print("decoder_hand keypoint_token_update", model.decoder_hand.keypoint_token_update)

    head = model.head_pose_hand
    for name in (
        "body_cont_dim", "num_shape_comps", "num_scale_comps", "num_hand_comps",
        "num_face_comps", "npose", "hand_pose_comps", "scale_mean", "scale_comps",
        "hand_pose_mean", "hand_joint_idxs_left", "hand_joint_idxs_right",
        "keypoint_mapping", "joint_rotation",
    ):
        value = getattr(head, name, "<missing>")
        print(f"head_pose_hand.{name:26s} {shape(value) if not isinstance(value, str) else value}")
    print("head_pose_hand.proj", shape(head.proj))

    body = model.head_pose
    for name in ("keypoint_mapping", "joint_rotation"):
        print(f"head_pose.{name:31s} {shape(getattr(body, name, '<missing>'))}")
    print("head_camera_hand", shape(getattr(model, "head_camera_hand", "<missing>")))
    print("bbox_embed", shape(model.bbox_embed))
    print("hand_cls_embed", shape(model.hand_cls_embed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
