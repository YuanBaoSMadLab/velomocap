"""Compare `sam3d_full.Sam3dFull` with the original `run_inference("full")`.

One process, one GPU: the reference four-pass pipeline runs first, then the
module, then every output that `run_inference` publishes is diffed. That is the
only end-to-end test of the hand crops, the fusion criteria, the keypoint prompt
and the final `mhr_forward` together.

    python tools/torch/sam3d_full_check.py --ckpt ... --mhr ... --config ...
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
import time
import traceback

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "onnx"))
sys.path.insert(0, HERE)

from sam3d_split import load_model  # noqa: E402
from sam3d_full import Sam3dFull  # noqa: E402


def diff(name, mine, theirs):
    if mine is None or theirs is None:
        print(f"  {name:20s} missing")
        return None
    mine = mine.detach().float()
    theirs = theirs.detach().float()
    if tuple(mine.shape) != tuple(theirs.shape):
        print(f"  {name:20s} shape {tuple(mine.shape)} vs {tuple(theirs.shape)}")
        return None
    delta = (mine - theirs).abs()
    print(
        f"  {name:20s} max abs {delta.max().item():.3e}  "
        f"mean {delta.mean().item():.3e}  "
        f"(|ref| max {theirs.abs().max().item():.3f})"
    )
    return delta.max().item()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=r"E:\RGBMocap\export\sam-3d-body")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--mhr", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--save", default=None)
    args = parser.parse_args()

    print("[1/4] loading model ...", flush=True)
    model = load_model(args.repo, args.ckpt, args.mhr, args.config).cuda()
    device = torch.device("cuda")
    # The shipped config runs the ViT in bfloat16 (`convert_to_fp16`) and
    # everything else in float32; `load_model` flattens it to float32 for
    # export. Put it back so the reference and the module run the same regime.
    backbone_dtype = torch.bfloat16
    model.backbone_dtype = backbone_dtype
    model.backbone.to(backbone_dtype)

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
    transform_hand = Compose(
        [
            GetBBoxCenterScale(padding=0.9),
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
    model._flow_tokens_only = False

    print("[2/4] reference run_inference(full) ...", flush=True)
    started = time.time()
    # The app's feature rows are token 0 of each decoder call; capture them the
    # same way `Sam3dTokenExtractor` does so the module's copies can be diffed.
    captured = {"body": [], "hand": []}

    def body_hook(module, inputs, output):
        captured["body"].append(output[0].detach())
        return None

    def hand_hook(module, inputs, output):
        captured["hand"].append(output[0].detach())
        return None

    handles = [
        model.decoder.register_forward_hook(body_hook),
        model.decoder_hand.register_forward_hook(hand_hook),
    ]
    try:
        with torch.no_grad():
            out = model.run_inference(
                image, batch, inference_type="full", transform_hand=transform_hand
            )
    except Exception:  # noqa: BLE001
        print("REFERENCE FAILED", flush=True)
        traceback.print_exc()
        return 1
    finally:
        for handle in handles:
            handle.remove()
    print(
        f"      done in {time.time() - started:.1f}s; captured "
        f"{len(captured['body'])} body / {len(captured['hand'])} hand token blocks",
        flush=True,
    )
    ref = out[0]["mhr"]

    frame = torch.from_numpy(image).to(device).permute(2, 0, 1).float()[None] / 255.0
    bbox_center = batch["bbox_center"][0, 0]
    bbox_scale = batch["bbox_scale"][0, 0]
    frame_size = torch.tensor(
        [[float(image.shape[1]), float(image.shape[0])]], device=device
    )

    print("[3/4] building the module ...", flush=True)
    module = Sam3dFull(model, backbone_dtype=backbone_dtype).eval().cuda()
    started = time.time()
    with torch.no_grad():
        got = module(frame, bbox_center[None], bbox_scale[None], cam_int, frame_size)
    print(f"      module forward {time.time() - started:.1f}s", flush=True)
    verts, j3d, j2d, pred_cam, cam_t, focal, model_params, joint_rots, hand_valid, \
        body_token, left_token, right_token, skel_state, pred_final = got
    print("      hand_valid", hand_valid.tolist(), flush=True)

    print("[4/4] comparing ...", flush=True)
    def first4(values):
        return ", ".join(f"{float(v):.6f}" for v in values.flatten()[:4])

    print(f"      module verts      [{first4(verts)}]", flush=True)
    print(f"      module kps3d      [{first4(j3d)}]", flush=True)
    print(f"      module kps2d      [{first4(j2d)}]", flush=True)
    print(f"      module pred_cam   [{first4(pred_cam)}]", flush=True)
    print(f"      module pred_cam_t [{first4(cam_t)}]", flush=True)
    print(f"      module params     [{first4(model_params)}]", flush=True)
    print(f"      module joint_rots [{first4(joint_rots)}]", flush=True)
    print(f"      reference verts   [{first4(ref['pred_vertices'])}]", flush=True)
    print(f"      reference kps2d   [{first4(ref['pred_keypoints_2d'])}]", flush=True)
    diff("pred_vertices", verts, ref["pred_vertices"])
    diff("pred_keypoints_3d", j3d, ref["pred_keypoints_3d"])
    diff("pred_keypoints_2d", j2d, ref["pred_keypoints_2d"])
    diff("pred_cam", pred_cam, ref["pred_cam"])
    diff("pred_cam_t", cam_t, ref["pred_cam_t"])
    diff("focal_length", focal, ref["focal_length"])
    diff("mhr_model_params", model_params, ref["mhr_model_params"])
    diff("joint_global_rots", joint_rots, ref["joint_global_rots"])
    diff("body_token", body_token, captured["body"][0][:, 0])
    diff("left_token", left_token, captured["hand"][0][:, 0])
    diff("right_token", right_token, captured["hand"][1][:, 0])
    # `skel_state` is raw (centimetres, MHR axes) while `pred_joint_coords` was
    # divided by 100 and flipped by `MHRHead.forward`.
    sign3 = torch.tensor([1.0, -1.0, -1.0], device=device)
    diff("skel_state xyz", skel_state[:, :, :3] * 0.01 * sign3,
         ref["pred_joint_coords"])
    print(f"      pred first4       [{', '.join(f'{float(v):.6f}' for v in pred_final.flatten()[:4])}]",
          flush=True)
    diff("pred_keypoints_2d_verts", None, ref.get("pred_keypoints_2d_verts"))

    if args.save:
        os.makedirs(args.save, exist_ok=True)
        # `velomocap-cli sam-full-torch --case` reads this exact layout:
        # i32 width, i32 height, f32 centre[2], scale[2], cam_int[9], frame.
        frame_nchw = frame[0].detach().cpu().numpy().astype("<f4")
        payload = struct.pack("<ii", image.shape[1], image.shape[0])
        payload += np.asarray(bbox_center.cpu().tolist(), "<f4").tobytes()
        payload += np.asarray(bbox_scale.cpu().tolist(), "<f4").tobytes()
        payload += np.asarray(cam_int[0].cpu().tolist(), "<f4").tobytes()
        payload += frame_nchw.tobytes()
        with open(os.path.join(args.save, "case.bin"), "wb") as fh:
            fh.write(payload)
        print(
            f"      wrote case.bin ({len(payload)/1e6:.1f} MB) — run "
            f"`velomocap-cli sam-full-torch --case {os.path.abspath(args.save)}`",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
