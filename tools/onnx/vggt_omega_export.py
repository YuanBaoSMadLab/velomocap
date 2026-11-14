"""Export VGGT-Omega (1B) camera path to ONNX.

The exported graph takes pre-normalised RGB frames plus the RoPE sin/cos table
(which depends only on the patch grid size) and returns the 9-D pose encoding.

    images    [1, N, 3, H, W]  float32, already (x - mean) / std?  NO -> raw 0..1
    rope_sin  [HW, 64]         float32  (HW = H/16 * W/16)
    rope_cos  [HW, 64]         float32
    -> pose_enc [1, N, 9]

RoPE for the prefix (cls / storage / camera / register) tokens is all-zero sin,
all-one cos, i.e. identity, so the wrapper concatenates that internally and the
attention never needs dynamic prefix arithmetic.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

PATCH_SIZE = 16
PATCH_TOKEN_START = 17  # 1 camera token + 16 register tokens
EMBED_DIM = 1024
NUM_HEADS = 16
HEAD_DIM = EMBED_DIM // NUM_HEADS  # 64


# --------------------------------------------------------------------------
# RoPE helper (host side, reused for reference outputs and by the Rust port)
# --------------------------------------------------------------------------
def rope_table(h_patches: int, w_patches: int) -> tuple[np.ndarray, np.ndarray]:
    """DINOv3 axial RoPE with base=100, normalize_coords="max", fp32."""
    d_head = HEAD_DIM
    n_periods = d_head // 4  # 16
    periods = 100.0 ** (2.0 * np.arange(n_periods, dtype=np.float64) / (d_head // 2))
    max_hw = float(max(h_patches, w_patches))
    coords_h = (np.arange(h_patches, dtype=np.float64) + 0.5) / max_hw
    coords_w = (np.arange(w_patches, dtype=np.float64) + 0.5) / max_hw
    gh, gw = np.meshgrid(coords_h, coords_w, indexing="ij")
    coords = np.stack([gh, gw], axis=-1).reshape(-1, 2)  # [HW, 2]
    coords = 2.0 * coords - 1.0
    angles = 2.0 * math.pi * coords[:, :, None] / periods[None, None, :]  # [HW,2,16]
    angles = angles.reshape(angles.shape[0], -1)  # [HW,32]
    angles = np.tile(angles, (1, 2))  # [HW,64]
    return np.sin(angles).astype(np.float32), np.cos(angles).astype(np.float32)


# --------------------------------------------------------------------------
# Patched attention: rope already padded, no prefix slicing
# --------------------------------------------------------------------------
def _apply_rope_full(self, q, k, rope):
    q_dtype = q.dtype
    k_dtype = k.dtype
    sin, cos = rope
    rope_dtype = sin.dtype
    q = q.to(dtype=rope_dtype)
    k = k.to(dtype=rope_dtype)
    q = (q * cos) + (_rotate_half(q) * sin)
    k = (k * cos) + (_rotate_half(k) * sin)
    return q.to(dtype=q_dtype), k.to(dtype=k_dtype)


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


# --------------------------------------------------------------------------
# Export wrapper
# --------------------------------------------------------------------------
class VGGTOmegaExport(nn.Module):
    def __init__(self, model, pad_sin: torch.Tensor, pad_cos: torch.Tensor):
        super().__init__()
        self.model = model
        self.agg = model.aggregator
        self.cam = model.camera_head
        self.patch_token_start = model.aggregator.patch_token_start
        self.depth = model.aggregator.depth
        self.inter_types = list(model.aggregator.inter_frame_attention_types)
        self.register_buffer("_vit_pad_sin", pad_sin["vit"], persistent=False)
        self.register_buffer("_vit_pad_cos", pad_cos["vit"], persistent=False)
        self.register_buffer("_agg_pad_sin", pad_sin["agg"], persistent=False)
        self.register_buffer("_agg_pad_cos", pad_cos["agg"], persistent=False)

    # -- DINOv3 patch-embed trunk (patch tokens only) ----------------------
    def _patch_tokens(self, x, rope_sin, rope_cos):
        vit = self.agg.patch_embed
        x = vit.patch_embed(x)
        b, hp, wp, _ = x.shape
        x = x.flatten(1, 2)
        cls_token = vit.cls_token + 0 * vit.mask_token
        x = torch.cat(
            [
                cls_token.expand(b, -1, -1),
                vit.storage_tokens.expand(b, -1, -1),
                x,
            ],
            dim=1,
        )
        sin = torch.cat([self._vit_pad_sin, rope_sin], dim=0).unsqueeze(0).unsqueeze(0)
        cos = torch.cat([self._vit_pad_cos, rope_cos], dim=0).unsqueeze(0).unsqueeze(0)
        rope = (sin, cos)
        for blk in vit.blocks:
            x = blk(x, rope)
        x = vit.norm(x)
        return x[:, vit.n_storage_tokens + 1 :]

    # -- alternating-attention aggregator ---------------------------------
    def _aggregate(self, patch_tokens, batch_size, num_frames, rope):
        def expand(tok):
            first = tok[:, 0:1].expand(batch_size, 1, *tok.shape[2:])
            rest = tok[:, 1:].expand(batch_size, num_frames - 1, *tok.shape[2:])
            return torch.cat([first, rest], dim=1).reshape(batch_size * num_frames, *tok.shape[2:])

        camera_token = expand(self.agg.camera_token)
        register_token = expand(self.agg.register_token)
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)
        _, num_tokens, embed_dim = tokens.shape
        pts = self.patch_token_start

        for idx in range(self.depth):
            tokens = tokens.reshape(batch_size * num_frames, num_tokens, embed_dim)
            tokens = self.agg.frame_blocks[idx](tokens, rope)
            frame_tokens = tokens.reshape(batch_size, num_frames, num_tokens, embed_dim)
            tokens = frame_tokens

            if self.inter_types[idx] == "global":
                t = tokens.reshape(batch_size, num_frames * num_tokens, embed_dim)
                t = self.agg.inter_frame_blocks[idx](t, None)
                tokens = t.reshape(batch_size, num_frames, num_tokens, embed_dim)
            else:
                cr = tokens[:, :, :pts].reshape(batch_size, num_frames * pts, embed_dim)
                pt = tokens[:, :, pts:].reshape(batch_size, num_frames * (num_tokens - pts), embed_dim)
                cr = self.agg.inter_frame_blocks[idx](cr, None)
                t = torch.cat([cr, pt], dim=1)
                cr = t[:, : num_frames * pts].reshape(batch_size, num_frames, pts, embed_dim)
                pt = t[:, num_frames * pts :].reshape(batch_size, num_frames, num_tokens - pts, embed_dim)
                tokens = torch.cat([cr, pt], dim=2)
        # Cached aggregator layers expose the concatenation of the frame-level and
        # the inter-frame tokens (1024 + 1024), which is what the heads consume.
        return torch.cat([frame_tokens, tokens], dim=-1)

    def _camera_head(self, tokens, batch_size, num_frames):
        cam = self.cam
        cr = tokens[:, :, : self.patch_token_start].float()
        cr = cam.token_norm(cr)
        cr = cr.reshape(batch_size, num_frames * self.patch_token_start, -1)
        for blk in cam.trunk:
            cr = blk(cr, None)
        cr = cr.reshape(batch_size, num_frames, self.patch_token_start, -1)
        ct = cam.trunk_norm(cr[:, :, 0])
        raw = cam.camera_branch(ct)
        translation = raw[..., :3]
        quaternion = raw[..., 3:7]
        fov = torch.relu(raw[..., 7:]) + 0.01
        return torch.cat([translation, quaternion, fov], dim=-1)

    def forward(self, images, rope_sin, rope_cos):
        batch_size, num_frames, channels, height, width = images.shape
        x = (images - self.agg._resnet_mean) / self.agg._resnet_std
        x = x.reshape(batch_size * num_frames, channels, height, width)

        patch_tokens = self._patch_tokens(x, rope_sin, rope_cos)

        sin = torch.cat([self._agg_pad_sin, rope_sin], dim=0).unsqueeze(0).unsqueeze(0)
        cos = torch.cat([self._agg_pad_cos, rope_cos], dim=0).unsqueeze(0).unsqueeze(0)
        tokens = self._aggregate(patch_tokens, batch_size, num_frames, (sin, cos))

        return self._camera_head(tokens, batch_size, num_frames)


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=r"E:\RGBMocap\export\vggt-omega")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=4, help="trace-time frame count")
    ap.add_argument("--height", type=int, default=288)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument(
        "--static-spatial",
        action="store_true",
        help="pin the image size to the trace shape instead of leaving it dynamic",
    )
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--skip-parity", action="store_true")
    ap.add_argument("--skip-ort", action="store_true")
    ap.add_argument("--ort-gpu", action="store_true")
    args = ap.parse_args()

    sys.path.insert(0, args.repo)
    from vggt_omega.models import VGGTOmega
    from vggt_omega.models.aggregator import Aggregator, RopePositionEmbedding
    from vggt_omega.models.layers.attention import SelfAttention

    torch.set_grad_enabled(False)

    print("[1/5] building model ...", flush=True)
    model = VGGTOmega(patch_size=PATCH_SIZE, embed_dim=EMBED_DIM, enable_camera=True,
                      enable_depth=False, autocast=False)
    model.eval()
    print("[2/5] loading checkpoint ...", flush=True)
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"      missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("      missing sample:", missing[:5])
    if unexpected:
        print("      unexpected sample:", unexpected[:5])
    del sd

    # RoPE identity padding for the prefix tokens.
    z = torch.zeros(PATCH_TOKEN_START, HEAD_DIM)
    o = torch.ones(PATCH_TOKEN_START, HEAD_DIM)
    z5, o5 = torch.zeros(5, HEAD_DIM), torch.ones(5, HEAD_DIM)
    wrapper = VGGTOmegaExport(model, {"vit": z5, "agg": z}, {"vit": o5, "agg": o})
    wrapper.eval()

    h, w = args.height, args.width
    hp, wp = h // PATCH_SIZE, w // PATCH_SIZE
    sin_np, cos_np = rope_table(hp, wp)
    sin = torch.from_numpy(sin_np)
    cos = torch.from_numpy(cos_np)
    n = args.frames
    images = torch.rand(1, n, 3, h, w)

    up = None
    if not args.skip_parity:
        # Feed the untouched upstream model through its own forward path before
        # the attention rope helper is patched.
        print(f"[3/5] upstream forward ({n} frames, {h}x{w}) ...", flush=True)
        t0 = time.time()
        up = model(images)["pose_enc"]
        print(f"      done in {time.time()-t0:.1f}s  shape={tuple(up.shape)}")

    # Patch attention rope application: the prefix rows are identity-padded, so
    # no dynamic prefix slicing is needed inside the exported graph.
    SelfAttention.apply_rope = _apply_rope_full

    print("[3b] wrapper forward ...", flush=True)
    t0 = time.time()
    ref = wrapper(images, sin, cos)
    print(f"      done in {time.time()-t0:.1f}s  shape={tuple(ref.shape)}")

    if args.check_only:
        print(ref[0, :2, :].numpy())
        return 0

    if up is not None:
        diff = (up - ref).abs()
        rel = diff.max() / up.abs().max().clamp_min(1e-9)
        print(f"      parity: max abs diff={diff.max().item():.3e}  rel={rel.item():.3e}")
        if diff.max().item() > 1e-3:
            raise SystemExit("upstream parity failed")

    dynamic_axes = {
        "images": {0: "batch", 1: "frames"},
        "rope_sin": {0: "tokens"},
        "rope_cos": {0: "tokens"},
        "pose_enc": {0: "batch", 1: "frames"},
    }
    if not args.static_spatial:
        # The token count is already driven by the rope tables, so the image
        # size can stay symbolic and one graph serves every aspect ratio.
        dynamic_axes["images"][3] = "height"
        dynamic_axes["images"][4] = "width"
    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    print(f"[4/5] exporting -> {args.out}", flush=True)
    t0 = time.time()
    torch.onnx.export(
        wrapper,
        (images, sin, cos),
        args.out,
        input_names=["images", "rope_sin", "rope_cos"],
        output_names=["pose_enc"],
        dynamic_axes=dynamic_axes,
        opset_version=args.opset,
        do_constant_folding=True,
        external_data=True,
        dynamo=False,
    )
    print(f"      exported in {time.time()-t0:.1f}s")

    meta = {
        "patch_size": PATCH_SIZE,
        "patch_token_start": PATCH_TOKEN_START,
        "embed_dim": EMBED_DIM,
        "num_heads": NUM_HEADS,
        "head_dim": HEAD_DIM,
        "trace_frames": n,
        "trace_height": h,
        "trace_width": w,
        "opset": args.opset,
        "inputs": ["images[1,N,3,H,W]", "rope_sin[HW,64]", "rope_cos[HW,64]"],
        "outputs": ["pose_enc[1,N,9]"],
    }
    with open(args.out + ".meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    np.save(os.path.join(out_dir, "ref_images.npy"), images.numpy())
    np.save(os.path.join(out_dir, "ref_rope_sin.npy"), sin_np)
    np.save(os.path.join(out_dir, "ref_rope_cos.npy"), cos_np)
    np.save(os.path.join(out_dir, "ref_pose_enc.npy"), ref.numpy())
    if up is not None:
        np.save(os.path.join(out_dir, "ref_upstream.npy"), up.numpy())

    if not args.skip_ort:
        print("[5/5] verifying with onnxruntime ...", flush=True)
        try:
            import onnxruntime as rt
        except ImportError:
            print("      onnxruntime not installed, skipping")
            return 0
        providers = ["CPUExecutionProvider"]
        if args.ort_gpu and "CUDAExecutionProvider" in rt.get_available_providers():
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        sess = rt.InferenceSession(args.out, providers=providers)
        print(f"      providers: {sess.get_providers()}")
        feeds = {
            "images": images.numpy(),
            "rope_sin": sin_np,
            "rope_cos": cos_np,
        }
        got = sess.run(["pose_enc"], feeds)[0]
        d = np.abs(got - ref.numpy())
        print(f"      max abs diff={d.max():.3e}  mean={d.mean():.3e}")
        np.save(os.path.join(out_dir, "ref_onnx.npy"), got)

        if n > 1 or True:
            # Dynamic frame axis: run with a different frame count.
            m = max(1, n - 1)
            img2 = images[:1, :m].numpy()
            got2 = sess.run(["pose_enc"], {"images": img2, "rope_sin": sin_np, "rope_cos": cos_np})[0]
            ref2 = wrapper(torch.from_numpy(img2), sin, cos).numpy()
            d2 = np.abs(got2 - ref2)
            print(f"      dynamic frames={m}: max abs diff={d2.max():.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
