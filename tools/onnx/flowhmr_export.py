"""Export the FlowHMR motion MMDiT to ONNX.

The network is one flow-matching step: given the current latent, the per-frame
video context and the timestep, it predicts x1. The 20-step Euler loop lives
outside the graph, so the exported model is called once per step.

Inputs
    x          [1, T, 491]        normalised motion latent
    cam_R      [1, T, 9]          per-frame relative camera rotation, row major
    feature    [1, T, 3072]       SAM-3D-Body body+hand tokens
    vtxt       [1, 1, 768]        text conditioning (always zeros for v2m)
    timestep   [1]                flow time in [0, 1]
    ctxt_valid [1, T]             bool, True where a video frame is real
    attn_mask  [1, 1, S, S]       additive mask over motion+context tokens

Output
    pred       [1, T, 491]        predicted x1

The narrowband mask is built by the caller so the graph stays independent of the
window sizes and of the sequence length.
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


# --------------------------------------------------------------------------
# Deterministic attention: identical to the reference "torch" mode, but written
# with plain matmul/softmax so ONNX export is guaranteed.
# --------------------------------------------------------------------------
def manual_attention(
    q,
    k,
    v,
    mode="torch",
    drop_rate=0.0,
    attn_mask=None,
    causal=False,
    cu_seqlens_q=None,
    cu_seqlens_kv=None,
    max_seqlen_q=None,
    max_seqlen_kv=None,
    batch_size=1,
    training=True,
    return_attn=False,
    gate=None,
):
    # q/k/v arrive as [B, L, H, D]
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    scale = 1.0 / (q.shape[-1] ** 0.5)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            scores = scores.masked_fill(~attn_mask, float("-inf"))
        else:
            scores = scores + attn_mask
    weights = torch.softmax(scores, dim=-1)
    out = torch.matmul(weights, v)
    if gate is not None:
        out = out * torch.sigmoid(gate.transpose(1, 2))
    out = out.transpose(1, 2)
    b, s, h, d = out.shape
    return out.reshape(b, s, h * d)


# --------------------------------------------------------------------------
def build_narrowband_mask(
    motion_len: int,
    ctxt_len: int,
    mm_window: int,
    mc_window: int,
    cm_window: int,
    cc_window: int,
    device=None,
    dtype=torch.float32,
) -> torch.Tensor:
    """Additive mask matching `mask_mode == "narrowband_v2m"`."""
    neg = float("-inf")

    def band(query_len, key_len, window, offset_q=0, offset_k=0, enabled=True):
        q = torch.arange(query_len, device=device)[:, None] + offset_q
        k = torch.arange(key_len, device=device)[None, :] + offset_k
        if not enabled:
            return torch.full((query_len, key_len), neg, device=device, dtype=dtype)
        inside = (q - k).abs() <= window
        return torch.where(
            inside,
            torch.zeros((), device=device, dtype=dtype),
            torch.full((), neg, device=device, dtype=dtype),
        )

    mm = band(motion_len, motion_len, mm_window)
    mc = band(motion_len, ctxt_len, mc_window)
    # T→M: cm_window < 0 means disabled.
    if cm_window < 0:
        cm = torch.full((ctxt_len, motion_len), neg, device=device, dtype=dtype)
    else:
        cm = band(ctxt_len, motion_len, cm_window)
    cc = band(ctxt_len, ctxt_len, cc_window)
    return torch.cat([torch.cat([mm, mc], 1), torch.cat([cm, cc], 1)], 0)


def build_refiner_mask(ctxt_valid: torch.Tensor) -> torch.Tensor:
    """Additive self-attention mask for the text refiner.

    Mirrors `IndividualTokenRefiner.forward`: a query/key pair is visible only
    when both tokens are valid; rows with nothing visible fall back to the
    diagonal so the softmax never sees an all-`-inf` row.
    """
    valid = ctxt_valid.bool()
    length = valid.shape[1]
    visible = valid[:, None, :] & valid[:, :, None]  # [B, L, L]
    idx = torch.arange(length, device=valid.device)
    diagonal = (idx[:, None] == idx[None, :]).unsqueeze(0).expand_as(visible)
    visible = torch.where(visible.any(dim=-1, keepdim=True), visible, diagonal)
    return torch.where(
        visible,
        torch.zeros((), dtype=torch.float32),
        torch.full((), float("-inf"), dtype=torch.float32),
    ).to(torch.float32)


class FlowHMRExport(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def _refine(self, ctxt_feat, timestep, ctxt_valid, refiner_mask):
        """Text refiner with host-supplied masks.

        `IndividualTokenRefiner` builds its diagonal fallback with `torch.eye`,
        which exports as an `EyeLike` node that ONNX Runtime cannot assign to an
        execution provider for a dynamic shape. The whole mask is therefore
        computed by the caller and passed in, which keeps the graph portable.
        """
        refiner = self.model.text_refiner
        t = refiner.timestep_encoder(timestep)
        mask_float = ctxt_valid.float().unsqueeze(-1)
        pooled = (ctxt_feat * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp_min(1e-5)
        c = t + refiner.context_encoder(pooled).unsqueeze(1)

        x = refiner.input_embedder(ctxt_feat)
        for block in refiner.individual_token_refiner.blocks:
            x = block(x, c, refiner_mask)
        return x

    def forward(self, x, cam_R, feature, vtxt, timestep, ctxt_valid, attn_mask, refiner_mask):
        m = self.model

        motion_feat = m.input_encoder(x)
        timestep_feat = m.timestep_encoder(timestep)
        vtxt_feat = m.vtxt_encoder(vtxt)
        adapter = timestep_feat + vtxt_feat

        ctxt_feat = m.ctxt_encoder["camera_R"](cam_R) + m.ctxt_encoder["feature"](feature)
        ctxt_feat = self._refine(ctxt_feat, timestep, ctxt_valid, refiner_mask)

        for block in m.double_blocks:
            motion_feat, ctxt_feat = block(
                motion_feat=motion_feat,
                text_feat=ctxt_feat,
                adapter=adapter,
                attn_mask=attn_mask,
            )

        split_len = motion_feat.shape[1]
        x_all = torch.cat((motion_feat, ctxt_feat), 1)
        for block in m.single_blocks:
            x_all = block(x=x_all, split_len=split_len, adapter=adapter, attn_mask=attn_mask)

        x_out = x_all[:, :split_len, ...]
        return m.final_layer(x_out, adapter)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=r"E:\RGBMocap\probe\src")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=64, help="trace-time motion length")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--skip-parity", action="store_true")
    ap.add_argument("--skip-ort", action="store_true")
    args = ap.parse_args()

    sys.path.insert(0, args.repo)
    os.chdir(args.repo)
    torch.set_grad_enabled(False)

    import yaml

    import flowhmr.network.hymotion_mmdit_for_v2m as mmdit_mod
    import flowhmr.network.token_refiner as refiner_mod
    from flowhmr.network.hymotion_mmdit_for_v2m import HunyuanMotionMMDiT

    # Deterministic attention: same maths as the reference "torch" mode.
    mmdit_mod.attention = manual_attention
    refiner_mod.attention = manual_attention

    with open(args.config, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    net_args = dict(cfg["network_module_args"])
    pipe_args = cfg["train_pipeline_args"]

    print("[1/5] building MMDiT ...", flush=True)
    model = HunyuanMotionMMDiT(**net_args).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"      {n_params / 1e9:.3f} B parameters")

    print("[2/5] loading checkpoint ...", flush=True)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    prefix = "motion_transformer."
    sub = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    if not sub:
        sub = dict(state)
    missing, unexpected = model.load_state_dict(sub, strict=False)
    print(f"      {len(sub)} tensors, missing={len(missing)}, unexpected={len(unexpected)}")
    if missing:
        raise SystemExit(f"missing weights: {missing[:8]}")
    if unexpected:
        print("      unexpected sample:", unexpected[:5])
    del ckpt, state, sub

    wrapper = FlowHMRExport(model).eval()

    motion_len = args.frames
    ctxt_len = motion_len
    # NOTE: only `narrowband_length` is scaled to frames by the reference
    # constructor; the v2m/tm/tt lengths are used as-is.
    mm_window = int(round(net_args.get("narrowband_length", 2.0) * 30.0))
    mc_raw = net_args.get("narrowband_v2m_length", -1.0)
    mc_window = int(round(mc_raw)) if mc_raw and mc_raw > 0 else max(motion_len, ctxt_len)
    tm_raw = net_args.get("narrowband_tm_length", 0.0)
    cm_window = -1 if tm_raw == 0 else int(round(tm_raw))
    tt_raw = net_args.get("narrowband_tt_length")
    if tt_raw is None:
        tt_raw = mc_raw
    cc_window = int(round(tt_raw)) if tt_raw and tt_raw > 0 else ctxt_len
    print(
        f"      windows: mm={mm_window} mc={mc_window} cm={cm_window} cc={cc_window}",
        flush=True,
    )

    attn_mask = build_narrowband_mask(
        motion_len, ctxt_len, mm_window, mc_window, cm_window, cc_window
    ).view(1, 1, motion_len + ctxt_len, motion_len + ctxt_len)

    x = torch.randn(1, motion_len, net_args["input_dim"])
    cam_R = torch.randn(1, motion_len, 9) * 0.1
    # Camera rotations are orthonormal-ish in practice; keep the trace realistic.
    eye = torch.eye(3).reshape(1, 1, 9).repeat(1, motion_len, 1)
    cam_R = eye + cam_R * 0.05
    feature = torch.randn(1, motion_len, net_args["ctxt_input_dim"]["feature"])
    vtxt = torch.zeros(1, 1, net_args["vtxt_input_dim"])
    timestep = torch.tensor([0.5])
    ctxt_valid = torch.ones(1, ctxt_len, dtype=torch.bool)
    refiner_mask = build_refiner_mask(ctxt_valid).unsqueeze(1)

    print(f"[3/5] reference forward (T={motion_len}) ...", flush=True)
    t0 = time.time()
    ref = wrapper(x, cam_R, feature, vtxt, timestep, ctxt_valid, attn_mask, refiner_mask)
    print(f"      done in {time.time()-t0:.1f}s shape={tuple(ref.shape)}")

    # Also verify the wrapper against the untouched model forward.
    if not args.skip_parity:
        print("[3b] parity against model.forward ...", flush=True)
        ctxt = {"camera_R": cam_R, "feature": feature}
        up = model(
            x=x,
            ctxt_input=ctxt,
            vtxt_input=vtxt,
            timesteps=timestep.expand(1),
            x_mask_temporal=torch.ones(1, motion_len, dtype=torch.bool),
            ctxt_mask_temporal=torch.ones(1, ctxt_len, dtype=torch.bool),
        )
        diff = (up - ref).abs().max().item()
        print(f"      max abs diff={diff:.3e}")
        if diff > 1e-3:
            raise SystemExit("wrapper does not match model.forward")

    dynamic_axes = {
        "x": {1: "motion_len"},
        "cam_R": {1: "motion_len"},
        "feature": {1: "motion_len"},
        "ctxt_valid": {1: "ctxt_len"},
        "attn_mask": {2: "seq", 3: "seq"},
        "refiner_mask": {2: "ctxt_len", 3: "ctxt_len"},
        "pred": {1: "motion_len"},
    }
    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    print(f"[4/5] exporting -> {args.out}", flush=True)
    t0 = time.time()
    torch.onnx.export(
        wrapper,
        (x, cam_R, feature, vtxt, timestep, ctxt_valid, attn_mask, refiner_mask),
        args.out,
        input_names=[
            "x",
            "cam_R",
            "feature",
            "vtxt",
            "timestep",
            "ctxt_valid",
            "attn_mask",
            "refiner_mask",
        ],
        output_names=["pred"],
        dynamic_axes=dynamic_axes,
        opset_version=args.opset,
        do_constant_folding=True,
        external_data=True,
        dynamo=False,
    )
    print(f"      exported in {time.time()-t0:.1f}s")

    meta = {
        "input_dim": net_args["input_dim"],
        "feat_dim": net_args["feat_dim"],
        "num_layers": net_args["num_layers"],
        "num_heads": net_args["num_heads"],
        "ctxt_feature_dim": net_args["ctxt_input_dim"]["feature"],
        "ctxt_camera_dim": net_args["ctxt_input_dim"]["camera_R"],
        "vtxt_input_dim": net_args["vtxt_input_dim"],
        "narrowband": {
            "mm": mm_window,
            "mc": mc_window,
            "cm": cm_window,
            "cc": cc_window,
        },
        "pred_type": pipe_args.get("pred_type"),
        "validation_steps": pipe_args["infer_noise_scheduler_cfg"]["validation_steps"],
        "solver": pipe_args.get("noise_scheduler_cfg", {}).get("method"),
        "fps": pipe_args.get("fps"),
        "mean_std": cfg.get("mean_std_dir"),
        "motion_fps": cfg.get("motion_fps"),
        "max_frame": cfg.get("max_frame"),
        "trace_frames": motion_len,
        "opset": args.opset,
        "inputs": [
            "x[1,T,491]",
            "cam_R[1,T,9]",
            "feature[1,T,3072]",
            "vtxt[1,1,768]",
            "timestep[1]",
            "ctxt_valid[1,T]",
            "attn_mask[1,1,2T,2T]",
            "refiner_mask[1,1,T,T]",
        ],
        "outputs": ["pred[1,T,491]"],
    }
    with open(args.out + ".meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    np.save(os.path.join(out_dir, "fh_x.npy"), x.numpy())
    np.save(os.path.join(out_dir, "fh_cam.npy"), cam_R.numpy())
    np.save(os.path.join(out_dir, "fh_feature.npy"), feature.numpy())
    np.save(os.path.join(out_dir, "fh_pred.npy"), ref.numpy())

    if not args.skip_ort:
        print("[5/5] verifying with onnxruntime ...", flush=True)
        try:
            import onnxruntime as rt
        except ImportError:
            print("      onnxruntime not installed, skipping")
            return 0
        providers = ["CPUExecutionProvider"]
        sess = rt.InferenceSession(args.out, providers=providers)
        feeds = {
            "x": x.numpy(),
            "cam_R": cam_R.numpy(),
            "feature": feature.numpy(),
            "vtxt": vtxt.numpy(),
            "timestep": timestep.numpy(),
            "ctxt_valid": ctxt_valid.numpy(),
            "attn_mask": attn_mask.numpy(),
            "refiner_mask": refiner_mask.numpy(),
        }
        got = sess.run(["pred"], feeds)[0]
        d = np.abs(got - ref.numpy())
        print(f"      max abs diff={d.max():.3e} mean={d.mean():.3e}")
        np.save(os.path.join(out_dir, "fh_pred_onnx.npy"), got)

        # Dynamic motion length: run with a shorter sequence.
        short = max(8, motion_len // 2)
        mask2 = build_narrowband_mask(
            short, short, mm_window, mc_window, cm_window, cc_window
        ).view(1, 1, 2 * short, 2 * short)
        valid2 = torch.ones(1, short, dtype=torch.bool)
        rmask2 = build_refiner_mask(valid2).unsqueeze(1)
        feeds2 = dict(feeds)
        feeds2["x"] = x[:, :short].numpy()
        feeds2["cam_R"] = cam_R[:, :short].numpy()
        feeds2["feature"] = feature[:, :short].numpy()
        feeds2["ctxt_valid"] = valid2.numpy()
        feeds2["attn_mask"] = mask2.numpy()
        feeds2["refiner_mask"] = rmask2.numpy()
        got2 = sess.run(["pred"], feeds2)[0]
        ref2 = wrapper(
            x[:, :short],
            cam_R[:, :short],
            feature[:, :short],
            vtxt,
            timestep,
            valid2,
            mask2,
            rmask2,
        ).numpy()
        print(f"      dynamic T={short}: max abs diff={np.abs(got2-ref2).max():.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
