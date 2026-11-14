"""Download the FlowHMR model files through the Windows certificate store.

Reuses RGB MoCap's own model registry, so paths, pinned revisions and SHA-256
checks are exactly the ones the application expects. The only change is that
TLS verification goes through ``truststore`` (Windows store), because this
machine routes the CDN hosts through a local proxy whose root CA OpenSSL does
not accept.

    python tools/fetch_models.py --reference E:\\RGBMocap\\toyxyz_RGBMocap --data G:\\RGBMocapData
    python tools/fetch_models.py --reference ... --data ... --list
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Files FlowHMR needs for a reconstruction. SMPL-H is gated and handled apart.
FLOW_KEYS = [
    "flowhmr",          # flowhmr_latest.ckpt
    "flow_config",      # config.yml
    "flow_metadata",    # config.json (fetched automatically by 'flowhmr' too)
    "flow_sam",         # SAM 3D Body model.ckpt
    "flow_sam_config",  # model_config.yaml
    "flow_mhr",         # mhr_model.pt
    "flow_vggt",        # vggt_omega_1b_512.pt (camera estimation)
]
TRACKING_KEYS = ["yolo", "person_reid"]
GATED_KEYS = ["smplh"]


def human(size: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.0f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--list", action="store_true", help="only report status")
    parser.add_argument("--with-tracking", action="store_true", default=True,
                        help="also fetch the shared person/pose detectors")
    args = parser.parse_args()

    reference = args.reference.resolve()
    data = args.data.resolve()
    data.mkdir(parents=True, exist_ok=True)

    try:
        import truststore

        truststore.inject_into_ssl()
    except Exception as error:  # pragma: no cover
        print(f"warning: truststore unavailable ({error})", file=sys.stderr)

    import os

    os.environ["RGB_MOCAP_DATA"] = str(data)
    sys.path.insert(0, str(reference))

    from mocap.models import BY_KEY, download, ready

    keys = list(FLOW_KEYS)
    if args.with_tracking:
        keys += TRACKING_KEYS

    print(f"{'key':<18}{'status':<12}{'size':>12}  file")
    print("-" * 78)
    total_missing = 0
    for key in keys:
        model = BY_KEY[key]
        state = "ready" if ready(model) else ("missing" if not model.path.is_file() else "incomplete")
        size = model.path.stat().st_size if model.path.is_file() else 0
        if state != "ready":
            total_missing += model.minimum_bytes
        print(f"{key:<18}{state:<12}{human(size):>12}  {model.path.relative_to(data)}")

    for key in GATED_KEYS:
        model = BY_KEY[key]
        state = "ready" if ready(model) else "GATED"
        print(f"{key:<18}{state:<12}{'-':>12}  {model.path.relative_to(data)}")

    if args.list:
        return 0
    if total_missing == 0:
        print("\nAll public FlowHMR files are present.")
        return 0

    print(f"\nDownloading about {human(total_missing)} ...")
    started = time.time()
    failures = []
    for key in keys:
        model = BY_KEY[key]
        if ready(model):
            continue
        print(f"\n[{key}] {model.name} ({human(model.minimum_bytes)})", flush=True)
        last = {"at": 0.0}

        def progress(kind, **info):
            now = time.time()
            if now - last["at"] < 3:
                return
            last["at"] = now
            current, total = info.get("current", 0), info.get("total", 0)
            if total:
                print(f"   {human(current)} / {human(total)} "
                      f"({100 * current / total:.0f}%)", flush=True)

        try:
            download(key, progress)
        except Exception as error:
            failures.append((key, str(error)[:200]))
            print(f"   FAILED: {error}", flush=True)
            continue
        if ready(model):
            print(f"   ok -> {model.path}", flush=True)
        else:
            failures.append((key, "verification failed after download"))
            print("   FAILED verification", flush=True)

    elapsed = time.time() - started
    print(f"\nfinished in {elapsed / 60:.1f} min")
    if failures:
        print("problems:")
        for key, reason in failures:
            print(f"  {key}: {reason}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
