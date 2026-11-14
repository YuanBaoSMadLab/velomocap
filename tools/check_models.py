"""Verify the model manifest against the files on disk.

Checks size for every entry and SHA-256 where the manifest records one.
This is the ground truth for "are my models installed correctly?", and the
same logic the in-app Model Manager uses.

    python tools/check_models.py
    python tools/check_models.py --manifest models/manifest.json --json out.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def human(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024:
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


def check(manifest_path: Path, verify_hash: bool) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf8"))
    root = manifest_path.parent
    results = {"groups": [], "ok": True}

    for group in manifest["groups"]:
        entries = []
        for spec in group["files"]:
            path = root / spec["path"].replace("/", "\\" if sys.platform == "win32" else "/")
            entry = {
                "key": spec["key"],
                "title": spec["title"],
                "path": str(path),
                "required": group.get("required", False) and not spec.get("optional", False),
                "gated": spec.get("gated", False),
            }
            if not path.is_file():
                entry["state"] = "missing"
                entry["detail"] = "not downloaded"
            else:
                size = path.stat().st_size
                expected = spec.get("bytes")
                entry["size"] = size
                if expected and size != expected:
                    entry["state"] = "size_mismatch"
                    entry["detail"] = f"{human(size)} != {human(expected)}"
                elif verify_hash and spec.get("sha256"):
                    actual = sha256(path)
                    if actual != spec["sha256"]:
                        entry["state"] = "hash_mismatch"
                        entry["detail"] = f"{actual[:16]}... != {spec['sha256'][:16]}..."
                    else:
                        entry["state"] = "ok"
                        entry["sha256"] = actual
                else:
                    entry["state"] = "ok"
            if entry["state"] != "ok" and entry["required"]:
                results["ok"] = False
            entries.append(entry)
        results["groups"].append(
            {
                "id": group["id"],
                "title": group["title"],
                "required": group.get("required", False),
                "files": entries,
            }
        )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "models" / "manifest.json")
    parser.add_argument("--no-hash", action="store_true", help="skip SHA-256 (faster)")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    if not args.manifest.is_file():
        print(f"error: manifest not found: {args.manifest}", file=sys.stderr)
        return 2

    results = check(args.manifest, verify_hash=not args.no_hash)

    symbols = {
        "ok": "OK  ",
        "missing": "MISS",
        "size_mismatch": "SIZE",
        "hash_mismatch": "HASH",
    }
    for group in results["groups"]:
        tag = "required" if group["required"] else "optional"
        print(f"\n{group['title']}  [{tag}]")
        print("-" * 76)
        for entry in group["files"]:
            mark = symbols[entry["state"]]
            size = human(entry["size"]) if "size" in entry else "-"
            gate = " (gated)" if entry["gated"] else ""
            print(f"  {mark:<5}{entry['title']:<38}{size:>10}{gate}")
            if entry["state"] != "ok":
                print(f"        {entry['path']}")
                print(f"        {entry.get('detail', '')}")

    print()
    if results["ok"]:
        print("All required models are present and verified.")
    else:
        print("Some required models are missing or invalid (see above).")
        print("Open the app's Model Manager to see where each file comes from.")

    if args.json:
        args.json.write_text(json.dumps(results, indent=2), encoding="utf8")
        print(f"wrote {args.json}")
    return 0 if results["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
