from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _human_bytes(n: int) -> str:
    value = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{n} B"


def dir_stats(path: Path) -> Dict[str, Any]:
    total_bytes = 0
    total_files = 0
    for item in path.rglob("*"):
        if item.is_file():
            total_files += 1
            total_bytes += item.stat().st_size
    return {
        "exists": path.exists(),
        "path": str(path),
        "files": total_files,
        "bytes": total_bytes,
        "human_size": _human_bytes(total_bytes),
    }


def _copy_tree_fallback(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.rglob("*"):
        rel = item.relative_to(src)
        target = dst / rel
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, target)


def sync_split(src: Path, dst: Path) -> str:
    if not src.exists() or not src.is_dir():
        raise FileNotFoundError(f"Missing processed split directory: {src}")
    try:
        if src.resolve() == dst.resolve():
            return "already_durable"
    except Exception:
        pass
    dst.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("rsync"):
        commands = [
            ["rsync", "-a", "--info=progress2", str(src) + "/", str(dst) + "/"],
            ["rsync", "-a", "--progress", str(src) + "/", str(dst) + "/"],
        ]
        for cmd in commands:
            try:
                subprocess.run(cmd, check=True)
                return "rsync"
            except subprocess.CalledProcessError:
                continue
    _copy_tree_fallback(src, dst)
    return "python_copy"


def build_manifest(*, split: str, src: Path, dst: Path, mode: str) -> Dict[str, Any]:
    return {
        "created_utc": _utc_now_iso(),
        "split": split,
        "copy_mode": mode,
        "source": dir_stats(src),
        "destination": dir_stats(dst),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Persist a processed SMART split to durable storage.")
    parser.add_argument("--src-root", default="/content/SMART/data/waymo_processed")
    parser.add_argument("--dst-root", default="/content/drive/MyDrive/wosac_experiments/datasets/waymo_processed")
    parser.add_argument("--split", choices=("training", "validation", "testing"), required=True)
    parser.add_argument("--manifest-json", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    src = Path(args.src_root).expanduser() / args.split
    dst = Path(args.dst_root).expanduser() / args.split

    print(f"source: {src}")
    print(f"destination: {dst}")
    print("source_stats_before:", json.dumps(dir_stats(src), indent=2, sort_keys=True))

    mode = sync_split(src, dst)
    manifest = build_manifest(split=args.split, src=src, dst=dst, mode=mode)

    print("manifest:", json.dumps(manifest, indent=2, sort_keys=True))

    if str(args.manifest_json).strip():
        manifest_path = Path(args.manifest_json).expanduser()
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"manifest_path: {manifest_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
