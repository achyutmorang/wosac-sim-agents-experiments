#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.platform.smart_modern_compat import patch_waymo_target_builder


def _run_pip(args: Iterable[str]) -> None:
    cmd = [sys.executable, "-m", "pip", *list(args)]
    print("[smart-train-setup] $", " ".join(cmd))
    rc = subprocess.run(cmd, check=False)
    if rc.returncode != 0:
        raise RuntimeError(f"pip command failed ({rc.returncode}): {' '.join(cmd)}")


def _can_import(module_name: str) -> tuple[bool, str]:
    try:
        importlib.import_module(module_name)
        return True, ""
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _ensure_import(module_name: str, package_spec: str) -> None:
    ok, err = _can_import(module_name)
    if ok:
        return
    print(f"[smart-train-setup] missing {module_name}: {err}")
    _run_pip(["install", "--upgrade", package_spec])
    ok, err = _can_import(module_name)
    if not ok:
        raise RuntimeError(f"Unable to import {module_name} after install: {err}")


def _ensure_waymo_open_dataset() -> None:
    ok, err = _can_import("waymo_open_dataset")
    if ok:
        return

    py_major, py_minor = sys.version_info[:2]
    print(f"[smart-train-setup] missing waymo_open_dataset: {err}")
    if (py_major, py_minor) >= (3, 12):
        candidates = [
            ["install", "--upgrade", "--no-deps", "waymo-open-dataset-tf-2-12-0==1.6.7"],
            ["install", "--upgrade", "--no-deps", "waymo-open-dataset-tf-2-12-0==1.6.4"],
        ]
    else:
        candidates = [
            ["install", "--upgrade", "waymo-open-dataset-tf-2-12-0==1.6.7"],
            ["install", "--upgrade", "waymo-open-dataset-tf-2-12-0==1.6.4"],
        ]

    errors: list[str] = []
    for args in candidates:
        try:
            _run_pip(args)
            ok, err = _can_import("waymo_open_dataset")
            if ok:
                return
            errors.append(f"{' '.join(args)} -> import failed: {err}")
        except Exception as exc:
            errors.append(f"{' '.join(args)} -> {type(exc).__name__}: {exc}")

    raise RuntimeError(
        "Unable to install/import waymo_open_dataset for SMART training. "
        + " | ".join(errors)
    )


def _resolve_torch_and_cuda_tags() -> tuple[str, str]:
    try:
        import torch
    except Exception as exc:
        raise RuntimeError(
            "SMART smoke training expects PyTorch to be preinstalled in Colab, but torch is not importable: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    match = re.match(r"^(\d+\.\d+\.\d+)", str(torch.__version__))
    if match is not None:
        torch_tag = match.group(1)
    else:
        short = re.match(r"^(\d+)\.(\d+)", str(torch.__version__))
        if short is None:
            raise RuntimeError(f"Unable to parse torch version: {torch.__version__!r}")
        torch_tag = f"{short.group(1)}.{short.group(2)}.0"

    cuda_version = getattr(torch.version, "cuda", None)
    cuda_tag = f"cu{str(cuda_version).replace('.', '')}" if cuda_version else "cpu"
    return torch_tag, cuda_tag


def _ensure_pyg_stack() -> None:
    torch_tag, cuda_tag = _resolve_torch_and_cuda_tags()
    wheel_index = f"https://data.pyg.org/whl/torch-{torch_tag}+{cuda_tag}.html"
    print(f"[smart-train-setup] using PyG wheel index: {wheel_index}")

    required_packages = [
        "torch_scatter",
        "torch_cluster",
    ]
    optional_packages = [
        "pyg_lib",
        "torch_sparse",
    ]

    for package_name in required_packages:
        ok, _ = _can_import(package_name)
        if ok:
            continue
        _run_pip(["install", "--upgrade", package_name, "-f", wheel_index])

    for package_name in optional_packages:
        ok, _ = _can_import(package_name)
        if ok:
            continue
        try:
            _run_pip(["install", "--upgrade", package_name, "-f", wheel_index])
        except Exception as exc:
            print(f"[smart-train-setup] optional package skipped: {package_name} ({type(exc).__name__}: {exc})")

    _ensure_import("torch_geometric", "torch_geometric")

    for module_name in ["torch_scatter", "torch_cluster", "torch_geometric"]:
        ok, err = _can_import(module_name)
        if not ok:
            raise RuntimeError(f"PyG dependency not importable after setup: {module_name} ({err})")


def _probe_smart_training_imports(smart_repo_dir: str) -> None:
    repo_dir = Path(str(smart_repo_dir)).expanduser().resolve()
    if not repo_dir.exists():
        raise FileNotFoundError(f"SMART repo dir does not exist: {repo_dir}")
    sys.path.insert(0, str(repo_dir))
    importlib.invalidate_caches()
    importlib.import_module("train")


def _apply_smart_modern_compat(smart_repo_dir: str) -> None:
    result = patch_waymo_target_builder(smart_repo_dir)
    status = "already-compatible" if result.already_compatible else "patched"
    print(f"[smart-train-setup] target_builder {status}: {result.target_builder_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ensure SMART smoke-training runtime on modern Colab.")
    parser.add_argument("--smart-repo-dir", type=str, default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _ensure_import("yaml", "PyYAML")
    _ensure_import("easydict", "easydict==1.13")
    _ensure_import("pytorch_lightning", "pytorch-lightning>=2.4,<2.6")
    _ensure_waymo_open_dataset()
    _ensure_pyg_stack()
    importlib.import_module("waymo_open_dataset.protos.sim_agents_submission_pb2")
    if str(args.smart_repo_dir).strip():
        _apply_smart_modern_compat(args.smart_repo_dir)
        _probe_smart_training_imports(args.smart_repo_dir)
    print("[smart-train-setup] runtime ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
