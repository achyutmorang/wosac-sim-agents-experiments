#!/usr/bin/env python3
from __future__ import annotations

import importlib
import re
import subprocess
import sys
from typing import Iterable


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


def main() -> int:
    _ensure_import("yaml", "PyYAML")
    _ensure_import("easydict", "easydict==1.13")
    _ensure_import("pytorch_lightning", "pytorch-lightning>=2.4,<2.6")
    _ensure_pyg_stack()
    print("[smart-train-setup] runtime ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
