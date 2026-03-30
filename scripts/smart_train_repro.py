#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import runpy
import sys
import traceback
from importlib import metadata
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch


def _bool_arg(explicit_true: bool, explicit_false: bool, default: bool) -> bool:
    if explicit_true and explicit_false:
        raise ValueError("Conflicting deterministic flags were provided.")
    if explicit_true:
        return True
    if explicit_false:
        return False
    return bool(default)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deterministic SMART training launcher.")
    parser.add_argument("--smart-repo-dir", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--save-ckpt-path", type=str, default="")
    parser.add_argument("--pretrain-ckpt", type=str, default="")
    parser.add_argument("--ckpt-path", type=str, default="")
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--no-deterministic", action="store_true")
    return parser.parse_args()


def _configure_reproducibility(*, seed: int, deterministic: bool) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("SMART_TRAIN_SEED", str(seed))
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    pl.seed_everything(seed, workers=True)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)
    if deterministic:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass


def _safe_version(package_name: str) -> str:
    try:
        return str(metadata.version(package_name))
    except Exception:
        return "not_installed"


def main() -> int:
    args = parse_args()
    deterministic = _bool_arg(args.deterministic, args.no_deterministic, True)
    seed = int(args.seed)
    _configure_reproducibility(seed=seed, deterministic=deterministic)

    smart_repo_dir = Path(args.smart_repo_dir).expanduser().resolve()
    train_py = smart_repo_dir / "train.py"
    if not train_py.exists():
        raise FileNotFoundError(f"Missing SMART train.py at: {train_py}")
    smart_repo_str = str(smart_repo_dir)
    if smart_repo_str not in sys.path:
        sys.path.insert(0, smart_repo_str)

    argv = [
        "train.py",
        "--config",
        str(args.config),
    ]
    if str(args.save_ckpt_path).strip():
        argv.extend(["--save_ckpt_path", str(args.save_ckpt_path).strip()])
    if str(args.pretrain_ckpt).strip():
        argv.extend(["--pretrain_ckpt", str(args.pretrain_ckpt).strip()])
    if str(args.ckpt_path).strip():
        argv.extend(["--ckpt_path", str(args.ckpt_path).strip()])

    os.chdir(str(smart_repo_dir))
    sys.argv = argv
    try:
        runpy.run_path(str(train_py), run_name="__main__")
    except Exception as exc:
        debug_payload = {
            "smart_repo_dir": str(smart_repo_dir),
            "config": str(args.config),
            "save_ckpt_path": str(args.save_ckpt_path),
            "seed": seed,
            "deterministic": bool(deterministic),
            "python_version": str(sys.version.split()[0]),
            "torch": _safe_version("torch"),
            "pytorch_lightning": _safe_version("pytorch-lightning"),
            "torch_geometric": _safe_version("torch-geometric"),
            "torch_scatter": _safe_version("torch-scatter"),
            "torch_cluster": _safe_version("torch-cluster"),
            "waymo_open_dataset": _safe_version("waymo-open-dataset-tf-2-12-0"),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        print("[smart-train-repro] training failed")
        print(json.dumps(debug_payload, indent=2, sort_keys=True))
        traceback.print_exc()
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
