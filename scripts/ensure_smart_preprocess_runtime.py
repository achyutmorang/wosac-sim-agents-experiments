#!/usr/bin/env python3
from __future__ import annotations

import importlib
import subprocess
import sys
from typing import Iterable


def _pip_install(args: Iterable[str]) -> None:
    cmd = [sys.executable, "-m", "pip", *list(args)]
    print("[smart-preprocess-setup] $", " ".join(cmd))
    rc = subprocess.run(cmd, check=False)
    if rc.returncode != 0:
        raise RuntimeError(f"pip install failed ({rc.returncode}): {' '.join(cmd)}")


def _import_ok(module_name: str) -> tuple[bool, str]:
    try:
        importlib.import_module(module_name)
        return True, ""
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _ensure_module(module_name: str, package_name: str) -> None:
    ok, err = _import_ok(module_name)
    if ok:
        return
    print(f"[smart-preprocess-setup] missing {module_name}: {err}")
    _pip_install(["install", "--upgrade", package_name])
    ok, err = _import_ok(module_name)
    if not ok:
        raise RuntimeError(f"Unable to import {module_name} after install: {err}")


def _ensure_waymo_open_dataset() -> None:
    ok, err = _import_ok("waymo_open_dataset")
    if ok:
        return

    py_major, py_minor = sys.version_info[:2]
    print(f"[smart-preprocess-setup] missing waymo_open_dataset: {err}")
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
            _pip_install(args)
            ok, err = _import_ok("waymo_open_dataset")
            if ok:
                return
            errors.append(f"{' '.join(args)} -> import failed: {err}")
        except Exception as exc:
            errors.append(f"{' '.join(args)} -> {type(exc).__name__}: {exc}")

    raise RuntimeError(
        "Unable to install/import waymo_open_dataset for SMART preprocessing. "
        + " | ".join(errors)
    )


def main() -> int:
    ok, err = _import_ok("tensorflow")
    if not ok:
        raise RuntimeError(
            "TensorFlow is required for SMART preprocessing but is not importable in this runtime: "
            f"{err}"
        )

    _ensure_module("easydict", "easydict==1.13")
    _ensure_module("pandas", "pandas")
    _ensure_waymo_open_dataset()

    # Validate the exact imports used by SMART's preprocessing entrypoint.
    importlib.import_module("tensorflow")
    importlib.import_module("torch")
    importlib.import_module("waymo_open_dataset.protos.scenario_pb2")
    print("[smart-preprocess-setup] runtime ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
