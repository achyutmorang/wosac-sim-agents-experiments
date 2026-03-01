from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS_COLAB_PATH = REPO_ROOT / "requirements-colab.txt"
NUMERIC_REPAIR_REQUIREMENTS = "numpy==2.2.6 scipy==1.14.1 pandas==2.2.3 scikit-learn==1.6.1"


def _run_cmd(args: List[str]) -> subprocess.CompletedProcess:
    print(f"[setup] $ {' '.join(args)}")
    return subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)


def _pip(args: List[str]) -> None:
    cmd = [sys.executable, "-m", "pip", *args]
    print(f"[setup] $ {' '.join(cmd)}")
    rc = subprocess.run(cmd, check=False)
    if rc.returncode != 0:
        raise RuntimeError(f"Command failed ({rc.returncode}): {' '.join(cmd)}")


def _probe_numpy_runtime() -> tuple[bool, str]:
    probe = _run_cmd(
        [
            sys.executable,
            "-c",
            "import numpy as np; from numpy._core.umath import _center, _expandtabs; print(np.__version__, np.__file__)",
        ]
    )
    if probe.returncode == 0:
        msg = probe.stdout.strip() or "NumPy probe succeeded."
        return True, msg
    err = (probe.stderr or probe.stdout).strip()
    return False, err


def _probe_core_runtime() -> tuple[bool, str]:
    probe = _run_cmd(
        [
            sys.executable,
            "-c",
            (
                "import jax, waymax, numpy as np, pandas as pd, scipy, sklearn; "
                "from numpy._core.umath import _center, _expandtabs; "
                "print('ok', np.__version__, pd.__version__, scipy.__version__, sklearn.__version__, jax.__version__)"
            ),
        ]
    )
    if probe.returncode == 0:
        return True, (probe.stdout.strip() or "core runtime probe succeeded")
    return False, ((probe.stderr or probe.stdout).strip())


def _repair_numeric_stack() -> None:
    _run_cmd([sys.executable, "-m", "pip", "uninstall", "-y", "numpy", "scipy", "pandas", "scikit-learn"])
    _pip(["install", "--no-cache-dir", "--force-reinstall", *NUMERIC_REPAIR_REQUIREMENTS.split()])


def _normalize_dist_name(name: str) -> str:
    return name.strip().lower().replace("_", "-")


def _parse_exact_pins(requirements_path: Path) -> Dict[str, str]:
    pins: Dict[str, str] = {}
    for raw in requirements_path.read_text().splitlines():
        line = raw.strip()
        if (not line) or line.startswith("#") or line.startswith("--") or line.startswith("git+"):
            continue
        if "==" not in line:
            continue
        name, version = line.split("==", 1)
        name = name.split("[", 1)[0].strip()
        version = version.strip()
        if name and version:
            pins[_normalize_dist_name(name)] = version
    return pins


def _installed_version(dist_name: str) -> str | None:
    try:
        return metadata.version(dist_name)
    except metadata.PackageNotFoundError:
        return None


def _collect_version_mismatches(requirements_path: Path) -> List[str]:
    pins = _parse_exact_pins(requirements_path)
    mismatches: List[str] = []
    for dist_name, expected in sorted(pins.items()):
        actual = _installed_version(dist_name)
        if actual != expected:
            mismatches.append(f"{dist_name}: have={actual!r}, want={expected!r}")
    return mismatches


def _install_requirements(requirements_path: Path, upgrade_pip: bool = False) -> None:
    if not requirements_path.exists():
        raise FileNotFoundError(f"Missing requirements file: {requirements_path}")
    if upgrade_pip:
        _pip(["install", "--upgrade", "pip"])
    _pip(["install", "-r", str(requirements_path)])


def run_deterministic_setup(
    force_reinstall: bool = False,
    auto_restart: bool = False,
    strict_lock: bool = True,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "ran_setup": True,
        "force_reinstall": bool(force_reinstall),
        "strict_lock": bool(strict_lock),
        "installed_requirements": False,
        "repaired_numeric_stack": False,
        "restart_required": False,
        "numpy_probe": "",
        "core_probe": "",
        "version_mismatches": [],
        "kernel_executable": sys.executable,
    }

    print("[setup] Starting deterministic environment bootstrap")

    mismatches = _collect_version_mismatches(REQUIREMENTS_COLAB_PATH) if strict_lock else []
    result["version_mismatches"] = mismatches

    core_ok, core_details = _probe_core_runtime()
    result["core_probe"] = core_details

    if (not force_reinstall) and core_ok and (not mismatches):
        print(f"[setup] Core runtime already healthy; skipping dependency install ({core_details}).")
    else:
        if (not force_reinstall) and (not mismatches) and (not core_ok):
            numpy_ok, numpy_details = _probe_numpy_runtime()
            if not numpy_ok:
                print(f"[setup] NumPy probe failed; applying targeted numeric repair.\n[setup] probe error: {numpy_details}")
                _repair_numeric_stack()
                result["repaired_numeric_stack"] = True
                core_ok, core_details = _probe_core_runtime()
                result["core_probe"] = core_details

        if force_reinstall or mismatches or (not core_ok):
            if force_reinstall:
                print("[setup] force_reinstall=True -> running dependency install.")
            elif mismatches:
                print("[setup] Detected lockfile mismatches -> running dependency install.")
            else:
                print(f"[setup] Core runtime unhealthy -> running dependency install.\n[setup] probe error: {core_details}")
            _install_requirements(REQUIREMENTS_COLAB_PATH, upgrade_pip=bool(force_reinstall))
            result["installed_requirements"] = True

    ok, details = _probe_numpy_runtime()
    if not ok:
        print("[setup] NumPy probe failed after dependency install, repairing numeric stack...")
        _repair_numeric_stack()
        result["repaired_numeric_stack"] = True
        ok, details = _probe_numpy_runtime()
        if not ok:
            raise RuntimeError(
                "NumPy runtime probe failed after repair attempt. Retry setup with force_reinstall=True.\n"
                f"Probe error: {details}"
            )
    print(f"[setup] NumPy probe passed ({details}).")
    result["numpy_probe"] = details

    core_ok, core_details = _probe_core_runtime()
    if not core_ok:
        raise RuntimeError(
            "Core runtime probe failed after setup. "
            f"Probe error: {core_details}\n"
            "Retry with force_reinstall=True."
        )
    print(f"[setup] Core runtime probe passed ({core_details}).")
    result["core_probe"] = core_details

    result["restart_required"] = bool(result["installed_requirements"] or result["repaired_numeric_stack"])
    if result["restart_required"]:
        print("Setup complete. Runtime restart is required before running experiment cells.")
        if auto_restart and os.environ.get("COLAB_RELEASE_TAG"):
            print("[setup] Auto-restart enabled. Restarting kernel in 2 seconds...")
            time.sleep(2)
            os.kill(os.getpid(), 9)
    else:
        print("Setup complete. No restart required; dependencies already matched.")

    return result


if __name__ == "__main__":
    force_flag = os.environ.get("FORCE_REINSTALL", "false").strip().lower()
    restart_flag = os.environ.get("AUTO_RESTART", "false").strip().lower()
    strict_flag = os.environ.get("STRICT_LOCK", "true").strip().lower()
    out = run_deterministic_setup(
        force_reinstall=force_flag in {"1", "true", "yes", "y"},
        auto_restart=restart_flag in {"1", "true", "yes", "y"},
        strict_lock=strict_flag not in {"0", "false", "no", "n"},
    )
    print("[setup] result:", json.dumps(out, indent=2))
