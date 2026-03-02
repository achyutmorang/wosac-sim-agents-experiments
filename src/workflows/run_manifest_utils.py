from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence


DEFAULT_PACKAGES: tuple[str, ...] = (
    "numpy",
    "pandas",
    "torch",
    "tensorflow",
    "jax",
    "waymo-open-dataset-tf-2-12-0",
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _to_serializable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_serializable(v) for v in value]
    return str(value)


def _safe_version(package_name: str) -> str:
    try:
        from importlib.metadata import version

        return str(version(package_name))
    except Exception:
        return "not_installed"


def detect_runtime_type() -> str:
    import os

    if str(os.environ.get("COLAB_TPU_ADDR", "")).strip():
        return "tpu"
    if str(os.environ.get("COLAB_GPU", "")).strip():
        return "gpu"
    if str(os.environ.get("NVIDIA_VISIBLE_DEVICES", "")).strip() not in {"", "none", "void"}:
        return "gpu"
    return "cpu"


def resolve_git_commit(repo_dir: str | Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(Path(str(repo_dir)).expanduser()), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        commit = out.strip()
        return commit if commit else "unknown"
    except Exception:
        return "unknown"


def collect_package_versions(package_names: Optional[Sequence[str]] = None) -> Dict[str, str]:
    names = list(package_names) if package_names is not None else list(DEFAULT_PACKAGES)
    return {str(name): _safe_version(str(name)) for name in names if str(name).strip()}


def write_json(path: str | Path, payload: Mapping[str, Any]) -> str:
    target = Path(str(path)).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(_to_serializable(dict(payload)), indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return str(target)


def build_training_run_manifest(
    *,
    run_id: str,
    run_tag: str,
    experiment_slug: str,
    run_name: str,
    run_prefix: str,
    persist_root: str,
    repo_root: str,
    config_hash: str,
    flow_summary: Optional[Mapping[str, Any]] = None,
    stage_flags: Optional[Mapping[str, Any]] = None,
    checkpoint_dir: str = "",
    resume_checkpoint_path: str = "",
    resume_checkpoint_source: str = "",
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    summary = dict(flow_summary or {})
    smart_repo_dir = str(summary.get("smart_repo_dir", "")).strip()
    manifest: Dict[str, Any] = {
        "run_id": str(run_id),
        "run_tag": str(run_tag),
        "created_utc": _utc_now_iso(),
        "experiment_slug": str(experiment_slug),
        "run_name": str(run_name),
        "run_prefix": str(run_prefix),
        "persist_root": str(persist_root),
        "repo_root": str(repo_root),
        "repo_commit": resolve_git_commit(repo_root),
        "smart_repo_dir": smart_repo_dir,
        "smart_repo_commit": resolve_git_commit(smart_repo_dir) if smart_repo_dir else "unknown",
        "config_hash": str(config_hash),
        "runtime": {
            "python_version": str(sys.version.split()[0]),
            "platform": str(platform.platform()),
            "runtime_type": detect_runtime_type(),
            "packages": collect_package_versions(),
        },
        "training": {
            "smart_profile": str(summary.get("smart_profile", "")),
            "smart_train_seed": summary.get("smart_train_seed"),
            "smart_deterministic_train": summary.get("smart_deterministic_train"),
            "smart_train_config": str(summary.get("smart_train_config", "")),
            "smart_val_config": str(summary.get("smart_val_config", "")),
            "smart_raw_data_root": str((summary.get("kwargs", {}) or {}).get("smart_raw_data_root", "")),
            "smart_processed_data_root": str((summary.get("kwargs", {}) or {}).get("smart_processed_data_root", "")),
            "checkpoint_dir": str(checkpoint_dir),
            "resume_checkpoint_path": str(resume_checkpoint_path),
            "resume_checkpoint_source": str(resume_checkpoint_source),
            "resume_checkpoint_exists": bool(Path(str(resume_checkpoint_path)).expanduser().exists())
            if str(resume_checkpoint_path).strip()
            else False,
            "resume_from_existing": bool(summary.get("resume_from_existing", True)),
            "stage_flags": _to_serializable(stage_flags or {}),
            "data_manifest": _to_serializable(summary.get("data_manifest", {})),
            "checkpoint_manifest": _to_serializable(summary.get("checkpoint_manifest", {})),
        },
    }
    if isinstance(extra, Mapping) and extra:
        manifest["extra"] = _to_serializable(dict(extra))
    return manifest
