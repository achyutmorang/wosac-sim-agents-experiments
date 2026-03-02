from __future__ import annotations

import csv
import hashlib
import json
import shlex
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence


_METRIC_ALIASES: Dict[str, tuple[str, ...]] = {
    "realism_meta_metric": ("realism_meta_metric", "realism", "meta_metric"),
    "simulated_collision_rate": ("simulated_collision_rate", "collision_rate", "collision"),
    "simulated_offroad_rate": ("simulated_offroad_rate", "offroad_rate", "offroad"),
    "simulated_traffic_light_violation_rate": (
        "simulated_traffic_light_violation_rate",
        "traffic_light_violation_rate",
        "tl_violation_rate",
    ),
}


@dataclass
class SmartBaselineFlowBundle:
    summary: Dict[str, Any]
    metrics: Dict[str, Optional[float]]
    command_plan: Dict[str, str]
    artifacts: Dict[str, str]


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


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = _to_serializable(dict(payload))
    path.write_text(json.dumps(serializable, indent=2, sort_keys=True, ensure_ascii=True) + "\n")


def _config_hash(config: Mapping[str, Any]) -> str:
    serializable = _to_serializable(dict(config))
    wire = json.dumps(serializable, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(wire.encode("utf-8")).hexdigest()


def _safe_float(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except Exception:
            return None
    return None


def _flatten_numeric_map(payload: Mapping[str, Any], prefix: str = "") -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key, value in payload.items():
        norm_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            out.update(_flatten_numeric_map(value, prefix=norm_key))
            continue
        num = _safe_float(value)
        if num is not None:
            out[norm_key.lower()] = num
    return out


def _extract_metric_values(flat_map: Mapping[str, float]) -> Dict[str, Optional[float]]:
    metrics: Dict[str, Optional[float]] = {k: None for k in _METRIC_ALIASES}
    for target_key, aliases in _METRIC_ALIASES.items():
        for alias in aliases:
            key = alias.lower()
            if key in flat_map:
                metrics[target_key] = flat_map[key]
                break
            suffix = f".{key}"
            matched_key = next((k for k in flat_map if k.endswith(suffix)), None)
            if matched_key is not None:
                metrics[target_key] = flat_map[matched_key]
                break
    return metrics


def _parse_metrics_json(path: Path) -> tuple[Dict[str, Optional[float]], str]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, Mapping):
        return ({k: None for k in _METRIC_ALIASES}, "json_non_mapping")
    flat = _flatten_numeric_map(payload)
    return _extract_metric_values(flat), "json"


def _parse_metrics_csv(path: Path) -> tuple[Dict[str, Optional[float]], str]:
    flat: Dict[str, float] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            metric_name = str(row.get("metric", "")).strip().lower()
            value = _safe_float(row.get("value"))
            if metric_name and value is not None:
                flat[metric_name] = value
    if not flat:
        return ({k: None for k in _METRIC_ALIASES}, "csv_empty")
    return _extract_metric_values(flat), "csv"


def _safe_git_commit(repo_root: Path) -> str:
    try:
        output = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        commit = output.strip()
        return commit if commit else "unknown"
    except Exception:
        return "unknown"


def _safe_repo_rev(repo_dir: Path) -> str:
    try:
        output = subprocess.check_output(
            ["git", "-C", str(repo_dir), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        rev = output.strip()
        return rev if rev else "unknown"
    except Exception:
        return "unknown"


def _safe_repo_commit(repo_dir: Path) -> str:
    try:
        output = subprocess.check_output(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        rev = output.strip()
        return rev if rev else "unknown"
    except Exception:
        return "unknown"


def _resolve_path(repo_root: Path, value: str) -> Path:
    p = Path(str(value)).expanduser()
    if p.is_absolute():
        return p
    return (repo_root / p).resolve()


def _resolve_config_path(repo_root: Path, value: str) -> str:
    text = str(value).strip()
    if not text:
        return text
    p = Path(text).expanduser()
    if p.is_absolute():
        return str(p)
    candidate = (repo_root / p).resolve()
    if candidate.exists():
        return str(candidate)
    return text


def _q(value: Any) -> str:
    return shlex.quote(str(value))


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _sha256_file(path: Path) -> str:
    if (not path.exists()) or (not path.is_file()):
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_manifest(path: Path, *, sample_size: int = 8, suffixes: Sequence[str] = ()) -> Dict[str, Any]:
    exists = path.exists() and path.is_dir()
    payload: Dict[str, Any] = {
        "path": str(path),
        "exists": bool(exists),
        "num_files": 0,
        "total_bytes": 0,
        "listing_sha256": "",
        "sample_files": [],
    }
    if not exists:
        return payload

    files: List[Path] = [p for p in path.rglob("*") if p.is_file()]
    if suffixes:
        norm_suffixes = tuple(str(s).lower() for s in suffixes if str(s).strip())
        files = [p for p in files if p.suffix.lower() in norm_suffixes]
    files = sorted(files)

    listing_rows: List[str] = []
    total_bytes = 0
    for file_path in files:
        try:
            size = int(file_path.stat().st_size)
        except Exception:
            size = 0
        total_bytes += size
        rel = str(file_path.relative_to(path))
        listing_rows.append(f"{rel}:{size}")

    listing_sha = hashlib.sha256("\n".join(listing_rows).encode("utf-8")).hexdigest() if listing_rows else ""
    payload.update(
        {
            "num_files": len(files),
            "total_bytes": int(total_bytes),
            "listing_sha256": listing_sha,
            "sample_files": [str(p.relative_to(path)) for p in files[: int(max(1, sample_size))]],
        }
    )
    return payload


def _collect_data_manifest(raw_data_root: str, processed_data_root: str) -> Dict[str, Any]:
    raw_root = Path(str(raw_data_root)).expanduser()
    processed_root = Path(str(processed_data_root)).expanduser()
    return {
        "raw_data_root": str(raw_root),
        "processed_data_root": str(processed_root),
        "splits": {
            "training": {
                "raw": _directory_manifest(raw_root / "training"),
                "processed": _directory_manifest(processed_root / "training", suffixes=(".pkl", ".pickle")),
            },
            "validation": {
                "raw": _directory_manifest(raw_root / "validation"),
                "processed": _directory_manifest(processed_root / "validation", suffixes=(".pkl", ".pickle")),
            },
        },
    }


def _collect_checkpoint_manifest(save_ckpt_path: str, pretrain_ckpt_path: str) -> Dict[str, Any]:
    save_dir = Path(str(save_ckpt_path)).expanduser() if str(save_ckpt_path).strip() else None
    manifest: Dict[str, Any] = {
        "save_ckpt_path": str(save_dir) if save_dir is not None else "",
        "save_ckpt_exists": bool(save_dir.exists()) if save_dir is not None else False,
        "checkpoints": [],
        "pretrain_ckpt_path": str(pretrain_ckpt_path).strip(),
        "pretrain_ckpt_sha256": "",
    }
    if save_dir is not None and save_dir.exists():
        ckpts = sorted(save_dir.glob("*.ckpt"))
        for path in ckpts:
            try:
                size = int(path.stat().st_size)
            except Exception:
                size = 0
            manifest["checkpoints"].append(
                {
                    "path": str(path),
                    "size_bytes": size,
                    "sha256": _sha256_file(path),
                }
            )
    pretrain = Path(str(pretrain_ckpt_path)).expanduser()
    if str(pretrain_ckpt_path).strip() and pretrain.exists():
        manifest["pretrain_ckpt_sha256"] = _sha256_file(pretrain)
    return manifest


def _sync_git_repo(repo_url: str, repo_branch: str, repo_dir: Path, repo_commit: str) -> Dict[str, Any]:
    target_commit = str(repo_commit).strip()
    if repo_dir.exists():
        subprocess.run(["git", "-C", str(repo_dir), "fetch", "origin"], check=True)
        subprocess.run(["git", "-C", str(repo_dir), "checkout", repo_branch], check=True)
        subprocess.run(["git", "-C", str(repo_dir), "pull", "--ff-only", "origin", repo_branch], check=True)
        mode = "updated"
    else:
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        clone_cmd = ["git", "clone", "-b", repo_branch]
        if not target_commit:
            clone_cmd.extend(["--depth", "1"])
        clone_cmd.extend([repo_url, str(repo_dir)])
        subprocess.run(clone_cmd, check=True)
        mode = "cloned"
    checked_out_ref = repo_branch
    if target_commit:
        subprocess.run(["git", "-C", str(repo_dir), "fetch", "--tags", "origin"], check=True)
        subprocess.run(["git", "-C", str(repo_dir), "checkout", target_commit], check=True)
        checked_out_ref = target_commit
    return {
        "mode": mode,
        "repo_rev": _safe_repo_rev(repo_dir),
        "repo_commit": _safe_repo_commit(repo_dir),
        "checked_out_ref": checked_out_ref,
    }


def _build_command_plan(
    *,
    repo_root: Path,
    smart_repo_dir: Path,
    train_config: str,
    val_config: str,
    ckpt_path: str,
    save_ckpt_path: str,
    raw_data_root: str,
    processed_data_root: str,
    install_pyg: bool,
    env_lockfile: str,
    train_seed: int,
    deterministic_train: bool,
    train_launcher_path: str,
) -> Dict[str, str]:
    setup_steps = [f"cd {_q(smart_repo_dir)}"]
    if str(env_lockfile).strip():
        setup_steps.append(f"python -m pip install -r {_q(env_lockfile)}")
    else:
        setup_steps.append("python -m pip install -r requirements.txt")
    setup_steps.append("bash scripts/install_pyg.sh" if install_pyg else "echo 'skip install_pyg.sh'")
    setup_cmd = " && ".join(setup_steps)

    preprocess_train_cmd = " && ".join(
        [
            f"cd {_q(smart_repo_dir)}",
            " ".join(
                [
                    "python data_preprocess.py",
                    f"--input_dir {_q(str(Path(raw_data_root) / 'training'))}",
                    f"--output_dir {_q(str(Path(processed_data_root) / 'training'))}",
                ]
            ),
        ]
    )
    preprocess_val_cmd = " && ".join(
        [
            f"cd {_q(smart_repo_dir)}",
            " ".join(
                [
                    "python data_preprocess.py",
                    f"--input_dir {_q(str(Path(raw_data_root) / 'validation'))}",
                    f"--output_dir {_q(str(Path(processed_data_root) / 'validation'))}",
                ]
            ),
        ]
    )

    train_parts: List[str] = [
        f"python {_q(train_launcher_path)}",
        f"--smart-repo-dir {_q(smart_repo_dir)}",
        f"--config {_q(train_config)}",
        f"--seed {int(train_seed)}",
    ]
    if deterministic_train:
        train_parts.append("--deterministic")
    else:
        train_parts.append("--no-deterministic")
    if save_ckpt_path:
        train_parts.append(f"--save-ckpt-path {_q(save_ckpt_path)}")
    env_prefix = f"SMART_TRAIN_SEED={int(train_seed)} PYTHONHASHSEED={int(train_seed)}"
    if deterministic_train:
        env_prefix += " CUBLAS_WORKSPACE_CONFIG=:4096:8"
    train_cmd = " && ".join(
        [
            f"cd {_q(repo_root)}",
            f"{env_prefix} {' '.join(train_parts)}",
        ]
    )

    val_parts = [
        f"cd {_q(smart_repo_dir)}",
        f"python val.py --config {_q(val_config)}",
    ]
    if ckpt_path:
        val_parts[-1] += f" --pretrain_ckpt {_q(ckpt_path)}"
    validate_cmd = " && ".join(val_parts)
    return {
        "setup_cmd": setup_cmd,
        "preprocess_train_cmd": preprocess_train_cmd,
        "preprocess_val_cmd": preprocess_val_cmd,
        "train_cmd": train_cmd,
        "validate_cmd": validate_cmd,
    }


def run_smart_baseline_flow(**kwargs: Any) -> SmartBaselineFlowBundle:
    repo_root = Path(str(kwargs.get("repo_root", "."))).resolve()
    run_tag = str(kwargs.get("run_tag", _utc_now_iso().replace("-", "").replace(":", "")))
    run_name = str(kwargs.get("run_name", "dev"))
    run_prefix = str(kwargs.get("run_prefix", "smart_baseline"))
    persist_root = Path(str(kwargs.get("persist_root", "/content/drive/MyDrive/wosac_experiments"))).expanduser()

    smart_repo_url = str(kwargs.get("smart_repo_url", "https://github.com/rainmaker22/SMART.git"))
    smart_repo_branch = str(kwargs.get("smart_repo_branch", "main"))
    smart_repo_commit = str(kwargs.get("smart_repo_commit", "")).strip()
    smart_repo_dir = Path(str(kwargs.get("smart_repo_dir", "/content/SMART")))
    train_config = _resolve_config_path(repo_root, str(kwargs.get("smart_train_config", "configs/train/train_scalable.yaml")))
    val_config = _resolve_config_path(
        repo_root, str(kwargs.get("smart_val_config", "configs/validation/validation_scalable.yaml"))
    )
    ckpt_path = str(kwargs.get("smart_ckpt_path", "")).strip()
    save_ckpt_path = str(kwargs.get("smart_save_ckpt_path", "")).strip()
    raw_data_root = str(kwargs.get("smart_raw_data_root", "/content/SMART/data/waymo/scenario"))
    processed_data_root = str(kwargs.get("smart_processed_data_root", "/content/SMART/data/waymo_processed"))
    install_pyg = bool(kwargs.get("smart_install_pyg", False))
    env_lockfile_arg = str(kwargs.get("smart_env_lockfile", "")).strip()
    env_lockfile = str(_resolve_path(repo_root, env_lockfile_arg)) if env_lockfile_arg else ""
    train_seed = _safe_int(kwargs.get("smart_train_seed", 2), 2)
    deterministic_train = bool(kwargs.get("smart_deterministic_train", True))
    smart_profile = str(kwargs.get("smart_profile", "default")).strip() or "default"
    train_launcher_arg = str(kwargs.get("smart_train_launcher_path", "scripts/smart_train_repro.py")).strip()
    train_launcher_path = str(_resolve_path(repo_root, train_launcher_arg))
    sync_smart_repo = bool(kwargs.get("sync_smart_repo", False))

    metrics: Dict[str, Optional[float]] = {k: None for k in _METRIC_ALIASES}
    metrics_source = "none"
    metric_ingest_error = ""
    metrics_json_arg = str(kwargs.get("official_metrics_json", "")).strip()
    metrics_csv_arg = str(kwargs.get("metrics_csv", "")).strip()
    if metrics_json_arg:
        path = Path(metrics_json_arg)
        try:
            if path.exists():
                metrics, metrics_source = _parse_metrics_json(path)
            else:
                metric_ingest_error = f"missing_json:{path}"
        except Exception as exc:
            metric_ingest_error = f"json_parse_error:{type(exc).__name__}"
    elif metrics_csv_arg:
        path = Path(metrics_csv_arg)
        try:
            if path.exists():
                metrics, metrics_source = _parse_metrics_csv(path)
            else:
                metric_ingest_error = f"missing_csv:{path}"
        except Exception as exc:
            metric_ingest_error = f"csv_parse_error:{type(exc).__name__}"

    sync_result: Dict[str, Any] = {
        "mode": "skipped",
        "repo_rev": _safe_repo_rev(smart_repo_dir),
        "repo_commit": _safe_repo_commit(smart_repo_dir),
        "checked_out_ref": "",
    }
    sync_error = ""
    if sync_smart_repo:
        try:
            sync_result = _sync_git_repo(
                repo_url=smart_repo_url,
                repo_branch=smart_repo_branch,
                repo_dir=smart_repo_dir,
                repo_commit=smart_repo_commit,
            )
        except Exception as exc:
            sync_error = f"{type(exc).__name__}: {exc}"
            sync_result = {"mode": "failed", "repo_rev": "unknown", "repo_commit": "unknown", "checked_out_ref": ""}

    command_plan = _build_command_plan(
        repo_root=repo_root,
        smart_repo_dir=smart_repo_dir,
        train_config=train_config,
        val_config=val_config,
        ckpt_path=ckpt_path,
        save_ckpt_path=save_ckpt_path,
        raw_data_root=raw_data_root,
        processed_data_root=processed_data_root,
        install_pyg=install_pyg,
        env_lockfile=env_lockfile,
        train_seed=train_seed,
        deterministic_train=deterministic_train,
        train_launcher_path=train_launcher_path,
    )
    data_manifest = _collect_data_manifest(raw_data_root=raw_data_root, processed_data_root=processed_data_root)
    checkpoint_manifest = _collect_checkpoint_manifest(save_ckpt_path=save_ckpt_path, pretrain_ckpt_path=ckpt_path)

    has_primary_metric = metrics.get("realism_meta_metric") is not None
    status = "metrics_loaded" if has_primary_metric else "ready"
    if sync_error:
        status = "sync_failed"

    serializable_kwargs = _to_serializable(dict(kwargs))
    run_dir = persist_root / f"{run_prefix}_{run_name}"
    outputs_dir = run_dir / "outputs"
    summary_path = outputs_dir / "smart_baseline_flow_summary.json"
    plan_path = outputs_dir / "smart_command_plan.json"
    metrics_path = outputs_dir / "smart_baseline_metrics_snapshot.json"
    data_manifest_path = outputs_dir / "smart_data_manifest.json"
    checkpoint_manifest_path = outputs_dir / "smart_checkpoint_manifest.json"

    summary: Dict[str, Any] = {
        "status": status,
        "run_tag": run_tag,
        "run_name": run_name,
        "run_prefix": run_prefix,
        "persist_root": str(persist_root),
        "repo_root": str(repo_root),
        "repo_commit": _safe_git_commit(repo_root),
        "smart_repo_url": smart_repo_url,
        "smart_repo_branch": smart_repo_branch,
        "smart_repo_commit_target": smart_repo_commit,
        "smart_repo_dir": str(smart_repo_dir),
        "smart_repo_sync": sync_result,
        "smart_profile": smart_profile,
        "smart_train_seed": int(train_seed),
        "smart_deterministic_train": bool(deterministic_train),
        "smart_env_lockfile": env_lockfile,
        "smart_train_launcher_path": train_launcher_path,
        "smart_train_config": train_config,
        "smart_val_config": val_config,
        "sync_error": sync_error,
        "metrics_source": metrics_source,
        "metric_ingest_error": metric_ingest_error,
        "data_manifest": data_manifest,
        "checkpoint_manifest": checkpoint_manifest,
        "created_utc": _utc_now_iso(),
        "config_hash": _config_hash({k: v for k, v in kwargs.items() if isinstance(k, str)}),
        "kwargs": serializable_kwargs,
    }

    artifacts: Dict[str, str] = {"run_dir": str(run_dir)}
    artifact_error = ""
    try:
        _write_json(summary_path, summary)
        _write_json(plan_path, command_plan)
        _write_json(
            metrics_path,
            {
                "run_tag": run_tag,
                "created_utc": summary["created_utc"],
                "metrics_source": metrics_source,
                "metrics": metrics,
            },
        )
        _write_json(data_manifest_path, data_manifest)
        _write_json(checkpoint_manifest_path, checkpoint_manifest)
        artifacts.update(
            {
                "summary_json": str(summary_path),
                "command_plan_json": str(plan_path),
                "metrics_json": str(metrics_path),
                "data_manifest_json": str(data_manifest_path),
                "checkpoint_manifest_json": str(checkpoint_manifest_path),
            }
        )
    except Exception as exc:
        artifact_error = f"{type(exc).__name__}: {exc}"
        summary["artifact_error"] = artifact_error

    return SmartBaselineFlowBundle(summary=summary, metrics=metrics, command_plan=command_plan, artifacts=artifacts)
