from __future__ import annotations

import hashlib
import json
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
    "diversity_score": ("diversity_score", "rollout_diversity", "sample_diversity", "diversity"),
}


@dataclass
class SmartEvalBundle:
    summary: Dict[str, Any]
    models: List[Dict[str, Any]]
    artifacts: Dict[str, str]


@dataclass
class SmartComparativeBundle:
    summary: Dict[str, Any]
    comparison: List[Dict[str, Any]]
    selection: Dict[str, Any]
    artifacts: Dict[str, str]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


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


def _flatten_numeric_map(payload: Mapping[str, Any], prefix: str = "") -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key, value in payload.items():
        norm_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            out.update(_flatten_numeric_map(value, prefix=norm_key))
            continue
        number = _safe_float(value)
        if number is not None:
            out[norm_key.lower()] = number
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
            matched = next((k for k in flat_map if k.endswith(suffix)), None)
            if matched is not None:
                metrics[target_key] = flat_map[matched]
                break
    return metrics


def _extract_metrics_from_json(path: Path) -> Dict[str, Optional[float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, Mapping) and isinstance(payload.get("metrics"), Mapping):
        source = payload["metrics"]
    elif isinstance(payload, Mapping):
        source = payload
    else:
        return {k: None for k in _METRIC_ALIASES}
    flat = _flatten_numeric_map(source)
    return _extract_metric_values(flat)


def _resolve_path(repo_root: Path, value: str) -> Path:
    p = Path(str(value)).expanduser()
    if p.is_absolute():
        return p
    return (repo_root / p).resolve()


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


def _sync_git_repo(repo_url: str, repo_branch: str, repo_dir: Path) -> Dict[str, Any]:
    if repo_dir.exists():
        subprocess.run(["git", "-C", str(repo_dir), "fetch", "origin"], check=True)
        subprocess.run(["git", "-C", str(repo_dir), "checkout", repo_branch], check=True)
        subprocess.run(["git", "-C", str(repo_dir), "pull", "--ff-only", "origin", repo_branch], check=True)
        mode = "updated"
    else:
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", "-b", repo_branch, repo_url, str(repo_dir)], check=True)
        mode = "cloned"
    return {"mode": mode, "repo_rev": _safe_repo_rev(repo_dir)}


def _inject_env(cmd: str, env_map: Mapping[str, Any], needle: str) -> str:
    if not env_map:
        return cmd
    env_prefix = " ".join([f"{k}={v}" for k, v in env_map.items()])
    if needle in cmd:
        return cmd.replace(needle, f"{env_prefix} {needle}", 1)
    return f"{env_prefix} {cmd}"


def _parse_models(models: Any) -> List[Dict[str, Any]]:
    if models is None:
        return []
    if isinstance(models, Sequence) and not isinstance(models, (bytes, bytearray, str)):
        out: List[Dict[str, Any]] = []
        for elem in models:
            if isinstance(elem, Mapping):
                out.append(dict(elem))
        return out
    if isinstance(models, str):
        text = models.strip()
        if not text:
            return []
        try:
            payload = json.loads(text)
        except Exception:
            return []
        if isinstance(payload, Sequence):
            out: List[Dict[str, Any]] = []
            for elem in payload:
                if isinstance(elem, Mapping):
                    out.append(dict(elem))
            return out
    return []


def _build_default_models(
    *,
    baseline_model_id: str,
    baseline_ckpt_path: str,
    variant_ckpt_paths: Sequence[str],
) -> List[Dict[str, Any]]:
    models: List[Dict[str, Any]] = []
    base_ckpt = str(baseline_ckpt_path).strip()
    if base_ckpt:
        models.append({"model_id": baseline_model_id, "checkpoint_path": base_ckpt, "env": {}})
    for idx, ckpt in enumerate(variant_ckpt_paths, start=1):
        ckpt_path = str(ckpt).strip()
        if not ckpt_path:
            continue
        models.append({"model_id": f"variant_{idx}", "checkpoint_path": ckpt_path, "env": {}})
    return models


def _parse_variant_ckpt_paths(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [str(v).strip() for v in value if str(v).strip()]
    return []


def _build_validate_cmd(
    *,
    smart_repo_dir: Path,
    val_config: str,
    checkpoint_path: str,
    env_map: Mapping[str, Any],
) -> str:
    parts = [
        f"cd {smart_repo_dir}",
        f"python val.py --config {val_config}",
    ]
    if str(checkpoint_path).strip():
        parts[-1] += f" --pretrain_ckpt {checkpoint_path}"
    validate_cmd = " && ".join(parts)
    return _inject_env(validate_cmd, env_map=env_map, needle="python val.py")


def run_smart_eval_flow(**kwargs: Any) -> SmartEvalBundle:
    repo_root = Path(str(kwargs.get("repo_root", "."))).resolve()
    run_tag = str(kwargs.get("run_tag", _utc_now_iso().replace("-", "").replace(":", "")))
    run_name = str(kwargs.get("run_name", "dev"))
    run_prefix = str(kwargs.get("run_prefix", "smart_eval"))
    persist_root = Path(str(kwargs.get("persist_root", "/content/drive/MyDrive/wosac_experiments"))).expanduser()

    smart_repo_url = str(kwargs.get("smart_repo_url", "https://github.com/rainmaker22/SMART.git"))
    smart_repo_branch = str(kwargs.get("smart_repo_branch", "main"))
    smart_repo_dir = Path(str(kwargs.get("smart_repo_dir", "/content/SMART")))
    smart_val_config = str(kwargs.get("smart_val_config", "configs/validation/validation_scalable.yaml"))
    sync_smart_repo = bool(kwargs.get("sync_smart_repo", False))

    models = _parse_models(kwargs.get("models"))
    if not models:
        models = _build_default_models(
            baseline_model_id=str(kwargs.get("baseline_model_id", "smart_baseline")),
            baseline_ckpt_path=str(kwargs.get("baseline_ckpt_path", "")),
            variant_ckpt_paths=_parse_variant_ckpt_paths(kwargs.get("variant_ckpt_paths")),
        )

    sync_result: Dict[str, Any] = {"mode": "skipped", "repo_rev": _safe_repo_rev(smart_repo_dir)}
    sync_error = ""
    if sync_smart_repo:
        try:
            sync_result = _sync_git_repo(
                repo_url=smart_repo_url,
                repo_branch=smart_repo_branch,
                repo_dir=smart_repo_dir,
            )
        except Exception as exc:
            sync_error = f"{type(exc).__name__}: {exc}"
            sync_result = {"mode": "failed", "repo_rev": "unknown"}

    metrics_dir_arg = str(kwargs.get("metrics_dir", "")).strip()
    metrics_dir = _resolve_path(repo_root, metrics_dir_arg) if metrics_dir_arg else None

    model_rows: List[Dict[str, Any]] = []
    for model in models:
        model_id = str(model.get("model_id", "")).strip()
        if not model_id:
            continue
        checkpoint_path = str(model.get("checkpoint_path", "")).strip()
        env_map = model.get("env", {})
        env_data = dict(env_map) if isinstance(env_map, Mapping) else {}
        validate_cmd = _build_validate_cmd(
            smart_repo_dir=smart_repo_dir,
            val_config=smart_val_config,
            checkpoint_path=checkpoint_path,
            env_map=env_data,
        )

        metrics = {k: None for k in _METRIC_ALIASES}
        metrics_source = "none"
        metrics_json_arg = str(model.get("metrics_json", "")).strip()
        metrics_path: Optional[Path] = None
        if metrics_json_arg:
            metrics_path = _resolve_path(repo_root, metrics_json_arg)
        elif metrics_dir is not None:
            metrics_path = metrics_dir / f"{model_id}.json"
        if metrics_path is not None:
            try:
                if metrics_path.exists():
                    metrics = _extract_metrics_from_json(metrics_path)
                    metrics_source = "json"
                else:
                    metrics_source = "missing"
            except Exception:
                metrics_source = "parse_error"

        model_rows.append(
            {
                "model_id": model_id,
                "checkpoint_path": checkpoint_path,
                "checkpoint_exists": bool(Path(checkpoint_path).expanduser().exists()) if checkpoint_path else False,
                "env": env_data,
                "validate_cmd": validate_cmd,
                "metrics_json": str(metrics_path) if metrics_path is not None else "",
                "metrics_source": metrics_source,
                "metrics": metrics,
            }
        )

    status = "ready"
    if sync_error:
        status = "sync_failed"
    elif not model_rows:
        status = "no_models"

    run_dir = persist_root / f"{run_prefix}_{run_name}"
    outputs_dir = run_dir / "outputs"
    summary_path = outputs_dir / "smart_eval_flow_summary.json"
    model_grid_path = outputs_dir / "smart_eval_model_grid.json"

    summary: Dict[str, Any] = {
        "status": status,
        "run_tag": run_tag,
        "run_name": run_name,
        "run_prefix": run_prefix,
        "persist_root": str(persist_root),
        "repo_root": str(repo_root),
        "smart_repo_url": smart_repo_url,
        "smart_repo_branch": smart_repo_branch,
        "smart_repo_dir": str(smart_repo_dir),
        "smart_repo_sync": sync_result,
        "sync_error": sync_error,
        "metrics_dir": str(metrics_dir) if metrics_dir is not None else "",
        "num_models": len(model_rows),
        "created_utc": _utc_now_iso(),
        "config_hash": _config_hash({k: v for k, v in kwargs.items() if isinstance(k, str)}),
    }

    artifacts: Dict[str, str] = {"run_dir": str(run_dir)}
    try:
        _write_json(summary_path, summary)
        _write_json(
            model_grid_path,
            {
                "run_tag": run_tag,
                "created_utc": summary["created_utc"],
                "models": model_rows,
            },
        )
        artifacts.update({"summary_json": str(summary_path), "model_grid_json": str(model_grid_path)})
    except Exception as exc:
        summary["artifact_error"] = f"{type(exc).__name__}: {exc}"

    return SmartEvalBundle(summary=summary, models=model_rows, artifacts=artifacts)


def _constraint_check(
    *,
    metrics: Mapping[str, Optional[float]],
    max_collision_rate: Optional[float],
    max_offroad_rate: Optional[float],
    max_tl_violation_rate: Optional[float],
    min_diversity_score: Optional[float],
) -> Dict[str, Any]:
    violations: List[str] = []

    def _upper(metric_key: str, bound: Optional[float], bound_key: str) -> None:
        if bound is None:
            return
        value = metrics.get(metric_key)
        if value is None:
            violations.append(f"missing:{metric_key}")
            return
        if float(value) > float(bound):
            violations.append(f"{metric_key}>{bound_key}")

    _upper("simulated_collision_rate", max_collision_rate, "max_collision_rate")
    _upper("simulated_offroad_rate", max_offroad_rate, "max_offroad_rate")
    _upper(
        "simulated_traffic_light_violation_rate",
        max_tl_violation_rate,
        "max_tl_violation_rate",
    )

    if min_diversity_score is not None:
        diversity = metrics.get("diversity_score")
        if diversity is None:
            violations.append("missing:diversity_score")
        elif float(diversity) < float(min_diversity_score):
            violations.append("diversity_score<min_diversity_score")

    return {"feasible": len(violations) == 0, "violations": violations}


def run_smart_comparative_flow(**kwargs: Any) -> SmartComparativeBundle:
    repo_root = Path(str(kwargs.get("repo_root", "."))).resolve()
    run_tag = str(kwargs.get("run_tag", _utc_now_iso().replace("-", "").replace(":", "")))
    run_name = str(kwargs.get("run_name", "dev"))
    run_prefix = str(kwargs.get("run_prefix", "smart_eval"))
    persist_root = Path(str(kwargs.get("persist_root", "/content/drive/MyDrive/wosac_experiments"))).expanduser()
    baseline_model_id = str(kwargs.get("baseline_model_id", "smart_baseline"))
    primary_metric = str(kwargs.get("primary_metric", "realism_meta_metric"))

    max_collision_rate = _safe_float(kwargs.get("max_collision_rate"))
    max_offroad_rate = _safe_float(kwargs.get("max_offroad_rate"))
    max_tl_violation_rate = _safe_float(kwargs.get("max_tl_violation_rate"))
    min_diversity_score = _safe_float(kwargs.get("min_diversity_score"))

    eval_models_json = _resolve_path(repo_root, str(kwargs.get("eval_models_json", "")).strip())
    payload = json.loads(eval_models_json.read_text(encoding="utf-8"))
    raw_models = payload.get("models", []) if isinstance(payload, Mapping) else []
    models: List[Dict[str, Any]] = [dict(m) for m in raw_models if isinstance(m, Mapping)]

    baseline = next((m for m in models if str(m.get("model_id")) == baseline_model_id), None)
    baseline_primary: Optional[float] = None
    if baseline is not None:
        baseline_primary = _safe_float(dict(baseline.get("metrics", {})).get(primary_metric))

    comparison: List[Dict[str, Any]] = []
    for model in models:
        model_id = str(model.get("model_id", ""))
        metrics = dict(model.get("metrics", {}))
        primary_value = _safe_float(metrics.get(primary_metric))
        constraint_check = _constraint_check(
            metrics=metrics,
            max_collision_rate=max_collision_rate,
            max_offroad_rate=max_offroad_rate,
            max_tl_violation_rate=max_tl_violation_rate,
            min_diversity_score=min_diversity_score,
        )
        delta_primary = None
        if baseline_primary is not None and primary_value is not None:
            delta_primary = float(primary_value) - float(baseline_primary)
        comparison.append(
            {
                "model_id": model_id,
                "primary_metric": primary_metric,
                "primary_value": primary_value,
                "delta_vs_baseline": delta_primary,
                "metrics": metrics,
                "feasible": constraint_check["feasible"],
                "violations": constraint_check["violations"],
                "is_baseline": model_id == baseline_model_id,
            }
        )

    candidates = [row for row in comparison if (not row["is_baseline"]) and (row["primary_value"] is not None)]
    feasible = [row for row in candidates if row["feasible"]]
    if feasible:
        selected = max(feasible, key=lambda row: float(row["primary_value"]))
        selection = {
            "status": "selected_feasible",
            "selected_model_id": selected["model_id"],
            "selected_primary_value": selected["primary_value"],
            "reason": "highest_primary_metric_among_feasible_candidates",
        }
    elif candidates:
        selected = max(candidates, key=lambda row: float(row["primary_value"]))
        selection = {
            "status": "selected_infeasible_fallback",
            "selected_model_id": selected["model_id"],
            "selected_primary_value": selected["primary_value"],
            "reason": "no_feasible_candidate_available",
        }
    else:
        selection = {
            "status": "no_candidates",
            "selected_model_id": "",
            "selected_primary_value": None,
            "reason": "no_candidate_metrics_available",
        }

    run_dir = persist_root / f"{run_prefix}_{run_name}"
    outputs_dir = run_dir / "outputs"
    summary_path = outputs_dir / "smart_comparative_summary.json"
    report_path = outputs_dir / "smart_comparative_report.json"

    summary: Dict[str, Any] = {
        "status": selection["status"],
        "run_tag": run_tag,
        "run_name": run_name,
        "run_prefix": run_prefix,
        "persist_root": str(persist_root),
        "repo_root": str(repo_root),
        "eval_models_json": str(eval_models_json),
        "baseline_model_id": baseline_model_id,
        "primary_metric": primary_metric,
        "num_models": len(models),
        "num_candidates": len(candidates),
        "num_feasible_candidates": len(feasible),
        "created_utc": _utc_now_iso(),
        "config_hash": _config_hash({k: v for k, v in kwargs.items() if isinstance(k, str)}),
    }

    artifacts: Dict[str, str] = {"run_dir": str(run_dir)}
    try:
        _write_json(summary_path, summary)
        _write_json(
            report_path,
            {
                "run_tag": run_tag,
                "summary": summary,
                "selection": selection,
                "comparison": comparison,
            },
        )
        artifacts.update({"summary_json": str(summary_path), "report_json": str(report_path)})
    except Exception as exc:
        summary["artifact_error"] = f"{type(exc).__name__}: {exc}"

    return SmartComparativeBundle(
        summary=summary,
        comparison=comparison,
        selection=selection,
        artifacts=artifacts,
    )
