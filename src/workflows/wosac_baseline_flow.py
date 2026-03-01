from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


_METRIC_ALIASES: Dict[str, tuple[str, ...]] = {
    "realism_meta_metric": (
        "realism_meta_metric",
        "realism",
        "meta_metric",
    ),
    "simulated_collision_rate": (
        "simulated_collision_rate",
        "collision_rate",
        "collision",
    ),
    "simulated_offroad_rate": (
        "simulated_offroad_rate",
        "offroad_rate",
        "offroad",
    ),
    "simulated_traffic_light_violation_rate": (
        "simulated_traffic_light_violation_rate",
        "traffic_light_violation_rate",
        "tl_violation_rate",
    ),
}


@dataclass
class WOSACBaselineFlowBundle:
    summary: Dict[str, Any]
    metrics: Dict[str, Optional[float]]
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
            exact_key = alias.lower()
            if exact_key in flat_map:
                metrics[target_key] = flat_map[exact_key]
                break
            suffix = f".{exact_key}"
            matched_key = next((k for k in flat_map if k.endswith(suffix)), None)
            if matched_key is not None:
                metrics[target_key] = flat_map[matched_key]
                break
    return metrics


def _parse_metrics_json(path: Path) -> tuple[Dict[str, Optional[float]], str]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, Mapping):
        return ({k: None for k in _METRIC_ALIASES}, "json_non_mapping")
    flat_map = _flatten_numeric_map(payload)
    metrics = _extract_metric_values(flat_map)
    return metrics, "json"


def _parse_metrics_csv(path: Path) -> tuple[Dict[str, Optional[float]], str]:
    flat_map: Dict[str, float] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            metric_name = str(row.get("metric", "")).strip().lower()
            value = _safe_float(row.get("value"))
            if metric_name and value is not None:
                flat_map[metric_name] = value
    if not flat_map:
        return ({k: None for k in _METRIC_ALIASES}, "csv_empty")
    metrics = _extract_metric_values(flat_map)
    return metrics, "csv"


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


def _config_hash(config: Mapping[str, Any]) -> str:
    serializable = _to_serializable(dict(config))
    wire = json.dumps(serializable, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(wire.encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = _to_serializable(dict(payload))
    path.write_text(json.dumps(serializable, indent=2, sort_keys=True, ensure_ascii=True) + "\n")


def run_wosac_baseline_flow(**kwargs: Any) -> WOSACBaselineFlowBundle:
    """Baseline workflow scaffold with evaluator-metric ingestion and artifact writing.

    This function is intentionally lightweight and Colab-friendly. It can ingest metrics from:
    - `official_metrics_json`: a JSON file produced by an evaluator
    - `metrics_csv`: a CSV file with columns `metric,value`
    """

    repo_root = Path(str(kwargs.get("repo_root", "."))).resolve()
    run_tag = str(kwargs.get("run_tag", _utc_now_iso().replace("-", "").replace(":", "")))
    run_name = str(kwargs.get("run_name", "dev"))
    run_prefix = str(kwargs.get("run_prefix", "wosac_baseline"))
    persist_root = Path(str(kwargs.get("persist_root", "/content/drive/MyDrive/wosac_experiments"))).expanduser()
    n_rollouts_per_scenario = int(kwargs.get("n_rollouts_per_scenario", 32))

    metrics: Dict[str, Optional[float]] = {k: None for k in _METRIC_ALIASES}
    metrics_source = "none"
    metric_ingest_error = ""

    metrics_json_arg = str(kwargs.get("official_metrics_json", "")).strip()
    metrics_csv_arg = str(kwargs.get("metrics_csv", "")).strip()
    if metrics_json_arg:
        metrics_path = Path(metrics_json_arg)
        try:
            if metrics_path.exists():
                metrics, metrics_source = _parse_metrics_json(metrics_path)
            else:
                metric_ingest_error = f"missing_json:{metrics_path}"
        except Exception as exc:
            metric_ingest_error = f"json_parse_error:{type(exc).__name__}"
    elif metrics_csv_arg:
        metrics_path = Path(metrics_csv_arg)
        try:
            if metrics_path.exists():
                metrics, metrics_source = _parse_metrics_csv(metrics_path)
            else:
                metric_ingest_error = f"missing_csv:{metrics_path}"
        except Exception as exc:
            metric_ingest_error = f"csv_parse_error:{type(exc).__name__}"

    has_primary_metric = metrics.get("realism_meta_metric") is not None
    status = "metrics_loaded" if has_primary_metric else "dry_run"
    message = (
        "Loaded evaluator metrics and wrote baseline artifacts."
        if has_primary_metric
        else "No evaluator metrics found yet. Generated dry-run artifacts and metadata."
    )

    run_dir = persist_root / f"{run_prefix}_{run_name}"
    outputs_dir = run_dir / "outputs"
    summary_path = outputs_dir / "wosac_baseline_flow_summary.json"
    metrics_path = outputs_dir / "wosac_baseline_metrics_snapshot.json"
    artifacts: Dict[str, str] = {}
    artifact_error = ""

    serializable_kwargs = _to_serializable(dict(kwargs))

    summary: Dict[str, Any] = {
        "status": status,
        "message": message,
        "run_tag": run_tag,
        "run_name": run_name,
        "run_prefix": run_prefix,
        "persist_root": str(persist_root),
        "n_rollouts_per_scenario": n_rollouts_per_scenario,
        "metrics_source": metrics_source,
        "metric_ingest_error": metric_ingest_error,
        "repo_root": str(repo_root),
        "repo_commit": _safe_git_commit(repo_root),
        "created_utc": _utc_now_iso(),
        "config_hash": _config_hash({k: v for k, v in kwargs.items() if isinstance(k, str)}),
        "kwargs": serializable_kwargs,
    }

    try:
        _write_json(summary_path, summary)
        _write_json(
            metrics_path,
            {
                "run_tag": run_tag,
                "created_utc": summary["created_utc"],
                "metrics_source": metrics_source,
                "metrics": metrics,
            },
        )
        artifacts = {
            "run_dir": str(run_dir),
            "summary_json": str(summary_path),
            "metrics_json": str(metrics_path),
        }
    except Exception as exc:
        artifact_error = f"{type(exc).__name__}: {exc}"
        artifacts = {"run_dir": str(run_dir)}

    if artifact_error:
        summary["artifact_error"] = artifact_error

    return WOSACBaselineFlowBundle(summary=summary, metrics=metrics, artifacts=artifacts)
