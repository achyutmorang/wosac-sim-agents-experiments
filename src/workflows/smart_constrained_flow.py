from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .smart_baseline_flow import run_smart_baseline_flow


_METRIC_ALIASES: Dict[str, tuple[str, ...]] = {
    "realism_meta_metric": ("realism_meta_metric", "realism", "meta_metric"),
    "simulated_collision_rate": ("simulated_collision_rate", "collision_rate", "collision"),
    "simulated_offroad_rate": ("simulated_offroad_rate", "offroad_rate", "offroad"),
    "simulated_traffic_light_violation_rate": (
        "simulated_traffic_light_violation_rate",
        "traffic_light_violation_rate",
        "tl_violation_rate",
    ),
    "diversity_score": (
        "diversity_score",
        "rollout_diversity",
        "trajectory_diversity",
        "sample_diversity",
        "diversity",
    ),
}


@dataclass
class SmartConstrainedBundle:
    summary: Dict[str, Any]
    baseline: Dict[str, Any]
    constraints: Dict[str, Optional[float]]
    variants: List[Dict[str, Any]]
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


def _config_hash(config: Mapping[str, Any]) -> str:
    serializable = _to_serializable(dict(config))
    wire = json.dumps(serializable, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(wire.encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = _to_serializable(dict(payload))
    path.write_text(json.dumps(serializable, indent=2, sort_keys=True, ensure_ascii=True) + "\n")


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
            exact = alias.lower()
            if exact in flat_map:
                metrics[target_key] = flat_map[exact]
                break
            suffix = f".{exact}"
            matched = next((k for k in flat_map if k.endswith(suffix)), None)
            if matched is not None:
                metrics[target_key] = flat_map[matched]
                break
    return metrics


def _extract_metrics_from_json(path: Path) -> Dict[str, Optional[float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, Mapping) and isinstance(payload.get("metrics"), Mapping):
        target = payload["metrics"]
    elif isinstance(payload, Mapping):
        target = payload
    else:
        return {k: None for k in _METRIC_ALIASES}
    flat = _flatten_numeric_map(target)
    return _extract_metric_values(flat)


def _parse_numeric_list(value: Any, default: Sequence[float]) -> List[float]:
    if value is None:
        return [float(v) for v in default]
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",") if p.strip()]
        out = [_safe_float(p) for p in parts]
        return [float(v) for v in out if v is not None] or [float(v) for v in default]
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        out = [_safe_float(v) for v in value]
        return [float(v) for v in out if v is not None] or [float(v) for v in default]
    one = _safe_float(value)
    if one is None:
        return [float(v) for v in default]
    return [float(one)]


def _parse_int_list(value: Any, default: Sequence[int]) -> List[int]:
    if value is None:
        return [int(v) for v in default]
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",") if p.strip()]
        out: List[int] = []
        for part in parts:
            num = _safe_float(part)
            if num is None:
                continue
            out.append(int(num))
        return out or [int(v) for v in default]
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        out: List[int] = []
        for elem in value:
            num = _safe_float(elem)
            if num is None:
                continue
            out.append(int(num))
        return out or [int(v) for v in default]
    one = _safe_float(value)
    if one is None:
        return [int(v) for v in default]
    return [int(one)]


def _format_variant_id(temperature: float, top_k: int, constraint_weight: float) -> str:
    t = str(temperature).replace(".", "p")
    c = str(constraint_weight).replace(".", "p")
    return f"t{t}_k{int(top_k)}_cw{c}"


def _inject_env(cmd: str, env: Mapping[str, Any], needle: str) -> str:
    env_prefix = " ".join([f"{k}={v}" for k, v in env.items()])
    if needle in cmd:
        return cmd.replace(needle, f"{env_prefix} {needle}", 1)
    return f"{env_prefix} {cmd}"


def _check_constraints(
    *,
    metrics: Mapping[str, Optional[float]],
    constraints: Mapping[str, Optional[float]],
) -> Dict[str, Any]:
    violations: List[str] = []

    def _check_upper(metric_key: str, limit_key: str) -> None:
        limit = constraints.get(limit_key)
        value = metrics.get(metric_key)
        if limit is None:
            return
        if value is None:
            violations.append(f"missing:{metric_key}")
            return
        if float(value) > float(limit):
            violations.append(f"{metric_key}>{limit_key}")

    _check_upper("simulated_collision_rate", "max_collision_rate")
    _check_upper("simulated_offroad_rate", "max_offroad_rate")
    _check_upper("simulated_traffic_light_violation_rate", "max_tl_violation_rate")

    min_diversity = constraints.get("min_diversity_score")
    diversity = metrics.get("diversity_score")
    if min_diversity is not None:
        if diversity is None:
            violations.append("missing:diversity_score")
        elif float(diversity) < float(min_diversity):
            violations.append("diversity_score<min_diversity_score")

    return {"feasible": len(violations) == 0, "violations": violations}


def _select_best_variant(
    variants: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    with_metrics = [v for v in variants if _safe_float(v.get("metrics", {}).get("realism_meta_metric")) is not None]
    feasible = [v for v in with_metrics if bool(v.get("constraint_check", {}).get("feasible"))]

    def _score(entry: Mapping[str, Any]) -> float:
        metrics = entry.get("metrics", {})
        value = _safe_float(metrics.get("realism_meta_metric"))
        return float(value) if value is not None else float("-inf")

    if feasible:
        best = max(feasible, key=_score)
        return {
            "status": "selected_feasible",
            "selected_variant_id": best.get("variant_id"),
            "selected_realism_meta_metric": _safe_float(best.get("metrics", {}).get("realism_meta_metric")),
            "reason": "highest_realism_among_feasible_variants",
            "num_feasible": len(feasible),
            "num_with_metrics": len(with_metrics),
        }
    if with_metrics:
        best = max(with_metrics, key=_score)
        return {
            "status": "selected_infeasible_fallback",
            "selected_variant_id": best.get("variant_id"),
            "selected_realism_meta_metric": _safe_float(best.get("metrics", {}).get("realism_meta_metric")),
            "reason": "no_feasible_variant_available",
            "num_feasible": 0,
            "num_with_metrics": len(with_metrics),
        }
    return {
        "status": "no_variant_metrics",
        "selected_variant_id": "",
        "selected_realism_meta_metric": None,
        "reason": "variant_metrics_not_found",
        "num_feasible": 0,
        "num_with_metrics": 0,
    }


def _resolve_path(repo_root: Path, value: str) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def run_smart_constrained_flow(**kwargs: Any) -> SmartConstrainedBundle:
    repo_root = Path(str(kwargs.get("repo_root", "."))).resolve()
    run_tag = str(kwargs.get("run_tag", _utc_now_iso().replace("-", "").replace(":", "")))
    run_name = str(kwargs.get("run_name", "dev"))
    run_prefix = str(kwargs.get("run_prefix", "smart_constrained"))
    persist_root = Path(str(kwargs.get("persist_root", "/content/drive/MyDrive/wosac_experiments"))).expanduser()

    baseline_kwargs = dict(kwargs)
    baseline_kwargs.setdefault("sync_smart_repo", False)
    baseline_bundle = run_smart_baseline_flow(**baseline_kwargs)

    temperatures = _parse_numeric_list(kwargs.get("temperatures"), default=[0.8, 1.0, 1.2])
    top_ks = _parse_int_list(kwargs.get("top_ks"), default=[8, 16])
    constraint_weights = _parse_numeric_list(kwargs.get("constraint_weights"), default=[0.05, 0.1, 0.2])
    limit_collision = _safe_float(kwargs.get("max_collision_rate", 0.06))
    limit_offroad = _safe_float(kwargs.get("max_offroad_rate", 0.03))
    limit_tl = _safe_float(kwargs.get("max_tl_violation_rate", 0.02))
    min_diversity = _safe_float(kwargs.get("min_diversity_score"))

    constraints: Dict[str, Optional[float]] = {
        "max_collision_rate": limit_collision,
        "max_offroad_rate": limit_offroad,
        "max_tl_violation_rate": limit_tl,
        "min_diversity_score": min_diversity,
    }

    baseline_metrics_path_arg = str(
        kwargs.get("baseline_metrics_json", "experiments/smart-baseline/results/smart_baseline_v0_metrics.json")
    ).strip()
    baseline_metrics_path = _resolve_path(repo_root, baseline_metrics_path_arg)
    baseline_metrics = {k: None for k in _METRIC_ALIASES}
    baseline_metrics_source = "none"
    if baseline_metrics_path.exists():
        try:
            baseline_metrics = _extract_metrics_from_json(baseline_metrics_path)
            baseline_metrics_source = "json"
        except Exception:
            baseline_metrics_source = "parse_error"

    variants: List[Dict[str, Any]] = []
    variant_metrics_dir_arg = str(kwargs.get("variant_metrics_dir", "")).strip()
    variant_metrics_dir = _resolve_path(repo_root, variant_metrics_dir_arg) if variant_metrics_dir_arg else None

    for temp in temperatures:
        for top_k in top_ks:
            for weight in constraint_weights:
                variant_id = _format_variant_id(temperature=temp, top_k=top_k, constraint_weight=weight)
                env_map = {
                    "SMART_TEMP": temp,
                    "SMART_TOP_K": int(top_k),
                    "SMART_CONSTRAINT_WEIGHT": weight,
                }
                train_cmd = _inject_env(baseline_bundle.command_plan["train_cmd"], env=env_map, needle="python train.py")
                validate_cmd = _inject_env(
                    baseline_bundle.command_plan["validate_cmd"],
                    env=env_map,
                    needle="python val.py",
                )
                variant_entry: Dict[str, Any] = {
                    "variant_id": variant_id,
                    "temperature": float(temp),
                    "top_k": int(top_k),
                    "constraint_weight": float(weight),
                    "setup_cmd": baseline_bundle.command_plan["setup_cmd"],
                    "preprocess_train_cmd": baseline_bundle.command_plan["preprocess_train_cmd"],
                    "preprocess_val_cmd": baseline_bundle.command_plan["preprocess_val_cmd"],
                    "train_cmd": train_cmd,
                    "validate_cmd": validate_cmd,
                    "metrics": {k: None for k in _METRIC_ALIASES},
                    "constraint_check": {"feasible": False, "violations": ["metrics_missing"]},
                }
                if variant_metrics_dir is not None:
                    metrics_path = variant_metrics_dir / f"{variant_id}.json"
                    variant_entry["metrics_json"] = str(metrics_path)
                    if metrics_path.exists():
                        try:
                            metrics = _extract_metrics_from_json(metrics_path)
                            variant_entry["metrics"] = metrics
                            variant_entry["constraint_check"] = _check_constraints(
                                metrics=metrics,
                                constraints=constraints,
                            )
                        except Exception as exc:
                            variant_entry["constraint_check"] = {
                                "feasible": False,
                                "violations": [f"metrics_parse_error:{type(exc).__name__}"],
                            }
                variants.append(variant_entry)

    selection = _select_best_variant(variants)

    if selection.get("selected_variant_id"):
        selected = next((v for v in variants if v.get("variant_id") == selection["selected_variant_id"]), None)
    else:
        selected = None
    selection["selected_variant"] = selected

    run_dir = persist_root / f"{run_prefix}_{run_name}"
    outputs_dir = run_dir / "outputs"
    summary_path = outputs_dir / "smart_constrained_flow_summary.json"
    variants_path = outputs_dir / "smart_constrained_variant_grid.json"
    selection_path = outputs_dir / "smart_constrained_selection.json"

    summary = {
        "status": selection["status"],
        "run_tag": run_tag,
        "run_name": run_name,
        "run_prefix": run_prefix,
        "persist_root": str(persist_root),
        "repo_root": str(repo_root),
        "created_utc": _utc_now_iso(),
        "config_hash": _config_hash({k: v for k, v in kwargs.items() if isinstance(k, str)}),
        "baseline_metrics_source": baseline_metrics_source,
        "baseline_metrics_json": str(baseline_metrics_path),
        "variant_metrics_dir": str(variant_metrics_dir) if variant_metrics_dir is not None else "",
        "num_variants": len(variants),
        "selection_status": selection["status"],
    }

    baseline_payload: Dict[str, Any] = {
        "summary": baseline_bundle.summary,
        "metrics": baseline_bundle.metrics,
        "command_plan": baseline_bundle.command_plan,
        "artifacts": baseline_bundle.artifacts,
        "baseline_metrics_reference": baseline_metrics,
        "baseline_metrics_reference_source": baseline_metrics_source,
    }

    artifacts: Dict[str, str] = {"run_dir": str(run_dir)}
    try:
        _write_json(summary_path, summary)
        _write_json(variants_path, {"run_tag": run_tag, "constraints": constraints, "variants": variants})
        _write_json(selection_path, selection)
        artifacts.update(
            {
                "summary_json": str(summary_path),
                "variant_grid_json": str(variants_path),
                "selection_json": str(selection_path),
            }
        )
    except Exception as exc:
        summary["artifact_error"] = f"{type(exc).__name__}: {exc}"

    return SmartConstrainedBundle(
        summary=summary,
        baseline=baseline_payload,
        constraints=constraints,
        variants=variants,
        selection=selection,
        artifacts=artifacts,
    )
