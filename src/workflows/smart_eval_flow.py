from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .model_eval_contract import (
    DEFAULT_BINDING_KEYS,
    DEFAULT_COMPATIBILITY_KEYS,
    compare_contract_signatures,
    contract_signature,
    load_json_mapping,
    load_simulation_manifest,
    sha256_file,
    validate_metrics_binding,
)


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


def _resolve_config_arg(*, repo_root: Path, smart_repo_dir: Path, value: str) -> str:
    text = str(value).strip()
    if not text:
        return text

    candidate = Path(text).expanduser()
    if candidate.is_absolute():
        return str(candidate)

    repo_candidate = (repo_root / candidate).resolve()
    if repo_candidate.exists():
        return str(repo_candidate)

    smart_candidate = (smart_repo_dir / candidate).resolve()
    if smart_candidate.exists():
        return str(smart_candidate)

    return text


def _paths_match(left: str, right: str) -> bool:
    if (not str(left).strip()) or (not str(right).strip()):
        return False
    try:
        return Path(str(left)).expanduser().resolve() == Path(str(right)).expanduser().resolve()
    except Exception:
        return str(left).strip() == str(right).strip()


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


def _normalize_key_list(value: Any, *, fallback: Sequence[str]) -> List[str]:
    if value is None:
        source = list(fallback)
    elif isinstance(value, str):
        source = [part.strip() for part in value.split(",")]
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        source = [str(v).strip() for v in value]
    else:
        source = list(fallback)

    out: List[str] = []
    for item in source:
        key = str(item).strip()
        if key and key not in out:
            out.append(key)
    return out if out else list(fallback)


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
    smart_val_config = _resolve_config_arg(
        repo_root=repo_root,
        smart_repo_dir=smart_repo_dir,
        value=str(kwargs.get("smart_val_config", "configs/validation/validation_scalable.yaml")),
    )
    sync_smart_repo = bool(kwargs.get("sync_smart_repo", False))
    strict_contract = bool(kwargs.get("strict_contract", False))
    require_metrics_binding = bool(kwargs.get("require_metrics_binding", strict_contract))
    verify_checkpoint_hash = bool(kwargs.get("verify_checkpoint_hash", strict_contract))
    binding_keys = _normalize_key_list(kwargs.get("binding_keys"), fallback=DEFAULT_BINDING_KEYS)
    compatibility_keys = _normalize_key_list(kwargs.get("compatibility_keys"), fallback=DEFAULT_COMPATIBILITY_KEYS)

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
    manifests_dir_arg = str(kwargs.get("manifests_dir", "")).strip()
    manifests_dir = _resolve_path(repo_root, manifests_dir_arg) if manifests_dir_arg else None

    model_rows: List[Dict[str, Any]] = []
    for model in models:
        model_id = str(model.get("model_id", "")).strip()
        if not model_id:
            continue

        manifest_json_arg = str(model.get("manifest_json", "")).strip()
        manifest_path: Optional[Path] = None
        if manifest_json_arg:
            manifest_path = _resolve_path(repo_root, manifest_json_arg)
        elif manifests_dir is not None:
            manifest_path = manifests_dir / f"{model_id}_simulation_manifest.json"

        manifest: Dict[str, Any] = {}
        manifest_source = "none"
        contract_errors: List[str] = []
        if manifest_path is not None:
            if manifest_path.exists():
                manifest, manifest_errors = load_simulation_manifest(manifest_path)
                manifest_source = "json"
                contract_errors.extend(manifest_errors)
            else:
                manifest_source = "missing"
                if strict_contract:
                    contract_errors.append("manifest_missing")

        checkpoint_path = str(model.get("checkpoint_path", "")).strip()
        if (not checkpoint_path) and manifest:
            checkpoint_path = str(manifest.get("checkpoint_path", "")).strip()

        if manifest:
            manifest_model_id = str(manifest.get("model_id", "")).strip()
            if manifest_model_id and (manifest_model_id != model_id):
                contract_errors.append("manifest_model_id_mismatch")
            manifest_ckpt_path = str(manifest.get("checkpoint_path", "")).strip()
            if manifest_ckpt_path and checkpoint_path and (not _paths_match(manifest_ckpt_path, checkpoint_path)):
                contract_errors.append("checkpoint_path_mismatch_manifest")

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
        metrics_payload: Dict[str, Any] = {}
        metrics_json_arg = str(model.get("metrics_json", "")).strip()
        metrics_path: Optional[Path] = None
        if metrics_json_arg:
            metrics_path = _resolve_path(repo_root, metrics_json_arg)
        elif metrics_dir is not None:
            metrics_path = metrics_dir / f"{model_id}.json"
        if metrics_path is not None:
            try:
                if metrics_path.exists():
                    metrics_payload = load_json_mapping(metrics_path)
                    metrics = _extract_metrics_from_json(metrics_path)
                    metrics_source = "json"
                else:
                    metrics_source = "missing"
                    if strict_contract:
                        contract_errors.append("metrics_json_missing")
            except Exception:
                metrics_source = "parse_error"
                if strict_contract:
                    contract_errors.append("metrics_json_parse_error")

        if require_metrics_binding:
            if not manifest:
                contract_errors.append("manifest_missing_for_metrics_binding")
            if metrics_source != "json":
                contract_errors.append("metrics_json_missing_for_binding")
            elif manifest:
                contract_errors.extend(
                    validate_metrics_binding(
                        metrics_payload,
                        manifest,
                        required_keys=binding_keys,
                    )
                )

        if verify_checkpoint_hash and manifest:
            expected_sha = str(manifest.get("checkpoint_sha256", "")).strip()
            if not expected_sha:
                contract_errors.append("manifest_missing_field:checkpoint_sha256")
            elif not checkpoint_path:
                contract_errors.append("checkpoint_missing_for_hash")
            else:
                actual_sha = sha256_file(checkpoint_path)
                if not actual_sha:
                    contract_errors.append("checkpoint_missing_for_hash")
                elif actual_sha != expected_sha:
                    contract_errors.append("checkpoint_sha256_mismatch")

        dedup_errors = list(dict.fromkeys([str(err) for err in contract_errors if str(err).strip()]))
        contract_valid = len(dedup_errors) == 0
        manifest_hash = str(manifest.get("manifest_sha256", "")).strip() if manifest else ""
        signature = contract_signature(manifest, keys=compatibility_keys) if manifest else {}

        model_rows.append(
            {
                "model_id": model_id,
                "checkpoint_path": checkpoint_path,
                "checkpoint_exists": bool(Path(checkpoint_path).expanduser().exists()) if checkpoint_path else False,
                "env": env_data,
                "validate_cmd": validate_cmd,
                "manifest_json": str(manifest_path) if manifest_path is not None else "",
                "manifest_source": manifest_source,
                "manifest_sha256": manifest_hash,
                "manifest": manifest,
                "metrics_json": str(metrics_path) if metrics_path is not None else "",
                "metrics_source": metrics_source,
                "metrics": metrics,
                "contract_valid": contract_valid,
                "contract_errors": dedup_errors,
                "contract_signature": signature,
            }
        )

    num_contract_invalid = sum(1 for row in model_rows if not bool(row.get("contract_valid", False)))
    status = "ready"
    if sync_error:
        status = "sync_failed"
    elif not model_rows:
        status = "no_models"
    elif strict_contract and (num_contract_invalid > 0):
        status = "contract_failed"

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
        "smart_val_config": smart_val_config,
        "smart_repo_sync": sync_result,
        "sync_error": sync_error,
        "metrics_dir": str(metrics_dir) if metrics_dir is not None else "",
        "manifests_dir": str(manifests_dir) if manifests_dir is not None else "",
        "strict_contract": strict_contract,
        "require_metrics_binding": require_metrics_binding,
        "verify_checkpoint_hash": verify_checkpoint_hash,
        "binding_keys": binding_keys,
        "compatibility_keys": compatibility_keys,
        "num_models": len(model_rows),
        "num_contract_invalid": num_contract_invalid,
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
    require_contract_compatibility = bool(kwargs.get("require_contract_compatibility", False))
    compatibility_keys = _normalize_key_list(kwargs.get("compatibility_keys"), fallback=DEFAULT_COMPATIBILITY_KEYS)

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
    baseline_signature: Dict[str, Any] = {}
    if baseline is not None:
        baseline_primary = _safe_float(dict(baseline.get("metrics", {})).get(primary_metric))
        signature_obj = baseline.get("contract_signature", {})
        if isinstance(signature_obj, Mapping):
            baseline_signature = dict(signature_obj)
        elif isinstance(baseline.get("manifest"), Mapping):
            baseline_signature = contract_signature(dict(baseline["manifest"]), keys=compatibility_keys)

    comparison: List[Dict[str, Any]] = []
    for model in models:
        model_id = str(model.get("model_id", ""))
        metrics = dict(model.get("metrics", {}))
        primary_value = _safe_float(metrics.get(primary_metric))
        contract_valid = bool(model.get("contract_valid", True))
        signature_obj = model.get("contract_signature", {})
        model_signature: Dict[str, Any] = {}
        if isinstance(signature_obj, Mapping):
            model_signature = dict(signature_obj)
        elif isinstance(model.get("manifest"), Mapping):
            model_signature = contract_signature(dict(model["manifest"]), keys=compatibility_keys)

        compatibility_violations: List[str] = []
        compatible_with_baseline = True
        if require_contract_compatibility and (model_id != baseline_model_id):
            if not contract_valid:
                compatibility_violations.append("candidate_contract_invalid")
            if not baseline_signature:
                compatibility_violations.append("baseline_signature_missing")
            if not model_signature:
                compatibility_violations.append("candidate_signature_missing")
            if baseline_signature and model_signature:
                compatibility_violations.extend(
                    compare_contract_signatures(
                        baseline_signature,
                        model_signature,
                        keys=compatibility_keys,
                    )
                )
            compatible_with_baseline = len(compatibility_violations) == 0

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
                "contract_valid": contract_valid,
                "contract_signature": model_signature,
                "compatible_with_baseline": compatible_with_baseline,
                "compatibility_violations": compatibility_violations,
            }
        )

    candidates = [row for row in comparison if (not row["is_baseline"]) and (row["primary_value"] is not None)]
    if require_contract_compatibility:
        candidates = [row for row in candidates if row["compatible_with_baseline"]]
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
    elif require_contract_compatibility:
        selection = {
            "status": "no_compatible_candidates",
            "selected_model_id": "",
            "selected_primary_value": None,
            "reason": "no_contract_compatible_candidate_available",
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
        "require_contract_compatibility": require_contract_compatibility,
        "compatibility_keys": compatibility_keys,
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
