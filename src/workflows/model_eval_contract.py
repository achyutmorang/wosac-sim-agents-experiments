from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple


DEFAULT_BINDING_KEYS: Tuple[str, ...] = (
    "manifest_sha256",
    "model_id",
    "scenario_set_hash",
    "evaluator_id",
    "metrics_config_hash",
    "n_rollouts",
    "num_history_seconds",
    "num_future_seconds",
    "seed",
)

DEFAULT_COMPATIBILITY_KEYS: Tuple[str, ...] = (
    "scenario_set_hash",
    "evaluator_id",
    "metrics_config_hash",
    "n_rollouts",
    "num_history_seconds",
    "num_future_seconds",
)

REQUIRED_MANIFEST_KEYS: Tuple[str, ...] = (
    "manifest_version",
    "created_utc",
    "run_tag",
    "model_id",
    "checkpoint_path",
    "checkpoint_sha256",
    "scenario_set_id",
    "scenario_set_hash",
    "evaluator_id",
    "metrics_config_hash",
    "n_rollouts",
    "num_history_seconds",
    "num_future_seconds",
    "seed",
    "manifest_sha256",
)


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


def _json_wire(payload: Mapping[str, Any]) -> bytes:
    serializable = _to_serializable(dict(payload))
    return json.dumps(
        serializable,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_json(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_json_wire(payload)).hexdigest()


def sha256_file(path: str | Path) -> str:
    p = Path(str(path)).expanduser()
    if not p.exists() or (not p.is_file()):
        return ""
    digest = hashlib.sha256()
    with p.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_keys(keys: Sequence[str] | None, fallback: Sequence[str]) -> List[str]:
    if keys is None:
        keys = fallback
    out: List[str] = []
    for key in keys:
        text = str(key).strip()
        if text and text not in out:
            out.append(text)
    return out if out else list(fallback)


def normalize_simulation_manifest(payload: Mapping[str, Any]) -> Dict[str, Any]:
    manifest = dict(_to_serializable(dict(payload)))
    manifest["manifest_version"] = str(manifest.get("manifest_version", "sim_eval_contract/v1"))
    if not str(manifest.get("checkpoint_sha256", "")).strip():
        manifest["checkpoint_sha256"] = sha256_file(str(manifest.get("checkpoint_path", "")))

    # Hash the canonical payload without the hash field itself.
    no_hash_payload = {k: v for k, v in manifest.items() if str(k) != "manifest_sha256"}
    manifest["manifest_sha256"] = sha256_json(no_hash_payload)
    return manifest


def write_simulation_manifest(path: str | Path, payload: Mapping[str, Any]) -> Dict[str, Any]:
    p = Path(str(path)).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    manifest = normalize_simulation_manifest(payload)
    p.write_text(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")
    return manifest


def load_json_mapping(path: str | Path) -> Dict[str, Any]:
    p = Path(str(path)).expanduser()
    payload = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        return {}
    return dict(payload)


def load_simulation_manifest(path: str | Path) -> Tuple[Dict[str, Any], List[str]]:
    p = Path(str(path)).expanduser()
    if not p.exists():
        return {}, ["manifest_missing"]
    try:
        manifest = load_json_mapping(p)
    except Exception as exc:
        return {}, [f"manifest_parse_error:{type(exc).__name__}"]
    if not manifest:
        return {}, ["manifest_non_mapping"]

    errors: List[str] = []
    for key in REQUIRED_MANIFEST_KEYS:
        if str(manifest.get(key, "")).strip() == "":
            errors.append(f"manifest_missing_field:{key}")

    provided_hash = str(manifest.get("manifest_sha256", "")).strip()
    expected_hash = sha256_json({k: v for k, v in manifest.items() if str(k) != "manifest_sha256"})
    if provided_hash and (provided_hash != expected_hash):
        errors.append("manifest_sha256_mismatch")

    return manifest, errors


def contract_signature(manifest: Mapping[str, Any], keys: Sequence[str] | None = None) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not isinstance(manifest, Mapping):
        return out
    for key in _normalize_keys(keys, DEFAULT_COMPATIBILITY_KEYS):
        out[key] = _to_serializable(manifest.get(key))
    return out


def compare_contract_signatures(
    baseline_signature: Mapping[str, Any],
    candidate_signature: Mapping[str, Any],
    *,
    keys: Sequence[str] | None = None,
) -> List[str]:
    mismatches: List[str] = []
    for key in _normalize_keys(keys, DEFAULT_COMPATIBILITY_KEYS):
        left = baseline_signature.get(key)
        right = candidate_signature.get(key)
        if left != right:
            mismatches.append(f"{key}_mismatch")
    return mismatches


def validate_metrics_binding(
    metrics_payload: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    required_keys: Sequence[str] | None = None,
) -> List[str]:
    errors: List[str] = []
    if not isinstance(metrics_payload, Mapping):
        return ["metrics_payload_non_mapping"]
    if not isinstance(manifest, Mapping) or (not manifest):
        return ["manifest_missing_for_binding"]

    for key in _normalize_keys(required_keys, DEFAULT_BINDING_KEYS):
        if key == "manifest_sha256":
            expected = manifest.get("manifest_sha256")
        else:
            expected = manifest.get(key)
        actual = metrics_payload.get(key)
        if actual is None:
            errors.append(f"metrics_binding_missing:{key}")
            continue
        if _to_serializable(actual) != _to_serializable(expected):
            errors.append(f"metrics_binding_mismatch:{key}")
    return errors
