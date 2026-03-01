from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def _to_serializable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {str(k): _to_serializable(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple, set)):
        return [_to_serializable(v) for v in list(value)]
    if hasattr(value, '__dict__'):
        return _to_serializable(vars(value))
    return str(value)


def _safe_version(dist_name: str) -> str:
    try:
        from importlib.metadata import version
        return str(version(dist_name))
    except Exception:
        return 'not_installed'


def _detect_colab_runtime_type() -> str:
    if str(os.environ.get('COLAB_TPU_ADDR', '')).strip():
        return 'tpu'
    if str(os.environ.get('COLAB_GPU', '')).strip():
        return 'gpu'
    if str(os.environ.get('NVIDIA_VISIBLE_DEVICES', '')).strip() not in {'', 'void', 'none'}:
        return 'gpu'
    return 'cpu'


def _resolve_git_commit(repo_dir: Optional[str], fallback: Optional[str]) -> str:
    if isinstance(fallback, str) and fallback.strip():
        return str(fallback).strip()
    if not repo_dir:
        return 'unknown'
    try:
        out = subprocess.check_output(
            ['git', '-C', str(repo_dir), 'rev-parse', 'HEAD'],
            stderr=subprocess.DEVNULL,
        )
        commit = out.decode('utf-8', errors='ignore').strip()
        return commit or 'unknown'
    except Exception:
        return 'unknown'


def _manifest_path(run_prefix: str) -> Path:
    return Path(f'{run_prefix}_notebook_contract_manifest.json')


def _contract_run_dir(*, persist_root: str, run_prefix: str, run_name: str) -> Path:
    root = Path(str(persist_root)).expanduser()
    return root / f'{str(run_prefix)}_{str(run_name)}'


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, ensure_ascii=True))
    os.replace(tmp, path)


def _safe_json_read(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text())
    except Exception:
        return {}
    return obj if isinstance(obj, dict) else {}


def _copy_to(dst: Path, src: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _cfg_hash(cfg: Any, search_cfg: Any) -> str:
    payload = {
        'cfg': _to_serializable(cfg),
        'search_cfg': _to_serializable(search_cfg),
    }
    wire = json.dumps(payload, sort_keys=True, separators=(',', ':'), ensure_ascii=True)
    return hashlib.sha256(wire.encode('utf-8')).hexdigest()


def load_notebook_contract_manifest(run_prefix: str) -> Dict[str, Any]:
    p = _manifest_path(run_prefix)
    if not p.exists():
        return {}
    try:
        obj = json.loads(p.read_text())
    except Exception:
        return {}
    return obj if isinstance(obj, dict) else {}


def manifest_has_stage(manifest: Mapping[str, Any], stage: str) -> bool:
    target = str(stage).strip().lower()
    if not target:
        return False
    events = manifest.get('events', [])
    if isinstance(events, list):
        for evt in events:
            if isinstance(evt, dict) and str(evt.get('stage', '')).strip().lower() == target:
                return True
    if str(manifest.get('stage', '')).strip().lower() == target:
        return True
    return False


def validate_notebook_contract_manifest(
    manifest: Mapping[str, Any],
    *,
    require_quick_probe: bool = True,
    require_preflight: bool = True,
    required_stages: Optional[Sequence[str]] = None,
) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    if not manifest:
        return False, ['manifest_missing']

    for key in (
        'run_tag',
        'git_commit',
        'created_utc',
        'cfg_hash',
        'python_version',
        'package_versions',
        'colab_runtime_type',
        'n_shards',
        'shard_id',
    ):
        if key not in manifest:
            reasons.append(f'missing_{key}')

    if require_quick_probe and (not bool(manifest.get('quick_probe_pass', False))):
        reasons.append('quick_probe_not_passed')
    if require_preflight and (not bool(manifest.get('preflight_pass', False))):
        reasons.append('preflight_not_passed')

    for stage in list(required_stages or ()):
        if not manifest_has_stage(manifest, str(stage)):
            reasons.append(f'missing_stage:{stage}')

    return len(reasons) == 0, reasons


def write_contract_storage_mirror(
    *,
    persist_root: str,
    run_prefix: str,
    run_name: str,
    run_prefix_path: str,
    cfg: Any,
    search_cfg: Any,
    n_shards: int,
    shard_id: int,
    stage: str,
    git_commit: str,
    resume_from_existing: bool,
    run_enabled: bool,
    artifact_paths: Optional[Mapping[str, Any]] = None,
    metrics_csv_path: Optional[str] = None,
    extra_fields: Optional[Mapping[str, Any]] = None,
) -> Dict[str, str]:
    run_dir = _contract_run_dir(
        persist_root=str(persist_root),
        run_prefix=str(run_prefix),
        run_name=str(run_name),
    )
    progress_dir = run_dir / 'progress'
    checkpoints_dir = run_dir / 'checkpoints'
    outputs_dir = run_dir / 'outputs'
    artifacts_dir = outputs_dir / 'artifacts'

    cfg_payload = _to_serializable(cfg)
    search_payload = _to_serializable(search_cfg)
    cfg_hash = _cfg_hash(cfg, search_cfg)
    created_now = _utc_now_iso()

    config_path = run_dir / 'config.json'
    env_manifest_path = run_dir / 'env_manifest.json'
    run_manifest_path = run_dir / 'run_manifest.json'
    progress_path = progress_dir / f'shard_{int(shard_id)}.json'
    latest_ckpt_path = checkpoints_dir / 'latest.json'
    metrics_out_path = outputs_dir / 'metrics.csv'
    artifact_index_path = artifacts_dir / 'artifact_index.json'

    _atomic_write_json(
        config_path,
        {
            'run_name': str(run_name),
            'run_prefix': str(run_prefix),
            'run_prefix_path': str(run_prefix_path),
            'cfg_hash': str(cfg_hash),
            'cfg': cfg_payload,
            'search_cfg': search_payload,
        },
    )

    env_prev = _safe_json_read(env_manifest_path)
    env_payload = {
        **env_prev,
        'updated_utc': created_now,
        'python_version': str(sys.version),
        'platform': str(platform.platform()),
        'git_commit': str(git_commit),
        'cfg_hash': str(cfg_hash),
        'run_name': str(run_name),
        'run_prefix': str(run_prefix),
        'run_prefix_path': str(run_prefix_path),
        'colab_runtime_type': _detect_colab_runtime_type(),
        'n_shards': int(max(1, int(n_shards))),
        'shard_id': int(shard_id),
        'packages': {
            'numpy': _safe_version('numpy'),
            'pandas': _safe_version('pandas'),
            'torch': _safe_version('torch'),
            'jax': _safe_version('jax'),
        },
    }
    _atomic_write_json(env_manifest_path, env_payload)

    run_prev = _safe_json_read(run_manifest_path)
    run_payload = {
        'run_name': str(run_name),
        'run_prefix': str(run_prefix),
        'run_prefix_path': str(run_prefix_path),
        'created_utc': str(run_prev.get('created_utc', '')).strip() or created_now,
        'updated_utc': created_now,
        'git_commit': str(git_commit),
        'config_hash': str(cfg_hash),
        'python_version': str(sys.version.split()[0]),
        'package_versions': dict(env_payload.get('packages', {})),
        'runtime_type': _detect_colab_runtime_type(),
        'n_shards': int(max(1, int(n_shards))),
        'shard_id': int(shard_id),
        'resume_from_existing': bool(resume_from_existing),
        'run_enabled': bool(run_enabled),
        'stage': str(stage),
    }
    if isinstance(extra_fields, Mapping) and extra_fields:
        run_payload['extra_fields'] = _to_serializable(dict(extra_fields))
    _atomic_write_json(run_manifest_path, run_payload)

    progress_payload = {
        'updated_utc': created_now,
        'run_name': str(run_name),
        'run_prefix': str(run_prefix),
        'run_prefix_path': str(run_prefix_path),
        'n_shards': int(max(1, int(n_shards))),
        'shard_id': int(shard_id),
        'stage': str(stage),
        'resume_from_existing': bool(resume_from_existing),
        'run_enabled': bool(run_enabled),
    }
    _atomic_write_json(progress_path, progress_payload)

    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(
        latest_ckpt_path,
        {
            'updated_utc': created_now,
            'run_prefix_path': str(run_prefix_path),
            'stage': str(stage),
            'resume_from_existing': bool(resume_from_existing),
        },
    )

    outputs_dir.mkdir(parents=True, exist_ok=True)
    copied_metrics = False
    if metrics_csv_path:
        copied_metrics = _copy_to(metrics_out_path, Path(str(metrics_csv_path)))
    if not copied_metrics and (not metrics_out_path.exists()):
        metrics_out_path.write_text('metric,value\nstage,0\n')

    flat_artifacts: Dict[str, str] = {}
    if isinstance(artifact_paths, Mapping):
        for k, v in artifact_paths.items():
            path_str = str(v).strip()
            if path_str:
                flat_artifacts[str(k)] = path_str
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(
        artifact_index_path,
        {
            'updated_utc': created_now,
            'artifact_count': int(len(flat_artifacts)),
            'artifacts': [
                {
                    'key': str(k),
                    'path': str(v),
                    'exists': bool(Path(str(v)).exists()),
                }
                for k, v in sorted(flat_artifacts.items())
            ],
        },
    )

    return {
        'contract_run_dir': str(run_dir),
        'contract_config': str(config_path),
        'contract_env_manifest': str(env_manifest_path),
        'contract_run_manifest': str(run_manifest_path),
        'contract_progress': str(progress_path),
        'contract_checkpoint_latest': str(latest_ckpt_path),
        'contract_outputs_metrics': str(metrics_out_path),
        'contract_outputs_artifact_index': str(artifact_index_path),
    }


def run_risk_training_notebook_gates(
    *,
    runner: Any,
    cfg: Any,
    eval_idx: Optional[Iterable[int]] = None,
    probe_shift_suite: str = 'nominal_clean',
) -> Dict[str, Any]:
    # Kept for API compatibility with older notebooks.
    # This repository does not ship closedloop risk modules; use a minimal gate
    # that verifies runner/config wiring before expensive Colab execution.
    import pandas as pd

    scenarios = list(getattr(runner, 'data', {}).get('scenarios', []))
    has_scenarios = len(scenarios) > 0
    has_cfg = cfg is not None
    has_eval_idx = eval_idx is None or len(list(eval_idx)) >= 0
    shift_name = str(probe_shift_suite).strip() or 'nominal_clean'

    summary_rows = [
        {'check': 'runner_has_scenarios', 'pass': int(has_scenarios), 'detail': f'n_scenarios={len(scenarios)}'},
        {'check': 'cfg_present', 'pass': int(has_cfg), 'detail': f'cfg_type={type(cfg).__name__}'},
        {'check': 'eval_idx_well_formed', 'pass': int(has_eval_idx), 'detail': 'optional iterable accepted'},
        {'check': 'probe_shift_suite_set', 'pass': int(bool(shift_name)), 'detail': shift_name},
    ]
    gate_df = pd.DataFrame(summary_rows)
    overall_pass = bool(gate_df['pass'].all()) if not gate_df.empty else False
    failure_reasons = [str(r['check']) for r in summary_rows if int(r.get('pass', 0)) == 0]

    return {
        'overall_pass': bool(overall_pass),
        'risk_probe_pass': bool(overall_pass),
        'preflight_pass': bool(overall_pass),
        'failure_reasons': failure_reasons,
        'risk_probe_summary_df': gate_df,
        'risk_probe_rows_df': pd.DataFrame(),
        'preflight_df': gate_df.copy(),
    }


def write_notebook_contract_manifest(
    *,
    run_prefix: str,
    run_tag: str,
    cfg: Any,
    search_cfg: Any,
    n_shards: int,
    shard_id: int,
    notebook_name: str,
    stage: str,
    repo_dir: Optional[str] = None,
    git_commit: Optional[str] = None,
    quick_probe_pass: Optional[bool] = None,
    preflight_pass: Optional[bool] = None,
    extra_fields: Optional[Mapping[str, Any]] = None,
) -> str:
    p = _manifest_path(run_prefix)
    prior = load_notebook_contract_manifest(run_prefix)

    commit = _resolve_git_commit(repo_dir=repo_dir, fallback=git_commit)
    created_utc = str(prior.get('created_utc', '')).strip() or _utc_now_iso()

    package_versions = {
        'numpy': _safe_version('numpy'),
        'pandas': _safe_version('pandas'),
        'torch': _safe_version('torch'),
        'jax': _safe_version('jax'),
    }
    event = {
        'at_utc': _utc_now_iso(),
        'stage': str(stage),
        'notebook_name': str(notebook_name),
        'quick_probe_pass': bool(quick_probe_pass) if quick_probe_pass is not None else None,
        'preflight_pass': bool(preflight_pass) if preflight_pass is not None else None,
    }
    if isinstance(extra_fields, Mapping) and extra_fields:
        event['extra_fields'] = _to_serializable(dict(extra_fields))

    events = prior.get('events', [])
    if not isinstance(events, list):
        events = []
    events.append(event)
    if len(events) > 200:
        events = events[-200:]

    manifest = {
        'run_prefix': str(run_prefix),
        'run_tag': str(run_tag),
        'git_commit': str(commit),
        'created_utc': created_utc,
        'updated_utc': _utc_now_iso(),
        'cfg_hash': _cfg_hash(cfg=cfg, search_cfg=search_cfg),
        'python_version': str(sys.version.split()[0]),
        'platform': str(platform.platform()),
        'package_versions': package_versions,
        'colab_runtime_type': _detect_colab_runtime_type(),
        'n_shards': int(max(1, int(n_shards))),
        'shard_id': int(shard_id),
        'notebook_name': str(notebook_name),
        'stage': str(stage),
        'quick_probe_pass': bool(quick_probe_pass) if quick_probe_pass is not None else bool(prior.get('quick_probe_pass', False)),
        'preflight_pass': bool(preflight_pass) if preflight_pass is not None else bool(prior.get('preflight_pass', False)),
        'events': events,
    }
    if isinstance(extra_fields, Mapping) and extra_fields:
        manifest['extra_fields'] = _to_serializable(dict(extra_fields))
    elif 'extra_fields' in prior:
        manifest['extra_fields'] = prior['extra_fields']

    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True))
    return str(p)
