from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from src.experiments import (
    ExperimentPack,
    experiment_pack_paths,
    get_experiment_pack,
    list_experiment_packs,
)

try:
    import pandas as _pd
except Exception:  # pragma: no cover
    _pd = None


@dataclass
class ExperimentBootstrapBundle:
    pack: ExperimentPack
    config: Dict[str, Any]
    paths: Dict[str, str]
    summary_table: Any


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def load_experiment_config(
    *,
    slug: str,
    repo_root: str | Path = ".",
    default_on_missing: bool = True,
) -> Dict[str, Any]:
    paths = experiment_pack_paths(repo_root=repo_root, slug=slug)
    payload = _load_json(paths["config_file"])
    if payload:
        return payload
    if not default_on_missing:
        raise FileNotFoundError(f"Config not found for experiment slug={slug!r}: {paths['config_file']}")
    pack = get_experiment_pack(slug)
    return {
        "slug": pack.slug,
        "title": pack.title,
        "objective": pack.objective,
        "run": {
            "run_name": "dev",
            "run_prefix": pack.slug.replace("-", "_"),
            "persist_root": "/content/drive/MyDrive/wosac_experiments",
            "n_shards": 1,
            "shard_id": 0,
            "resume_from_existing": True,
            "run_enabled": True,
        },
    }


def _build_table(rows: list[dict[str, Any]]) -> Any:
    if _pd is None:
        return rows
    return _pd.DataFrame(rows)


def bootstrap_experiment_pack(
    *,
    slug: str,
    repo_root: str | Path = ".",
    overrides: Optional[Mapping[str, Any]] = None,
) -> ExperimentBootstrapBundle:
    pack = get_experiment_pack(slug)
    cfg = load_experiment_config(slug=slug, repo_root=repo_root, default_on_missing=True)
    cfg = dict(cfg)
    if overrides:
        cfg.update(dict(overrides))

    paths = experiment_pack_paths(repo_root=repo_root, slug=slug)
    path_map = {k: str(v) for k, v in paths.items()}
    rows = [
        {"field": "slug", "value": pack.slug},
        {"field": "title", "value": pack.title},
        {"field": "objective", "value": pack.objective},
        {"field": "notebook", "value": ", ".join(pack.notebooks)},
        {"field": "workflow", "value": ", ".join(pack.workflows)},
        {"field": "config_file", "value": path_map["config_file"]},
    ]
    return ExperimentBootstrapBundle(
        pack=pack,
        config=cfg,
        paths=path_map,
        summary_table=_build_table(rows),
    )


def list_experiment_pack_table() -> Any:
    rows = []
    for pack in list_experiment_packs():
        rows.append(
            {
                "slug": pack.slug,
                "title": pack.title,
                "n_notebooks": len(pack.notebooks),
                "n_workflows": len(pack.workflows),
                "n_modules": len(pack.modules),
                "tags": ",".join(pack.tags),
            }
        )
    rows = sorted(rows, key=lambda r: r["slug"])
    return _build_table(rows)
