from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from .spec import ExperimentPack, normalize_slug


_PACKS: Sequence[ExperimentPack] = (
    ExperimentPack(
        slug="wosac-baseline",
        title="WOSAC Baseline",
        objective="Establish reproducible Sim Agents baseline metrics before novelty attempts.",
        notebooks=(
            "experiments/wosac-baseline/notebooks/wosac_baseline_colab.ipynb",
        ),
        workflows=(
            "src/workflows/wosac_baseline_flow.py",
        ),
        modules=(
            "src/workflows/wosac_baseline_flow.py",
            "src/workflows/experiment_flow.py",
            "src/workflows/notebook_contract.py",
            "src/platform",
        ),
        config_paths=(
            "configs/experiments/wosac-baseline.json",
        ),
        tags=("wosac", "sim-agents", "baseline", "colab", "waymax", "womd"),
    ),
    ExperimentPack(
        slug="smart-baseline",
        title="SMART Baseline Wrapper",
        objective="Reproduce SMART baseline with thin wrapper flow and WOSAC-aligned reporting.",
        notebooks=(
            "experiments/smart-baseline/notebooks/smart-baseline_colab.ipynb",
        ),
        workflows=(
            "src/workflows/smart_baseline_flow.py",
        ),
        modules=(
            "src/workflows/smart_baseline_flow.py",
            "src/workflows/experiment_flow.py",
            "src/platform",
        ),
        config_paths=(
            "configs/experiments/smart-baseline.json",
        ),
        tags=("wosac", "sim-agents", "smart", "baseline", "colab", "external-wrapper"),
    ),
    ExperimentPack(
        slug="smart-constrained",
        title="SMART Constrained Probabilistic Variant",
        objective="Optimize SMART-style variant under explicit safety/diversity constraints and compare with baseline.",
        notebooks=(
            "experiments/smart-constrained/notebooks/smart-constrained_colab.ipynb",
        ),
        workflows=(
            "src/workflows/smart_constrained_flow.py",
        ),
        modules=(
            "src/workflows/smart_constrained_flow.py",
            "src/workflows/smart_baseline_flow.py",
            "src/workflows/experiment_flow.py",
            "src/platform",
        ),
        config_paths=(
            "configs/experiments/smart-constrained.json",
        ),
        tags=("wosac", "sim-agents", "smart", "constrained", "probabilistic", "colab"),
    ),
)


def list_experiment_packs() -> List[ExperimentPack]:
    return list(_PACKS)


def get_experiment_pack(slug: str) -> ExperimentPack:
    key = normalize_slug(slug)
    for pack in _PACKS:
        if pack.slug == key:
            return pack
    raise KeyError(f"Unknown experiment pack: {slug!r}")


def find_experiment_packs(query: str, tags: Optional[Iterable[str]] = None) -> List[ExperimentPack]:
    q = str(query).strip().lower()
    wanted_tags = {normalize_slug(t) for t in list(tags or ()) if str(t).strip()}
    out: List[ExperimentPack] = []
    for pack in _PACKS:
        hay = " ".join([pack.slug, pack.title, pack.objective, " ".join(pack.tags)]).lower()
        if q and (q not in hay):
            continue
        if wanted_tags and not wanted_tags.issubset(set(pack.tags)):
            continue
        out.append(pack)
    return out


def validate_pack_paths(repo_root: str | Path, pack: ExperimentPack) -> Dict[str, List[str]]:
    root = Path(repo_root).expanduser().resolve()
    required: List[str] = []
    required.extend(pack.notebooks)
    required.extend(pack.workflows)
    required.extend(pack.modules)
    required.extend(pack.config_paths)

    existing: List[str] = []
    missing: List[str] = []
    for rel in required:
        p = root / rel
        if p.exists():
            existing.append(rel)
        else:
            missing.append(rel)
    return {"existing": existing, "missing": missing}


def validate_registry(repo_root: str | Path) -> Dict[str, Dict[str, List[str]]]:
    out: Dict[str, Dict[str, List[str]]] = {}
    for pack in _PACKS:
        out[pack.slug] = validate_pack_paths(repo_root, pack)
    return out
