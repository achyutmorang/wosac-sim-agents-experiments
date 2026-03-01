from __future__ import annotations

from src.experiments import list_experiment_packs, validate_registry


def test_pack_registry_has_wosac_baseline() -> None:
    slugs = {pack.slug for pack in list_experiment_packs()}
    assert "wosac-baseline" in slugs


def test_registry_paths_resolve_for_repo_root() -> None:
    status = validate_registry(".")
    assert "wosac-baseline" in status
    assert status["wosac-baseline"]["missing"] == []
