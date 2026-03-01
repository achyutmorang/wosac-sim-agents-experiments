from src.experiments import PACKS


def test_pack_registry_has_baseline() -> None:
    slugs = {item['slug'] for item in PACKS}
    assert 'wosac-baseline' in slugs
