from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from src.platform.smart_rollout_contract import require_official_rollout_count
from src.platform.smart_rollout_paths import normalize_dataset_paths, normalize_path_value


def test_normalize_path_value_resolves_relative_local_path(tmp_path: Path) -> None:
    base = tmp_path / "SMART"
    base.mkdir(parents=True, exist_ok=True)
    resolved = normalize_path_value("data/valid_demo", base_dir=base)
    assert resolved == str((base / "data/valid_demo").resolve())


def test_normalize_path_value_preserves_remote_and_empty_values(tmp_path: Path) -> None:
    base = tmp_path / "SMART"
    assert normalize_path_value("gs://bucket/path", base_dir=base) == "gs://bucket/path"
    assert normalize_path_value("", base_dir=base) == ""
    assert normalize_path_value(None, base_dir=base) is None


def test_normalize_dataset_paths_updates_dataset_entries(tmp_path: Path) -> None:
    base = tmp_path / "SMART"
    cfg = SimpleNamespace(
        Dataset=SimpleNamespace(
            root=".",
            val_raw_dir=["data/valid_demo"],
            val_processed_dir=None,
            train_raw_dir="/abs/train",
        )
    )
    out = normalize_dataset_paths(cfg, smart_repo_dir=base)
    assert out.Dataset.root == str(base.resolve())
    assert out.Dataset.val_raw_dir == [str((base / "data/valid_demo").resolve())]
    assert out.Dataset.val_processed_dir is None
    assert out.Dataset.train_raw_dir == "/abs/train"


def test_require_official_rollout_count_rejects_nonstandard_count_for_validation_inputs() -> None:
    with pytest.raises(ValueError, match="requires rollout_count=32"):
        require_official_rollout_count(
            rollout_count=4,
            scenario_proto_path="",
            scenario_proto_dir="",
            scenario_tfrecords="gs://bucket/validation.tfrecord-*",
            strict_validation=False,
        )


def test_require_official_rollout_count_allows_default_count() -> None:
    require_official_rollout_count(
        rollout_count=32,
        scenario_proto_path="",
        scenario_proto_dir="",
        scenario_tfrecords="gs://bucket/validation.tfrecord-*",
        strict_validation=False,
    )
