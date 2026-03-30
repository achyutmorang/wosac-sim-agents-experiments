from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence


def is_remote_path(text: str) -> bool:
    value = str(text).strip()
    return value.startswith("gs://") or value.startswith("s3://") or value.startswith("http://") or value.startswith("https://")


def normalize_path_value(value: Any, *, base_dir: Path) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text or is_remote_path(text):
            return text
        candidate = Path(text).expanduser()
        if candidate.is_absolute():
            return str(candidate)
        return str((base_dir / candidate).resolve())
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        return [normalize_path_value(elem, base_dir=base_dir) for elem in value]
    return value


def normalize_dataset_paths(config: Any, *, smart_repo_dir: Path) -> Any:
    dataset = getattr(config, "Dataset", None)
    if dataset is None:
        return config
    for key in (
        "root",
        "train_raw_dir",
        "val_raw_dir",
        "test_raw_dir",
        "train_processed_dir",
        "val_processed_dir",
        "test_processed_dir",
    ):
        if hasattr(dataset, key):
            setattr(dataset, key, normalize_path_value(getattr(dataset, key), base_dir=smart_repo_dir))
    return config
