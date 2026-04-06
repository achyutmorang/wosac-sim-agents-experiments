from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch


def _load_script_module():
    script_path = Path(__file__).resolve().parent.parent / "scripts" / "smart_preprocess_resumable.py"
    spec = importlib.util.spec_from_file_location("smart_preprocess_resumable_script", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("smart_preprocess_resumable_script", module)
    spec.loader.exec_module(module)
    return module


def test_parse_args_accepts_legacy_underscore_flags() -> None:
    module = _load_script_module()
    with patch.object(
        sys,
        "argv",
        [
            "smart_preprocess_resumable.py",
            "--smart_repo_dir",
            "/content/SMART",
            "--input_dir",
            "/tmp/raw",
            "--output_dir",
            "/tmp/processed",
            "--split",
            "training",
        ],
    ):
        args = module.parse_args()

    assert args.smart_repo_dir == "/content/SMART"
    assert args.input_dir == "/tmp/raw"
    assert args.output_dir == "/tmp/processed"


def test_parse_args_accepts_hyphen_flags() -> None:
    module = _load_script_module()
    with patch.object(
        sys,
        "argv",
        [
            "smart_preprocess_resumable.py",
            "--smart-repo-dir",
            "/content/SMART",
            "--input-dir",
            "/tmp/raw",
            "--output-dir",
            "/tmp/processed",
            "--split",
            "validation",
        ],
    ):
        args = module.parse_args()

    assert args.smart_repo_dir == "/content/SMART"
    assert args.input_dir == "/tmp/raw"
    assert args.output_dir == "/tmp/processed"
