from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_run_smart_baseline_script_dry_run(tmp_path: Path) -> None:
    config_payload = {
        "slug": "smart-baseline",
        "title": "SMART Baseline Wrapper",
        "objective": "Smoke-test command plan generation.",
        "run": {
            "run_name": "smoke",
            "run_prefix": "smart_baseline",
            "persist_root": str(tmp_path / "persist"),
        },
        "smart": {
            "repo_url": "https://github.com/rainmaker22/SMART.git",
            "branch": "main",
            "repo_dir": str(tmp_path / "SMART"),
            "train_config": "configs/train/train_scalable.yaml",
            "val_config": "configs/validation/validation_scalable.yaml",
            "ckpt_path": "",
            "raw_data_root": str(tmp_path / "raw"),
            "processed_data_root": str(tmp_path / "processed"),
            "install_pyg": False,
        },
    }
    config_path = tmp_path / "smart-baseline.json"
    config_path.write_text(json.dumps(config_payload), encoding="utf-8")

    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_smart_baseline.py"
    proc = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--config",
            str(config_path),
            "--repo-root",
            str(repo_root),
            "--no-sync-smart-repo",
            "--print-only",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "python val.py" in proc.stdout
    assert "eval.py" not in proc.stdout
    plan_path = tmp_path / "persist" / "smart_baseline_smoke" / "outputs" / "smart_command_plan.json"
    assert plan_path.exists()
