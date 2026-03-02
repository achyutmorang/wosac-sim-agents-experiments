from __future__ import annotations

import json
from pathlib import Path

from src.workflows import run_smart_baseline_flow


def test_smart_baseline_flow_ready_without_sync(tmp_path: Path) -> None:
    lockfile = tmp_path / "lock.txt"
    lockfile.write_text("torch==1.12.0\n", encoding="utf-8")
    launcher = tmp_path / "launcher.py"
    launcher.write_text("print('launcher')\n", encoding="utf-8")

    bundle = run_smart_baseline_flow(
        repo_root=tmp_path,
        persist_root=tmp_path / "persist",
        run_prefix="smart_baseline",
        run_name="dev",
        run_tag="20260302T000000Z",
        sync_smart_repo=False,
        smart_env_lockfile=str(lockfile),
        smart_train_launcher_path=str(launcher),
        smart_train_seed=13,
        smart_profile="paper_repro",
    )

    assert bundle.summary["status"] == "ready"
    assert bundle.summary["smart_repo_sync"]["mode"] == "skipped"
    assert "setup_cmd" in bundle.command_plan
    assert "train_cmd" in bundle.command_plan
    assert "--seed 13" in bundle.command_plan["train_cmd"]
    assert "smart_train_repro.py" not in bundle.command_plan["train_cmd"]  # custom launcher path used in this test
    assert "python val.py" in bundle.command_plan["validate_cmd"]
    assert str(lockfile) in bundle.command_plan["setup_cmd"]
    assert bundle.summary["smart_train_seed"] == 13
    assert bundle.summary["smart_profile"] == "paper_repro"
    assert "eval.py" not in bundle.command_plan["validate_cmd"]
    assert "summary_json" in bundle.artifacts
    assert "command_plan_json" in bundle.artifacts
    assert "data_manifest_json" in bundle.artifacts
    assert "checkpoint_manifest_json" in bundle.artifacts
    data_manifest = json.loads(Path(bundle.artifacts["data_manifest_json"]).read_text(encoding="utf-8"))
    assert "splits" in data_manifest


def test_smart_baseline_flow_ingests_metrics_json(tmp_path: Path) -> None:
    metrics_input = {
        "scores": {
            "realism_meta_metric": 0.7591,
            "simulated_collision_rate": 0.041,
            "simulated_offroad_rate": 0.015,
            "simulated_traffic_light_violation_rate": 0.011,
        }
    }
    metrics_path = tmp_path / "smart_metrics.json"
    metrics_path.write_text(json.dumps(metrics_input))

    bundle = run_smart_baseline_flow(
        repo_root=tmp_path,
        persist_root=tmp_path / "persist",
        run_prefix="smart_baseline",
        run_name="dev",
        run_tag="20260302T000000Z",
        sync_smart_repo=False,
        official_metrics_json=str(metrics_path),
    )

    assert bundle.summary["status"] == "metrics_loaded"
    assert bundle.summary["metrics_source"] == "json"
    assert bundle.metrics["realism_meta_metric"] == 0.7591
    assert bundle.metrics["simulated_collision_rate"] == 0.041
    assert bundle.metrics["simulated_offroad_rate"] == 0.015
    assert bundle.metrics["simulated_traffic_light_violation_rate"] == 0.011


def test_smart_baseline_flow_records_checkpoint_hashes(tmp_path: Path) -> None:
    ckpt_dir = tmp_path / "ckpts"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "epoch=00.ckpt"
    ckpt_path.write_bytes(b"checkpoint-data")

    bundle = run_smart_baseline_flow(
        repo_root=tmp_path,
        persist_root=tmp_path / "persist",
        run_prefix="smart_baseline",
        run_name="dev",
        run_tag="20260302T000000Z",
        sync_smart_repo=False,
        smart_save_ckpt_path=str(ckpt_dir),
    )

    ckpt_manifest = bundle.summary["checkpoint_manifest"]
    ckpts = list(ckpt_manifest.get("checkpoints", []))
    assert len(ckpts) == 1
    assert ckpts[0]["path"] == str(ckpt_path)
    assert ckpts[0]["sha256"]
