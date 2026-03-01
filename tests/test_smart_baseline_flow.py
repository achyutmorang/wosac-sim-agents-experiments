from __future__ import annotations

import json
from pathlib import Path

from src.workflows import run_smart_baseline_flow


def test_smart_baseline_flow_ready_without_sync(tmp_path: Path) -> None:
    bundle = run_smart_baseline_flow(
        repo_root=tmp_path,
        persist_root=tmp_path / "persist",
        run_prefix="smart_baseline",
        run_name="dev",
        run_tag="20260302T000000Z",
        sync_smart_repo=False,
    )

    assert bundle.summary["status"] == "ready"
    assert bundle.summary["smart_repo_sync"]["mode"] == "skipped"
    assert "setup_cmd" in bundle.command_plan
    assert "train_cmd" in bundle.command_plan
    assert "summary_json" in bundle.artifacts
    assert "command_plan_json" in bundle.artifacts


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
