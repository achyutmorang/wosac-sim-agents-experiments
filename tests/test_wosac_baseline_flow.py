from __future__ import annotations

import json
from pathlib import Path

from src.workflows import run_wosac_baseline_flow


def test_wosac_baseline_flow_dry_run_writes_artifacts(tmp_path: Path) -> None:
    bundle = run_wosac_baseline_flow(
        repo_root=tmp_path,
        persist_root=tmp_path / "persist",
        run_prefix="wosac_baseline",
        run_name="dev",
        run_tag="20260302T000000Z",
    )

    assert bundle.summary["status"] == "dry_run"
    assert bundle.metrics["realism_meta_metric"] is None
    assert "summary_json" in bundle.artifacts
    assert "metrics_json" in bundle.artifacts

    summary_path = Path(bundle.artifacts["summary_json"])
    metrics_path = Path(bundle.artifacts["metrics_json"])
    assert summary_path.exists()
    assert metrics_path.exists()


def test_wosac_baseline_flow_ingests_metrics_json(tmp_path: Path) -> None:
    metrics_input = {
        "results": {
            "realism_meta_metric": 0.7812,
            "simulated_collision_rate": 0.037,
            "simulated_offroad_rate": 0.014,
            "simulated_traffic_light_violation_rate": 0.009,
        }
    }
    input_path = tmp_path / "official_metrics.json"
    input_path.write_text(json.dumps(metrics_input))

    bundle = run_wosac_baseline_flow(
        repo_root=tmp_path,
        persist_root=tmp_path / "persist",
        run_prefix="wosac_baseline",
        run_name="dev",
        run_tag="20260302T000000Z",
        official_metrics_json=str(input_path),
    )

    assert bundle.summary["status"] == "metrics_loaded"
    assert bundle.summary["metrics_source"] == "json"
    assert bundle.metrics["realism_meta_metric"] == 0.7812
    assert bundle.metrics["simulated_collision_rate"] == 0.037
    assert bundle.metrics["simulated_offroad_rate"] == 0.014
    assert bundle.metrics["simulated_traffic_light_violation_rate"] == 0.009
