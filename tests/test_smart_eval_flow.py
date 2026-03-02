from __future__ import annotations

import json
from pathlib import Path

from src.workflows import run_smart_comparative_flow, run_smart_eval_flow


def test_smart_eval_flow_builds_validate_commands_and_ingests_metrics(tmp_path: Path) -> None:
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    (metrics_dir / "smart_baseline.json").write_text(
        json.dumps(
            {
                "metrics": {
                    "realism_meta_metric": 0.751,
                    "simulated_collision_rate": 0.052,
                    "simulated_offroad_rate": 0.021,
                    "simulated_traffic_light_violation_rate": 0.010,
                }
            }
        ),
        encoding="utf-8",
    )
    (metrics_dir / "variant_1.json").write_text(
        json.dumps(
            {
                "metrics": {
                    "realism_meta_metric": 0.764,
                    "simulated_collision_rate": 0.059,
                    "simulated_offroad_rate": 0.020,
                    "simulated_traffic_light_violation_rate": 0.011,
                }
            }
        ),
        encoding="utf-8",
    )

    bundle = run_smart_eval_flow(
        repo_root=tmp_path,
        persist_root=tmp_path / "persist",
        run_prefix="smart_eval",
        run_name="dev",
        run_tag="20260302T000000Z",
        sync_smart_repo=False,
        smart_repo_dir=str(tmp_path / "SMART"),
        models=[
            {"model_id": "smart_baseline", "checkpoint_path": "/tmp/smart_baseline.ckpt"},
            {
                "model_id": "variant_1",
                "checkpoint_path": "/tmp/variant_1.ckpt",
                "env": {"SMART_TEMP": 1.1},
            },
        ],
        metrics_dir=str(metrics_dir),
    )

    assert bundle.summary["status"] == "ready"
    assert bundle.summary["num_models"] == 2
    assert "model_grid_json" in bundle.artifacts
    baseline = next(m for m in bundle.models if m["model_id"] == "smart_baseline")
    variant = next(m for m in bundle.models if m["model_id"] == "variant_1")
    assert "python val.py" in baseline["validate_cmd"]
    assert "--pretrain_ckpt /tmp/smart_baseline.ckpt" in baseline["validate_cmd"]
    assert "SMART_TEMP=1.1 python val.py" in variant["validate_cmd"]
    assert variant["metrics"]["realism_meta_metric"] == 0.764


def test_smart_comparative_flow_selects_best_feasible_candidate(tmp_path: Path) -> None:
    eval_payload = {
        "models": [
            {
                "model_id": "smart_baseline",
                "metrics": {
                    "realism_meta_metric": 0.750,
                    "simulated_collision_rate": 0.055,
                    "simulated_offroad_rate": 0.022,
                    "simulated_traffic_light_violation_rate": 0.010,
                },
            },
            {
                "model_id": "variant_1",
                "metrics": {
                    "realism_meta_metric": 0.770,
                    "simulated_collision_rate": 0.080,
                    "simulated_offroad_rate": 0.022,
                    "simulated_traffic_light_violation_rate": 0.010,
                },
            },
            {
                "model_id": "variant_2",
                "metrics": {
                    "realism_meta_metric": 0.763,
                    "simulated_collision_rate": 0.050,
                    "simulated_offroad_rate": 0.021,
                    "simulated_traffic_light_violation_rate": 0.009,
                },
            },
        ]
    }
    eval_json = tmp_path / "smart_eval_model_grid.json"
    eval_json.write_text(json.dumps(eval_payload), encoding="utf-8")

    bundle = run_smart_comparative_flow(
        repo_root=tmp_path,
        persist_root=tmp_path / "persist",
        run_prefix="smart_eval",
        run_name="dev",
        run_tag="20260302T000000Z",
        eval_models_json=str(eval_json),
        baseline_model_id="smart_baseline",
        max_collision_rate=0.06,
        max_offroad_rate=0.03,
        max_tl_violation_rate=0.02,
    )

    assert bundle.selection["status"] == "selected_feasible"
    assert bundle.selection["selected_model_id"] == "variant_2"
    assert "report_json" in bundle.artifacts
