from __future__ import annotations

import json
from pathlib import Path

from src.workflows import (
    run_smart_comparative_flow,
    run_smart_eval_flow,
    write_simulation_manifest,
)


def test_smart_eval_flow_builds_validate_commands_and_ingests_metrics(tmp_path: Path) -> None:
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    wrapper_config = tmp_path / "experiments" / "smart-baseline" / "configs" / "validation_scalable_paper_repro.yaml"
    wrapper_config.parent.mkdir(parents=True, exist_ok=True)
    wrapper_config.write_text("Dataset:\n  val_raw_dir: []\n", encoding="utf-8")

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
        smart_val_config="experiments/smart-baseline/configs/validation_scalable_paper_repro.yaml",
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
    assert str(wrapper_config) in baseline["validate_cmd"]
    assert "--pretrain_ckpt /tmp/smart_baseline.ckpt" in baseline["validate_cmd"]
    assert "python " in baseline["rollout_cmd"]
    assert "scripts/smart_rollout_export.py" in baseline["rollout_cmd"]
    assert "--output-path " in baseline["rollout_cmd"]
    assert baseline["scenario_rollouts_path"].endswith("smart_baseline.binproto")
    assert "SMART_TEMP=1.1 python val.py" in variant["validate_cmd"]
    assert "SMART_TEMP=1.1 python " in variant["rollout_cmd"]
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


def test_smart_eval_flow_strict_contract_checks_manifest_binding(tmp_path: Path) -> None:
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    baseline_ckpt = ckpt_dir / "smart_baseline.ckpt"
    baseline_ckpt.write_bytes(b"baseline")

    manifests_dir = tmp_path / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifests_dir / "smart_baseline_simulation_manifest.json"
    manifest = write_simulation_manifest(
        manifest_path,
        {
            "created_utc": "2026-03-03T00:00:00Z",
            "run_tag": "20260303T000000Z",
            "model_id": "smart_baseline",
            "checkpoint_path": str(baseline_ckpt),
            "scenario_set_id": "womd_validation",
            "scenario_set_hash": "sha256:scenariohash",
            "evaluator_id": "waymo_open_dataset.sim_agents_metrics.challenge_2025",
            "metrics_config_hash": "sha256:metricsconfighash",
            "n_rollouts": 32,
            "num_history_seconds": 1,
            "num_future_seconds": 8,
            "seed": 2,
        },
    )

    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "smart_baseline.json").write_text(
        json.dumps(
            {
                "manifest_sha256": manifest["manifest_sha256"],
                "model_id": "smart_baseline",
                "scenario_set_hash": "sha256:scenariohash",
                "evaluator_id": "waymo_open_dataset.sim_agents_metrics.challenge_2025",
                "metrics_config_hash": "sha256:metricsconfighash",
                "n_rollouts": 32,
                "num_history_seconds": 1,
                "num_future_seconds": 8,
                "seed": 2,
                "metrics": {
                    "realism_meta_metric": 0.751,
                    "simulated_collision_rate": 0.052,
                    "simulated_offroad_rate": 0.021,
                    "simulated_traffic_light_violation_rate": 0.010,
                },
            }
        ),
        encoding="utf-8",
    )

    bundle = run_smart_eval_flow(
        repo_root=tmp_path,
        persist_root=tmp_path / "persist",
        run_prefix="smart_eval",
        run_name="dev",
        run_tag="20260303T000000Z",
        sync_smart_repo=False,
        smart_repo_dir=str(tmp_path / "SMART"),
        models=[{"model_id": "smart_baseline", "checkpoint_path": str(baseline_ckpt)}],
        metrics_dir=str(metrics_dir),
        manifests_dir=str(manifests_dir),
        strict_contract=True,
        require_metrics_binding=True,
        verify_checkpoint_hash=True,
    )

    assert bundle.summary["status"] == "ready"
    assert bundle.summary["num_contract_invalid"] == 0
    assert bundle.models[0]["contract_valid"] is True
    assert bundle.models[0]["manifest_sha256"] == manifest["manifest_sha256"]


def test_smart_comparative_flow_filters_incompatible_contracts(tmp_path: Path) -> None:
    eval_payload = {
        "models": [
            {
                "model_id": "smart_baseline",
                "contract_valid": True,
                "contract_signature": {
                    "scenario_set_hash": "sha256:scenarioA",
                    "evaluator_id": "eval-v1",
                    "metrics_config_hash": "cfgA",
                    "n_rollouts": 32,
                    "num_history_seconds": 1,
                    "num_future_seconds": 8,
                },
                "metrics": {
                    "realism_meta_metric": 0.750,
                    "simulated_collision_rate": 0.055,
                    "simulated_offroad_rate": 0.022,
                    "simulated_traffic_light_violation_rate": 0.010,
                },
            },
            {
                "model_id": "variant_incompatible",
                "contract_valid": True,
                "contract_signature": {
                    "scenario_set_hash": "sha256:scenarioB",
                    "evaluator_id": "eval-v1",
                    "metrics_config_hash": "cfgA",
                    "n_rollouts": 32,
                    "num_history_seconds": 1,
                    "num_future_seconds": 8,
                },
                "metrics": {
                    "realism_meta_metric": 0.790,
                    "simulated_collision_rate": 0.050,
                    "simulated_offroad_rate": 0.021,
                    "simulated_traffic_light_violation_rate": 0.009,
                },
            },
            {
                "model_id": "variant_compatible",
                "contract_valid": True,
                "contract_signature": {
                    "scenario_set_hash": "sha256:scenarioA",
                    "evaluator_id": "eval-v1",
                    "metrics_config_hash": "cfgA",
                    "n_rollouts": 32,
                    "num_history_seconds": 1,
                    "num_future_seconds": 8,
                },
                "metrics": {
                    "realism_meta_metric": 0.770,
                    "simulated_collision_rate": 0.051,
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
        run_tag="20260303T000000Z",
        eval_models_json=str(eval_json),
        baseline_model_id="smart_baseline",
        max_collision_rate=0.06,
        max_offroad_rate=0.03,
        max_tl_violation_rate=0.02,
        require_contract_compatibility=True,
    )

    assert bundle.selection["status"] == "selected_feasible"
    assert bundle.selection["selected_model_id"] == "variant_compatible"
