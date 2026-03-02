from __future__ import annotations

import json
from pathlib import Path

from src.workflows import run_smart_constrained_flow


def test_smart_constrained_flow_generates_variant_grid(tmp_path: Path) -> None:
    bundle = run_smart_constrained_flow(
        repo_root=tmp_path,
        persist_root=tmp_path / "persist",
        run_prefix="smart_constrained",
        run_name="dev",
        run_tag="20260302T000000Z",
        sync_smart_repo=False,
        temperatures=[0.9, 1.1],
        top_ks=[8],
        constraint_weights=[0.1, 0.2],
    )

    assert bundle.summary["status"] == "no_variant_metrics"
    assert len(bundle.variants) == 4
    first = bundle.variants[0]
    assert "SMART_TEMP=" in first["train_cmd"]
    assert "SMART_TOP_K=" in first["train_cmd"]
    assert "SMART_CONSTRAINT_WEIGHT=" in first["validate_cmd"]
    assert "summary_json" in bundle.artifacts
    assert "variant_grid_json" in bundle.artifacts
    assert "selection_json" in bundle.artifacts


def test_smart_constrained_flow_selects_best_feasible_variant(tmp_path: Path) -> None:
    metrics_dir = tmp_path / "variant_metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    dry = run_smart_constrained_flow(
        repo_root=tmp_path,
        persist_root=tmp_path / "persist",
        run_prefix="smart_constrained",
        run_name="dev",
        run_tag="20260302T000000Z",
        sync_smart_repo=False,
        temperatures=[0.8],
        top_ks=[8, 16],
        constraint_weights=[0.1],
    )
    ids = [v["variant_id"] for v in dry.variants]
    assert len(ids) == 2

    payloads = {
        ids[0]: {
            "metrics": {
                "realism_meta_metric": 0.761,
                "simulated_collision_rate": 0.050,
                "simulated_offroad_rate": 0.020,
                "simulated_traffic_light_violation_rate": 0.010,
            }
        },
        ids[1]: {
            "metrics": {
                "realism_meta_metric": 0.770,
                "simulated_collision_rate": 0.090,
                "simulated_offroad_rate": 0.020,
                "simulated_traffic_light_violation_rate": 0.010,
            }
        },
    }
    for variant_id, payload in payloads.items():
        (metrics_dir / f"{variant_id}.json").write_text(json.dumps(payload), encoding="utf-8")

    scored = run_smart_constrained_flow(
        repo_root=tmp_path,
        persist_root=tmp_path / "persist",
        run_prefix="smart_constrained",
        run_name="dev",
        run_tag="20260302T000000Z",
        sync_smart_repo=False,
        variant_metrics_dir=str(metrics_dir),
        temperatures=[0.8],
        top_ks=[8, 16],
        constraint_weights=[0.1],
        max_collision_rate=0.06,
        max_offroad_rate=0.03,
        max_tl_violation_rate=0.02,
    )

    assert scored.selection["status"] == "selected_feasible"
    assert scored.selection["selected_variant_id"] == ids[0]


def test_smart_constrained_flow_auto_resume_variant_checkpoint(tmp_path: Path) -> None:
    persist_root = tmp_path / "persist"
    run_prefix = "smart_constrained"
    run_name = "dev"
    variant_id = "t0p8_k8_cw0p1"
    ckpt_dir = persist_root / f"{run_prefix}_{run_name}" / "checkpoints" / "variants" / variant_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    resume_ckpt = ckpt_dir / "epoch=09.ckpt"
    resume_ckpt.write_bytes(b"resume-me")

    bundle = run_smart_constrained_flow(
        repo_root=tmp_path,
        persist_root=persist_root,
        run_prefix=run_prefix,
        run_name=run_name,
        run_tag="20260302T000000Z",
        sync_smart_repo=False,
        temperatures=[0.8],
        top_ks=[8],
        constraint_weights=[0.1],
        resume_from_existing=True,
    )

    assert len(bundle.variants) == 1
    v = bundle.variants[0]
    assert v["resume_checkpoint_path"] == str(resume_ckpt)
    assert "--ckpt-path" in v["train_cmd"]
    assert str(resume_ckpt) in v["train_cmd"]
