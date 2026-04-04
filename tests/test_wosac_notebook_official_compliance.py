from __future__ import annotations

import json
from pathlib import Path


def test_wosac_baseline_notebook_has_official_sim_agents_pipeline() -> None:
    root = Path(__file__).resolve().parents[1]
    notebook_path = root / "experiments" / "wosac-baseline" / "notebooks" / "wosac_baseline_colab.ipynb"
    nb = json.loads(notebook_path.read_text(encoding="utf-8"))
    all_source = "\n".join("".join(cell.get("source", [])) for cell in nb.get("cells", []))

    required_tokens = (
        "from waymo_open_dataset.protos import scenario_pb2",
        "from waymo_open_dataset.protos import sim_agents_submission_pb2",
        "from waymo_open_dataset.utils.sim_agents import submission_specs",
        "metrics.compute_scenario_metrics_for_bundle",
        "submission_specs.validate_scenario_rollouts",
        "SimAgentsChallengeSubmission",
    )
    for token in required_tokens:
        assert token in all_source


def test_smart_baseline_eval_notebook_keeps_official_rollout_count_for_smoke_eval() -> None:
    root = Path(__file__).resolve().parents[1]
    notebook_path = root / "experiments" / "smart-baseline" / "notebooks" / "smart_baseline_eval_colab.ipynb"
    nb = json.loads(notebook_path.read_text(encoding="utf-8"))
    all_source = "\n".join("".join(cell.get("source", [])) for cell in nb.get("cells", []))

    assert (
        "DEFAULT_EVAL_ROLLOUT_COUNT = OFFICIAL_REQUIRED_ROLLOUT_COUNT if SMOKE_EVAL_DEFAULT else "
        "int(RUN.get('n_rollouts_per_scenario', OFFICIAL_REQUIRED_ROLLOUT_COUNT))"
    ) in all_source
    assert "Official Sim Agents evaluation requires SMART_ROLLOUT_COUNT=" in all_source
