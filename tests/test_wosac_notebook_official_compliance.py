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
