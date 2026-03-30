from __future__ import annotations

import pytest

from src.platform.smart_rollout_submission import (
    build_joint_scene_spec,
    build_scenario_rollouts_spec,
    constant_z_future,
    scenario_id_from_value,
    valid_agent_indices,
)


def test_scenario_id_from_value_decodes_strings_and_char_codes() -> None:
    assert scenario_id_from_value("abc123") == "abc123"
    assert scenario_id_from_value(["abc123"]) == "abc123"
    assert scenario_id_from_value([97, 98, 99, 49, 50, 51]) == "abc123"


def test_valid_agent_indices_uses_current_step_mask() -> None:
    valid = [
        [False, True, True],
        [False, False, True],
        [True, False, False],
    ]
    assert valid_agent_indices(valid, current_step=1) == [0]
    assert valid_agent_indices(valid, current_step=2) == [0, 1]


def test_constant_z_future_uses_last_delta_when_available() -> None:
    positions = [
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.3],
        [0.0, 0.0, 1.7],
    ]
    valid = [True, True, True]
    assert constant_z_future(positions, valid, current_step=2, horizon=3) == pytest.approx([2.1, 2.5, 2.9])


def test_build_joint_scene_spec_keeps_only_agents_valid_at_current_step() -> None:
    pred_xy = [
        [[1.0, 2.0], [2.0, 3.0]],
        [[10.0, 20.0], [20.0, 30.0]],
    ]
    pred_heading = [
        [0.1, 0.2],
        [1.1, 1.2],
    ]
    position_history = [
        [[0.0, 0.0, 1.0], [0.0, 0.0, 1.1], [0.0, 0.0, 1.2]],
        [[0.0, 0.0, 5.0], [0.0, 0.0, 5.0], [0.0, 0.0, 5.0]],
    ]
    valid_history = [
        [True, True, True],
        [True, False, False],
    ]
    object_ids = [101, 202]

    joint_scene = build_joint_scene_spec(
        pred_xy=pred_xy,
        pred_heading=pred_heading,
        position_history=position_history,
        valid_history=valid_history,
        object_ids=object_ids,
        current_step=2,
    )

    assert len(joint_scene["simulated_trajectories"]) == 1
    traj = joint_scene["simulated_trajectories"][0]
    assert traj["object_id"] == 101
    assert traj["center_x"] == [1.0, 2.0]
    assert traj["center_y"] == [2.0, 3.0]
    assert traj["heading"] == [0.1, 0.2]


def test_build_scenario_rollouts_spec_requires_scenario_id() -> None:
    spec = build_scenario_rollouts_spec(scenario_id=[115, 49], rollout_joint_scenes=[{"simulated_trajectories": []}])
    assert spec["scenario_id"] == "s1"
    assert len(spec["joint_scenes"]) == 1
