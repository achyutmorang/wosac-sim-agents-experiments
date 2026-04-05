from __future__ import annotations

from pathlib import Path

from src.workflows import (
    choose_focal_object_id,
    find_processed_scenario_file,
    rank_visualization_candidates,
    select_visualization_scenario,
)


def test_rank_visualization_candidates_prefers_representative_safe() -> None:
    payload = {
        "per_scenario": [
            {
                "scenario_id": "safe_low",
                "metametric": 0.55,
                "min_average_displacement_error": 2.0,
                "simulated_collision_rate": 0.0,
                "simulated_offroad_rate": 0.0,
                "simulated_traffic_light_violation_rate": 0.0,
            },
            {
                "scenario_id": "safe_mid",
                "metametric": 0.63,
                "min_average_displacement_error": 1.4,
                "simulated_collision_rate": 0.0,
                "simulated_offroad_rate": 0.0,
                "simulated_traffic_light_violation_rate": 0.0,
            },
            {
                "scenario_id": "safe_high",
                "metametric": 0.79,
                "min_average_displacement_error": 1.1,
                "simulated_collision_rate": 0.0,
                "simulated_offroad_rate": 0.0,
                "simulated_traffic_light_violation_rate": 0.0,
            },
            {
                "scenario_id": "unsafe_best",
                "metametric": 0.90,
                "min_average_displacement_error": 0.9,
                "simulated_collision_rate": 0.2,
                "simulated_offroad_rate": 0.0,
                "simulated_traffic_light_violation_rate": 0.0,
            },
        ]
    }

    ranked = rank_visualization_candidates(payload, strategy="representative_safe", limit=3)
    assert [row["scenario_id"] for row in ranked] == ["safe_mid", "safe_low", "safe_high"]
    assert all(row["selection_pool"] == "safe" for row in ranked)


def test_select_visualization_scenario_honors_explicit_override() -> None:
    payload = {
        "per_scenario": [
            {
                "scenario_id": "a",
                "metametric": 0.5,
                "min_average_displacement_error": 1.0,
                "simulated_collision_rate": 0.0,
                "simulated_offroad_rate": 0.0,
                "simulated_traffic_light_violation_rate": 0.0,
            },
            {
                "scenario_id": "b",
                "metametric": 0.7,
                "min_average_displacement_error": 0.8,
                "simulated_collision_rate": 0.0,
                "simulated_offroad_rate": 0.0,
                "simulated_traffic_light_violation_rate": 0.0,
            },
        ]
    }
    selected = select_visualization_scenario(payload, scenario_id="a")
    assert selected["scenario_id"] == "a"
    assert selected["selection_strategy"] == "explicit"


def test_find_processed_scenario_file_prefers_split_dir(tmp_path: Path) -> None:
    root = tmp_path / "waymo_processed"
    (root / "validation").mkdir(parents=True)
    target = root / "validation" / "abc123.pkl"
    target.write_bytes(b"payload")

    resolved = find_processed_scenario_file(processed_root=root, scenario_id="abc123")
    assert resolved == target


class _State:
    def __init__(self, x: float, y: float, valid: bool = True) -> None:
        self.center_x = x
        self.center_y = y
        self.valid = valid


class _Track:
    def __init__(self, object_id: int, points: list[tuple[float, float]]) -> None:
        self.id = object_id
        self.states = [_State(x, y, True) for x, y in points]


class _Scenario:
    def __init__(self) -> None:
        self.sdc_track_index = 1
        self.current_time_index = 2
        self.tracks = [
            _Track(100, [(0, 0), (1, 0), (2, 0)]),
            _Track(200, [(10, 10), (11, 10), (12, 10)]),
            _Track(300, [(12, 12), (13, 12), (14, 12)]),
        ]


def test_choose_focal_object_id_prefers_sdc_if_simulated() -> None:
    scenario_rollout_spec = {
        "scenario_id": "demo",
        "joint_scenes": [
            {
                "simulated_trajectories": [
                    {"object_id": 100, "center_x": [0.0], "center_y": [0.0], "heading": [0.0]},
                    {"object_id": 200, "center_x": [1.0], "center_y": [1.0], "heading": [0.0]},
                ]
            }
        ],
    }

    focal = choose_focal_object_id(
        scenario=_Scenario(),
        scenario_rollout_spec=scenario_rollout_spec,
        rollout_index=0,
    )
    assert focal == 200
