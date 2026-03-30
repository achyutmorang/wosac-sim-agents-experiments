from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence


def _to_python(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    return value


def _flatten_singletons(value: Any) -> Any:
    cur = value
    while isinstance(cur, (list, tuple)) and len(cur) == 1:
        cur = cur[0]
    return cur


def scenario_id_from_value(value: Any) -> str:
    cur = _flatten_singletons(_to_python(value))
    if cur is None:
        return ""
    if isinstance(cur, bytes):
        return cur.decode("utf-8", errors="ignore").strip()
    if isinstance(cur, str):
        return cur.strip()
    if isinstance(cur, (list, tuple)):
        if all(isinstance(item, str) and len(item) == 1 for item in cur):
            return "".join(cur).strip()
        if all(isinstance(item, (int, float)) for item in cur):
            chars: List[str] = []
            for raw in cur:
                num = int(raw)
                if num <= 0:
                    continue
                try:
                    chars.append(chr(num))
                except Exception:
                    continue
            return "".join(chars).strip()
        if len(cur) == 1:
            return scenario_id_from_value(cur[0])
    return str(cur).strip()


def int_list_from_value(value: Any) -> List[int]:
    cur = _flatten_singletons(_to_python(value))
    if cur is None:
        return []
    if isinstance(cur, (int, float)):
        return [int(cur)]
    if isinstance(cur, (list, tuple)):
        out: List[int] = []
        for item in cur:
            if isinstance(item, (list, tuple)):
                out.extend(int_list_from_value(item))
                continue
            if isinstance(item, (int, float)):
                out.append(int(item))
        return out
    return []


def bool_matrix_from_value(value: Any) -> List[List[bool]]:
    rows = _to_python(value)
    if not isinstance(rows, (list, tuple)):
        raise TypeError("Expected a 2D bool-like sequence.")
    out: List[List[bool]] = []
    for row in rows:
        if not isinstance(row, (list, tuple)):
            raise TypeError("Expected a 2D bool-like sequence.")
        out.append([bool(item) for item in row])
    return out


def float_matrix_from_value(value: Any) -> List[List[float]]:
    rows = _to_python(value)
    if not isinstance(rows, (list, tuple)):
        raise TypeError("Expected a 2D float-like sequence.")
    out: List[List[float]] = []
    for row in rows:
        if not isinstance(row, (list, tuple)):
            raise TypeError("Expected a 2D float-like sequence.")
        out.append([float(item) for item in row])
    return out


def float_tensor3_from_value(value: Any) -> List[List[List[float]]]:
    mats = _to_python(value)
    if not isinstance(mats, (list, tuple)):
        raise TypeError("Expected a 3D float-like sequence.")
    out: List[List[List[float]]] = []
    for mat in mats:
        out.append(float_matrix_from_value(mat))
    return out


def valid_agent_indices(valid_history: Any, *, current_step: int) -> List[int]:
    valid = bool_matrix_from_value(valid_history)
    out: List[int] = []
    for idx, row in enumerate(valid):
        if current_step < len(row) and bool(row[current_step]):
            out.append(idx)
    return out


def constant_z_future(
    position_history: Any,
    valid_history: Sequence[bool],
    *,
    current_step: int,
    horizon: int,
) -> List[float]:
    positions = float_matrix_from_value(position_history)
    if current_step >= len(positions):
        raise IndexError(f"current_step={current_step} exceeds position history length={len(positions)}")
    current_pos = positions[current_step]
    current_z = float(current_pos[2]) if len(current_pos) > 2 else 0.0
    dz = 0.0
    prev_step = current_step - 1
    if prev_step >= 0 and current_step < len(valid_history) and prev_step < len(valid_history):
        if bool(valid_history[current_step]) and bool(valid_history[prev_step]):
            prev_pos = positions[prev_step]
            prev_z = float(prev_pos[2]) if len(prev_pos) > 2 else current_z
            dz = current_z - prev_z
    return [current_z + dz * float(step + 1) for step in range(int(horizon))]


def build_joint_scene_spec(
    *,
    pred_xy: Any,
    pred_heading: Any,
    position_history: Any,
    valid_history: Any,
    object_ids: Any,
    current_step: int,
) -> Dict[str, Any]:
    xy = float_tensor3_from_value(pred_xy)
    heading = float_matrix_from_value(pred_heading)
    positions = float_tensor3_from_value(position_history)
    valid = bool_matrix_from_value(valid_history)
    ids = int_list_from_value(object_ids)

    if not (len(xy) == len(heading) == len(positions) == len(valid) == len(ids)):
        raise ValueError(
            "Prediction/history tensor lengths disagree: "
            f"pred_xy={len(xy)} pred_heading={len(heading)} positions={len(positions)} valid={len(valid)} ids={len(ids)}"
        )

    sim_indices = valid_agent_indices(valid, current_step=current_step)
    trajectories: List[Dict[str, Any]] = []
    for idx in sim_indices:
        agent_xy = xy[idx]
        agent_heading = heading[idx]
        if len(agent_xy) != len(agent_heading):
            raise ValueError(f"Agent {idx} has mismatched future lengths: xy={len(agent_xy)} heading={len(agent_heading)}")
        center_x = [float(step[0]) for step in agent_xy]
        center_y = [float(step[1]) for step in agent_xy]
        center_z = constant_z_future(
            positions[idx],
            valid[idx],
            current_step=current_step,
            horizon=len(agent_xy),
        )
        trajectories.append(
            {
                "object_id": int(ids[idx]),
                "center_x": center_x,
                "center_y": center_y,
                "center_z": center_z,
                "heading": [float(v) for v in agent_heading],
            }
        )
    return {"simulated_trajectories": trajectories}


def build_scenario_rollouts_spec(
    *,
    scenario_id: Any,
    rollout_joint_scenes: Iterable[Dict[str, Any]],
) -> Dict[str, Any]:
    sid = scenario_id_from_value(scenario_id)
    if not sid:
        raise ValueError("Missing scenario_id for ScenarioRollouts export.")
    joint_scenes = [dict(scene) for scene in rollout_joint_scenes]
    if not joint_scenes:
        raise ValueError(f"Scenario {sid} produced no joint scenes.")
    return {"scenario_id": sid, "joint_scenes": joint_scenes}


def scenario_rollouts_proto_from_spec(sim_agents_submission_pb2: Any, spec: Dict[str, Any]) -> Any:
    joint_scenes = []
    for scene in spec.get("joint_scenes", []):
        trajectories = []
        for traj in scene.get("simulated_trajectories", []):
            trajectories.append(
                sim_agents_submission_pb2.SimulatedTrajectory(
                    center_x=traj.get("center_x", []),
                    center_y=traj.get("center_y", []),
                    center_z=traj.get("center_z", []),
                    heading=traj.get("heading", []),
                    object_id=int(traj.get("object_id", -1)),
                )
            )
        joint_scenes.append(sim_agents_submission_pb2.JointScene(simulated_trajectories=trajectories))
    return sim_agents_submission_pb2.ScenarioRollouts(
        scenario_id=str(spec.get("scenario_id", "")).strip(),
        joint_scenes=joint_scenes,
    )


def submission_proto_from_specs(sim_agents_submission_pb2: Any, specs: Sequence[Dict[str, Any]]) -> Any:
    return sim_agents_submission_pb2.SimAgentsChallengeSubmission(
        scenario_rollouts=[scenario_rollouts_proto_from_spec(sim_agents_submission_pb2, spec) for spec in specs],
        submission_type=sim_agents_submission_pb2.SimAgentsChallengeSubmission.SIM_AGENTS_SUBMISSION,
    )
