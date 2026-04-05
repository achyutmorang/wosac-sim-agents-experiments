from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

from src.platform.smart_rollout_paths import normalize_dataset_paths
from src.workflows.wosac_official_metrics import _load_rollouts, _load_scenarios


DEFAULT_SELECTION_STRATEGY = "representative_safe"
SUPPORTED_SELECTION_STRATEGIES = (
    "representative_safe",
    "best_safe",
    "lowest_min_ade_safe",
    "explicit",
)
DEFAULT_PROCESSED_SPLIT = "validation"
DEFAULT_VIDEO_FPS = 10
DEFAULT_VIDEO_DPI = 120
TOKEN_SHIFT = 5


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def _load_json(path: str | Path) -> Dict[str, Any]:
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected mapping JSON at {path}")
    return payload


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        if isinstance(value, str) and not value.strip():
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_per_scenario_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    normalized = {
        "scenario_id": str(row.get("scenario_id", "")).strip(),
        "metametric": _safe_float(row.get("metametric")),
        "min_average_displacement_error": _safe_float(row.get("min_average_displacement_error")),
        "simulated_collision_rate": _safe_float(row.get("simulated_collision_rate")),
        "simulated_offroad_rate": _safe_float(row.get("simulated_offroad_rate")),
        "simulated_traffic_light_violation_rate": _safe_float(row.get("simulated_traffic_light_violation_rate")),
    }
    normalized["safe"] = scenario_is_safe(normalized)
    return normalized


def scenario_is_safe(row: Mapping[str, Any]) -> bool:
    return (
        (_safe_float(row.get("simulated_collision_rate")) == 0.0)
        and (_safe_float(row.get("simulated_offroad_rate")) == 0.0)
        and (_safe_float(row.get("simulated_traffic_light_violation_rate")) == 0.0)
    )


def load_visualization_metrics(metrics_json_path: str | Path) -> Dict[str, Any]:
    payload = _load_json(metrics_json_path)
    per_scenario = payload.get("per_scenario")
    if not isinstance(per_scenario, list) or not per_scenario:
        raise ValueError(f"Official metrics JSON has no per_scenario rows: {metrics_json_path}")
    payload["per_scenario"] = [_normalize_per_scenario_row(row) for row in per_scenario if isinstance(row, Mapping)]
    return payload


def rank_visualization_candidates(
    metrics_payload: Mapping[str, Any],
    *,
    strategy: str = DEFAULT_SELECTION_STRATEGY,
    limit: int = 5,
) -> List[Dict[str, Any]]:
    rows_raw = metrics_payload.get("per_scenario", [])
    rows = [_normalize_per_scenario_row(row) for row in rows_raw if isinstance(row, Mapping)]
    rows = [row for row in rows if row["scenario_id"]]
    if not rows:
        raise ValueError("No per-scenario metrics available for visualization.")

    strategy_text = str(strategy or DEFAULT_SELECTION_STRATEGY).strip().lower()
    if strategy_text not in SUPPORTED_SELECTION_STRATEGIES:
        raise ValueError(f"Unsupported visualization selection strategy: {strategy}")

    safe_rows = [dict(row) for row in rows if row["safe"]]
    pool: List[Dict[str, Any]]
    pool_name = "safe" if safe_rows else "all"
    pool = safe_rows if safe_rows else [dict(row) for row in rows]

    metric_values = [row["metametric"] for row in pool if row["metametric"] is not None]
    median_meta = median(metric_values) if metric_values else None
    for row in pool:
        row["selection_pool"] = pool_name
        row["selection_strategy"] = strategy_text
        row["representative_gap"] = (
            abs(float(row["metametric"]) - float(median_meta))
            if (median_meta is not None and row["metametric"] is not None)
            else math.inf
        )

    def sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
        meta = row.get("metametric")
        meta_sort = -float(meta) if meta is not None else math.inf
        min_ade = row.get("min_average_displacement_error")
        ade_sort = float(min_ade) if min_ade is not None else math.inf
        rep_gap = float(row.get("representative_gap", math.inf))
        sid = str(row.get("scenario_id", ""))
        if strategy_text == "representative_safe":
            return (rep_gap, ade_sort, meta_sort, sid)
        if strategy_text == "best_safe":
            return (meta_sort, ade_sort, sid)
        if strategy_text == "lowest_min_ade_safe":
            return (ade_sort, rep_gap, meta_sort, sid)
        return (sid,)

    pool.sort(key=sort_key)
    return pool[: max(int(limit), 1)]


def select_visualization_scenario(
    metrics_payload: Mapping[str, Any],
    *,
    scenario_id: str = "",
    strategy: str = DEFAULT_SELECTION_STRATEGY,
) -> Dict[str, Any]:
    explicit_id = str(scenario_id).strip()
    rows_raw = metrics_payload.get("per_scenario", [])
    rows = [_normalize_per_scenario_row(row) for row in rows_raw if isinstance(row, Mapping)]
    if explicit_id:
        for row in rows:
            if row["scenario_id"] == explicit_id:
                picked = dict(row)
                picked["selection_strategy"] = "explicit"
                picked["selection_pool"] = "explicit"
                picked["selection_reason"] = "scenario_id override"
                return picked
        raise ValueError(f"Requested scenario_id is absent from official metrics JSON: {explicit_id}")

    ranked = rank_visualization_candidates(metrics_payload, strategy=strategy, limit=1)
    picked = dict(ranked[0])
    picked["selection_reason"] = (
        "safe scenario nearest median metametric"
        if picked.get("selection_strategy") == "representative_safe"
        else f"ranked by {picked.get('selection_strategy')}"
    )
    return picked


def find_processed_scenario_file(
    *,
    processed_root: str | Path,
    scenario_id: str,
    split: str = DEFAULT_PROCESSED_SPLIT,
) -> Path:
    root = Path(processed_root).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"Missing SMART processed root: {root}")
    candidates: List[Path] = []
    split_dir = root / split
    search_roots = [split_dir, root] if split_dir.exists() else [root]
    for base in search_roots:
        for suffix in (".pkl", ".pickle"):
            candidate = base / f"{scenario_id}{suffix}"
            if candidate.exists() and candidate.is_file():
                return candidate
        candidates.extend(list(base.glob(f"{scenario_id}.*")))
    raise FileNotFoundError(
        f"Could not find processed SMART sample for scenario_id={scenario_id} under {search_roots}. "
        f"Nearby matches: {[str(p) for p in candidates[:5]]}"
    )


def _prepare_imports(smart_repo_dir: Path, *, repo_root: Path) -> None:
    smart_repo = str(smart_repo_dir)
    if smart_repo not in sys.path:
        sys.path.insert(0, smart_repo)
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


def _seed_all(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np  # pylint: disable=import-outside-toplevel

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch  # pylint: disable=import-outside-toplevel

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _load_visualization_rollout_specs(
    *,
    scenario_rollouts_path: str,
) -> List[Dict[str, Any]]:
    from waymo_open_dataset.protos import sim_agents_submission_pb2  # pylint: disable=import-outside-toplevel

    raw_rollouts = _load_rollouts(Path(scenario_rollouts_path).expanduser(), sim_agents_submission_pb2=sim_agents_submission_pb2)
    specs: List[Dict[str, Any]] = []
    for rollout in raw_rollouts:
        joint_scenes = []
        for scene in rollout.joint_scenes:
            trajectories = []
            for traj in scene.simulated_trajectories:
                trajectories.append(
                    {
                        "object_id": int(traj.object_id),
                        "center_x": [float(v) for v in traj.center_x],
                        "center_y": [float(v) for v in traj.center_y],
                        "center_z": [float(v) for v in traj.center_z],
                        "heading": [float(v) for v in traj.heading],
                    }
                )
            joint_scenes.append({"simulated_trajectories": trajectories})
        specs.append({"scenario_id": str(rollout.scenario_id).strip(), "joint_scenes": joint_scenes})
    return specs


def _select_rollout_spec(
    rollout_specs: Sequence[Mapping[str, Any]],
    *,
    scenario_id: str,
) -> tuple[int, Dict[str, Any]]:
    for index, spec in enumerate(rollout_specs):
        if str(spec.get("scenario_id", "")).strip() == str(scenario_id).strip():
            return index, dict(spec)
    raise ValueError(f"Scenario {scenario_id} is absent from the rollout proto.")


def _load_visualization_scenario(
    *,
    scenario_id: str,
    scenario_proto_path: str = "",
    scenario_proto_dir: str = "",
    scenario_tfrecords: Any = None,
) -> Any:
    from waymo_open_dataset.protos import scenario_pb2  # pylint: disable=import-outside-toplevel

    scenarios = _load_scenarios(
        required_ids=[scenario_id],
        scenario_pb2=scenario_pb2,
        scenario_proto_path=scenario_proto_path,
        scenario_proto_dir=scenario_proto_dir,
        scenario_tfrecords=scenario_tfrecords,
    )
    scenario = scenarios.get(str(scenario_id).strip())
    if scenario is None:
        raise ValueError(
            "Could not resolve scenario proto required for visualization. "
            f"scenario_id={scenario_id}, scenario_proto_path={scenario_proto_path}, "
            f"scenario_proto_dir={scenario_proto_dir}, scenario_tfrecords={scenario_tfrecords}"
        )
    return scenario


def _track_by_object_id(scenario: Any) -> Dict[int, Any]:
    return {int(track.id): track for track in scenario.tracks}


def _state_xy(track: Any, step: int) -> Optional[tuple[float, float]]:
    if step < 0 or step >= len(track.states):
        return None
    state = track.states[step]
    if not bool(getattr(state, "valid", False)):
        return None
    return float(state.center_x), float(state.center_y)


def _track_valid_points(track: Any, start: int, end: int) -> List[tuple[float, float]]:
    out: List[tuple[float, float]] = []
    for step in range(max(start, 0), min(end, len(track.states))):
        point = _state_xy(track, step)
        if point is not None:
            out.append(point)
    return out


def _simulated_object_ids(scenario_rollout_spec: Mapping[str, Any], rollout_index: int) -> List[int]:
    joint_scenes = scenario_rollout_spec.get("joint_scenes", [])
    if rollout_index < 0 or rollout_index >= len(joint_scenes):
        raise IndexError(
            f"rollout_index={rollout_index} exceeds available joint scenes={len(joint_scenes)} "
            f"for scenario_id={scenario_rollout_spec.get('scenario_id', '')}"
        )
    return [int(traj.get("object_id", -1)) for traj in joint_scenes[rollout_index].get("simulated_trajectories", [])]


def choose_focal_object_id(
    *,
    scenario: Any,
    scenario_rollout_spec: Mapping[str, Any],
    rollout_index: int,
    preferred_object_id: int | None = None,
) -> int:
    simulated_ids = [oid for oid in _simulated_object_ids(scenario_rollout_spec, rollout_index) if oid >= 0]
    if not simulated_ids:
        raise ValueError(f"No simulated trajectories found for rollout_index={rollout_index}")

    preferred = int(preferred_object_id or 0)
    if preferred and preferred in simulated_ids:
        return preferred

    sdc_index = int(getattr(scenario, "sdc_track_index", -1))
    if 0 <= sdc_index < len(scenario.tracks):
        sdc_id = int(scenario.tracks[sdc_index].id)
        if sdc_id in simulated_ids:
            return sdc_id

    current_step = int(getattr(scenario, "current_time_index", 10))
    track_by_id = _track_by_object_id(scenario)
    current_positions: List[tuple[int, float, float]] = []
    for object_id in simulated_ids:
        track = track_by_id.get(object_id)
        if track is None:
            continue
        point = _state_xy(track, current_step)
        if point is not None:
            current_positions.append((object_id, point[0], point[1]))

    if current_positions:
        mean_x = sum(x for _, x, _ in current_positions) / float(len(current_positions))
        mean_y = sum(y for _, _, y in current_positions) / float(len(current_positions))
        current_positions.sort(key=lambda row: (row[1] - mean_x) ** 2 + (row[2] - mean_y) ** 2)
        return int(current_positions[0][0])

    return int(simulated_ids[0])


def _resolve_rollout_seed(*, base_seed: int, scenario_export_index: int, rollout_index: int) -> int:
    return int(base_seed) + int(scenario_export_index) * 1000 + int(rollout_index)


def _resolve_config_path(*, repo_root: Path, smart_repo_dir: Path, value: str) -> Path:
    text = str(value).strip()
    if not text:
        raise ValueError("SMART visualization requires a non-empty config path.")

    candidate = Path(text).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()

    repo_candidate = (repo_root / candidate).resolve()
    if repo_candidate.exists():
        return repo_candidate

    smart_candidate = (smart_repo_dir / candidate).resolve()
    if smart_candidate.exists():
        return smart_candidate

    return repo_candidate


def _prepare_single_batch(
    *,
    smart_repo_dir: Path,
    config_path: str,
    pretrain_ckpt: str,
    processed_root: str,
    processed_split: str,
    scenario_id: str,
    rollout_seed: int,
) -> Dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    _prepare_imports(smart_repo_dir, repo_root=repo_root)

    import pickle  # pylint: disable=import-outside-toplevel

    import torch  # pylint: disable=import-outside-toplevel
    from torch_geometric.data import Batch  # pylint: disable=import-outside-toplevel

    from smart.datasets.preprocess import TokenProcessor  # pylint: disable=import-outside-toplevel
    from smart.model import SMART  # pylint: disable=import-outside-toplevel
    from smart.transforms import WaymoTargetBuilder  # pylint: disable=import-outside-toplevel
    from smart.utils.config import load_config_act  # pylint: disable=import-outside-toplevel
    from smart.utils.log import Logging  # pylint: disable=import-outside-toplevel

    resolved_config_path = _resolve_config_path(repo_root=repo_root, smart_repo_dir=smart_repo_dir, value=config_path)
    config = load_config_act(str(resolved_config_path))
    config = normalize_dataset_paths(config, smart_repo_dir=smart_repo_dir)
    model_config = config.Model

    processed_file = find_processed_scenario_file(
        processed_root=processed_root,
        scenario_id=scenario_id,
        split=processed_split,
    )
    with processed_file.open("rb") as handle:
        raw_data = pickle.load(handle)

    processor = TokenProcessor(2048)
    data = processor.preprocess(raw_data)
    transform = WaymoTargetBuilder(model_config.num_historical_steps, model_config.decoder.num_future_steps)
    data = transform(data)
    batch = Batch.from_data_list([data])

    model = SMART(model_config)
    logger = Logging().log(level="DEBUG")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_params_from_file(
        filename=pretrain_ckpt,
        logger=logger,
        to_cpu=(device.type == "cpu"),
    )
    model = model.to(device)
    model.eval()
    model.inference_token = True

    _seed_all(int(rollout_seed))
    batch = copy.deepcopy(batch)
    batch = batch.to(device)
    batch = model.match_token_map(batch)
    batch = model.sample_pt_pred(batch)
    batch["agent"]["av_index"] += batch["agent"]["ptr"][:-1]
    with torch.inference_mode():
        pred = model.inference(batch)

    return {
        "model": model,
        "batch": batch,
        "pred": pred,
        "processed_file": str(processed_file),
        "device": str(device),
        "config_path": str(resolved_config_path),
    }


def compute_smart_token_trace(
    *,
    smart_repo_dir: str,
    config_path: str,
    pretrain_ckpt: str,
    processed_root: str,
    scenario_id: str,
    scenario_export_index: int,
    rollout_index: int = 0,
    base_seed: int = 2,
    focal_object_id: int,
    processed_split: str = DEFAULT_PROCESSED_SPLIT,
) -> Dict[str, Any]:
    bundle = _prepare_single_batch(
        smart_repo_dir=Path(smart_repo_dir).expanduser().resolve(),
        config_path=config_path,
        pretrain_ckpt=pretrain_ckpt,
        processed_root=processed_root,
        processed_split=processed_split,
        scenario_id=scenario_id,
        rollout_seed=_resolve_rollout_seed(
            base_seed=base_seed,
            scenario_export_index=scenario_export_index,
            rollout_index=rollout_index,
        ),
    )

    batch = bundle["batch"]
    pred = bundle["pred"]

    agent_ids_raw = batch["agent"]["id"]
    if hasattr(agent_ids_raw, "detach"):
        agent_ids = [int(v) for v in agent_ids_raw.detach().cpu().tolist()]
    else:
        agent_ids = [int(v) for v in list(agent_ids_raw)]
    try:
        focal_index = agent_ids.index(int(focal_object_id))
    except ValueError as exc:
        raise ValueError(f"focal_object_id={focal_object_id} is absent from the SMART batch ids.") from exc

    token_ids = pred["next_token_idx"][focal_index].detach().cpu().tolist()
    token_probs = pred["pred_prob"][focal_index].detach().cpu().tolist()
    pred_traj = pred["pred_traj"][focal_index].detach().cpu().tolist()
    pred_head = pred["pred_head"][focal_index].detach().cpu().tolist()

    token_trace = {
        "scenario_id": str(scenario_id),
        "focal_object_id": int(focal_object_id),
        "scenario_export_index": int(scenario_export_index),
        "rollout_index": int(rollout_index),
        "base_seed": int(base_seed),
        "rollout_seed": _resolve_rollout_seed(
            base_seed=base_seed,
            scenario_export_index=scenario_export_index,
            rollout_index=rollout_index,
        ),
        "token_shift": TOKEN_SHIFT,
        "token_ids": [int(v) for v in token_ids],
        "token_probs": [float(v) for v in token_probs],
        "pred_traj": pred_traj,
        "pred_head": [float(v) for v in pred_head],
        "processed_file": str(bundle["processed_file"]),
        "device": str(bundle["device"]),
        "config_path": str(bundle["config_path"]),
    }
    wire = json.dumps(token_trace, sort_keys=True).encode("utf-8")
    token_trace["sha256"] = f"sha256:{hashlib.sha256(wire).hexdigest()}"
    return token_trace


def _extract_map_polylines(scenario: Any) -> Dict[str, List[List[tuple[float, float]]]]:
    out: Dict[str, List[List[tuple[float, float]]]] = {
        "lane": [],
        "road_line": [],
        "road_edge": [],
        "crosswalk": [],
        "speed_bump": [],
        "driveway": [],
    }
    for feature in scenario.map_features:
        if feature.HasField("lane") and len(feature.lane.polyline) > 0:
            out["lane"].append([(float(p.x), float(p.y)) for p in feature.lane.polyline])
        elif feature.HasField("road_line") and len(feature.road_line.polyline) > 0:
            out["road_line"].append([(float(p.x), float(p.y)) for p in feature.road_line.polyline])
        elif feature.HasField("road_edge") and len(feature.road_edge.polyline) > 0:
            out["road_edge"].append([(float(p.x), float(p.y)) for p in feature.road_edge.polyline])
        elif feature.HasField("crosswalk") and len(feature.crosswalk.polygon) > 0:
            out["crosswalk"].append([(float(p.x), float(p.y)) for p in feature.crosswalk.polygon])
        elif feature.HasField("speed_bump") and len(feature.speed_bump.polygon) > 0:
            out["speed_bump"].append([(float(p.x), float(p.y)) for p in feature.speed_bump.polygon])
        elif feature.HasField("driveway") and len(feature.driveway.polygon) > 0:
            out["driveway"].append([(float(p.x), float(p.y)) for p in feature.driveway.polygon])
    return out


def _extract_rollout_trajectories(
    scenario_rollout_spec: Mapping[str, Any],
    *,
    rollout_index: int,
) -> Dict[int, Dict[str, List[float]]]:
    trajectories = {}
    joint_scene = scenario_rollout_spec.get("joint_scenes", [])[rollout_index]
    for traj in joint_scene.get("simulated_trajectories", []):
        object_id = int(traj.get("object_id", -1))
        trajectories[object_id] = {
            "center_x": [float(v) for v in traj.get("center_x", [])],
            "center_y": [float(v) for v in traj.get("center_y", [])],
            "center_z": [float(v) for v in traj.get("center_z", [])],
            "heading": [float(v) for v in traj.get("heading", [])],
        }
    if not trajectories:
        raise ValueError(f"No simulated trajectories found for rollout_index={rollout_index}.")
    return trajectories


def _view_radius_for_scene(
    *,
    focal_center: tuple[float, float],
    track_by_id: Mapping[int, Any],
    rollout_trajs: Mapping[int, Mapping[str, Sequence[float]]],
    focal_object_id: int,
    current_step: int,
    user_radius: float,
) -> float:
    radius = float(user_radius)
    if radius > 0:
        return radius

    x0, y0 = focal_center
    distances: List[float] = []
    for object_id, traj in rollout_trajs.items():
        xs = [float(v) for v in traj.get("center_x", [])]
        ys = [float(v) for v in traj.get("center_y", [])]
        distances.extend([math.hypot(x - x0, y - y0) for x, y in zip(xs, ys)])
        track = track_by_id.get(object_id)
        if track is not None:
            distances.extend(
                [
                    math.hypot(x - x0, y - y0)
                    for x, y in _track_valid_points(track, max(0, current_step - 10), current_step + 1)
                ]
            )

    inferred = max(distances) if distances else 30.0
    inferred = max(inferred + 10.0, 35.0)
    return min(inferred, 120.0)


def _render_scene_to_mp4(
    *,
    scenario: Any,
    scenario_rollout_spec: Mapping[str, Any],
    rollout_index: int,
    focal_object_id: int,
    token_trace: Optional[Mapping[str, Any]],
    output_mp4: Path,
    fps: int,
    dpi: int,
    view_radius: float,
) -> Dict[str, Any]:
    import matplotlib  # pylint: disable=import-outside-toplevel

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel
    from matplotlib.animation import FFMpegWriter  # pylint: disable=import-outside-toplevel

    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        raise RuntimeError("ffmpeg is required to write MP4 files in Colab.")

    track_by_id = _track_by_object_id(scenario)
    rollout_trajs = _extract_rollout_trajectories(scenario_rollout_spec, rollout_index=rollout_index)
    if focal_object_id not in rollout_trajs:
        raise ValueError(f"Focal object {focal_object_id} is absent from the selected rollout.")

    current_step = int(getattr(scenario, "current_time_index", 10))
    focal_track = track_by_id.get(int(focal_object_id))
    focal_history = _track_valid_points(focal_track, 0, current_step + 1) if focal_track is not None else []
    focal_future_logged = _track_valid_points(
        focal_track,
        current_step + 1,
        current_step + 1 + len(rollout_trajs[int(focal_object_id)]["center_x"]),
    ) if focal_track is not None else []

    if focal_history:
        focal_center = focal_history[-1]
    else:
        focal_center = (
            float(rollout_trajs[int(focal_object_id)]["center_x"][0]),
            float(rollout_trajs[int(focal_object_id)]["center_y"][0]),
        )

    radius = _view_radius_for_scene(
        focal_center=focal_center,
        track_by_id=track_by_id,
        rollout_trajs=rollout_trajs,
        focal_object_id=focal_object_id,
        current_step=current_step,
        user_radius=view_radius,
    )

    map_polylines = _extract_map_polylines(scenario)
    frame_count = len(rollout_trajs[int(focal_object_id)]["center_x"])

    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal")
    ax.set_facecolor("#f7f5ef")
    center_x, center_y = focal_center
    ax.set_xlim(center_x - radius, center_x + radius)
    ax.set_ylim(center_y - radius, center_y + radius)
    ax.set_title(f"SMART Baseline Rollout | scenario={scenario.scenario_id}")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

    for polyline in map_polylines["lane"]:
        xs, ys = zip(*polyline)
        ax.plot(xs, ys, color="#b8c1cc", linewidth=0.8, alpha=0.8, zorder=1)
    for polyline in map_polylines["road_line"]:
        xs, ys = zip(*polyline)
        ax.plot(xs, ys, color="#d4d9df", linewidth=0.6, alpha=0.7, zorder=1)
    for polyline in map_polylines["road_edge"]:
        xs, ys = zip(*polyline)
        ax.plot(xs, ys, color="#9aa4ad", linewidth=1.0, alpha=0.7, zorder=1)
    for key in ("crosswalk", "speed_bump", "driveway"):
        for polyline in map_polylines[key]:
            xs, ys = zip(*polyline)
            ax.fill(xs, ys, color="#eae3d2", alpha=0.25, zorder=0)

    for track in scenario.tracks:
        hist = _track_valid_points(track, 0, current_step + 1)
        if len(hist) >= 2:
            xs, ys = zip(*hist)
            ax.plot(xs, ys, color="#5f6c7b", linewidth=0.8, alpha=0.25, zorder=2)

    if len(focal_history) >= 2:
        xs, ys = zip(*focal_history)
        ax.plot(xs, ys, color="#f28e2b", linewidth=2.0, alpha=0.9, zorder=3, label="focal history")
    if len(focal_future_logged) >= 2:
        xs, ys = zip(*focal_future_logged)
        ax.plot(xs, ys, color="#2ca02c", linewidth=1.5, alpha=0.8, linestyle="--", zorder=3, label="logged future")

    other_lines: Dict[int, Any] = {}
    other_points: Dict[int, Any] = {}
    for object_id in rollout_trajs:
        if int(object_id) == int(focal_object_id):
            continue
        line, = ax.plot([], [], color="#4c78a8", linewidth=1.0, alpha=0.75, zorder=4)
        point, = ax.plot([], [], marker="o", markersize=3, color="#4c78a8", alpha=0.85, zorder=5)
        other_lines[int(object_id)] = line
        other_points[int(object_id)] = point

    focal_line, = ax.plot([], [], color="#d62728", linewidth=2.2, alpha=0.95, zorder=6, label="simulated future")
    focal_point, = ax.plot([], [], marker="o", markersize=6, color="#d62728", zorder=7)
    focal_heading, = ax.plot([], [], color="#d62728", linewidth=2.0, zorder=7)
    focal_chunk, = ax.plot([], [], color="#ffbf00", linewidth=3.0, alpha=0.95, zorder=8, label="current token chunk")
    info_text = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#cccccc"},
        zorder=9,
    )
    ax.legend(loc="lower right")

    writer = FFMpegWriter(fps=max(int(fps), 1), metadata={"artist": "Codex"}, bitrate=2200)

    with writer.saving(fig, str(output_mp4), dpi=max(int(dpi), 72)):
        for frame_idx in range(frame_count):
            for object_id, traj in rollout_trajs.items():
                xs = traj["center_x"][: frame_idx + 1]
                ys = traj["center_y"][: frame_idx + 1]
                if int(object_id) == int(focal_object_id):
                    focal_line.set_data(xs, ys)
                    focal_point.set_data([xs[-1]], [ys[-1]])
                    heading_vals = traj["heading"]
                    heading = heading_vals[min(frame_idx, len(heading_vals) - 1)] if heading_vals else 0.0
                    heading_len = max(radius * 0.08, 3.0)
                    hx = xs[-1] + heading_len * math.cos(heading)
                    hy = ys[-1] + heading_len * math.sin(heading)
                    focal_heading.set_data([xs[-1], hx], [ys[-1], hy])
                else:
                    other_lines[int(object_id)].set_data(xs, ys)
                    other_points[int(object_id)].set_data([xs[-1]], [ys[-1]])

            if token_trace:
                token_step = min(frame_idx // int(token_trace.get("token_shift", TOKEN_SHIFT)), len(token_trace.get("token_ids", [])) - 1)
                token_ids = token_trace.get("token_ids", [])
                token_probs = token_trace.get("token_probs", [])
                token_id = token_ids[token_step] if token_step >= 0 and token_step < len(token_ids) else None
                token_prob = token_probs[token_step] if token_step >= 0 and token_step < len(token_probs) else None
                chunk_start = max(token_step * int(token_trace.get("token_shift", TOKEN_SHIFT)), 0)
                chunk_end = min(chunk_start + int(token_trace.get("token_shift", TOKEN_SHIFT)), frame_count)
                chunk_x = rollout_trajs[int(focal_object_id)]["center_x"][chunk_start:chunk_end]
                chunk_y = rollout_trajs[int(focal_object_id)]["center_y"][chunk_start:chunk_end]
                focal_chunk.set_data(chunk_x, chunk_y)
                info_text.set_text(
                    "\n".join(
                        [
                            f"scenario_id: {scenario.scenario_id}",
                            f"rollout_index: {rollout_index}",
                            f"focal_object_id: {focal_object_id}",
                            f"frame: {frame_idx + 1}/{frame_count}",
                            f"token_step: {token_step + 1}/{len(token_ids)}",
                            f"selected_token: {token_id}",
                            f"selected_prob: {token_prob:.4f}" if token_prob is not None else "selected_prob: n/a",
                        ]
                    )
                )
            else:
                focal_chunk.set_data([], [])
                info_text.set_text(
                    "\n".join(
                        [
                            f"scenario_id: {scenario.scenario_id}",
                            f"rollout_index: {rollout_index}",
                            f"focal_object_id: {focal_object_id}",
                            f"frame: {frame_idx + 1}/{frame_count}",
                            "token_trace: unavailable",
                        ]
                    )
                )

            writer.grab_frame()

    plt.close(fig)
    return {
        "output_mp4": str(output_mp4),
        "output_exists": output_mp4.exists(),
        "output_size": output_mp4.stat().st_size if output_mp4.exists() else 0,
        "frame_count": int(frame_count),
        "fps": int(fps),
        "dpi": int(dpi),
        "view_radius": float(radius),
        "ffmpeg_path": str(ffmpeg_path),
    }


def render_smart_rollout_video(
    *,
    scenario_rollouts_path: str,
    output_mp4: str,
    scenario_proto_path: str = "",
    scenario_proto_dir: str = "",
    scenario_tfrecords: Any = None,
    official_metrics_json: str = "",
    scenario_id: str = "",
    selection_strategy: str = DEFAULT_SELECTION_STRATEGY,
    selection_limit: int = 5,
    rollout_index: int = 0,
    focal_object_id: int = 0,
    smart_repo_dir: str = "",
    smart_config_path: str = "",
    pretrain_ckpt: str = "",
    processed_root: str = "",
    processed_split: str = DEFAULT_PROCESSED_SPLIT,
    seed: int = 2,
    fps: int = DEFAULT_VIDEO_FPS,
    dpi: int = DEFAULT_VIDEO_DPI,
    view_radius: float = 0.0,
    skip_token_trace: bool = False,
) -> Dict[str, Any]:
    metrics_payload = load_visualization_metrics(official_metrics_json) if str(official_metrics_json).strip() else None
    if not str(scenario_id).strip():
        if metrics_payload is None:
            raise ValueError("Either scenario_id or official_metrics_json is required for visualization selection.")
        selection = select_visualization_scenario(metrics_payload, strategy=selection_strategy)
        scenario_id = selection["scenario_id"]
        candidates = rank_visualization_candidates(metrics_payload, strategy=selection_strategy, limit=selection_limit)
    else:
        selection = {
            "scenario_id": str(scenario_id).strip(),
            "selection_strategy": "explicit",
            "selection_reason": "scenario_id override",
        }
        candidates = (
            rank_visualization_candidates(metrics_payload, strategy=selection_strategy, limit=selection_limit)
            if metrics_payload is not None
            else []
        )

    rollout_specs = _load_visualization_rollout_specs(scenario_rollouts_path=scenario_rollouts_path)
    scenario_export_index, scenario_rollout_spec = _select_rollout_spec(rollout_specs, scenario_id=scenario_id)
    scenario = _load_visualization_scenario(
        scenario_id=scenario_id,
        scenario_proto_path=scenario_proto_path,
        scenario_proto_dir=scenario_proto_dir,
        scenario_tfrecords=scenario_tfrecords,
    )
    resolved_focal_object_id = choose_focal_object_id(
        scenario=scenario,
        scenario_rollout_spec=scenario_rollout_spec,
        rollout_index=rollout_index,
        preferred_object_id=focal_object_id if int(focal_object_id) > 0 else None,
    )

    token_trace: Optional[Dict[str, Any]] = None
    token_trace_error = ""
    if not skip_token_trace:
        required = {
            "smart_repo_dir": str(smart_repo_dir).strip(),
            "smart_config_path": str(smart_config_path).strip(),
            "pretrain_ckpt": str(pretrain_ckpt).strip(),
            "processed_root": str(processed_root).strip(),
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise ValueError(f"Token trace reconstruction requires: {missing}")
        try:
            token_trace = compute_smart_token_trace(
                smart_repo_dir=smart_repo_dir,
                config_path=smart_config_path,
                pretrain_ckpt=pretrain_ckpt,
                processed_root=processed_root,
                processed_split=processed_split,
                scenario_id=scenario_id,
                scenario_export_index=scenario_export_index,
                rollout_index=rollout_index,
                base_seed=seed,
                focal_object_id=resolved_focal_object_id,
            )
        except Exception as exc:
            token_trace_error = f"{type(exc).__name__}: {exc}"
            raise

    output_path = Path(output_mp4).expanduser()
    render_summary = _render_scene_to_mp4(
        scenario=scenario,
        scenario_rollout_spec=scenario_rollout_spec,
        rollout_index=rollout_index,
        focal_object_id=resolved_focal_object_id,
        token_trace=token_trace,
        output_mp4=output_path,
        fps=fps,
        dpi=dpi,
        view_radius=view_radius,
    )

    token_trace_path = output_path.with_suffix(".token_trace.json")
    summary_path = output_path.with_suffix(".json")
    if token_trace is not None:
        _write_json(token_trace_path, token_trace)

    summary: Dict[str, Any] = {
        "created_utc": _utc_now_iso(),
        "scenario_id": str(scenario_id),
        "selection": selection,
        "candidate_scenarios": candidates,
        "scenario_export_index": int(scenario_export_index),
        "scenario_rollouts_path": str(Path(scenario_rollouts_path).expanduser()),
        "scenario_proto_path": str(scenario_proto_path),
        "scenario_proto_dir": str(scenario_proto_dir),
        "scenario_tfrecords": str(scenario_tfrecords),
        "rollout_index": int(rollout_index),
        "focal_object_id": int(resolved_focal_object_id),
        "token_trace_path": str(token_trace_path) if token_trace is not None else "",
        "token_trace_error": str(token_trace_error),
        "token_trace_present": bool(token_trace is not None),
        "smart_config_path": str(smart_config_path),
        "pretrain_ckpt": str(pretrain_ckpt),
        "processed_root": str(processed_root),
        "processed_split": str(processed_split),
        "seed": int(seed),
    }
    summary.update(render_summary)
    _write_json(summary_path, summary)
    summary["summary_json"] = str(summary_path)
    return summary
