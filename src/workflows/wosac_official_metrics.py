from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _to_serializable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_serializable(v) for v in value]
    return str(value)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_to_serializable(dict(payload)), indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _parse_csv_like_paths(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [str(v).strip() for v in value if str(v).strip()]
    return []


def _is_gcs_uri(value: str) -> bool:
    return str(value).strip().startswith("gs://")


def _expand_tfrecord_inputs(paths_value: Any, *, tf: Any) -> List[str]:
    resolved: List[str] = []
    for raw in _parse_csv_like_paths(paths_value):
        text = str(raw).strip()
        if not text:
            continue

        # Expand globs (local or GCS) first.
        if "*" in text:
            for match in sorted(tf.io.gfile.glob(text)):
                norm = str(match).strip()
                if norm and norm not in resolved:
                    resolved.append(norm)
            continue

        if _is_gcs_uri(text):
            if tf.io.gfile.exists(text) or tf.io.gfile.glob(text):
                if text not in resolved:
                    resolved.append(text)
            continue

        local = Path(text).expanduser()
        if local.exists() and local.is_file():
            norm = str(local)
            if norm not in resolved:
                resolved.append(norm)
    return resolved


def _challenge_type(challenge_name: str, submission_specs: Any) -> Any:
    text = str(challenge_name).strip().upper()
    if text in {"SIM_AGENTS", "SIMAGENTS"}:
        return submission_specs.ChallengeType.SIM_AGENTS
    if text in {"SCENARIO_GEN", "SCENARIOGEN"}:
        return submission_specs.ChallengeType.SCENARIO_GEN
    raise ValueError(f"Unsupported challenge_type: {challenge_name}")


def _load_rollouts(path: Path, sim_agents_submission_pb2: Any) -> List[Any]:
    raw = path.read_bytes()
    submission = sim_agents_submission_pb2.SimAgentsChallengeSubmission()
    submission.ParseFromString(raw)
    if len(submission.scenario_rollouts) > 0:
        return [rollout for rollout in submission.scenario_rollouts]

    one = sim_agents_submission_pb2.ScenarioRollouts()
    one.ParseFromString(raw)
    if str(one.scenario_id).strip() and len(one.joint_scenes) > 0:
        return [one]
    raise ValueError(f"Unable to parse scenario rollouts from: {path}")


def _load_scenario_from_proto(path: Path, scenario_pb2: Any) -> Any:
    scenario = scenario_pb2.Scenario()
    scenario.ParseFromString(path.read_bytes())
    if not str(scenario.scenario_id).strip():
        raise ValueError(f"Invalid scenario proto (missing scenario_id): {path}")
    return scenario


def _load_scenarios(
    *,
    required_ids: Sequence[str],
    scenario_pb2: Any,
    scenario_proto_path: str = "",
    scenario_proto_dir: str = "",
    scenario_tfrecords: Any = None,
) -> Dict[str, Any]:
    scenario_by_id: Dict[str, Any] = {}
    targets = [str(s).strip() for s in required_ids if str(s).strip()]
    target_set = set(targets)
    if not target_set:
        return scenario_by_id

    proto_path_text = str(scenario_proto_path).strip()
    if proto_path_text:
        path = Path(proto_path_text).expanduser()
        if path.exists() and path.is_file():
            scenario = _load_scenario_from_proto(path, scenario_pb2=scenario_pb2)
            if scenario.scenario_id in target_set:
                scenario_by_id[str(scenario.scenario_id)] = scenario

    proto_dir_text = str(scenario_proto_dir).strip()
    if proto_dir_text:
        root = Path(proto_dir_text).expanduser()
        for sid in targets:
            if sid in scenario_by_id:
                continue
            for suffix in (".pb", ".binproto", ".scenario.pb", ".proto"):
                candidate = root / f"{sid}{suffix}"
                if candidate.exists() and candidate.is_file():
                    scenario_by_id[sid] = _load_scenario_from_proto(candidate, scenario_pb2=scenario_pb2)
                    break

    missing = [sid for sid in targets if sid not in scenario_by_id]
    if missing:
        import tensorflow as tf  # pylint: disable=import-outside-toplevel

        tfrecord_inputs = _expand_tfrecord_inputs(scenario_tfrecords, tf=tf)
        if tfrecord_inputs:
            dataset = tf.data.TFRecordDataset(tfrecord_inputs)
            for scenario_bytes in dataset.as_numpy_iterator():
                scenario = scenario_pb2.Scenario.FromString(scenario_bytes)
                sid = str(scenario.scenario_id).strip()
                if sid in target_set and sid not in scenario_by_id:
                    scenario_by_id[sid] = scenario
                    if len(scenario_by_id) >= len(target_set):
                        break

    return scenario_by_id


def compute_official_metrics_from_rollouts(
    *,
    scenario_rollouts_path: str,
    scenario_proto_path: str = "",
    scenario_proto_dir: str = "",
    scenario_tfrecords: Any = None,
    challenge_type: str = "SIM_AGENTS",
    metrics_config_textproto: str = "",
    output_metrics_json: str = "",
    binding_fields: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    rollouts_path = Path(str(scenario_rollouts_path)).expanduser()
    if not rollouts_path.exists():
        raise FileNotFoundError(f"Missing scenario rollouts proto: {rollouts_path}")

    from google.protobuf import text_format  # pylint: disable=import-outside-toplevel
    from waymo_open_dataset.protos import scenario_pb2  # pylint: disable=import-outside-toplevel
    from waymo_open_dataset.protos import sim_agents_metrics_pb2  # pylint: disable=import-outside-toplevel
    from waymo_open_dataset.protos import sim_agents_submission_pb2  # pylint: disable=import-outside-toplevel
    from waymo_open_dataset.utils.sim_agents import submission_specs  # pylint: disable=import-outside-toplevel
    from waymo_open_dataset.wdl_limited.sim_agents_metrics import metrics  # pylint: disable=import-outside-toplevel

    challenge = _challenge_type(challenge_type, submission_specs=submission_specs)
    if str(metrics_config_textproto).strip():
        cfg = sim_agents_metrics_pb2.SimAgentMetricsConfig()
        text_format.Parse(Path(str(metrics_config_textproto).expanduser()).read_text(encoding="utf-8"), cfg)
    else:
        cfg = metrics.load_metrics_config(challenge)

    rollouts_list = _load_rollouts(rollouts_path, sim_agents_submission_pb2=sim_agents_submission_pb2)
    scenario_ids = [str(r.scenario_id).strip() for r in rollouts_list if str(r.scenario_id).strip()]
    scenarios = _load_scenarios(
        required_ids=scenario_ids,
        scenario_pb2=scenario_pb2,
        scenario_proto_path=scenario_proto_path,
        scenario_proto_dir=scenario_proto_dir,
        scenario_tfrecords=scenario_tfrecords,
    )
    missing = [sid for sid in scenario_ids if sid not in scenarios]
    if missing:
        raise ValueError(f"Missing scenario protos for scenario_ids: {missing}")

    per_scenario = []
    scenario_metrics_protos = []
    for rollout in rollouts_list:
        sid = str(rollout.scenario_id).strip()
        metric_proto = metrics.compute_scenario_metrics_for_bundle(
            cfg,
            scenarios[sid],
            rollout,
            challenge_type=challenge,
        )
        scenario_metrics_protos.append(metric_proto)
        per_scenario.append(
            {
                "scenario_id": sid,
                "metametric": float(metric_proto.metametric),
                "simulated_collision_rate": float(metric_proto.simulated_collision_rate),
                "simulated_offroad_rate": float(metric_proto.simulated_offroad_rate),
                "simulated_traffic_light_violation_rate": float(metric_proto.simulated_traffic_light_violation_rate),
                "min_average_displacement_error": float(metric_proto.min_average_displacement_error),
            }
        )

    aggregate_proto = (
        metrics.aggregate_scenario_metrics(scenario_metrics_protos)
        if len(scenario_metrics_protos) > 1
        else scenario_metrics_protos[0]
    )
    bucketed = metrics.aggregate_metrics_to_buckets(cfg, aggregate_proto)

    metrics_payload = {
        "realism_meta_metric": float(bucketed.realism_meta_metric),
        "kinematic_metrics": float(bucketed.kinematic_metrics),
        "interactive_metrics": float(bucketed.interactive_metrics),
        "map_based_metrics": float(bucketed.map_based_metrics),
        "min_ade": float(bucketed.min_ade),
        "simulated_collision_rate": float(bucketed.simulated_collision_rate),
        "simulated_offroad_rate": float(bucketed.simulated_offroad_rate),
        "simulated_traffic_light_violation_rate": float(bucketed.simulated_traffic_light_violation_rate),
    }

    cfg_wire = cfg.SerializeToString()
    config_sha = hashlib.sha256(cfg_wire).hexdigest() if cfg_wire else ""
    payload: Dict[str, Any] = {
        "created_utc": _utc_now_iso(),
        "challenge_type": str(challenge_type),
        "scenario_rollouts_path": str(rollouts_path),
        "scenario_count": len(scenario_metrics_protos),
        "scenario_ids": scenario_ids,
        "metrics_config_sha256": config_sha,
        "metrics": metrics_payload,
        "per_scenario": per_scenario,
        "source": "official_waymo_metrics_api",
    }
    if isinstance(binding_fields, Mapping) and binding_fields:
        payload.update({str(k): _to_serializable(v) for k, v in binding_fields.items()})

    out_path = str(output_metrics_json).strip()
    if out_path:
        _write_json(Path(out_path).expanduser(), payload)
        payload["output_metrics_json"] = str(Path(out_path).expanduser())
    return payload
