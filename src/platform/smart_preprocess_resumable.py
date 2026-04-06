from __future__ import annotations

import importlib.util
import json
import os
import pickle
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


STATE_DIRNAME = ".preprocess_state"


@dataclass(frozen=True)
class PreprocessRunPlan:
    input_dir: Path
    output_dir: Path
    state_dir: Path
    progress_json: Path
    shards: List[Path]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def default_state_dir(output_dir: Path) -> Path:
    return output_dir / STATE_DIRNAME


def shard_marker_path(state_dir: Path, shard_name: str) -> Path:
    return state_dir / "shards" / f"{Path(shard_name).name}.json"


def list_input_shards(input_dir: Path) -> List[Path]:
    return sorted([p for p in input_dir.glob("*.tfrecord*") if p.is_file()])


def is_nonempty_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except Exception:
        return False


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def shard_marker_complete(marker: Mapping[str, Any]) -> bool:
    return str(marker.get("status", "")).strip().lower() == "complete"


def build_run_plan(
    *,
    input_dir: Path,
    output_dir: Path,
    state_dir: Path | None = None,
    progress_json: Path | None = None,
    max_shards: int = 0,
) -> PreprocessRunPlan:
    resolved_state_dir = (state_dir or default_state_dir(output_dir)).expanduser()
    resolved_progress = (progress_json or (resolved_state_dir / "progress.json")).expanduser()
    shards = list_input_shards(input_dir.expanduser())
    if int(max_shards) > 0:
        shards = shards[: int(max_shards)]
    return PreprocessRunPlan(
        input_dir=input_dir.expanduser(),
        output_dir=output_dir.expanduser(),
        state_dir=resolved_state_dir,
        progress_json=resolved_progress,
        shards=shards,
    )


def read_shard_markers(state_dir: Path) -> List[Dict[str, Any]]:
    shard_dir = state_dir / "shards"
    if not shard_dir.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for path in sorted(shard_dir.glob("*.json")):
        payload = load_json(path)
        if payload:
            rows.append(payload)
    return rows


def build_progress_payload(*, plan: PreprocessRunPlan) -> Dict[str, Any]:
    markers = read_shard_markers(plan.state_dir)
    by_name = {str(row.get("shard_name", "")): row for row in markers}
    complete = 0
    running = 0
    failed = 0
    pending = 0
    selected_names = [path.name for path in plan.shards]
    for shard_name in selected_names:
        marker = by_name.get(shard_name, {})
        status = str(marker.get("status", "")).strip().lower()
        if status == "complete":
            complete += 1
        elif status == "running":
            running += 1
        elif status == "failed":
            failed += 1
        else:
            pending += 1

    processed = 0
    skipped_existing = 0
    skipped_no_tracks = 0
    for marker in markers:
        processed += int(marker.get("processed_scenarios", 0) or 0)
        skipped_existing += int(marker.get("skipped_existing_scenarios", 0) or 0)
        skipped_no_tracks += int(marker.get("skipped_no_tracks_to_predict", 0) or 0)

    return {
        "created_utc": utc_now_iso(),
        "input_dir": str(plan.input_dir),
        "output_dir": str(plan.output_dir),
        "state_dir": str(plan.state_dir),
        "selected_shards": len(plan.shards),
        "selected_shard_names": selected_names,
        "status_counts": {
            "complete": complete,
            "running": running,
            "failed": failed,
            "pending": pending,
        },
        "scenario_counts": {
            "processed": processed,
            "skipped_existing": skipped_existing,
            "skipped_no_tracks_to_predict": skipped_no_tracks,
        },
    }


def atomic_pickle_dump(payload: Any, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + f".tmp-{os.getpid()}-{time.time_ns()}")
    try:
        with temp_path.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temp_path, output_path)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


def load_smart_data_preprocess_module(smart_repo_dir: Path):
    module_path = smart_repo_dir / "data_preprocess.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Missing SMART preprocess entrypoint: {module_path}")
    smart_repo_text = str(smart_repo_dir)
    if smart_repo_text not in sys.path:
        sys.path.insert(0, smart_repo_text)
    spec = importlib.util.spec_from_file_location("smart_upstream_data_preprocess", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load SMART preprocess module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_processed_scenario_payload(upstream: Any, scenario: Any) -> tuple[str, Dict[str, Any] | None]:
    save_infos = upstream.process_single_data(scenario)
    map_info = save_infos["map_infos"]
    track_info = save_infos["track_infos"]
    scenario_id = str(save_infos["scenario_id"])
    tracks_to_predict = save_infos["tracks_to_predict"]
    sdc_track_index = int(save_infos["sdc_track_index"])
    if len(tracks_to_predict["track_index"]) < 1:
        return scenario_id, None

    av_id = track_info["object_id"][sdc_track_index]
    dynamic_map_infos = save_infos["dynamic_map_infos"]
    tf_lights = upstream.process_dynamic_map(dynamic_map_infos)
    tf_current_light = tf_lights.loc[tf_lights["time_step"] == "11"]
    map_data = upstream.get_map_features(map_info, tf_current_light)
    new_agents_array = upstream.process_agent(track_info, tracks_to_predict, sdc_track_index, scenario_id, 0, 91)

    payload: Dict[str, Any] = {
        "scenario_id": new_agents_array["scenario_id"].values[0],
        "city": new_agents_array["city"].values[0],
        "agent": upstream.get_agent_features(new_agents_array, av_id, num_historical_steps=11),
    }
    payload.update(map_data)
    return scenario_id, payload


def _write_shard_marker(
    *,
    marker_path: Path,
    shard_name: str,
    source_path: Path,
    output_dir: Path,
    state_dir: Path,
    payload: Mapping[str, Any],
) -> None:
    body = {
        "shard_name": shard_name,
        "source_path": str(source_path),
        "output_dir": str(output_dir),
        "state_dir": str(state_dir),
    }
    body.update(dict(payload))
    write_json(marker_path, body)


def process_shard(
    *,
    smart_repo_dir: Path,
    shard_path: Path,
    output_dir: Path,
    state_dir: Path,
    skip_existing: bool = True,
    log_every: int = 25,
) -> Dict[str, Any]:
    marker_path = shard_marker_path(state_dir, shard_path.name)
    running_payload: Dict[str, Any] = {
        "status": "running",
        "started_utc": utc_now_iso(),
        "completed_utc": "",
        "records_seen": 0,
        "processed_scenarios": 0,
        "skipped_existing_scenarios": 0,
        "skipped_no_tracks_to_predict": 0,
        "last_scenario_id": "",
        "error": "",
        "smart_repo_dir": str(smart_repo_dir),
    }
    _write_shard_marker(
        marker_path=marker_path,
        shard_name=shard_path.name,
        source_path=shard_path,
        output_dir=output_dir,
        state_dir=state_dir,
        payload=running_payload,
    )

    upstream = load_smart_data_preprocess_module(smart_repo_dir)
    import tensorflow as tf  # type: ignore
    from waymo_open_dataset.protos import scenario_pb2  # type: ignore

    dataset = tf.data.TFRecordDataset(str(shard_path), compression_type="", num_parallel_reads=1)

    try:
        for record_idx, raw_record in enumerate(dataset, start=1):
            scenario = scenario_pb2.Scenario()
            scenario.ParseFromString(bytearray(raw_record.numpy()))
            scenario_id = str(scenario.scenario_id)
            output_path = output_dir / f"{scenario_id}.pkl"

            running_payload["records_seen"] = int(record_idx)
            running_payload["last_scenario_id"] = scenario_id

            if skip_existing and is_nonempty_file(output_path):
                running_payload["skipped_existing_scenarios"] = int(running_payload["skipped_existing_scenarios"]) + 1
            else:
                built_scenario_id, processed_payload = build_processed_scenario_payload(upstream, scenario)
                output_path = output_dir / f"{built_scenario_id}.pkl"
                if processed_payload is None:
                    running_payload["skipped_no_tracks_to_predict"] = int(
                        running_payload["skipped_no_tracks_to_predict"]
                    ) + 1
                else:
                    atomic_pickle_dump(processed_payload, output_path)
                    running_payload["processed_scenarios"] = int(running_payload["processed_scenarios"]) + 1

            if record_idx % max(1, int(log_every)) == 0:
                print(
                    "[smart-preprocess] "
                    f"shard={shard_path.name} records_seen={running_payload['records_seen']} "
                    f"processed={running_payload['processed_scenarios']} "
                    f"skipped_existing={running_payload['skipped_existing_scenarios']} "
                    f"skipped_no_tracks={running_payload['skipped_no_tracks_to_predict']}"
                )
                _write_shard_marker(
                    marker_path=marker_path,
                    shard_name=shard_path.name,
                    source_path=shard_path,
                    output_dir=output_dir,
                    state_dir=state_dir,
                    payload=running_payload,
                )
    except Exception as exc:
        running_payload["status"] = "failed"
        running_payload["completed_utc"] = utc_now_iso()
        running_payload["error"] = f"{type(exc).__name__}: {exc}"
        _write_shard_marker(
            marker_path=marker_path,
            shard_name=shard_path.name,
            source_path=shard_path,
            output_dir=output_dir,
            state_dir=state_dir,
            payload=running_payload,
        )
        raise

    running_payload["status"] = "complete"
    running_payload["completed_utc"] = utc_now_iso()
    _write_shard_marker(
        marker_path=marker_path,
        shard_name=shard_path.name,
        source_path=shard_path,
        output_dir=output_dir,
        state_dir=state_dir,
        payload=running_payload,
    )
    return dict(running_payload)


def pending_shards(plan: PreprocessRunPlan) -> List[Path]:
    out: List[Path] = []
    for shard_path in plan.shards:
        marker = load_json(shard_marker_path(plan.state_dir, shard_path.name))
        if not shard_marker_complete(marker):
            out.append(shard_path)
    return out


def preprocess_shards(
    *,
    smart_repo_dir: Path,
    input_dir: Path,
    output_dir: Path,
    state_dir: Path | None = None,
    progress_json: Path | None = None,
    max_shards: int = 0,
    skip_existing: bool = True,
    log_every: int = 25,
) -> Dict[str, Any]:
    plan = build_run_plan(
        input_dir=input_dir,
        output_dir=output_dir,
        state_dir=state_dir,
        progress_json=progress_json,
        max_shards=0,
    )
    plan.output_dir.mkdir(parents=True, exist_ok=True)
    plan.state_dir.mkdir(parents=True, exist_ok=True)
    (plan.state_dir / "shards").mkdir(parents=True, exist_ok=True)

    selected_shards = pending_shards(plan)
    if int(max_shards) > 0:
        selected_shards = selected_shards[: int(max_shards)]
    selected_plan = PreprocessRunPlan(
        input_dir=plan.input_dir,
        output_dir=plan.output_dir,
        state_dir=plan.state_dir,
        progress_json=plan.progress_json,
        shards=selected_shards,
    )

    initial_progress = build_progress_payload(plan=selected_plan if selected_shards else plan)
    initial_progress["status"] = "starting"
    initial_progress["smart_repo_dir"] = str(smart_repo_dir)
    write_json(plan.progress_json, initial_progress)

    processed_shard_rows: List[Dict[str, Any]] = []
    for shard_idx, shard_path in enumerate(selected_shards, start=1):
        print(
            "[smart-preprocess] "
            f"processing shard {shard_idx}/{len(selected_shards)}: {shard_path.name}"
        )
        result = process_shard(
            smart_repo_dir=smart_repo_dir,
            shard_path=shard_path,
            output_dir=plan.output_dir,
            state_dir=plan.state_dir,
            skip_existing=skip_existing,
            log_every=log_every,
        )
        processed_shard_rows.append(result)
        progress = build_progress_payload(plan=plan)
        progress["status"] = "running"
        progress["smart_repo_dir"] = str(smart_repo_dir)
        progress["last_completed_shard"] = shard_path.name
        write_json(plan.progress_json, progress)

    final_progress = build_progress_payload(plan=plan)
    final_progress["status"] = "complete" if not pending_shards(plan) else "partial"
    final_progress["smart_repo_dir"] = str(smart_repo_dir)
    final_progress["processed_shards_this_run"] = [row.get("shard_name", "") for row in processed_shard_rows]
    write_json(plan.progress_json, final_progress)
    return final_progress
