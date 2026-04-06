from __future__ import annotations

from pathlib import Path

from src.platform.smart_preprocess_resumable import (
    build_progress_payload,
    build_run_plan,
    default_state_dir,
    shard_marker_path,
    write_json,
)


def test_default_state_dir_is_under_output_dir(tmp_path: Path) -> None:
    output_dir = tmp_path / "processed" / "training"
    assert default_state_dir(output_dir) == output_dir / ".preprocess_state"


def test_shard_marker_path_uses_shard_basename(tmp_path: Path) -> None:
    marker = shard_marker_path(tmp_path, "/tmp/path/training.tfrecord-00012-of-01000")
    assert marker == tmp_path / "shards" / "training.tfrecord-00012-of-01000.json"


def test_build_progress_payload_counts_marker_statuses(tmp_path: Path) -> None:
    input_dir = tmp_path / "raw"
    output_dir = tmp_path / "processed"
    state_dir = output_dir / ".preprocess_state"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "shards").mkdir(parents=True, exist_ok=True)

    for shard_name in ("a.tfrecord-00000", "b.tfrecord-00001", "c.tfrecord-00002"):
        (input_dir / shard_name).write_bytes(b"raw")

    write_json(
        shard_marker_path(state_dir, "a.tfrecord-00000"),
        {
            "shard_name": "a.tfrecord-00000",
            "status": "complete",
            "processed_scenarios": 10,
            "skipped_existing_scenarios": 2,
            "skipped_no_tracks_to_predict": 1,
        },
    )
    write_json(
        shard_marker_path(state_dir, "b.tfrecord-00001"),
        {
            "shard_name": "b.tfrecord-00001",
            "status": "running",
            "processed_scenarios": 4,
            "skipped_existing_scenarios": 0,
            "skipped_no_tracks_to_predict": 0,
        },
    )

    plan = build_run_plan(input_dir=input_dir, output_dir=output_dir, state_dir=state_dir)
    payload = build_progress_payload(plan=plan)

    assert payload["selected_shards"] == 3
    assert payload["status_counts"]["complete"] == 1
    assert payload["status_counts"]["running"] == 1
    assert payload["status_counts"]["pending"] == 1
    assert payload["scenario_counts"]["processed"] == 14
