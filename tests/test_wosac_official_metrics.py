from __future__ import annotations

from src.workflows.wosac_official_metrics import _format_missing_scenarios_error


def test_format_missing_scenarios_error_explains_demo_mismatch() -> None:
    message = _format_missing_scenarios_error(
        required_ids=["1a146468873b7871", "1c83f56236e33b4"],
        missing_ids=["1a146468873b7871", "1c83f56236e33b4"],
        scenario_proto_path="",
        scenario_proto_dir="",
        scenario_tfrecords="gs://waymo_open_dataset_motion_v_1_3_1/uncompressed/scenario/validation/validation.tfrecord-*",
    )

    assert "SMART smoke/demo rollouts" in message
    assert "data/valid_demo" in message
    assert "official WOSAC metrics" in message
    assert "1a146468873b7871" in message
    assert "gs://waymo_open_dataset_motion_v_1_3_1/uncompressed/scenario/validation/validation.tfrecord-*" in message
