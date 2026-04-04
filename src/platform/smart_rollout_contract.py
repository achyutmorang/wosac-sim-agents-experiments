from __future__ import annotations


DEFAULT_OFFICIAL_ROLLOUT_COUNT = 32


def require_official_rollout_count(
    *,
    rollout_count: int,
    scenario_proto_path: str,
    scenario_proto_dir: str,
    scenario_tfrecords: str,
    strict_validation: bool,
) -> None:
    has_validation_source = any(
        str(value).strip()
        for value in (
            scenario_proto_path,
            scenario_proto_dir,
            scenario_tfrecords,
        )
    )
    if (has_validation_source or bool(strict_validation)) and int(rollout_count) != DEFAULT_OFFICIAL_ROLLOUT_COUNT:
        raise ValueError(
            "Official Sim Agents validation requires rollout_count=32 "
            f"(got {int(rollout_count)}). Bound the number of scenarios instead."
        )
