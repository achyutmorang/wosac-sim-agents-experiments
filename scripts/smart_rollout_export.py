#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import traceback
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.platform.smart_rollout_paths import normalize_dataset_paths, normalize_path_value
from src.platform.smart_rollout_submission import (
    build_joint_scene_spec,
    build_scenario_rollouts_spec,
    scenario_id_from_value,
    scenario_rollouts_proto_from_spec,
    submission_proto_from_specs,
)


CURRENT_TIME_INDEX_FALLBACK = 10
DEFAULT_ROLLOUT_COUNT = 32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export SMART rollouts as Waymo ScenarioRollouts/submission protobufs.")
    parser.add_argument("--smart-repo-dir", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--pretrain-ckpt", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--rollout-count", type=int, default=DEFAULT_ROLLOUT_COUNT)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--scenario-proto-path", type=str, default="")
    parser.add_argument("--scenario-proto-dir", type=str, default="")
    parser.add_argument("--scenario-tfrecords", type=str, default="")
    parser.add_argument("--strict-validation", action="store_true")
    return parser.parse_args()


def _safe_version(package_name: str) -> str:
    try:
        return str(metadata.version(package_name))
    except Exception:
        return "not_installed"


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _prepare_imports(smart_repo_dir: Path, repo_root: Path) -> None:
    smart_repo = str(smart_repo_dir)
    if smart_repo not in sys.path:
        sys.path.insert(0, smart_repo)
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


def _build_dataset(config_path: str, *, smart_repo_dir: Path) -> tuple[Any, Any, Any, Any]:
    _prepare_imports(smart_repo_dir, repo_root=Path(__file__).resolve().parents[1])
    from smart.datasets.scalable_dataset import MultiDataset  # pylint: disable=import-outside-toplevel
    from smart.model import SMART  # pylint: disable=import-outside-toplevel
    from smart.transforms import WaymoTargetBuilder  # pylint: disable=import-outside-toplevel
    from smart.utils.config import load_config_act  # pylint: disable=import-outside-toplevel

    resolved_config_path = normalize_path_value(config_path, base_dir=smart_repo_dir)
    config = load_config_act(str(resolved_config_path))
    config = normalize_dataset_paths(config, smart_repo_dir=smart_repo_dir)
    data_config = config.Dataset
    model_config = config.Model
    dataset = MultiDataset(
        root=data_config.root,
        split="val",
        raw_dir=data_config.val_raw_dir,
        processed_dir=data_config.val_processed_dir,
        transform=WaymoTargetBuilder(model_config.num_historical_steps, model_config.decoder.num_future_steps),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=data_config.batch_size,
        shuffle=False,
        num_workers=data_config.num_workers,
        pin_memory=data_config.pin_memory,
        persistent_workers=True if data_config.num_workers > 0 else False,
    )
    model = SMART(model_config)
    return config, dataset, dataloader, model


def _prepare_batch(model: Any, batch: Any) -> Any:
    batch = model.match_token_map(batch)
    batch = model.sample_pt_pred(batch)
    if isinstance(batch, Batch):
        batch["agent"]["av_index"] += batch["agent"]["ptr"][:-1]
    return batch


def _resolve_current_step(model: Any) -> int:
    hist_steps = int(getattr(model, "num_historical_steps", 0) or 0)
    if hist_steps > 0:
        return hist_steps - 1
    return CURRENT_TIME_INDEX_FALLBACK


def _run_single_rollout(
    *,
    model: Any,
    raw_batch: Any,
    device: torch.device,
    rollout_seed: int,
) -> tuple[str, Dict[str, Any]]:
    _seed_all(rollout_seed)
    batch = copy.deepcopy(raw_batch)
    if isinstance(batch, Batch) and int(getattr(batch, "num_graphs", 1)) != 1:
        raise ValueError(
            "smart_rollout_export.py currently expects validation batch_size=1; "
            f"received num_graphs={int(getattr(batch, 'num_graphs', -1))}."
        )
    if hasattr(batch, "to"):
        batch = batch.to(device)
    batch = _prepare_batch(model, batch)
    with torch.inference_mode():
        pred = model.inference(batch)

    scenario_id = scenario_id_from_value(batch["scenario_id"])
    if not scenario_id:
        raise ValueError("Missing scenario_id in SMART validation batch.")

    current_step = _resolve_current_step(model)
    joint_scene = build_joint_scene_spec(
        pred_xy=pred["pred_traj"],
        pred_heading=pred["pred_head"],
        position_history=batch["agent"]["position"],
        valid_history=batch["agent"]["valid_mask"],
        object_ids=batch["agent"]["id"],
        current_step=current_step,
    )
    return scenario_id, joint_scene


def _load_validation_scenarios(
    *,
    scenario_ids: Sequence[str],
    scenario_proto_path: str,
    scenario_proto_dir: str,
    scenario_tfrecords: str,
) -> Dict[str, Any]:
    if not (str(scenario_proto_path).strip() or str(scenario_proto_dir).strip() or str(scenario_tfrecords).strip()):
        return {}

    from waymo_open_dataset.protos import scenario_pb2  # pylint: disable=import-outside-toplevel
    from src.workflows.wosac_official_metrics import _load_scenarios  # pylint: disable=import-outside-toplevel

    return _load_scenarios(
        required_ids=scenario_ids,
        scenario_pb2=scenario_pb2,
        scenario_proto_path=scenario_proto_path,
        scenario_proto_dir=scenario_proto_dir,
        scenario_tfrecords=scenario_tfrecords,
    )


def _validate_rollouts(
    *,
    rollout_specs: Sequence[Dict[str, Any]],
    scenarios_by_id: Dict[str, Any],
) -> int:
    if not scenarios_by_id:
        return 0
    from waymo_open_dataset.protos import sim_agents_submission_pb2  # pylint: disable=import-outside-toplevel
    from waymo_open_dataset.utils.sim_agents import submission_specs  # pylint: disable=import-outside-toplevel

    validated = 0
    for spec in rollout_specs:
        sid = str(spec.get("scenario_id", "")).strip()
        scenario = scenarios_by_id.get(sid)
        if scenario is None:
            continue
        proto = scenario_rollouts_proto_from_spec(sim_agents_submission_pb2, spec)
        submission_specs.validate_scenario_rollouts(proto, scenario)
        validated += 1
    return validated


def _write_submission(*, rollout_specs: Sequence[Dict[str, Any]], output_path: Path) -> None:
    from waymo_open_dataset.protos import sim_agents_submission_pb2  # pylint: disable=import-outside-toplevel

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = submission_proto_from_specs(sim_agents_submission_pb2, rollout_specs)
    output_path.write_bytes(payload.SerializeToString())


def main() -> int:
    args = parse_args()
    smart_repo_dir = Path(args.smart_repo_dir).expanduser().resolve()
    output_path = Path(args.output_path).expanduser()

    try:
        _prepare_imports(smart_repo_dir, repo_root=REPO_ROOT)
        _seed_all(int(args.seed))
        config, dataset, dataloader, model = _build_dataset(args.config, smart_repo_dir=smart_repo_dir)
        current_step = _resolve_current_step(model)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        from smart.utils.log import Logging  # pylint: disable=import-outside-toplevel

        logger = Logging().log(level="DEBUG")
        model.load_params_from_file(
            filename=args.pretrain_ckpt,
            logger=logger,
            to_cpu=(device.type == "cpu"),
        )
        model = model.to(device)
        model.eval()
        model.inference_token = True

        rollout_specs: List[Dict[str, Any]] = []
        scenario_ids: List[str] = []
        for scenario_index, raw_batch in enumerate(dataloader):
            scenario_joint_scenes: List[Dict[str, Any]] = []
            scenario_id = ""
            for rollout_idx in range(int(args.rollout_count)):
                rollout_seed = int(args.seed) + scenario_index * 1000 + rollout_idx
                scenario_id, joint_scene = _run_single_rollout(
                    model=model,
                    raw_batch=raw_batch,
                    device=device,
                    rollout_seed=rollout_seed,
                )
                scenario_joint_scenes.append(joint_scene)
            rollout_specs.append(
                build_scenario_rollouts_spec(
                    scenario_id=scenario_id,
                    rollout_joint_scenes=scenario_joint_scenes,
                )
            )
            scenario_ids.append(scenario_id)

        scenarios_by_id = _load_validation_scenarios(
            scenario_ids=scenario_ids,
            scenario_proto_path=args.scenario_proto_path,
            scenario_proto_dir=args.scenario_proto_dir,
            scenario_tfrecords=args.scenario_tfrecords,
        )
        if args.strict_validation and scenario_ids:
            missing = [sid for sid in scenario_ids if sid not in scenarios_by_id]
            if missing:
                raise ValueError(f"Missing scenario protos required for strict validation: {missing}")
        validated_scenarios = _validate_rollouts(rollout_specs=rollout_specs, scenarios_by_id=scenarios_by_id)
        _write_submission(rollout_specs=rollout_specs, output_path=output_path)

        summary = {
            "status": "ok",
            "output_path": str(output_path),
            "output_exists": output_path.exists(),
            "output_size": output_path.stat().st_size if output_path.exists() else 0,
            "scenario_count": len(rollout_specs),
            "scenario_ids": scenario_ids,
            "rollout_count": int(args.rollout_count),
            "current_time_index": int(current_step),
            "validated_scenarios": int(validated_scenarios),
            "strict_validation": bool(args.strict_validation),
            "config": str(args.config),
            "checkpoint": str(args.pretrain_ckpt),
            "device": str(device),
            "torch": _safe_version("torch"),
            "torch_geometric": _safe_version("torch-geometric"),
            "waymo_open_dataset": _safe_version("waymo-open-dataset-tf-2-12-0"),
        }
        print("[smart-rollout-export] export complete")
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        debug_payload = {
            "smart_repo_dir": str(smart_repo_dir),
            "config": str(args.config),
            "pretrain_ckpt": str(args.pretrain_ckpt),
            "output_path": str(output_path),
            "rollout_count": int(args.rollout_count),
            "seed": int(args.seed),
            "python_version": str(sys.version.split()[0]),
            "torch": _safe_version("torch"),
            "torch_geometric": _safe_version("torch-geometric"),
            "waymo_open_dataset": _safe_version("waymo-open-dataset-tf-2-12-0"),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        print("[smart-rollout-export] export failed")
        print(json.dumps(debug_payload, indent=2, sort_keys=True))
        traceback.print_exc()
        raise


if __name__ == "__main__":
    raise SystemExit(main())
