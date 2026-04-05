#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.workflows.smart_visualization import DEFAULT_SELECTION_STRATEGY, render_smart_rollout_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a SMART rollout visualization MP4 and save it to Drive.")
    parser.add_argument("--scenario-rollouts-path", type=str, required=True)
    parser.add_argument("--output-mp4", type=str, required=True)
    parser.add_argument("--official-metrics-json", type=str, default="")
    parser.add_argument("--scenario-id", type=str, default="")
    parser.add_argument("--selection-strategy", type=str, default=DEFAULT_SELECTION_STRATEGY)
    parser.add_argument("--selection-limit", type=int, default=5)
    parser.add_argument("--rollout-index", type=int, default=0)
    parser.add_argument("--focal-object-id", type=int, default=0)
    parser.add_argument("--scenario-proto-path", type=str, default="")
    parser.add_argument("--scenario-proto-dir", type=str, default="")
    parser.add_argument("--scenario-tfrecords", type=str, default="")
    parser.add_argument("--smart-repo-dir", type=str, default="")
    parser.add_argument("--smart-config-path", type=str, default="")
    parser.add_argument("--pretrain-ckpt", type=str, default="")
    parser.add_argument("--processed-root", type=str, default="")
    parser.add_argument("--processed-split", type=str, default="validation")
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--view-radius", type=float, default=0.0)
    parser.add_argument("--skip-token-trace", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        summary = render_smart_rollout_video(
            scenario_rollouts_path=args.scenario_rollouts_path,
            output_mp4=args.output_mp4,
            official_metrics_json=args.official_metrics_json,
            scenario_id=args.scenario_id,
            selection_strategy=args.selection_strategy,
            selection_limit=args.selection_limit,
            rollout_index=args.rollout_index,
            focal_object_id=args.focal_object_id,
            scenario_proto_path=args.scenario_proto_path,
            scenario_proto_dir=args.scenario_proto_dir,
            scenario_tfrecords=args.scenario_tfrecords,
            smart_repo_dir=args.smart_repo_dir,
            smart_config_path=args.smart_config_path,
            pretrain_ckpt=args.pretrain_ckpt,
            processed_root=args.processed_root,
            processed_split=args.processed_split,
            seed=args.seed,
            fps=args.fps,
            dpi=args.dpi,
            view_radius=args.view_radius,
            skip_token_trace=bool(args.skip_token_trace),
        )
        print("[smart-rollout-visualize] render complete")
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        payload = {
            "error_type": type(exc).__name__,
            "error": str(exc),
            "scenario_rollouts_path": args.scenario_rollouts_path,
            "official_metrics_json": args.official_metrics_json,
            "scenario_id": args.scenario_id,
            "selection_strategy": args.selection_strategy,
            "rollout_index": args.rollout_index,
            "output_mp4": args.output_mp4,
        }
        print("[smart-rollout-visualize] render failed")
        print(json.dumps(payload, indent=2, sort_keys=True))
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
