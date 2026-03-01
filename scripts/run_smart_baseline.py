#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.workflows import run_smart_baseline_flow


def _load_config(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Config is not a mapping: {path}")
    return dict(payload)


def _bool_arg(explicit_true: bool, explicit_false: bool, default: bool) -> bool:
    if explicit_true and explicit_false:
        raise ValueError("Conflicting boolean flags were provided.")
    if explicit_true:
        return True
    if explicit_false:
        return False
    return bool(default)


def _build_flow_kwargs(config: Mapping[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    run_cfg = dict(config.get("run", {}))
    smart_cfg = dict(config.get("smart", {}))

    kwargs: Dict[str, Any] = {
        "repo_root": str(args.repo_root),
        "run_tag": args.run_tag or run_cfg.get("run_tag", ""),
        "run_name": args.run_name or run_cfg.get("run_name", "dev"),
        "run_prefix": args.run_prefix or run_cfg.get("run_prefix", "smart_baseline"),
        "persist_root": args.persist_root or run_cfg.get("persist_root", "/content/drive/MyDrive/wosac_experiments"),
        "smart_repo_url": args.smart_repo_url or smart_cfg.get("repo_url", "https://github.com/rainmaker22/SMART.git"),
        "smart_repo_branch": args.smart_repo_branch or smart_cfg.get("branch", "main"),
        "smart_repo_dir": args.smart_repo_dir or smart_cfg.get("repo_dir", "/content/SMART"),
        "smart_train_config": args.train_config or smart_cfg.get("train_config", "configs/train/train_scalable.yaml"),
        "smart_val_config": args.val_config or smart_cfg.get("val_config", "configs/validation/validation_scalable.yaml"),
        "smart_ckpt_path": args.ckpt_path if args.ckpt_path is not None else smart_cfg.get("ckpt_path", ""),
        "smart_raw_data_root": args.raw_data_root or smart_cfg.get("raw_data_root", "/content/SMART/data/waymo/scenario"),
        "smart_processed_data_root": args.processed_data_root
        or smart_cfg.get("processed_data_root", "/content/SMART/data/waymo_processed"),
        "smart_install_pyg": _bool_arg(args.install_pyg, args.no_install_pyg, bool(smart_cfg.get("install_pyg", True))),
        "sync_smart_repo": _bool_arg(args.sync_smart_repo, args.no_sync_smart_repo, True),
        "official_metrics_json": args.official_metrics_json or "",
        "metrics_csv": args.metrics_csv or "",
    }
    if not kwargs["run_tag"]:
        kwargs.pop("run_tag")
    return kwargs


def _stage_commands(plan: Mapping[str, str], args: argparse.Namespace) -> Dict[str, str]:
    run_all = bool(args.all_stages)
    do_setup = run_all or bool(args.setup)
    do_preprocess = run_all or bool(args.preprocess)
    do_train = run_all or bool(args.train)
    do_validate = run_all or bool(args.validate)

    stages: Dict[str, str] = {}
    if do_setup:
        stages["setup"] = plan["setup_cmd"]
    if do_preprocess:
        stages["preprocess_train"] = plan["preprocess_train_cmd"]
        stages["preprocess_val"] = plan["preprocess_val_cmd"]
    if do_train:
        stages["train"] = plan["train_cmd"]
    if do_validate:
        stages["validate"] = plan["validate_cmd"]
    return stages


def _run_shell(cmd: str) -> None:
    subprocess.run(["bash", "-lc", cmd], check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One-command SMART baseline runner around the wrapper workflow.",
    )
    parser.add_argument("--config", type=str, default="configs/experiments/smart-baseline.json")
    parser.add_argument("--repo-root", type=str, default=".")
    parser.add_argument("--run-tag", type=str, default="")
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--run-prefix", type=str, default="")
    parser.add_argument("--persist-root", type=str, default="")

    parser.add_argument("--smart-repo-url", type=str, default="")
    parser.add_argument("--smart-repo-branch", type=str, default="")
    parser.add_argument("--smart-repo-dir", type=str, default="")
    parser.add_argument("--train-config", type=str, default="")
    parser.add_argument("--val-config", type=str, default="")
    parser.add_argument("--ckpt-path", type=str, default=None)
    parser.add_argument("--raw-data-root", type=str, default="")
    parser.add_argument("--processed-data-root", type=str, default="")

    parser.add_argument("--sync-smart-repo", action="store_true")
    parser.add_argument("--no-sync-smart-repo", action="store_true")
    parser.add_argument("--install-pyg", action="store_true")
    parser.add_argument("--no-install-pyg", action="store_true")

    parser.add_argument("--official-metrics-json", type=str, default="")
    parser.add_argument("--metrics-csv", type=str, default="")
    parser.add_argument("--env-lockfile", type=str, default="")

    parser.add_argument("--all-stages", action="store_true")
    parser.add_argument("--setup", action="store_true")
    parser.add_argument("--preprocess", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--print-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config file: {config_path}")
    config = _load_config(config_path)

    kwargs = _build_flow_kwargs(config=config, args=args)
    bundle = run_smart_baseline_flow(**kwargs)

    print("[flow] summary")
    print(json.dumps(bundle.summary, indent=2, sort_keys=True))
    print("[flow] command_plan")
    print(json.dumps(bundle.command_plan, indent=2, sort_keys=True))

    lockfile_arg = args.env_lockfile.strip()
    if lockfile_arg:
        lockfile = Path(lockfile_arg).expanduser().resolve()
        if not lockfile.exists():
            raise FileNotFoundError(f"Missing env lockfile: {lockfile}")
        env_cmd = f"python -m pip install -r {lockfile}"
        print(f"[env] {env_cmd}")
        if not args.print_only:
            _run_shell(env_cmd)

    stages = _stage_commands(plan=bundle.command_plan, args=args)
    if not stages:
        print("[stages] none selected; generated artifacts only.")
        return 0

    print(f"[stages] selected={list(stages.keys())}")
    for stage, cmd in stages.items():
        print(f"[stage:{stage}] {cmd}")
        if args.print_only:
            continue
        _run_shell(cmd)
    print("[done] SMART baseline stages completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
