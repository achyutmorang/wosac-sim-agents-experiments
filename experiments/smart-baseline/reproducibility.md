# SMART Reproducibility Patch Set

This note captures current upstream reproducibility risks and how this wrapper pack handles them.

## Upstream Risks (Observed)
1. Evaluation command mismatch:
`README.md` shows `python eval.py ...`, but repository exposes `val.py` and no `eval.py`.
2. Checkpoint availability:
upstream README discusses pretrain release intent, but no checkpoints are bundled in the public repo.
3. Rigid legacy dependency stack:
the upstream setup expects `PyTorch 1.12.x + CUDA 11.3 + Linux`, plus PyG wheels bound to that ABI.
4. Demo-first default train config:
`configs/train/train_scalable.yaml` defaults to `data/valid_demo` with `total_steps: 32`.
5. No upstream automated tests:
correctness depends on manual command execution.

## Patch Set in This Repo
1. Corrected evaluation path in wrapper command plan:
`src/workflows/smart_baseline_flow.py` generates `python val.py ...`.
2. One-command orchestration script:
`scripts/run_smart_baseline.py` supports dry-run planning, stage selection, and execution.
3. Strict dependency lock for exact upstream profile:
`experiments/smart-baseline/env/requirements-smart-exact-cu113.txt`.
4. Deterministic training launcher around upstream `train.py`:
`scripts/smart_train_repro.py` sets fixed seed and deterministic flags, then calls SMART `train.py` unchanged.
5. Pinned upstream commit support:
`smart.repo_commit` in `configs/experiments/smart-baseline.json` locks SMART checkout revision.
6. Profiled configs:
`smart.profiles.smoke` (demo) and `smart.profiles.paper_repro` (full WOMD-processed paths + non-smoke budget).
7. Run manifests:
baseline flow writes data split manifests and checkpoint SHA256 manifests into run outputs.
8. Smoke tests for command behavior:
`tests/test_smart_baseline_flow.py` and `tests/test_run_smart_baseline_script.py`.

## Recommended Execution Modes
1. Inspect mode (Colab-first):
use notebook + dry-run command plan and optional stage execution.
2. Paper-repro baseline mode (Linux/CUDA11.3):
use `--profile paper_repro` so run uses pinned SMART commit, strict lockfile, deterministic seed, and paper-repro configs.

## One-Command Examples
Dry-run only (safe):
```bash
python3 scripts/run_smart_baseline.py \
  --config configs/experiments/smart-baseline.json \
  --profile paper_repro \
  --no-sync-smart-repo \
  --print-only
```

Train + validate (when environment and data are ready):
```bash
python3 scripts/run_smart_baseline.py \
  --config configs/experiments/smart-baseline.json \
  --profile paper_repro \
  --sync-smart-repo \
  --setup --preprocess --train --validate
```
