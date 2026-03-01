# Baseline Results

This folder stores versioned baseline metric artifacts used for reproducibility and thesis reporting.

## Files
- `baseline_v0_metrics.json`: canonical baseline metrics snapshot.

## Update Rule
Only overwrite `baseline_v0_metrics.json` from the notebook:
- `experiments/wosac-baseline/notebooks/01_colab_smoke_test.ipynb`

Each write must include:
- `run_tag`
- `git_commit`
- `cfg_hash`
- metric keys for primary and safety diagnostics

## Promotion Rule
Create `baseline_v1_metrics.json`, `baseline_v2_metrics.json`, etc. only after a controlled methodological change with the same config contract and explicit changelog in commit history.
