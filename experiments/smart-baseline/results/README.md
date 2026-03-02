# SMART Baseline Results

This folder stores SMART-wrapper baseline metrics and reproducibility artifacts.

## Files
- `smart_baseline_v0_metrics.json`: canonical SMART wrapper baseline snapshot.
- `smart_baseline_train_v0.json`: training-stage snapshot including profile, checkpoint dir, and command plan.

## Update rule
Only update `smart_baseline_v0_metrics.json` from:
- `experiments/smart-baseline/notebooks/smart-baseline_colab.ipynb`

The artifact must include:
- `run_tag`
- `repo_commit`
- `config_hash`
- `smart_train_seed`
- checkpoint hashes (`checkpoint_manifest`)
- WOSAC-aligned metrics when official evaluator output is available
