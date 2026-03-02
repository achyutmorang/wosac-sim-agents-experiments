# SMART Constrained Probabilistic Variant

## Objective
Train/evaluate a constrained probabilistic variant over SMART baseline and compare on WOSAC metrics.

## Notebook
- Open in Colab: https://colab.research.google.com/github/achyutmorang/wosac-sim-agents-experiments/blob/main/experiments/smart-constrained/notebooks/smart-constrained_colab.ipynb
- `experiments/smart-constrained/notebooks/smart-constrained_colab.ipynb`
- Open simulation in Colab: https://colab.research.google.com/github/achyutmorang/wosac-sim-agents-experiments/blob/main/experiments/smart-constrained/notebooks/smart_rollout_simulation_colab.ipynb
- Simulation notebook: `experiments/smart-constrained/notebooks/smart_rollout_simulation_colab.ipynb`
- Open checkpoint eval in Colab: https://colab.research.google.com/github/achyutmorang/wosac-sim-agents-experiments/blob/main/experiments/smart-constrained/notebooks/smart_checkpoint_eval_colab.ipynb
- Checkpoint eval notebook: `experiments/smart-constrained/notebooks/smart_checkpoint_eval_colab.ipynb`
- Open comparative eval in Colab: https://colab.research.google.com/github/achyutmorang/wosac-sim-agents-experiments/blob/main/experiments/smart-constrained/notebooks/smart_comparative_eval_colab.ipynb
- Comparative eval notebook: `experiments/smart-constrained/notebooks/smart_comparative_eval_colab.ipynb`

## Workflow Entrypoint
- `src/workflows/smart_constrained_flow.py`

## Config
- `configs/experiments/smart-constrained.json`

## Results Artifacts
- `experiments/smart-constrained/results/README.md`
- `experiments/smart-constrained/results/smart_constrained_v0_metrics.json`
- `experiments/smart-constrained/results/smart_rollout_simulation_v0.json`
- `experiments/smart-constrained/results/smart_checkpoint_eval_v0.json`
- `experiments/smart-constrained/results/smart_comparative_v0_report.json`

## Metrics Ingestion Convention
Set environment variable before Step 3:
- `WOSAC_VARIANT_METRICS_DIR`: directory containing per-variant JSON files named `<variant_id>.json`

Simulation notebook uses:
- `SMART_BASELINE_CKPT`: baseline checkpoint path
- `SMART_VARIANT_CKPTS`: comma-separated variant checkpoint paths
- `WOSAC_SCENARIO_SET_ID`: scenario split identifier (e.g. `womd_validation`)
- `WOSAC_SCENARIO_SET_HASH`: immutable scenario-set hash you use for this run
- `WOSAC_EVALUATOR_ID`: evaluator implementation identifier
- `WOSAC_METRICS_CONFIG_HASH`: metrics config hash/tag
- `WOSAC_SIM_MANIFESTS_DIR`: directory where per-model simulation manifests are written

Checkpoint eval notebook uses:
- `WOSAC_MODEL_METRICS_DIR`: directory with `<model_id>.json` metric files
- `WOSAC_SIM_MANIFESTS_DIR`: directory with `<model_id>_simulation_manifest.json`

Each metrics JSON should include binding keys (`manifest_sha256`, `model_id`, scenario/evaluator/config hashes, rollout settings) so strict contract checks can verify provenance.

The workflow selects the best feasible variant by:
1. safety constraints (`collision`, `offroad`, `traffic-light` bounds), then
2. highest `realism_meta_metric` among feasible variants.
