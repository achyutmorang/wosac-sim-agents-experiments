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
  - Default sweep is now pilot-sized (`2 x 1 x 2 = 4` variants) to reduce compute.

Notebook run modes:
- `WOSAC_RUN_MODE=pilot` (default): uses pilot sweep defaults and small shard staging defaults.
- `WOSAC_RUN_MODE=full`: expands to full-style sweep defaults (`3 x 2 x 3 = 18`) unless env overrides are set.

## Artifact Location
- Persist artifacts to Google Drive under:
  - `/content/drive/MyDrive/wosac_experiments/<run_prefix>_<run_name>/outputs/`
  - run-scoped outputs: `/content/drive/MyDrive/wosac_experiments/<run_prefix>_<run_name>/outputs/<run_tag>/`

## Metrics Ingestion Convention
Set environment variable before Step 3:
- `WOSAC_VARIANT_METRICS_DIR`: directory containing per-variant JSON files named `<variant_id>.json`

Simulation notebook uses:
- `SMART_BASELINE_CKPT`: baseline checkpoint path
- `SMART_VARIANT_CKPTS`: comma-separated variant checkpoint paths
- `SMART_BASELINE_ROLLOUTS_PROTO`: baseline `ScenarioRollouts`/submission binproto path
- `SMART_VARIANT_ROLLOUTS_PROTOS`: comma-separated rollout proto paths (aligned with variant checkpoints)
- `WOSAC_SCENARIO_SET_ID`: scenario split identifier (e.g. `womd_validation`)
- `WOSAC_SCENARIO_SET_HASH`: immutable scenario-set hash you use for this run
- `WOSAC_EVALUATOR_ID`: evaluator implementation identifier
- `WOSAC_METRICS_CONFIG_HASH`: metrics config hash/tag
- `WOSAC_SIM_MANIFESTS_DIR`: directory where per-model simulation manifests are written
- `WOSAC_SCENARIO_PROTO_PATH`: optional single scenario proto path
- `WOSAC_SCENARIO_PROTO_DIR`: optional directory of `<scenario_id>.pb` files
- `WOSAC_SCENARIO_TFRECORDS`: optional fallback TFRecord path list (comma-separated; supports local and `gs://`/glob inputs)
- `WOSAC_AUTO_RESUME=1` (default): auto-discover latest run output dirs/checkpoints/manifests
- `WOSAC_RUN_SIM_PENDING_ONLY=1` (default): execute only models whose rollout proto is missing

Checkpoint eval notebook uses:
- `WOSAC_SIM_MANIFESTS_DIR`: directory with `<model_id>_simulation_manifest.json`
- `WOSAC_MODEL_METRICS_DIR`: optional output directory for computed `<model_id>.json` metrics (auto-created)
- `WOSAC_SCENARIO_PROTO_PATH` / `WOSAC_SCENARIO_PROTO_DIR` / `WOSAC_SCENARIO_TFRECORDS`: scenario sources for official metric computation
- `WOSAC_RECOMPUTE_METRICS=0` (default): reuse existing metric JSONs when present
- `WOSAC_READY_MODELS_ONLY=1` (default): evaluate only models with available rollout protos

Metrics JSON files are computed inline in notebook via official Waymo APIs and include binding keys (`manifest_sha256`, `model_id`, scenario/evaluator/config hashes, rollout settings) so strict contract checks can verify provenance.

The workflow selects the best feasible variant by:
1. safety constraints (`collision`, `offroad`, `traffic-light` bounds), then
2. highest `realism_meta_metric` among feasible variants.

## GCS Data Staging (Training Notebook)
- Constrained training notebook stages raw WOMD shards from Waymo GCS to local Colab paths used by SMART.
- Key env vars:
  - `SMART_DATA_SOURCE=gcs_stage` (default)
  - `SMART_GCS_DATASET_ROOT=gs://waymo_open_dataset_motion_v_1_3_1/uncompressed/scenario`
  - `SMART_GCS_TRAIN_SPLIT=training`, `SMART_GCS_VAL_SPLIT=validation`
  - `SMART_GCS_TRAIN_SHARDS`, `SMART_GCS_VAL_SHARDS`
  - `SMART_RUN_DATA_STAGE=1` (default), `SMART_FORCE_DATA_REDOWNLOAD=0` (default)
- Preprocess auto-skip logic is enabled by default when processed outputs already exist; override with `SMART_FORCE_PREPROCESS=1`.
