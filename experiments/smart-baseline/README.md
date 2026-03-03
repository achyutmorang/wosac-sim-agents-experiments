# SMART Baseline Wrapper

## Objective
Reproduce SMART baseline with a thin wrapper and evaluate under WOSAC-aligned reporting.

## Notebook
- Open in Colab: https://colab.research.google.com/github/achyutmorang/wosac-sim-agents-experiments/blob/main/experiments/smart-baseline/notebooks/smart-baseline_colab.ipynb
- `experiments/smart-baseline/notebooks/smart-baseline_colab.ipynb`

## Workflow Entrypoint
- `src/workflows/smart_baseline_flow.py`
- `scripts/run_smart_baseline.py` (one-command runner)

## Config
- `configs/experiments/smart-baseline.json`
- `experiments/smart-baseline/configs/train_scalable_paper_repro.yaml`
- `experiments/smart-baseline/configs/validation_scalable_paper_repro.yaml`

## Artifact Location
- Persist artifacts to Google Drive under:
  - `/content/drive/MyDrive/wosac_experiments/<run_prefix>_<run_name>/outputs/`
  - run-scoped outputs: `/content/drive/MyDrive/wosac_experiments/<run_prefix>_<run_name>/outputs/<run_tag>/`

## Reproducibility Pack
- `experiments/smart-baseline/reproducibility.md`
- strict env lock: `experiments/smart-baseline/env/requirements-smart-exact-cu113.txt`
- deterministic launcher: `scripts/smart_train_repro.py`

## Profiles
- `smoke`: quick sanity mode using SMART demo config (`data/valid_demo`).
- `paper_repro`: pinned SMART commit + fixed seed + full WOMD processed split config.

Notebook run modes:
- `WOSAC_RUN_MODE=pilot` (default): small GCS shard staging defaults for fast iteration.
- `WOSAC_RUN_MODE=full`: larger shard staging defaults for longer training runs.

Set profile in notebook/script:
- env var `SMART_BASELINE_PROFILE=paper_repro` (notebook)
- CLI `--profile paper_repro` (script)

## GCS Data Staging (Training Notebook)
- Default behavior stages WOMD shards from Waymo GCS to local Colab storage under `SMART.raw_data_root`.
- Key env vars:
  - `SMART_DATA_SOURCE=gcs_stage` (default)
  - `SMART_GCS_DATASET_ROOT=gs://waymo_open_dataset_motion_v_1_3_1/uncompressed/scenario`
  - `SMART_GCS_TRAIN_SPLIT=training`, `SMART_GCS_VAL_SPLIT=validation`
  - `SMART_GCS_TRAIN_SHARDS`, `SMART_GCS_VAL_SHARDS`
  - `SMART_RUN_DATA_STAGE=1` (default), `SMART_FORCE_DATA_REDOWNLOAD=0` (default)
- Raw TFRecords stay on local Colab disk; checkpoints/artifacts persist to Drive.

## Preprocess Resume Policy
- Notebook auto-detects existing processed `.pkl/.pickle` files.
- Default behavior: skip preprocessing when processed outputs already exist.
- Override with `SMART_FORCE_PREPROCESS=1`.

## Resume Behavior
- Training auto-resumes from the latest checkpoint in `SMART_BASELINE_CKPT_DIR` when `run.resume_from_existing=true` and no explicit checkpoint override is passed.
- Optional explicit override: `SMART_RESUME_CKPT=/content/drive/.../epoch=XX.ckpt`.
- Notebook writes `run_manifest.json` per run with resolved resume checkpoint, config hash, repo commits, and environment versions.

Quick dry-run:
```bash
python3 scripts/run_smart_baseline.py \
  --config configs/experiments/smart-baseline.json \
  --profile paper_repro \
  --no-sync-smart-repo \
  --print-only
```

## Optional Evaluator Ingestion
Set one environment variable before running Step 3:
- `WOSAC_OFFICIAL_METRICS_JSON`: path to evaluator JSON output
- `WOSAC_OFFICIAL_METRICS_CSV`: path to CSV with `metric,value` columns

## Notes
- This experiment is intended as an external benchmark baseline.
- Keep method changes out of this pack; use separate packs for Gap-2 variants.
