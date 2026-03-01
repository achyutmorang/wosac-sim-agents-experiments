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

## Results Artifacts
- `experiments/smart-baseline/results/README.md`
- `experiments/smart-baseline/results/smart_baseline_v0_metrics.json`

## Reproducibility Pack
- `experiments/smart-baseline/reproducibility.md`
- strict env lock: `experiments/smart-baseline/env/requirements-smart-exact-cu113.txt`

Quick dry-run:
```bash
python3 scripts/run_smart_baseline.py \
  --config configs/experiments/smart-baseline.json \
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
