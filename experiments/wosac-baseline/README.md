# wosac-baseline

## Objective
Establish a reproducible baseline submission pipeline and first valid metric snapshot for the WOSAC Sim Agents benchmark.

## Notebook
- Open in Colab (smoke): https://colab.research.google.com/github/achyutmorang/wosac-sim-agents-experiments/blob/main/experiments/wosac-baseline/notebooks/01_colab_smoke_test.ipynb
- Open in Colab (baseline): https://colab.research.google.com/github/achyutmorang/wosac-sim-agents-experiments/blob/main/experiments/wosac-baseline/notebooks/wosac_baseline_colab.ipynb
- `experiments/wosac-baseline/notebooks/01_colab_smoke_test.ipynb`
- `experiments/wosac-baseline/notebooks/wosac_baseline_colab.ipynb`

`wosac_baseline_colab.ipynb` now follows the official Sim Agents tutorial stages:
- `ScenarioRollouts` generation + protobuf validation via `submission_specs`.
- official metrics call via `metrics.compute_scenario_metrics_for_bundle`.
- optional `SimAgentsChallengeSubmission` shard/tar packaging.
- direct `gs://` TFRecord discovery from Waymo GCS when `WOSAC_SAMPLE_TFRECORD` is not preset; notebook now auto-tries common Motion bucket roots (`v1.3.1/v1.2.0`, with and without `uncompressed/scenario`) and common split names.
- notebook prints active `gcloud` account before probing GCS; set `WOSAC_FORCE_GCS_AUTH=1` to force re-authentication in Colab.
- runtime-safe dependency bootstrap for modern Colab (Python 3.12 compatibility mode + explicit fallback guidance to Python 3.10/3.11 when required).

## Workflow Entrypoint
- `src/workflows/wosac_baseline_flow.py`

## Config
- `configs/experiments/wosac-baseline.json`

## Literature and References
- `experiments/wosac-baseline/lit_survey.md`
- `experiments/wosac-baseline/smart_centered_implementation_comparison.md`
- `experiments/wosac-baseline/references/README.md`

## Experiment Contract
- `experiments/wosac-baseline/experiment_contract.md`

## Artifact Location
- Persist artifacts to Google Drive under:
  - `/content/drive/MyDrive/wosac_experiments/<run_prefix>_<run_name>/outputs/`

## Optional Evaluator Ingestion
Set one of these environment variables before running Step 5 in the smoke notebook:
- `WOSAC_OFFICIAL_METRICS_JSON`: path to evaluator JSON output
- `WOSAC_OFFICIAL_METRICS_CSV`: path to CSV with `metric,value` columns

## Inputs
- Waymo Open Motion data access and challenge constraints.
- Official submission format (`SimAgentsChallengeSubmission`).
- Official evaluator / metric definitions.

## Minimal Deliverables
1. Successful local generation of valid rollouts.
2. Valid submission package structure.
3. One metric report archived with date, commit hash, and config hash.

## Exit Criteria
- Baseline run is reproducible on a second execution with acceptable variance.
- All secondary safety diagnostics are logged (collision/offroad/TL violation rates).
