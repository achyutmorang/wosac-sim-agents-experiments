# wosac-baseline

## Objective
Establish a reproducible baseline submission pipeline and first valid metric snapshot for the WOSAC Sim Agents benchmark.

## Notebook
- `experiments/wosac-baseline/notebooks/wosac_baseline_colab.ipynb`

## Workflow Entrypoint
- `src/workflows/wosac_baseline_flow.py`

## Config
- `configs/experiments/wosac-baseline.json`

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
