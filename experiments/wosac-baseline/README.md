# wosac-baseline

## Objective
Establish a reproducible baseline submission pipeline and first valid metric snapshot for the WOSAC Sim Agents benchmark.

## Notebook
- `experiments/wosac-baseline/notebooks/01_colab_smoke_test.ipynb`
- `experiments/wosac-baseline/notebooks/wosac_baseline_colab.ipynb`

## Workflow Entrypoint
- `src/workflows/wosac_baseline_flow.py`

## Config
- `configs/experiments/wosac-baseline.json`

## Literature and References
- `experiments/wosac-baseline/lit_survey.md`
- `experiments/wosac-baseline/references/README.md`

## Experiment Contract
- `experiments/wosac-baseline/experiment_contract.md`

## Results Artifacts
- `experiments/wosac-baseline/results/README.md`
- `experiments/wosac-baseline/results/baseline_v0_metrics.json`

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
