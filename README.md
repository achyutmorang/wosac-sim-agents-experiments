# WOSAC Sim Agents Experiments

A research workspace for reproducible simulation-model development on the Waymo Open Sim Agents Challenge (WOSAC).

## Motivation

This repository exists to solve one practical research problem: how to move from paper-level ideas to measurable, benchmark-aligned evidence without getting lost in unstructured experimentation. The project is designed around three requirements:

1. Reproducibility: every run should be traceable by config, code revision, and saved artifacts.
2. Comparability: baseline and variant models must be evaluated under the same protocol.
3. Iteration speed: model changes should be isolated, testable, and reversible.

## Research Goal

The current research direction is:

- establish a stable SMART baseline,
- introduce constrained probabilistic variants,
- improve `realism_meta_metric` while preventing regressions on key safety rates (collision, offroad, traffic-light violations).

The benchmark task and evaluator contract are fixed by WOSAC. Novelty comes from method design, optimization strategy, and controlled ablation.

## Current Status

As of March 2, 2026, the repository contains three active experiment tracks:

1. `wosac-baseline`
- benchmark contract, first baseline pipeline, literature synthesis, and reference artifacts.

2. `smart-baseline`
- SMART wrapper workflow with reproducibility patch set (`val.py` path correction, strict env lock option, smoke tests, one-command runner).

3. `smart-constrained`
- constrained variant sweep workflow,
- separated checkpoint evaluation workflow,
- separated comparative evaluation workflow with feasibility-first model selection.

Repository test status: `17 passed` (`PYTHONPATH=. pytest -q`).

## Repository Structure

- `src/workflows/`: experiment workflows and orchestration APIs.
- `src/platform/`: runtime/bootstrap utilities.
- `configs/experiments/`: experiment configs.
- `experiments/`: per-track assets (notebooks, results, references, docs).
- `tests/`: regression and workflow tests.
- `notes/`: research reasoning and formal problem statements.

## Experiment Tracks

### `wosac-baseline`
- path: `experiments/wosac-baseline/`
- purpose: benchmark grounding and first reproducible baseline contract.
- key docs:
  - `experiments/wosac-baseline/experiment_contract.md`
  - `experiments/wosac-baseline/lit_survey.md`
  - `experiments/wosac-baseline/smart_centered_implementation_comparison.md`

### `smart-baseline`
- path: `experiments/smart-baseline/`
- purpose: reproduce SMART behavior in this pipeline and generate baseline reference artifacts.
- key files:
  - `src/workflows/smart_baseline_flow.py`
  - `scripts/run_smart_baseline.py`
  - `experiments/smart-baseline/reproducibility.md`

### `smart-constrained`
- path: `experiments/smart-constrained/`
- purpose: implement constrained probabilistic variants and compare against SMART baseline.
- key files:
  - `src/workflows/smart_constrained_flow.py`
  - `src/workflows/smart_eval_flow.py`
  - `experiments/smart-constrained/results/`

## Evaluation Strategy

1. Freeze baseline configuration and checkpoint.
2. Train variant models under matched data/compute protocol.
3. Run checkpoint-level evaluation.
4. Run comparative analysis with explicit feasibility constraints.
5. Accept a variant only if primary realism improves without safety regression.

## Creating a New Experiment Pack

```bash
python3 scripts/new_experiment.py \
  --slug my-variant \
  --title "My Variant" \
  --objective "Test one controlled change against smart-baseline"
```

Generated assets include config, workflow stub, notebook scaffold, and experiment README.

## References

- Leaderboard and code availability snapshot: `references/leaderboard_2025.md`
- SMART context and positioning: `references/smart.md`
- Public reference repo fetch script: `scripts/fetch_public_repos.sh`

## Run Tests

```bash
pip install -r requirements-dev.txt
PYTHONPATH=. pytest -q
```
