# WOSAC Sim Agents Experiments

Colab-first repository for WOSAC research with thin notebooks, deterministic runtime bootstrap, and resumable Drive-backed artifacts.

## Why This Design
This repo intentionally follows the strongest structure from `waymax-simulation-experiments`:
- notebook orchestration only,
- reusable Python modules in `src/`,
- deterministic Colab setup,
- explicit run manifests and resume contracts.

## Colab-First Structure
- `requirements-colab.txt`: locked Colab dependency set.
- `scripts/colab_setup.py`: deterministic environment install/probe/restart handling.
- `src/platform/`: Colab bootstrap, repo sync, Drive mount, setup cache, runtime profiles.
- `notebooks/NOTEBOOK_DESIGN_CONTRACT.md`: required notebook execution contract.
- `notebooks/templates/`: starter Colab template.
- `experiments/<slug>/notebooks/`: experiment-specific Colab notebooks.
- `configs/experiments/`: runtime/config JSON per experiment.
- `src/workflows/`: notebook-facing orchestration APIs.

## Current Pack
- `wosac-baseline`
  - Open in Colab (smoke): [01_colab_smoke_test.ipynb](https://colab.research.google.com/github/achyutmorang/wosac-sim-agents-experiments/blob/main/experiments/wosac-baseline/notebooks/01_colab_smoke_test.ipynb)
  - Open in Colab (baseline): [wosac_baseline_colab.ipynb](https://colab.research.google.com/github/achyutmorang/wosac-sim-agents-experiments/blob/main/experiments/wosac-baseline/notebooks/wosac_baseline_colab.ipynb)
  - Smoke notebook: `experiments/wosac-baseline/notebooks/01_colab_smoke_test.ipynb`
  - Notebook: `experiments/wosac-baseline/notebooks/wosac_baseline_colab.ipynb`
  - Config: `configs/experiments/wosac-baseline.json`
  - Workflow: `src/workflows/wosac_baseline_flow.py`
  - Contract: `experiments/wosac-baseline/experiment_contract.md`
  - Results: `experiments/wosac-baseline/results/baseline_v0_metrics.json`
- `smart-baseline`
  - Open in Colab: [smart-baseline_colab.ipynb](https://colab.research.google.com/github/achyutmorang/wosac-sim-agents-experiments/blob/main/experiments/smart-baseline/notebooks/smart-baseline_colab.ipynb)
  - Notebook: `experiments/smart-baseline/notebooks/smart-baseline_colab.ipynb`
  - Config: `configs/experiments/smart-baseline.json`
  - Workflow: `src/workflows/smart_baseline_flow.py`
  - One-command runner: `scripts/run_smart_baseline.py`
  - Repro notes: `experiments/smart-baseline/reproducibility.md`
  - Results: `experiments/smart-baseline/results/smart_baseline_v0_metrics.json`
- `smart-constrained`
  - Open in Colab: [smart-constrained_colab.ipynb](https://colab.research.google.com/github/achyutmorang/wosac-sim-agents-experiments/blob/main/experiments/smart-constrained/notebooks/smart-constrained_colab.ipynb)
  - Notebook: `experiments/smart-constrained/notebooks/smart-constrained_colab.ipynb`
  - Checkpoint eval notebook: `experiments/smart-constrained/notebooks/smart_checkpoint_eval_colab.ipynb`
  - Comparative eval notebook: `experiments/smart-constrained/notebooks/smart_comparative_eval_colab.ipynb`
  - Config: `configs/experiments/smart-constrained.json`
  - Workflow: `src/workflows/smart_constrained_flow.py`
  - Eval workflow: `src/workflows/smart_eval_flow.py`
  - Results: `experiments/smart-constrained/results/smart_constrained_v0_metrics.json`

## Colab Run Order (Baseline)
1. Open the baseline notebook in Colab.
2. Run Step 1 cell (repo sync + runtime bootstrap).
3. If restart is requested, restart runtime and rerun Step 1.
4. Run config + fast-fail cells.
5. Execute experiment-specific cells and persist artifacts in Drive.

## Scaffolding New Experiments
```bash
python3 scripts/new_experiment.py \
  --slug my-variant \
  --title "My Variant" \
  --objective "Test one controlled change against wosac-baseline"
```

Generated files:
- `experiments/<slug>/README.md`
- `experiments/<slug>/notebooks/<slug>_colab.ipynb`
- `configs/experiments/<slug>.json`
- `src/workflows/<slug>_flow.py`
- `src/experiments/papers/<slug>/__init__.py`

## Public References
- Leaderboard/report/code status: `references/leaderboard_2025.md`
- SMART context: `references/smart.md`
- Public external repos (local clones): `./scripts/fetch_public_repos.sh`

## Local Tests
```bash
pip install -r requirements-dev.txt
PYTHONPATH=. pytest -q
```
