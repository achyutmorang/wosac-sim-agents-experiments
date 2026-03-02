# WOSAC Sim Agents Experiments

Reproducible research repository for WOSAC simulation modeling.

## Motivation

This project is built to turn benchmark ideas into measurable evidence quickly and cleanly:

- fix one baseline,
- introduce controlled variants,
- compare under identical evaluation protocol,
- keep every run traceable by config + artifacts.

## Main Research Goal

Establish a stable SMART baseline, then develop constrained probabilistic variants that improve `realism_meta_metric` without regressing safety-critical rates (`collision`, `offroad`, `traffic-light violation`).

## Current Status

Active tracks:

1. `wosac-baseline`: benchmark contract, survey, reference artifacts.
2. `smart-baseline`: SMART wrapper + reproducibility patch set.
3. `smart-constrained`: constrained variant sweep + separate simulation, strict eval, and comparative eval.

Latest test status: `20 passed` (`PYTHONPATH=. pytest -q`).

## Key Paths

- Workflows: `src/workflows/`
- Experiment configs: `configs/experiments/`
- Experiment packs: `experiments/`
- Research notes: `notes/`
- Tests: `tests/`

## Recommended Run Order

1. Run `smart-baseline` to freeze baseline checkpoint/metrics.
2. Run `smart-constrained` training/sweep.
3. Run rollout simulation and write per-model manifests.
4. Run strict checkpoint evaluation with metrics-manifest binding checks.
5. Run comparative evaluation and select feasible, contract-compatible best variant.

## New Experiment Scaffold

```bash
python3 scripts/new_experiment.py \
  --slug my-variant \
  --title "My Variant" \
  --objective "Test one controlled change against smart-baseline"
```

## References

- `references/leaderboard_2025.md`
- `references/smart.md`

## Tests

```bash
pip install -r requirements-dev.txt
PYTHONPATH=. pytest -q
```
