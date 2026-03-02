# WOSAC Baseline Experiment Contract

Date: March 2, 2026

## 1. Objective
Establish a reproducible WOSAC baseline run before any novelty work. The baseline must produce a valid run artifact with traceable config, code revision, and metrics schema.

## 2. Hypothesis
If we keep a deterministic Colab-first setup and evaluate with the official WOSAC metric structure, we can repeatedly generate baseline artifacts with stable safety diagnostics and no manual runtime patching.

## 3. Primary Metric
- `realism_meta_metric` (official WOSAC primary score)

## 4. Secondary Diagnostics
- `simulated_collision_rate`
- `simulated_offroad_rate`
- `simulated_traffic_light_violation_rate`
- Kinematic and interaction metric groups from the evaluator report

## 5. Baseline Definition
- Experiment slug: `wosac-baseline`
- Notebook entrypoint: `experiments/wosac-baseline/notebooks/01_colab_smoke_test.ipynb`
- Config file: `configs/experiments/wosac-baseline.json`
- Workflow entrypoint: `src/workflows/wosac_baseline_flow.py`
- Output artifact root: `/content/drive/MyDrive/wosac_experiments/<run_prefix>_<run_name>/outputs/`

## 6. Compute Budget
- Runtime: Google Colab (CPU or T4 GPU)
- Per run time budget: <= 2 hours for smoke/validation runs
- Weekly baseline budget: <= 12 runtime hours
- Storage root: `/content/drive/MyDrive/wosac_experiments`

## 7. Acceptance Gates
1. Bootstrap, repo sync, and dependency setup complete without manual shell fixes.
2. Run config validation passes (`n_shards`, `shard_id`, persistence path checks).
3. Workflow call completes and writes a metrics artifact.
4. Artifact captures `run_tag`, `git_commit`, `cfg_hash`, and metric keys.

## 8. Stop Criteria (for baseline phase)
1. Two independent Colab sessions produce valid artifacts with identical config hash.
2. No unresolved bootstrap/runtime failures across two consecutive runs.
3. Baseline artifact schema is stable and ready for ablation comparisons.

## 9. Immediate Next Ablation (single change only)
After baseline stability is proven, run one safety-dominant objective ablation focused on collision/offroad reduction while keeping all other settings fixed.
