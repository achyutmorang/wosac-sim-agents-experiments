# WOSAC Sim Agents Experiments

Focused experimentation repository for the Waymo Open Sim Agents Challenge (WOSAC), built to avoid diffuse problem framing and maximize reproducible leaderboard progress.

## Why This Pivot
Your previous repository proved strong engineering discipline, but spread effort across multiple broad tracks (`closedloop-core`, `surprise-potential`, `risk-uq-suite`). This repo keeps one fixed target:

- Task: WOSAC-style multi-agent closed-loop simulation.
- Objective: improve benchmark metrics with rigorous baselines.
- Process: fast, repeatable experiment cycles with clear stop/go criteria.

## Fixed Problem Definition
Given 1s history + map context, generate 32 joint 8s futures for all valid objects, following Sim Agents submission rules.

Primary benchmark outputs (from official evaluator):
- `realism_meta_metric` (higher is better; official ranking score)
- `kinematic_metrics`
- `interactive_metrics`
- `map_based_metrics`

Secondary diagnostics:
- `min_ade`
- `simulated_collision_rate`
- `simulated_offroad_rate`
- `simulated_traffic_light_violation_rate`

Official metric code and config are in the Waymo Open Dataset repository and linked in [references/leaderboard_2025.md](references/leaderboard_2025.md).

## Repository Layout
- `experiments/`: experiment packs with one objective each.
- `configs/experiments/`: central runtime and baseline configs.
- `references/`: leaderboard methods, reports, and code-status tracking.
- `notes/`: research decisions, failures, and thesis-ready reflections.
- `scripts/`: small automation scripts for scaffolding and repo sync.
- `src/experiments/`: reusable experiment-pack scaffolding helpers.

## Current Pack
- `wosac-baseline`: minimal reproducible baseline to establish your first credible score before attempting novel modifications.

## Week-1 Execution Plan
1. Reproduce a complete submission flow from the official Waymo tutorial.
2. Freeze one baseline config in `configs/experiments/wosac-baseline.json`.
3. Record first metric snapshot in `notes/` with date and commit hash.
4. Change one variable at a time (architecture, rollout policy, or training objective).
5. Keep/rollback changes strictly based on benchmark deltas.

## Leaderboard Method Tracking
Top-2025 methods and code availability status are tracked in [references/leaderboard_2025.md](references/leaderboard_2025.md).

## SMART and TrajTok Context
A compact SMART-oriented reading and implementation map is tracked in [references/smart.md](references/smart.md).

## Quick Start
```bash
cd wosac-sim-agents-experiments
python3 scripts/new_experiment.py \
  --slug my-variant \
  --title "My Variant" \
  --objective "Test one concrete change against wosac-baseline"
```

## Scope Guardrails
- No new research direction unless baseline is stable and measurable.
- No multi-track branching until one main track has repeatable gains.
- No claims without metric deltas on official challenge outputs.
