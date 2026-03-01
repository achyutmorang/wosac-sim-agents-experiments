# Main Research Idea From First Principles

Date: March 1, 2026

## 1. Problem Formulation (First Principles)

### 1.1 What the benchmark actually asks
Given 1 second of history plus map context, generate **32 joint futures** over **8 seconds** for all valid agents in a scene.

### 1.2 What is optimized
The official ranking objective is `realism_meta_metric` (higher is better), which is a weighted aggregation over kinematic, interaction, and map-adherence likelihood terms.

### 1.3 Consequence
A method is only useful if it improves the official metric bundle:
- `realism_meta_metric` (primary)
- `kinematic_metrics`, `interactive_metrics`, `map_based_metrics` (bucket diagnostics)
- `min_ade`, `simulated_collision_rate`, `simulated_offroad_rate`, `simulated_traffic_light_violation_rate` (safety/quality diagnostics)

### 1.4 Core scientific tension
To score well, a model must preserve:
1. Marginal realism (single-agent motion statistics)
2. Joint realism (interaction consistency)
3. Closed-loop stability (errors must not snowball under rollout)

This is the central difficulty of SimAgents.

## 2. Main Research Idea

### 2.1 Thesis of the idea
Use a **benchmark-first, baseline-first** strategy:
1. Reproduce a credible WOSAC baseline end-to-end.
2. Borrow proven architectural priors from public strong methods (especially SMART-like tokenized modeling, TrafficBots family, and UniMM insights).
3. Introduce one controlled change at a time and accept/reject it strictly by official metrics.

### 2.2 Why this is right for us
- It eliminates vague framing drift.
- It enforces measurable progress.
- It is robust for beginner-level research execution.

## 3. Methodology (Operational)

### Stage A: Reproducible Baseline
- Freeze one config (`configs/experiments/wosac-baseline.json`).
- Produce repeatable baseline metrics and manifests.
- Confirm variance is small across reruns.

### Stage B: Controlled Variant Ladder
Test exactly one knob per run:
1. **Representation**: tokenization granularity / state encoding
2. **Dynamics learning**: objective or conditioning changes
3. **Sampling/control**: decoding, reranking, post-training policy improvements

### Stage C: Evaluation Discipline
Each run logs:
- commit hash
- config hash
- runtime profile
- full metric bundle
- pass/fail decision vs baseline

A change is kept only if it improves primary metric without unacceptable safety regression.

## 4. First-Principles Guardrails

1. No multi-track branching before one strong baseline line is stable.
2. No novelty claims without metric deltas.
3. No hidden notebook logic; reusable code belongs in `src/`.
4. Every expensive run must be resume-safe (Colab interruption tolerant).

## 5. Popular Repositories Attempting SimAgents/WOSAC

The list below is a curated snapshot of public, commonly referenced repositories as of March 1, 2026.

## 5.1 Core official infrastructure (must-use)
| Repository | Why it matters | Stars |
|---|---|---:|
| [waymo-research/waymo-open-dataset](https://github.com/waymo-research/waymo-open-dataset) | Official challenge tooling, tutorial notebook, evaluator interfaces | 3241 |
| [waymo-research/waymax](https://github.com/waymo-research/waymax) | Official JAX simulator used by many challenge pipelines | 1040 |

## 5.2 Popular public method repositories (directly relevant)
| Repository | Relevance to SimAgents | Stars |
|---|---|---:|
| [rainmaker22/SMART](https://github.com/rainmaker22/SMART) | SMART method (token-based multi-agent generation), widely used reference | 247 |
| [zhejz/TrafficBots](https://github.com/zhejz/TrafficBots) | Strong world-model baseline lineage for multi-agent simulation | 72 |
| [zhejz/TrafficBotsV1.5](https://github.com/zhejz/TrafficBotsV1.5) | Explicitly reported as 3rd place Waymo Open Sim Agent Challenge 2024 | 39 |
| [wangwenxi-handsome/Joint-Multipathpp](https://github.com/wangwenxi-handsome/Joint-Multipathpp) | Sim Agent competition 2023 solution-style codebase | 31 |
| [Longzhong-Lin/UniMM](https://github.com/Longzhong-Lin/UniMM) | WOSAC 2025 honorable mention paper repo | 21 |

## 5.3 Additional challenge-attempt repositories (lower adoption)
| Repository | Note | Stars |
|---|---|---:|
| [hansungkim98122/Sim-Agent-SMART-RL](https://github.com/hansungkim98122/Sim-Agent-SMART-RL) | 2025 WOMD Sim Agents challenge attempt combining SMART + RL | 2 |
| [vita-student-projects/Sim_agents_pytorch](https://github.com/vita-student-projects/Sim_agents_pytorch) | Pytorch implementation around official tutorial flow | 0 |
| [neumyor/TrajTokenization](https://github.com/neumyor/TrajTokenization) | Trajectory tokenization repository, potentially related to TrajTok-style ideas | 0 |

## 5.4 Public project pages / partial releases from top 2025 methods
| Repository | Status |
|---|---|
| [projrlftsim/projrlftsim.github.io](https://github.com/projrlftsim/projrlftsim.github.io) | RLFTSim project page repository, not full training/inference code |
| [rlftsim/rlftsim.github.io](https://github.com/rlftsim/rlftsim.github.io) | RLFTSim project page template/history |

## 5.5 Missing official code (important)
- TrajTok (1st, WOSAC 2025): no clearly linked official full code repository yet.
- comBOT (3rd, WOSAC 2025): no clearly linked official full code repository yet.

## 6. Decision Rule for Our Work
For our repository, “main idea progress” means:
- baseline reproduced,
- one controlled variant tested,
- official metric movement explained,
- clear keep/rollback decision documented.

This keeps the research path rigorous and thesis-defensible.
