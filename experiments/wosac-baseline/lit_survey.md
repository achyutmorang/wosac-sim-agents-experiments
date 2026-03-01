# WOSAC Sim Agents Literature Survey

Date: March 1, 2026  
Scope: problem framing, benchmark understanding, method landscape, implementation limitations, and actionable research gaps for this repository.

## 1. Research Objective and Framing

### 1.1 Target question
How can we improve realism and safety diagnostics on the Waymo Open Sim Agents Challenge (WOSAC) while keeping experiments reproducible under Colab constraints?

### 1.2 First-principles formulation
Given scene context `c` (map + 1 second history), simulate 32 futures for all valid agents over 8 seconds:

```text
p(S_{1:T}^{ADV}, S_{1:T}^{world} | c)
```

Challenge factorization (from benchmark rules):

```text
p(S_{1:T}^{ADV}, S_{1:T}^{world} | c)
= product_t pi_ADV(S_t^{ADV} | s_{<t}^{ADV}, s_{<t}^{world}, c)
            * p(S_t^{world} | s_{<t}^{ADV}, s_{<t}^{world}, c)
```

The benchmark score is the realism meta-metric (higher is better), so our practical optimization target is not generic trajectory loss; it is the evaluator-defined realism bundle.

## 2. Benchmark Deep Dive

### 2.1 Official metric structure (2025 config)
From `challenge_2025_sim_agents_config.textproto` (Waymo evaluator):

| Feature likelihood term | Weight in meta-metric |
|---|---:|
| `linear_speed_likelihood` | 0.05 |
| `linear_acceleration_likelihood` | 0.05 |
| `angular_speed_likelihood` | 0.05 |
| `angular_acceleration_likelihood` | 0.05 |
| `distance_to_nearest_object_likelihood` | 0.10 |
| `collision_indication_likelihood` | 0.25 |
| `time_to_collision_likelihood` | 0.10 |
| `distance_to_road_edge_likelihood` | 0.05 |
| `offroad_indication_likelihood` | 0.25 |
| `traffic_light_violation_likelihood` | 0.05 |

Operational takeaway:
- Collision and offroad dominate (`0.25 + 0.25 = 0.50`).
- Interaction terms are heavily rewarded, but motion smoothness still matters.

### 2.2 Why this benchmark is hard
1. Single logged future vs simulated distribution mismatch.
2. Closed-loop compounding error over 8 seconds.
3. Joint multi-agent consistency, not independent marginal prediction.
4. Metric is likelihood-based realism proxy, not direct planning utility.

### 2.3 Benchmark limitations (important for thesis claims)
1. Realism proxy is not identical to downstream planning safety.
2. Only 32 rollout samples can underrepresent tail events.
3. Compute and implementation complexity are weakly reflected in score.
4. Metric weighting can bias optimization toward specific failure types.

## 3. Literature Taxonomy (Most Relevant to This Repo)

## 3.1 Benchmark and simulator foundations
| Work | Contribution | Limitation for our objective |
|---|---|---|
| WOSAC benchmark paper (NeurIPS D&B 2023) | Defines challenge task, evaluator framing, and imitation likelihood setup | Gives task and metric, not a method recipe |
| Waymax (2023) | High-throughput JAX simulator substrate for closed-loop AV research | Infrastructure, not a winning SimAgents method itself |

## 3.2 Token/autoregressive simulation methods
| Work | Contribution | Limitation |
|---|---|---|
| SMART (2024) | Next-token multi-agent simulation; strong open code baseline | Challenge-specific adaptation details still require heavy engineering |
| TrajTok (2025, arXiv/tech report) | Tokenized trajectory approach behind top 2025 result | Public full training/inference repo still unclear |
| BehaviorGPT (2024) | GPT-style behavior generation for driving trajectories | Benchmark transfer details to WOSAC need validation |

## 3.3 Mixture/world-model family
| Work | Contribution | Limitation |
|---|---|---|
| TrafficBots (2023) | Closed-loop world-model perspective for multi-agent simulation | Original repo focused on WOMD motion challenge setup |
| TrafficBots V1.5 (2024) | Practical Sim Agents challenge solution with improved architecture | High compute/training cost; limited turnkey reproducibility |
| UniMM (2025) | Unified mixture-model framing with closed-loop sample generation | Public repo exists, but full release status remains partial |
| Multiverse Transformer (2023) | Strong generative sequence model for autonomous driving behavior | Integration cost into WOSAC evaluator pipeline is non-trivial |

## 3.4 RL/post-training reliability directions
| Work | Contribution | Limitation |
|---|---|---|
| RLFTSim (2025 project/report lineage) | RL fine-tuning perspective for realism and controllability | Public full code release not clearly available |
| RIFT (2025) | Closed-loop RL fine-tuning evidence for traffic simulation realism/control | Not a direct drop-in WOSAC baseline package |
| Reliable Simulated Driving Agents (2025) | Emphasizes reliability and safety in simulated agents | Benchmark-specific transfer requires method adaptation |
| ForSim (2026) | New simulation direction with stronger modeling assumptions | Too recent; stability and reproducibility still uncertain |

## 4. Popular SimAgents/WOSAC Repository Landscape

Snapshot verified on March 1, 2026 (stars are point-in-time and may change).

| Repository | Role | Stars | What is useful | Main limitation |
|---|---|---:|---|---|
| [waymo-research/waymo-open-dataset](https://github.com/waymo-research/waymo-open-dataset) | Official benchmark tooling | 3241 | evaluator/tutorial/specs | Not a winning method implementation |
| [waymo-research/waymax](https://github.com/waymo-research/waymax) | Official simulator | 1040 | scalable closed-loop backend | Needs substantial method code on top |
| [rainmaker22/SMART](https://github.com/rainmaker22/SMART) | Major open method baseline | 247 | complete training/eval code path | Requires adaptation to our exact protocol |
| [zhejz/TrafficBots](https://github.com/zhejz/TrafficBots) | World-model baseline lineage | 72 | strong conceptual baseline | Original focus differs from current WOSAC setup |
| [zhejz/TrafficBotsV1.5](https://github.com/zhejz/TrafficBotsV1.5) | WOSAC 2024 challenge code | 39 | explicit sim-agent challenge pipeline | Heavy compute and setup burden |
| [wangwenxi-handsome/Joint-Multipathpp](https://github.com/wangwenxi-handsome/Joint-Multipathpp) | 2023 sim-agent competition code | 31 | challenge-oriented implementation ideas | Older stack and limited maintainability |
| [Longzhong-Lin/UniMM](https://github.com/Longzhong-Lin/UniMM) | WOSAC 2025 honorable mention repo | 21 | modern mixture-model framing | Full code release still marked “coming soon” |
| [hansungkim98122/Sim-Agent-SMART-RL](https://github.com/hansungkim98122/Sim-Agent-SMART-RL) | Challenge attempt (SMART+RL) | 2 | practical adaptation hints | Small-scale, less validated |
| [neumyor/TrajTokenization](https://github.com/neumyor/TrajTokenization) | Trajectory tokenization codebase | 0 | tokenization experimentation ideas | Not clearly official TrajTok challenge code |
| [vita-student-projects/Sim_agents_pytorch](https://github.com/vita-student-projects/Sim_agents_pytorch) | Tutorial-style implementation | 0 | simple learning scaffold | Not SOTA-oriented |
| [projrlftsim/projrlftsim.github.io](https://github.com/projrlftsim/projrlftsim.github.io) | RLFTSim project page | 1 | method overview | not full code |
| [rlftsim/rlftsim.github.io](https://github.com/rlftsim/rlftsim.github.io) | RLFTSim project page mirror | 0 | historical context | not full code |

## 5. Reproducibility and Implementation Gap Analysis

### 5.1 Gap A: leaderboard opacity
For top 2025 methods (TrajTok, RLFTSim, comBOT), publicly accessible full training code is incomplete or unclear.

Impact:
- exact reproduction of winner settings is blocked;
- we need principled approximation using public baselines.

### 5.2 Gap B: benchmark-to-code impedance mismatch
Official evaluator is standardized, but training pipelines are heterogeneous across repos.

Impact:
- reproducibility depends on custom data preprocessing, submission packaging, and runtime assumptions;
- many repos are not one-command reproducible.

### 5.3 Gap C: compute asymmetry
Several strong repos assume multi-GPU, multi-day training.

Impact:
- naive reproduction is impractical for Colab-centric workflows;
- methods must be reduced to ablations that still preserve benchmark validity.

### 5.4 Gap D: weak failure-mode instrumentation
Most public repos optimize end score but provide limited causal diagnosis of:
- why realism changed,
- which metric bucket moved,
- whether safety diagnostics regressed.

Impact:
- hard to derive thesis-grade causal narratives.

## 6. Problem-Focused Limitations in Current Literature

1. Strong methods often improve aggregate realism but under-report per-bucket tradeoffs.
2. Closed-loop distribution shift handling is discussed, but diagnostic standards are inconsistent.
3. Public code frequently omits full submission reproducibility (data prep + packaging + evaluator parity).
4. Benchmark reporting is leaderboard-first, but ablation transparency is often insufficient for method attribution.

## 7. Main Research Gap for This Repo

Our target gap is not “invent a new benchmark.” It is:

```text
Build a reproducible, Colab-compatible, benchmark-faithful methodology
that maps controlled implementation changes -> official metric deltas ->
clear causal interpretation.
```

This is distinct from pure model novelty and directly thesis-defensible.

## 8. Methodology Derived from Survey

### Stage 1: Baseline lock
- Reproduce one stable baseline end-to-end.
- Freeze config and submission pipeline.

### Stage 2: Structured variant ladder
- Tokenization variant
- Architecture variant
- Sampling / post-training variant
(only one axis per experiment)

### Stage 3: Evaluation contract
For every run, record:
- meta-metric and three bucket metrics,
- safety diagnostics,
- config hash, commit hash, runtime profile,
- keep/rollback decision with reason.

### Stage 4: Thesis-ready analysis
Write negative and positive results with explicit causal claims limited to observed deltas.

## 9. Falsifiable Research Hypotheses

1. A SMART-derived baseline with strict evaluator parity can exceed tutorial-level baselines on `realism_meta_metric` without increasing collision/offroad rates.
2. Closed-loop sample generation strategies (as emphasized by UniMM/TrafficBots lineage) improve interaction/map bucket stability more than pure token scaling alone.
3. RL fine-tuning variants can improve controllability but risk regressions unless constrained by metric-bucket acceptance criteria.

## 10. Local Paper Corpus Downloaded

Downloaded PDFs are in:
- `experiments/wosac-baseline/references/pdfs/`
- `experiments/wosac-baseline/references/pdfs_manifest.txt`

Notable availability note:
- Direct storage links for some 2025 winner technical reports (TrajTok, RLFTSim, comBOT) currently return `403` from public access paths tested on March 1, 2026.
- We included TrajTok arXiv paper and UniMM technical report/public paper; RLFTSim/comBOT are represented via available project/report references in this survey.

## 11. Primary Sources

- WOSAC challenge page: https://waymo.com/open/challenges/2025/sim-agents/
- WOSAC benchmark paper (NeurIPS D&B 2023): https://papers.nips.cc/paper_files/paper/2023/hash/b96ce67b2f2d45e4ab315e13a6b5b9c5-Abstract-Datasets_and_Benchmarks.html
- Waymo Open Dataset repo: https://github.com/waymo-research/waymo-open-dataset
- Waymax repo: https://github.com/waymo-research/waymax
- SMART repo/paper: https://github.com/rainmaker22/SMART , https://arxiv.org/abs/2405.15677
- TrafficBots repo/paper: https://github.com/zhejz/TrafficBots , https://arxiv.org/abs/2303.04116
- TrafficBotsV1.5 repo/paper: https://github.com/zhejz/TrafficBotsV1.5 , https://arxiv.org/abs/2406.10898
- UniMM repo/paper: https://github.com/Longzhong-Lin/UniMM , https://arxiv.org/abs/2501.17015
- TrajTok paper: https://arxiv.org/abs/2506.21618
