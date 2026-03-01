# WOSAC Sim Agents Literature Survey (Comparative, PDF-Grounded)

Date: March 2, 2026  
Scope: comparative analysis of problem formulations, methodologies, assumptions, limitations, and research-gap patterns for WOSAC-oriented work.

## 0. Corpus and Method

This survey is grounded in the local reference PDF corpus under:
- `experiments/wosac-baseline/references/pdfs/`

Covered works:
- Benchmark and infra: WOSAC (2023), Waymax (2023)
- Challenge and trajectory-generation methods: MVTA/MVTE (2023), TrafficBots (2023), TrafficBots V1.5 (2024), SMART (2024), BehaviorGPT (2024), TrajTok (2025), UniMM (2025)
- RL and reliability lines: RIFT (2025), Building reliable sim driving agents by scaling self-play (2025), ForSim (2026)

Important note on leaderboard date consistency:
- TrajTok arXiv v1 dated June 23, 2025 reports `realism=0.7852` and shows second place in the paper's submission-period table.
- Public challenge tables may differ by phase/time; always tie claims to a dated source.

## 1. Benchmark Formulation (Ground Truth Problem Definition)

## 1.1 WOSAC task setup (Montali et al., 2023)

Core task:
- Inputs: map + 1.1s history.
- Output: 8s closed-loop future rollout (80 steps at 10 Hz), up to 128 agents.
- Requirement: 32 stochastic rollouts per scenario.

Required factorization (paper formulation):
- Closed-loop autoregressive simulation is mandatory.
- ADV policy and environment dynamics are factorized so ADV policy can be swapped.
- Agent-centric multi-agent factorization for environment dynamics is allowed.

Evaluation design:
- Approximate NLL under simulated distribution, computed from submitted samples.
- Uses histogram-based likelihood from 32 rollouts.
- Composite metric built from component metrics (kinematic, interaction, map-based).
- Collision and road-departure were manually upweighted (2x versus other components in 2023 setting).

## 1.2 Benchmark assumptions and implications

Explicit benchmark assumptions/constraints from WOSAC paper:
- Simulate agents valid at handover time (`t=0`).
- Object dimensions fixed to last observed history state.
- No specific motion model enforced (submit states directly).

Implications for research design:
- Better trajectory regression alone does not guarantee better realism metric.
- Calibration over safety-critical components (collision/offroad) can move ranking disproportionately.
- 32-sample stochastic quality is central; deterministic one-shot quality is insufficient.

## 2. Comparative Survey by Work

## 2.1 Benchmark and simulator foundations

### WOSAC benchmark paper (2023)
- Formulation: factorized closed-loop generative simulation with ADV/environment decomposition.
- Methodology: benchmark and metric design, not a single simulator model.
- Assumptions: histogram-based approximation of likelihood; time-aggregated metric components.
- Limitations/open issues stated: open questions on scene-centric vs agent-centric policies, planning depth needed, and benchmark/property design itself.

### Waymax (2023)
- Formulation: differentiable, multi-agent simulator instantiated from real-world logs.
- Methodology: JAX accelerator-first infrastructure, route-aware closed-loop metrics, bundled baseline agents.
- Assumptions: behavior-level simulation focus (not sensor simulation); scenarios initialized from WOMD logs.
- Limitations: baseline agents are not near challenge-top behavior quality; infrastructure solves "how to run" more than "what wins WOSAC".

## 2.2 Challenge-focused simulation methods

### Multiverse Transformer (MVTA/MVTE, 2023)
- Formulation: WOSAC-consistent 0.1s autoregressive rollout with ADV/environment factorization.
- Methodology:
  - Transformer scene encoder + autoregressive decoder.
  - GMM multi-modal head.
  - Receding-horizon prediction (predict 1s, execute first 0.1s).
  - Variable-length history aggregation.
  - Periodic top-k stochastic sampling to trade off realism/diversity.
- Assumptions:
  - Factorized architecture satisfies challenge interchangeability requirements.
  - `z` kept constant from initial state in their setup.
- Limitations:
  - Top-k sampling increases diversity but can increase drift/kinematic artifacts.
  - Preliminary collision-avoidance loss did not improve realism in their report.

### TrafficBots (2023)
- Formulation: data-driven traffic simulation as a world model; recurrent closed-loop multi-agent policy.
- Methodology:
  - Destination-conditioned + time-invariant latent personality (CVAE).
  - Shared vectorized context and attention for scalability.
- Assumptions:
  - Uncertainty treated via GT-future-conditioned decomposition in formulation discussion.
  - Time-invariant personality as style factor.
- Limitations:
  - Strong conceptual framing but not top benchmark realism versus later token methods.
  - Original work explicitly leaves player-agent training as future work.

### TrafficBots V1.5 (2024)
- Formulation: shared policy for all agents, conditioned on destination and personality.
- Methodology:
  - TrafficBots + HPTR pairwise-relative transformer.
  - Scheduled teacher forcing during training.
  - Scenario filtering at inference.
- Assumptions:
  - Altitude (`z`) held constant from last observed value at inference.
  - Unicycle dynamics used for all agent types in their report.
- Limitations (explicitly discussed):
  - No ablation studies in report.
  - Weaker interactive metrics and high minADE relative to top GPT-style entries.
  - Authors note compute limitations and manual rough parameter tuning.

### SMART (2024)
- Formulation: trajectory/map tokenization + decoder-only next-token prediction.
- Methodology:
  - Motion and road-vector tokenization.
  - GPT-style autoregressive decoding.
  - Noise-injected rolling matching to counter compounding error/distribution shift.
- Assumptions:
  - Discrete token vocabulary sufficiently captures driving behavior modes.
  - Replanning and token horizon coupling are primary stability controls.
- Limitations:
  - Tokenization and autoregression still face OOD drift risks (explicitly discussed in method rationale).
  - Large token/data scaling pressures can complicate reproduction at low compute.

### BehaviorGPT (2024)
- Formulation:
  - Patch-level autoregressive factorization over multi-agent trajectories.
  - Per-patch agent-level factorization with a local independence assumption.
- Methodology:
  - Fully autoregressive decoder-only transformer.
  - Next-Patch Prediction Paradigm (NP3) for longer-horizon reasoning.
  - Triple-attention design (temporal, agent-agent, agent-map).
- Assumptions:
  - Agents are conditionally independent within patch horizon.
  - Patching improves semantic reasoning versus pointwise next-token.
- Limitations (explicitly shown):
  - Failure cases still show compounding-error drift (offroad example in paper).

### TrajTok technical report (2025)
- Formulation: discrete NTP behavior model, built as tokenizer/loss upgrade over SMART-family setup.
- Methodology:
  - Hybrid tokenizer (rule-based gridding + data-driven filtering/expansion).
  - Symmetry enforcement via flipped trajectories.
  - Spatial-aware label smoothing (distance-aware non-uniform smoothing).
- Assumptions:
  - Better tokenizer should cover plausible trajectory space beyond empirical data support.
  - Symmetry and coverage improve robustness/generalization.
- Limitations:
  - Code stated as future release in report.
  - Performance sensitivity to vocabulary size and tokenizer hyperparameters.
  - Built on SMART-tiny baseline stack, so reproducibility depends on that ecosystem.

### UniMM (2025 paper + 2025 technical report)
- Formulation:
  - Unifies continuous and discrete methods under a common mixture-model view.
  - Autoregressive simulation with small-step decomposition and per-agent factorization.
- Methodology:
  - Systematic study over mixture-model configurations:
    - positive component matching,
    - continuous regression,
    - prediction horizon,
    - component count.
  - Closed-loop sample generation for distribution-shift mitigation.
  - Additional handling of shortcut-learning and off-policy issues.
- Assumptions:
  - With small update interval, per-agent conditional independence is tractable.
  - GPT-like token methods are special cases of anchor-based mixture models.
- Limitations:
  - Higher implementation complexity than single-stack token baselines.
  - Requires careful closed-loop data generation policy to avoid shortcut/off-policy failure modes.

## 2.3 RL fine-tuning and reliability lines

### RIFT (2025)
- Formulation: dual-stage AV-centric pipeline.
  - Stage 1: IL pretraining in data-driven simulator.
  - Stage 2: RL fine-tuning in physics-based simulator.
- Methodology:
  - Group-relative fine-tuning over candidate modalities.
  - CBV (critical background vehicle) selection via interaction analysis.
  - Surrogate objective for stable optimization.
- Assumptions:
  - A quality IL model exists before RL stage.
  - Theoretical assumptions for analysis (support floor, boundedness, regularity).
- Limitations (paper discussion):
  - Fine-tuning may overfit simulator specifics, creating sim-to-real transfer gaps.
  - Dependence on quality of pretraining/reference behavior remains.

### Building reliable sim driving agents by scaling self-play (2025)
- Formulation: reliability-first objective (goal reach + no collision + no offroad) in POSG setting.
- Methodology:
  - Large-scale decentralized self-play PPO in GPUDrive.
  - Task-specific reward shaping and batched scenario scaling.
- Assumptions:
  - Agents should reach WOMD endpoint goals within 91 steps.
  - Semi-realistic observation design, no history in observation.
- Limitations for WOSAC transfer:
  - Objective is reliability/goal completion, not WOSAC likelihood realism.
  - May optimize away diversity characteristics relevant for WOSAC realism score.

### ForSim (2026)
- Formulation: stepwise closed-loop forward simulation for multimodal candidate rollout.
- Methodology:
  - Trajectory-Aligned Rollout for controlled traffic agent.
  - Stepwise Prediction Rollout for other agents (interaction-aware updates each virtual step).
  - Integrated with RIFT-style group-relative optimization.
- Assumptions:
  - Physically grounded propagation (PID + bicycle model).
  - Best spatiotemporal match to reference trajectory preserves modality consistency.
- Limitations:
  - Added rollout complexity and dependency on predictor quality.
  - Preprint-stage method; independent reproduction outside RIFT ecosystem remains non-trivial.

## 3. Cross-Paper Comparison Matrix

| Work | Problem Formulation Style | Core Method | Main Assumption | Main Limitation |
|---|---|---|---|---|
| WOSAC 2023 | Factorized closed-loop generative benchmark | Histogram NLL realism metric over 32 rollouts | Factorization + hand-tuned metric weighting represent realism | Proxy realism is not full safety/planning utility |
| Waymax 2023 | Data-driven simulator substrate | JAX differentiable multi-agent simulation infra | Behavior-level simulation sufficient for research loops | Not itself a top SimAgents solver |
| MVTA/MVTE 2023 | Closed-loop autoregressive with ADV/world split | Transformer + GMM + receding horizon + top-k periodic sampling | Factorized decomposition and periodic sampling stabilize rollout | Diversity-vs-drift tradeoff remains |
| TrafficBots 2023 | World-model simulation with destination/personality | CVAE + transformer shared policy | Time-invariant personality captures style | Falls behind later tokenized approaches |
| TrafficBots V1.5 2024 | Shared policy with pairwise-relative encoding | CVAE + HPTR + scheduled teacher forcing | Constant z and unicycle dynamics are acceptable approximations | Limited ablation and weaker interaction metrics |
| SMART 2024 | Discrete token autoregression | Decoder-only NTP with trajectory/map tokens | Tokenization + rolling matching can control covariate shift | Reproduction cost and token design sensitivity |
| BehaviorGPT 2024 | Patch-level autoregressive factorization | Decoder-only NP3 with triple attention | Within-patch agent independence | Compounding error still visible in failure cases |
| TrajTok 2025 | Tokenizer-centric enhancement over NTP | Hybrid tokenizer + spatial-aware smoothing | Better coverage/symmetry improves realism | Full code release not available in report |
| UniMM 2025 | Unified mixture-model view of continuous+discrete | Config sweeps + closed-loop sample generation | Small-step per-agent decomposition | Higher methodological complexity |
| RIFT 2025 | IL-to-RL dual-platform fine-tuning | Group-relative RL over candidate modalities | AV-centric CBV targeting sufficient for controllability | Potential simulator overfitting and transfer gap |
| Reliable self-play 2025 | Reliability-first objective in POSG | Large-scale PPO self-play | Goal/no-collision/no-offroad criteria define useful behavior | Objective mismatch with WOSAC realism benchmark |
| ForSim 2026 | Stepwise forward simulation for multimodal rollouts | Trajectory-aligned + stepwise interactive propagation | Dynamic consistency via physical propagation | Integration complexity and early-stage reproducibility |

## 4. Patterns Across the Literature

## 4.1 Converging pattern: autoregressive closed-loop is non-negotiable
All top-performing directions explicitly model sequential rollout rather than one-shot open-loop forecasting.

## 4.2 Dominant representation shift: continuous mixture -> discrete tokenization
From 2023 to 2025, strong leaderboard methods increasingly use token-based NTP formulations, while UniMM reframes this as a broader mixture-model configuration choice rather than a strict paradigm break.

## 4.3 Covariate-shift mitigation is the central bottleneck
Across families, improvements repeatedly target closed-loop mismatch:
- scheduled teacher forcing,
- noise-injected tokenization/rolling matching,
- explicit closed-loop sample generation,
- RL fine-tuning in interactive simulators.

## 4.4 Recurring assumptions that may cap realism
Common assumptions that simplify training but may constrain realism:
- short-horizon per-agent conditional independence,
- fixed/approximated dynamics channels (for example constant z),
- simplified physics updates for non-primary agents,
- objective proxying through weighted metric bundles.

## 4.5 Reproducibility gap remains large
Several strong methods report excellent realism, but full end-to-end public training/evaluation stacks are still inconsistent, especially for top 2025 entries.

## 5. Potential Research Gaps (Actionable for This Repo)

## Gap 1: Architecture vs data-configuration confounding
Observation:
- UniMM suggests closed-loop sample generation can explain much of the discrete-vs-continuous gap.

Research question:
- How much realism gain comes from data configuration alone when architecture is fixed?

Minimal experiment:
- Keep one baseline architecture fixed; compare open-loop, scheduled teacher forcing, and closed-loop sample generation under same compute budget.

## Gap 2: Safety-heavy metric components vs behavior diversity
Observation:
- WOSAC composition rewards collision/offroad strongly, while realism also needs multimodality.

Research question:
- Which training knobs improve safety components without collapsing diversity?

Minimal experiment:
- Controlled sweep over replan frequency, sampling temperature/top-k, and token vocabulary coverage; report collision/offroad + diversity proxies together.

## Gap 3: Independence assumptions at patch/step horizon
Observation:
- Multiple works assume per-agent independence within short horizons.

Research question:
- Does explicit short-horizon coordination improve interaction metrics enough to justify complexity?

Minimal experiment:
- Compare independent-head versus lightweight coordinated interaction head on identical tokenization and training data.

## Gap 4: Physics-aware rollout for non-primary agents
Observation:
- ForSim shows gains when other-agent rollouts become stepwise reactive.

Research question:
- In WOSAC-like setup, do interaction metrics improve when non-primary agents use stepwise prediction instead of static rollout assumptions?

Minimal experiment:
- Swap other-agent propagation module only; hold policy fixed.

## Gap 5: Colab-feasible reproducibility science
Observation:
- Many papers optimize score but underreport reproducibility variance.

Research question:
- What is the metric variance across seeds/runtimes under constrained Colab budgets?

Minimal experiment:
- Run three-seed baseline reproducibility protocol with fixed config hash and artifact logging.

## 6. Thesis-Oriented Positioning

A strong thesis narrative can be built around:
1. Benchmark-faithful baseline reproduction.
2. Controlled ablations tied to specific assumptions from literature.
3. Explicit negative results (for example, when a mitigation improves one bucket and harms another).
4. Reproducibility and methodological clarity as the main contribution, not only raw leaderboard chasing.

This is publishable if claims are causal, bounded, and fully traceable to fixed experimental contracts.

## 7. References (Local PDF Corpus)

- [The Waymo Open Sim Agents Challenge (NeurIPS 2023 Datasets and Benchmarks)](references/pdfs/wosac_challenge_2023_neurips_db.pdf)
- [Waymax: An Accelerated, Data-Driven Simulator for Large-Scale Autonomous Driving Research (2023)](references/pdfs/waymax_2023.pdf)
- [Multiverse Transformer: 1st Place Solution for Waymo Open Sim Agents Challenge 2023](references/pdfs/multiverse_transformer_2023.pdf)
- [TrafficBots: Towards World Models for Autonomous Driving Simulation and Motion Prediction (2023)](references/pdfs/trafficbots_2023.pdf)
- [TrafficBots V1.5 Technical Report (WOSAC 2024)](references/pdfs/trafficbots_v1_5_2024.pdf)
- [SMART: Scalable Multi-agent Real-time Simulation via Next-token Prediction (NeurIPS 2024)](references/pdfs/smart_2024.pdf)
- [BehaviorGPT: Smart Agent Simulation for Autonomous Driving with Next-Patch Prediction (2024)](references/pdfs/behaviorgpt_2024.pdf)
- [TrajTok Technical Report for 2025 Waymo Open Sim Agents Challenge](references/pdfs/trajtok_2025_arxiv.pdf)
- [Revisit Mixture Models for Multi-Agent Simulation: Experimental Study within a Unified Framework (2025)](references/pdfs/unimm_2025_arxiv.pdf)
- [UniMM Technical Report for Waymo Open Sim Agents Challenge 2025](references/pdfs/wosac_2025_unimm_technical_report.pdf)
- [RIFT: Group-Relative RL Fine-Tuning for Realistic and Controllable Traffic Simulation (2025)](references/pdfs/rift_2025.pdf)
- [Building reliable sim driving agents by scaling self-play (2025)](references/pdfs/reliable_simulated_driving_agents_2025.pdf)
- [ForSim: Stepwise Forward Simulation for Traffic Policy Fine-Tuning (2026)](references/pdfs/forsim_2026.pdf)
