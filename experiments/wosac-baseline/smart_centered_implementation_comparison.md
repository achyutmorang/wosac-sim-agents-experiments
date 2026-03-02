# SMART-Centered Comparative Analysis of WOSAC Implementations

Date: March 2, 2026  
Scope: compare major WOSAC-focused implementations using SMART as the reference baseline, and identify high-value research gaps for this repository.

## 0. Sources and Evidence Level

Primary local sources used:
- `experiments/wosac-baseline/lit_survey.md`
- `references/leaderboard_2025.md`
- local PDFs in `experiments/wosac-baseline/references/pdfs/` (SMART, TrajTok, UniMM and related literature)

Evidence levels in this document:
- `A`: Paper/report + usable public code path exists.
- `B`: Paper/report available, but full code/checkpoints unavailable or partial.
- `C`: Leaderboard/project-page metadata only; technical details incomplete in local corpus.

## 1. Shared Problem Formulation (What All Methods Are Solving)

All compared methods target the same benchmark-level problem:

- Input: map context + short history (`~1.1s`) for active agents at handover.
- Output: closed-loop future simulation (`8s`, `10Hz`, up to 128 agents), with stochasticity (`32` rollouts per scenario).
- Evaluation: WOSAC realism composite and component safety/interaction/map metrics (collision/offroad/TL violation are critical ranking drivers).

In short, they solve:
- `simulate P(trajectories_{1:T} | history, map)` in closed loop,
- under interaction realism and safety-aware benchmark scoring.

Differences are mostly in:
- representation (`continuous` vs `tokenized`),
- training signal and rollout correction,
- inference-time sampling/control,
- practical reproducibility.

## 2. SMART Baseline (Reference Point)

Evidence level: `A` (paper + public code).

SMART formulation:
- trajectory/map tokenization + decoder-only autoregressive next-token prediction.
- explicit closed-loop rollout; mitigation for compounding error via rolling matching and noise injection.

Why SMART is a good anchor:
- public code exists and can be wrapped in this repo,
- strong historical performance for Sim Agents style tasks,
- architecture is simple enough to extend for controlled ablations,
- directly aligned with token-based trend seen in top methods.

Known limitations to keep in view:
- token/vocabulary design sensitivity,
- environment rigidity for exact reproduction (legacy CUDA/PyTorch stack),
- OOD drift remains a core challenge despite rolling matching.

## 3. Method-by-Method Comparison Against SMART

## 3.1 TrajTok (WOSAC 2025 top entry family)

Evidence level: `B` (report/paper-level details; no full official public training code in this repo context).

How it formulates the same shared problem:
- keeps the same WOSAC closed-loop, stochastic multi-agent rollout target as SMART.
- still uses discrete autoregressive behavior modeling.

Delta vs SMART:
- main innovation is tokenizer/training target quality, not a full paradigm change.
- hybrid tokenizer (rule-based + data-driven filtering/expansion), symmetry handling, spatial-aware label smoothing.

Likely gains:
- better support coverage of plausible trajectories,
- smoother probability mass assignment across nearby behavioral modes,
- improved realism without changing benchmark definition.

Limitations/repro risks:
- strong sensitivity to tokenizer hyperparameters and vocabulary design.
- no complete public stack makes exact engineering replication difficult.

What this suggests for your work:
- tokenizer/loss-space improvements are high-impact and benchmark-relevant.
- but pure token improvements may still under-address explicit safety-constrained uncertainty calibration.

## 3.2 RLFTSim (WOSAC 2025 second place)

Evidence level: `C` in this local workspace (leaderboard link/project-page status known; detailed code path missing).

How it formulates the same shared problem:
- still evaluated on the same WOSAC closed-loop realism task.

Known differentiator (from available metadata naming and related literature lineages):
- likely stronger RL/fine-tuning emphasis on interaction quality and rollout robustness.

Comparison to SMART:
- SMART is largely supervised token autoregression with closed-loop corrections.
- RLFTSim-style methods likely add policy optimization pressure beyond pure imitation.

Limitation in current comparison:
- cannot claim architecture-level differences confidently without technical report text/code in local corpus.

What this suggests for your work:
- there is a plausible value in post-imitation fine-tuning for rollout stability,
- but reproducibility and objective mismatch risks are high unless metric alignment is explicit.

## 3.3 comBOT (WOSAC 2025 third place)

Evidence level: `C` in this local workspace (leaderboard metadata only).

How it formulates the same shared problem:
- same WOSAC task and metric contract.

Comparison to SMART:
- unknown architectural specifics from current local artifacts.
- practical difference for your plan is reproducibility availability, not benchmark definition.

What this suggests for your work:
- comBOT should currently be treated as an external report-level comparator, not an implementable baseline.
- avoid overfitting your contribution narrative to unverifiable details.

## 3.4 UniMM (WOSAC 2025 honorable mention)

Evidence level: `B` (paper/report available; public repo exists but full release still limited).

How it formulates the same shared problem:
- same closed-loop stochastic multi-agent simulation objective under WOSAC metrics.

Delta vs SMART:
- conceptual reframing: discrete token models and continuous models are viewed under one mixture-model umbrella.
- emphasizes configuration and closed-loop sample generation choices as primary realism drivers.

Likely gains:
- potentially better control over multimodality and calibration via mixture formulation choices.
- explicitly studies where performance comes from (architecture vs data-generation/training policy).

Limitations/repro risks:
- higher implementation complexity than a single token stack.
- strong dependence on closed-loop data regeneration policy quality.

What this suggests for your work:
- supports your planned focus on uncertainty and constrained probabilistic control.
- indicates that training-data/rollout policy design may matter as much as model family.

## 3.5 TrafficBots V1.5 / BehaviorGPT (important public comparators, non-2025-top)

Evidence level: `A/B` depending on artifact.

Why include:
- they are concrete, inspectable baselines for ablations when top-2025 code is unavailable.

Delta vs SMART:
- TrafficBots line: CVAE/world-model style with scheduled teacher forcing and scenario filtering.
- BehaviorGPT: patch-level autoregressive factorization with local independence assumptions.

Usefulness:
- helps isolate whether gains come from tokenization, rollout strategy, or objective shaping.

## 4. Unified Comparison Matrix (SMART as Center)

| Method | Shared WOSAC Problem Alignment | Main Delta vs SMART | Strength | Key Limitation | Reproducibility Risk |
|---|---|---|---|---|---|
| SMART | Direct | Baseline reference | Open code + strong tokenized AR baseline | Token/OOD sensitivity | Medium |
| TrajTok | Direct | Better tokenizer + smoothing/symmetry | Better trajectory support coverage | Hyperparameter sensitivity; no full code | High |
| RLFTSim | Direct | Likely RL fine-tuning layer | Potentially better closed-loop robustness | Method internals not locally inspectable | Very high |
| comBOT | Direct | Unknown in local corpus | Top-ranking evidence exists | Technical details unavailable here | Very high |
| UniMM | Direct | Mixture-model reframing + closed-loop sample generation | Strong lens on calibration + data policy effects | Higher system complexity | High |
| TrafficBots V1.5 | Direct | CVAE/world-model trajectory style | Public baseline lineage | Weaker reported top metrics | Medium |
| BehaviorGPT | Direct | Patch-level AR factorization | Better long-horizon representation attempt | Compounding error still observed | Medium |

## 5. Patterns That Matter for Your Contribution

## 5.1 Same benchmark, different failure-control strategy
- SMART/TrajTok focus on representation + token learning improvements.
- UniMM emphasizes probabilistic formulation and generation-policy choices.
- RL-oriented methods emphasize policy robustness after imitation.

## 5.2 Most methods still rely on soft handling of hard safety constraints
- Collision/offroad are scored heavily, but many pipelines optimize proxies and hope constraints emerge.
- This leaves room for explicit constrained probabilistic optimization.

## 5.3 Closed-loop distribution shift is still the core bottleneck
- Different methods attack it differently (rolling matching, scheduled teacher forcing, sample generation, RL fine-tuning),
- but none fully remove long-horizon drift and safety-diversity tension.

## 5.4 Reproducibility is now a scientific gap, not just an engineering issue
- Top leaderboard methods without full code/checkpoints make fair comparison hard.
- A transparent, reproducible baseline-to-variant pipeline is itself a publishable contribution enabler.

## 6. High-Value Research Gaps (Prioritized)

## Gap A (Highest): Safety-diversity Pareto under probabilistic rollout

Observed:
- Strong methods improve realism, but safety metrics can improve by collapsing behavioral diversity.

Open question:
- Can we improve safety metrics while preserving calibrated multimodal behavior?

Why this is publishable:
- Directly benchmark-aligned, measurable, and under-addressed by pure tokenizer changes.

## Gap B (High): Explicit constrained decoding/training for closed-loop simulation

Observed:
- Most pipelines treat constraints as implicit through data/loss weighting.

Open question:
- Can constraints (collision/offroad/TL risk budgets) be integrated into sampling or objective without destroying realism?

Why this is publishable:
- Strong theoretical framing + practical metric impact.

## Gap C (High): Architecture vs generation-policy confounding

Observed:
- UniMM suggests closed-loop sample generation policy can explain major gains.

Open question:
- With architecture fixed (SMART baseline), how much gain comes from generation/training policy alone?

Why this is publishable:
- Clean ablation value; helps resolve current literature ambiguity.

## Gap D (Medium): Risk-calibrated uncertainty quality

Observed:
- Many methods optimize aggregate realism but not explicit uncertainty calibration under rare interactions.

Open question:
- Does probabilistic calibration improve tail safety metrics and robustness?

Why this is publishable:
- Matches your probabilistic deep learning direction and benchmark needs.

## 7. Concrete Research Positioning for This Repository

Recommended problem framing:
- “Given a SMART-style closed-loop token simulator, design a constrained probabilistic training/inference scheme that improves WOSAC realism meta-metric while maintaining or improving safety-critical component metrics, under a fully reproducible Colab-first pipeline.”

Minimal defensible experimental ladder:
1. Reproduce SMART wrapper baseline end-to-end (already in progress in this repo).
2. Add one constrained probabilistic variant only (no architecture explosion).
3. Run fixed-budget ablations on:
   - sampling temperature/top-k,
   - constraint penalty/budget,
   - closed-loop data regeneration ratio.
4. Report:
   - primary metric (`realism_meta_metric`),
   - safety components (collision/offroad/TL),
   - diversity proxies,
   - reproducibility artifacts (config hash, run manifest).

## 8. Thesis-Writing Value

Even before SOTA gains, this comparative framing gives thesis-strength contributions:
- a rigorous reproduction-first benchmark protocol,
- a SMART-centered method comparison with explicit evidence levels,
- controlled ablation evidence separating architecture gains from training-policy gains,
- a constrained probabilistic method proposal directly tied to benchmark metrics.

## 9. Known Unknowns (To Keep Honest)

- RLFTSim/comBOT technical internals are incomplete in local artifacts as of March 2, 2026.
- Any claim about their exact architecture/training details should be marked provisional until their reports/code are locally ingested and parsed.

