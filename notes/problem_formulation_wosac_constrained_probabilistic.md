# Rigorous Problem Formulation: Constrained Probabilistic WOSAC Simulation

Date: March 2, 2026

## 1. Scope and Goal

We treat WOSAC Sim Agents as a stochastic closed-loop multi-agent simulation problem: for each scenario, given map context and short history, generate 32 plausible joint futures over an 8-second horizon. The research goal is to improve realism while explicitly controlling safety-critical behavior, using a SMART-style autoregressive token model as the baseline backbone.

## 2. Scenario-Level Mathematical Setup

A scenario is indexed by $n\in\{1,\dots,N\}$.

For scenario $n$:

- map and static context: $m_n$
- active agents at handover: $\mathcal{A}_n$, with $|\mathcal{A}_n|\le 128$
- observed history for each agent $i\in\mathcal{A}_n$: $x^{(n)}_{i,-H:0}$
- future horizon: $T=80$
- stochastic rollouts required: $K=32$

Conditioning context:

$$
C_n:=\left(m_n,\{x^{(n)}_{i,-H:0}\}_{i\in\mathcal{A}_n}\right).
$$

A sampled joint rollout $k$ is:

$$
Y^{(n,k)}_{1:T}:=\{x^{(n,k)}_{i,1:T}\}_{i\in\mathcal{A}_n}.
$$

The simulator induces a conditional distribution $q_{\theta}(Y_{1:T}\mid C)$, where $\theta$ are model parameters.

## 3. SMART-Compatible Autoregressive Factorization

Let the motion token for agent $i$ at time $t$ be $s_{i,t}\in\mathcal{V}$. Using SMART-style token autoregression:

$$
q_{\theta}(s_{1:T}\mid C)
=\prod_{t=1}^{T}\prod_{i\in\mathcal{A}_t}
q_{\theta}\big(s_{i,t}\mid s_{<t},C\big).
$$

With decoder $g$, continuous rollout is:

$$
x_{t+1}=F\big(x_t,g(s_t),m\big),
$$

where $F$ is the simulation propagation operator.

## 4. Shared Benchmark Objective (Evaluation Space)

Let evaluator output over the 32 sampled rollouts be represented by $\mathcal{M}(\{Y^{(k)}\}_{k=1}^{K},C)$.

Key quantities:

- primary realism score: $R$
- safety components: $C_{col},\;C_{off},\;C_{tl}$
- diversity/calibration proxy: $D$

Population-level target:

$$
\max_{\theta} J(\theta)
:=\mathbb{E}_{C\sim\mathcal{D}}\left[R\left(\{Y^{(k)}\}_{k=1}^{K};C\right)\right].
$$

Since leaderboard rank also depends on safety components, we use an explicit constrained formulation.

## 5. Main Research Formulation: Constrained Probabilistic Optimization

We define:

$$
\begin{aligned}
\max_{\theta,\eta}\quad & \mathbb{E}[R] \\
\text{s.t.}\quad
& \mathbb{E}[C_{col}]\le\epsilon_{col}, \\
& \mathbb{E}[C_{off}]\le\epsilon_{off}, \\
& \mathbb{E}[C_{tl}]\le\epsilon_{tl}, \\
& \mathbb{E}[D]\ge\delta_{min}.
\end{aligned}
$$

Here, $\eta$ are inference-policy controls, $\epsilon_{col},\epsilon_{off},\epsilon_{tl}$ are safety budgets, and $\delta_{min}$ is a diversity floor.

This keeps the benchmark unchanged, but changes optimization to explicitly manage realism-safety tradeoffs.

## 6. Surrogate Training Objective (Practical)

Because official metrics are not directly differentiable end-to-end, optimize:

$$
\mathcal{L}(\theta,\eta,\lambda,\mu)
=\mathcal{L}_{base}(\theta)
+\sum_{j\in\{col,off,tl\}}\lambda_j\,[\hat C_j(\theta,\eta)-\epsilon_j]_+
+\mu\,[\delta_{min}-\hat D(\theta,\eta)]_+,
$$

with multipliers constrained as $\lambda_j\ge 0$ and $\mu\ge 0$.

Multiplier updates:

$$
\lambda_j\leftarrow\left[\lambda_j+\alpha_{\lambda}(\hat C_j-\epsilon_j)\right]_+,
$$

$$
\mu\leftarrow\left[\mu+\alpha_{\mu}(\delta_{min}-\hat D)\right]_+.
$$

## 7. Optional Tail-Risk Extension

To penalize rare catastrophic failures, enforce tail constraints such as:

$$
\mathrm{CVaR}_{\tau}(C_{col})\le\bar\epsilon_{col},
$$

and similarly for offroad and traffic-light violations.

## 8. Hypotheses (Paper-Consistent)

- H1: With SMART architecture fixed, constrained probabilistic optimization improves safety without reducing primary realism.
- H2: A substantial fraction of gains can come from objective/rollout-policy design, not only architecture changes.
- H3: Diversity-aware constraints prevent mode-collapse masquerading as safety improvement.
- H4: Tail-risk constraints reduce catastrophic rollout modes better than mean-only constraints.

## 9. Identifiability and Ablation Discipline

Hold fixed:

- SMART-compatible base architecture/tokenizer family
- data split and preprocessing
- compute budget and training steps
- evaluator and metric ingestion path

Vary only:

- constraint budgets $\epsilon_{\cdot}$
- multiplier schedules
- decoding policy $\eta$
- optional tail-risk terms

This yields interpretable causal ablations instead of uncontrolled exploration.

## 10. Acceptance Criteria

A variant is accepted only if:

$$
\Delta R>0,
$$

$$
\Delta C_{col}\le 0,\;\Delta C_{off}\le 0,\;\Delta C_{tl}\le 0,
$$

and diversity remains above threshold.

## 11. Consistency With Existing Formulations

- WOSAC defines the shared task/evaluator contract.
- SMART provides the concrete tokenized closed-loop baseline.
- TrajTok shows tokenizer/loss refinements can improve realism within the same task.
- UniMM shows formulation and closed-loop generation policy can drive large gains.

Hence this formulation is literature-consistent and contribution-ready: same benchmark problem, explicit constrained probabilistic optimization as the methodological novelty.

## 12. One-Line Formal Statement

Given context $C$, learn a stochastic closed-loop policy $q_{\theta,\eta}(Y_{1:T}\mid C)$ that maximizes expected WOSAC realism under explicit safety and diversity constraints, using SMART-style autoregressive token modeling as the baseline representation.

