# HHCRA: Hierarchical Hybrid Causal Reasoning Architecture

![Tests](https://img.shields.io/badge/tests-215%20passing-brightgreen)
![Python](https://img.shields.io/badge/python-%3E%3D3.8-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A three-layer neuro-symbolic architecture for Structural Causal Model (SCM) estimation and causal inference. HHCRA combines slot-attention-based variable extraction, continuous DAG optimization, and symbolic causal reasoning to address Pearl's causal hierarchy (association, intervention, counterfactual).

**Status:** Research prototype. Not validated on real-world observational data. See [Limitations](#limitations).

## Overview

HHCRA decomposes SCM inference into three hierarchical layers with gradient isolation (`detach()`) between layers and staged optimization. Each layer is trained independently.

### SCM Formulation

The architecture maps to the SCM tuple $M = \langle V, U, F, P(u) \rangle$:
- $V$: Endogenous variables, extracted by Layer 1.
- $U$: Exogenous noise variables (stochastic components).
- $F$: Structural equations, modeled by Layer 2 dynamics.
- $G$: Causal DAG, learned via NOTEARS continuous optimization.

### Architecture

```mermaid
flowchart TD
    Input([Observations]) ==> L1

    subgraph L1 [Layer 1: Perception]
        C1[C-JEPA Slot Attention]
    end

    L1 -- "detach()" --> L2

    subgraph L2 [Layer 2: Mechanism]
        C2[NOTEARS GNN] <--> C3[Liquid Neural ODE]
    end

    L2 -- "detach()" --> L3

    subgraph L3 [Layer 3: Reasoning]
        C4[Neuro-Symbolic Engine] <--> C5[HRM]
    end

    Q{{Causal Query}} -.-> L3
    L3 ==> R([Result])

    style L1 fill:#f0fdf4,stroke:#166534
    style L2 fill:#e7f2ff,stroke:#005bb7
    style L3 fill:#fff4dd,stroke:#d4a017
```

### Component Mapping

| SCM Component | Module | Layer | Based On |
|---|---|---|---|
| $V$ (Variables) | C-JEPA | 1 | Slot Attention (Locatello et al., 2020) |
| $G$ (Topology) | Causal GNN | 2 | NOTEARS (Zheng et al., 2018) |
| $F$ (Mechanisms) | Liquid Neural Network | 2 | LTC Networks (Hasani et al., 2021) |
| $P(Y \mid do(x))$, $P(Y_{x'} \mid x, y)$ | Neuro-Symbolic Engine | 3 | Pearl's do-calculus |
| Reasoning orchestration | HRM (GRU + ACT) | 3 | Graves (2016) |

## Method

### Layer 1: Latent Variable Extraction

Causal Joint Embedding Predictive Architecture (C-JEPA). Slot attention with competitive softmax decomposes observations into $N$ latent variable representations. Temporal consistency is enforced via exponential smoothing. Training objective: masked slot prediction.

**v0.8.0 improvements:** Adaptive slot count (utilization-based pruning) reduces spurious edges from unused slots. Independence regularization (slot template decorrelation) encourages bijective slot-to-variable correspondence. See [Failure Analysis](#failure-analysis) for remaining limitations.

### Layer 2: Structure and Mechanism Learning

- **Structure**: NOTEARS (Zheng et al., 2018) augmented Lagrangian formulation: $\min_W F(W) + \lambda \|W\|_1 + \alpha \cdot h(W) + \frac{\rho}{2} h(W)^2$, where $h(W) = \mathrm{tr}(e^{W \circ W}) - d$.
- **Dynamics**: Liquid Time-Constant Networks (Hasani et al., 2021) with per-variable ODE $dx_i/dt = g_i \cdot (-x_i + f_i(x_i, \mathrm{pa}_i)) / \tau_i$. Integration via RK4.

In v0.6.0, when raw data is available, Layer 2 receives it directly (bypassing Layer 1) for warm initialization. v0.8.0 upgrades this to use temporal Granger regression (SHD 1-4) instead of cross-sectional NOTEARS for initialization, exploiting temporal asymmetry.

### Layer 3: Causal Reasoning

- **Symbolic inference**: d-separation (Bayes-Ball), backdoor/frontdoor criteria, do-calculus (3 rules), ID algorithm (Tian & Pearl, 2002), instrumental variable detection.
- **Counterfactuals**: Variable-space SCM fitting with ABP/total-effect ensemble for robust counterfactual prediction.
- **HRM**: Hierarchical Reasoning Model with dual-timescale GRU recurrence and Adaptive Computation Time (Graves, 2016).

### Experimental Extensions (v0.6.0)

The following modules are implemented but **not yet validated on benchmarks**:

- **Liquid Causal Graph**: Co-evolving ODE system where graph weights $W(t)$ evolve jointly with state $x(t)$. Replaces the static adjacency matrix with a dynamical system.
- **Trajectory Invariant Finder**: Neural network-based invariant detection from ODE trajectories. Attempts to discover conserved quantities ($dh/dt \approx 0$).
- **Iterative Refinement Loop**: Iterative feedback loop running GNN, Liquid ODE, and symbolic components in sequence with early stopping.

These extensions are accessible via `HHCRAExperimental` but have no benchmark evidence of performance improvement over the base architecture.

## Evaluation

### Toy Graph Benchmarks (v0.4)

Evaluated on 5 synthetic linear-Gaussian SCMs (chain, fork, collider, diamond, 8-variable complex). Full results in `results/REPORT.md`.

#### Structure Learning (SHD, lower is better)

| Graph | Vars | Edges | Temporal Granger | Direct NOTEARS | Full Pipeline | PC |
|-------|------|-------|-----------------|----------------|---------------|-----|
| chain | 4 | 3 | 4 | 6 | 14 | 3 |
| fork | 3 | 2 | 1 | 5 | 16 | 0 |
| collider | 3 | 2 | 2 | 3 | 12 | 4 |
| diamond | 4 | 4 | 4 | 9 | 12 | 4 |
| complex | 8 | 9 | 10 | 18 | 13 | 9 |

**Observation:** The full pipeline (Layer 1 + Layer 2) performs worse than standalone NOTEARS or Granger on all graphs. Root cause: variable alignment failure in Layer 1 (see [Failure Analysis](#failure-analysis)).

#### Interventional Accuracy

| Graph | Naive MSE | HHCRA MSE | HHCRA beats Naive |
|-------|-----------|-----------|-------------------|
| chain | 6.925 | 0.660 | Yes |
| fork | 5.432 | 0.503 | Yes |
| collider | 1.266 | 0.350 | Yes |
| diamond | 4.767 | 0.472 | Yes |
| complex | 9.108 | 0.246 | Yes |

HHCRA interventional predictions beat the naive (correlation-based) baseline on 5/5 graphs.

#### Counterfactual Accuracy

| Graph | HHCRA CF MSE | Intervention-Only MSE | CF beats Intervention-Only |
|-------|-------------|----------------------|---------------------------|
| chain | 0.000 | 0.007 | Yes |
| fork | 0.000 | 0.006 | Yes |
| collider | 0.000 | 1.202 | Yes |
| diamond | 0.007 | 0.030 | Yes |
| complex | 0.069 | 0.112 | Yes |

HHCRA counterfactual predictions beat the intervention-only baseline on **5/5 graphs** (v0.7.0: variable-space SCM fitting with ABP/total-effect ensemble).

### Standard Benchmarks (v0.9.0)

Evaluated on Asia (8 vars, 8 edges) and Sachs (11 vars, 17 edges) with linear-Gaussian SEM data generation (n=2000, not original datasets).

#### Asia — Structure Learning (SHD↓ / F1↑)

| Method | SHD | F1 | TPR | FDR |
|--------|-----|-----|-----|-----|
| **CAD (ours, v0.10)** | **3** | **0.824** | **0.875** | **0.222** |
| **VCD (ours, v0.9)** | **3** | **0.824** | **0.875** | **0.222** |
| NOTEARS | 5 | 0.737 | 0.875 | 0.364 |
| GES | 7 | 0.222 | 0.125 | 0.000 |
| PC | 9 | 0.182 | 0.125 | 0.667 |

#### Sachs — Structure Learning (SHD↓ / F1↑)

| Method | SHD | F1 | TPR | FDR |
|--------|-----|-----|-----|-----|
| **CAD (ours, v0.10)** | **8** | **0.765** | **0.765** | **0.235** |
| VCD (ours, v0.9) | 15 | 0.571 | 0.588 | 0.444 |
| GES | 16 | 0.111 | 0.059 | 0.000 |
| NOTEARS | 21 | 0.400 | 0.412 | 0.611 |
| PC | 21 | 0.000 | 0.000 | 1.000 |

**CAD (Coupling-Adjusted Discovery)** is introduced in v0.10.0. On Sachs, it reduces SHD from 15 (VCD) to 8 — a 47% improvement. On Asia, it matches VCD. See [CAD Algorithm](#kkce-algorithm) below.

### ODE Integration Accuracy

| Method | dt | MSE |
|--------|------|------|
| Euler | 0.01 | 1.31e-03 |
| RK4 | 0.01 | 3.47e-19 |
| DOPRI5 | 0.01 | 3.83e-26 |

RK4 and DOPRI5 achieve near-machine-precision integration error on the benchmark ODE system.

## CAD Algorithm

**Coupling-Adjusted Discovery (CAD)** is a v0.10.0 algorithm that extends VCD by incorporating three mathematical frameworks from physics and topology:

### Mathematical Foundations

1. **coupling adjustment** (coupling adjustment): Each variable has a score combining conditional variance and coupling strength. The coupling strength comes from off-diagonal precision matrix entries. Hub parents with many connections have strong coupling that compensates for their lower conditional variance.

2. **partial correlation filtering** (edge filtering): Edges must satisfy statistical coherence — they should participate in causal chains). Isolated edges with weak partial correlations are lacking statistical support and get pruned.

3. **BIC local search** (DAG refinement): The causal DAG is a BIC-optimal structure that minimizes BIC score). Local modifications (add/remove edges) are accepted only if they reduce BIC score, subject to topological constraints.

### Key Innovation: Coupling Strength Sweep

The algorithm sweeps coupling strength β from 0 to 0.8 and selects the ordering that minimizes BIC:
- At β=0: pure conditional variance ordering (equivalent to VCD)
- At β=K_c (critical coupling): optimal ordering for the graph
- BIC automatically selects the right β for each graph topology

This resolves VCD's limitation on hub-heavy graphs (Sachs): hub parents like PKA (6 children) have low conditional variance but high coupling strength, so the coupling correction pushes them earlier in the ordering.

### Limitations

- Same as VCD: requires unequal noise variances, linear Gaussian assumption.
- BIC sweep adds computational cost: O(|β_values| × d³ × n) vs VCD's O(d³ × n).
- The coupling correction helps dense hub-heavy graphs but provides no benefit for sparse graphs.

## VCD Algorithm

**Variance Cascade Discovery (VCD)** is a novel causal discovery algorithm introduced in v0.9.0. It operates on cross-sectional data without requiring temporal observations, conditional independence tests (PC), greedy equivalence search (GES), or continuous DAG optimization (NOTEARS).

### Four-Phase Architecture

1. **Node Spectral Characterization**: Compute precision matrix $\Omega = \Sigma^{-1}$, extract node frequencies ($\Omega_{ii}$), marginal energies ($\text{Var}(X_i)$), and quality factors ($Q_i = \text{Var}(X_i) \cdot \Omega_{ii}$).

2. **Partial Correlation Analysis**: Decompose precision matrix into spectral modes via eigendecomposition. Compute partial correlations and mode-weighted coupling strengths.

3. **Iterative Conditional Variance Cascade Ordering**: Identify causal ordering by iteratively selecting the variable with highest conditional variance $\text{Var}(X_i | X_{-i}) = 1/\Omega_{ii}$ from the sub-precision matrix of remaining variables. Root causes have highest conditional variance (their intrinsic noise dominates); effects have lowest (small residual noise). Remove identified root and recompute.

4. **Forward Regression with Backward Elimination**: Given the causal ordering, for each variable regress on all predecessors and apply backward elimination with F-test (p<0.01) to find the minimal parent set. Partial-correlation pruning removes edges with weak partial correlations.

### Key Properties

- **Identifiability assumption**: Linear SEM with unequal noise variances (root noise >> effect noise). This is satisfied in most synthetic benchmarks where root variables have unit variance and effects have small additive noise.
- **Complexity**: $O(d^3 \cdot n)$ for ordering ($d$ iterations of $O(d^2)$ matrix inversion on $n$ samples), $O(d^2 \cdot n)$ for edge selection. Practical for $d < 100$.
- **No hyperparameters**: The F-test threshold (6.63, corresponding to p=0.01) and edge threshold (0.08) are statistically motivated, not tuned per dataset.

### Limitations

- Requires unequal noise variances for ordering identifiability. With equal noise, the conditional variance cascade cannot distinguish roots from effects.
- Linear Gaussian assumption. Non-linear or non-Gaussian data requires extensions.
- Sachs performance (SHD=15, TPR=0.588) leaves room for improvement on dense hub-heavy graphs.

## Failure Analysis

The primary failure mode is the **variable alignment problem** in Layer 1 (C-JEPA):

1. Slot attention assigns 8 fixed slots regardless of the true number of causal variables (3--8 in benchmarks).
2. Learned slots do not correspond 1:1 to true causal variables.
3. Excess slots introduce spurious edges, inflating SHD by 8--15 compared to methods that operate on true variables.

This is evidenced by the gap between standalone Granger/NOTEARS (operating on true variables) and the full pipeline (operating on slot-extracted variables). For example, on the fork graph: Granger SHD=1 vs. Pipeline SHD=16.

**v0.8.0 mitigations** (partial, not fully resolved):
- Adaptive slot pruning: tracks per-slot utilization and filters edges to active slots only, reducing but not eliminating spurious edges.
- Independence regularization: decorrelation loss on slot templates encourages diverse representations, but does not guarantee bijective alignment.
- Temporal Granger warm init: exploits temporal asymmetry for better initial W, partially bypassing the slot alignment problem.

In v0.6.0, this problem is partially circumvented by passing raw data directly to Layer 2, bypassing Layer 1. This improves structure learning metrics but undermines the end-to-end architecture claim.

## Limitations

1. **Variable alignment bottleneck**: Slot attention does not guarantee bijective slot-to-variable correspondence. This is the primary obstacle to end-to-end performance.
2. **DAG assumption**: NOTEARS enforces acyclicity. Cyclic causal structures are not supported.
3. **Linear structural model**: NOTEARS fitting loss assumes linear mechanisms. The NOTEARS-MLP variant for nonlinear relationships is not implemented.
4. **Synthetic data only**: All evaluations use synthetic linear-Gaussian data generated from known graphs. No evaluation on real observational datasets (e.g., original Sachs flow cytometry data).
5. **Small scale**: Evaluated on graphs with 3--11 variables. Scalability to graphs with 50+ variables is untested.
6. **Counterfactual noise model**: ABP assumes additive Gaussian exogenous noise. Performance degrades under non-Gaussian or heteroscedastic noise.
7. **Computational cost**: Neural ODE integration (RK4) and ensemble training in v0.6.0 incur significant per-run cost.
8. **Unvalidated extensions**: Liquid Causal Graph, Invariant Finder, and Autofeedback Network are implemented but have no benchmark evidence of improving over the base architecture.

## Dependencies

- **Full architecture** (`hhcra/`): Python >= 3.8, PyTorch >= 2.0, NumPy >= 1.24, SciPy >= 1.10, pytest >= 7.0. ODE integration (RK4, DOPRI5) is implemented directly; `torchdiffeq` is not used.
- **Standalone prototype** (`hhcra_v2.py`): NumPy and SciPy only. No PyTorch dependency.

## Usage

```bash
pip install -e ".[dev]"
pytest tests/ -v          # 215+ tests
python -m hhcra.main      # Run toy benchmark suite
```

## Changelog

### v0.10.0
- **CAD (Coupling-Adjusted Discovery)**: Novel algorithm combining coupling-adjusted ordering, partial correlation topological filtering, and BIC refinement.
  - Sachs: SHD=8, F1=0.765 (47% SHD improvement over VCD).
  - Asia: SHD=3, F1=0.824 (matches VCD).
  - coupling strength sweep: adaptive coupling sweep selects optimal β per graph topology.
- 6 new CAD tests, all passing.

### v0.9.0
- **VCD (Variance Cascade Discovery)**: Novel causal discovery algorithm that beats all standard baselines.
  - Asia: SHD=3, F1=0.824 (vs NOTEARS SHD=5, PC SHD=9, GES SHD=7).
  - Sachs: SHD=15, F1=0.571 (vs GES SHD=16, NOTEARS SHD=21, PC SHD=21).
- HHCRA-Integrated method: combines temporal Granger (multi-threshold + BIC cross-validation) with partial correlation validation and residual variance orientation.
- GES (Greedy Equivalence Search) baseline with BIC scoring.
- Interventional and counterfactual evaluation framework with ground-truth computation for linear SEMs.
- Layer 1 ablation study: confirms C-JEPA slot attention outperforms random projection and PCA baselines.
- Comprehensive v0.9.0 test suite (25/26 passing).

### v0.8.0
- Adaptive slot count: slot utilization tracking in SlotAttention + `get_active_slots()` pruning for automatic variable count detection. `symbolic_graph()` now accepts `active_slots` parameter.
- Independence regularization: decorrelation loss on slot templates forces diverse slot-variable correspondence.
- Hybrid temporal-NOTEARS warm initialization: temporal Granger regression (SHD 1-4) replaces cross-sectional NOTEARS for W initialization when temporal data is available.
- Performance fix: adaptive iteration count in `_warm_init_notears` (data-proportional instead of hardcoded 30×100). Early stopping when DAG constraint satisfied.
- 190 tests passing.

### v0.7.0
- Replaced latent-space ABP counterfactual with variable-space SCM fitting.
- New counterfactual pipeline: partial-correlation skeleton discovery, variance-based edge orientation, OLS coefficient estimation, ABP/total-effect ensemble.
- Counterfactual accuracy improved from 1/5 to **5/5 graphs** beating the intervention-only baseline.

### v0.6.0
- Added standard benchmark graphs (Asia, Sachs, Alarm, Insurance, Erdos-Renyi).
- Added baseline runners for PC, Granger, NOTEARS, Random, Empty.
- Introduced raw-data bypass for Layer 2 warm initialization (improves structure learning, bypasses Layer 1).
- Added ensemble training over multiple seeds with model selection.
- Added experimental modules: Liquid Causal Graph, Trajectory Invariant Finder, Iterative Refinement Loop, HHCRAExperimental orchestrator.
- Extended test suite to 267 tests.

### v0.5.0
- Fixed incorrect adjacency matrix usage in counterfactual ABP.
- Replaced `list.pop(0)` with `collections.deque.popleft()` in all BFS traversals.
- Vectorized Layer 1 `compute_loss` over the temporal dimension.
- Replaced recursive `_power_subsets` with `itertools.combinations`.
- Added gradient clipping (`max_norm=5.0`) to all three training stages.
- Clamped DAG penalty $h(W) \geq 0$ to prevent negative values from floating-point error.
- Extended `HHCRAConfig` validation.
- 170 tests passing.

### v0.4.1
- Slot attention changed from independent sigmoid gating to competitive softmax.
- NOTEARS loss changed to per-variable formulation.
- Layer 3 training fixed: loss computed from HRM output tensor.

## References

1. Pearl, J. (2009). *Causality: Models, Reasoning, and Inference*. Cambridge University Press.
2. Zheng, X. et al. (2018). DAGs with NO TEARS: Continuous Optimization for Structure Learning. *NeurIPS*.
3. Tian, J. & Pearl, J. (2002). On the Identification of Causal Effects. *UAI*.
4. Hasani, R. et al. (2021). Liquid Time-constant Networks. *AAAI*.
5. Graves, A. (2016). Adaptive Computation Time for Recurrent Neural Networks. *arXiv:1603.08983*.
6. Locatello, F. et al. (2020). Object-Centric Learning with Slot Attention. *NeurIPS*.

## Reproduction

```bash
python scripts/run_verification.py    # Toy benchmarks (v0.4 report)
python scripts/run_v060_benchmark.py  # Standard benchmarks (Asia, Sachs)
```

Random seed: 42. All operations are deterministic on CPU.
