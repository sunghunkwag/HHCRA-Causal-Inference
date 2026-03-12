# HHCRA: Hierarchical Hybrid Causal Reasoning Architecture

A modular, three-layer architecture for causal inference spanning all three rungs of Pearl's Ladder of Causation.

## Abstract

Contemporary deep learning systems are largely confined to Pearl's first rung of causation—learning associational patterns from observational data without genuine causal reasoning capability. We present **HHCRA (Hierarchical Hybrid Causal Reasoning Architecture)**, a neuro-symbolic framework that decomposes the Structural Causal Model (SCM) into five differentiable modules organized across three hierarchical layers. The architecture supports observational inference, interventional reasoning via do-calculus, and counterfactual computation through Abduction-Action-Prediction (ABP).

| SCM Component | Module | Layer |
|---|---|---|
| **V** (variables) | C-JEPA (slot attention, mask prediction) | 1 |
| **G** (graph) | Causal GNN with NOTEARS (Zheng et al., 2018) | 2 |
| **F** (mechanisms) | Liquid Neural Network (Neural ODE, RK4) | 2 |
| **do-calculus / counterfactuals** | Neuro-Symbolic Engine (3 rules + ID algorithm) | 3 |
| **Reasoning orchestration** | HRM (GRU + Adaptive Computation Time) | 3 |

## Architecture Overview

**Layer 1 — Representation.** C-JEPA extracts causal variable representations via slot attention and temporal smoothing with mask-prediction learning.

**Layer 2 — Structure & Mechanism.** A Causal GNN learns the DAG structure through NOTEARS augmented Lagrangian optimization (`h(W) = tr(e^{W∘W}) − d = 0`), while a Liquid Neural Network models structural equations as per-variable Neural ODEs integrated via RK4.

**Layer 3 — Reasoning.** A Neuro-Symbolic Engine implements the full do-calculus (three rules), the ID algorithm (Tian & Pearl, 2002), and instrumental variable detection. A GRU-based Hierarchical Reasoning Module (HRM) with Adaptive Computation Time (ACT) orchestrates multi-step causal queries.

### Inter-Layer Coupling

- **Intra-layer (tight):** Shared gradient flow within each layer (GNN ⟷ Liquid Net; Neuro-Symbolic ⟷ HRM).
- **Inter-layer (interface):** `.detach()` boundaries between layers enforce staged training with no gradient crossing.
- **Top-down (feedback):** Layer 3 emits diagnostic signals for structure/variable revision in lower layers.

## Pearl's Ladder Coverage

| Rung | Query | Mechanism |
|---|---|---|
| 1 — Observation | P(Y\|X) | C-JEPA + GNN + Liquid Net joint encoding |
| 2 — Intervention | P(Y\|do(X)) | do-calculus rules + ID algorithm + ODE edge-cutting |
| 3 — Counterfactual | P(Y\_{x'}\|X=x, Y=y) | ABP: MLE noise abduction → SCM modification → propagation |

## Implementation

All components are implemented in PyTorch with full gradient support across 10 development phases:

1. PyTorch migration (`nn.Module`, staged training)
2. NOTEARS structure learning (augmented Lagrangian, SHD/TPR/FDR metrics)
3. Neural ODE Liquid Net (RK4, adaptive, per-variable dynamics)
4. Complete do-calculus engine (3 rules, ID algorithm, IV detection, ABP)
5. GRU-based HRM (ACT, learned reset, hierarchical timescales)
6. Benchmark suite (5 canonical graphs, baselines, metrics)
7. World model (physics environment, grounding loop, error routing)
8. Self-modification (RSI loop: evaluate → bottleneck → modify → verify)
9. Perception (vision CNN, time-series, text encoders, modality router)
10. Agent loop (causal bandit, gridworld, closed-loop reasoning)

**Verification:** 144 unit tests across 9 test files; all passing.

## Usage

```bash
pip install -e ".[dev]"
python -m hhcra.main        # demo + benchmarks
pytest tests/ -v            # test suite
```

**Requirements:** Python ≥ 3.8, PyTorch ≥ 2.0, NumPy, SciPy, torchdiffeq

## Theoretical Foundations

The architecture is grounded in Pearl's SCM framework **SCM = (V, U, F, G)**, where *V* denotes endogenous variables, *U* exogenous noise, *F* structural equations, and *G* the causal graph. Key theoretical building blocks include:

- **NOTEARS** (Zheng et al., 2018) — continuous DAG structure learning via the acyclicity constraint.
- **ID Algorithm** (Tian & Pearl, 2002) — general causal effect identifiability beyond backdoor/frontdoor criteria.
- **do-calculus** (Pearl, 2009) — three rules for reducing interventional to observational distributions.
- **Liquid Time-constant Networks** (Hasani et al., 2021) — adaptive ODE-based dynamics.
- **Adaptive Computation Time** (Graves, 2016) — learned halting for variable-depth reasoning.

## References

1. Pearl, J. (2009). *Causality: Models, Reasoning, and Inference*. Cambridge University Press.
2. Zheng, X. et al. (2018). DAGs with NO TEARS: Continuous Optimization for Structure Learning. *NeurIPS*.
3. Tian, J. & Pearl, J. (2002). On the Identification of Causal Effects. *UAI*.
4. Nam, H. et al. (2026). Causal-JEPA: Learning World Models through Object-Level Latent Interventions. *arXiv:2602.11389*.
5. Hasani, R. et al. (2021). Liquid Time-constant Networks. *AAAI*.
6. Wang, G. et al. (2025). Hierarchical Reasoning Model. *arXiv:2506.21734*.
7. Graves, A. (2016). Adaptive Computation Time for Recurrent Neural Networks. *arXiv:1603.08983*.
8. LeCun, Y. (2022). A Path Towards Autonomous Machine Intelligence. *OpenReview*.

## License

MIT License
