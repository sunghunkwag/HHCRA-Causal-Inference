# HHCRA: Hierarchical Hybrid Causal Reasoning Architecture

A 5-component, 3-layer modular architecture for causal inference covering all three rungs of Pearl's Ladder of Causation.

## Abstract

Current deep learning architectures operate primarily on Pearl's first rung (observation/association), learning statistical correlations from data without the capacity for genuine causal reasoning. This repository presents **HHCRA (Hierarchical Hybrid Causal Reasoning Architecture)**, a hierarchical hybrid system that integrates five heterogeneous neural and symbolic components into a unified three-layer architecture capable of observational inference, interventional reasoning (do-calculus), and counterfactual computation.

The architecture decomposes the Structural Causal Model (SCM) into learnable modules:

| SCM Component | Architecture Module | Layer |
|---|---|---|
| **V** (causal variables) | C-JEPA (Causal Joint Embedding Predictive Architecture) | Layer 1 |
| **G** (graph structure) | Causal GNN with NOTEARS optimization | Layer 2 |
| **F** (structural equations) | Liquid Neural Network (Neural ODE, RK4) | Layer 2 |
| **do-calculus / counterfactuals** | Neuro-Symbolic Engine (3 rules + ID algorithm) | Layer 3 |
| **Reasoning orchestration** | HRM (GRU + Adaptive Computation Time) | Layer 3 |

## Architecture

```
┌──────────────────────────────────────────────────────┐
│  Layer 3: Neuro-Symbolic ⟷ HRM                      │  Tight Coupling
│  Full do-calculus (3 rules), ID algorithm,           │
│  instrumental variables, GRU + ACT reasoning         │
└──────────┬───────────────────────────────────────────┘
           │ Interface: .detach() — no gradient crossing
           ↕ Feedback: structure/variable revision requests
┌──────────┴───────────────────────────────────────────┐
│  Layer 2: GNN ⟷ Liquid Neural Net                   │  Tight Coupling
│  NOTEARS augmented Lagrangian DAG learning +         │
│  Neural ODE (RK4) mechanism modeling                 │
└──────────┬───────────────────────────────────────────┘
           │ Interface: .detach() — no gradient crossing
           ↕ Feedback: variable resolution adjustment
┌──────────┴───────────────────────────────────────────┐
│  Layer 1: C-JEPA                                     │  Standalone
│  Slot attention, temporal smoothing,                 │
│  mask-prediction learning                            │
└──────────────────────────────────────────────────────┘

External Systems (Phase 7-10):
  World Model → prediction grounding & error correction
  Self-Modification → RSI loop (evaluate → bottleneck → modify → verify)
  Perception → vision, time-series, text → latent slots
  Agent → closed-loop: perceive → reason → plan → act → learn
```

### Connection Types

1. **Tight Coupling** — Shared gradients within layers (GNN⟷Liquid Net, Neuro-Symbolic⟷HRM)
2. **Interface Coupling** — `.detach()` between layers, no gradient crossing, staged training
3. **Feedback Coupling** — Top-down diagnostic signals from Layer 3 to Layers 2 and 1

## Implementation Status

All components implemented in PyTorch with full gradient flow:

| Phase | Component | Status |
|---|---|---|
| 1 | PyTorch Migration (nn.Module, staged training) | Complete |
| 2 | NOTEARS Structure Learning (augmented Lagrangian, SHD/TPR/FDR) | Complete |
| 3 | Neural ODE Liquid Net (RK4, adaptive, per-variable dynamics) | Complete |
| 4 | Full do-calculus (3 rules, ID algorithm, IV detection, ABP) | Complete |
| 5 | GRU HRM (ACT, learned reset, H/L-module timescales) | Complete |
| 6 | Benchmark Suite (5 graphs, baselines, metrics) | Complete |
| 7 | World Model (physics env, grounding loop, error routing) | Complete |
| 8 | Self-Modification (RSI loop, bottleneck detection, reversible mods) | Complete |
| 9 | Perception (vision CNN, time-series, text, modality router) | Complete |
| 10 | Agent Loop (causal bandit, gridworld, closed-loop reasoning) | Complete |

## Pearl's Ladder Coverage

| Rung | Operation | Implementation |
|---|---|---|
| **Rung 1: Observation** | P(Y\|X) | C-JEPA + GNN + Liquid Net joint encoding |
| **Rung 2: Intervention** | P(Y\|do(X)) | do-calculus (3 rules) + ID algorithm + ODE edge-cutting |
| **Rung 3: Counterfactual** | P(Y\_{x'}\|X=x, Y=y) | Abduction-Action-Prediction (MLE noise, modified SCM) |

## Key Upgrades from v2 Prototype

### Phase 2: NOTEARS Structure Learning
- Replaces simplified Granger scoring with continuous optimization
- Augmented Lagrangian: `min F(W) + λ·h(W) + (ρ/2)·h(W)²` where `h(W) = tr(e^(W∘W)) - d`
- Causal discovery metrics: SHD, TPR, FDR against ground truth

### Phase 3: Neural ODE Mechanisms
- RK4 integration replacing fixed-step Euler
- Per-variable ODE: `dx_i/dt = f_i(x_j : A[i,j]>0)` with learned f_i
- Intervention do(X_i=x): cut row i, clamp x_i, re-integrate

### Phase 4: Complete Neuro-Symbolic Engine
- Do-calculus 3 rules (insertion/deletion of observations, action/observation exchange, insertion/deletion of actions)
- ID algorithm (Tian & Pearl 2002) for general identifiability
- Instrumental variable detection when backdoor/frontdoor fail
- Proper ABP counterfactual: abduction (MLE noise) → action (modify SCM) → prediction (propagate with U)

### Phase 5: GRU-based HRM
- H-module and L-module are GRU recurrent networks
- Adaptive Computation Time (ACT): learned halting probability
- Learned reset: H-module decides WHEN to reset L-module

## Verification

144 tests across 8 test files:

```
tests/test_layer1.py     — C-JEPA (slots, gradients, temporal smoothing)
tests/test_layer2.py     — GNN + Liquid (NOTEARS, RK4, tight coupling)
tests/test_layer3.py     — NeuroSym + HRM (d-sep, do-calculus, ACT, GRU)
tests/test_pipeline.py   — Full pipeline, gradient isolation, staged training
tests/test_benchmarks.py — 12 original tests + causal discovery metrics
tests/test_world_model.py— Physics env, grounding loop, error routing
tests/test_self_mod.py   — RSI loop, bottleneck detection, reversible mods
tests/test_perception.py — Vision, time-series, text, modality router
tests/test_agent.py      — Causal bandit, gridworld, agent episodes

Result: 144/144 tests passed
```

## File Structure

```
HHCRA-Causal-Inference/
├── README.md
├── LICENSE
├── requirements.txt
├── setup.py
├── hhcra/
│   ├── __init__.py
│   ├── config.py              # HHCRAConfig
│   ├── causal_graph.py        # CausalGraphData + CausalQueryType
│   ├── layer1_cjepa.py        # C-JEPA (slot attention, mask prediction)
│   ├── layer2_mechanism.py    # GNN (NOTEARS) + Liquid Net (Neural ODE)
│   ├── layer3_reasoning.py    # Neuro-Symbolic (do-calculus, ID) + HRM (GRU, ACT)
│   ├── interfaces.py          # LayerInterface + FeedbackRouter
│   ├── architecture.py        # HHCRA main class (staged training)
│   ├── benchmarks.py          # 5-graph benchmark suite + baselines
│   ├── world_model.py         # World model + grounding loop
│   ├── self_modification.py   # RSI loop (evaluate → modify → verify)
│   ├── perception.py          # Vision/TimeSeries/Text perception
│   ├── agent.py               # Closed-loop causal agent
│   └── main.py                # Demo + benchmark runner
├── environments/
│   ├── simple_physics.py      # Spring-connected objects
│   ├── causal_bandit.py       # Multi-armed bandit with causal structure
│   └── causal_gridworld.py    # Grid with causal object relationships
├── tests/
│   ├── conftest.py
│   ├── test_layer1.py
│   ├── test_layer2.py
│   ├── test_layer3.py
│   ├── test_pipeline.py
│   ├── test_benchmarks.py
│   ├── test_world_model.py
│   ├── test_self_mod.py
│   ├── test_perception.py
│   └── test_agent.py
└── hhcra_v2.py                # Original NumPy prototype (kept for reference)
```

## Usage

```bash
# Install
pip install -e ".[dev]"

# Run full demo
python -m hhcra.main

# Run tests
pytest tests/ -v

# Run original prototype
python hhcra_v2.py
```

Requirements: Python 3.8+, PyTorch >= 2.0, NumPy, SciPy, torchdiffeq

## Theoretical Foundation

The architecture is grounded in Pearl's Structural Causal Model framework:

- **SCM = (V, U, F, G)** where V is endogenous variables, U is exogenous noise, F is structural equations, G is the causal graph
- **Three Rungs of Causation**: Association P(Y|X), Intervention P(Y|do(X)), Counterfactual P(Y_{x'}|X=x, Y=y)
- **NOTEARS** (Zheng et al. 2018): Continuous optimization for DAG structure learning via acyclicity constraint h(W) = tr(e^(W∘W)) - d = 0
- **ID Algorithm** (Tian & Pearl 2002): General identifiability beyond backdoor/frontdoor criteria
- **do-calculus**: Pearl's three rules for reducing interventional distributions to observational ones

## References

- Pearl, J. (2009). *Causality: Models, Reasoning, and Inference*. Cambridge University Press.
- Zheng, X. et al. (2018). DAGs with NO TEARS: Continuous Optimization for Structure Learning. NeurIPS 2018.
- Tian, J. & Pearl, J. (2002). On the Identification of Causal Effects. UAI 2002.
- Nam, H. et al. (2026). Causal-JEPA: Learning World Models through Object-Level Latent Interventions. arXiv:2602.11389.
- Hasani, R. et al. (2021). Liquid Time-constant Networks. AAAI 2021.
- Wang, G. et al. (2025). Hierarchical Reasoning Model. arXiv:2506.21734.
- Graves, A. (2016). Adaptive Computation Time for Recurrent Neural Networks. arXiv:1603.08983.
- LeCun, Y. (2022). A Path Towards Autonomous Machine Intelligence. OpenReview.

## License

MIT License
