# HHCRA: Hierarchical Hybrid Causal Reasoning Architecture

A 5-component, 3-layer modular architecture for causal inference covering all three rungs of Pearl's Ladder of Causation.

## Abstract

Current deep learning architectures operate primarily on Pearl's first rung (observation/association), learning statistical correlations from data without the capacity for genuine causal reasoning. This repository presents **HHCRA (Hierarchical Hybrid Causal Reasoning Architecture)**, a hierarchical hybrid system that integrates five heterogeneous neural and symbolic components into a unified three-layer architecture capable of observational inference, interventional reasoning (do-calculus), and counterfactual computation.

The architecture decomposes the Structural Causal Model (SCM) into learnable modules:

| SCM Component | Architecture Module | Layer |
|---|---|---|
| **V** (causal variables) | C-JEPA (Causal Joint Embedding Predictive Architecture) | Layer 1 |
| **G** (graph structure) | Causal Graph Neural Network | Layer 2 |
| **F** (structural equations) | Liquid Neural Network (ODE dynamics) | Layer 2 |
| **do-calculus / counterfactuals** | Neuro-Symbolic Reasoning Engine | Layer 3 |
| **Reasoning orchestration** | Hierarchical Reasoning Model (HRM) | Layer 3 |

## Architecture

```
┌─────────────────────────────────────────────────┐
│  Layer 3: Neuro-Symbolic ⟷ HRM                 │  Tight Coupling
│  Formal causal operations + reasoning control    │
│  (d-separation, backdoor/frontdoor, do-calculus, │
│   counterfactuals, multi-step orchestration)     │
└──────────┬──────────────────────────────────────┘
           │ Interface: symbolic graph + trajectories
           ↕ Feedback: structure/variable revision requests
┌──────────┴──────────────────────────────────────┐
│  Layer 2: GNN ⟷ Liquid Neural Net              │  Tight Coupling
│  Causal structure + dynamical mechanisms         │
│  (directed graph learning, ODE-based dynamics,   │
│   DAG enforcement, intervention via do(X))       │
└──────────┬──────────────────────────────────────┘
           │ Interface: latent variable tensors
           ↕ Feedback: variable resolution adjustment
┌──────────┴──────────────────────────────────────┐
│  Layer 1: C-JEPA                                │  Standalone
│  Causal latent variable extraction               │
│  (object-level masking, slot attention,          │
│   temporal encoding)                             │
└─────────────────────────────────────────────────┘
```

### Connection Types

The architecture employs three distinct connection types between components:

1. **Tight Coupling** — Shared computation and joint optimization within layers (GNN⟷Liquid Net, Neuro-Symbolic⟷HRM)
2. **Interface Coupling** — Explicit data structures passed between layers with no gradient flow, enabling independent training
3. **Feedback Coupling** — Top-down diagnostic signals from Layer 3 to Layers 2 and 1 for self-correction (structure revision, variable refinement)

## Pearl's Ladder Coverage

| Rung | Operation | Implementation |
|---|---|---|
| **Rung 1: Observation** | P(Y\|X) | C-JEPA + GNN + Liquid Net joint encoding |
| **Rung 2: Intervention** | P(Y\|do(X)) | Neuro-symbolic do-calculus + Liquid Net edge-cutting ODE propagation |
| **Rung 3: Counterfactual** | P(Y\_{x'}\|X=x, Y=y) | Abduction-Action-Prediction via Neuro-symbolic + Liquid Net |

## Key Components

### Layer 1: C-JEPA

Inspired by [Nam et al. (2026)](https://arxiv.org/abs/2602.11389), C-JEPA extracts causally relevant latent variables through object-level masking that induces latent interventions. Unlike patch-level masking, object-level masking forces the model to reason about causal interactions between entities to predict masked object trajectories.

### Layer 2: GNN + Liquid Neural Network

- **Causal GNN**: Learns directed adjacency matrix using temporal Granger-style causality scoring with NOTEARS-style DAG enforcement and L1 sparsity
- **Liquid Net**: Models dynamical mechanisms as continuous-time ODEs (dx/dt = gate · (-x + f(x,I)) / τ), inspired by [Hasani et al. (2021)](https://arxiv.org/abs/2006.04439). Supports intervention via edge cutting and variable clamping

These two are tightly coupled: GNN topology determines Liquid Net ODE structure, and Liquid Net fitting errors inform GNN structure revision.

### Layer 3: Neuro-Symbolic + HRM

- **Neuro-Symbolic Engine**: Implements Pearl's causal inference operations — d-separation (Bayes-Ball algorithm), backdoor criterion, frontdoor criterion, identifiability checking, and counterfactual computation
- **HRM** ([Wang et al., 2025](https://arxiv.org/abs/2506.21734)): Orchestrates multi-step causal reasoning with dual-timescale recurrent modules. H-module (slow) sets reasoning strategy; L-module (fast) executes concrete computations. On convergence failure, H-module resets L-module for backtracking

## Verification

The implementation includes a 12-test verification suite:

```
[PASS] Layer 1: Latent variable extraction
[PASS] Layer 2: Graph is valid DAG
[PASS] Layer 2: Graph density reasonable
[PASS] Layer 3: d-Separation algorithm functional
[PASS] Layer 3: Identifiability check works
[PASS] Pipeline: Observational query P(X3|X0)
[PASS] Pipeline: Interventional query P(X3|do(X0=2))
[PASS] Pipeline: Counterfactual query P(Y_{x'}|X=x,Y=y)
[PASS] Pipeline: Feedback mechanism activates
[PASS] Architecture: Full forward pass completes
[PASS] HRM: Reasoning produces valid trace
[PASS] Liquid Net: Intervention changes output

Result: 12/12 tests passed
```

## Usage

```bash
# Run the complete pipeline with synthetic causal data
python hhcra_v2.py
```

Requirements: Python 3.8+, NumPy, SciPy

The synthetic data generator creates observations from a known causal graph (X0→X1→X3, X0→X2→X3, X2→X4), enabling ground-truth validation of learned causal structure and interventional/counterfactual predictions.

## Theoretical Foundation

The architecture is grounded in Pearl's Structural Causal Model framework:

- **SCM = (V, U, F, G)** where V is endogenous variables, U is exogenous noise, F is structural equations, G is the causal graph
- **Three Rungs of Causation**: Association P(Y|X), Intervention P(Y|do(X)), Counterfactual P(Y_{x'}|X=x, Y=y)
- **Identifiability**: The neuro-symbolic engine verifies whether a causal effect is identifiable from observational data using graphical criteria before attempting estimation
- **do-calculus**: Pearl's three rules for reducing interventional distributions to observational ones, implemented via backdoor and frontdoor adjustment

The architectural insight is that each SCM component maps naturally to a specific neural/symbolic module, and these modules can be organized into a hierarchical hybrid system with well-defined interfaces.

## Citation

If you use this architecture in your research, please cite:

```bibtex
@software{hhcra2026,
  title={HHCRA: Hierarchical Hybrid Causal Reasoning Architecture},
  year={2026},
  url={https://github.com/[username]/HHCRA-Causal-Inference}
}
```

## References

- Pearl, J. (2009). *Causality: Models, Reasoning, and Inference*. Cambridge University Press.
- Nam, H. et al. (2026). Causal-JEPA: Learning World Models through Object-Level Latent Interventions. arXiv:2602.11389.
- Hasani, R. et al. (2021). Liquid Time-constant Networks. AAAI 2021.
- Wang, G. et al. (2025). Hierarchical Reasoning Model. arXiv:2506.21734.
- Zheng, X. et al. (2018). DAGs with NO TEARS. NeurIPS 2018.
- LeCun, Y. (2022). A Path Towards Autonomous Machine Intelligence. OpenReview.

## License

MIT License
