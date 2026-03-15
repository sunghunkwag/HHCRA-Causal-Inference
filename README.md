# HHCRA v2.0: Hierarchical Hybrid Causal Reasoning Architecture

A 3-layer causal reasoning system rebuilt from failure analysis of the original HHCRA.

## What Changed from v1.x

The original HHCRA (v0.1–v0.10) had a PyTorch-based 3-layer architecture. Deep analysis revealed that most components were broken:

| Component | Original | Failure | v2.0 Replacement |
|-----------|----------|---------|-------------------|
| Layer 1 | C-JEPA SlotAttention | 8 fixed slots, no spatial structure → V-alignment failure (SHD 12-16) | ICA + noise-floor dim detection |
| Layer 2 structure | NOTEARS neural GNN | Neural training degraded Granger warm-init | CODA (CV ordering + held-out BIC) |
| Layer 2 mechanism | Liquid Neural ODE | Trained on reconstruction, not causal mechanisms | OLS SCM fitting |
| Layer 3 symbolic | d-sep, backdoor, do-calculus | **Working correctly** | Preserved as-is |
| Layer 3 HRM | GRU + ACT reasoning | loss=0.01*norm (no causal objective) | Removed (direct dispatch) |
| Agent | perceive/plan/act/ground | Good design, broken model underneath | Preserved + info-theoretic intervention selection |

## Architecture

```
Layer 1: ICA Variable Extraction
    - FastICA with logcosh nonlinearity
    - Noise-floor dimensionality detection
    - Direct mode when obs_dim <= n_vars

Layer 2: CODA Structure Learning + OLS SCM
    - Cross-validated ordering with held-out BIC scoring
    - Conditional variance ordering (not varsortability-dependent)
    - OLS coefficient estimation per node

Layer 3: NeuroSymbolicEngine (from original HHCRA)
    - d-Separation (Bayes-Ball algorithm)
    - Backdoor/frontdoor criteria
    - Complete do-calculus (3 rules)
    - ABP counterfactual (Abduction-Action-Prediction)
    - Identifiability checking

Agent: ActiveCausalAgent
    - Beta-distributed edge beliefs
    - Information-theoretic intervention selection
    - Contrastive signal from do-vs-observe comparison
```

## Pearl's Causal Hierarchy

All three rungs of Pearl's ladder:

```python
from hhcra import HHCRA, CausalQueryType
from coda.data import generate_linear_sem_data, ASIA_TRUE_DAG

X, _ = generate_linear_sem_data(ASIA_TRUE_DAG, n=2000, seed=42)
model = HHCRA()
model.fit(X)

# Rung 1: P(Y|X)
r1 = model.query(CausalQueryType.OBSERVATIONAL, X=0, Y=3)

# Rung 2: P(Y|do(X=2))
r2 = model.query(CausalQueryType.INTERVENTIONAL, X=0, Y=3, x_value=2.0)

# Rung 3: P(Y_{x'}|X=x,Y=y)
r3 = model.query(CausalQueryType.COUNTERFACTUAL, X=0, Y=3,
                 factual_values=X[0], cf_x=X[0,0]+2.0)
```

## Benchmark Results

| Dataset | d | edges | SHD | F1 | V-sort | empty SHD |
|---------|---|-------|-----|-----|--------|-----------|
| Chain (4,3) | 4 | 3 | 2 | 0.333 | 1.000 | 3 |
| Fork (3,2) | 3 | 2 | 1 | 0.500 | 0.500 | 2 |
| Collider (3,2) | 3 | 2 | 0 | 1.000 | 1.000 | 2 |
| Asia (8,8) | 8 | 8 | 6 | 0.526 | 1.000 | 8 |
| Asia STD | 8 | 8 | 6 | 0.526 | 0.750 | 8 |
| Sachs (11,17) | 11 | 17 | 12 | 0.471 | 0.824 | 17 |
| ER-20 raw | 20 | 34 | 53 | 0.330 | 0.941 | 34 |

**Honest assessment:** CODA beats sortnregress (trivial baseline) on 7/8 benchmarks, but loses to PC/GES/LiNGAM (standard methods). The architecture value is in the integrated reasoning pipeline (Pearl's 3 rungs + symbolic engine + active agent), not in structure learning alone.

## Project Structure

```
hhcra/                    # HHCRA v2.0 — integrated architecture
├── __init__.py
├── architecture.py       # HHCRA main class (ICA + CODA + Symbolic)
├── symbolic.py           # NeuroSymbolicEngine (d-sep, backdoor, ABP)
├── graph.py              # CausalGraphData structures
└── agent.py              # ActiveCausalAgent (info-theoretic interventions)

coda/                     # Structure learning + inference
├── discovery.py          # CODA algorithm + sortnregress baselines
├── scm.py                # Linear SCM fitting
├── inference.py          # Interventional + counterfactual queries
├── metrics.py            # SHD, F1, varsortability, R²-sortability
├── data.py               # DAG generators, Sachs/Asia ground truth
└── baselines.py          # PC/GES/LiNGAM wrappers (requires causal-learn)

tests/                    # 114 tests
├── test_hhcra.py         # HHCRA integration tests (34 tests)
├── test_discovery.py     # CODA algorithm tests
├── test_inference.py     # Causal inference tests
├── test_metrics.py       # Metric tests
├── test_data.py          # Data generation tests
└── test_scm.py           # SCM fitting tests
```

## Installation

```bash
pip install numpy scipy scikit-learn
pip install -e .

# Optional: PC/GES/LiNGAM baselines
pip install causal-learn
```

## Running Tests

```bash
pytest tests/ -v --ignore=tests/test_baselines.py  # without causal-learn
pytest tests/ -v                                      # with causal-learn
```

## Known Limitations

1. **Linear models only.** Assumes X_j = Σ w_ij X_i + e_j.
2. **Loses to PC/GES on structure learning.** CODA's ordering-based approach is less principled than conditional independence testing.
3. **Synthetic benchmarks only.** Not validated on real Sachs flow cytometry data.
4. **No latent confounders.** Assumes causal sufficiency.
5. **Scale limited.** Tested up to d=20 nodes.

## References

- Pearl. "Causality: Models, Reasoning, and Inference." Cambridge, 2009.
- Reisach et al. "Beware of the Simulated DAG!" NeurIPS 2021.
- Spirtes, Glymour, Scheines. "Causation, Prediction, and Search." MIT Press, 2000.
- Sachs et al. "Causal Protein-Signaling Networks." Science 2005.

## License

MIT
