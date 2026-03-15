# CODA: Cross-validated Ordering for DAG Alignment

A varsortability-aware causal discovery framework with honest benchmarking.

## Status: Research in Progress

**CODA does not beat standard causal discovery methods (PC, GES, LiNGAM) on current benchmarks.** It beats the trivial sortnregress baseline, but that is not a meaningful contribution — sortnregress exists to diagnose benchmark artifacts, not as a real competitor.

This repository is published for transparency. The honest results are below.

## Benchmark Results (SHD, lower = better)

All experiments on synthetic linear SEM data. Varsortability and R²-sortability reported for every dataset.

| Dataset | V-sort | empty | SNR | PC | GES | LiNGAM | CODA |
|---------|--------|-------|-----|----|----|--------|------|
| Asia raw (d=8, 8 edges) | 1.000 | 8 | 5 | **1** | **0** | **0** | 6 |
| Asia standardized | 0.750 | 8 | 14 | **1** | **0** | 7 | 6 |
| Asia low-weight | 0.750 | 8 | 11 | **0** | **0** | 8 | 5 |
| Sachs synthetic (d=11, 17 edges) | 0.824 | 17 | 27 | **10** | 15 | **1** | 12 |
| Sachs standardized | 0.706 | 17 | 29 | **10** | 15 | 12 | 12 |
| ER-2 d=20 raw (34 edges) | 0.941 | 34 | 57 | **16** | 20 | **15** | 53 |
| ER-2 d=20 standardized | 0.529 | 34 | 125 | **16** | 20 | 64 | 53 |

**Winner by dataset:** PC/GES/LiNGAM win every benchmark. CODA is competitive with LiNGAM only on standardized data.

## What CODA Does

Three-stage ordering-based structure learning:
1. Generate candidate topological orderings (conditional variance, random restarts, variance sort)
2. For each ordering, fit parents via cross-validated Lasso (LassoCV + adaptive noise floor)
3. Score each DAG on held-out data using BIC; return best

## Why It Doesn't Work (Yet)

1. **PC uses conditional independence tests** which directly encode the Markov property. CODA's ordering + regression approach is less principled.
2. **GES has provable consistency** for Gaussian models with BIC score. CODA's greedy ordering search has no such guarantee.
3. **LiNGAM exploits non-Gaussianity** for full identifiability. CODA doesn't use distribution shape information.
4. **CODA's ordering search is O(d³n)** per candidate, making it slower than PC (O(d^q) for sparse graphs) without being more accurate.

## What Would Make This Publishable

For arXiv-level contribution, CODA would need one of:

1. **Theoretical identifiability result**: Prove that conditional-variance ordering recovers the true topological order under specific conditions (e.g., additive noise with bounded signal-to-noise ratio).
2. **Hybrid PC-CODA**: Use CODA's ordering as initialization for PC's constraint-based refinement. If this is faster than PC alone with equal accuracy, that's a practical contribution.
3. **Real-data advantage**: Demonstrate superior performance on real (non-synthetic) data where PC/GES assumptions are violated (nonlinear, non-Gaussian, latent confounders).
4. **Scalability**: Show CODA scales better than PC/GES for d > 100 (unlikely with current O(d³n) complexity).

## Installation

```bash
pip install -e .
# Requires: numpy, scipy, scikit-learn
# Optional: causal-learn (for PC/GES/LiNGAM baselines)
pip install causal-learn
```

## Quick Start

```python
from coda import coda_discover, shd, varsortability
from coda.data import generate_linear_sem_data, ASIA_TRUE_DAG

# Generate data
X, W = generate_linear_sem_data(ASIA_TRUE_DAG, n=2000, seed=42)

# Diagnostic: check if benchmark is trivially solvable
print(f"Varsortability: {varsortability(X, ASIA_TRUE_DAG):.3f}")

# Run CODA
result = coda_discover(X, n_restarts=10, seed=42)
print(f"SHD: {shd(result['adj'], ASIA_TRUE_DAG)}")

# Compare with PC (the method you should probably use instead)
from coda.baselines import run_pc
adj_pc = run_pc(X)
print(f"PC SHD: {shd(adj_pc, ASIA_TRUE_DAG)}")
```

## Pearl's Causal Hierarchy

CODA includes inference for all three rungs of Pearl's ladder:

```python
from coda import fit_linear_scm, interventional_mean, counterfactual

scm = fit_linear_scm(X, result['adj'])

# Rung 2: E[X5 | do(X0 = 3)]
mean = interventional_mean(scm, target=5, intervention_node=0, intervention_value=3.0)

# Rung 3: Counterfactual
cf = counterfactual(scm, factual_values=X[0], intervention_node=0, counterfactual_value=5.0)
```

## Project Structure

```
coda/
├── __init__.py        # Public API
├── discovery.py       # CODA algorithm + sortnregress baselines
├── baselines.py       # PC, GES, LiNGAM wrappers (causal-learn)
├── scm.py             # Linear SCM fitting
├── inference.py       # Interventional + counterfactual queries
├── metrics.py         # SHD, F1, varsortability, R²-sortability
└── data.py            # DAG generators, Sachs/Asia ground truth
tests/                 # 80+ tests
scripts/
└── run_benchmark.py   # Full benchmark with honest reporting
```

## Running Tests & Benchmarks

```bash
pytest tests/ -v
python scripts/run_benchmark.py
```

## Known Limitations

1. **Linear models only** — no nonlinear mechanisms
2. **Synthetic benchmarks only** — not validated on real data
3. **Loses to PC/GES/LiNGAM** — the primary finding of this benchmark
4. **O(d³n) complexity** — not scalable beyond ~50 nodes
5. **No latent variables** — assumes causal sufficiency
6. **Identifiability** — linear Gaussian SEMs yield only equivalence classes

## References

- Reisach, Seiler, Weichwald. "Beware of the Simulated DAG!" NeurIPS 2021.
- Spirtes, Glymour, Scheines. "Causation, Prediction, and Search." MIT Press, 2000.
- Chickering. "Optimal Structure Identification with GES." JMLR, 2002.
- Shimizu et al. "A Linear Non-Gaussian Acyclic Model for Causal Discovery." JMLR, 2006.
- Pearl. "Causality: Models, Reasoning, and Inference." Cambridge, 2009.

## License

MIT
