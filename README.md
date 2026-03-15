# CODA: Cross-validated Ordering for DAG Alignment

A varsortability-aware causal discovery framework with honest benchmarking.

## What This Is

CODA is a causal structure learning algorithm that addresses three known failure modes in the NOTEARS/GOLEM lineage:

1. **Fixed regression thresholds** → cross-validated parent selection (LassoCV + adaptive noise floor)
2. **Variance-based ordering exploit** ([Reisach et al., 2021](https://arxiv.org/abs/2102.13647)) → conditional-variance ordering that does not exploit varsortability
3. **Single BIC ordering** → ensemble of candidate orderings scored on held-out data

## What This Is Not

- This is **not** a neural method. It is classical statistics (OLS + Lasso + BIC).
- It operates on **linear additive noise models** only. For nonlinear causal discovery, see [SCORE](https://arxiv.org/abs/2106.12801), [DiffAN](https://arxiv.org/abs/2210.06201), or [NoGAM](https://arxiv.org/abs/2304.03265).
- It has been tested on **synthetic data** and **synthetic-from-real-DAG data** (Sachs, Asia). It has **not** been validated on the original Sachs flow cytometry measurements.
- The current SHD numbers reflect performance on linear Gaussian / non-Gaussian SEMs. Real-world causal discovery remains an open problem.

## Honest Benchmarking

Every experiment reports:
- **Varsortability** and **R²-sortability** of the test data ([Reisach et al., 2021/2023](https://arxiv.org/abs/2102.13647))
- **sortnregress** baseline (sort by variance + Lasso — the trivial method that matches NOTEARS on standard benchmarks)
- **R²-sortnregress** baseline (scale-invariant variant)
- **Empty graph** baseline (SHD = number of true edges — the floor any method must beat)

If CODA does not beat sortnregress on a given dataset, the table says so.

## Installation

```bash
pip install -e .
# or
pip install numpy scipy scikit-learn
```

## Quick Start

```python
from coda import coda_discover, shd, varsortability
from coda.data import generate_linear_sem_data, ASIA_TRUE_DAG

# Generate data
X, W = generate_linear_sem_data(ASIA_TRUE_DAG, n=2000, seed=42)

# Check varsortability (diagnostic)
print(f"Varsortability: {varsortability(X, ASIA_TRUE_DAG):.3f}")

# Run CODA
result = coda_discover(X, n_restarts=10, seed=42)
print(f"SHD: {shd(result['adj'], ASIA_TRUE_DAG)}")
print(f"Strategy: {result['strategy']}")
```

## Pearl's Causal Hierarchy

CODA includes inference for all three rungs:

```python
from coda import fit_linear_scm, interventional_mean, counterfactual

# Fit SCM from discovered DAG
scm = fit_linear_scm(X, result['adj'])

# Rung 2: Interventional — E[X5 | do(X0 = 3)]
mean = interventional_mean(scm, target=5, intervention_node=0, intervention_value=3.0)

# Rung 3: Counterfactual — "What would X2 be if X0 had been 5?"
cf = counterfactual(scm, factual_values=X[0], intervention_node=0, counterfactual_value=5.0)
```

## Project Structure

```
coda/
├── __init__.py        # Public API
├── discovery.py       # CODA algorithm + sortnregress baselines
├── scm.py             # Linear SCM fitting
├── inference.py       # Interventional + counterfactual queries
├── metrics.py         # SHD, F1, varsortability, R²-sortability
└── data.py            # DAG generators, Sachs/Asia ground truth
tests/
├── conftest.py        # Shared fixtures
├── test_metrics.py    # Metric tests
├── test_data.py       # Data generation tests
├── test_discovery.py  # Discovery algorithm tests
├── test_scm.py        # SCM fitting tests
└── test_inference.py  # Causal inference tests
scripts/
└── run_benchmark.py   # Full benchmark with honest reporting
```

## Running Tests

```bash
pytest tests/ -v
```

## Running Benchmarks

```bash
python scripts/run_benchmark.py
```

## Known Limitations

1. **Linear models only.** The entire framework assumes X_j = Σ w_ij X_i + e_j. Nonlinear causal mechanisms are not handled.
2. **Synthetic benchmarks.** While the Sachs DAG structure is real, the data is synthetic. Real flow cytometry data may behave differently.
3. **Identifiability.** Linear Gaussian SEMs are identifiable only up to Markov equivalence classes. CODA picks a single DAG via BIC, which may not be the true one even with infinite data. Non-Gaussian noise improves identifiability (LiNGAM theory).
4. **Scale.** Tested up to d=20 nodes. The conditional-variance ordering step is O(d³n), limiting scalability.
5. **No latent variables.** Assumes causal sufficiency (all common causes observed).

## References

- Reisach, Seiler, Weichwald. "Beware of the Simulated DAG! Causal Discovery Benchmarks May Be Easy To Game." NeurIPS 2021.
- Reisach et al. "Scale-Free Structure Learning with Regularized Regression." NeurIPS 2023.
- Zheng et al. "DAGs with NO TEARS." NeurIPS 2018.
- Ng, Ghassami, Zhang. "On the Role of Sparsity and DAG Constraints for Learning Linear DAGs." NeurIPS 2020 (GOLEM).
- Rolland et al. "Score Matching Enables Causal Discovery of Nonlinear Additive Noise Models." ICML 2022 (SCORE).
- Pearl. "Causality: Models, Reasoning, and Inference." Cambridge, 2009.
- Sachs et al. "Causal Protein-Signaling Networks Derived from Multiparameter Single-Cell Data." Science 2005.

## License

MIT
