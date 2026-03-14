# Phase C: Counterfactual Accuracy Results

Comparison of counterfactual prediction accuracy.
Counterfactual: 'What would Y have been if X were x' instead of x?'
Seed: 42. Data: B=16, T=20.

## Method Description

- **HHCRA Counterfactual (v0.7.0)**: Variable-space SCM fitting + ABP ensemble.
  Step 1: Learn DAG skeleton via partial correlations (precision matrix).
  Step 2: Orient edges using variance ordering (root noise >> mechanism noise).
  Step 3: Fit OLS coefficients on discovered parents.
  Step 4: ABP (Abduction-Action-Prediction) in variable space.
  Step 5: Total-effect estimation as fallback for Markov-equivalent edges.
  Per-target selection: use ABP when methods agree, total-effect when they disagree.

- **Intervention-only baseline**: Computes do(X=x') using the TRUE graph
  with TRUE coefficients but sets noise to zero. Error = noise magnitude.

## Results Table

| Graph | HHCRA CF MSE | Int-Only MSE | HHCRA CF beats Int-Only | Test Pairs |
|-------|-------------|-------------|------------------------|------------|
| chain | 0.000075 | 0.006998 | Yes | 6 |
| fork | 0.000199 | 0.005751 | Yes | 2 |
| collider | 0.000200 | 1.202451 | Yes | 2 |
| diamond | 0.007256 | 0.029942 | Yes | 5 |
| complex | 0.068552 | 0.112186 | Yes | 21 |

## Success Criterion

HHCRA CF beats Int-Only on **5/5 graphs**. Target (>= 3/5): **MET**

## Analysis

v0.7.0 fixed the counterfactual prediction by operating in the original
variable space rather than in the C-JEPA latent space. The prior approach
(v0.4) computed ABP in the latent representation, where the noise estimation
was meaningless because latent dimensions don't correspond to structural
equations. The new approach learns the SCM directly from variable data using
partial correlations for skeleton discovery and variance-based orientation,
then applies proper ABP with OLS-fitted coefficients. An ensemble with
total-effect estimation handles edges where direction is unidentifiable
from Gaussian data (Markov equivalence).

## Reproduction

```bash
python -c "from hhcra.verification import run_phase_c; run_phase_c()"
```
