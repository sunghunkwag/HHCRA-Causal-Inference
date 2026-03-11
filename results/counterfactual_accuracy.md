# Phase C: Counterfactual Accuracy Results

Comparison of counterfactual prediction accuracy.
Counterfactual: 'What would Y have been if X were x' instead of x?'
Seed: 42. Data: B=16, T=20.

## Method Description

- **HHCRA Counterfactual**: Uses ABP (Abduction-Action-Prediction) procedure.
  Step 1: Infer exogenous noise U = Y_observed - Y_predicted.
  Step 2: Modify graph (cut incoming edges to X).
  Step 3: Propagate through modified SCM: Y_cf = f(parents|do(X=x')) + U.

- **Intervention-only baseline**: Just computes do(X=x') without noise.
  Ignores the abducted noise U, so predictions differ from factual by the
  noise component. Should be worse than proper counterfactual.

## Results Table

| Graph | HHCRA CF MSE | Int-Only MSE | HHCRA CF beats Int-Only | Test Pairs |
|-------|-------------|-------------|------------------------|------------|
| chain | 1.338807 | 0.006998 | No | 6 |
| fork | 1.484083 | 0.005751 | No | 2 |
| collider | 1.055567 | 1.202451 | Yes | 2 |
| diamond | 1.158430 | 0.029942 | No | 5 |
| complex | 0.671305 | 0.112186 | No | 21 |

## Success Criterion

HHCRA CF beats Int-Only on 1/5 graphs. Target (>= 3/5): **NOT MET**

## Analysis

The ABP procedure should improve over intervention-only predictions because
it preserves the exogenous noise structure from the factual observation.
In practice, the accuracy depends on whether C-JEPA correctly decomposes
the observation into causal variables, and whether the Liquid Net accurately
models the mechanisms.

## Reproduction

```bash
python -c "from hhcra.verification import run_phase_c; run_phase_c()"
```
