# Phase B: Interventional Accuracy Results

Comparison of interventional prediction accuracy: P(Y|do(X=x))
with x_val=2.0 for all source-target pairs in each graph.
Seed: 42. Data: B=16, T=20.

## Results Table

| Graph | Naive MSE | HHCRA MSE | Oracle MSE | Edges (learned/true) | HHCRA beats Naive |
|-------|-----------|-----------|------------|---------------------|-------------------|
| chain | 6.924618 | 0.659925 | 0.000000 | 1/3 | Yes |
| fork | 5.431816 | 0.503369 | 0.000000 | 0/2 | Yes |
| collider | 1.265557 | 0.350117 | 0.000000 | 0/2 | Yes |
| diamond | 4.766776 | 0.472293 | 0.000000 | 1/4 | Yes |
| complex | 9.108443 | 0.245919 | 0.000000 | 6/9 | Yes |

## Success Criterion

HHCRA beats Naive on 5/5 graphs. Target (>= 3/5): **MET**

## Analysis

- **Naive baseline**: Uses correlation(X,Y) * x_val, ignoring confounders.
  On fork structures, correlation includes both direct and confounded paths,
  so Naive overestimates. On chains, Naive captures some signal through
  correlation but is biased.

- **Oracle baseline**: Uses true graph structure and true coefficients.
  This is the theoretical lower bound on MSE (zero noise case).

- **HHCRA**: Must learn both structure AND mechanisms from observations.
  Errors come from: (1) C-JEPA variable misalignment, (2) incorrect
  graph structure, (3) Liquid Net mechanism approximation error.


## Reproduction

```bash
python -c "from hhcra.verification import run_phase_b; run_phase_b()"
```
