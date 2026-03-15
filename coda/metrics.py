"""
Evaluation metrics for causal discovery.

Includes structural metrics (SHD, F1) and diagnostic metrics
(varsortability, R²-sortability) following Reisach et al. (2021, 2023).

References:
    Reisach et al., "Beware of the Simulated DAG!", NeurIPS 2021.
    Reisach et al., "Scale-Free Structure Learning", NeurIPS 2023.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from itertools import permutations


# ---------------------------------------------------------------------------
# Structural metrics
# ---------------------------------------------------------------------------

def shd(pred: NDArray, true: NDArray) -> int:
    """Structural Hamming Distance between two adjacency matrices.

    Counts: missing edges + extra edges + reversed edges.
    pred[i,j] = 1 means i -> j.

    Parameters
    ----------
    pred : (d, d) binary array — predicted adjacency.
    true : (d, d) binary array — ground-truth adjacency.

    Returns
    -------
    int — SHD value (lower is better, 0 = perfect).
    """
    pred = np.asarray(pred, dtype=bool)
    true = np.asarray(true, dtype=bool)
    assert pred.shape == true.shape, f"Shape mismatch: {pred.shape} vs {true.shape}"

    # Extra edges (in pred but not in true)
    extra = pred & ~true
    # Missing edges (in true but not in pred)
    missing = ~pred & true
    # Reversed: pred has j->i where true has i->j.
    # For each (j,i) in extra, check if (i,j) is in missing.
    # Each reversed pair appears once in reversed_edges (at the extra position).
    reversed_edges = extra & missing.T
    n_reversed = int(np.sum(reversed_edges))
    n_extra = int(np.sum(extra)) - n_reversed
    n_missing = int(np.sum(missing)) - n_reversed
    return n_extra + n_missing + n_reversed


def f1_score_dag(pred: NDArray, true: NDArray) -> dict:
    """Precision, recall, F1 for directed edges.

    Parameters
    ----------
    pred : (d, d) binary array.
    true : (d, d) binary array.

    Returns
    -------
    dict with keys: tp, fp, fn, precision, recall, f1.
    """
    pred = np.asarray(pred, dtype=bool)
    true = np.asarray(true, dtype=bool)

    tp = int(np.sum(pred & true))
    fp = int(np.sum(pred & ~true))
    fn = int(np.sum(~pred & true))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


# ---------------------------------------------------------------------------
# Varsortability diagnostics (Reisach et al., 2021)
# ---------------------------------------------------------------------------

def varsortability(X: NDArray, true_dag: NDArray) -> float:
    """Fraction of directed edges where Var(child) > Var(parent).

    A value near 1.0 means marginal variance increases along causal order,
    so the trivial sortnregress baseline will perform well. Values near 0.5
    indicate no exploitable variance pattern.

    Parameters
    ----------
    X : (n, d) data matrix.
    true_dag : (d, d) binary adjacency matrix. true_dag[i,j]=1 means i->j.

    Returns
    -------
    float in [0, 1]. Fraction of edges respecting variance ordering.
    """
    X = np.asarray(X, dtype=np.float64)
    true_dag = np.asarray(true_dag, dtype=bool)
    variances = np.var(X, axis=0)

    edges = np.argwhere(true_dag)
    if len(edges) == 0:
        return 0.5  # No edges — undefined, return neutral

    count = sum(1 for i, j in edges if variances[j] > variances[i])
    return count / len(edges)


def r2_sortability(X: NDArray, true_dag: NDArray) -> float:
    """Scale-invariant sortability metric (Reisach et al., 2023).

    For each edge i->j, checks if R²(j | pa(j)) > R²(i | pa(i))
    where pa(·) denotes parents in the true DAG.
    Unlike varsortability, this is invariant to marginal rescaling.

    Parameters
    ----------
    X : (n, d) data matrix.
    true_dag : (d, d) binary adjacency matrix.

    Returns
    -------
    float in [0, 1].
    """
    X = np.asarray(X, dtype=np.float64)
    true_dag = np.asarray(true_dag, dtype=bool)
    d = X.shape[1]

    # Compute R² for each node
    r2 = np.zeros(d)
    for j in range(d):
        parents = np.where(true_dag[:, j])[0]
        if len(parents) == 0:
            r2[j] = 0.0  # Root nodes have R² = 0
        else:
            X_pa = X[:, parents]
            # OLS: R² = 1 - SS_res / SS_tot
            X_pa_c = X_pa - X_pa.mean(axis=0)
            y_c = X[:, j] - X[:, j].mean()
            ss_tot = np.sum(y_c ** 2)
            if ss_tot < 1e-15:
                r2[j] = 0.0
            else:
                beta = np.linalg.lstsq(X_pa_c, y_c, rcond=None)[0]
                residuals = y_c - X_pa_c @ beta
                ss_res = np.sum(residuals ** 2)
                r2[j] = 1.0 - ss_res / ss_tot

    edges = np.argwhere(true_dag)
    if len(edges) == 0:
        return 0.5

    count = sum(1 for i, j in edges if r2[j] > r2[i])
    return count / len(edges)


def topological_order_from_dag(dag: NDArray) -> list[int]:
    """Kahn's algorithm for topological sort.

    Parameters
    ----------
    dag : (d, d) binary adjacency matrix.

    Returns
    -------
    list of node indices in topological order.

    Raises
    ------
    ValueError if the graph contains a cycle.
    """
    dag = np.asarray(dag, dtype=bool)
    d = dag.shape[0]
    in_degree = dag.sum(axis=0).astype(int)
    queue = [i for i in range(d) if in_degree[i] == 0]
    order = []

    while queue:
        node = queue.pop(0)
        order.append(node)
        for child in range(d):
            if dag[node, child]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

    if len(order) != d:
        raise ValueError("Graph contains a cycle — not a DAG.")
    return order
