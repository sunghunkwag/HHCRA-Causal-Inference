"""
Structural Causal Model (SCM) fitting.

Given a discovered DAG and observational data, fits a linear SCM
    X_j = sum_{i in pa(j)} w_ij * X_i + e_j
using OLS regression for each node given its parents.

The fitted SCM supports:
    - Interventional queries (do-calculus)
    - Counterfactual reasoning (abduction-action-prediction)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field


@dataclass
class LinearSCM:
    """A fitted linear structural causal model.

    Attributes
    ----------
    adj : (d, d) binary adjacency matrix. adj[i,j]=1 means i->j.
    weights : (d, d) weighted adjacency. weights[i,j] = coefficient.
    noise_std : (d,) estimated noise standard deviation per node.
    intercepts : (d,) intercept per node.
    topo_order : list of node indices in topological order.
    d : number of variables.
    """

    adj: NDArray
    weights: NDArray
    noise_std: NDArray
    intercepts: NDArray
    topo_order: list[int]
    d: int = field(init=False)

    def __post_init__(self):
        self.d = self.adj.shape[0]

    def predict(self, exogenous: NDArray | None = None, seed: int = 0) -> NDArray:
        """Forward-sample from the SCM.

        Parameters
        ----------
        exogenous : (n, d) noise terms. If None, sampled from fitted noise.
        seed : Random seed (used if exogenous is None).

        Returns
        -------
        (n, d) generated data.
        """
        if exogenous is None:
            rng = np.random.RandomState(seed)
            n = 1000
            exogenous = np.column_stack([
                rng.randn(n) * self.noise_std[j]
                for j in range(self.d)
            ])

        n = exogenous.shape[0]
        X = np.zeros((n, self.d), dtype=np.float64)

        for j in self.topo_order:
            parents = np.where(self.adj[:, j] > 0)[0]
            X[:, j] = self.intercepts[j] + exogenous[:, j]
            if len(parents) > 0:
                X[:, j] += X[:, parents] @ self.weights[parents, j]

        return X


def fit_linear_scm(X: NDArray, adj: NDArray) -> LinearSCM:
    """Fit a linear SCM given data and a DAG structure.

    For each node j, fits:
        X_j = intercept_j + sum_{i in pa(j)} w_ij * X_i + e_j

    Parameters
    ----------
    X : (n, d) observational data.
    adj : (d, d) binary adjacency matrix.

    Returns
    -------
    LinearSCM instance with fitted parameters.
    """
    X = np.asarray(X, dtype=np.float64)
    adj = np.asarray(adj, dtype=np.float64)
    n, d = X.shape

    weights = np.zeros((d, d), dtype=np.float64)
    noise_std = np.zeros(d, dtype=np.float64)
    intercepts = np.zeros(d, dtype=np.float64)

    # Topological order via Kahn's
    from coda.metrics import topological_order_from_dag
    topo = topological_order_from_dag(adj)

    for j in topo:
        parents = np.where(adj[:, j] > 0)[0]
        y = X[:, j]

        if len(parents) == 0:
            intercepts[j] = y.mean()
            residuals = y - intercepts[j]
        else:
            # OLS with intercept
            X_pa = np.column_stack([np.ones(n), X[:, parents]])
            beta = np.linalg.lstsq(X_pa, y, rcond=None)[0]
            intercepts[j] = beta[0]
            weights[parents, j] = beta[1:]
            residuals = y - X_pa @ beta

        noise_std[j] = max(np.std(residuals), 1e-10)

    return LinearSCM(
        adj=adj,
        weights=weights,
        noise_std=noise_std,
        intercepts=intercepts,
        topo_order=topo,
    )
