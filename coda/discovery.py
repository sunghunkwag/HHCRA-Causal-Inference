"""
Causal discovery algorithms.

Main algorithm: CODA (Cross-validated Ordering for DAG Alignment).
Baselines: sortnregress, R²-sortnregress (Reisach et al., 2021/2023).

CODA addresses three known failure modes of variance-based methods:
    1. Fixed regression thresholds → cross-validated parent selection
    2. Variance-based ordering (varsortability) → conditional-variance ordering
    3. Single BIC ordering → ensemble + held-out validation

References:
    Reisach et al., "Beware of the Simulated DAG!", NeurIPS 2021.
    Reisach et al., "Scale-Free Structure Learning", NeurIPS 2023.
"""

from __future__ import annotations

import warnings
import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LassoCV, Lasso
from sklearn.model_selection import KFold


# ===================================================================
# Baselines (Reisach et al., 2021/2023)
# ===================================================================

def sortnregress(X: NDArray, alpha: float | None = None) -> NDArray:
    """Var-sort + Lasso regression baseline (Reisach et al., 2021).

    Sorts variables by increasing marginal variance to recover topological
    order, then runs Lasso to prune edges. This trivial baseline matches
    NOTEARS/GOLEM on high-varsortability data.

    Parameters
    ----------
    X : (n, d) data matrix (should NOT be standardized for this to work).
    alpha : Lasso regularization. If None, uses LassoCV.

    Returns
    -------
    (d, d) binary adjacency matrix.
    """
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape

    # Sort by increasing marginal variance
    variances = np.var(X, axis=0)
    order = np.argsort(variances)  # ascending variance = topological order

    return _regress_in_order(X, order, alpha)


def r2_sortnregress(X: NDArray, alpha: float | None = None) -> NDArray:
    """R²-sort + Lasso regression baseline (Reisach et al., 2023).

    Scale-invariant version: standardize data, then sort by increasing
    marginal variance of residuals from regressing on all predecessors.

    Parameters
    ----------
    X : (n, d) data matrix.
    alpha : Lasso regularization. If None, uses LassoCV.

    Returns
    -------
    (d, d) binary adjacency matrix.
    """
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape

    # Standardize
    X_std = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-15)

    # Greedy ordering by increasing R²
    remaining = list(range(d))
    order = []

    for _ in range(d):
        if len(order) == 0:
            # First node: pick the one with smallest variance (in standardized
            # data this is ~1.0 for all, so pick by raw variance as tiebreaker)
            variances = [np.var(X_std[:, j]) for j in remaining]
            idx = remaining[np.argmin(variances)]
        else:
            # Pick the node whose R² when regressed on current order is smallest
            best_idx = None
            best_r2 = float("inf")
            X_pred = X_std[:, order]
            for j in remaining:
                y = X_std[:, j]
                y_c = y - y.mean()
                ss_tot = np.sum(y_c ** 2)
                if ss_tot < 1e-15:
                    r2 = 0.0
                else:
                    beta = np.linalg.lstsq(X_pred, y_c, rcond=None)[0]
                    resid = y_c - X_pred @ beta
                    r2 = 1.0 - np.sum(resid ** 2) / ss_tot
                    r2 = max(r2, 0.0)
                if r2 < best_r2:
                    best_r2 = r2
                    best_idx = j
            idx = best_idx

        order.append(idx)
        remaining.remove(idx)

    return _regress_in_order(X_std, order, alpha)


# ===================================================================
# CODA algorithm
# ===================================================================

def coda_discover(
    X: NDArray,
    n_restarts: int = 10,
    cv_folds: int = 5,
    val_fraction: float = 0.3,
    seed: int = 42,
    verbose: bool = False,
) -> dict:
    """CODA: Cross-validated Ordering for DAG Alignment.

    Three-stage algorithm:
        1. Generate candidate topological orderings via multiple strategies
        2. For each ordering, fit parents using cross-validated Lasso
        3. Score each DAG on held-out data using BIC; return best

    CODA is designed to work on standardized data and does NOT exploit
    varsortability. Internal standardization is applied.

    Parameters
    ----------
    X : (n, d) data matrix.
    n_restarts : Number of random ordering restarts.
    cv_folds : Folds for cross-validated parent selection.
    val_fraction : Fraction of data held out for ordering scoring.
    seed : Random seed.
    verbose : Print progress.

    Returns
    -------
    dict with keys:
        - "adj" : (d, d) binary adjacency matrix (best DAG)
        - "order" : best topological ordering
        - "bic" : BIC score of best DAG
        - "strategy" : which ordering strategy won
        - "all_scores" : dict mapping strategy name -> BIC
    """
    rng = np.random.RandomState(seed)
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape

    # Always standardize internally
    X_std = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-15)

    # Train/validation split
    n_val = max(int(n * val_fraction), d + 1)
    n_train = n - n_val
    indices = rng.permutation(n)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]
    X_train = X_std[train_idx]
    X_val = X_std[val_idx]

    # Generate candidate orderings
    candidates = {}

    # Strategy 1: Conditional variance ordering (our primary method)
    cond_order = _conditional_variance_order(X_train, rng)
    candidates["cond_var"] = cond_order

    # Strategy 2: Multiple random starts refined by greedy BIC
    for r in range(n_restarts):
        perm = rng.permutation(d).tolist()
        refined = _greedy_refine_order(X_train, perm)
        candidates[f"random_{r}"] = refined

    # Strategy 3: Variance ordering (for comparison — this IS sortnregress)
    var_order = np.argsort(np.var(X_train, axis=0)).tolist()
    candidates["var_sort"] = var_order

    # Strategy 4: Reverse variance ordering (anti-varsortability check)
    candidates["var_sort_rev"] = var_order[::-1]

    # Score all candidates on validation set
    best_bic = float("inf")
    best_result = None
    all_scores = {}

    for name, order in candidates.items():
        adj = _cv_regress_in_order(X_train, order, cv_folds=cv_folds)
        bic = _compute_bic(X_val, adj, order)
        all_scores[name] = round(bic, 2)

        if verbose:
            n_edges = int(adj.sum())
            print(f"  {name}: BIC={bic:.1f}, edges={n_edges}")

        if bic < best_bic:
            best_bic = bic
            best_result = {
                "adj": adj,
                "order": order,
                "bic": round(bic, 2),
                "strategy": name,
            }

    best_result["all_scores"] = all_scores
    return best_result


# ===================================================================
# Internal helpers
# ===================================================================

def _regress_in_order(
    X: NDArray,
    order: list[int],
    alpha: float | None = None,
) -> NDArray:
    """Given a topological order, fit parents via Lasso regression.

    Parameters
    ----------
    X : (n, d) data matrix.
    order : List of node indices in topological order.
    alpha : Lasso alpha. If None, use LassoCV.

    Returns
    -------
    (d, d) binary adjacency matrix.
    """
    n, d = X.shape
    adj = np.zeros((d, d), dtype=np.float64)

    for idx, j in enumerate(order):
        if idx == 0:
            continue  # First in order has no parents
        predecessors = order[:idx]
        X_pred = X[:, predecessors]
        y = X[:, j]

        if alpha is None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = LassoCV(cv=5, max_iter=5000)
                model.fit(X_pred, y)
        else:
            model = Lasso(alpha=alpha, max_iter=5000)
            model.fit(X_pred, y)

        for k, p in enumerate(predecessors):
            if abs(model.coef_[k]) > 1e-6:
                adj[p, j] = 1.0

    return adj


def _cv_regress_in_order(
    X: NDArray,
    order: list[int],
    cv_folds: int = 5,
) -> NDArray:
    """Cross-validated Lasso regression along a topological order.

    Uses LassoCV with adaptive noise-floor filtering.

    Parameters
    ----------
    X : (n, d) standardized data.
    order : Candidate topological order.
    cv_folds : Number of CV folds.

    Returns
    -------
    (d, d) binary adjacency matrix.
    """
    n, d = X.shape
    adj = np.zeros((d, d), dtype=np.float64)
    noise_floor = 2.0 / np.sqrt(n)  # Adaptive threshold

    for idx, j in enumerate(order):
        if idx == 0:
            continue
        predecessors = order[:idx]
        X_pred = X[:, predecessors]
        y = X[:, j]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = LassoCV(cv=min(cv_folds, n), max_iter=10000)
            model.fit(X_pred, y)

        for k, p in enumerate(predecessors):
            if abs(model.coef_[k]) > noise_floor:
                adj[p, j] = 1.0

    return adj


def _conditional_variance_order(X: NDArray, rng: np.random.RandomState) -> list[int]:
    """Order nodes by increasing conditional variance.

    Unlike marginal variance ordering (varsortability-exploiting),
    this uses the variance of residuals after regressing out the best
    single predictor. Root nodes have high residual variance (no good
    predictor), so they appear first.

    Parameters
    ----------
    X : (n, d) standardized data.
    rng : Random state.

    Returns
    -------
    list of node indices in estimated topological order.
    """
    n, d = X.shape
    remaining = list(range(d))
    order = []

    for step in range(d):
        if len(order) == 0:
            # First node: smallest max-correlation with any other node
            # (root nodes should be least predictable from others)
            best_node = None
            best_max_corr = float("inf")
            for j in remaining:
                max_corr = 0.0
                for k in remaining:
                    if k == j:
                        continue
                    corr = abs(np.corrcoef(X[:, j], X[:, k])[0, 1])
                    max_corr = max(max_corr, corr)
                if max_corr < best_max_corr:
                    best_max_corr = max_corr
                    best_node = j
            order.append(best_node)
            remaining.remove(best_node)
        else:
            # Pick node with smallest R² when regressed on current order
            X_pred = X[:, order]
            best_node = None
            best_r2 = float("inf")
            for j in remaining:
                y = X[:, j]
                ss_tot = np.sum((y - y.mean()) ** 2)
                if ss_tot < 1e-15:
                    r2 = 0.0
                else:
                    beta = np.linalg.lstsq(X_pred, y - y.mean(), rcond=None)[0]
                    resid = (y - y.mean()) - X_pred @ beta
                    r2 = 1.0 - np.sum(resid ** 2) / ss_tot
                    r2 = max(r2, 0.0)
                if r2 < best_r2:
                    best_r2 = r2
                    best_node = j
            order.append(best_node)
            remaining.remove(best_node)

    return order


def _greedy_refine_order(X: NDArray, order: list[int]) -> list[int]:
    """Refine a topological order by greedy adjacent swaps.

    Iteratively swaps adjacent pairs if it improves BIC.

    Parameters
    ----------
    X : (n, d) data matrix.
    order : Initial ordering.

    Returns
    -------
    Refined ordering.
    """
    d = len(order)
    order = list(order)
    improved = True

    while improved:
        improved = False
        for i in range(d - 1):
            # Try swapping order[i] and order[i+1]
            new_order = order.copy()
            new_order[i], new_order[i + 1] = new_order[i + 1], new_order[i]

            bic_old = _fast_local_bic(X, order, i)
            bic_new = _fast_local_bic(X, new_order, i)

            if bic_new < bic_old - 1e-6:
                order = new_order
                improved = True

    return order


def _fast_local_bic(X: NDArray, order: list[int], pos: int) -> float:
    """Compute BIC for nodes at position pos and pos+1 in the order.

    Only evaluates the two affected nodes for speed.

    Parameters
    ----------
    X : (n, d) data matrix.
    order : Current ordering.
    pos : Position of the first node in the swap.

    Returns
    -------
    Sum of BIC for the two nodes.
    """
    n, d = X.shape
    total_bic = 0.0

    for offset in [0, 1]:
        idx = pos + offset
        j = order[idx]
        if idx == 0:
            # No predecessors
            ss = np.sum((X[:, j] - X[:, j].mean()) ** 2)
            total_bic += n * np.log(ss / n + 1e-15)
        else:
            predecessors = order[:idx]
            X_pred = X[:, predecessors]
            y = X[:, j]
            y_c = y - y.mean()
            beta = np.linalg.lstsq(X_pred, y_c, rcond=None)[0]
            resid = y_c - X_pred @ beta
            ss_res = np.sum(resid ** 2)
            k_params = len(predecessors)
            total_bic += n * np.log(ss_res / n + 1e-15) + k_params * np.log(n)

    return total_bic


def _compute_bic(X: NDArray, adj: NDArray, order: list[int]) -> float:
    """Compute BIC score for a DAG on given data.

    BIC = n * log(RSS/n) + k * log(n) summed over all nodes.

    Parameters
    ----------
    X : (n, d) data matrix (validation set).
    adj : (d, d) binary adjacency matrix.
    order : Topological ordering used.

    Returns
    -------
    Total BIC score (lower is better).
    """
    n, d = X.shape
    total_bic = 0.0

    for j in range(d):
        parents = np.where(adj[:, j] > 0)[0]
        y = X[:, j]
        y_c = y - y.mean()
        ss_tot = np.sum(y_c ** 2)

        if len(parents) == 0:
            ss_res = ss_tot
            k_params = 0
        else:
            X_pa = X[:, parents]
            beta = np.linalg.lstsq(X_pa, y_c, rcond=None)[0]
            resid = y_c - X_pa @ beta
            ss_res = np.sum(resid ** 2)
            k_params = len(parents)

        # BIC for this node
        total_bic += n * np.log(ss_res / n + 1e-15) + k_params * np.log(n)

    return total_bic
