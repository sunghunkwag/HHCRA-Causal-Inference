"""
Standard causal discovery baselines via causal-learn.

Wraps PC, GES, and ICA-LiNGAM with a unified interface that returns
(d, d) binary adjacency matrices, matching CODA's output format.

These are the actual methods CODA must beat to be a credible contribution.
sortnregress (in discovery.py) is the minimum bar; these are the real bar.

References:
    Spirtes et al., "Causation, Prediction, and Search" (2000) — PC.
    Chickering, "Optimal Structure Identification with GES", JMLR 2002.
    Shimizu et al., "A Linear Non-Gaussian Acyclic Model", JMLR 2006 — LiNGAM.
"""

from __future__ import annotations

import warnings
import numpy as np
from numpy.typing import NDArray


def run_pc(X: NDArray, alpha: float = 0.05) -> NDArray:
    """PC algorithm (constraint-based).

    Scale-invariant: not affected by varsortability.
    Recovers Markov equivalence class (CPDAG), not a specific DAG.
    We convert CPDAG to DAG by orienting undirected edges arbitrarily.

    Parameters
    ----------
    X : (n, d) data matrix.
    alpha : Significance level for conditional independence tests.

    Returns
    -------
    (d, d) binary adjacency matrix.
    """
    from causallearn.search.ConstraintBased.PC import pc

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cg = pc(X, alpha=alpha, indep_test="fisherz", stable=True)

    return _graph_to_adj(cg.G, X.shape[1])


def run_ges(X: NDArray) -> NDArray:
    """GES algorithm (score-based).

    Greedy Equivalence Search. Scale-invariant.
    Returns a CPDAG; we orient remaining undirected edges.

    Parameters
    ----------
    X : (n, d) data matrix.

    Returns
    -------
    (d, d) binary adjacency matrix.
    """
    from causallearn.search.ScoreBased.GES import ges

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = ges(X, score_func="local_score_BIC")

    return _graph_to_adj(result["G"], X.shape[1])


def run_lingam(X: NDArray) -> NDArray:
    """ICA-LiNGAM (functional causal model-based).

    Assumes linear non-Gaussian acyclic model. Identifies full DAG
    (not just equivalence class) when non-Gaussianity holds.
    NOT scale-invariant; performance depends on noise distribution.

    Parameters
    ----------
    X : (n, d) data matrix.

    Returns
    -------
    (d, d) binary adjacency matrix.
    """
    from causallearn.search.FCMBased.lingam import ICALiNGAM

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ICALiNGAM()
        model.fit(X)

    # LiNGAM returns weighted adjacency; threshold to binary
    W = model.adjacency_matrix_
    # W[i,j] != 0 means j -> i (causal-learn convention)
    # We need adj[i,j] = 1 means i -> j
    adj = (np.abs(W.T) > 0.05).astype(np.float64)
    np.fill_diagonal(adj, 0)
    return adj


def run_direct_lingam(X: NDArray) -> NDArray:
    """DirectLiNGAM (no ICA; regression-based).

    More stable than ICA-LiNGAM. Also assumes non-Gaussian noise.

    Parameters
    ----------
    X : (n, d) data matrix.

    Returns
    -------
    (d, d) binary adjacency matrix.
    """
    from causallearn.search.FCMBased.lingam import DirectLiNGAM

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = DirectLiNGAM()
        model.fit(X)

    W = model.adjacency_matrix_
    adj = (np.abs(W.T) > 0.05).astype(np.float64)
    np.fill_diagonal(adj, 0)
    return adj


# ===================================================================
# Internal helpers
# ===================================================================

def _graph_to_adj(G, d: int) -> NDArray:
    """Convert causal-learn GeneralGraph to binary adjacency matrix.

    Handles directed (->), undirected (--), and bidirected (<->) edges.
    Undirected edges are oriented by lower-index → higher-index (arbitrary).

    Parameters
    ----------
    G : causal-learn GeneralGraph object.
    d : Number of nodes.

    Returns
    -------
    (d, d) binary adjacency matrix.
    """
    adj = np.zeros((d, d), dtype=np.float64)
    graph_matrix = G.graph  # (d, d) matrix with edge type encoding

    for i in range(d):
        for j in range(d):
            if i == j:
                continue
            # causal-learn encoding:
            # graph[i,j] = -1, graph[j,i] = 1  → i -> j
            # graph[i,j] = -1, graph[j,i] = -1 → i -- j (undirected)
            # graph[i,j] = 1, graph[j,i] = 1   → i <-> j (bidirected)
            if graph_matrix[i, j] == -1 and graph_matrix[j, i] == 1:
                adj[i, j] = 1.0  # i -> j
            elif graph_matrix[i, j] == -1 and graph_matrix[j, i] == -1:
                # Undirected: orient lower -> higher (arbitrary but consistent)
                if i < j:
                    adj[i, j] = 1.0

    return adj
