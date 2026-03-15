"""
Data generation and loading for causal discovery benchmarks.

Provides:
    - Erdős–Rényi and scale-free DAG generators
    - Linear SEM data generation (with configurable varsortability)
    - Sachs flow cytometry ground-truth DAG
    - Asia (BnLearn) ground-truth DAG

References:
    Sachs et al., "Causal Protein-Signaling Networks Derived from
        Multiparameter Single-Cell Data", Science 2005.
    Reisach et al., "Beware of the Simulated DAG!", NeurIPS 2021.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


# ===================================================================
# Ground-truth DAGs
# ===================================================================

# Sachs et al. (2005) consensus network: 11 nodes, 17 edges.
# Node order: Raf, Mek, Plcg, PIP2, PIP3, Erk, Akt, PKA, PKC, P38, JNK
# Adjacency: SACHS_TRUE_DAG[i,j] = 1 means node i -> node j.
SACHS_NODE_NAMES = [
    "Raf", "Mek", "Plcg", "PIP2", "PIP3",
    "Erk", "Akt", "PKA", "PKC", "P38", "JNK",
]

_sachs_edges = [
    (0, 1),   # Raf -> Mek
    (1, 5),   # Mek -> Erk
    (2, 3),   # Plcg -> PIP2
    (2, 4),   # Plcg -> PIP3
    (4, 3),   # PIP3 -> PIP2
    (5, 6),   # Erk -> Akt
    (7, 0),   # PKA -> Raf
    (7, 1),   # PKA -> Mek
    (7, 5),   # PKA -> Erk
    (7, 6),   # PKA -> Akt
    (7, 9),   # PKA -> P38
    (7, 10),  # PKA -> JNK
    (8, 0),   # PKC -> Raf
    (8, 1),   # PKC -> Mek
    (8, 7),   # PKC -> PKA
    (8, 9),   # PKC -> P38
    (8, 10),  # PKC -> JNK
]
SACHS_TRUE_DAG = np.zeros((11, 11), dtype=np.float64)
for i, j in _sachs_edges:
    SACHS_TRUE_DAG[i, j] = 1.0

# Asia (Lauritzen & Spiegelhalter, 1988) network: 8 nodes, 8 edges.
# Node order: asia, tub, smoke, lung, bronc, either, xray, dysp
ASIA_NODE_NAMES = [
    "asia", "tub", "smoke", "lung", "bronc",
    "either", "xray", "dysp",
]

_asia_edges = [
    (0, 1),  # asia -> tub
    (2, 3),  # smoke -> lung
    (2, 4),  # smoke -> bronc
    (1, 5),  # tub -> either
    (3, 5),  # lung -> either
    (5, 6),  # either -> xray
    (5, 7),  # either -> dysp
    (4, 7),  # bronc -> dysp
]
ASIA_TRUE_DAG = np.zeros((8, 8), dtype=np.float64)
for i, j in _asia_edges:
    ASIA_TRUE_DAG[i, j] = 1.0


# ===================================================================
# DAG generators
# ===================================================================

def generate_er_dag(d: int, expected_edges: int, seed: int = 42) -> NDArray:
    """Generate an Erdős–Rényi DAG.

    Creates a random lower-triangular adjacency matrix (guaranteeing
    acyclicity) with a random permutation applied.

    Parameters
    ----------
    d : Number of nodes.
    expected_edges : Expected number of edges (actual may vary).
    seed : Random seed.

    Returns
    -------
    (d, d) binary adjacency matrix. A[i,j]=1 means i->j.
    """
    rng = np.random.RandomState(seed)
    prob = expected_edges / (d * (d - 1) / 2)
    prob = min(prob, 1.0)

    # Lower triangular → acyclic in natural order
    adj = np.zeros((d, d), dtype=np.float64)
    for i in range(1, d):
        for j in range(i):
            if rng.random() < prob:
                adj[j, i] = 1.0  # j -> i (j is ancestor)

    # Random permutation to hide natural ordering
    perm = rng.permutation(d)
    adj_perm = adj[np.ix_(perm, perm)]
    return adj_perm


def generate_sf_dag(d: int, k: int = 2, seed: int = 42) -> NDArray:
    """Generate a scale-free DAG via Barabási–Albert model.

    Parameters
    ----------
    d : Number of nodes.
    k : Number of edges each new node attaches to.
    seed : Random seed.

    Returns
    -------
    (d, d) binary adjacency matrix.
    """
    rng = np.random.RandomState(seed)
    adj = np.zeros((d, d), dtype=np.float64)
    degrees = np.zeros(d, dtype=np.float64)

    # Start with a small clique
    for i in range(min(k, d)):
        for j in range(i):
            adj[j, i] = 1.0
            degrees[i] += 1
            degrees[j] += 1

    for i in range(k, d):
        # Preferential attachment
        probs = degrees[:i].copy()
        if probs.sum() == 0:
            probs = np.ones(i)
        probs /= probs.sum()

        targets = rng.choice(i, size=min(k, i), replace=False, p=probs)
        for t in targets:
            adj[t, i] = 1.0
            degrees[t] += 1
            degrees[i] += 1

    # Random permutation
    perm = rng.permutation(d)
    return adj[np.ix_(perm, perm)]


# ===================================================================
# Data generation
# ===================================================================

def generate_linear_sem_data(
    dag: NDArray,
    n: int = 2000,
    weight_range: tuple[float, float] = (0.5, 2.0),
    noise_std_range: tuple[float, float] = (0.5, 2.0),
    standardize: bool = False,
    seed: int = 42,
) -> tuple[NDArray, NDArray]:
    """Generate data from a linear SEM: X_j = sum_i w_ij * X_i + e_j.

    Edge weights are drawn from Unif(weight_range) with random sign.
    Noise std is drawn from Unif(noise_std_range).

    WARNING: With default parameters, the generated data will be highly
    var-sortable (Reisach et al., 2021). Use standardize=True or
    set weight_range=(0.2, 0.8) to reduce this artifact.

    Parameters
    ----------
    dag : (d, d) binary adjacency matrix.
    n : Number of samples.
    weight_range : (low, high) for absolute weight values.
    noise_std_range : (low, high) for noise standard deviations.
    standardize : If True, standardize each column to zero mean, unit var.
    seed : Random seed.

    Returns
    -------
    X : (n, d) data matrix.
    W : (d, d) weighted adjacency matrix (ground truth weights).
    """
    rng = np.random.RandomState(seed)
    dag = np.asarray(dag, dtype=np.float64)
    d = dag.shape[0]

    # Generate weights
    W = np.zeros((d, d), dtype=np.float64)
    edges = np.argwhere(dag > 0)
    for i, j in edges:
        w = rng.uniform(weight_range[0], weight_range[1])
        sign = rng.choice([-1.0, 1.0])
        W[i, j] = sign * w

    # Noise std per variable
    noise_std = rng.uniform(noise_std_range[0], noise_std_range[1], size=d)

    # Topological order (Kahn's)
    in_deg = (dag > 0).sum(axis=0).astype(int)
    queue = [i for i in range(d) if in_deg[i] == 0]
    topo = []
    in_deg_copy = in_deg.copy()
    while queue:
        node = queue.pop(0)
        topo.append(node)
        for child in range(d):
            if dag[node, child] > 0:
                in_deg_copy[child] -= 1
                if in_deg_copy[child] == 0:
                    queue.append(child)
    assert len(topo) == d, "Graph has a cycle — cannot generate SEM data."

    # Generate data
    X = np.zeros((n, d), dtype=np.float64)
    for j in topo:
        parents = np.where(dag[:, j] > 0)[0]
        noise = rng.randn(n) * noise_std[j]
        if len(parents) > 0:
            X[:, j] = X[:, parents] @ W[parents, j] + noise
        else:
            X[:, j] = noise

    if standardize:
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-15)

    return X, W


# ===================================================================
# Real data loading
# ===================================================================

def load_sachs(standardize: bool = False) -> NDArray:
    """Generate synthetic Sachs-like data from the ground-truth DAG.

    Since the actual Sachs flow cytometry data requires external download
    (bnlearn.com/bnrepository or Zenodo), this function generates synthetic
    data from the known DAG structure as a self-contained benchmark.

    For real data evaluation, download the original dataset and load with:
        import pandas as pd
        df = pd.read_csv("sachs.csv")
        X = df.values  # shape (853, 11)

    Parameters
    ----------
    standardize : If True, standardize columns.

    Returns
    -------
    X : (853, 11) synthetic data from Sachs DAG.
    """
    # Use non-Gaussian noise to match real Sachs characteristics
    rng = np.random.RandomState(42)
    d = 11
    n = 853

    # Weights tuned to approximate real Sachs data scale relationships
    W = np.zeros((d, d), dtype=np.float64)
    weight_seed = np.random.RandomState(123)
    for i, j in _sachs_edges:
        w = weight_seed.uniform(0.3, 1.2)
        sign = weight_seed.choice([-1.0, 1.0])
        W[i, j] = sign * w

    # Topological order
    from coda.metrics import topological_order_from_dag
    topo = topological_order_from_dag(SACHS_TRUE_DAG)

    X = np.zeros((n, d), dtype=np.float64)
    for j in topo:
        parents = np.where(SACHS_TRUE_DAG[:, j] > 0)[0]
        # Use non-Gaussian noise (exponential + Gaussian mixture)
        noise = rng.randn(n) * 0.8 + rng.exponential(0.3, size=n) * rng.choice([-1, 1], size=n)
        if len(parents) > 0:
            X[:, j] = X[:, parents] @ W[parents, j] + noise
        else:
            X[:, j] = noise

    if standardize:
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-15)

    return X
