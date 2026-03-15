"""
Causal inference on fitted linear SCMs.

Implements Pearl's causal hierarchy:
    Rung 1: Observational — P(Y | X=x) (standard conditioning)
    Rung 2: Interventional — P(Y | do(X=x)) (graph surgery)
    Rung 3: Counterfactual — P(Y_x | X=x', Y=y') (abduction-action-prediction)

References:
    Pearl, "Causality" (2009), Chapters 3 and 7.
    Pearl, "The Book of Why" (2018).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from coda.scm import LinearSCM


def interventional_mean(
    scm: LinearSCM,
    target: int,
    intervention_node: int,
    intervention_value: float,
    n_samples: int = 10000,
    seed: int = 42,
) -> float:
    """Estimate E[X_target | do(X_intervention = value)].

    Performs graph surgery (removes all edges into intervention_node)
    and forward-samples from the mutilated model.

    Parameters
    ----------
    scm : Fitted LinearSCM.
    target : Index of the target variable.
    intervention_node : Index of the variable being intervened on.
    intervention_value : Value to set.
    n_samples : Number of Monte Carlo samples.
    seed : Random seed.

    Returns
    -------
    float — estimated interventional mean.
    """
    rng = np.random.RandomState(seed)
    d = scm.d

    # Generate exogenous noise
    exogenous = np.column_stack([
        rng.randn(n_samples) * scm.noise_std[j]
        for j in range(d)
    ])

    # Forward-sample with intervention (graph surgery)
    X = np.zeros((n_samples, d), dtype=np.float64)

    for j in scm.topo_order:
        if j == intervention_node:
            X[:, j] = intervention_value
            continue

        parents = np.where(scm.adj[:, j] > 0)[0]
        X[:, j] = scm.intercepts[j] + exogenous[:, j]
        if len(parents) > 0:
            X[:, j] += X[:, parents] @ scm.weights[parents, j]

    return float(np.mean(X[:, target]))


def interventional_distribution(
    scm: LinearSCM,
    target: int,
    intervention_node: int,
    intervention_value: float,
    n_samples: int = 10000,
    seed: int = 42,
) -> NDArray:
    """Sample from P(X_target | do(X_intervention = value)).

    Parameters
    ----------
    scm : Fitted LinearSCM.
    target : Target variable index.
    intervention_node : Intervened variable index.
    intervention_value : Intervention value.
    n_samples : Number of samples.
    seed : Random seed.

    Returns
    -------
    (n_samples,) array of samples from interventional distribution.
    """
    rng = np.random.RandomState(seed)
    d = scm.d

    exogenous = np.column_stack([
        rng.randn(n_samples) * scm.noise_std[j]
        for j in range(d)
    ])

    X = np.zeros((n_samples, d), dtype=np.float64)

    for j in scm.topo_order:
        if j == intervention_node:
            X[:, j] = intervention_value
            continue

        parents = np.where(scm.adj[:, j] > 0)[0]
        X[:, j] = scm.intercepts[j] + exogenous[:, j]
        if len(parents) > 0:
            X[:, j] += X[:, parents] @ scm.weights[parents, j]

    return X[:, target]


def counterfactual(
    scm: LinearSCM,
    factual_values: NDArray,
    intervention_node: int,
    counterfactual_value: float,
) -> NDArray:
    """Compute counterfactual: "What would Y have been if X had been x'?"

    Three-step procedure (Pearl, 2009):
        1. ABDUCTION: Infer exogenous noise U from factual observation.
        2. ACTION: Apply intervention do(X = x') in mutilated model.
        3. PREDICTION: Forward-propagate with inferred U.

    Parameters
    ----------
    scm : Fitted LinearSCM.
    factual_values : (d,) observed values for all variables.
    intervention_node : Variable to set counterfactually.
    counterfactual_value : Counterfactual value for intervention_node.

    Returns
    -------
    (d,) counterfactual values for all variables.
    """
    factual_values = np.asarray(factual_values, dtype=np.float64)
    d = scm.d

    # Step 1: ABDUCTION — infer exogenous noise
    U = np.zeros(d, dtype=np.float64)
    for j in scm.topo_order:
        parents = np.where(scm.adj[:, j] > 0)[0]
        predicted = scm.intercepts[j]
        if len(parents) > 0:
            predicted += factual_values[parents] @ scm.weights[parents, j]
        U[j] = factual_values[j] - predicted

    # Steps 2-3: ACTION + PREDICTION
    X_cf = np.zeros(d, dtype=np.float64)
    for j in scm.topo_order:
        if j == intervention_node:
            X_cf[j] = counterfactual_value
            continue

        parents = np.where(scm.adj[:, j] > 0)[0]
        X_cf[j] = scm.intercepts[j] + U[j]
        if len(parents) > 0:
            X_cf[j] += X_cf[parents] @ scm.weights[parents, j]

    return X_cf
