"""Shared test fixtures for CODA test suite."""

import pytest
import numpy as np
from coda.data import (
    generate_er_dag,
    generate_linear_sem_data,
    SACHS_TRUE_DAG,
    ASIA_TRUE_DAG,
)


@pytest.fixture
def chain3_dag():
    """Simple chain: 0 -> 1 -> 2."""
    dag = np.zeros((3, 3))
    dag[0, 1] = 1
    dag[1, 2] = 1
    return dag


@pytest.fixture
def fork3_dag():
    """Fork: 0 -> 1, 0 -> 2."""
    dag = np.zeros((3, 3))
    dag[0, 1] = 1
    dag[0, 2] = 1
    return dag


@pytest.fixture
def collider3_dag():
    """Collider: 0 -> 2, 1 -> 2."""
    dag = np.zeros((3, 3))
    dag[0, 2] = 1
    dag[1, 2] = 1
    return dag


@pytest.fixture
def asia_dag():
    """Asia network (8 nodes, 8 edges)."""
    return ASIA_TRUE_DAG.copy()


@pytest.fixture
def sachs_dag():
    """Sachs network (11 nodes, 17 edges)."""
    return SACHS_TRUE_DAG.copy()


@pytest.fixture
def er10_dag():
    """Random ER DAG with 10 nodes."""
    return generate_er_dag(10, expected_edges=15, seed=42)


@pytest.fixture
def chain3_data(chain3_dag):
    """Data from chain graph with known weights."""
    X, W = generate_linear_sem_data(chain3_dag, n=2000, seed=42)
    return X, W, chain3_dag


@pytest.fixture
def asia_data(asia_dag):
    """Data from Asia graph."""
    X, W = generate_linear_sem_data(asia_dag, n=2000, seed=42)
    return X, W, asia_dag
