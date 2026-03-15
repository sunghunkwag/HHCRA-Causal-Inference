"""Tests for coda.data."""

import numpy as np
import pytest
from coda.data import (
    generate_er_dag,
    generate_sf_dag,
    generate_linear_sem_data,
    load_sachs,
    SACHS_TRUE_DAG,
    SACHS_NODE_NAMES,
    ASIA_TRUE_DAG,
    ASIA_NODE_NAMES,
)
from coda.metrics import topological_order_from_dag


class TestGroundTruthDAGs:
    """Verify ground-truth DAG definitions."""

    def test_sachs_shape(self):
        assert SACHS_TRUE_DAG.shape == (11, 11)

    def test_sachs_edges(self):
        assert int(SACHS_TRUE_DAG.sum()) == 17

    def test_sachs_is_dag(self):
        """Sachs graph must be acyclic."""
        order = topological_order_from_dag(SACHS_TRUE_DAG)
        assert len(order) == 11

    def test_sachs_node_count(self):
        assert len(SACHS_NODE_NAMES) == 11

    def test_sachs_no_self_loops(self):
        assert np.trace(SACHS_TRUE_DAG) == 0

    def test_asia_shape(self):
        assert ASIA_TRUE_DAG.shape == (8, 8)

    def test_asia_edges(self):
        assert int(ASIA_TRUE_DAG.sum()) == 8

    def test_asia_is_dag(self):
        order = topological_order_from_dag(ASIA_TRUE_DAG)
        assert len(order) == 8

    def test_asia_node_count(self):
        assert len(ASIA_NODE_NAMES) == 8


class TestDAGGenerators:
    """Test random DAG generation."""

    def test_er_dag_is_acyclic(self):
        dag = generate_er_dag(20, expected_edges=30, seed=42)
        order = topological_order_from_dag(dag)
        assert len(order) == 20

    def test_er_dag_no_self_loops(self):
        dag = generate_er_dag(10, expected_edges=15, seed=42)
        assert np.trace(dag) == 0

    def test_er_dag_shape(self):
        dag = generate_er_dag(15, expected_edges=20, seed=42)
        assert dag.shape == (15, 15)

    def test_er_dag_reproducible(self):
        dag1 = generate_er_dag(10, expected_edges=15, seed=42)
        dag2 = generate_er_dag(10, expected_edges=15, seed=42)
        assert np.array_equal(dag1, dag2)

    def test_er_dag_different_seeds(self):
        dag1 = generate_er_dag(10, expected_edges=15, seed=42)
        dag2 = generate_er_dag(10, expected_edges=15, seed=99)
        assert not np.array_equal(dag1, dag2)

    def test_sf_dag_is_acyclic(self):
        dag = generate_sf_dag(20, k=2, seed=42)
        order = topological_order_from_dag(dag)
        assert len(order) == 20

    def test_sf_dag_no_self_loops(self):
        dag = generate_sf_dag(10, k=2, seed=42)
        assert np.trace(dag) == 0


class TestSEMDataGeneration:
    """Test linear SEM data generation."""

    def test_output_shape(self, chain3_dag):
        X, W = generate_linear_sem_data(chain3_dag, n=500, seed=42)
        assert X.shape == (500, 3)
        assert W.shape == (3, 3)

    def test_weights_match_dag(self, chain3_dag):
        _, W = generate_linear_sem_data(chain3_dag, n=500, seed=42)
        # Nonzero weights only where dag has edges
        for i in range(3):
            for j in range(3):
                if chain3_dag[i, j] == 0:
                    assert W[i, j] == 0.0

    def test_reproducible(self, asia_dag):
        X1, W1 = generate_linear_sem_data(asia_dag, n=500, seed=42)
        X2, W2 = generate_linear_sem_data(asia_dag, n=500, seed=42)
        assert np.array_equal(X1, X2)
        assert np.array_equal(W1, W2)

    def test_standardized_data(self, chain3_dag):
        X, _ = generate_linear_sem_data(
            chain3_dag, n=5000, seed=42, standardize=True
        )
        # Means should be ~0, stds should be ~1
        assert np.allclose(X.mean(axis=0), 0, atol=0.05)
        assert np.allclose(X.std(axis=0), 1, atol=0.05)

    def test_default_high_varsortability(self, asia_dag):
        """Default parameters should produce high varsortability (known issue)."""
        from coda.metrics import varsortability
        X, _ = generate_linear_sem_data(asia_dag, n=2000, seed=42)
        v = varsortability(X, asia_dag)
        # Default weight_range=(0.5, 2.0) → high varsortability
        assert v >= 0.7


class TestSachsLoader:
    """Test Sachs synthetic data loader."""

    def test_shape(self):
        X = load_sachs()
        assert X.shape == (853, 11)

    def test_standardized_shape(self):
        X = load_sachs(standardize=True)
        assert X.shape == (853, 11)

    def test_standardized_stats(self):
        X = load_sachs(standardize=True)
        assert np.allclose(X.mean(axis=0), 0, atol=0.05)
        assert np.allclose(X.std(axis=0), 1, atol=0.05)

    def test_reproducible(self):
        X1 = load_sachs()
        X2 = load_sachs()
        assert np.array_equal(X1, X2)
