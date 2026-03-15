"""Tests for coda.metrics."""

import numpy as np
import pytest
from coda.metrics import (
    shd,
    f1_score_dag,
    varsortability,
    r2_sortability,
    topological_order_from_dag,
)
from coda.data import generate_linear_sem_data


class TestSHD:
    """Structural Hamming Distance tests."""

    def test_identical_graphs(self, chain3_dag):
        assert shd(chain3_dag, chain3_dag) == 0

    def test_empty_vs_chain(self, chain3_dag):
        empty = np.zeros_like(chain3_dag)
        assert shd(empty, chain3_dag) == 2  # 2 missing edges

    def test_extra_edge(self, chain3_dag):
        pred = chain3_dag.copy()
        pred[0, 2] = 1  # Add spurious edge
        assert shd(pred, chain3_dag) == 1

    def test_reversed_edge(self):
        true = np.zeros((3, 3))
        true[0, 1] = 1
        pred = np.zeros((3, 3))
        pred[1, 0] = 1  # Reversed
        assert shd(pred, true) == 1  # One reversal

    def test_completely_wrong(self, chain3_dag):
        wrong = np.zeros_like(chain3_dag)
        wrong[2, 0] = 1
        wrong[2, 1] = 1
        result = shd(wrong, chain3_dag)
        assert result > 0

    def test_empty_graph_shd(self):
        """Empty graph vs empty graph = 0."""
        empty = np.zeros((5, 5))
        assert shd(empty, empty) == 0

    def test_sachs_empty_shd(self, sachs_dag):
        """Empty prediction on Sachs = 17 (all edges missing)."""
        empty = np.zeros_like(sachs_dag)
        assert shd(empty, sachs_dag) == 17


class TestF1:
    """F1 score tests."""

    def test_perfect_recovery(self, chain3_dag):
        result = f1_score_dag(chain3_dag, chain3_dag)
        assert result["f1"] == 1.0
        assert result["tp"] == 2
        assert result["fp"] == 0
        assert result["fn"] == 0

    def test_empty_prediction(self, chain3_dag):
        empty = np.zeros_like(chain3_dag)
        result = f1_score_dag(empty, chain3_dag)
        assert result["recall"] == 0.0
        assert result["fn"] == 2

    def test_extra_edges(self, chain3_dag):
        pred = np.ones_like(chain3_dag)
        np.fill_diagonal(pred, 0)
        result = f1_score_dag(pred, chain3_dag)
        assert result["tp"] == 2
        assert result["fp"] == 4  # 6 total - 2 true = 4 false


class TestVarsortability:
    """Varsortability diagnostic tests."""

    def test_high_varsortability_raw_data(self, chain3_dag):
        """Default SEM parameters produce high varsortability."""
        X, _ = generate_linear_sem_data(chain3_dag, n=2000, seed=42)
        v = varsortability(X, chain3_dag)
        assert v >= 0.5  # Should be high for raw data

    def test_reduced_varsortability_standardized(self, chain3_dag):
        """Standardized data should have lower varsortability."""
        X, _ = generate_linear_sem_data(chain3_dag, n=2000, seed=42, standardize=True)
        v = varsortability(X, chain3_dag)
        # Standardized data has equal marginal variances → v ≈ 0.5
        assert v <= 0.75

    def test_varsortability_range(self, asia_dag):
        """Varsortability should be in [0, 1]."""
        X, _ = generate_linear_sem_data(asia_dag, n=2000, seed=42)
        v = varsortability(X, asia_dag)
        assert 0.0 <= v <= 1.0

    def test_empty_dag(self):
        """Empty DAG returns 0.5 (neutral)."""
        dag = np.zeros((3, 3))
        X = np.random.randn(100, 3)
        assert varsortability(X, dag) == 0.5


class TestR2Sortability:
    """R²-sortability diagnostic tests."""

    def test_r2_range(self, asia_dag):
        X, _ = generate_linear_sem_data(asia_dag, n=2000, seed=42)
        r2 = r2_sortability(X, asia_dag)
        assert 0.0 <= r2 <= 1.0

    def test_r2_empty_dag(self):
        dag = np.zeros((3, 3))
        X = np.random.randn(100, 3)
        assert r2_sortability(X, dag) == 0.5


class TestTopologicalOrder:
    """Topological sort tests."""

    def test_chain(self, chain3_dag):
        order = topological_order_from_dag(chain3_dag)
        assert len(order) == 3
        assert order.index(0) < order.index(1) < order.index(2)

    def test_fork(self, fork3_dag):
        order = topological_order_from_dag(fork3_dag)
        assert order[0] == 0  # Root must be first

    def test_cycle_raises(self):
        cycle = np.zeros((3, 3))
        cycle[0, 1] = 1
        cycle[1, 2] = 1
        cycle[2, 0] = 1
        with pytest.raises(ValueError, match="cycle"):
            topological_order_from_dag(cycle)

    def test_asia(self, asia_dag):
        order = topological_order_from_dag(asia_dag)
        assert len(order) == 8
        # asia (0) must come before tub (1)
        assert order.index(0) < order.index(1)
