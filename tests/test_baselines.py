"""Tests for coda.baselines — PC, GES, LiNGAM wrappers."""

import numpy as np
import pytest
from coda.baselines import run_pc, run_ges, run_lingam, run_direct_lingam
from coda.data import generate_linear_sem_data, ASIA_TRUE_DAG
from coda.metrics import shd


@pytest.fixture
def asia_raw_data():
    X, _ = generate_linear_sem_data(ASIA_TRUE_DAG, n=2000, seed=42)
    return X


class TestPC:
    def test_returns_correct_shape(self, asia_raw_data):
        adj = run_pc(asia_raw_data)
        assert adj.shape == (8, 8)

    def test_no_self_loops(self, asia_raw_data):
        adj = run_pc(asia_raw_data)
        assert np.trace(adj) == 0

    def test_binary_output(self, asia_raw_data):
        adj = run_pc(asia_raw_data)
        assert set(np.unique(adj)).issubset({0.0, 1.0})

    def test_reasonable_shd(self, asia_raw_data):
        adj = run_pc(asia_raw_data)
        s = shd(adj, ASIA_TRUE_DAG)
        assert s <= 8  # PC should be good on linear Gaussian


class TestGES:
    def test_returns_correct_shape(self, asia_raw_data):
        adj = run_ges(asia_raw_data)
        assert adj.shape == (8, 8)

    def test_no_self_loops(self, asia_raw_data):
        adj = run_ges(asia_raw_data)
        assert np.trace(adj) == 0

    def test_reasonable_shd(self, asia_raw_data):
        adj = run_ges(asia_raw_data)
        s = shd(adj, ASIA_TRUE_DAG)
        assert s <= 4  # GES should be very good on linear Gaussian


class TestLiNGAM:
    def test_returns_correct_shape(self, asia_raw_data):
        adj = run_lingam(asia_raw_data)
        assert adj.shape == (8, 8)

    def test_no_self_loops(self, asia_raw_data):
        adj = run_lingam(asia_raw_data)
        assert np.trace(adj) == 0

    def test_binary_output(self, asia_raw_data):
        adj = run_lingam(asia_raw_data)
        assert set(np.unique(adj)).issubset({0.0, 1.0})


class TestDirectLiNGAM:
    def test_returns_correct_shape(self, asia_raw_data):
        adj = run_direct_lingam(asia_raw_data)
        assert adj.shape == (8, 8)

    def test_no_self_loops(self, asia_raw_data):
        adj = run_direct_lingam(asia_raw_data)
        assert np.trace(adj) == 0
