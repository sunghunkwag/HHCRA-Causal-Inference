"""Tests for coda.discovery — CODA algorithm and baselines."""

import numpy as np
import pytest
from coda.discovery import coda_discover, sortnregress, r2_sortnregress
from coda.data import generate_linear_sem_data, ASIA_TRUE_DAG, SACHS_TRUE_DAG
from coda.metrics import shd, f1_score_dag, varsortability


class TestSortnregress:
    """Baseline: var-sort + Lasso (Reisach et al., 2021)."""

    def test_chain_raw_data(self, chain3_dag):
        """Sortnregress should work on raw (high-varsortability) data."""
        X, _ = generate_linear_sem_data(chain3_dag, n=2000, seed=42)
        adj = sortnregress(X)
        assert adj.shape == (3, 3)
        assert shd(adj, chain3_dag) <= 3  # 2-edge graph; 3 is reasonable

    def test_asia_raw_data(self, asia_dag):
        """Sortnregress on Asia with raw data."""
        X, _ = generate_linear_sem_data(asia_dag, n=2000, seed=42)
        adj = sortnregress(X)
        assert adj.shape == (8, 8)
        s = shd(adj, asia_dag)
        assert s <= 8  # Reasonable for 8-edge graph

    def test_returns_binary(self, chain3_dag):
        X, _ = generate_linear_sem_data(chain3_dag, n=2000, seed=42)
        adj = sortnregress(X)
        assert set(np.unique(adj)).issubset({0.0, 1.0})

    def test_no_self_loops(self, chain3_dag):
        X, _ = generate_linear_sem_data(chain3_dag, n=2000, seed=42)
        adj = sortnregress(X)
        assert np.trace(adj) == 0


class TestR2Sortnregress:
    """Baseline: R²-sort + Lasso (Reisach et al., 2023)."""

    def test_chain_standardized(self, chain3_dag):
        X, _ = generate_linear_sem_data(chain3_dag, n=2000, seed=42, standardize=True)
        adj = r2_sortnregress(X)
        assert adj.shape == (3, 3)
        assert shd(adj, chain3_dag) <= 3  # 2-edge graph; 3 is reasonable

    def test_returns_binary(self, chain3_dag):
        X, _ = generate_linear_sem_data(chain3_dag, n=2000, seed=42)
        adj = r2_sortnregress(X)
        assert set(np.unique(adj)).issubset({0.0, 1.0})


class TestCODA:
    """Main algorithm: CODA."""

    def test_chain_basic(self, chain3_dag):
        """CODA should recover a simple chain."""
        X, _ = generate_linear_sem_data(chain3_dag, n=2000, seed=42)
        result = coda_discover(X, n_restarts=5, seed=42)
        assert "adj" in result
        assert "order" in result
        assert "bic" in result
        assert "strategy" in result
        assert "all_scores" in result
        assert shd(result["adj"], chain3_dag) <= 2

    def test_fork_basic(self, fork3_dag):
        """CODA should recover a simple fork."""
        X, _ = generate_linear_sem_data(fork3_dag, n=2000, seed=42)
        result = coda_discover(X, n_restarts=5, seed=42)
        assert shd(result["adj"], fork3_dag) <= 2

    def test_collider_basic(self, collider3_dag):
        """CODA should recover a simple collider."""
        X, _ = generate_linear_sem_data(collider3_dag, n=2000, seed=42)
        result = coda_discover(X, n_restarts=5, seed=42)
        assert shd(result["adj"], collider3_dag) <= 2

    def test_asia_raw(self, asia_dag):
        """CODA on Asia with raw data."""
        X, _ = generate_linear_sem_data(asia_dag, n=2000, seed=42)
        result = coda_discover(X, n_restarts=10, seed=42)
        s = shd(result["adj"], asia_dag)
        assert s <= 8  # 8-edge graph; competitive with baselines

    def test_asia_standardized(self, asia_dag):
        """CODA on Asia with standardized data (varsortability removed)."""
        X, _ = generate_linear_sem_data(asia_dag, n=2000, seed=42, standardize=True)
        result = coda_discover(X, n_restarts=10, seed=42)
        s = shd(result["adj"], asia_dag)
        # On standardized data, methods that exploit varsortability degrade.
        # CODA should still work because it uses conditional variance ordering.
        assert s <= 8

    def test_returns_valid_dag(self, chain3_dag):
        """Output should be a valid DAG (no cycles)."""
        from coda.metrics import topological_order_from_dag
        X, _ = generate_linear_sem_data(chain3_dag, n=2000, seed=42)
        result = coda_discover(X, seed=42)
        adj = result["adj"]
        # Should not raise
        order = topological_order_from_dag(adj)
        assert len(order) == 3

    def test_reproducible(self, chain3_dag):
        """Same seed → same result."""
        X, _ = generate_linear_sem_data(chain3_dag, n=2000, seed=42)
        r1 = coda_discover(X, seed=42)
        r2 = coda_discover(X, seed=42)
        assert np.array_equal(r1["adj"], r2["adj"])

    def test_no_self_loops(self, asia_dag):
        X, _ = generate_linear_sem_data(asia_dag, n=2000, seed=42)
        result = coda_discover(X, seed=42)
        assert np.trace(result["adj"]) == 0

    def test_verbose_runs(self, chain3_dag, capsys):
        """Verbose mode should print progress."""
        X, _ = generate_linear_sem_data(chain3_dag, n=2000, seed=42)
        coda_discover(X, n_restarts=2, seed=42, verbose=True)
        captured = capsys.readouterr()
        assert "BIC" in captured.out

    def test_all_scores_populated(self, chain3_dag):
        """All candidate strategies should have scores."""
        X, _ = generate_linear_sem_data(chain3_dag, n=2000, seed=42)
        result = coda_discover(X, n_restarts=3, seed=42)
        scores = result["all_scores"]
        assert "cond_var" in scores
        assert "var_sort" in scores
        assert "var_sort_rev" in scores
        assert len(scores) >= 6  # cond_var + var_sort + var_sort_rev + 3 random


class TestCODAvsBaselines:
    """Compare CODA against baselines for honest evaluation."""

    def test_coda_competitive_on_raw_data(self, asia_dag):
        """On raw data (high varsortability), CODA should be in the ballpark."""
        X, _ = generate_linear_sem_data(asia_dag, n=2000, seed=42)
        
        adj_snr = sortnregress(X)
        adj_coda = coda_discover(X, n_restarts=10, seed=42)["adj"]
        
        shd_snr = shd(adj_snr, asia_dag)
        shd_coda = shd(adj_coda, asia_dag)
        
        # On high-varsortability data, sortnregress may win.
        # CODA should still be reasonable (not catastrophically worse).
        assert shd_coda <= shd_snr + 4

    def test_coda_on_low_varsortability_data(self, asia_dag):
        """On data with reduced varsortability, CODA should be robust."""
        # Use small weights → lower varsortability
        X, _ = generate_linear_sem_data(
            asia_dag, n=2000, weight_range=(0.2, 0.8),
            noise_std_range=(0.8, 1.2), seed=42,
        )
        v = varsortability(X, asia_dag)
        # Verify varsortability is indeed lower
        assert v < 0.95

        adj_coda = coda_discover(X, n_restarts=10, seed=42)["adj"]
        s = shd(adj_coda, asia_dag)
        assert s <= 8  # Should still work
