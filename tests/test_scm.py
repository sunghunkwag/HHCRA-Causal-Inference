"""Tests for coda.scm — SCM fitting."""

import numpy as np
import pytest
from coda.scm import fit_linear_scm, LinearSCM
from coda.data import generate_linear_sem_data


class TestFitLinearSCM:
    """Test OLS-based SCM fitting."""

    def test_chain_weight_recovery(self, chain3_dag):
        """Fitted weights should approximate true weights."""
        X, W_true = generate_linear_sem_data(chain3_dag, n=5000, seed=42)
        scm = fit_linear_scm(X, chain3_dag)

        # Check that fitted weights are close to true
        for i in range(3):
            for j in range(3):
                if chain3_dag[i, j] > 0:
                    assert abs(scm.weights[i, j] - W_true[i, j]) < 0.1, \
                        f"Weight [{i},{j}]: true={W_true[i,j]:.3f}, fitted={scm.weights[i,j]:.3f}"

    def test_zero_weights_where_no_edge(self, chain3_dag):
        """Weights should be zero where there is no edge."""
        X, _ = generate_linear_sem_data(chain3_dag, n=2000, seed=42)
        scm = fit_linear_scm(X, chain3_dag)
        for i in range(3):
            for j in range(3):
                if chain3_dag[i, j] == 0:
                    assert scm.weights[i, j] == 0.0

    def test_noise_std_positive(self, chain3_dag):
        X, _ = generate_linear_sem_data(chain3_dag, n=2000, seed=42)
        scm = fit_linear_scm(X, chain3_dag)
        assert all(s > 0 for s in scm.noise_std)

    def test_topo_order_valid(self, asia_dag):
        X, _ = generate_linear_sem_data(asia_dag, n=2000, seed=42)
        scm = fit_linear_scm(X, asia_dag)
        assert len(scm.topo_order) == 8

    def test_predict_shape(self, chain3_dag):
        X, _ = generate_linear_sem_data(chain3_dag, n=2000, seed=42)
        scm = fit_linear_scm(X, chain3_dag)
        X_gen = scm.predict(seed=0)
        assert X_gen.shape == (1000, 3)

    def test_predict_with_exogenous(self, chain3_dag):
        X, _ = generate_linear_sem_data(chain3_dag, n=2000, seed=42)
        scm = fit_linear_scm(X, chain3_dag)
        exo = np.random.randn(50, 3)
        X_gen = scm.predict(exogenous=exo)
        assert X_gen.shape == (50, 3)

    def test_generated_data_similar_distribution(self, chain3_dag):
        """Generated data should have similar statistics to training data."""
        X, _ = generate_linear_sem_data(chain3_dag, n=5000, seed=42)
        scm = fit_linear_scm(X, chain3_dag)
        X_gen = scm.predict(seed=0)

        # Means should be in the same ballpark
        for j in range(3):
            assert abs(X.mean(axis=0)[j] - X_gen.mean(axis=0)[j]) < 1.0
