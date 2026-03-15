"""Tests for coda.inference — interventional and counterfactual queries."""

import numpy as np
import pytest
from coda.inference import interventional_mean, interventional_distribution, counterfactual
from coda.scm import fit_linear_scm
from coda.data import generate_linear_sem_data


class TestInterventionalMean:
    """Test do-calculus interventional queries."""

    def test_chain_intervention_propagates(self, chain3_dag):
        """do(X0 = c) should affect X1 and X2 in a chain."""
        X, W = generate_linear_sem_data(chain3_dag, n=5000, seed=42)
        scm = fit_linear_scm(X, chain3_dag)

        # Baseline: do(X0 = 0)
        mean_base = interventional_mean(scm, target=2, intervention_node=0,
                                         intervention_value=0.0, seed=42)
        # do(X0 = 5) should change X2
        mean_high = interventional_mean(scm, target=2, intervention_node=0,
                                         intervention_value=5.0, seed=42)
        # Effect should be nonzero
        assert abs(mean_high - mean_base) > 0.1

    def test_chain_no_backward_effect(self, chain3_dag):
        """do(X2 = c) should NOT affect X0 in a chain."""
        X, _ = generate_linear_sem_data(chain3_dag, n=5000, seed=42)
        scm = fit_linear_scm(X, chain3_dag)

        mean_0a = interventional_mean(scm, target=0, intervention_node=2,
                                       intervention_value=0.0, seed=42)
        mean_0b = interventional_mean(scm, target=0, intervention_node=2,
                                       intervention_value=10.0, seed=42)
        # X0 should not be affected by do(X2)
        assert abs(mean_0a - mean_0b) < 0.5

    def test_fork_d_separation(self, fork3_dag):
        """In fork 0->1, 0->2: do(X1=c) should NOT affect X2."""
        X, _ = generate_linear_sem_data(fork3_dag, n=5000, seed=42)
        scm = fit_linear_scm(X, fork3_dag)

        mean_a = interventional_mean(scm, target=2, intervention_node=1,
                                      intervention_value=0.0, seed=42)
        mean_b = interventional_mean(scm, target=2, intervention_node=1,
                                      intervention_value=10.0, seed=42)
        assert abs(mean_a - mean_b) < 0.5

    def test_self_intervention(self, chain3_dag):
        """do(X0 = 5) → E[X0] = 5."""
        X, _ = generate_linear_sem_data(chain3_dag, n=5000, seed=42)
        scm = fit_linear_scm(X, chain3_dag)

        mean = interventional_mean(scm, target=0, intervention_node=0,
                                    intervention_value=5.0, seed=42)
        assert abs(mean - 5.0) < 0.01


class TestInterventionalDistribution:
    """Test interventional distribution sampling."""

    def test_output_shape(self, chain3_dag):
        X, _ = generate_linear_sem_data(chain3_dag, n=2000, seed=42)
        scm = fit_linear_scm(X, chain3_dag)
        samples = interventional_distribution(
            scm, target=2, intervention_node=0,
            intervention_value=1.0, n_samples=500, seed=42,
        )
        assert samples.shape == (500,)

    def test_mean_matches_interventional_mean(self, chain3_dag):
        X, _ = generate_linear_sem_data(chain3_dag, n=5000, seed=42)
        scm = fit_linear_scm(X, chain3_dag)

        mean_func = interventional_mean(
            scm, target=2, intervention_node=0,
            intervention_value=2.0, n_samples=50000, seed=42,
        )
        samples = interventional_distribution(
            scm, target=2, intervention_node=0,
            intervention_value=2.0, n_samples=50000, seed=42,
        )
        assert abs(np.mean(samples) - mean_func) < 0.01


class TestCounterfactual:
    """Test counterfactual reasoning (abduction-action-prediction)."""

    def test_chain_counterfactual(self, chain3_dag):
        """Counterfactual should change downstream but preserve noise."""
        X, _ = generate_linear_sem_data(chain3_dag, n=5000, seed=42)
        scm = fit_linear_scm(X, chain3_dag)

        # Take first observation as factual
        factual = X[0]
        cf = counterfactual(scm, factual, intervention_node=0,
                           counterfactual_value=factual[0] + 5.0)

        # X0 should be set to counterfactual value
        assert abs(cf[0] - (factual[0] + 5.0)) < 1e-10
        # X1, X2 should change (propagation of intervention)
        assert cf[1] != factual[1]

    def test_identity_counterfactual(self, chain3_dag):
        """Setting X0 to its actual value should return ~same values."""
        X, _ = generate_linear_sem_data(chain3_dag, n=5000, seed=42)
        scm = fit_linear_scm(X, chain3_dag)

        factual = X[0]
        cf = counterfactual(scm, factual, intervention_node=0,
                           counterfactual_value=factual[0])
        # Should be approximately identical (up to floating point)
        assert np.allclose(cf, factual, atol=1e-6)

    def test_no_backward_counterfactual(self, chain3_dag):
        """Counterfactual on X2 should not change X0."""
        X, _ = generate_linear_sem_data(chain3_dag, n=5000, seed=42)
        scm = fit_linear_scm(X, chain3_dag)

        factual = X[0]
        cf = counterfactual(scm, factual, intervention_node=2,
                           counterfactual_value=100.0)
        # X0 should be unchanged
        assert abs(cf[0] - factual[0]) < 1e-10
        # X1 should be unchanged (X2 is downstream of X1)
        assert abs(cf[1] - factual[1]) < 1e-10

    def test_output_shape(self, chain3_dag):
        X, _ = generate_linear_sem_data(chain3_dag, n=2000, seed=42)
        scm = fit_linear_scm(X, chain3_dag)
        cf = counterfactual(scm, X[0], intervention_node=0,
                           counterfactual_value=0.0)
        assert cf.shape == (3,)
