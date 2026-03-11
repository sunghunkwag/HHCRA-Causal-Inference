"""Tests for benchmark suite and evaluation metrics."""

import pytest
import torch
import numpy as np

from hhcra.config import HHCRAConfig
from hhcra.architecture import HHCRA
from hhcra.main import generate_causal_data, run_verification_tests


class TestVerificationSuite:
    """Run the original 12 verification tests from hhcra_v2.py."""

    def test_all_12_pass(self, trained_model, observations, ground_truth):
        """All 12 original verification tests must pass."""
        with torch.no_grad():
            results = run_verification_tests(trained_model, observations, ground_truth)
        passed = sum(1 for r in results if r.passed)
        total = len(results)
        assert total == 12, f"Expected 12 tests, got {total}"
        # Print details for any failures
        for r in results:
            if not r.passed:
                print(f"FAILED: {r}")
        assert passed == total, f"Only {passed}/{total} tests passed"


class TestCausalDiscoveryMetrics:
    """Test causal discovery metrics: SHD, TPR, FDR."""

    def test_shd_exists(self, trained_model, ground_truth):
        metrics = trained_model.layer2.gnn.compute_metrics(
            ground_truth['true_adjacency'])
        assert 'shd' in metrics
        assert isinstance(metrics['shd'], int)

    def test_tpr_range(self, trained_model, ground_truth):
        metrics = trained_model.layer2.gnn.compute_metrics(
            ground_truth['true_adjacency'])
        assert 0 <= metrics['tpr'] <= 1

    def test_fdr_range(self, trained_model, ground_truth):
        metrics = trained_model.layer2.gnn.compute_metrics(
            ground_truth['true_adjacency'])
        assert 0 <= metrics['fdr'] <= 1


class TestDataGeneration:
    def test_generates_correct_shapes(self):
        obs, gt = generate_causal_data(B=4, T=8, obs_dim=48)
        assert obs.shape == (4, 8, 48)
        assert gt['true_adjacency'].shape == (5, 5)

    def test_ground_truth_is_dag(self):
        _, gt = generate_causal_data()
        adj = gt['true_adjacency']
        N = adj.shape[0]
        in_degree = adj.sum(axis=1).astype(int)
        queue = [i for i in range(N) if in_degree[i] == 0]
        visited = 0
        while queue:
            node = queue.pop(0)
            visited += 1
            for child in range(N):
                if adj[child, node] > 0:
                    in_degree[child] -= 1
                    if in_degree[child] == 0:
                        queue.append(child)
        assert visited == N

    def test_interventional_effect(self):
        _, gt = generate_causal_data()
        expected = 0.3 * (0.7 * 2.0) + 0.6 * (0.5 * 2.0)
        assert abs(gt['do_x0_2_effect_on_x3'] - expected) < 1e-6

    def test_returns_torch_tensor(self):
        obs, _ = generate_causal_data()
        assert isinstance(obs, torch.Tensor)
        assert obs.dtype == torch.float32
