"""Tests for Phase 8: Self-Modifying Architecture."""

import pytest
import torch
import numpy as np

from hhcra.config import HHCRAConfig
from hhcra.architecture import HHCRA
from hhcra.self_modification import (
    ArchitectureEvaluator, BottleneckDetector, StructureModifier,
    ModificationVerifier, SelfModificationEngine, PerformanceMetrics,
)
from hhcra.main import generate_causal_data


@pytest.fixture
def eval_setup():
    config = HHCRAConfig(train_epochs_l1=3, train_epochs_l2=5, train_epochs_l3=3)
    observations, gt = generate_causal_data(B=4, T=8)
    torch.manual_seed(42)
    model = HHCRA(config)
    model.train_all(observations, verbose=False)
    model.eval()
    return model, observations, gt


class TestArchitectureEvaluator:
    def test_evaluate_returns_metrics(self, eval_setup):
        model, obs, gt = eval_setup
        evaluator = ArchitectureEvaluator()
        metrics = evaluator.evaluate(model, obs, gt)
        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.overall >= 0

    def test_metrics_fields(self, eval_setup):
        model, obs, gt = eval_setup
        evaluator = ArchitectureEvaluator()
        metrics = evaluator.evaluate(model, obs, gt)
        assert hasattr(metrics, 'layer1_reconstruction')
        assert hasattr(metrics, 'layer2_shd')
        assert hasattr(metrics, 'layer2_mechanism_mse')
        assert hasattr(metrics, 'layer3_convergence')
        assert hasattr(metrics, 'layer3_identifiability')


class TestBottleneckDetector:
    def test_detect_returns_bottleneck(self):
        detector = BottleneckDetector()
        metrics = PerformanceMetrics(
            layer1_reconstruction=0.5,
            layer2_shd=15,
            layer2_mechanism_mse=0.3,
            layer3_convergence=0.1,
            layer3_identifiability=0.8,
        )
        bottleneck, reason = detector.detect(metrics)
        assert isinstance(bottleneck, str)
        assert isinstance(reason, str)

    def test_identifies_worst_layer(self):
        detector = BottleneckDetector()
        metrics = PerformanceMetrics(
            layer1_reconstruction=0.8,
            layer2_shd=0,
            layer2_mechanism_mse=0.0,
            layer3_convergence=0.01,  # Very bad
            layer3_identifiability=0.9,
        )
        bottleneck, _ = detector.detect(metrics)
        assert 'layer3' in bottleneck


class TestStructureModifier:
    def test_propose_modification(self):
        modifier = StructureModifier()
        config = HHCRAConfig()
        result = modifier.propose_modification('layer2_structure', config)
        assert result is not None
        param, value = result
        assert isinstance(param, str)

    def test_apply_modification(self):
        modifier = StructureModifier()
        config = HHCRAConfig(edge_threshold=0.35)
        new_config = modifier.apply_modification(config, 'edge_threshold', 0.3)
        assert new_config.edge_threshold == 0.3
        assert config.edge_threshold == 0.35  # Original unchanged

    def test_all_bottleneck_types_handled(self):
        modifier = StructureModifier()
        config = HHCRAConfig()
        for bottleneck in ['layer1', 'layer2_structure', 'layer2_mechanism',
                           'layer3_convergence', 'layer3_identifiability']:
            result = modifier.propose_modification(bottleneck, config)
            assert result is not None, f"No proposal for {bottleneck}"


class TestSelfModificationEngine:
    def test_rsi_loop_runs(self, eval_setup):
        model, obs, gt = eval_setup
        engine = SelfModificationEngine()
        # Run a short loop
        history = engine.run_rsi_loop(
            model, obs, gt, num_iterations=3, verbose=False)
        assert len(history) == 3

    def test_modification_log(self, eval_setup):
        model, obs, gt = eval_setup
        engine = SelfModificationEngine()
        engine.run_rsi_loop(model, obs, gt, num_iterations=3, verbose=False)
        summary = engine.get_modification_summary()
        assert 'total_attempted' in summary
        assert 'accepted' in summary
        assert 'rejected' in summary

    def test_modifications_are_reversible(self):
        """Verify rejected modifications don't persist."""
        modifier = StructureModifier()
        config = HHCRAConfig()
        original = config.edge_threshold
        new_config = modifier.apply_modification(config, 'edge_threshold', 0.5)
        assert config.edge_threshold == original  # Original unchanged
