"""Tests for the full HHCRA pipeline and architecture integration."""

import pytest
import torch
import torch.nn as nn
import numpy as np

from hhcra.config import HHCRAConfig
from hhcra.causal_graph import CausalQueryType
from hhcra.architecture import HHCRA
from hhcra.interfaces import LayerInterface


class TestHHCRA:
    def test_is_nn_module(self, config):
        model = HHCRA(config)
        assert isinstance(model, nn.Module)

    def test_forward_pass(self, config, observations):
        model = HHCRA(config)
        fwd = model.forward(observations)
        assert 'latent' in fwd
        assert 'layer2' in fwd
        assert 'graph' in fwd

    def test_forward_latent_shape(self, config, observations):
        model = HHCRA(config)
        fwd = model.forward(observations)
        B, T = observations.shape[:2]
        assert fwd['latent'].shape == (B, T, config.num_vars, config.latent_dim)

    def test_observational_query(self, trained_model, observations):
        with torch.no_grad():
            r = trained_model.query(observations, CausalQueryType.OBSERVATIONAL,
                                     X=0, Y=3, verbose=False)
        assert r['answer'] is not None
        assert not torch.any(torch.isnan(r['answer'])).item()

    def test_interventional_query(self, trained_model, observations):
        xv = torch.full((trained_model.config.latent_dim,), 2.0)
        with torch.no_grad():
            r = trained_model.query(observations, CausalQueryType.INTERVENTIONAL,
                                     X=0, Y=3, x_value=xv, verbose=False)
        assert r['identifiability'] is not None

    def test_counterfactual_query(self, trained_model, observations):
        D = trained_model.config.latent_dim
        B = observations.shape[0]
        fx = torch.full((B, D), 1.0)
        fy = torch.full((B, D), 0.5)
        cfx = torch.full((D,), -1.0)
        with torch.no_grad():
            r = trained_model.query(observations, CausalQueryType.COUNTERFACTUAL,
                                     X=0, Y=3, factual_x=fx, factual_y=fy,
                                     counterfactual_x=cfx, verbose=False)
        assert r['identifiability'] is not None

    def test_feedback_mechanism(self, trained_model, observations):
        """Feedback should be generated for non-identifiable queries."""
        with torch.no_grad():
            r = trained_model.query(observations, CausalQueryType.INTERVENTIONAL,
                                     X=trained_model.config.num_vars - 1, Y=0,
                                     verbose=False)
        # Query should complete regardless of identifiability
        assert 'identifiability' in r

    def test_intervention_changes_output(self, trained_model, observations):
        """Intervention should produce different results than observation."""
        with torch.no_grad():
            fwd = trained_model.forward(observations)
            adj = torch.tensor(fwd['graph'].adjacency, dtype=torch.float32)
            traj = fwd['layer2']['trajectories']
            D = trained_model.config.latent_dim

            obs_y = traj[:, -1, 3, :].mean().item()
            xv = torch.full((D,), 5.0)
            int_traj = trained_model.layer2.liquid.intervene(traj, adj, {0: xv})
            int_y = int_traj[:, -1, 3, :].mean().item()

        assert abs(obs_y - int_y) > 1e-6

    def test_graph_is_dag(self, trained_model):
        graph = trained_model.layer2.symbolic_graph()
        assert graph.is_dag()

    def test_graph_density_reasonable(self, trained_model):
        graph = trained_model.layer2.symbolic_graph()
        N = len(graph.nodes)
        max_e = N * (N - 1)
        density = graph.edge_count() / max_e if max_e > 0 else 0
        assert 0.0 < density < 0.7


class TestLayerInterface:
    def test_l1_to_l2_detaches(self):
        """Interface L1->L2 should detach gradients."""
        x = torch.randn(4, 8, 8, 10, requires_grad=True)
        y = LayerInterface.l1_to_l2(x)
        assert not y.requires_grad

    def test_l2_to_l3_detaches(self):
        """Interface L2->L3 should detach gradients."""
        x = torch.randn(4, 8, 8, 10, requires_grad=True)
        y = LayerInterface.l2_to_l3(x)
        assert not y.requires_grad


class TestGradientIsolation:
    def test_no_gradient_crossing_l1_to_l2(self, config, observations):
        """Training Layer 2 should NOT update Layer 1 parameters."""
        model = HHCRA(config)
        # Record L1 params
        l1_params_before = {n: p.clone() for n, p in model.layer1.named_parameters()}

        # Train Layer 2 only
        model.train_layer2(observations, verbose=False)

        # L1 params should be unchanged
        for n, p in model.layer1.named_parameters():
            assert torch.equal(p, l1_params_before[n]), \
                f"Layer 1 param {n} changed during Layer 2 training"

    def test_no_gradient_crossing_l2_to_l3(self, config, observations):
        """Training Layer 3 should NOT update Layer 2 parameters."""
        model = HHCRA(config)
        model.train_layer1(observations, verbose=False)
        model.train_layer2(observations, verbose=False)

        l2_params_before = {n: p.clone() for n, p in model.layer2.named_parameters()}

        model.train_layer3(observations, verbose=False)

        for n, p in model.layer2.named_parameters():
            assert torch.equal(p, l2_params_before[n]), \
                f"Layer 2 param {n} changed during Layer 3 training"


class TestStagedTraining:
    def test_train_layer1(self, config, observations):
        model = HHCRA(config)
        model.train_layer1(observations, verbose=False)
        assert len(model.layer1.loss_history) > 0

    def test_train_layer2(self, config, observations):
        model = HHCRA(config)
        model.train_layer1(observations, verbose=False)
        model.train_layer2(observations, verbose=False)
        graph = model.layer2.symbolic_graph()
        assert graph.is_dag()

    def test_train_layer3(self, config, observations):
        model = HHCRA(config)
        model.train_layer3(observations, verbose=False)
        # HRM should have some depth history
        assert len(model.layer3.hrm.depth_history) > 0

    def test_train_all(self, config, observations):
        model = HHCRA(config)
        model.train_all(observations, verbose=False)
        # Should complete without error and produce valid graph
        graph = model.layer2.symbolic_graph()
        assert graph.is_dag()
