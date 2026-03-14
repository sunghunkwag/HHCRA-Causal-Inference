"""Tests for Layer 2: GNN + Liquid Neural Network (Mechanism Layer)."""

import pytest
import torch
import torch.nn as nn
import numpy as np

from hhcra.config import HHCRAConfig
from hhcra.layer2_mechanism import CausalGNN, LiquidNeuralNet, MechanismLayer


class TestCausalGNN:
    def test_adjacency_shape(self, config):
        gnn = CausalGNN(config)
        A = gnn.adjacency()
        assert A.shape == (config.num_vars, config.num_vars)

    def test_no_self_loops(self, config):
        gnn = CausalGNN(config)
        A = gnn.adjacency()
        diag = torch.diag(A)
        assert torch.allclose(diag, torch.zeros_like(diag))

    def test_adjacency_range(self, config):
        gnn = CausalGNN(config)
        A = gnn.adjacency()
        assert A.min() >= 0.0
        assert A.max() <= 1.0

    def test_hard_adjacency_binary(self, config):
        gnn = CausalGNN(config)
        A = gnn.adjacency(hard=True)
        assert torch.all((A == 0) | (A == 1))

    def test_dag_penalty_nonnegative(self, config):
        gnn = CausalGNN(config)
        pen = gnn.dag_penalty()
        assert pen.item() >= -1e-6

    def test_is_nn_module(self, config):
        gnn = CausalGNN(config)
        assert isinstance(gnn, nn.Module)

    def test_notears_loss_computes(self, config):
        gnn = CausalGNN(config)
        latent = torch.randn(4, 8, config.num_vars, config.latent_dim)
        loss = gnn.notears_loss(latent)
        assert loss.dim() == 0

    def test_gradient_flow_through_adjacency(self, config):
        """Verify gradients flow through GNN adjacency."""
        gnn = CausalGNN(config)
        latent = torch.randn(4, 8, config.num_vars, config.latent_dim)
        loss = gnn.notears_loss(latent)
        loss.backward()
        assert gnn.W.grad is not None
        assert gnn.W.grad.abs().sum() > 0

    def test_prune_to_dag(self, config):
        gnn = CausalGNN(config)
        # Set some random weights
        with torch.no_grad():
            gnn.W.data = torch.randn(config.num_vars, config.num_vars) * 2
            gnn.W.data.fill_diagonal_(-10.0)
        gnn.prune_to_dag()
        A = gnn.adjacency(hard=True)
        assert gnn._is_dag(A)

    def test_compute_shd(self, config, ground_truth):
        gnn = CausalGNN(config)
        shd = gnn.compute_shd(ground_truth['true_adjacency'])
        assert isinstance(shd, int)
        assert shd >= 0

    def test_compute_metrics(self, config, ground_truth):
        gnn = CausalGNN(config)
        metrics = gnn.compute_metrics(ground_truth['true_adjacency'])
        assert 'shd' in metrics
        assert 'tpr' in metrics
        assert 'fdr' in metrics
        assert 0 <= metrics['tpr'] <= 1
        assert 0 <= metrics['fdr'] <= 1


class TestLiquidNeuralNet:
    def test_evolve_shape(self, config):
        liquid = LiquidNeuralNet(config)
        embeddings = torch.randn(4, 8, config.num_vars, config.latent_dim)
        adj = torch.rand(config.num_vars, config.num_vars) * 0.5
        traj = liquid.evolve(embeddings, adj)
        assert traj.shape == embeddings.shape

    def test_evolve_no_nan(self, config):
        liquid = LiquidNeuralNet(config)
        embeddings = torch.randn(4, 8, config.num_vars, config.latent_dim)
        adj = torch.rand(config.num_vars, config.num_vars) * 0.5
        traj = liquid.evolve(embeddings, adj)
        assert not torch.any(torch.isnan(traj)).item()

    def test_is_nn_module(self, config):
        liquid = LiquidNeuralNet(config)
        assert isinstance(liquid, nn.Module)

    def test_intervene_changes_output(self, config):
        liquid = LiquidNeuralNet(config)
        embeddings = torch.randn(4, 8, config.num_vars, config.latent_dim)
        adj = torch.rand(config.num_vars, config.num_vars) * 0.5

        traj_obs = liquid.evolve(embeddings, adj)
        xv = torch.full((config.latent_dim,), 5.0)
        traj_int = liquid.intervene(embeddings, adj, {0: xv})

        assert not torch.allclose(traj_obs, traj_int)

    def test_rk4_integration(self, config):
        """Test that RK4 integration works."""
        config_rk4 = HHCRAConfig(liquid_method="rk4")
        liquid = LiquidNeuralNet(config_rk4)
        embeddings = torch.randn(2, 4, config_rk4.num_vars, config_rk4.latent_dim)
        adj = torch.rand(config_rk4.num_vars, config_rk4.num_vars) * 0.5
        traj = liquid.evolve(embeddings, adj)
        assert not torch.any(torch.isnan(traj)).item()

    def test_gradient_flow(self, config):
        """Verify gradients flow through Liquid Net."""
        liquid = LiquidNeuralNet(config)
        embeddings = torch.randn(2, 4, config.num_vars, config.latent_dim)
        adj = torch.rand(config.num_vars, config.num_vars) * 0.5
        traj = liquid.evolve(embeddings, adj)
        loss = traj.mean()
        loss.backward()
        grad_count = sum(1 for p in liquid.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
        assert grad_count > 0


class TestMechanismLayer:
    def test_forward_returns_dict(self, config):
        layer = MechanismLayer(config)
        latent = torch.randn(4, 8, config.num_vars, config.latent_dim)
        out = layer(latent)
        assert 'embeddings' in out
        assert 'adjacency' in out
        assert 'trajectories' in out

    def test_symbolic_graph(self, config):
        layer = MechanismLayer(config)
        latent = torch.randn(4, 8, config.num_vars, config.latent_dim)
        layer(latent)
        graph = layer.symbolic_graph()
        assert graph.is_dag()

    def test_tight_coupling_gradients(self, config):
        """GNN and Liquid Net share gradients within Layer 2."""
        layer = MechanismLayer(config)
        latent = torch.randn(2, 4, config.num_vars, config.latent_dim)
        loss = layer.compute_loss(latent)
        loss.backward()
        # Both GNN and Liquid should have gradients
        gnn_grads = sum(1 for p in layer.gnn.parameters()
                        if p.grad is not None and p.grad.abs().sum() > 0)
        liquid_grads = sum(1 for p in layer.liquid.parameters()
                          if p.grad is not None and p.grad.abs().sum() > 0)
        assert gnn_grads > 0, "GNN has no gradients"
        assert liquid_grads > 0, "Liquid Net has no gradients"

    def test_handle_feedback(self, config):
        layer = MechanismLayer(config)
        old_w = layer.gnn.W.data[1, 0].item()
        layer.handle_feedback({'remove_edge': (1, 0)})
        new_w = layer.gnn.W.data[1, 0].item()
        assert new_w == -5.0

    def test_dag_after_training(self, trained_model):
        """After training, the graph should be a valid DAG."""
        graph = trained_model.layer2.symbolic_graph()
        assert graph.is_dag()

    def test_temporal_granger_warm_init(self, config):
        """v0.8.0: Temporal Granger init should set W from temporal signal."""
        gnn = CausalGNN(config)
        raw_data = torch.randn(8, 10, config.num_vars)
        w_before = gnn.W.data.clone()
        gnn.warm_init_from_data(
            torch.randn(8, 10, config.num_vars, config.latent_dim),
            raw_data=raw_data)
        w_after = gnn.W.data
        assert not torch.equal(w_before, w_after), \
            "W should change after temporal Granger init"

    def test_adaptive_notears_iterations(self, config):
        """v0.8.0: Adaptive iterations should not timeout on small data."""
        import time
        gnn = CausalGNN(config)
        raw_data = torch.randn(4, 8, config.num_vars)
        start = time.time()
        gnn._warm_init_notears(raw_data)
        elapsed = time.time() - start
        assert elapsed < 30.0, f"_warm_init_notears took {elapsed:.1f}s (should be <30s)"

    def test_symbolic_graph_with_active_slots(self, config):
        """v0.8.0: symbolic_graph should filter by active slots."""
        layer = MechanismLayer(config)
        latent = torch.randn(4, 8, config.num_vars, config.latent_dim)
        layer(latent)
        # Only allow slots 0, 1, 2
        active = torch.tensor([0, 1, 2])
        graph = layer.symbolic_graph(active_slots=active)
        # All edges should be between active slots
        for p, c, w in graph.edges:
            assert p in {0, 1, 2} and c in {0, 1, 2}, \
                f"Edge ({p},{c}) involves inactive slot"
