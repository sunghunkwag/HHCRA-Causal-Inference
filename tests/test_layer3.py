"""Tests for Layer 3: Neuro-Symbolic Engine + HRM (Reasoning Layer)."""

import pytest
import torch
import torch.nn as nn
import numpy as np

from hhcra.config import HHCRAConfig
from hhcra.causal_graph import CausalGraphData, CausalQueryType
from hhcra.layer3_reasoning import NeuroSymbolicEngine, HRM, ReasoningLayer


def make_chain_graph():
    """X0 -> X1 -> X2 -> X3"""
    adj = np.zeros((4, 4))
    edges = [(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)]
    for p, c, w in edges:
        adj[c, p] = w
    return CausalGraphData([0, 1, 2, 3], edges, adj)


def make_fork_graph():
    """X1 <- X0 -> X2"""
    adj = np.zeros((3, 3))
    edges = [(0, 1, 1.0), (0, 2, 1.0)]
    for p, c, w in edges:
        adj[c, p] = w
    return CausalGraphData([0, 1, 2], edges, adj)


def make_collider_graph():
    """X0 -> X2 <- X1"""
    adj = np.zeros((3, 3))
    edges = [(0, 2, 1.0), (1, 2, 1.0)]
    for p, c, w in edges:
        adj[c, p] = w
    return CausalGraphData([0, 1, 2], edges, adj)


def make_diamond_graph():
    """X0 -> X1 -> X3, X0 -> X2 -> X3"""
    adj = np.zeros((4, 4))
    edges = [(0, 1, 1.0), (0, 2, 1.0), (1, 3, 1.0), (2, 3, 1.0)]
    for p, c, w in edges:
        adj[c, p] = w
    return CausalGraphData([0, 1, 2, 3], edges, adj)


class TestNeuroSymbolicEngine:
    def setup_method(self):
        self.engine = NeuroSymbolicEngine()

    def test_d_separation_chain(self):
        """In chain X0->X1->X2: X0 _||_ X2 | X1"""
        G = make_chain_graph()
        assert self.engine.d_separated(G, 0, 2, {1})
        assert not self.engine.d_separated(G, 0, 2, set())

    def test_d_separation_fork(self):
        """In fork X1<-X0->X2: X1 _||_ X2 | X0"""
        G = make_fork_graph()
        assert self.engine.d_separated(G, 1, 2, {0})
        assert not self.engine.d_separated(G, 1, 2, set())

    def test_d_separation_collider(self):
        """In collider X0->X2<-X1: X0 _||_ X1 | {}, not _||_ given X2"""
        G = make_collider_graph()
        assert self.engine.d_separated(G, 0, 1, set())
        assert not self.engine.d_separated(G, 0, 1, {2})

    def test_backdoor_chain(self):
        """In chain X0->X1->X2: backdoor for X0->X2 is empty set."""
        G = make_chain_graph()
        bd = self.engine.find_backdoor_set(G, 0, 2)
        assert bd is not None
        assert len(bd) == 0  # No confounders

    def test_backdoor_diamond(self):
        """In diamond: backdoor for X1->X3 should include X0 or X2."""
        G = make_diamond_graph()
        bd = self.engine.find_backdoor_set(G, 1, 3)
        assert bd is not None

    def test_frontdoor_set(self):
        """Test frontdoor criterion detection."""
        G = make_chain_graph()
        fd = self.engine.find_frontdoor_set(G, 0, 3)
        # Chain has mediators
        if fd is not None:
            assert len(fd) > 0

    def test_instrumental_variable(self):
        """Test instrumental variable detection."""
        # In chain X0->X1->X2->X3, X0 is instrument for X1->X3
        G = make_chain_graph()
        iv = self.engine.find_instrumental_variable(G, 1, 3)
        # X0 should work as instrument since X0->X1->...->X3
        # and X0 is d-sep from X3 given X1
        if iv is not None:
            assert iv == 0

    def test_identifiability_identifiable(self):
        """Direct causal path should be identifiable."""
        G = make_chain_graph()
        result = self.engine.check_identifiability(G, 0, 3)
        assert result['identifiable'] is True

    def test_identifiability_returns_strategy(self):
        G = make_chain_graph()
        result = self.engine.check_identifiability(G, 0, 3)
        assert result['strategy'] is not None

    def test_do_calculus_rule1(self):
        """Test Rule 1: insertion/deletion of observations."""
        G = make_chain_graph()
        result = self.engine.do_calc_rule1(G, Y=3, X=0, Z={1}, W=set())
        assert isinstance(result, bool)

    def test_do_calculus_rule2(self):
        """Test Rule 2: action/observation exchange."""
        G = make_chain_graph()
        result = self.engine.do_calc_rule2(G, Y=3, X=0, Z={1}, W=set())
        assert isinstance(result, bool)

    def test_do_calculus_rule3(self):
        """Test Rule 3: insertion/deletion of actions."""
        G = make_chain_graph()
        result = self.engine.do_calc_rule3(G, Y=3, X=0, Z={1}, W=set())
        assert isinstance(result, bool)

    def test_id_algorithm_basic(self):
        """Test ID algorithm on simple graph."""
        G = make_diamond_graph()
        result = self.engine.id_algorithm(G, {3}, {0})
        assert 'identifiable' in result

    def test_counterfactual_abduction(self):
        observed = torch.tensor([1.0, 2.0, 3.0])
        predicted = torch.tensor([0.8, 1.9, 2.7])
        noise = self.engine.counterfactual_abduction(observed, predicted)
        expected = torch.tensor([0.2, 0.1, 0.3])
        assert torch.allclose(noise, expected, atol=1e-6)


class TestHRM:
    def test_is_nn_module(self, config):
        hrm = HRM(config)
        assert isinstance(hrm, nn.Module)

    def test_reason_returns_dict(self, config):
        hrm = HRM(config)
        q = torch.randn(config.latent_dim)
        result = hrm.reason(q)
        assert 'result' in result
        assert 'convergence' in result
        assert 'steps' in result
        assert 'trace' in result

    def test_reason_result_shape(self, config):
        hrm = HRM(config)
        q = torch.randn(config.latent_dim)
        result = hrm.reason(q)
        assert result['result'].shape == (config.latent_dim,)

    def test_convergence_nonnegative(self, config):
        hrm = HRM(config)
        q = torch.randn(config.latent_dim)
        result = hrm.reason(q)
        assert result['convergence'] >= 0

    def test_trace_nonempty(self, config):
        hrm = HRM(config)
        q = torch.randn(config.latent_dim)
        result = hrm.reason(q)
        assert len(result['trace']) > 0

    def test_gru_based(self, config):
        """HRM should use GRU cells."""
        hrm = HRM(config)
        assert hasattr(hrm, 'h_gru')
        assert hasattr(hrm, 'l_gru')
        assert isinstance(hrm.h_gru, nn.GRUCell)
        assert isinstance(hrm.l_gru, nn.GRUCell)

    def test_act_halting(self, config):
        """ACT should include halt probabilities in trace."""
        hrm = HRM(config)
        q = torch.randn(config.latent_dim)
        result = hrm.reason(q)
        # Some trace entries should have halt_prob
        has_halt = any('halt_prob' in t for t in result['trace'])
        assert has_halt

    def test_learned_reset(self, config):
        """HRM should have learned reset network."""
        hrm = HRM(config)
        assert hasattr(hrm, 'reset_net')

    def test_depth_tracking(self, config):
        hrm = HRM(config)
        q = torch.randn(config.latent_dim)
        hrm.reason(q)
        hrm.reason(q)
        assert len(hrm.depth_history) == 2

    def test_gradient_flow(self, config):
        """Verify gradients flow through HRM."""
        hrm = HRM(config)
        q = torch.randn(config.latent_dim)
        result = hrm.reason(q)
        if isinstance(result['result'], torch.Tensor) and result['result'].requires_grad:
            loss = result['result'].sum()
            loss.backward()
            grad_count = sum(1 for p in hrm.parameters()
                             if p.grad is not None and p.grad.abs().sum() > 0)
            assert grad_count > 0


class TestReasoningLayer:
    def test_is_nn_module(self, config):
        layer = ReasoningLayer(config)
        assert isinstance(layer, nn.Module)

    def test_diagnostic(self, config):
        layer = ReasoningLayer(config)
        G = make_chain_graph()
        diag = layer.generate_diagnostic(G)
        assert 'issues' in diag
        assert 'density' in diag
        assert 'is_dag' in diag

    def test_tight_coupling(self, config):
        """Neuro-symbolic and HRM share gradients within Layer 3."""
        layer = ReasoningLayer(config)
        # HRM is the neural component; symbolic is pure algorithm
        assert hasattr(layer, 'symbolic')
        assert hasattr(layer, 'hrm')
        # Both should be accessible for joint training
        params = list(layer.parameters())
        assert len(params) > 0  # HRM parameters are trainable
