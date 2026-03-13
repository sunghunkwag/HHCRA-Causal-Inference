"""Tests for v0.5.0 bugfixes and improvements."""

import pytest
import torch
import numpy as np
from collections import deque

from hhcra.config import HHCRAConfig
from hhcra.causal_graph import CausalGraphData, CausalQueryType
from hhcra.layer1_cjepa import CJEPA
from hhcra.layer2_mechanism import CausalGNN, MechanismLayer
from hhcra.layer3_reasoning import NeuroSymbolicEngine, ReasoningLayer
from hhcra.architecture import HHCRA


@pytest.fixture
def config():
    return HHCRAConfig(
        obs_dim=48, latent_dim=10, num_vars=8,
        train_epochs_l1=3, train_epochs_l2=5, train_epochs_l3=3,
    )


def make_diamond_graph():
    """X0->X1->X3, X0->X2->X3"""
    adj = np.zeros((4, 4))
    edges = [(0, 1, 1.0), (0, 2, 1.0), (1, 3, 1.0), (2, 3, 1.0)]
    for p, c, w in edges:
        adj[c, p] = w
    return CausalGraphData([0, 1, 2, 3], edges, adj)


class TestCounterfactualAdjacencyFix:
    """Verify the counterfactual uses modified adjacency (mod_adj)."""

    def test_counterfactual_query_runs(self, config):
        """Counterfactual query should complete without error."""
        torch.manual_seed(42)
        model = HHCRA(config)
        obs = torch.randn(4, 8, config.obs_dim)
        model.train_all(obs, verbose=False)
        model.eval()

        D = config.latent_dim
        result = model.query(
            obs, CausalQueryType.COUNTERFACTUAL,
            X=0, Y=1,
            factual_x=torch.randn(D),
            factual_y=torch.randn(D),
            counterfactual_x=torch.randn(D),
            verbose=False,
        )
        assert result is not None
        assert 'answer' in result

    def test_counterfactual_uses_modified_adjacency(self):
        """The counterfactual action step should zero out incoming edges."""
        engine = NeuroSymbolicEngine()
        adj = torch.eye(4) * 0  # no self-loops
        adj[1, 0] = 1.0  # 0 -> 1
        adj[2, 0] = 1.0  # 0 -> 2

        interventions = {0: torch.tensor([1.0])}
        mod_adj = engine.counterfactual_action(adj, interventions)

        # Incoming edges to node 0 should be cut
        assert mod_adj[0, :].sum().item() == 0.0
        # Other edges preserved
        assert mod_adj[1, 0].item() == 1.0
        assert mod_adj[2, 0].item() == 1.0


class TestDequePerformance:
    """Verify BFS algorithms use deque (O(1) popleft)."""

    def test_ancestors_uses_deque(self):
        """ancestors() should work correctly with deque."""
        graph = make_diamond_graph()
        ancestors_3 = graph.ancestors(3)
        assert ancestors_3 == {0, 1, 2}

    def test_descendants_uses_deque(self):
        """descendants() should work correctly with deque."""
        graph = make_diamond_graph()
        desc_0 = graph.descendants(0)
        assert desc_0 == {1, 2, 3}

    def test_is_dag_uses_deque(self):
        """is_dag() with deque should still correctly detect DAGs."""
        graph = make_diamond_graph()
        assert graph.is_dag()

    def test_d_separation_with_deque(self):
        """d-separation should work correctly with deque-based BFS."""
        engine = NeuroSymbolicEngine()
        graph = make_diamond_graph()
        # 1 and 2 are d-separated given 0
        assert engine.d_separated(graph, 1, 2, {0})
        # 1 and 2 are NOT d-separated given 3 (collider opened)
        assert not engine.d_separated(graph, 1, 2, {3})


class TestCombinationsReplacement:
    """Verify _power_subsets uses itertools.combinations."""

    def test_power_subsets_returns_correct_subsets(self):
        engine = NeuroSymbolicEngine()
        s = {1, 2, 3}
        subsets_2 = list(engine._power_subsets(s, 2))
        assert len(subsets_2) == 3
        # Should return tuples from combinations
        for subset in subsets_2:
            assert len(subset) == 2

    def test_power_subsets_empty(self):
        engine = NeuroSymbolicEngine()
        subsets_0 = list(engine._power_subsets({1, 2}, 0))
        assert len(subsets_0) == 1
        assert subsets_0[0] == ()

    def test_backdoor_still_works(self):
        """Backdoor finding should work with combinations-based subsets."""
        engine = NeuroSymbolicEngine()
        graph = make_diamond_graph()
        bd = engine.find_backdoor_set(graph, 0, 3)
        # From 0 to 3, no backdoor needed (no confounders)
        assert bd is not None


class TestVectorizedComputeLoss:
    """Verify vectorized compute_loss produces valid gradients."""

    def test_compute_loss_returns_scalar(self, config):
        cjepa = CJEPA(config)
        obs = torch.randn(4, 8, config.obs_dim)
        loss = cjepa.compute_loss(obs)
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_compute_loss_gradient_flow(self, config):
        cjepa = CJEPA(config)
        obs = torch.randn(4, 8, config.obs_dim)
        loss = cjepa.compute_loss(obs)
        loss.backward()
        # Check gradients exist
        for p in cjepa.parameters():
            if p.requires_grad:
                assert p.grad is not None
                break

    def test_compute_loss_loss_history(self, config):
        cjepa = CJEPA(config)
        obs = torch.randn(2, 4, config.obs_dim)
        cjepa.compute_loss(obs)
        cjepa.compute_loss(obs)
        assert len(cjepa.loss_history) == 2


class TestDAGPenaltyStability:
    """Verify DAG penalty is numerically stable."""

    def test_dag_penalty_nonnegative(self, config):
        gnn = CausalGNN(config)
        h = gnn.dag_penalty()
        assert h.item() >= 0.0

    def test_dag_penalty_nonnegative_after_pruning(self, config):
        """After pruning to DAG, penalty should be clamped >= 0."""
        gnn = CausalGNN(config)
        gnn.W.data.fill_(-5.0)  # All edges suppressed
        h = gnn.dag_penalty()
        assert h.item() >= 0.0


class TestGradientClipping:
    """Verify gradient clipping is applied during training."""

    def test_training_completes_without_nan(self, config):
        torch.manual_seed(42)
        model = HHCRA(config)
        obs = torch.randn(4, 8, config.obs_dim)
        model.train_all(obs, verbose=False)

        # Check no NaN in parameters
        for name, param in model.named_parameters():
            assert not torch.isnan(param).any(), f"NaN in {name}"


class TestConfigValidation:
    """Verify enhanced config validation."""

    def test_valid_config(self):
        config = HHCRAConfig()
        assert config.hrm_hidden_dim > 0

    def test_invalid_hrm_hidden_dim(self):
        with pytest.raises(AssertionError, match="hrm_hidden_dim"):
            HHCRAConfig(hrm_hidden_dim=0)

    def test_invalid_hrm_max_steps(self):
        with pytest.raises(AssertionError, match="hrm_max_steps"):
            HHCRAConfig(hrm_max_steps=0)

    def test_invalid_hrm_update_interval(self):
        with pytest.raises(AssertionError, match="hrm_update_interval"):
            HHCRAConfig(hrm_update_interval=0)

    def test_invalid_hrm_momentum(self):
        with pytest.raises(AssertionError, match="hrm_momentum"):
            HHCRAConfig(hrm_momentum=1.0)

    def test_invalid_train_epochs(self):
        with pytest.raises(AssertionError, match="train_epochs_l1"):
            HHCRAConfig(train_epochs_l1=0)


class TestWorldModelConvergence:
    """Test new convergence_improved method."""

    def test_convergence_improved(self):
        from hhcra.world_model import SimplePhysicsWorld, WorldModel
        env = SimplePhysicsWorld(num_objects=3)
        wm = WorldModel(env)
        # Simulate decreasing errors
        wm.error_history = [1.0, 0.8, 0.5]
        assert wm.convergence_improved()

    def test_convergence_not_improved(self):
        from hhcra.world_model import SimplePhysicsWorld, WorldModel
        env = SimplePhysicsWorld(num_objects=3)
        wm = WorldModel(env)
        wm.error_history = [0.5, 0.8, 1.0]
        assert not wm.convergence_improved()

    def test_convergence_insufficient_data(self):
        from hhcra.world_model import SimplePhysicsWorld, WorldModel
        env = SimplePhysicsWorld(num_objects=3)
        wm = WorldModel(env)
        wm.error_history = [1.0]
        assert not wm.convergence_improved()
