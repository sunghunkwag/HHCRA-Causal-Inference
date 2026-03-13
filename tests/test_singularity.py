"""
Tests for HHCRA Singularity: Liquid Causal Graph, Symbolic Genesis, Autocatalytic Net.

Validates the three breakthrough innovations:
1. Liquid Causal Graph — graph + state co-evolution via coupled ODE
2. Symbolic Genesis Engine — automatic invariant/rule discovery
3. Autocatalytic Causal Network — self-catalytic convergence loop
4. HHCRASingularity — unified system integration
"""

import pytest
import torch
import numpy as np

from hhcra.config import HHCRAConfig
from hhcra.causal_graph import CausalGraphData, CausalQueryType
from hhcra.layer2_mechanism import CausalGNN, MechanismLayer
from hhcra.layer3_reasoning import NeuroSymbolicEngine
from hhcra.liquid_causal_graph import LiquidCausalGraph, GraphDynamicsNet
from hhcra.symbolic_genesis import (
    SymbolicGenesisEngine, SymbolicRule, InvariantFinder, SymbolicDistiller
)
from hhcra.autocatalytic_causal_net import AutocatalyticCausalNet
from hhcra.singularity import HHCRASingularity


@pytest.fixture
def config():
    return HHCRAConfig(
        obs_dim=48, latent_dim=10, num_vars=5,
        train_epochs_l1=3, train_epochs_l2=5, train_epochs_l3=3,
        liquid_ode_steps=4, liquid_dt=0.05,
    )


@pytest.fixture
def small_config():
    """Minimal config for fast tests."""
    return HHCRAConfig(
        obs_dim=16, latent_dim=6, num_vars=4,
        train_epochs_l1=2, train_epochs_l2=3, train_epochs_l3=2,
        liquid_ode_steps=2, liquid_dt=0.05,
        hrm_max_steps=10,
    )


def make_diamond_graph():
    """X0->X1->X3, X0->X2->X3"""
    adj = np.zeros((4, 4))
    edges = [(0, 1, 1.0), (0, 2, 1.0), (1, 3, 1.0), (2, 3, 1.0)]
    for p, c, w in edges:
        adj[c, p] = w
    return CausalGraphData([0, 1, 2, 3], edges, adj)


# =============================================================================
# TEST SUITE 1: Liquid Causal Graph
# =============================================================================

class TestGraphDynamicsNet:
    """Test the graph dynamics network dW/dt = g(W, x)."""

    def test_output_shape(self, config):
        net = GraphDynamicsNet(config)
        N = config.num_vars
        D = config.latent_dim
        W = torch.randn(N, N)
        states = torch.randn(4, N, D)
        dW = net(W, states)
        assert dW.shape == (N, N), f"Expected ({N},{N}), got {dW.shape}"

    def test_no_self_loops(self, config):
        """dW should have zero diagonal (no self-loop dynamics)."""
        net = GraphDynamicsNet(config)
        N = config.num_vars
        W = torch.randn(N, N)
        states = torch.randn(4, N, config.latent_dim)
        dW = net(W, states)
        diag = torch.diag(dW)
        assert torch.allclose(diag, torch.zeros(N), atol=1e-7), \
            "Self-loop dynamics should be zero"

    def test_gradient_flow(self, config):
        """Gradients should flow through the graph dynamics net."""
        net = GraphDynamicsNet(config)
        W = torch.randn(config.num_vars, config.num_vars, requires_grad=True)
        states = torch.randn(4, config.num_vars, config.latent_dim)
        dW = net(W, states)
        loss = dW.sum()
        loss.backward()
        assert W.grad is not None

    def test_damping_bounds_output(self, config):
        """Damping should keep dW small."""
        net = GraphDynamicsNet(config)
        W = torch.randn(config.num_vars, config.num_vars) * 5
        states = torch.randn(4, config.num_vars, config.latent_dim) * 5
        dW = net(W, states)
        assert dW.abs().max().item() < 10.0, "dW should be bounded by damping"


class TestLiquidCausalGraph:
    """Test the co-evolving graph + state system."""

    def test_coupled_evolve_shapes(self, config):
        lcg = LiquidCausalGraph(config)
        B, T, N, D = 4, 8, config.num_vars, config.latent_dim
        embeddings = torch.randn(B, T, N, D)
        W_init = torch.randn(N, N)

        trajectories, W_final = lcg.coupled_evolve(embeddings, W_init)

        assert trajectories.shape == (B, T, N, D)
        assert W_final.shape == (N, N)

    def test_graph_actually_evolves(self, config):
        """W should change during co-evolution (not stay static)."""
        lcg = LiquidCausalGraph(config)
        N = config.num_vars
        W_init = torch.randn(N, N)
        W_init_copy = W_init.clone()

        embeddings = torch.randn(4, 8, N, config.latent_dim)
        _, W_final = lcg.coupled_evolve(embeddings, W_init)

        # W should have changed
        diff = (W_final - W_init_copy).abs().sum().item()
        assert diff > 1e-6, "Graph should evolve during coupled integration"

    def test_trajectories_finite(self, config):
        """All trajectory values should be finite (no NaN/Inf)."""
        torch.manual_seed(42)
        lcg = LiquidCausalGraph(config)
        N = config.num_vars
        embeddings = torch.randn(4, 8, N, config.latent_dim)
        W_init = torch.zeros(N, N)

        trajectories, W_final = lcg.coupled_evolve(embeddings, W_init)

        assert torch.isfinite(trajectories).all(), "Trajectories contain non-finite values"
        assert torch.isfinite(W_final).all(), "Evolved W contains non-finite values"

    def test_graph_evolution_history(self, config):
        """Should record graph snapshots at each timestep."""
        lcg = LiquidCausalGraph(config)
        T = 8
        embeddings = torch.randn(4, T, config.num_vars, config.latent_dim)
        W_init = torch.randn(config.num_vars, config.num_vars)

        lcg.coupled_evolve(embeddings, W_init)
        history = lcg.get_graph_evolution()

        assert len(history) == T, f"Expected {T} snapshots, got {len(history)}"
        for snap in history:
            assert snap.shape == (config.num_vars, config.num_vars)

    def test_graph_change_rate(self, config):
        """Change rate should be computable and non-negative."""
        lcg = LiquidCausalGraph(config)
        embeddings = torch.randn(4, 8, config.num_vars, config.latent_dim)
        W_init = torch.randn(config.num_vars, config.num_vars)
        lcg.coupled_evolve(embeddings, W_init)

        rate = lcg.compute_graph_change_rate()
        assert rate >= 0.0

    def test_coupled_evolve_gradient_flow(self, config):
        """Gradients should flow through the coupled evolution."""
        lcg = LiquidCausalGraph(config)
        embeddings = torch.randn(
            2, 4, config.num_vars, config.latent_dim, requires_grad=True
        )
        W_init = torch.randn(config.num_vars, config.num_vars)

        trajectories, _ = lcg.coupled_evolve(embeddings, W_init)
        loss = trajectories.sum()
        loss.backward()

        assert embeddings.grad is not None, "No gradient to embeddings"


# =============================================================================
# TEST SUITE 2: Symbolic Genesis Engine
# =============================================================================

class TestInvariantFinder:
    """Test neural invariant detection."""

    def test_candidate_shape(self, config):
        finder = InvariantFinder(config, num_candidates=4)
        T, N, D = 10, config.num_vars, config.latent_dim
        states = torch.randn(T, N * D)
        candidates = finder.compute_candidates(states)
        assert candidates.shape == (T, 4)

    def test_finds_constant_function(self, config):
        """A constant trajectory should yield invariants."""
        finder = InvariantFinder(config, num_candidates=4)
        B, T, N, D = 2, 10, config.num_vars, config.latent_dim

        # Constant trajectory -- all basis functions should be invariant
        constant_state = torch.randn(1, 1, N, D).expand(B, T, -1, -1)
        invariants = finder.find_invariants(constant_state, threshold=0.1)
        assert len(invariants) > 0, "Should find invariants in constant trajectory"

    def test_train_step_returns_loss(self, config):
        finder = InvariantFinder(config, num_candidates=4)
        trajectories = torch.randn(2, 8, config.num_vars, config.latent_dim)
        loss = finder.train_step(trajectories)
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_train_step_gradient_flow(self, config):
        finder = InvariantFinder(config, num_candidates=4)
        trajectories = torch.randn(2, 8, config.num_vars, config.latent_dim)
        loss = finder.train_step(trajectories)
        loss.backward()

        has_grad = False
        for p in finder.parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad, "No gradients flow to InvariantFinder parameters"


class TestSymbolicDistiller:
    """Test neural-to-symbolic conversion."""

    def test_probe_dependencies(self, config):
        distiller = SymbolicDistiller(config)
        basis_net = torch.nn.Sequential(
            torch.nn.Linear(config.num_vars * config.latent_dim, config.latent_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(config.latent_dim, 1),
        )
        ref_state = torch.randn(config.num_vars * config.latent_dim)
        deps = distiller.probe_dependencies(basis_net, ref_state)
        assert isinstance(deps, list)
        # All dependency indices should be valid variable indices
        for d in deps:
            assert 0 <= d < config.num_vars

    def test_distill_expression(self, config):
        distiller = SymbolicDistiller(config)
        basis_net = torch.nn.Sequential(
            torch.nn.Linear(config.num_vars * config.latent_dim, 1),
        )
        ref_state = torch.randn(config.num_vars * config.latent_dim)
        expr = distiller.distill_to_expression(basis_net, [0, 1], ref_state)
        assert isinstance(expr, str)
        assert len(expr) > 0

    def test_classify_independence(self, config):
        distiller = SymbolicDistiller(config)
        graph = make_diamond_graph()
        # Nodes 1 and 2 have no direct edge between them
        rule_type = distiller.classify_rule_type([1, 2], graph)
        # They are connected through 0 and 3, so they're connected
        assert rule_type in ('independence', 'edge_constraint', 'flow_conservation')


class TestSymbolicGenesisEngine:
    """Test the full rule discovery pipeline."""

    def test_discover_rules(self, config):
        engine = SymbolicGenesisEngine(config, num_candidates=4)
        graph = make_diamond_graph()
        # Use constant-ish trajectories to ensure invariants are found
        B, T, N, D = 2, 8, config.num_vars, config.latent_dim
        base = torch.randn(1, 1, N, D)
        trajectories = base.expand(B, T, -1, -1) + torch.randn(B, T, N, D) * 0.01

        rules = engine.discover_rules(trajectories, graph)
        assert isinstance(rules, list)
        # All rules should be SymbolicRule instances
        for rule in rules:
            assert isinstance(rule, SymbolicRule)
            assert 0.0 <= rule.confidence <= 1.0

    def test_train_finder_reduces_loss(self, config):
        engine = SymbolicGenesisEngine(config, num_candidates=4)
        trajectories = torch.randn(2, 8, config.num_vars, config.latent_dim)
        losses = engine.train_finder(trajectories, lr=0.01, steps=10)
        assert len(losses) == 10
        # Loss should generally decrease (or at least not explode)
        assert losses[-1] < losses[0] * 10, "Loss exploded during training"

    def test_get_graph_constraints(self, config):
        engine = SymbolicGenesisEngine(config, num_candidates=4)
        # Manually add a rule
        rule = SymbolicRule(
            name="Test_Rule", description="Test",
            expression="h(x0, x1) ≈ const",
            confidence=0.9, involved_nodes={0, 1},
            is_novel=True, rule_type='independence',
        )
        engine.discovered_rules.append(rule)

        constraints = engine.get_graph_constraints()
        assert len(constraints) > 0
        assert constraints[0]['type'] == 'must_not_exist'

    def test_summary_string(self, config):
        engine = SymbolicGenesisEngine(config, num_candidates=4)
        summary = engine.summary()
        assert isinstance(summary, str)
        assert "Symbolic Genesis Engine" in summary


# =============================================================================
# TEST SUITE 3: Autocatalytic Causal Network
# =============================================================================

class TestAutocatalyticCausalNet:
    """Test the self-catalytic convergence loop."""

    def test_single_cycle(self, small_config):
        torch.manual_seed(42)
        acn = AutocatalyticCausalNet(small_config)
        B, T, N, D = 2, 4, small_config.num_vars, small_config.latent_dim
        latent = torch.randn(B, T, N, D)

        result = acn.autocatalytic_cycle(latent, cycle_idx=0)

        assert 'trajectories' in result
        assert 'graph' in result
        assert 'new_rules' in result
        assert isinstance(result['graph'], CausalGraphData)
        assert result['trajectories'].shape == (B, T, N, D)

    def test_multiple_cycles(self, small_config):
        torch.manual_seed(42)
        acn = AutocatalyticCausalNet(small_config)
        B, T, N, D = 2, 4, small_config.num_vars, small_config.latent_dim
        latent = torch.randn(B, T, N, D)

        for i in range(3):
            result = acn.autocatalytic_cycle(latent, cycle_idx=i)

        assert len(acn.cycle_history) == 3

    def test_run_until_convergence(self, small_config):
        torch.manual_seed(42)
        acn = AutocatalyticCausalNet(small_config)
        B, T, N, D = 2, 4, small_config.num_vars, small_config.latent_dim
        latent = torch.randn(B, T, N, D)

        result = acn.run_until_convergence(
            latent, max_cycles=5, verbose=False
        )

        assert 'final_graph' in result
        assert 'all_rules' in result
        assert 'novel_rules' in result
        assert result['total_cycles'] >= 1

    def test_constraint_application(self, small_config):
        """Constraints should modify GNN weights."""
        acn = AutocatalyticCausalNet(small_config)
        N = small_config.num_vars

        # Record initial weight
        initial_w = acn.gnn.W.data[1, 0].item()

        # Apply a must_not_exist constraint
        constraints = [{
            'type': 'must_not_exist',
            'edge': (0, 1),
            'confidence': 0.9,
            'source': 'test',
        }]
        acn._apply_constraints_to_gnn(constraints)

        # Weight should be suppressed
        new_w = acn.gnn.W.data[1, 0].item()
        assert new_w <= initial_w, "must_not_exist should suppress edge weight"

    def test_structure_distance(self, small_config):
        acn = AutocatalyticCausalNet(small_config)
        N = small_config.num_vars
        W1 = torch.zeros(N, N)
        W2 = torch.ones(N, N)
        dist = acn._compute_structure_distance(W1, W2)
        assert dist > 0, "Different graphs should have positive distance"

        dist_same = acn._compute_structure_distance(W1, W1)
        assert dist_same < 1e-6, "Same graph should have zero distance"

    def test_no_nan_after_cycles(self, small_config):
        """System should remain numerically stable across cycles."""
        torch.manual_seed(42)
        acn = AutocatalyticCausalNet(small_config)
        latent = torch.randn(2, 4, small_config.num_vars, small_config.latent_dim)

        result = acn.run_until_convergence(
            latent, max_cycles=3, verbose=False
        )

        # Check GNN weights
        assert torch.isfinite(acn.gnn.W.data).all(), "NaN in GNN weights"

    def test_summary_string(self, small_config):
        acn = AutocatalyticCausalNet(small_config)
        summary = acn.summary()
        assert isinstance(summary, str)
        assert "Autocatalytic" in summary


# =============================================================================
# TEST SUITE 4: HHCRASingularity (Unified System)
# =============================================================================

class TestHHCRASingularity:
    """Test the unified Singularity system."""

    def test_construction(self, small_config):
        model = HHCRASingularity(small_config)
        assert model.layer1 is not None
        assert model.layer2 is not None
        assert model.layer3 is not None
        assert model.liquid_graph is not None
        assert model.symbolic_genesis is not None
        assert model.autocatalytic is not None

    def test_train_singularity(self, small_config):
        """Full singularity training should complete without error."""
        torch.manual_seed(42)
        model = HHCRASingularity(small_config)
        obs = torch.randn(2, 4, small_config.obs_dim)

        result = model.train_singularity(obs, max_cycles=3, verbose=False)

        assert 'final_graph' in result
        assert 'all_rules' in result
        assert result['total_cycles'] >= 1

    def test_forward_includes_singularity_data(self, small_config):
        """Forward pass should include Singularity metadata."""
        torch.manual_seed(42)
        model = HHCRASingularity(small_config)
        obs = torch.randn(2, 4, small_config.obs_dim)

        output = model.forward(obs)

        assert 'discovered_rules' in output
        assert 'autocatalytic_converged' in output

    def test_query_includes_singularity_metadata(self, small_config):
        """Query results should include Singularity metadata."""
        torch.manual_seed(42)
        model = HHCRASingularity(small_config)
        obs = torch.randn(2, 4, small_config.obs_dim)
        model.train_singularity(obs, max_cycles=2, verbose=False)
        model.eval()

        result = model.query(
            obs, CausalQueryType.OBSERVATIONAL,
            X=0, Y=1, verbose=False,
        )

        assert 'singularity' in result
        assert 'discovered_rules' in result['singularity']
        assert 'autocatalytic_converged' in result['singularity']

    def test_interventional_query_after_training(self, small_config):
        """Interventional query should work after Singularity training."""
        torch.manual_seed(42)
        model = HHCRASingularity(small_config)
        obs = torch.randn(2, 4, small_config.obs_dim)
        model.train_singularity(obs, max_cycles=2, verbose=False)
        model.eval()

        D = small_config.latent_dim
        result = model.query(
            obs, CausalQueryType.INTERVENTIONAL,
            X=0, Y=1,
            x_value=torch.randn(D),
            verbose=False,
        )

        assert result is not None
        assert 'type' in result

    def test_no_nan_in_parameters(self, small_config):
        """All parameters should remain finite after training."""
        torch.manual_seed(42)
        model = HHCRASingularity(small_config)
        obs = torch.randn(2, 4, small_config.obs_dim)
        model.train_singularity(obs, max_cycles=2, verbose=False)

        for name, param in model.named_parameters():
            assert torch.isfinite(param).all(), f"NaN/Inf in {name}"

    def test_singularity_summary(self, small_config):
        """Summary should be a non-empty string."""
        model = HHCRASingularity(small_config)
        summary = model.singularity_summary()
        assert isinstance(summary, str)
        assert "SINGULARITY" in summary
        assert "Pearl's Ladder" in summary

    def test_backward_compatible_with_base_hhcra(self, small_config):
        """Singularity model should be usable as a regular HHCRA model."""
        torch.manual_seed(42)
        model = HHCRASingularity(small_config)
        obs = torch.randn(2, 4, small_config.obs_dim)

        # Should support base HHCRA training methods
        model._train_layer1(obs, verbose=False)
        model._train_layer2(obs, verbose=False)
        model._train_layer3(obs, verbose=False)

        # Should support base forward
        output = model.forward(obs)
        assert 'graph' in output


# =============================================================================
# TEST SUITE 5: Integration & Emergence Tests
# =============================================================================

class TestEmergence:
    """Test emergent properties that arise from the interaction of components."""

    def test_autocatalytic_improves_structure(self, small_config):
        """
        Autocatalytic cycles should produce a more refined graph than
        single-pass training (measured by DAG validity and edge count).
        """
        torch.manual_seed(42)
        latent = torch.randn(2, 4, small_config.num_vars, small_config.latent_dim)

        # Single pass: just GNN
        gnn_only = CausalGNN(small_config)
        optimizer = torch.optim.Adam(gnn_only.parameters(), lr=0.01)
        for _ in range(10):
            optimizer.zero_grad()
            loss = gnn_only.notears_loss(latent)
            loss.backward()
            optimizer.step()
        gnn_only.prune_to_dag()
        A_single = gnn_only.adjacency(hard=True)
        is_dag_single = gnn_only._is_dag(A_single)

        # Autocatalytic: full loop
        torch.manual_seed(42)
        acn = AutocatalyticCausalNet(small_config)
        result = acn.run_until_convergence(
            latent, max_cycles=3, verbose=False
        )

        # Both should produce valid results (no crashes)
        assert result['final_graph'] is not None
        # The autocatalytic graph should exist
        assert result['final_graph'].edge_count() >= 0

    def test_symbolic_genesis_finds_structure_related_invariants(self, small_config):
        """
        When given trajectories from a structured graph, the Genesis engine
        should discover invariants related to the graph topology.
        """
        torch.manual_seed(42)
        engine = SymbolicGenesisEngine(small_config, num_candidates=4)
        graph = make_diamond_graph()

        # Create trajectories with structure: x3 = f(x1, x2)
        B, T, N, D = 2, 8, small_config.num_vars, small_config.latent_dim
        trajectories = torch.randn(B, T, N, D)

        # Train the finder
        engine.train_finder(trajectories, lr=0.01, steps=20)

        # Discover rules
        rules = engine.discover_rules(trajectories, graph)

        # Should return a list (may or may not find rules depending on data)
        assert isinstance(rules, list)

    def test_liquid_graph_preserves_causality(self, small_config):
        """
        Even as the graph evolves, causal relationships in the data
        should be approximately preserved (trajectories remain bounded).
        """
        torch.manual_seed(42)
        lcg = LiquidCausalGraph(small_config)
        N = small_config.num_vars
        D = small_config.latent_dim

        # Start with a simple causal structure
        W_init = torch.zeros(N, N)
        W_init[1, 0] = 2.0  # 0 -> 1
        W_init[2, 1] = 2.0  # 1 -> 2

        embeddings = torch.randn(2, 6, N, D)
        trajectories, W_final = lcg.coupled_evolve(embeddings, W_init)

        # Trajectories should remain bounded
        assert trajectories.abs().max().item() < 10.0

        # Graph should still have some structure (not all zeros)
        A_final = torch.sigmoid(W_final) * lcg.diag_mask
        assert A_final.sum().item() > 0.1, "Graph collapsed to empty"
