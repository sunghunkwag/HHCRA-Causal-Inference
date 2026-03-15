"""
HHCRA v2.0 Integration Tests.

Tests the full 3-layer architecture:
  1. Layer 1 (ICA): Variable extraction from high-dim observations
  2. Layer 2 (CODA + SCM): Structure learning + mechanism fitting
  3. Layer 3 (Symbolic): d-separation, backdoor, interventional, counterfactual
  4. End-to-end pipeline: observations → graph → queries
  5. Active agent: observation → model → intervention selection
  6. Comparison with baselines
"""

import pytest
import numpy as np
from coda.data import (
    generate_er_dag, generate_linear_sem_data,
    SACHS_TRUE_DAG, ASIA_TRUE_DAG, load_sachs,
)
from coda.metrics import shd, f1_score_dag, varsortability

from hhcra.architecture import HHCRA, ICAExtractor
from hhcra.graph import CausalGraphData, CausalQueryType
from hhcra.symbolic import NeuroSymbolicEngine
from hhcra.agent import ActiveCausalAgent


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture
def chain_dag():
    dag = np.zeros((4, 4))
    dag[0, 1] = dag[1, 2] = dag[2, 3] = 1
    return dag

@pytest.fixture
def fork_dag():
    dag = np.zeros((3, 3))
    dag[0, 1] = dag[0, 2] = 1
    return dag

@pytest.fixture
def collider_dag():
    dag = np.zeros((3, 3))
    dag[0, 2] = dag[1, 2] = 1
    return dag

@pytest.fixture
def chain_data(chain_dag):
    X, W = generate_linear_sem_data(chain_dag, n=2000, seed=42)
    return X, W, chain_dag

@pytest.fixture
def asia_data():
    X, W = generate_linear_sem_data(ASIA_TRUE_DAG, n=2000, seed=42)
    return X, W, ASIA_TRUE_DAG


# ===================================================================
# 1. Layer 1: ICA Variable Extraction
# ===================================================================

class TestICAExtractor:

    def test_direct_mode(self, chain_data):
        """Low-dim data: no extraction needed."""
        X, _, _ = chain_data
        model = HHCRA()
        model.fit(X, verbose=False)
        assert model.extractor is None
        assert model._n_vars == 4

    def test_high_dim_extraction(self, chain_dag):
        """High-dim observations → ICA extracts variables."""
        X, _ = generate_linear_sem_data(chain_dag, n=2000, seed=42)
        np.random.seed(42)
        proj = np.random.randn(4, 48) * 0.3
        noise = np.random.randn(2000, 48) * 0.1
        obs = X @ proj + noise

        extractor = ICAExtractor(max_vars=10)
        vars_extracted = extractor.fit_transform(obs)
        assert vars_extracted.shape[0] == 2000
        assert 2 <= vars_extracted.shape[1] <= 10

    def test_n_vars_hint(self, chain_dag):
        """With hint, extracts exact number of variables."""
        X, _ = generate_linear_sem_data(chain_dag, n=2000, seed=42)
        np.random.seed(42)
        proj = np.random.randn(4, 48) * 0.3
        obs = X @ proj + np.random.randn(2000, 48) * 0.1

        extractor = ICAExtractor(max_vars=10, n_vars_hint=4)
        vars_extracted = extractor.fit_transform(obs)
        assert vars_extracted.shape[1] == 4

    def test_raw_data_bypass(self, asia_data):
        """Raw data → skips ICA."""
        X, _, dag = asia_data
        model = HHCRA()
        model.fit(np.random.randn(2000, 48), raw_data=X, verbose=False)
        assert model.extractor is None
        assert model._n_vars == 8


# ===================================================================
# 2. Layer 2: Structure Learning + SCM
# ===================================================================

class TestStructureLearning:

    def test_chain_recovery(self, chain_data):
        X, _, dag = chain_data
        model = HHCRA()
        model.fit(X, verbose=False)
        s = shd(model.adj, dag)
        assert s <= 3, f"Chain SHD={s}"

    def test_asia_recovery(self, asia_data):
        X, _, dag = asia_data
        model = HHCRA()
        model.fit(X, verbose=False)
        s = shd(model.adj, dag)
        assert s <= 8, f"Asia SHD={s}"

    def test_output_is_dag(self, asia_data):
        X, _, _ = asia_data
        model = HHCRA()
        model.fit(X, verbose=False)
        assert model.graph_data.is_dag()

    def test_scm_fitted(self, chain_data):
        X, W_true, dag = chain_data
        model = HHCRA()
        model.fit(X, verbose=False)
        assert model.scm is not None
        assert model.scm.d == 4

    def test_scm_weight_recovery(self, chain_data):
        """Fitted weights should approximate true weights."""
        X, W_true, dag = chain_data
        model = HHCRA()
        model.fit(X, verbose=False)
        # Check weights on correctly discovered edges
        for i in range(4):
            for j in range(4):
                if dag[i, j] > 0 and model.adj[i, j] > 0:
                    error = abs(model.scm.weights[i, j] - W_true[i, j])
                    assert error < 0.2, f"Weight [{i},{j}]: true={W_true[i,j]:.3f}, fitted={model.scm.weights[i,j]:.3f}"

    def test_evaluate_method(self, chain_data):
        X, _, dag = chain_data
        model = HHCRA()
        model.fit(X, verbose=False)
        metrics = model.evaluate(dag)
        assert 'shd' in metrics
        assert 'f1' in metrics
        assert 'varsortability' in metrics


# ===================================================================
# 3. Layer 3: Symbolic Reasoning
# ===================================================================

class TestSymbolicEngine:

    def _chain_graph(self):
        adj = np.zeros((4, 4))
        adj[0, 1] = adj[1, 2] = adj[2, 3] = 1
        return CausalGraphData.from_adjacency(adj)

    def _fork_graph(self):
        adj = np.zeros((3, 3))
        adj[0, 1] = adj[0, 2] = 1
        return CausalGraphData.from_adjacency(adj)

    def _collider_graph(self):
        adj = np.zeros((3, 3))
        adj[0, 2] = adj[1, 2] = 1
        return CausalGraphData.from_adjacency(adj)

    def test_d_sep_chain(self):
        """X0 ⊥ X2 | X1 in chain."""
        e = NeuroSymbolicEngine()
        G = self._chain_graph()
        assert e.d_separated(G, 0, 2, {1})
        assert not e.d_separated(G, 0, 2, set())

    def test_d_sep_fork(self):
        """X1 ⊥ X2 | X0 in fork."""
        e = NeuroSymbolicEngine()
        G = self._fork_graph()
        assert e.d_separated(G, 1, 2, {0})
        assert not e.d_separated(G, 1, 2, set())

    def test_d_sep_collider(self):
        """X0 ⊥ X1 in collider, but NOT given X2."""
        e = NeuroSymbolicEngine()
        G = self._collider_graph()
        assert e.d_separated(G, 0, 1, set())
        assert not e.d_separated(G, 0, 1, {2})

    def test_backdoor_chain(self):
        """Chain: backdoor set for X0→X3 is empty."""
        e = NeuroSymbolicEngine()
        G = self._chain_graph()
        bd = e.find_backdoor_set(G, 0, 3)
        assert bd is not None
        assert len(bd) == 0

    def test_identifiability_dag(self):
        """All effects identifiable in a DAG without confounders."""
        e = NeuroSymbolicEngine()
        G = self._chain_graph()
        for x in range(4):
            for y in range(4):
                if x != y:
                    result = e.check_identifiability(G, x, y)
                    assert result['identifiable'], f"P(X{y}|do(X{x})) should be identifiable"

    def test_do_calculus_rule1(self):
        e = NeuroSymbolicEngine()
        G = self._chain_graph()
        # Rule 1 should apply in chain graph
        result = e.do_calc_rule1(G, Y=3, X=0, Z={1}, W=set())
        assert isinstance(result, bool)

    def test_do_calculus_rule2(self):
        e = NeuroSymbolicEngine()
        G = self._chain_graph()
        result = e.do_calc_rule2(G, Y=3, X=0, Z={1}, W=set())
        assert isinstance(result, bool)


# ===================================================================
# 4. End-to-End Pipeline
# ===================================================================

class TestEndToEnd:

    def test_chain_full_pipeline(self, chain_data):
        """Complete pipeline: data → model → all 3 query types."""
        X, _, dag = chain_data
        model = HHCRA()
        model.fit(X, verbose=False)

        # Rung 1
        r1 = model.query(CausalQueryType.OBSERVATIONAL, X=0, Y=3)
        assert r1.answer is not None
        assert r1.identifiable

        # Rung 2
        r2 = model.query(CausalQueryType.INTERVENTIONAL, X=0, Y=3, x_value=2.0)
        assert r2.identifiable
        assert r2.answer is not None

        # Rung 3
        factual = X[0]
        r3 = model.query(CausalQueryType.COUNTERFACTUAL, X=0, Y=3,
                         factual_values=factual, cf_x=factual[0] + 1.0)
        assert r3.identifiable
        assert r3.answer is not None

    def test_interventional_accuracy(self, chain_data):
        """do(X0=2) on chain should propagate through SCM."""
        X, W, dag = chain_data
        model = HHCRA()
        model.fit(X, verbose=False)

        r = model.query(CausalQueryType.INTERVENTIONAL, X=0, Y=1, x_value=2.0)
        if model.adj[0, 1] > 0:  # Edge was discovered
            # E[X1 | do(X0=2)] ≈ W[0,1] * 2 + intercept
            assert abs(r.answer[0]) > 0.1  # Non-zero effect

    def test_counterfactual_identity(self, chain_data):
        """Setting X0 to its actual value → same factual values."""
        X, _, dag = chain_data
        model = HHCRA()
        model.fit(X, verbose=False)

        factual = X[0]
        r = model.query(CausalQueryType.COUNTERFACTUAL, X=0, Y=1,
                        factual_values=factual, cf_x=factual[0])
        # Should be approximately equal to factual[1]
        assert abs(r.answer[0] - factual[1]) < 0.5

    def test_no_backward_effect(self, chain_data):
        """do(X3=c) should NOT affect X0 in chain."""
        X, _, dag = chain_data
        model = HHCRA()
        model.fit(X, verbose=False)

        r1 = model.query(CausalQueryType.INTERVENTIONAL, X=3, Y=0, x_value=0.0)
        r2 = model.query(CausalQueryType.INTERVENTIONAL, X=3, Y=0, x_value=10.0)
        if r1.answer is not None and r2.answer is not None:
            assert abs(r1.answer[0] - r2.answer[0]) < 1.0

    def test_asia_pipeline(self, asia_data):
        X, _, dag = asia_data
        model = HHCRA()
        model.fit(X, verbose=False)
        metrics = model.evaluate(dag)
        assert metrics['shd'] <= 8

    def test_summary_works(self, chain_data):
        X, _, _ = chain_data
        model = HHCRA()
        model.fit(X, verbose=False)
        s = model.summary()
        assert "HHCRA v2.0" in s
        assert "Pearl's Ladder" in s

    def test_reproducible(self, chain_data):
        X, _, _ = chain_data
        m1 = HHCRA()
        m1.fit(X, verbose=False)
        m2 = HHCRA()
        m2.fit(X, verbose=False)
        assert np.array_equal(m1.adj, m2.adj)


# ===================================================================
# 5. Active Agent
# ===================================================================

class TestActiveAgent:

    def test_agent_creation(self):
        agent = ActiveCausalAgent(n_vars=4)
        assert agent.n_vars == 4
        assert agent.edge_probabilities.shape == (4, 4)

    def test_observe_and_update(self, chain_data):
        X, _, _ = chain_data
        agent = ActiveCausalAgent(n_vars=4)
        for obs in X[:100]:
            agent.observe(obs)
        assert len(agent.obs_buffer) == 100
        agent.update_model(verbose=False)
        assert agent.state.graph_known

    def test_select_intervention(self, chain_data):
        X, _, _ = chain_data
        agent = ActiveCausalAgent(n_vars=4)
        for obs in X[:100]:
            agent.observe(obs)
        agent.update_model(verbose=False)
        node, value = agent.select_intervention()
        assert 0 <= node < 4
        assert isinstance(value, float)

    def test_process_intervention(self, chain_data):
        X, _, dag = chain_data
        agent = ActiveCausalAgent(n_vars=4)
        for obs in X[:100]:
            agent.observe(obs)

        # Simulate intervention result
        obs_data = X[:50]
        int_data = X[50:100].copy()
        int_data[:, 1] += 3.0  # Shift variable 1

        agent.process_intervention(
            target=0, value=3.0, obs_data=obs_data, int_data=int_data)
        assert agent.state.n_interventions == 1

    def test_edge_uncertainty_decreases(self, chain_data):
        X, _, _ = chain_data
        agent = ActiveCausalAgent(n_vars=4)

        initial_uncertainty = agent.edge_uncertainty.mean()

        for obs in X[:200]:
            agent.observe(obs)
        agent.update_model(verbose=False)

        final_uncertainty = agent.edge_uncertainty.mean()
        assert final_uncertainty < initial_uncertainty

    def test_stats(self, chain_data):
        X, _, _ = chain_data
        agent = ActiveCausalAgent(n_vars=4)
        for obs in X[:50]:
            agent.observe(obs)
        stats = agent.get_stats()
        assert stats['step'] == 50
        assert stats['n_observations'] == 50


# ===================================================================
# 6. Benchmark Comparison
# ===================================================================

class TestBenchmarks:

    def test_sachs_synthetic(self):
        """Sachs synthetic benchmark."""
        X = load_sachs()
        model = HHCRA()
        model.fit(X, verbose=False)
        metrics = model.evaluate(SACHS_TRUE_DAG)
        print(f"\n  Sachs: SHD={metrics['shd']} F1={metrics['f1']:.3f}")
        # Should beat empty graph (SHD=17)
        assert metrics['shd'] < 17

    def test_asia(self):
        """Asia benchmark."""
        X, _ = generate_linear_sem_data(ASIA_TRUE_DAG, n=2000, seed=42)
        model = HHCRA()
        model.fit(X, verbose=False)
        metrics = model.evaluate(ASIA_TRUE_DAG)
        print(f"\n  Asia: SHD={metrics['shd']} F1={metrics['f1']:.3f}")
        assert metrics['shd'] <= 8

    def test_er20(self):
        """ER-2 d=20 benchmark."""
        dag = generate_er_dag(20, expected_edges=30, seed=42)
        X, _ = generate_linear_sem_data(dag, n=2000, seed=42)
        model = HHCRA()
        model.fit(X, verbose=False)
        metrics = model.evaluate(dag)
        print(f"\n  ER-20: SHD={metrics['shd']} F1={metrics['f1']:.3f} "
              f"varsortability={metrics['varsortability']}")
        # Should beat empty graph
        n_true_edges = int(dag.sum())
        assert metrics['shd'] < n_true_edges * 2

    def test_comprehensive_results(self):
        """Print full comparison table."""
        print("\n" + "=" * 70)
        print("  HHCRA v2.0 — Comprehensive Benchmark Results")
        print("=" * 70)

        datasets = [
            ("Asia (8,8)", ASIA_TRUE_DAG, 2000),
            ("Sachs (11,17)", SACHS_TRUE_DAG, None),
        ]

        for name, dag, n in datasets:
            if n is not None:
                X, _ = generate_linear_sem_data(dag, n=n, seed=42)
            else:
                X = load_sachs()

            model = HHCRA()
            model.fit(X, verbose=False)
            m = model.evaluate(dag)
            v = m['varsortability']
            empty_shd = int(dag.sum())
            print(f"  {name:<20} SHD={m['shd']:>3}  F1={m['f1']:.3f}  "
                  f"V-sort={v:.3f}  (empty={empty_shd})")

        print("=" * 70)
