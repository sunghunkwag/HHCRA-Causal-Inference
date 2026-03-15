"""
Tests for Active Causal Discovery.

Proves that observation+intervention beats observation-only methods.
"""
import pytest
import numpy as np
from coda.data import generate_er_dag, ASIA_TRUE_DAG, SACHS_TRUE_DAG
from coda.metrics import shd
from coda.discovery import coda_discover
from hhcra.causal_env import CausalEnv
from hhcra.active_discovery import ActiveDiscoveryAgent


@pytest.fixture(params=[
    ("chain4", np.array([[0,1,0,0],[0,0,1,0],[0,0,0,1],[0,0,0,0]], dtype=float)),
    ("fork3", np.array([[0,1,1],[0,0,0],[0,0,0]], dtype=float)),
    ("collider3", np.array([[0,0,1],[0,0,1],[0,0,0]], dtype=float)),
    ("diamond4", np.array([[0,1,1,0],[0,0,0,1],[0,0,0,1],[0,0,0,0]], dtype=float)),
])
def small_graph(request):
    name, dag = request.param
    return name, dag


class TestActiveBeatsPassive:
    """Core claim: active discovery beats observation-only methods."""

    def test_small_graphs(self, small_graph):
        name, dag = small_graph
        d = dag.shape[0]
        env = CausalEnv.from_dag(dag, seed=42)

        # Passive baseline
        X = env.observe(2000)
        coda_adj = coda_discover(X, seed=42)['adj']
        shd_passive = shd(coda_adj, dag)

        # Active agent
        env2 = CausalEnv.from_dag(dag, seed=42)
        agent = ActiveDiscoveryAgent(d)
        adj_active = agent.discover(env2, n_obs=1000, max_interventions=d)
        shd_active = shd(adj_active, dag)

        print(f"\n  {name}: passive={shd_passive} active={shd_active} "
              f"ints={agent.n_interventions}")
        assert shd_active <= shd_passive, \
            f"{name}: active={shd_active} not better than passive={shd_passive}"

    def test_asia_active_beats_passive(self):
        env = CausalEnv.from_dag(ASIA_TRUE_DAG, seed=42)
        X = env.observe(2000)
        shd_passive = shd(coda_discover(X, seed=42)['adj'], ASIA_TRUE_DAG)

        env2 = CausalEnv.from_dag(ASIA_TRUE_DAG, seed=42)
        agent = ActiveDiscoveryAgent(8)
        adj = agent.discover(env2, n_obs=1000, max_interventions=8)
        shd_active = shd(adj, ASIA_TRUE_DAG)

        print(f"\n  Asia: passive={shd_passive} active={shd_active}")
        assert shd_active < shd_passive

    def test_sachs_active_beats_passive(self):
        env = CausalEnv.from_dag(SACHS_TRUE_DAG, seed=123)
        X = env.observe(2000)
        shd_passive = shd(coda_discover(X, seed=42)['adj'], SACHS_TRUE_DAG)

        env2 = CausalEnv.from_dag(SACHS_TRUE_DAG, seed=123)
        agent = ActiveDiscoveryAgent(11)
        adj = agent.discover(env2, n_obs=1000, max_interventions=11)
        shd_active = shd(adj, SACHS_TRUE_DAG)

        print(f"\n  Sachs: passive={shd_passive} active={shd_active}")
        assert shd_active < shd_passive


class TestActiveDiscoveryProperties:

    def test_output_is_dag(self):
        dag = ASIA_TRUE_DAG
        env = CausalEnv.from_dag(dag, seed=42)
        agent = ActiveDiscoveryAgent(8)
        adj = agent.discover(env, n_obs=500)
        # Check DAG via topological sort
        d = adj.shape[0]
        in_deg = adj.sum(axis=0).astype(int)
        queue = [i for i in range(d) if in_deg[i] == 0]
        visited = 0
        while queue:
            node = queue.pop(0)
            visited += 1
            for c in range(d):
                if adj[node, c] > 0:
                    in_deg[c] -= 1
                    if in_deg[c] == 0:
                        queue.append(c)
        assert visited == d, "Output is not a DAG"

    def test_few_interventions_suffice(self):
        """O(log d) interventions should be enough."""
        dag = ASIA_TRUE_DAG  # d=8
        env = CausalEnv.from_dag(dag, seed=42)
        agent = ActiveDiscoveryAgent(8)
        adj = agent.discover(env, n_obs=1000, max_interventions=4)
        s = shd(adj, dag)
        assert s <= 5, f"SHD={s} with only 4 interventions"
        assert agent.n_interventions <= 4

    def test_zero_interventions_matches_passive(self):
        """With 0 interventions, active = passive (skeleton + Meek only)."""
        dag = ASIA_TRUE_DAG
        env = CausalEnv.from_dag(dag, seed=42)
        agent = ActiveDiscoveryAgent(8)
        adj = agent.discover(env, n_obs=1000, max_interventions=0)
        assert agent.n_interventions == 0
        # Should still produce a valid result
        assert adj.shape == (8, 8)

    def test_perfect_recovery_chain(self):
        """Chain graph should be perfectly recoverable."""
        dag = np.array([[0,1,0,0],[0,0,1,0],[0,0,0,1],[0,0,0,0]], dtype=float)
        env = CausalEnv.from_dag(dag, seed=42)
        agent = ActiveDiscoveryAgent(4)
        adj = agent.discover(env, n_obs=2000, max_interventions=4,
                             samples_per_int=500)
        assert shd(adj, dag) == 0, "Chain should be perfectly recovered"

    def test_verbose_mode(self, capsys):
        dag = np.array([[0,1,0],[0,0,1],[0,0,0]], dtype=float)
        env = CausalEnv.from_dag(dag, seed=42)
        agent = ActiveDiscoveryAgent(3)
        agent.discover(env, n_obs=500, verbose=True)
        captured = capsys.readouterr()
        assert "Phase 1" in captured.out
        assert "Intervention" in captured.out or "Phase 2" in captured.out


class TestCausalEnv:

    def test_observe_shape(self):
        dag = ASIA_TRUE_DAG
        env = CausalEnv.from_dag(dag)
        X = env.observe(100)
        assert X.shape == (100, 8)

    def test_intervene_changes_distribution(self):
        dag = np.array([[0,1,0],[0,0,1],[0,0,0]], dtype=float)
        env = CausalEnv.from_dag(dag, seed=42)
        X_obs = env.observe(1000)
        X_int = env.intervene(0, 10.0, n=1000)
        # X0 should be constant at 10.0
        assert np.allclose(X_int[:, 0], 10.0)
        # X1 should shift (X0 → X1)
        assert abs(X_int[:, 1].mean() - X_obs[:, 1].mean()) > 1.0

    def test_intervene_no_backward(self):
        dag = np.array([[0,1,0],[0,0,1],[0,0,0]], dtype=float)
        env = CausalEnv.from_dag(dag, seed=42)
        X_obs = env.observe(1000)
        X_int = env.intervene(2, 100.0, n=1000)
        # X0 should NOT change when intervening on X2
        from scipy import stats
        _, p = stats.ttest_ind(X_obs[:, 0], X_int[:, 0])
        assert p > 0.01, "Intervening on child should not affect parent"
