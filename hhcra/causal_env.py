"""
Causal Environment for active discovery experiments.

The agent can:
  - observe(n): draw samples from P(X) — free
  - intervene(node, value, n): draw samples from P(X|do(X_i=v)) — costs budget

Hidden DAG is NOT visible to the agent.
"""
import numpy as np
from coda.metrics import topological_order_from_dag


class CausalEnv:
    """Environment with hidden causal structure."""

    def __init__(self, true_dag, weights, noise_std, seed=42):
        self.true_dag = true_dag
        self.weights = weights
        self.noise_std = noise_std
        self.d = true_dag.shape[0]
        self._rng = np.random.RandomState(seed)
        self._topo = topological_order_from_dag(true_dag)
        self.intervention_count = 0

    def observe(self, n=200):
        """Draw n observational samples."""
        return self._sample(n)

    def intervene(self, node, value, n=200):
        """Perform do(X_node=value), return n samples."""
        self.intervention_count += 1
        return self._sample(n, do_node=node, do_value=value)

    def _sample(self, n, do_node=None, do_value=None):
        X = np.zeros((n, self.d))
        for j in self._topo:
            if j == do_node:
                X[:, j] = do_value
                continue
            parents = np.where(self.true_dag[:, j] > 0)[0]
            noise = self._rng.randn(n) * self.noise_std[j]
            if len(parents) > 0:
                X[:, j] = X[:, parents] @ self.weights[parents, j] + noise
            else:
                X[:, j] = noise
        return X

    @staticmethod
    def from_dag(dag, seed=42):
        rng = np.random.RandomState(seed)
        d = dag.shape[0]
        W = np.zeros((d, d))
        for i, j in zip(*np.where(dag > 0)):
            W[i, j] = rng.choice([-1, 1]) * rng.uniform(0.5, 2.0)
        noise_std = rng.uniform(0.5, 1.5, size=d)
        return CausalEnv(dag, W, noise_std, seed=seed + 1)
