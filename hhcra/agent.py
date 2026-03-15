"""Active Causal Discovery Agent. Pure numpy."""
from __future__ import annotations
import numpy as np
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass
from hhcra.architecture import HHCRA
from hhcra.graph import CausalQueryType

@dataclass
class AgentState:
    step: int = 0
    total_reward: float = 0.0
    graph_known: bool = False
    n_interventions: int = 0

class ActiveCausalAgent:
    """Agent that discovers causal structure via observation + targeted intervention."""

    def __init__(self, n_vars: int, obs_buffer_size: int = 500):
        self.n_vars = n_vars
        self.obs_buffer_size = obs_buffer_size
        self.obs_buffer: List[np.ndarray] = []
        self.model: Optional[HHCRA] = None
        self.edge_alpha = np.ones((n_vars, n_vars))
        self.edge_beta = np.ones((n_vars, n_vars))
        np.fill_diagonal(self.edge_alpha, 0)
        np.fill_diagonal(self.edge_beta, 0)
        self.state = AgentState()

    @property
    def edge_probabilities(self) -> np.ndarray:
        total = self.edge_alpha + self.edge_beta
        return self.edge_alpha / np.maximum(total, 1e-10)

    @property
    def edge_uncertainty(self) -> np.ndarray:
        a, b = self.edge_alpha, self.edge_beta
        total = a + b
        return (a * b) / (total ** 2 * (total + 1) + 1e-10)

    def observe(self, observation: np.ndarray):
        self.obs_buffer.append(observation)
        if len(self.obs_buffer) > self.obs_buffer_size:
            self.obs_buffer.pop(0)
        self.state.step += 1

    def update_model(self, verbose: bool = False):
        if len(self.obs_buffer) < 20:
            return
        X = np.array(self.obs_buffer)
        self.model = HHCRA(n_vars_hint=self.n_vars)
        self.model.fit(X, verbose=verbose)
        self.state.graph_known = True
        if self.model.adj is not None:
            for i in range(self.n_vars):
                for j in range(self.n_vars):
                    if i == j: continue
                    if self.model.adj[i, j] > 0:
                        self.edge_alpha[i, j] += 1.0
                    else:
                        self.edge_beta[i, j] += 1.0

    def select_intervention(self) -> Tuple[int, float]:
        """Pick node with highest adjacent edge uncertainty."""
        unc = self.edge_uncertainty
        scores = unc.sum(axis=1) + unc.sum(axis=0)
        best = int(np.argmax(scores))
        val = 2.0
        if self.model and self.model._var_data is not None:
            val = 2.0 * max(np.std(self.model._var_data[:, best]), 0.5)
        return best, val

    def process_intervention(self, target: int, value: float,
                             obs_data: np.ndarray, int_data: np.ndarray):
        from scipy import stats
        self.state.n_interventions += 1
        d = obs_data.shape[1]
        for j in range(d):
            if j == target: continue
            _, p = stats.ttest_ind(obs_data[:, j], int_data[:, j], equal_var=False)
            if p < 0.01:
                self.edge_alpha[target, j] += 2.0
            else:
                self.edge_beta[target, j] += 2.0

    def get_stats(self) -> Dict:
        return {
            'step': self.state.step,
            'n_interventions': self.state.n_interventions,
            'graph_known': self.state.graph_known,
            'mean_uncertainty': float(self.edge_uncertainty.mean()),
            'n_observations': len(self.obs_buffer),
        }
