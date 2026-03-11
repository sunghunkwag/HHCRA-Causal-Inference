"""Simple physics environment for world model grounding."""

import numpy as np
from typing import Tuple, Optional

from hhcra.causal_graph import CausalGraphData


class SimplePhysicsEnv:
    """
    Simple physics with objects connected by causal relationships.
    Objects have positions and velocities, connected by spring-like forces.
    """

    def __init__(self, num_objects: int = 4, seed: int = 42):
        self.num_objects = num_objects
        self.rng = np.random.RandomState(seed)
        self.dt = 0.1
        self.damping = 0.98
        self.gravity = -0.1

        self.connections = []
        for i in range(num_objects - 1):
            self.connections.append((i, i + 1, 0.5))

        self.positions = np.zeros((num_objects, 2))
        self.velocities = np.zeros((num_objects, 2))
        self.reset()

    def reset(self) -> np.ndarray:
        self.positions = self.rng.randn(self.num_objects, 2) * 0.5
        self.velocities = np.zeros((self.num_objects, 2))
        return self._observe()

    def step(self, action: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float, bool, dict]:
        forces = np.zeros_like(self.velocities)
        forces[:, 1] += self.gravity

        for i, j, k in self.connections:
            diff = self.positions[j] - self.positions[i]
            dist = np.linalg.norm(diff) + 1e-8
            force = k * diff / dist
            forces[i] += force
            forces[j] -= force

        if action is not None and len(action) >= 2:
            forces[0] += action[:2]

        self.velocities += forces * self.dt
        self.velocities *= self.damping
        self.positions += self.velocities * self.dt

        return self._observe(), 0.0, False, {}

    def intervene(self, var_idx: int, value: float) -> np.ndarray:
        saved = (self.positions.copy(), self.velocities.copy())
        self.positions[var_idx, 0] = value
        self.velocities[var_idx] = 0.0
        for _ in range(10):
            self.step()
            self.positions[var_idx, 0] = value
            self.velocities[var_idx] = 0.0
        obs = self._observe()
        self.positions, self.velocities = saved
        return obs

    def _observe(self) -> np.ndarray:
        return np.concatenate([self.positions.flatten(), self.velocities.flatten()])

    def get_true_graph(self) -> CausalGraphData:
        N = self.num_objects
        adj = np.zeros((N, N))
        edges = []
        for i, j, k in self.connections:
            adj[j, i] = 1.0
            edges.append((i, j, float(k)))
        return CausalGraphData(list(range(N)), edges, adj)
