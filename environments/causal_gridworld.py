"""
CausalGridWorld: Grid with objects that have causal relationships.

Agent must figure out causal relationships between objects to solve
the grid efficiently. E.g., pushing A causes B to move.
"""

import numpy as np
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass

from hhcra.causal_graph import CausalGraphData


@dataclass
class GridObject:
    """An object on the grid with position and properties."""
    name: str
    x: int
    y: int
    movable: bool = True


class CausalGridWorld:
    """
    Grid world where objects have hidden causal relationships.

    Objects interact causally: pushing object A may move object B.
    Agent must discover these relationships to reach the goal efficiently.
    """

    def __init__(self, width: int = 6, height: int = 6,
                 num_objects: int = 4, seed: int = 42):
        self.width = width
        self.height = height
        self.num_objects = num_objects
        self.rng = np.random.RandomState(seed)

        # Actions: 0=up, 1=down, 2=left, 3=right, 4=push
        self.num_actions = 5

        # Causal links: pushing obj i also moves obj j
        self.causal_links: List[Tuple[int, int, int, int]] = []
        # (source_obj, target_obj, dx, dy)

        self.objects: List[GridObject] = []
        self.agent_x = 0
        self.agent_y = 0
        self.goal_x = width - 1
        self.goal_y = height - 1
        self.steps = 0
        self.max_steps = 100

        self._setup()

    def _setup(self):
        """Set up objects and causal links."""
        # Place objects
        positions = set()
        positions.add((0, 0))  # Agent start
        positions.add((self.width - 1, self.height - 1))  # Goal

        for i in range(self.num_objects):
            while True:
                x = self.rng.randint(0, self.width)
                y = self.rng.randint(0, self.height)
                if (x, y) not in positions:
                    positions.add((x, y))
                    self.objects.append(GridObject(f"obj_{i}", x, y))
                    break

        # Create causal links (chain structure)
        for i in range(self.num_objects - 1):
            dx = self.rng.choice([-1, 0, 1])
            dy = self.rng.choice([-1, 0, 1])
            self.causal_links.append((i, i + 1, dx, dy))

    def reset(self) -> np.ndarray:
        """Reset and return observation."""
        self.objects = []
        self.causal_links = []
        self._setup()
        self.agent_x = 0
        self.agent_y = 0
        self.steps = 0
        return self._observe()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """Execute action, return (obs, reward, done, info)."""
        self.steps += 1

        # Move agent
        dx, dy = [(0, 1), (0, -1), (-1, 0), (1, 0), (0, 0)][action]
        new_x = max(0, min(self.width - 1, self.agent_x + dx))
        new_y = max(0, min(self.height - 1, self.agent_y + dy))

        # Check for push action
        if action == 4:
            self._push_nearby()
        else:
            # Check collision with objects
            blocked = False
            for obj in self.objects:
                if obj.x == new_x and obj.y == new_y and obj.movable:
                    # Push the object
                    self._push_object(obj, dx, dy)
                    blocked = False  # Can push through
            if not blocked:
                self.agent_x = new_x
                self.agent_y = new_y

        # Reward
        at_goal = (self.agent_x == self.goal_x and self.agent_y == self.goal_y)
        reward = 1.0 if at_goal else -0.01  # Step penalty
        done = at_goal or self.steps >= self.max_steps

        info = {
            'at_goal': at_goal,
            'steps': self.steps,
        }

        return self._observe(), reward, done, info

    def _push_object(self, obj: GridObject, dx: int, dy: int):
        """Push object and propagate causal effects."""
        if not obj.movable:
            return

        obj_idx = self.objects.index(obj)
        obj.x = max(0, min(self.width - 1, obj.x + dx))
        obj.y = max(0, min(self.height - 1, obj.y + dy))

        # Propagate causal effects
        for src, tgt, cdx, cdy in self.causal_links:
            if src == obj_idx:
                target_obj = self.objects[tgt]
                if target_obj.movable:
                    target_obj.x = max(0, min(self.width - 1, target_obj.x + cdx))
                    target_obj.y = max(0, min(self.height - 1, target_obj.y + cdy))

    def _push_nearby(self):
        """Push nearest object."""
        for obj in self.objects:
            dist = abs(obj.x - self.agent_x) + abs(obj.y - self.agent_y)
            if dist <= 1 and obj.movable:
                dx = np.sign(obj.x - self.agent_x)
                dy = np.sign(obj.y - self.agent_y)
                if dx == 0 and dy == 0:
                    dx = 1  # Push right by default
                self._push_object(obj, int(dx), int(dy))
                break

    def _observe(self) -> np.ndarray:
        """Return observation vector."""
        obs = [self.agent_x / self.width, self.agent_y / self.height]
        for obj in self.objects:
            obs.extend([obj.x / self.width, obj.y / self.height])
        obs.extend([self.goal_x / self.width, self.goal_y / self.height])
        return np.array(obs, dtype=np.float32)

    def intervene(self, obj_idx: int, x: int, y: int) -> np.ndarray:
        """Intervene: set object position and observe causal effects."""
        if 0 <= obj_idx < len(self.objects):
            old_x, old_y = self.objects[obj_idx].x, self.objects[obj_idx].y
            dx = x - old_x
            dy = y - old_y
            self.objects[obj_idx].x = x
            self.objects[obj_idx].y = y
            # Propagate causal effects
            for src, tgt, cdx, cdy in self.causal_links:
                if src == obj_idx:
                    self.objects[tgt].x = max(0, min(
                        self.width - 1, self.objects[tgt].x + cdx))
                    self.objects[tgt].y = max(0, min(
                        self.height - 1, self.objects[tgt].y + cdy))
        return self._observe()

    def get_observation_dim(self) -> int:
        return 2 + 2 * self.num_objects + 2  # agent + objects + goal

    def get_true_graph(self) -> CausalGraphData:
        """Return true causal graph of object relationships."""
        N = self.num_objects
        adj = np.zeros((N, N))
        edges = []
        for src, tgt, dx, dy in self.causal_links:
            adj[tgt, src] = 1.0
            edges.append((src, tgt, 1.0))
        return CausalGraphData(list(range(N)), edges, adj)
