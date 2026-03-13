"""
CausalGraphData and CausalQueryType definitions.

These are the core data structures shared between layers.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Tuple, Set, Optional

from collections import deque

import numpy as np
import torch


class CausalQueryType(Enum):
    """Three rungs of Pearl's Ladder."""
    OBSERVATIONAL = "P(Y|X)"
    INTERVENTIONAL = "P(Y|do(X))"
    COUNTERFACTUAL = "P(Y_x'|X=x,Y=y)"


@dataclass
class CausalGraphData:
    """Symbolic representation of a directed causal graph."""
    nodes: List[int]
    edges: List[Tuple[int, int, float]]  # (parent, child, weight)
    adjacency: np.ndarray  # adj[child, parent] = 1 means parent->child

    def parents(self, n: int) -> Set[int]:
        return {p for p, c, _ in self.edges if c == n}

    def children(self, n: int) -> Set[int]:
        return {c for p, c, _ in self.edges if p == n}

    def ancestors(self, n: int) -> Set[int]:
        result = set()
        queue = deque(self.parents(n))
        while queue:
            x = queue.popleft()
            if x not in result:
                result.add(x)
                queue.extend(self.parents(x))
        return result

    def descendants(self, n: int) -> Set[int]:
        result = set()
        queue = deque(self.children(n))
        while queue:
            x = queue.popleft()
            if x not in result:
                result.add(x)
                queue.extend(self.children(x))
        return result

    def has_edge(self, parent: int, child: int) -> bool:
        return any(p == parent and c == child for p, c, _ in self.edges)

    def edge_count(self) -> int:
        return len(self.edges)

    def is_dag(self) -> bool:
        """Verify acyclicity via topological sort (Kahn's algorithm)."""
        N = len(self.nodes)
        in_deg = {n: 0 for n in self.nodes}
        for p, c, _ in self.edges:
            in_deg[c] += 1
        queue = deque(n for n in self.nodes if in_deg[n] == 0)
        visited = 0
        while queue:
            node = queue.popleft()
            visited += 1
            for ch in self.children(node):
                in_deg[ch] -= 1
                if in_deg[ch] == 0:
                    queue.append(ch)
        return visited == N

    def to_torch_adjacency(self, device: str = "cpu") -> torch.Tensor:
        """Convert adjacency to torch tensor."""
        return torch.tensor(self.adjacency, dtype=torch.float32, device=device)
