"""Causal graph data structures. No PyTorch dependency."""
from enum import Enum
from dataclasses import dataclass
from typing import List, Tuple, Set
from collections import deque
import numpy as np

class CausalQueryType(Enum):
    OBSERVATIONAL = "P(Y|X)"
    INTERVENTIONAL = "P(Y|do(X))"
    COUNTERFACTUAL = "P(Y_x'|X=x,Y=y)"

@dataclass
class CausalGraphData:
    nodes: List[int]
    edges: List[Tuple[int, int, float]]
    adjacency: np.ndarray

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

    def edge_count(self) -> int:
        return len(self.edges)

    def is_dag(self) -> bool:
        N = len(self.nodes)
        in_deg = np.zeros(N, dtype=int)
        for p, c, _ in self.edges:
            in_deg[c] += 1
        queue = deque(i for i in range(N) if in_deg[i] == 0)
        visited = 0
        while queue:
            node = queue.popleft()
            visited += 1
            for c in self.children(node):
                in_deg[c] -= 1
                if in_deg[c] == 0:
                    queue.append(c)
        return visited == N

    @staticmethod
    def from_adjacency(adj: np.ndarray) -> 'CausalGraphData':
        d = adj.shape[0]
        nodes = list(range(d))
        edges = [(i, j, float(adj[i, j])) for i in range(d) for j in range(d) if adj[i, j] > 0]
        return CausalGraphData(nodes=nodes, edges=edges, adjacency=adj.copy())
