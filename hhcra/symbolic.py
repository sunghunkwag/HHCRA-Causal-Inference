"""
NeuroSymbolicEngine: Causal reasoning via Pearl's framework.

Ported from original HHCRA Layer 3 — the one component that worked.
Accepts CausalGraphData objects. Pure Python/numpy.
"""
from __future__ import annotations
import numpy as np
from typing import Optional, Set, Dict, Any
from itertools import combinations
from collections import deque
from hhcra.graph import CausalGraphData


class NeuroSymbolicEngine:
    """Pearl's causal inference: d-sep, backdoor, frontdoor, do-calculus, ABP."""

    def d_separated(self, G: CausalGraphData, X: int, Y: int, Z: Set[int]) -> bool:
        """Test X ⊥ Y | Z via Bayes-Ball."""
        visited = set()
        queue = deque([(X, 'up')])
        while queue:
            node, direction = queue.popleft()
            if node == Y:
                return False
            if (node, direction) in visited:
                continue
            visited.add((node, direction))
            if direction == 'up':
                if node not in Z:
                    for p in G.parents(node):
                        queue.append((p, 'up'))
                    for c in G.children(node):
                        queue.append((c, 'down'))
            else:
                if node not in Z:
                    for c in G.children(node):
                        queue.append((c, 'down'))
                if node in Z:
                    for p in G.parents(node):
                        queue.append((p, 'up'))
        return True

    def do_calc_rule1(self, G: CausalGraphData, Y: int, X: int,
                      Z: Set[int], W: Set[int]) -> bool:
        G_mut = self._remove_incoming(G, {X})
        z_node = next(iter(Z)) if Z else Y
        return self.d_separated(G_mut, Y, z_node, W | {X})

    def do_calc_rule2(self, G: CausalGraphData, Y: int, X: int,
                      Z: Set[int], W: Set[int]) -> bool:
        G_mut = self._remove_incoming(G, {X})
        G_mut = self._remove_outgoing(G_mut, Z)
        z_node = next(iter(Z)) if Z else Y
        return self.d_separated(G_mut, Y, z_node, W | {X})

    def do_calc_rule3(self, G: CausalGraphData, Y: int, X: int,
                      Z: Set[int], W: Set[int]) -> bool:
        G_x_bar = self._remove_incoming(G, {X})
        z_s = set()
        for z in Z:
            w_ancestors = set()
            for w in W:
                w_ancestors |= G_x_bar.ancestors(w)
            if z in w_ancestors or z in W:
                z_s.add(z)
        G_mut = self._remove_incoming(G_x_bar, z_s)
        z_node = next(iter(Z)) if Z else Y
        return self.d_separated(G_mut, Y, z_node, W | {X})

    def _remove_incoming(self, G: CausalGraphData, nodes: Set[int]) -> CausalGraphData:
        new_edges = [(p, c, w) for p, c, w in G.edges if c not in nodes]
        new_adj = G.adjacency.copy()
        for n in nodes:
            new_adj[:, n] = 0.0
        return CausalGraphData(G.nodes[:], new_edges, new_adj)

    def _remove_outgoing(self, G: CausalGraphData, nodes: Set[int]) -> CausalGraphData:
        new_edges = [(p, c, w) for p, c, w in G.edges if p not in nodes]
        new_adj = G.adjacency.copy()
        for n in nodes:
            new_adj[n, :] = 0.0
        return CausalGraphData(G.nodes[:], new_edges, new_adj)

    def find_backdoor_set(self, G: CausalGraphData, X: int, Y: int) -> Optional[Set[int]]:
        desc_X = G.descendants(X)
        candidates = set(range(len(G.nodes))) - {X, Y} - desc_X
        for size in range(len(candidates) + 1):
            for subset in combinations(sorted(candidates), size):
                Z = set(subset)
                if self._blocks_backdoor(G, X, Y, Z):
                    return Z
        return None

    def find_frontdoor_set(self, G: CausalGraphData, X: int, Y: int) -> Optional[Set[int]]:
        paths = self._directed_paths(G, X, Y)
        if not paths:
            return None
        mediators = set()
        for path in paths:
            mediators.update(path[1:-1])
        if not mediators:
            return None
        for m in mediators:
            if not self._blocks_backdoor(G, m, Y, {X}):
                return None
        return mediators

    def check_identifiability(self, G: CausalGraphData, X: int, Y: int) -> Dict:
        bd = self.find_backdoor_set(G, X, Y)
        if bd is not None:
            return {'identifiable': True, 'strategy': 'backdoor', 'adjustment_set': bd}
        fd = self.find_frontdoor_set(G, X, Y)
        if fd is not None:
            return {'identifiable': True, 'strategy': 'frontdoor', 'adjustment_set': fd}
        paths = self._directed_paths(G, X, Y)
        if not paths:
            return {'identifiable': True, 'strategy': 'no_causal_path', 'adjustment_set': set()}
        return {'identifiable': False, 'strategy': None, 'adjustment_set': None}

    def _blocks_backdoor(self, G, X, Y, Z):
        adj_mut = G.adjacency.copy()
        adj_mut[X, :] = 0
        G_mut = CausalGraphData(G.nodes[:],
                                [(p, c, w) for p, c, w in G.edges if p != X],
                                adj_mut)
        return self.d_separated(G_mut, X, Y, Z)

    def _directed_paths(self, G, start, end, max_depth=15):
        paths = []
        queue = deque([[start]])
        while queue:
            path = queue.popleft()
            node = path[-1]
            if len(path) > max_depth:
                continue
            if node == end and len(path) > 1:
                paths.append(path)
                continue
            for child in G.children(node):
                if child not in path:
                    queue.append(path + [child])
        return paths
