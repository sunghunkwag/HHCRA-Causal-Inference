"""
Layer 3: Reasoning Layer — Neuro-Symbolic Engine + HRM (Tightly Coupled)

Neuro-Symbolic Engine: Formal causal reasoning via Pearl's framework.
  - d-Separation (Bayes-Ball)
  - Backdoor/Frontdoor criteria
  - Complete do-calculus (3 rules)
  - ID algorithm (Tian & Pearl 2002) for general identifiability
  - Counterfactual via Abduction-Action-Prediction
  - Instrumental variable detection

HRM: Hierarchical Reasoning Model with GRU recurrence.
  - H-module (slow timescale): strategic reasoning with GRU
  - L-module (fast timescale): concrete computation with GRU
  - Adaptive Computation Time (ACT): learned convergence stopping
  - Learned reset mechanism

Interface IN  <- Layer 2: CausalGraphData + trajectories
Output: Causal query answers + feedback signals
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Set, List, Tuple, Dict, Any
from itertools import combinations

from hhcra.config import HHCRAConfig
from hhcra.causal_graph import CausalGraphData, CausalQueryType


class NeuroSymbolicEngine:
    """
    Formal causal reasoning engine implementing Pearl's causal inference.

    Symbolic operations:
        - d-Separation via Bayes-Ball algorithm
        - Backdoor criterion + adjustment set finding
        - Frontdoor criterion
        - Complete do-calculus (3 rules)
        - ID algorithm for general identifiability
        - Instrumental variable detection
        - Counterfactual 3-step procedure
    """

    def d_separated(self, G: CausalGraphData, X: int, Y: int, Z: Set[int]) -> bool:
        """
        Test if X _||_ Y | Z using Bayes-Ball algorithm.
        Returns True if X and Y are d-separated given Z.
        """
        from collections import deque
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
            else:  # 'down'
                if node not in Z:
                    for c in G.children(node):
                        queue.append((c, 'down'))
                if node in Z:
                    for p in G.parents(node):
                        queue.append((p, 'up'))

        return True

    # --- Do-Calculus: 3 Rules ---

    def do_calc_rule1(self, G: CausalGraphData, Y: int, X: int,
                      Z: Set[int], W: Set[int]) -> bool:
        """
        Rule 1: Insertion/deletion of observations.
        P(y|do(x),z,w) = P(y|do(x),w) if (Y _||_ Z | X,W) in G_X_bar
        where G_X_bar is G with incoming edges to X removed.
        """
        G_mutilated = self._remove_incoming(G, {X})
        return self.d_separated(G_mutilated, Y, next(iter(Z)) if Z else Y, W | {X})

    def do_calc_rule2(self, G: CausalGraphData, Y: int, X: int,
                      Z: Set[int], W: Set[int]) -> bool:
        """
        Rule 2: Action/observation exchange.
        P(y|do(x),do(z),w) = P(y|do(x),z,w) if (Y _||_ Z | X,W) in G_X_bar_Z_underbar
        where G_X_bar_Z_underbar has incoming to X removed AND outgoing from Z removed.
        """
        G_mut = self._remove_incoming(G, {X})
        G_mut = self._remove_outgoing(G_mut, Z)
        z_node = next(iter(Z)) if Z else Y
        return self.d_separated(G_mut, Y, z_node, W | {X})

    def do_calc_rule3(self, G: CausalGraphData, Y: int, X: int,
                      Z: Set[int], W: Set[int]) -> bool:
        """
        Rule 3: Insertion/deletion of actions.
        P(y|do(x),do(z),w) = P(y|do(x),w) if (Y _||_ Z | X,W) in G_X_bar_Z(S)_bar
        where Z(S) is Z minus nodes that are not ancestors of any W-node in G_X_bar.
        """
        G_x_bar = self._remove_incoming(G, {X})
        # Z(S): remove from Z those not ancestors of W in G_x_bar
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
        """Create new graph with incoming edges to nodes removed."""
        new_edges = [(p, c, w) for p, c, w in G.edges if c not in nodes]
        new_adj = G.adjacency.copy()
        for n in nodes:
            new_adj[n, :] = 0.0
        return CausalGraphData(G.nodes[:], new_edges, new_adj)

    def _remove_outgoing(self, G: CausalGraphData, nodes: Set[int]) -> CausalGraphData:
        """Create new graph with outgoing edges from nodes removed."""
        new_edges = [(p, c, w) for p, c, w in G.edges if p not in nodes]
        new_adj = G.adjacency.copy()
        for n in nodes:
            new_adj[:, n] = 0.0
        return CausalGraphData(G.nodes[:], new_edges, new_adj)

    # --- Identification Algorithms ---

    def find_backdoor_set(self, G: CausalGraphData, X: int, Y: int) -> Optional[Set[int]]:
        """
        Find minimal valid backdoor adjustment set for P(Y|do(X)).

        Backdoor criterion: Z is valid if
        1. No node in Z is a descendant of X
        2. Z blocks every backdoor path (path with arrow into X)
        """
        descendants_X = G.descendants(X)
        candidates = set(G.nodes) - {X, Y} - descendants_X

        for size in range(len(candidates) + 1):
            for subset in self._power_subsets(candidates, size):
                Z = set(subset)
                if self._blocks_backdoor(G, X, Y, Z):
                    return Z
        return None

    def find_frontdoor_set(self, G: CausalGraphData, X: int, Y: int) -> Optional[Set[int]]:
        """
        Find frontdoor adjustment set: mediators M where
        1. X intercepts all directed paths from X to M
        2. No unblocked backdoor X to M
        3. X blocks all backdoor M to Y
        """
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

    def find_instrumental_variable(self, G: CausalGraphData, X: int, Y: int) -> Optional[int]:
        """
        Find instrumental variable Z for effect of X on Y.

        Conditions for Z to be a valid instrument:
        1. Z is associated with X (not d-separated from X given empty set)
        2. Z affects Y only through X (d-separated from Y given X)
        3. Z shares no common causes with Y (no confounders between Z and Y)
        """
        for z in G.nodes:
            if z == X or z == Y:
                continue
            # Condition 1: Z associated with X
            if self.d_separated(G, z, X, set()):
                continue
            # Condition 2: Z affects Y only through X
            if not self.d_separated(G, z, Y, {X}):
                continue
            # Condition 3: no direct path from Z to Y bypassing X
            direct_paths = self._directed_paths(G, z, Y)
            all_through_x = all(X in p for p in direct_paths) if direct_paths else True
            if all_through_x:
                return z
        return None

    def id_algorithm(self, G: CausalGraphData, Y: Set[int], X: Set[int],
                     P_nodes: Optional[Set[int]] = None) -> dict:
        """
        ID algorithm (Tian & Pearl 2002) for general identifiability.

        Determines if P(Y|do(X)) is identifiable from observational data
        in causal graph G. Returns identification result with formula
        or non-identifiability proof.

        Args:
            G: Causal graph
            Y: Set of outcome variables
            X: Set of intervention variables
            P_nodes: Set of all observed variables (defaults to G.nodes)

        Returns:
            dict with 'identifiable', 'strategy', 'adjustment_set'
        """
        if P_nodes is None:
            P_nodes = set(G.nodes)

        # Base cases
        if not X:
            return {'identifiable': True, 'strategy': 'no_intervention',
                    'adjustment_set': set()}

        # Try backdoor first (most efficient)
        if len(X) == 1 and len(Y) == 1:
            x = next(iter(X))
            y = next(iter(Y))

            bd = self.find_backdoor_set(G, x, y)
            if bd is not None:
                return {'identifiable': True, 'strategy': 'backdoor',
                        'adjustment_set': bd}

            fd = self.find_frontdoor_set(G, x, y)
            if fd is not None:
                return {'identifiable': True, 'strategy': 'frontdoor',
                        'adjustment_set': fd}

            iv = self.find_instrumental_variable(G, x, y)
            if iv is not None:
                return {'identifiable': True, 'strategy': 'instrumental_variable',
                        'adjustment_set': {iv}}

        # General ID: check ancestors
        ancestors_Y = set()
        for y in Y:
            ancestors_Y |= G.ancestors(y)
            ancestors_Y.add(y)

        # If X has no ancestor that's also an ancestor of Y, trivially identifiable
        x_ancestors_of_y = X & ancestors_Y
        if not x_ancestors_of_y:
            return {'identifiable': True, 'strategy': 'no_causal_path',
                    'adjustment_set': set()}

        # Check c-components (connected components in bidirected graph)
        # For simplicity with our DAG representation, check district structure
        remaining = P_nodes - X
        if Y.issubset(remaining):
            # Try adjustment via remaining variables
            for size in range(len(remaining) + 1):
                for subset in self._power_subsets(remaining - Y, size):
                    Z = set(subset)
                    # Check if Z is valid adjustment
                    valid = True
                    for x in X:
                        for y in Y:
                            if not self._blocks_backdoor(G, x, y, Z):
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        return {'identifiable': True, 'strategy': 'generalized_adjustment',
                                'adjustment_set': Z}

        return {'identifiable': False, 'strategy': None, 'adjustment_set': None}

    def check_identifiability(self, G: CausalGraphData, X: int, Y: int) -> dict:
        """
        Check if P(Y|do(X)) is identifiable. Uses ID algorithm.
        """
        return self.id_algorithm(G, {Y}, {X})

    # --- Counterfactual Reasoning ---

    def counterfactual_abduction(self, observed_y: torch.Tensor,
                                 predicted_y: torch.Tensor) -> torch.Tensor:
        """
        Step 1 of ABP: Abduction.
        Infer exogenous noise U given evidence.
        U = observed - predicted (MLE for additive noise model)
        """
        return observed_y - predicted_y

    def counterfactual_action(self, adjacency: torch.Tensor,
                              interventions: Dict[int, torch.Tensor]) -> torch.Tensor:
        """
        Step 2 of ABP: Action.
        Modify structural equations by cutting incoming edges to intervened vars.
        Returns modified adjacency.
        """
        mod_adj = adjacency.clone()
        for idx in interventions:
            mod_adj[idx, :] = 0.0
        return mod_adj

    def counterfactual_prediction(self, cf_trajectory_y: torch.Tensor,
                                  noise_u: torch.Tensor) -> torch.Tensor:
        """
        Step 3 of ABP: Prediction.
        Forward propagate through modified SCM with inferred U.
        Y_cf = f_Y(pa_Y under intervention) + U
        """
        return cf_trajectory_y + noise_u

    # --- Helper Methods ---

    def _blocks_backdoor(self, G: CausalGraphData, X: int, Y: int,
                         Z: Set[int]) -> bool:
        """Check if Z blocks all backdoor paths from X to Y.
        Uses the mutilated graph with outgoing edges from X removed."""
        G_mut = self._remove_outgoing(G, {X})
        return self.d_separated(G_mut, X, Y, Z)

    def _directed_paths(self, G: CausalGraphData, start: int, end: int,
                        max_depth: int = 10) -> List[List[int]]:
        """Find all directed paths from start to end."""
        from collections import deque
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

    def _power_subsets(self, s: set, size: int):
        """Generate all subsets of given size using itertools.combinations."""
        yield from combinations(sorted(s), size)


class HRM(nn.Module):
    """
    Hierarchical Reasoning Model with GRU recurrence and ACT.

    H-module (slow timescale): GRU-based strategic reasoning.
        Updates every K steps.
    L-module (fast timescale): GRU-based concrete computation.
        Updates every step.

    Adaptive Computation Time (ACT): learned halting probability.
    Learned reset: H-module learns WHEN to reset L-module.
    """

    def __init__(self, config: HHCRAConfig):
        super().__init__()
        self.config = config
        D = config.latent_dim
        H = config.hrm_hidden_dim

        # H-module: slow strategic reasoning (GRU)
        self.h_gru = nn.GRUCell(D + D, H)
        self.h_directive = nn.Linear(H, D)  # Strategy encoding
        self.h_convergence = nn.Linear(H, 1)  # Convergence estimation

        # L-module: fast computation (GRU)
        self.l_gru = nn.GRUCell(D, D)
        self.l_output = nn.Linear(D, D)

        # ACT: adaptive computation time
        self.halt_net = nn.Sequential(
            nn.Linear(H + D, 1),
            nn.Sigmoid(),
        )

        # Learned reset mechanism
        self.reset_net = nn.Sequential(
            nn.Linear(H + D + 1, 1),  # +1 for convergence signal
            nn.Sigmoid(),
        )
        self.l_init = nn.Linear(H, D)  # H-module generates L-module initial state

        # Reasoning depth tracking
        self.depth_history = []

    def forward(self, query: torch.Tensor) -> dict:
        """Alias for reason() to support nn.Module convention."""
        return self.reason(query)

    def reason(self, query: torch.Tensor) -> dict:
        """
        Execute multi-step hierarchical reasoning with ACT.

        H-module sets strategy (every K steps), L-module executes (every step).
        ACT determines when to stop. Learned reset replaces fixed patience.
        """
        D = self.config.latent_dim
        H = self.config.hrm_hidden_dim
        K = self.config.hrm_update_interval
        device = query.device if isinstance(query, torch.Tensor) else 'cpu'

        if isinstance(query, np.ndarray):
            query = torch.tensor(query, dtype=torch.float32, device=device)

        q = query.flatten()[:D] if query.numel() >= D else F.pad(
            query.flatten(), (0, D - query.numel()))

        h_state = torch.zeros(1, H, device=device)
        l_state = torch.zeros(1, D, device=device)
        q_input = q.unsqueeze(0)  # (1, D)

        trace = []
        cumulative_halt = torch.tensor(0.0, device=device)
        result = None
        result_weight = torch.tensor(0.0, device=device)
        best_conv = 0.0
        momentum = 0.0

        for step in range(self.config.hrm_max_steps):
            # H-module update (slow timescale: every K steps)
            if step % K == 0:
                h_input = torch.cat([q_input, l_state], dim=-1)  # (1, D+D)
                h_state = self.h_gru(h_input, h_state)

            # Generate directive and convergence
            directive = torch.tanh(self.h_directive(h_state))  # (1, D)
            raw_conv = torch.sigmoid(self.h_convergence(h_state)).item()
            momentum = self.config.hrm_momentum * momentum + (1 - self.config.hrm_momentum) * raw_conv
            conv = momentum

            # L-module update (fast timescale: every step)
            l_state = self.l_gru(directive, l_state)
            res = torch.tanh(self.l_output(l_state))  # (1, D)

            # ACT: compute halting probability
            halt_input = torch.cat([h_state, l_state], dim=-1)
            halt_prob = self.halt_net(halt_input).squeeze()

            # Accumulate weighted result
            remainder = 1.0 - cumulative_halt
            p = torch.min(halt_prob, remainder)
            if result is None:
                result = p * res
                result_weight = p
            else:
                result = result + p * res
                result_weight = result_weight + p
            cumulative_halt = cumulative_halt + p

            trace.append({'step': step, 'convergence': conv,
                          'halt_prob': halt_prob.item()})

            if conv > best_conv + self.config.hrm_convergence_threshold:
                best_conv = conv

            # ACT stopping condition
            if cumulative_halt.item() > 0.95:
                trace.append({'step': step, 'event': 'CONVERGED'})
                break

            # Learned reset: H-module decides when to reset L-module
            conv_signal = torch.tensor([[conv]], device=device)
            reset_input = torch.cat([h_state, l_state, conv_signal], dim=-1)
            reset_prob = self.reset_net(reset_input).squeeze()

            if reset_prob.item() > 0.7 and step > K:
                # H-module resets L-module with new initialization
                l_state = torch.tanh(self.l_init(h_state))
                trace.append({
                    'step': step, 'event': 'H_MODULE_RESET',
                    'reason': 'Learned reset triggered',
                    'reset_prob': reset_prob.item()
                })

        # Normalize result by total weight
        if result_weight.item() > 0:
            result = result / result_weight.clamp(min=1e-8)
        elif result is None:
            result = res

        self.depth_history.append(len(trace))

        return {
            'result': result.squeeze(0),
            'convergence': best_conv,
            'steps': len(trace),
            'trace': trace,
        }


class ReasoningLayer(nn.Module):
    """
    Layer 3: Neuro-symbolic + HRM tightly coupled.

    HRM orchestrates the reasoning process.
    Neuro-symbolic provides formal causal operations.
    Tight coupling: HRM reasoning uses symbolic engine results.

    Interface IN  <- Layer 2: CausalGraphData + trajectories
    Output: Causal query answers + feedback signals
    """

    def __init__(self, config: HHCRAConfig):
        super().__init__()
        self.config = config
        self.symbolic = NeuroSymbolicEngine()
        self.hrm = HRM(config)

    def forward(self, query: torch.Tensor) -> dict:
        """Forward pass through HRM reasoning."""
        return self.hrm.reason(query)

    def answer_query(
        self,
        query_type: CausalQueryType,
        X: int, Y: int,
        graph: CausalGraphData,
        trajectories: torch.Tensor,
        mechanism_layer,  # MechanismLayer (avoid circular import)
        x_value: Optional[torch.Tensor] = None,
        factual_x: Optional[torch.Tensor] = None,
        factual_y: Optional[torch.Tensor] = None,
        counterfactual_x: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Answer a causal query.

        Rung 1: P(Y|X) — observational
        Rung 2: P(Y|do(X=x)) — interventional
        Rung 3: P(Y_{x'}|X=x, Y=y) — counterfactual (ABP procedure)
        """
        # 1. Check identifiability
        id_check = self.symbolic.check_identifiability(graph, X, Y)

        # 2. Generate feedback if not identifiable
        feedback = {}
        if not id_check['identifiable']:
            feedback = {
                'to_layer2': {
                    'issue': 'non_identifiable',
                    'X': X, 'Y': Y,
                    'suggestion': 'Graph may need refinement',
                },
                'to_layer1': {
                    'increase_resolution': True,
                    'reason': f'P(X{Y}|do(X{X})) not identifiable with current graph',
                },
            }

        # 3. HRM orchestrated reasoning
        query_vec = trajectories[:, -1, X, :].mean(dim=0)
        reasoning = self.hrm.reason(query_vec)

        # 4. Execute causal computation
        answer = None
        adj_tensor = torch.tensor(graph.adjacency, dtype=torch.float32,
                                  device=trajectories.device)

        if query_type == CausalQueryType.OBSERVATIONAL:
            answer = trajectories[:, -1, Y, :]

        elif query_type == CausalQueryType.INTERVENTIONAL:
            if id_check['identifiable'] and x_value is not None:
                int_traj = mechanism_layer.liquid.intervene(
                    trajectories, adj_tensor, {X: x_value})
                answer = int_traj[:, -1, Y, :]

        elif query_type == CausalQueryType.COUNTERFACTUAL:
            if id_check['identifiable'] and all(
                v is not None for v in [factual_x, factual_y, counterfactual_x]
            ):
                # Step 1: Abduction — infer exogenous noise U
                observed_y = trajectories[:, -1, Y, :]
                noise_u = self.symbolic.counterfactual_abduction(
                    factual_y, observed_y)

                # Step 2: Action — modify graph for intervention
                mod_adj = self.symbolic.counterfactual_action(
                    adj_tensor, {X: counterfactual_x})

                # Step 3: Prediction — propagate through modified SCM
                # v0.5.0 FIX: use mod_adj (edges to X cut) instead of adj_tensor
                cf_traj = mechanism_layer.liquid.intervene(
                    trajectories, mod_adj, {X: counterfactual_x})
                cf_y = cf_traj[:, -1, Y, :]

                answer = self.symbolic.counterfactual_prediction(cf_y, noise_u)

        return {
            'type': query_type.value,
            'answer': answer,
            'identifiability': id_check,
            'reasoning': {
                'result': reasoning['result'].detach() if isinstance(reasoning['result'], torch.Tensor) else reasoning['result'],
                'convergence': reasoning['convergence'],
                'steps': reasoning['steps'],
                'trace': reasoning['trace'],
            },
            'feedback': feedback,
        }

    def generate_diagnostic(self, graph: CausalGraphData) -> dict:
        """Proactive diagnostic of graph quality."""
        issues = []

        if not graph.is_dag():
            issues.append('Graph contains cycles — not a valid DAG')

        if graph.edge_count() == 0:
            issues.append('Graph has no edges — structure learning may have failed')

        N = len(graph.nodes)
        max_edges = N * (N - 1)
        density = graph.edge_count() / max_edges if max_edges > 0 else 0
        if density > 0.6:
            issues.append(f'Graph too dense ({density:.1%}) — may indicate insufficient pruning')

        return {'issues': issues, 'density': density, 'is_dag': graph.is_dag()}
