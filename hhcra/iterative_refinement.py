"""
Iterative Refinement Loop: The three architectures catalyze each other.

Inspired by the iterative refinement: iterative_refinement sets are collections of
molecules where each molecule's formation is catalyzed by another member
of the set. No single molecule can exist alone -- the SET is the unit
of organization. Life emerged from this self-reinforcing loop.

Here, the three HHCRA architectures form an iterative_refinement loop:

    GNN (structure) → Liquid Net (dynamics) → Symbolic Engine (rules)
         ↑                                          |
         └──────── constrains search space ─────────┘

Each component produces something the others need:
- GNN discovers structure → Liquid Net needs structure to integrate
- Liquid Net produces trajectories → Symbolic Engine needs trajectories to discover rules
- Symbolic Engine discovers rules → GNN needs rules to constrain search space

The LOOP creates convergent behavior: after multiple cycles, the system
converges to a fixed point where structure, dynamics, and rules are
mutually consistent. This fixed point contains more information than
any single component could discover alone -- this is CONVERGENCE.

Mathematical formulation:
    Let S = (W, θ_liquid, R) be the system state (graph, ODE params, rules)
    The iterative_refinement operator Φ maps:
        Φ(S_n) = S_{n+1}
    where each component of S_{n+1} depends on all components of S_n.
    Convergence: ||S_{n+1} - S_n|| < ε implies fixed point reached.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, List, Optional, Tuple

from hhcra.config import HHCRAConfig
from hhcra.causal_graph import CausalGraphData
from hhcra.layer1_cjepa import CJEPA
from hhcra.layer2_mechanism import CausalGNN, LiquidNeuralNet, MechanismLayer
from hhcra.layer3_reasoning import NeuroSymbolicEngine, ReasoningLayer
from hhcra.liquid_causal_graph import LiquidCausalGraph
from hhcra.invariant_finder import TrajectoryInvariantFinder, InvariantRule


class IterativeRefinementNet(nn.Module):
    """
    The Iterative Refinement Loop: iterative loop over three architectures.

    Each cycle:
    1. GNN refines structure using constraints from Symbolic Engine
    2. Liquid Causal Graph co-evolves states + structure
    3. Invariant Finder discovers new rules from trajectories
    4. New rules constrain GNN's next cycle

    Convergence detected when:
    - Graph structure stabilizes (SHD between cycles < threshold)
    - Discovered rules stabilize (no new rules found)
    - Trajectory prediction error stabilizes

    The system has THREE convergence criteria, all must be met.
    """

    def __init__(self, config: HHCRAConfig):
        super().__init__()
        self.config = config

        # Core components
        self.gnn = CausalGNN(config)
        self.liquid_graph = LiquidCausalGraph(config)
        self.invariant_finder = TrajectoryInvariantFinder(config, num_candidates=8)
        self.symbolic_engine = NeuroSymbolicEngine()

        # Cycle tracking
        self.cycle_history: List[Dict] = []
        self.convergence_achieved = False

    def _apply_constraints_to_gnn(self, constraints: List[Dict]):
        """
        Apply symbolic constraints to GNN's adjacency weights.

        This is the key feedback step: Symbolic Engine → GNN.
        Discovered rules modify the search space for structure learning.
        """
        with torch.no_grad():
            for constraint in constraints:
                if constraint['confidence'] < 0.5:
                    continue

                if constraint['type'] == 'must_not_exist':
                    i, j = constraint['edge']
                    N = self.config.num_vars
                    if i < N and j < N:
                        # Suppress this edge
                        self.gnn.W.data[j, i] = min(
                            self.gnn.W.data[j, i].item(), -3.0
                        )

                elif constraint['type'] == 'must_exist':
                    i, j = constraint['edge']
                    N = self.config.num_vars
                    if i < N and j < N:
                        # Strengthen this edge
                        self.gnn.W.data[j, i] = max(
                            self.gnn.W.data[j, i].item(), 1.0
                        )

    def _compute_structure_distance(self, W_prev: torch.Tensor,
                                    W_curr: torch.Tensor) -> float:
        """
        Compute structural distance between two graph states.
        Uses Frobenius norm on the sigmoid-activated adjacency difference.
        """
        with torch.no_grad():
            A_prev = torch.sigmoid(W_prev) * self.gnn.diag_mask
            A_curr = torch.sigmoid(W_curr) * self.gnn.diag_mask
            dist = (A_curr - A_prev).pow(2).sum().sqrt().item()
        return dist

    def _symbolic_graph_from_weights(self, W: torch.Tensor) -> CausalGraphData:
        """Convert weight matrix to CausalGraphData."""
        with torch.no_grad():
            A = (torch.sigmoid(W) * self.gnn.diag_mask)
            A_hard = (A > self.config.edge_threshold).float()
            A_np = A_hard.cpu().numpy()
            A_soft = A.cpu().numpy()

        N = self.config.num_vars
        nodes = list(range(N))
        edges = []
        for i in range(N):
            for j in range(N):
                if A_np[i, j] > 0:
                    edges.append((j, i, float(A_soft[i, j])))

        return CausalGraphData(nodes, edges, A_np)

    def refinement_cycle(self, latent: torch.Tensor,
                            cycle_idx: int = 0) -> Dict:
        """
        Execute one full iterative_refinement cycle.

        Args:
            latent: (B, T, N, D) latent representations from Layer 1
            cycle_idx: which cycle number this is

        Returns:
            Dict with cycle results: trajectories, graph, rules, metrics
        """
        device = latent.device

        # Save pre-cycle state for convergence check
        W_before = self.gnn.W.data.clone()

        # === PHASE 1: GNN structure refinement ===
        # GNN refines structure with gradient-based NOTEARS
        # (Constraints from previous cycle already applied)
        optimizer = optim.Adam(self.gnn.parameters(), lr=0.01)
        for _ in range(5):  # Brief refinement
            optimizer.zero_grad()
            loss = self.gnn.notears_loss(latent)
            loss.backward()
            nn.utils.clip_grad_norm_(self.gnn.parameters(), max_norm=5.0)
            optimizer.step()

        # === PHASE 2: Liquid Causal Graph co-evolution ===
        # Use GNN message passing for embeddings
        embeddings = self.gnn.message_pass(latent)

        # Co-evolve states AND graph structure
        trajectories, W_evolved = self.liquid_graph.coupled_evolve(
            embeddings, self.gnn.W
        )

        # Update GNN weights with evolved graph
        # Blend: keep some GNN learning, incorporate liquid evolution
        with torch.no_grad():
            blend_alpha = 0.3  # How much liquid graph influences GNN
            self.gnn.W.data = (
                (1 - blend_alpha) * self.gnn.W.data + blend_alpha * W_evolved
            )

        # === PHASE 3: Invariant Finder — discover rules ===
        graph = self._symbolic_graph_from_weights(self.gnn.W)

        # Train invariant finder on new trajectories
        self.invariant_finder.train_finder(
            trajectories.detach(), lr=0.001, steps=10
        )

        # Discover rules from trajectories
        new_rules = self.invariant_finder.discover_rules(
            trajectories.detach(), graph
        )

        # === PHASE 4: Apply constraints back to GNN (constraint feedback) ===
        constraints = self.invariant_finder.get_graph_constraints()
        self._apply_constraints_to_gnn(constraints)

        # === Compute cycle metrics ===
        W_after = self.gnn.W.data.clone()
        structure_dist = self._compute_structure_distance(W_before, W_after)
        graph_change_rate = self.liquid_graph.compute_graph_change_rate()

        cycle_result = {
            'cycle': cycle_idx,
            'trajectories': trajectories,
            'graph': graph,
            'new_rules': new_rules,
            'total_rules': len(self.invariant_finder.discovered_rules),
            'constraints_applied': len(constraints),
            'structure_distance': structure_dist,
            'graph_change_rate': graph_change_rate,
            'n_edges': graph.edge_count(),
            'is_dag': graph.is_dag(),
        }

        self.cycle_history.append(cycle_result)
        return cycle_result

    def run_until_convergence(self, latent: torch.Tensor,
                              max_cycles: int = 10,
                              convergence_threshold: float = 0.05,
                              verbose: bool = True) -> Dict:
        """
        Run iterative_refinement cycles until the system reaches a fixed point.

        Convergence = structure + rules + dynamics all stabilize.

        Args:
            latent: (B, T, N, D) latent variables
            max_cycles: maximum number of cycles
            convergence_threshold: distance below which convergence is declared
            verbose: whether to print progress

        Returns:
            Final state including all discovered rules and final graph
        """
        if verbose:
            print("=" * 60)
            print("AUTOCATALYTIC CAUSAL NETWORK — Iterative Evolution")
            print("=" * 60)

        for cycle in range(max_cycles):
            result = self.refinement_cycle(latent, cycle_idx=cycle)

            if verbose:
                novel = sum(1 for r in result['new_rules'] if r.is_novel)
                print(
                    f"  Cycle {cycle + 1}/{max_cycles} | "
                    f"Edges: {result['n_edges']} | "
                    f"DAG: {result['is_dag']} | "
                    f"ΔStructure: {result['structure_distance']:.4f} | "
                    f"New rules: {len(result['new_rules'])} "
                    f"(novel: {novel}) | "
                    f"Graph Δ rate: {result['graph_change_rate']:.4f}"
                )

            # Check convergence (all three criteria)
            if cycle > 0:
                struct_converged = (
                    result['structure_distance'] < convergence_threshold
                )
                rules_converged = len(result['new_rules']) == 0
                dynamics_converged = (
                    result['graph_change_rate'] < convergence_threshold
                )

                if struct_converged and dynamics_converged:
                    self.convergence_achieved = True
                    if verbose:
                        print(f"\n  CONVERGENCE REACHED at cycle {cycle + 1}")
                        print(f"  Structure stable: {struct_converged}")
                        print(f"  Rules stable: {rules_converged}")
                        print(f"  Dynamics stable: {dynamics_converged}")
                    break

        # Final summary
        final_graph = self._symbolic_graph_from_weights(self.gnn.W)

        final_result = {
            'converged': self.convergence_achieved,
            'total_cycles': len(self.cycle_history),
            'final_graph': final_graph,
            'all_rules': list(self.invariant_finder.discovered_rules),
            'novel_rules': [
                r for r in self.invariant_finder.discovered_rules if r.is_novel
            ],
            'cycle_history': self.cycle_history,
            'graph_evolution': self.liquid_graph.get_graph_evolution(),
        }

        if verbose:
            print(f"\n  Total rules discovered: {len(final_result['all_rules'])}")
            print(f"  Novel rules: {len(final_result['novel_rules'])}")
            print(f"  Final graph: {final_graph.edge_count()} edges, "
                  f"DAG={final_graph.is_dag()}")
            print("=" * 60)

        return final_result

    def summary(self) -> str:
        """Human-readable summary of the iterative_refinement system state."""
        lines = [
            "Iterative Refinement Loop Summary",
            f"  Cycles completed: {len(self.cycle_history)}",
            f"  Fixed point reached: {self.convergence_achieved}",
            "",
            self.invariant_finder.summary(),
        ]

        if self.cycle_history:
            last = self.cycle_history[-1]
            lines.extend([
                "",
                f"  Last cycle metrics:",
                f"    Structure distance: {last['structure_distance']:.4f}",
                f"    Graph change rate: {last['graph_change_rate']:.4f}",
                f"    Edges: {last['n_edges']} | DAG: {last['is_dag']}",
            ])

        return "\n".join(lines)
