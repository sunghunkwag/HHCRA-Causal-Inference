"""
HHCRA Experimental: Unified system integrating all three experimental extensions (unvalidated).

This is the orchestrator that combines:
1. Liquid Causal Graph — graph structure as a dynamical system
2. Trajectory Invariant Finder — automatic rule discovery from dynamics
3. Iterative Refinement Loop — iterative convergence loop

The Experimental system extends the base HHCRA architecture by replacing
the static pipeline (L1 → L2 → L3) with a iterative system where:
- The graph evolves continuously (not just during training)
- Rules are discovered (not hardcoded)
- All components catalyze each other's improvement

Usage:
    model = HHCRAExperimental(config)
    model.train_experimental(observations)  # Full pipeline with iterative refinement
    result = model.query(observations, ...)  # Enhanced causal queries

The existing HHCRA API is preserved for backward compatibility.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Optional, Dict

from hhcra.config import HHCRAConfig
from hhcra.causal_graph import CausalGraphData, CausalQueryType
from hhcra.layer1_cjepa import CJEPA
from hhcra.layer2_mechanism import MechanismLayer
from hhcra.layer3_reasoning import ReasoningLayer
from hhcra.interfaces import LayerInterface, FeedbackRouter
from hhcra.liquid_causal_graph import LiquidCausalGraph
from hhcra.invariant_finder import TrajectoryInvariantFinder
from hhcra.iterative_refinement import IterativeRefinementNet


class HHCRAExperimental(nn.Module):
    """
    HHCRA Experimental: The complete system with all three innovations.

    Architecture:
        Layer 1: C-JEPA (Perception) — unchanged
        Layer 2: GNN + Liquid Causal Graph (Structure + Co-evolving Dynamics)
        Layer 3: Neuro-Symbolic + HRM + Invariant Finder (Reasoning + Rule Discovery)
        Meta:    Autofeedback Loop (Self-feedback convergence)

    The key difference from base HHCRA:
        Base:        L1 → L2 → L3 (one-shot, static)
        Experimental: L1 → [L2 ↔ L3 ↔ L2 ↔ L3 ...] → convergence (iterative)
    """

    def __init__(self, config: Optional[HHCRAConfig] = None):
        super().__init__()
        self.config = config or HHCRAConfig()

        # Original layers (preserved for compatibility)
        self.layer1 = CJEPA(self.config)
        self.layer2 = MechanismLayer(self.config)
        self.layer3 = ReasoningLayer(self.config)
        self.feedback_router = FeedbackRouter()

        # === SINGULARITY INNOVATIONS ===

        # Innovation 1: Liquid Causal Graph
        self.liquid_graph = LiquidCausalGraph(self.config)

        # Innovation 2: Trajectory Invariant Finder
        self.invariant_finder = TrajectoryInvariantFinder(
            self.config, num_candidates=8
        )

        # Innovation 3: Autofeedback Network
        self.iterative_refinement = IterativeRefinementNet(self.config)

    # === SINGULARITY TRAINING ===

    def train_experimental(self, observations: torch.Tensor,
                          max_cycles: int = 10,
                          verbose: bool = True) -> Dict:
        """
        Full experimental training pipeline.

        Phase 1: Standard staged training (L1, L2, L3)
        Phase 2: Autofeedback cycles until convergence

        The iterative_refinement phase is where the iterative loop: the system
        discovers its own rules and refines its own structure.
        """
        if verbose:
            print("=" * 60)
            print("HHCRA SINGULARITY — Training Pipeline")
            print("=" * 60)

        # Phase 1: Standard staged training
        if verbose:
            print("\n--- Phase 1: Standard Staged Training ---")

        self._train_layer1(observations, verbose)
        self._train_layer2(observations, verbose)
        self._train_layer3(observations, verbose)

        # Phase 2: Autofeedback evolution
        if verbose:
            print("\n--- Phase 2: Autofeedback Iterative Evolution ---")

        # Get latent from trained Layer 1
        with torch.no_grad():
            latent = self.layer1.extract_variables(observations)

        # Sync GNN weights to iterative_refinement network
        self.iterative_refinement.gnn.W.data.copy_(self.layer2.gnn.W.data)

        # Run iterative_refinement cycles
        ac_result = self.iterative_refinement.run_until_convergence(
            latent, max_cycles=max_cycles, verbose=verbose
        )

        # Sync evolved weights back to Layer 2
        self.layer2.gnn.W.data.copy_(self.iterative_refinement.gnn.W.data)
        self.layer2.gnn.prune_to_dag()

        if verbose:
            print("\n" + self.experimental_summary())

        return ac_result

    def _train_layer1(self, observations: torch.Tensor, verbose: bool):
        """Stage 1: Train C-JEPA."""
        optimizer = optim.Adam(self.layer1.parameters(), lr=self.config.cjepa_lr)
        if verbose:
            print("\nSTAGE 1: C-JEPA (Latent Causal Variable Extraction)")

        for ep in range(self.config.train_epochs_l1):
            optimizer.zero_grad()
            loss = self.layer1.compute_loss(observations)
            loss.backward()
            nn.utils.clip_grad_norm_(self.layer1.parameters(), max_norm=5.0)
            optimizer.step()

            if verbose and (ep + 1) % max(1, self.config.train_epochs_l1 // 3) == 0:
                print(f"  Epoch {ep + 1}/{self.config.train_epochs_l1} | "
                      f"Loss: {loss.item():.6f}")

    def _train_layer2(self, observations: torch.Tensor, verbose: bool):
        """Stage 2: Train GNN + Liquid Net."""
        optimizer = optim.Adam(self.layer2.parameters(), lr=self.config.layer2_lr)
        if verbose:
            print(f"\nSTAGE 2: GNN + Liquid Net (Structure + Mechanisms)")

        with torch.no_grad():
            latent = self.layer1.extract_variables(observations)

        for ep in range(self.config.train_epochs_l2):
            optimizer.zero_grad()
            loss = self.layer2.compute_loss(latent)
            loss.backward()
            nn.utils.clip_grad_norm_(self.layer2.parameters(), max_norm=5.0)
            optimizer.step()

            if (ep + 1) % 5 == 0:
                self.layer2.gnn.update_lagrangian()

            if verbose and (ep + 1) % max(1, self.config.train_epochs_l2 // 3) == 0:
                with torch.no_grad():
                    A = self.layer2.gnn.adjacency(hard=True)
                    n_edges = int(A.sum().item())
                    is_dag = self.layer2.gnn._is_dag(A)
                print(f"  Epoch {ep + 1}/{self.config.train_epochs_l2} | "
                      f"Edges: {n_edges} | DAG: {is_dag} | Loss: {loss.item():.4f}")

        self.layer2.gnn.prune_to_dag()

    def _train_layer3(self, observations: torch.Tensor, verbose: bool):
        """Stage 3: Train HRM."""
        optimizer = optim.Adam(
            self.layer3.hrm.parameters(), lr=self.config.layer3_lr
        )
        if verbose:
            print(f"\nSTAGE 3: HRM (Reasoning Orchestration)")

        for ep in range(self.config.train_epochs_l3):
            optimizer.zero_grad()
            q = torch.randn(self.config.latent_dim)
            r = self.layer3.hrm.reason(q)

            if isinstance(r['result'], torch.Tensor) and r['result'].requires_grad:
                stability_loss = r['result'].norm()
                q2 = torch.randn(self.config.latent_dim)
                r2 = self.layer3.hrm.reason(q2)
                if isinstance(r2['result'], torch.Tensor) and r2['result'].requires_grad:
                    total_loss = 0.01 * stability_loss + 0.01 * r2['result'].norm()
                    total_loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.layer3.hrm.parameters(), max_norm=5.0
                    )
                    optimizer.step()

            if verbose and (ep + 1) % max(1, self.config.train_epochs_l3 // 3) == 0:
                print(f"  Epoch {ep + 1}/{self.config.train_epochs_l3} | "
                      f"Steps: {r['steps']} | Conv: {r['convergence']:.4f}")

    # === ENHANCED QUERY ===

    def forward(self, observations: torch.Tensor) -> dict:
        """Full forward pass through all layers."""
        latent = self.layer1.extract_variables(observations)
        latent_detached = LayerInterface.l1_to_l2(latent)
        l2_out = self.layer2(latent_detached)
        graph = self.layer2.symbolic_graph()

        return {
            'latent': latent,
            'layer2': l2_out,
            'graph': graph,
            'discovered_rules': list(self.invariant_finder.discovered_rules),
            'liquid_graph_evolution': self.liquid_graph.get_graph_evolution(),
            'iterative_refinement_converged': self.iterative_refinement.convergence_achieved,
        }

    def query(self, observations: torch.Tensor, query_type: CausalQueryType,
              X: int, Y: int, verbose: bool = True, **kwargs) -> dict:
        """Answer a causal query through the enhanced pipeline."""
        fwd = self.forward(observations)
        traj_detached = LayerInterface.l2_to_l3(fwd['layer2']['trajectories'])

        result = self.layer3.answer_query(
            query_type, X, Y, fwd['graph'],
            traj_detached, self.layer2, **kwargs
        )

        # Enrich with Experimental metadata
        result['experimental'] = {
            'discovered_rules': len(self.invariant_finder.discovered_rules),
            'novel_rules': sum(
                1 for r in self.invariant_finder.discovered_rules if r.is_novel
            ),
            'iterative_refinement_converged': self.iterative_refinement.convergence_achieved,
            'refinement_cycles': len(self.iterative_refinement.cycle_history),
        }

        # Route feedback
        if result.get('feedback'):
            self.feedback_router.route(
                result['feedback'], self.layer2, self.layer1, verbose
            )

        return result

    # === DIAGNOSTICS ===

    def experimental_summary(self) -> str:
        """Comprehensive summary of the Experimental system state."""
        sg = self.layer2.symbolic_graph()
        diag = self.layer3.generate_diagnostic(sg)

        lines = [
            "",
            "=" * 60,
            "HHCRA SINGULARITY — System Summary",
            "=" * 60,
            "",
            "Base Architecture:",
            f"  Layer 1: C-JEPA ({self.config.num_vars} slots, "
            f"dim={self.config.latent_dim})",
            f"  Layer 2: GNN + Liquid Net "
            f"(method={self.config.liquid_method})",
            f"  Layer 3: Neuro-Symbolic + HRM",
            "",
            "Experimental Extensions (unvalidated):",
            f"  [1] Liquid Causal Graph: graph evolves as ODE",
            f"      Graph change rate: "
            f"{self.liquid_graph.compute_graph_change_rate():.4f}",
            f"  [2] Invariant Finder: {len(self.invariant_finder.discovered_rules)}"
            f" rules discovered",
        ]

        novel = sum(
            1 for r in self.invariant_finder.discovered_rules if r.is_novel
        )
        lines.append(f"      Novel rules: {novel}")

        lines.extend([
            f"  [3] Autofeedback Loop: "
            f"{len(self.iterative_refinement.cycle_history)} cycles",
            f"      Fixed point reached: "
            f"{self.iterative_refinement.convergence_achieved}",
            "",
            "Learned Causal Graph:",
            f"  Nodes: {len(sg.nodes)} | Edges: {sg.edge_count()} | "
            f"DAG: {sg.is_dag()}",
            f"  Density: {diag['density']:.1%}",
        ])

        for p, c, w in sg.edges[:10]:
            lines.append(f"    X{p} -> X{c} (weight: {w:.3f})")
        if sg.edge_count() > 10:
            lines.append(f"    ... ({sg.edge_count() - 10} more)")

        if self.invariant_finder.discovered_rules:
            lines.extend(["", "Discovered Rules:"])
            for rule in self.invariant_finder.discovered_rules[:5]:
                tag = "NOVEL" if rule.is_novel else "known"
                lines.append(
                    f"  [{tag}] {rule.name}: {rule.expression} "
                    f"(conf={rule.confidence:.3f})"
                )
            remaining = len(self.invariant_finder.discovered_rules) - 5
            if remaining > 0:
                lines.append(f"  ... ({remaining} more)")

        lines.extend([
            "",
            "Pearl's Ladder Coverage:",
            "  Rung 1 (Observation):    P(Y|X)",
            "  Rung 2 (Intervention):   P(Y|do(X)) via Liquid Causal Graph",
            "  Rung 3 (Counterfactual): P(Y_{x'}|X=x) via ABP + Genesis Rules",
            "",
            "=" * 60,
        ])

        return "\n".join(lines)
