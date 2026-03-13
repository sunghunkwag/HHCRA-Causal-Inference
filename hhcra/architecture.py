"""
HHCRA: Hierarchical Hybrid Causal Reasoning Architecture

Main orchestrator class combining all 3 layers:
    Layer 1: C-JEPA (Perception)
    Layer 2: GNN + Liquid Net (Mechanism) [tight coupling]
    Layer 3: Neuro-symbolic + HRM (Reasoning) [tight coupling]

Staged training API: train_layer1(), train_layer2(), train_layer3()
All layers are nn.Module subclasses with proper gradient flow.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Optional

from hhcra.config import HHCRAConfig
from hhcra.causal_graph import CausalGraphData, CausalQueryType
from hhcra.layer1_cjepa import CJEPA
from hhcra.layer2_mechanism import MechanismLayer
from hhcra.layer3_reasoning import ReasoningLayer
from hhcra.interfaces import LayerInterface, FeedbackRouter


class HHCRA(nn.Module):
    """
    HHCRA: Hierarchical Hybrid Causal Reasoning Architecture

    5 components in 3 layers:
        Layer 1: C-JEPA                    (Perception)
        Layer 2: GNN + Liquid Net          (Mechanism)  [tight coupling]
        Layer 3: Neuro-symbolic + HRM      (Reasoning)  [tight coupling]

    3 connection types:
        Tight coupling:    Within-layer shared computation
        Interface coupling: Between-layer .detach() (no gradient crossing)
        Feedback coupling:  Top-down diagnostic signals

    Full Pearl's Ladder coverage:
        Rung 1: Observation   P(Y|X)
        Rung 2: Intervention  P(Y|do(X))
        Rung 3: Counterfactual P(Y_{x'}|X=x, Y=y)
    """

    def __init__(self, config: Optional[HHCRAConfig] = None):
        super().__init__()
        self.config = config or HHCRAConfig()
        self.layer1 = CJEPA(self.config)
        self.layer2 = MechanismLayer(self.config)
        self.layer3 = ReasoningLayer(self.config)
        self.feedback_router = FeedbackRouter()

    # --- Core Pipeline ---

    def forward(self, observations: torch.Tensor) -> dict:
        """Full forward pass through all 3 layers."""
        # Layer 1: Perception
        latent = self.layer1.extract_variables(observations)

        # Interface: L1 -> L2 (detach — no gradient crossing between layers)
        latent_detached = LayerInterface.l1_to_l2(latent)

        # Layer 2: Mechanism
        l2_out = self.layer2(latent_detached)

        # Interface: L2 -> L3 (symbolic conversion)
        graph = self.layer2.symbolic_graph()

        return {
            'latent': latent,
            'layer2': l2_out,
            'graph': graph,
        }

    def query(self, observations: torch.Tensor, query_type: CausalQueryType,
              X: int, Y: int, verbose: bool = True, **kwargs) -> dict:
        """Answer a causal query through the complete pipeline."""
        fwd = self.forward(observations)

        # Interface: L2 -> L3 trajectories detached
        traj_detached = LayerInterface.l2_to_l3(fwd['layer2']['trajectories'])

        result = self.layer3.answer_query(
            query_type, X, Y, fwd['graph'],
            traj_detached,
            self.layer2, **kwargs)

        # Route feedback to lower layers
        if result.get('feedback'):
            self.feedback_router.route(
                result['feedback'], self.layer2, self.layer1, verbose)

        return result

    # --- Staged Training API ---

    def train_layer1(self, observations: torch.Tensor, verbose: bool = True):
        """Stage 1: Train C-JEPA (latent variable extraction)."""
        optimizer = optim.Adam(self.layer1.parameters(), lr=self.config.cjepa_lr)

        if verbose:
            print("=" * 60)
            print("STAGE 1: C-JEPA (Latent Causal Variable Extraction)")
            print("=" * 60)

        for ep in range(self.config.train_epochs_l1):
            optimizer.zero_grad()
            loss = self.layer1.compute_loss(observations)
            loss.backward()
            nn.utils.clip_grad_norm_(self.layer1.parameters(), max_norm=5.0)
            optimizer.step()

            if verbose and (ep + 1) % max(1, self.config.train_epochs_l1 // 3) == 0:
                print(f"  Epoch {ep + 1}/{self.config.train_epochs_l1} | "
                      f"Mask-prediction loss: {loss.item():.6f}")

    def train_layer2(self, observations: torch.Tensor, verbose: bool = True):
        """Stage 2: Train GNN + Liquid Net (structure + mechanisms)."""
        optimizer = optim.Adam(self.layer2.parameters(), lr=self.config.layer2_lr)

        if verbose:
            print(f"\n{'=' * 60}")
            print("STAGE 2: GNN + Liquid Net (Structure + Mechanisms)")
            print("=" * 60)

        # Get latent from trained Layer 1 (detached — no gradient to L1)
        with torch.no_grad():
            latent = self.layer1.extract_variables(observations)

        for ep in range(self.config.train_epochs_l2):
            optimizer.zero_grad()
            loss = self.layer2.compute_loss(latent)
            loss.backward()
            nn.utils.clip_grad_norm_(self.layer2.parameters(), max_norm=5.0)
            optimizer.step()

            # Update NOTEARS Lagrangian multipliers periodically
            if (ep + 1) % 5 == 0:
                h = self.layer2.gnn.update_lagrangian()

            if verbose and (ep + 1) % max(1, self.config.train_epochs_l2 // 3) == 0:
                with torch.no_grad():
                    A = self.layer2.gnn.adjacency(hard=True)
                    n_edges = int(A.sum().item())
                    is_dag = self.layer2.gnn._is_dag(A)
                    dag_pen = self.layer2.gnn.dag_penalty().item()
                print(f"  Epoch {ep + 1}/{self.config.train_epochs_l2} | "
                      f"Edges: {n_edges} | DAG: {is_dag} | "
                      f"DAG penalty: {dag_pen:.4f} | Loss: {loss.item():.4f}")

        # Final DAG enforcement
        self.layer2.gnn.prune_to_dag()

        if verbose:
            with torch.no_grad():
                A_final = self.layer2.gnn.adjacency(hard=True)
            print(f"  Final: {int(A_final.sum().item())} edges, "
                  f"DAG verified: {self.layer2.gnn._is_dag(A_final)}")

    def train_layer3(self, observations: torch.Tensor, verbose: bool = True):
        """
        Stage 3: Train HRM (reasoning orchestration).

        v0.4.1 FIX: Loss is computed from HRM output tensor (has gradient)
        instead of from detached convergence scalar. The old code used
        torch.tensor(r['convergence']) which created a new leaf tensor with
        no gradient connection to HRM parameters — so train_layer3 was a
        no-op that never updated any weights.
        """
        optimizer = optim.Adam(self.layer3.hrm.parameters(), lr=self.config.layer3_lr)

        if verbose:
            print(f"\n{'=' * 60}")
            print("STAGE 3: HRM (Reasoning Orchestration)")
            print("=" * 60)

        for ep in range(self.config.train_epochs_l3):
            optimizer.zero_grad()
            q = torch.randn(self.config.latent_dim)
            r = self.layer3.hrm.reason(q)

            # v0.4.1 FIX: compute loss from output tensor that retains gradient
            # Old: conv_loss = F.mse_loss(torch.tensor(r['convergence']), ...)
            #      ^ torch.tensor(float) is a NEW LEAF — zero gradient to HRM
            # New: loss from r['result'] which flows through GRU/halt networks
            if isinstance(r['result'], torch.Tensor) and r['result'].requires_grad:
                # Output stability: encourage bounded, non-degenerate outputs
                stability_loss = r['result'].norm()
                # Run second query for gradient diversity
                q2 = torch.randn(self.config.latent_dim)
                r2 = self.layer3.hrm.reason(q2)
                if isinstance(r2['result'], torch.Tensor) and r2['result'].requires_grad:
                    total_loss = 0.01 * stability_loss + 0.01 * r2['result'].norm()
                    total_loss.backward()
                    nn.utils.clip_grad_norm_(self.layer3.hrm.parameters(), max_norm=5.0)
                    optimizer.step()

            if verbose and (ep + 1) % max(1, self.config.train_epochs_l3 // 3) == 0:
                resets = sum(1 for t in r['trace']
                             if 'event' in t and t['event'] == 'H_MODULE_RESET')
                print(f"  Epoch {ep + 1}/{self.config.train_epochs_l3} | "
                      f"Steps: {r['steps']} | Conv: {r['convergence']:.4f} | "
                      f"Resets: {resets}")

    def train_all(self, observations: torch.Tensor, verbose: bool = True):
        """
        Full staged training pipeline.
        Each stage trains its layer independently (no gradient crossing).
        """
        self.train_layer1(observations, verbose)
        self.train_layer2(observations, verbose)
        self.train_layer3(observations, verbose)

        if verbose:
            print(f"\n{'=' * 60}")
            print("TRAINING COMPLETE")
            print("=" * 60)

    # Legacy compatibility
    def train_model(self, observations: torch.Tensor, verbose: bool = True):
        """Alias for train_all."""
        self.train_all(observations, verbose)

    # --- Diagnostics ---

    def summary(self) -> str:
        sg = self.layer2.symbolic_graph()
        diag = self.layer3.generate_diagnostic(sg)
        lines = [
            "",
            "=" * 60,
            "HHCRA: Hierarchical Hybrid Causal Reasoning Architecture",
            "=" * 60,
            "",
            "Architecture:",
            f"  Layer 1: C-JEPA",
            f"    {self.config.num_vars} latent variable slots from "
            f"{self.config.obs_dim}-dim observations",
            f"    Latent dim: {self.config.latent_dim}",
            f"",
            f"  Layer 2: GNN + Liquid Neural Network [tightly coupled]",
            f"    GNN: NOTEARS continuous optimization ({self.config.num_vars} nodes)",
            f"    Liquid Net: Neural ODE (method={self.config.liquid_method}, "
            f"dt={self.config.liquid_dt}, steps={self.config.liquid_ode_steps})",
            f"",
            f"  Layer 3: Neuro-Symbolic + HRM [tightly coupled]",
            f"    Neuro-symbolic: d-separation, backdoor/frontdoor, "
            f"do-calculus (3 rules), ID algorithm",
            f"    HRM: GRU + ACT, {self.config.hrm_max_steps} max steps, "
            f"H-update every {self.config.hrm_update_interval} steps",
            f"",
            f"Connections:",
            f"  L1 -> L2: Interface (.detach() — no gradient crossing)",
            f"  L2 -> L3: Interface (.detach() — no gradient crossing)",
            f"  L3 -> L2: Feedback (structure revision requests)",
            f"  L3 -> L1: Feedback (variable resolution adjustment)",
            f"",
            f"Learned Causal Graph:",
            f"  Nodes: {len(sg.nodes)} | Edges: {sg.edge_count()} | DAG: {sg.is_dag()}",
            f"  Density: {diag['density']:.1%}",
        ]

        for p, c, w in sg.edges[:15]:
            lines.append(f"    X{p} -> X{c} (weight: {w:.3f})")
        if sg.edge_count() > 15:
            lines.append(f"    ... ({sg.edge_count() - 15} more)")

        if diag['issues']:
            lines.append(f"\n  Diagnostics:")
            for iss in diag['issues']:
                lines.append(f"    WARNING: {iss}")

        lines.extend([
            "",
            "Pearl's Ladder Coverage:",
            "  Rung 1 (Observation):    P(Y|X)",
            "  Rung 2 (Intervention):   P(Y|do(X)) via do-calculus + Neural ODE",
            "  Rung 3 (Counterfactual): P(Y_{x'}|X=x) via ABP procedure",
        ])

        return "\n".join(lines)
