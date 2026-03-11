"""
LayerInterface and FeedbackRouter for inter-layer communication.

Defines the coupling types:
  - Tight coupling: within-layer shared computation (GNN<->Liquid, NeuroSym<->HRM)
  - Interface coupling: between layers with .detach() — no gradient crossing
  - Feedback coupling: top-down diagnostic signals from L3 -> L2 -> L1
"""

import torch
from typing import Optional

from hhcra.layer1_cjepa import CJEPA
from hhcra.layer2_mechanism import MechanismLayer


class LayerInterface:
    """
    Manages data flow between layers with proper gradient isolation.

    L1 -> L2: latent variables with .detach() (no gradient crossing)
    L2 -> L3: symbolic graph + trajectories with .detach()
    """

    @staticmethod
    def l1_to_l2(latent: torch.Tensor) -> torch.Tensor:
        """Interface: Layer 1 output -> Layer 2 input. Detach gradients."""
        return latent.detach()

    @staticmethod
    def l2_to_l3(trajectories: torch.Tensor) -> torch.Tensor:
        """Interface: Layer 2 output -> Layer 3 input. Detach gradients."""
        return trajectories.detach()


class FeedbackRouter:
    """Routes diagnostic feedback from upper to lower layers."""

    @staticmethod
    def route(feedback: dict, layer2: MechanismLayer, layer1: CJEPA,
              verbose: bool = True):
        if not feedback:
            return

        if 'to_layer2' in feedback and feedback['to_layer2']:
            fb = feedback['to_layer2']
            if verbose:
                print(f"    [Feedback L3->L2] X{fb.get('X', '?')}->X{fb.get('Y', '?')}: "
                      f"{fb.get('issue', '')}")
            layer2.handle_feedback(fb)

        if 'to_layer1' in feedback and feedback['to_layer1']:
            fb = feedback['to_layer1']
            if verbose:
                print(f"    [Feedback L3->L1] {fb.get('reason', '')}")
            layer1.handle_feedback(fb)
