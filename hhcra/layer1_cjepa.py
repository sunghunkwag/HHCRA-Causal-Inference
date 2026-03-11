"""
Layer 1: C-JEPA (Causal Joint Embedding Predictive Architecture)

Extracts causally relevant latent variables from high-dimensional observations.
Uses object-level masking to induce latent interventions, forcing the model
to learn causal interaction patterns.

Interface OUT -> Layer 2: latent_vars (B, T, num_vars, latent_dim)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from hhcra.config import HHCRAConfig


class SlotAttention(nn.Module):
    """Decompose latent vector into N variable slots via attention."""

    def __init__(self, num_vars: int, latent_dim: int):
        super().__init__()
        self.num_vars = num_vars
        self.latent_dim = latent_dim

        self.W_key = nn.Linear(latent_dim, latent_dim, bias=False)
        self.W_value = nn.Linear(latent_dim, latent_dim, bias=False)
        self.W_query = nn.ModuleList([
            nn.Linear(latent_dim, latent_dim, bias=True)
            for _ in range(num_vars)
        ])
        self.scale = latent_dim ** 0.5

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args: z (B, latent_dim)
        Returns: slots (B, num_vars, latent_dim)
        """
        B = z.shape[0]
        keys = self.W_key(z)      # (B, D)
        values = self.W_value(z)  # (B, D)

        slots = []
        for v in range(self.num_vars):
            queries = self.W_query[v](z)  # (B, D)
            attn = torch.sum(queries * keys, dim=-1, keepdim=True) / self.scale
            attn = torch.sigmoid(attn)
            slot = attn * values + (1 - attn) * queries
            slots.append(slot)

        slots = torch.stack(slots, dim=1)  # (B, N, D)
        slots = F.normalize(slots, dim=-1, eps=1e-8)
        return slots


class CJEPA(nn.Module):
    """
    Layer 1: Perception Layer

    Extracts causally relevant latent variables (V in SCM) from
    high-dimensional observations.
    """

    def __init__(self, config: HHCRAConfig):
        super().__init__()
        self.config = config
        D_obs = config.obs_dim
        D_lat = config.latent_dim
        N = config.num_vars

        # Encoder: observation -> latent space
        self.encoder = nn.Linear(D_obs, D_lat)

        # Slot attention
        self.slot_attention = SlotAttention(N, D_lat)

        # Temporal smoothing
        self.W_temporal = nn.Linear(D_lat, D_lat, bias=False)
        self.temporal_alpha = 0.7

        # Predictor: masked slot prediction
        self.predictor = nn.Sequential(
            nn.Linear(D_lat, D_lat),
            nn.Tanh(),
        )

        self.loss_history = []

    def extract_variables(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Extract latent causal variables from observations.

        Args: observations (B, T, obs_dim)
        Returns: latent_vars (B, T, num_vars, latent_dim)
        """
        B, T, _ = observations.shape
        N = self.config.num_vars
        D = self.config.latent_dim

        all_slots = []
        prev_slots = torch.zeros(B, N, D, device=observations.device)

        for t in range(T):
            z = torch.tanh(self.encoder(observations[:, t, :]))  # (B, D)
            slots = self.slot_attention(z)  # (B, N, D)

            if t > 0:
                temporal = torch.tanh(
                    self.W_temporal(prev_slots.reshape(B * N, D))
                ).reshape(B, N, D)
                slots = self.temporal_alpha * slots + (1 - self.temporal_alpha) * temporal

            all_slots.append(slots)
            prev_slots = slots

        latent = torch.stack(all_slots, dim=1)  # (B, T, N, D)
        return latent

    def compute_loss(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Compute mask-prediction loss for training.
        Masks random variable slots and predicts from context.
        """
        latent = self.extract_variables(observations)
        B, T, N, D = latent.shape
        num_mask = max(1, int(N * self.config.mask_ratio))

        total_loss = torch.tensor(0.0, device=observations.device)
        count = 0

        for b in range(B):
            mask_idx = torch.randperm(N, device=observations.device)[:num_mask]
            visible_mask = torch.ones(N, dtype=torch.bool, device=observations.device)
            visible_mask[mask_idx] = False

            for t in range(T):
                target = latent[b, t, mask_idx, :]  # (num_mask, D)
                context = latent[b, t, visible_mask, :].mean(dim=0)  # (D,)
                pred = self.predictor(context).unsqueeze(0).expand(num_mask, -1)
                total_loss = total_loss + F.mse_loss(pred, target.detach())
                count += 1

        loss = total_loss / max(count, 1)
        self.loss_history.append(loss.item())
        return loss

    def handle_feedback(self, feedback: dict):
        """Handle feedback from upper layers (no gradient, config-level)."""
        if feedback.get('increase_resolution'):
            with torch.no_grad():
                for q in self.slot_attention.W_query:
                    q.weight.add_(torch.randn_like(q.weight) * 0.02)
                    if q.bias is not None:
                        q.bias.add_(torch.randn_like(q.bias) * 0.05)
