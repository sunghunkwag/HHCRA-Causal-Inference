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
    """
    Decompose latent vector into N variable slots via competitive attention.

    v0.4.1 FIX: Uses softmax over slots (competitive) instead of independent
    sigmoid gating. Each slot competes for the input signal, forcing
    specialization on different aspects of the input. This is closer to
    Locatello et al. (2020) and resolves the V-alignment bottleneck where
    all slots would extract identical information.

    Previous issue: sigmoid gating was independent per slot, so no competition
    existed — all N slots could attend equally to the same input features,
    producing near-identical representations and inflating Pipeline SHD.
    """

    def __init__(self, num_vars: int, latent_dim: int, num_iters: int = 3):
        super().__init__()
        self.num_vars = num_vars
        self.latent_dim = latent_dim
        self.num_iters = num_iters
        self.scale = latent_dim ** 0.5

        # Learnable slot initialization
        self.slot_mu = nn.Parameter(torch.randn(1, num_vars, latent_dim) * 0.05)
        self.slot_log_sigma = nn.Parameter(torch.zeros(1, num_vars, latent_dim))

        # Shared projections for competitive attention
        self.W_key = nn.Linear(latent_dim, latent_dim, bias=False)
        self.W_value = nn.Linear(latent_dim, latent_dim, bias=False)
        self.W_query = nn.Linear(latent_dim, latent_dim, bias=False)

        # GRU update for iterative slot refinement
        self.gru = nn.GRUCell(latent_dim, latent_dim)

        # Slot-level residual MLP
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
        )

        self.norm_input = nn.LayerNorm(latent_dim)
        self.norm_slots = nn.LayerNorm(latent_dim)
        self.norm_mlp = nn.LayerNorm(latent_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args: z (B, latent_dim)
        Returns: slots (B, num_vars, latent_dim)
        """
        B = z.shape[0]
        N = self.num_vars
        D = self.latent_dim

        # Project input
        inputs = self.norm_input(z).unsqueeze(1)  # (B, 1, D)
        k = self.W_key(inputs)    # (B, 1, D)
        v = self.W_value(inputs)  # (B, 1, D)

        # Initialize slots with learned parameters + noise for diversity
        slots = self.slot_mu.expand(B, -1, -1).clone()
        if self.training:
            sigma = self.slot_log_sigma.exp().expand(B, -1, -1)
            slots = slots + sigma * torch.randn_like(slots)

        # Iterative competitive attention
        for _ in range(self.num_iters):
            slots_normed = self.norm_slots(slots)
            q = self.W_query(slots_normed)  # (B, N, D)

            # Attention logits: (B, N, 1)
            attn_logits = torch.bmm(q, k.transpose(1, 2)) / self.scale

            # COMPETITIVE: softmax over slots (dim=1), not independent sigmoid
            # This forces slots to compete for the input signal
            attn = F.softmax(attn_logits, dim=1)  # (B, N, 1)

            # v0.8.0: Track per-slot utilization for adaptive slot pruning
            self.slot_utilization = attn.squeeze(-1).mean(dim=0)  # (N,)

            # Weighted value: each slot gets its share
            updates = attn * v  # (B, N, D) via broadcast

            # GRU update for slot refinement
            updates_flat = updates.reshape(B * N, D)
            slots_flat = slots.reshape(B * N, D)
            slots = self.gru(updates_flat, slots_flat).reshape(B, N, D)

            # Residual MLP
            slots = slots + self.mlp(self.norm_mlp(slots))

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

        # v0.6.0: Deeper encoder with residual connection
        # 2-layer MLP preserves more signal from observations
        hidden_dim = max(D_lat * 2, D_obs // 2)
        self.encoder = nn.Sequential(
            nn.Linear(D_obs, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, D_lat),
        )
        # Residual projection (skip connection from obs to latent space)
        self.encoder_skip = nn.Linear(D_obs, D_lat, bias=False)

        # Slot attention (competitive, v0.4.1)
        self.slot_attention = SlotAttention(N, D_lat, num_iters=config.slot_attention_iters)

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
            obs_t = observations[:, t, :]
            # v0.6.0: Deep encoder + residual skip connection
            z = torch.tanh(self.encoder(obs_t) + self.encoder_skip(obs_t))  # (B, D)
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

    def get_active_slots(self, threshold=0.05):
        """Return indices of slots with utilization above threshold."""
        if hasattr(self.slot_attention, 'slot_utilization'):
            return (self.slot_attention.slot_utilization > threshold).nonzero().flatten()
        return torch.arange(self.config.num_vars)

    def compute_loss(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Compute mask-prediction loss for training.
        Masks random variable slots and predicts from context.

        v0.5.0: Vectorized over T dimension (eliminates inner loop).
        Per-batch masking is retained to ensure diverse mask patterns.
        """
        latent = self.extract_variables(observations)
        B, T, N, D = latent.shape
        num_mask = max(1, int(N * self.config.mask_ratio))

        total_loss = torch.tensor(0.0, device=observations.device)

        for b in range(B):
            mask_idx = torch.randperm(N, device=observations.device)[:num_mask]
            visible_mask = torch.ones(N, dtype=torch.bool, device=observations.device)
            visible_mask[mask_idx] = False

            # Vectorized over T: (T, num_mask, D) and (T, D)
            target = latent[b, :, mask_idx, :]  # (T, num_mask, D)
            context = latent[b, :, visible_mask, :].mean(dim=1)  # (T, D)
            pred = self.predictor(context).unsqueeze(1).expand(-1, num_mask, -1)  # (T, num_mask, D)
            total_loss = total_loss + F.mse_loss(pred, target.detach())

        loss = total_loss / B

        # v0.8.0: Independence regularization — decorrelate slots
        # Use slot_attention parameters directly (not full latent graph) to
        # keep autograd graph small and avoid timeout in multi-call scenarios.
        slot_mu = self.slot_attention.slot_mu.squeeze(0)  # (N, D)
        normed_mu = F.normalize(slot_mu, dim=-1)
        sim_mu = normed_mu @ normed_mu.T  # (N, N)
        off_diag = sim_mu - torch.eye(N, device=observations.device)
        independence_loss = off_diag.pow(2).mean()
        loss = loss + 0.1 * independence_loss

        self.loss_history.append(loss.item())
        return loss

    def handle_feedback(self, feedback: dict):
        """Handle feedback from upper layers (no gradient, config-level)."""
        if feedback.get('increase_resolution'):
            with torch.no_grad():
                # Perturb slot initialization to encourage re-specialization
                self.slot_attention.slot_mu.data.add_(
                    torch.randn_like(self.slot_attention.slot_mu) * 0.05)
