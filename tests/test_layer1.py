"""Tests for Layer 1: C-JEPA (Causal Joint Embedding Predictive Architecture)."""

import pytest
import torch
import torch.nn as nn

from hhcra.config import HHCRAConfig
from hhcra.layer1_cjepa import CJEPA, SlotAttention


class TestSlotAttention:
    def test_output_shape(self, config):
        sa = SlotAttention(config.num_vars, config.latent_dim, num_iters=config.slot_attention_iters)
        z = torch.randn(4, config.latent_dim)
        slots = sa(z)
        assert slots.shape == (4, config.num_vars, config.latent_dim)

    def test_normalized_output(self, config):
        sa = SlotAttention(config.num_vars, config.latent_dim)
        z = torch.randn(4, config.latent_dim)
        slots = sa(z)
        norms = slots.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=0.1)

    def test_slot_diversity(self, config):
        """v0.4.1 FIX VALIDATION: competitive attention produces diverse slots."""
        torch.manual_seed(42)
        sa = SlotAttention(config.num_vars, config.latent_dim, num_iters=3)
        sa.train()  # Training mode adds noise for slot diversity
        z = torch.randn(2, config.latent_dim)
        slots = sa(z)
        # Slots should not all be identical — std across slot dim should be > 0
        slot_std = slots[0].std(dim=0).mean()
        assert slot_std > 0.001, f"Slot diversity too low: {slot_std:.4f}"


class TestCJEPA:
    def test_extract_variables_shape(self, config, observations):
        model = CJEPA(config)
        latent = model.extract_variables(observations)
        B, T = observations.shape[:2]
        assert latent.shape == (B, T, config.num_vars, config.latent_dim)

    def test_extract_variables_no_nan(self, config, observations):
        model = CJEPA(config)
        latent = model.extract_variables(observations)
        assert not torch.any(torch.isnan(latent)).item()

    def test_is_nn_module(self, config):
        model = CJEPA(config)
        assert isinstance(model, nn.Module)

    def test_has_parameters(self, config):
        model = CJEPA(config)
        params = list(model.parameters())
        assert len(params) > 0

    def test_compute_loss_returns_scalar(self, config, observations):
        model = CJEPA(config)
        loss = model.compute_loss(observations)
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_gradient_flow(self, config, observations):
        """Verify gradients flow through C-JEPA."""
        model = CJEPA(config)
        loss = model.compute_loss(observations)
        loss.backward()
        grad_count = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
        assert grad_count > 0, "No gradients flowed through C-JEPA"

    def test_handle_feedback(self, config):
        model = CJEPA(config)
        old_mu = model.slot_attention.slot_mu.clone()
        model.handle_feedback({'increase_resolution': True})
        new_mu = model.slot_attention.slot_mu
        assert not torch.equal(old_mu, new_mu)

    def test_temporal_smoothing(self, config, observations):
        """Verify temporal smoothing affects output."""
        model = CJEPA(config)
        latent = model.extract_variables(observations)
        # First timestep should differ from later ones due to temporal smoothing
        t0 = latent[:, 0, :, :]
        t1 = latent[:, 1, :, :]
        # They should be different (temporal smoothing kicks in at t>0)
        assert not torch.allclose(t0, t1)

    def test_adaptive_slot_pruning(self, config):
        """v0.8.0: Unused slots should have low utilization."""
        torch.manual_seed(42)
        model = CJEPA(config)
        obs = torch.randn(4, 8, config.obs_dim)
        model.extract_variables(obs)
        active = model.get_active_slots()
        util = model.slot_attention.slot_utilization
        assert util.std() > 0.001, "All slots equally used — no specialization"

    def test_slot_utilization_tracked(self, config):
        """v0.8.0: SlotAttention should track slot_utilization."""
        sa = SlotAttention(config.num_vars, config.latent_dim)
        z = torch.randn(4, config.latent_dim)
        sa(z)
        assert hasattr(sa, 'slot_utilization')
        assert sa.slot_utilization.shape == (config.num_vars,)

    def test_independence_regularization(self, config):
        """v0.8.0: Independence loss includes decorrelation term."""
        torch.manual_seed(42)
        model = CJEPA(config)
        # Use small batch/time to stay fast
        obs = torch.randn(2, 4, config.obs_dim)
        loss = model.compute_loss(obs)
        # The loss should include the independence term (non-zero)
        assert loss.item() > 0
        # Verify gradient flows through independence loss
        loss.backward()
        grad_count = sum(1 for p in model.parameters()
                         if p.grad is not None and p.grad.abs().sum() > 0)
        assert grad_count > 0, "Independence loss has no gradient"

    def test_get_active_slots_returns_tensor(self, config):
        """v0.8.0: get_active_slots returns a tensor of indices."""
        model = CJEPA(config)
        obs = torch.randn(2, 4, config.obs_dim)
        model.extract_variables(obs)
        active = model.get_active_slots()
        assert isinstance(active, torch.Tensor)
        assert active.dim() == 1
