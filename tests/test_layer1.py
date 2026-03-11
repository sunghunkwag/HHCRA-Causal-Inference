"""Tests for Layer 1: C-JEPA (Causal Joint Embedding Predictive Architecture)."""

import pytest
import torch
import torch.nn as nn

from hhcra.config import HHCRAConfig
from hhcra.layer1_cjepa import CJEPA, SlotAttention


class TestSlotAttention:
    def test_output_shape(self, config):
        sa = SlotAttention(config.num_vars, config.latent_dim)
        z = torch.randn(4, config.latent_dim)
        slots = sa(z)
        assert slots.shape == (4, config.num_vars, config.latent_dim)

    def test_normalized_output(self, config):
        sa = SlotAttention(config.num_vars, config.latent_dim)
        z = torch.randn(4, config.latent_dim)
        slots = sa(z)
        norms = slots.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=0.1)


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
        old_bias = model.slot_attention.W_query[0].bias.clone()
        model.handle_feedback({'increase_resolution': True})
        new_bias = model.slot_attention.W_query[0].bias
        assert not torch.equal(old_bias, new_bias)

    def test_temporal_smoothing(self, config, observations):
        """Verify temporal smoothing affects output."""
        model = CJEPA(config)
        latent = model.extract_variables(observations)
        # First timestep should differ from later ones due to temporal smoothing
        t0 = latent[:, 0, :, :]
        t1 = latent[:, 1, :, :]
        # They should be different (temporal smoothing kicks in at t>0)
        assert not torch.allclose(t0, t1)
