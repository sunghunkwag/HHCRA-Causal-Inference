"""Tests for Phase 9: Multi-Modal Perception Interface."""

import pytest
import torch
import torch.nn as nn

from hhcra.config import HHCRAConfig
from hhcra.perception import (
    VisionPerception, TimeSeriesPerception, TextPerception,
    PerceptionRouter, create_default_perception, PerceptionInterface,
)


@pytest.fixture
def config():
    return HHCRAConfig(num_vars=8, latent_dim=10)


class TestVisionPerception:
    def test_is_nn_module(self, config):
        vp = VisionPerception(config)
        assert isinstance(vp, nn.Module)

    def test_encode_single_frame(self, config):
        vp = VisionPerception(config)
        img = torch.randn(2, 3, 64, 64)
        slots = vp.encode(img)
        assert slots.shape == (2, config.num_vars, config.latent_dim)

    def test_encode_video(self, config):
        vp = VisionPerception(config)
        video = torch.randn(2, 5, 3, 64, 64)  # B, T, C, H, W
        slots = vp.encode(video)
        assert slots.shape == (2, 5, config.num_vars, config.latent_dim)

    def test_normalized_output(self, config):
        vp = VisionPerception(config)
        img = torch.randn(2, 3, 64, 64)
        slots = vp.encode(img)
        norms = slots.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=0.1)

    def test_input_type(self, config):
        vp = VisionPerception(config)
        assert vp.input_type() == "vision"


class TestTimeSeriesPerception:
    def test_encode_shape(self, config):
        tsp = TimeSeriesPerception(config, input_channels=3)
        data = torch.randn(4, 20, 3)  # B, T, channels
        slots = tsp.encode(data)
        assert slots.shape == (4, config.num_vars, config.latent_dim)

    def test_is_nn_module(self, config):
        tsp = TimeSeriesPerception(config)
        assert isinstance(tsp, nn.Module)

    def test_input_type(self, config):
        tsp = TimeSeriesPerception(config)
        assert tsp.input_type() == "time_series"


class TestTextPerception:
    def test_encode_shape(self, config):
        tp = TextPerception(config, vocab_size=100)
        tokens = torch.randint(0, 100, (3, 10))  # B, seq_len
        slots = tp.encode(tokens)
        assert slots.shape == (3, config.num_vars, config.latent_dim)

    def test_causal_claims_shape(self, config):
        tp = TextPerception(config, vocab_size=100)
        tokens = torch.randint(0, 100, (3, 10))
        adj = tp.extract_causal_claims(tokens)
        assert adj.shape == (3, config.num_vars, config.num_vars)

    def test_no_self_loops(self, config):
        tp = TextPerception(config, vocab_size=100)
        tokens = torch.randint(0, 100, (3, 10))
        adj = tp.extract_causal_claims(tokens)
        for b in range(3):
            diag = torch.diag(adj[b])
            assert torch.allclose(diag, torch.zeros_like(diag))

    def test_input_type(self, config):
        tp = TextPerception(config)
        assert tp.input_type() == "text"


class TestPerceptionRouter:
    def test_register_and_route(self, config):
        router = PerceptionRouter(config)
        vp = VisionPerception(config)
        router.register_module_type("vision", vp)

        img = torch.randn(2, 3, 64, 64)
        slots = router.route(img, "vision")
        assert slots.shape == (2, config.num_vars, config.latent_dim)

    def test_unknown_modality_raises(self, config):
        router = PerceptionRouter(config)
        with pytest.raises(ValueError):
            router.route(torch.randn(1, 10), "unknown")

    def test_available_modalities(self, config):
        router = create_default_perception(config)
        modalities = router.available_modalities()
        assert "vision" in modalities
        assert "time_series" in modalities
        assert "text" in modalities

    def test_default_perception_works(self, config):
        router = create_default_perception(config)
        # Vision
        img = torch.randn(1, 3, 64, 64)
        v_slots = router.route(img, "vision")
        assert v_slots.shape[-1] == config.latent_dim
        # Time series
        ts = torch.randn(1, 20, 1)
        t_slots = router.route(ts, "time_series")
        assert t_slots.shape[-1] == config.latent_dim
