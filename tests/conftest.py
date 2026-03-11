"""Shared test fixtures for HHCRA tests."""

import pytest
import torch
import numpy as np

from hhcra.config import HHCRAConfig
from hhcra.architecture import HHCRA
from hhcra.main import generate_causal_data


@pytest.fixture
def config():
    """Standard test configuration with reduced epochs for speed."""
    return HHCRAConfig(
        obs_dim=48, latent_dim=10, num_vars=8,
        mask_ratio=0.3, slot_attention_iters=3,
        gnn_lr=0.05, gnn_l1_penalty=0.02, gnn_dag_penalty=0.5,
        edge_threshold=0.35,
        liquid_ode_steps=8, liquid_dt=0.05,
        hrm_max_steps=30, hrm_patience=4, hrm_momentum=0.9,
        train_epochs_l1=5, train_epochs_l2=10, train_epochs_l3=5,
    )


@pytest.fixture
def observations_and_gt():
    """Generate synthetic causal data."""
    return generate_causal_data(B=4, T=8, obs_dim=48, seed=42)


@pytest.fixture
def observations(observations_and_gt):
    return observations_and_gt[0]


@pytest.fixture
def ground_truth(observations_and_gt):
    return observations_and_gt[1]


@pytest.fixture
def trained_model(config, observations):
    """A trained HHCRA model for testing."""
    torch.manual_seed(42)
    model = HHCRA(config)
    model.train_all(observations, verbose=False)
    model.eval()
    return model
