"""
Phase 9: Multi-Modal Perception Interface

Extracts causal variables from raw sensory input — not just synthetic data.
Routes different input modalities to appropriate perception modules.

Interface: raw observation -> latent_vars compatible with Layer 1.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Tuple, Dict
from abc import ABC, abstractmethod

from hhcra.config import HHCRAConfig
from hhcra.causal_graph import CausalGraphData


class PerceptionInterface(ABC):
    """Abstract base class for perception modules."""

    @abstractmethod
    def encode(self, raw_input: torch.Tensor) -> torch.Tensor:
        """
        Encode raw input to latent variables compatible with Layer 1.

        Returns: (B, T, num_vars, latent_dim) or (B, num_vars, latent_dim)
        """
        pass

    @abstractmethod
    def input_type(self) -> str:
        """Return the type of input this module handles."""
        pass


class VisionPerception(nn.Module, PerceptionInterface):
    """
    Vision perception: extract object-centric representations from images.

    Uses a CNN encoder to extract spatial features, then slot attention
    to decompose into object slots compatible with C-JEPA latent format.
    """

    def __init__(self, config: HHCRAConfig, image_channels: int = 3,
                 image_size: int = 64):
        super().__init__()
        self.config = config
        self.image_size = image_size

        # CNN encoder for spatial features
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 4, stride=2, padding=1),
            nn.ReLU(),
        )

        # Compute flattened size
        with torch.no_grad():
            dummy = torch.zeros(1, image_channels, image_size, image_size)
            feat = self.encoder(dummy)
            self.feat_dim = feat.flatten(1).shape[1]

        # Project to slot-compatible representation
        self.project = nn.Linear(self.feat_dim, config.num_vars * config.latent_dim)

    def encode(self, raw_input: torch.Tensor) -> torch.Tensor:
        """
        Encode images to latent slots.

        Args:
            raw_input: (B, C, H, W) or (B, T, C, H, W) images

        Returns:
            (B, [T,] num_vars, latent_dim) latent slots
        """
        has_time = raw_input.dim() == 5
        if has_time:
            B, T, C, H, W = raw_input.shape
            x = raw_input.reshape(B * T, C, H, W)
        else:
            B = raw_input.shape[0]
            T = 1
            x = raw_input

        features = self.encoder(x).flatten(1)  # (B*T, feat_dim)
        slots = self.project(features)  # (B*T, N*D)
        slots = slots.reshape(-1, self.config.num_vars, self.config.latent_dim)
        slots = F.normalize(slots, dim=-1)

        if has_time:
            slots = slots.reshape(B, T, self.config.num_vars, self.config.latent_dim)

        return slots

    def input_type(self) -> str:
        return "vision"


class TimeSeriesPerception(nn.Module, PerceptionInterface):
    """
    Time series perception: encode sensor data to causal variable slots.

    Uses sliding window + 1D convolution to extract temporal features,
    then maps to C-JEPA compatible latent format.
    """

    def __init__(self, config: HHCRAConfig, input_channels: int = 1,
                 window_size: int = 16):
        super().__init__()
        self.config = config
        self.window_size = window_size

        # 1D CNN for temporal feature extraction
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.project = nn.Linear(64, config.num_vars * config.latent_dim)

    def encode(self, raw_input: torch.Tensor) -> torch.Tensor:
        """
        Encode time series to latent slots.

        Args:
            raw_input: (B, T, channels) time series data

        Returns:
            (B, num_vars, latent_dim) latent slots
        """
        B, T, C = raw_input.shape
        x = raw_input.permute(0, 2, 1)  # (B, C, T)
        features = self.encoder(x).squeeze(-1)  # (B, 64)
        slots = self.project(features)
        slots = slots.reshape(B, self.config.num_vars, self.config.latent_dim)
        return F.normalize(slots, dim=-1)

    def input_type(self) -> str:
        return "time_series"


class TextPerception(nn.Module, PerceptionInterface):
    """
    Text perception: extract causal claims from natural language.

    Maps text to CausalGraphData directly (can bypass Layer 1-2).
    Enables: "smoking causes cancer" -> edge(smoking, cancer)

    Uses a simple bag-of-words encoder for demonstration.
    """

    def __init__(self, config: HHCRAConfig, vocab_size: int = 1000,
                 embed_dim: int = 64):
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, config.num_vars * config.latent_dim),
        )

        # Causal relation extractor
        self.relation_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, config.num_vars * config.num_vars),
        )

    def encode(self, raw_input: torch.Tensor) -> torch.Tensor:
        """
        Encode tokenized text to latent slots.

        Args:
            raw_input: (B, seq_len) token indices

        Returns:
            (B, num_vars, latent_dim) latent slots
        """
        B = raw_input.shape[0]
        embedded = self.embedding(raw_input)  # (B, seq_len, embed_dim)
        pooled = embedded.mean(dim=1)  # (B, embed_dim)

        slots = self.encoder(pooled)
        slots = slots.reshape(B, self.config.num_vars, self.config.latent_dim)
        return F.normalize(slots, dim=-1)

    def extract_causal_claims(self, raw_input: torch.Tensor) -> torch.Tensor:
        """
        Extract causal graph structure from text.

        Returns: (B, num_vars, num_vars) adjacency predictions
        """
        embedded = self.embedding(raw_input).mean(dim=1)
        adj_logits = self.relation_head(embedded)
        adj = torch.sigmoid(adj_logits.reshape(-1, self.config.num_vars, self.config.num_vars))
        # Remove self-loops
        mask = 1.0 - torch.eye(self.config.num_vars, device=adj.device)
        return adj * mask

    def input_type(self) -> str:
        return "text"


class PerceptionRouter(nn.Module):
    """Routes input to correct perception module based on modality."""

    def __init__(self, config: HHCRAConfig):
        super().__init__()
        self.config = config
        self.modules_dict: Dict[str, PerceptionInterface] = {}

    def register_module_type(self, name: str, module: PerceptionInterface):
        """Register a perception module for a given input type."""
        self.modules_dict[name] = module
        if isinstance(module, nn.Module):
            self.add_module(f"perception_{name}", module)

    def route(self, raw_input: torch.Tensor,
              modality: str) -> torch.Tensor:
        """Route input to appropriate perception module."""
        if modality not in self.modules_dict:
            raise ValueError(f"Unknown modality: {modality}. "
                             f"Available: {list(self.modules_dict.keys())}")
        return self.modules_dict[modality].encode(raw_input)

    def available_modalities(self) -> List[str]:
        return list(self.modules_dict.keys())


def create_default_perception(config: HHCRAConfig) -> PerceptionRouter:
    """Create a PerceptionRouter with all default perception modules."""
    router = PerceptionRouter(config)
    router.register_module_type("vision", VisionPerception(config))
    router.register_module_type("time_series", TimeSeriesPerception(config))
    router.register_module_type("text", TextPerception(config))
    return router
