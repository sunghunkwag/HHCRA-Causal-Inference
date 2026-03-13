"""
HHCRAConfig: Master configuration for the complete architecture.
"""

from dataclasses import dataclass


@dataclass
class HHCRAConfig:
    """Master configuration for the HHCRA architecture."""
    # Observation space
    obs_dim: int = 48
    num_true_vars: int = 5

    # Architecture
    latent_dim: int = 10
    num_vars: int = 8

    # Layer 1: C-JEPA
    mask_ratio: float = 0.3
    slot_attention_iters: int = 3
    cjepa_lr: float = 0.001

    # Layer 2: GNN (NOTEARS)
    gnn_lr: float = 0.05
    gnn_l1_penalty: float = 0.02
    gnn_dag_penalty: float = 0.5
    edge_threshold: float = 0.30  # v0.6.0: lowered from 0.35 (adaptive threshold overrides)
    notears_lambda: float = 0.01
    notears_rho: float = 1.0
    notears_rho_max: float = 1e16
    notears_h_tol: float = 1e-8
    notears_max_iter: int = 50

    # Layer 2: Liquid Net
    liquid_ode_steps: int = 8
    liquid_dt: float = 0.05
    liquid_method: str = "rk4"  # "euler", "rk4", "dopri5"

    # Layer 3: HRM
    hrm_max_steps: int = 30
    hrm_patience: int = 4
    hrm_momentum: float = 0.9
    hrm_convergence_threshold: float = 0.01
    hrm_hidden_dim: int = 20
    hrm_update_interval: int = 3  # H-module updates every K steps

    # Training
    train_epochs_l1: int = 15
    train_epochs_l2: int = 30
    train_epochs_l3: int = 10
    layer2_lr: float = 0.001
    layer3_lr: float = 0.001

    # Device
    device: str = "cpu"

    def __post_init__(self):
        """Validate critical hyperparameters at construction time."""
        assert self.num_vars > 0, f"num_vars must be positive, got {self.num_vars}"
        assert self.latent_dim > 0, f"latent_dim must be positive, got {self.latent_dim}"
        assert self.obs_dim > 0, f"obs_dim must be positive, got {self.obs_dim}"
        assert self.notears_rho > 0, f"notears_rho must be positive, got {self.notears_rho}"
        assert self.liquid_dt > 0, f"liquid_dt must be positive, got {self.liquid_dt}"
        assert self.liquid_ode_steps > 0, (
            f"liquid_ode_steps must be positive, got {self.liquid_ode_steps}"
        )
        assert 0.0 < self.edge_threshold < 1.0, (
            f"edge_threshold must be in (0, 1), got {self.edge_threshold}"
        )
        assert 0.0 < self.mask_ratio < 1.0, (
            f"mask_ratio must be in (0, 1), got {self.mask_ratio}"
        )
        assert self.liquid_method in ("euler", "rk4", "dopri5"), (
            f"liquid_method must be one of 'euler', 'rk4', 'dopri5', got '{self.liquid_method}'"
        )
        assert self.hrm_hidden_dim > 0, (
            f"hrm_hidden_dim must be positive, got {self.hrm_hidden_dim}"
        )
        assert self.hrm_max_steps > 0, (
            f"hrm_max_steps must be positive, got {self.hrm_max_steps}"
        )
        assert self.hrm_update_interval > 0, (
            f"hrm_update_interval must be positive, got {self.hrm_update_interval}"
        )
        assert 0.0 <= self.hrm_momentum < 1.0, (
            f"hrm_momentum must be in [0, 1), got {self.hrm_momentum}"
        )
        assert self.train_epochs_l1 > 0, (
            f"train_epochs_l1 must be positive, got {self.train_epochs_l1}"
        )
        assert self.train_epochs_l2 > 0, (
            f"train_epochs_l2 must be positive, got {self.train_epochs_l2}"
        )
        assert self.train_epochs_l3 > 0, (
            f"train_epochs_l3 must be positive, got {self.train_epochs_l3}"
        )
