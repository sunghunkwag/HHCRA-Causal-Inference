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
    edge_threshold: float = 0.35
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
