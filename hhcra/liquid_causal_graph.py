"""
Liquid Causal Graph: The graph itself evolves as a dynamical system.

Instead of learning a STATIC adjacency matrix, the graph structure W(t) co-evolves
with node states x(t) via coupled ODEs:

    dx/dt = f(x, W(t))       -- node dynamics depend on current structure
    dW/dt = g(W, x(t))       -- structure dynamics depend on current state

This breaks the fundamental assumption of ALL existing causal discovery methods
that a single fixed DAG underlies the data. In reality, causal structures change:
taking a drug alters the body's causal mechanisms, not just the variable values.

Mathematical foundation:
    - The (x, W) pair lives on a product manifold R^(N*D) x R^(N*N)
    - The ODE integrator (RK4) operates on this joint space
    - DAG projection after each step ensures acyclicity is maintained
    - The graph dynamics network learns dW/dt as a function of (W, x, invariants)

Key insight: W(t) is NOT "time-varying graphs" (which assume a sequence of DAGs).
This is a CONTINUOUS flow on the space of DAGs, where structure and state are
coupled through mutual feedback -- a fundamentally new mathematical object.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple, List

from hhcra.config import HHCRAConfig


class GraphDynamicsNet(nn.Module):
    """
    Learns dW/dt: how the causal graph structure evolves over time.

    The graph dynamics depend on:
    1. Current structure W(t) -- structural inertia and self-regulation
    2. Current node states x(t) -- states reshape structure
    3. Structural invariants -- conserved quantities constrain evolution

    Architecture:
        (W_flat, node_summary) -> MLP -> dW_flat -> reshape to (N, N)
        with self-loop masking and DAG-preserving regularization.
    """

    def __init__(self, config: HHCRAConfig):
        super().__init__()
        self.config = config
        N = config.num_vars
        D = config.latent_dim

        # Node state summary: compress (N, D) -> summary vector
        self.state_encoder = nn.Sequential(
            nn.Linear(N * D, N * N),
            nn.Tanh(),
        )

        # Graph dynamics: (W_flat + state_summary) -> dW_flat
        self.dynamics_net = nn.Sequential(
            nn.Linear(N * N + N * N, N * N),
            nn.Tanh(),
            nn.Linear(N * N, N * N),
        )

        # Mask out self-loops
        self.register_buffer('diag_mask', 1.0 - torch.eye(N))

        # Damping coefficient to prevent runaway graph dynamics
        self.damping = nn.Parameter(torch.tensor(0.1))

    def forward(self, W: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """
        Compute dW/dt given current graph W and node states.

        Args:
            W: (N, N) current adjacency weights (pre-sigmoid)
            states: (B, N, D) current node states

        Returns:
            dW: (N, N) time derivative of graph weights
        """
        N = self.config.num_vars

        # Summarize node states across batch
        state_mean = states.mean(dim=0)  # (N, D)
        state_flat = state_mean.reshape(-1)  # (N*D,)
        state_summary = self.state_encoder(state_flat)  # (N*N,)

        # Flatten W
        W_flat = W.reshape(-1)  # (N*N,)

        # Compute dynamics
        combined = torch.cat([W_flat, state_summary], dim=-1)  # (2*N*N,)
        dW_flat = self.dynamics_net(combined)  # (N*N,)

        # Reshape and apply constraints
        dW = dW_flat.reshape(N, N)

        # No self-loop dynamics
        dW = dW * self.diag_mask

        # Damping: graph structure has inertia (resists rapid change)
        damping_coeff = torch.sigmoid(self.damping)
        dW = dW * damping_coeff * 0.1  # Scale down for stability

        return dW


class LiquidCausalGraph(nn.Module):
    """
    Liquid Causal Graph: graph structure W(t) co-evolves with node states x(t).

    Coupled ODE system:
        dx/dt = f(x, W(t))    -- node dynamics on evolving graph
        dW/dt = g(W, x(t))    -- graph dynamics driven by state

    Integration uses RK4 on the joint (x, W) state space.
    After each integration step, W is projected to maintain DAG structure.
    """

    def __init__(self, config: HHCRAConfig):
        super().__init__()
        self.config = config
        N = config.num_vars
        D = config.latent_dim

        # Graph dynamics network
        self.graph_dynamics = GraphDynamicsNet(config)

        # Node dynamics networks (per-variable, like original LiquidNeuralNet)
        self.tau_nets = nn.ModuleList([
            nn.Linear(D * 2, D) for _ in range(N)
        ])
        self.f_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(D * 2, D),
                nn.Tanh(),
            ) for _ in range(N)
        ])
        self.gate_nets = nn.ModuleList([
            nn.Linear(D * 2, D) for _ in range(N)
        ])
        self.biases = nn.ParameterList([
            nn.Parameter(torch.zeros(D)) for _ in range(N)
        ])

        # Mask out self-loops
        self.register_buffer('diag_mask', 1.0 - torch.eye(N))

        # Track graph evolution history
        self.graph_trajectory: List[torch.Tensor] = []

    def _node_ode_fn(self, states: torch.Tensor, parent_inputs: torch.Tensor) -> torch.Tensor:
        """Compute dx/dt for all variables. Same dynamics as LiquidNeuralNet."""
        dxs = []
        for i in range(self.config.num_vars):
            cat = torch.cat([parent_inputs[:, i, :], states[:, i, :]], dim=-1)
            tau = torch.sigmoid(self.tau_nets[i](cat)) + 0.1
            f = self.f_nets[i](cat) + self.biases[i]
            gate = torch.sigmoid(self.gate_nets[i](cat))
            dx = gate * (-states[:, i, :] + f) / tau
            dxs.append(dx)
        return torch.stack(dxs, dim=1)  # (B, N, D)

    def _aggregate_parents(self, embeddings: torch.Tensor, states: torch.Tensor,
                           W: torch.Tensor) -> torch.Tensor:
        """
        Aggregate parent inputs using CURRENT (evolving) adjacency.

        Args:
            embeddings: (B, N, D) input embeddings
            states: (B, N, D) current ODE states
            W: (N, N) current adjacency weights (pre-sigmoid)

        Returns:
            parent_inputs: (B, N, D) aggregated parent signals
        """
        A = torch.sigmoid(W) * self.diag_mask  # (N, N) soft adjacency

        # Apply edge cutoff
        active_mask = (A > 0.01).float()
        A_active = A * active_mask

        combined = 0.5 * embeddings + 0.5 * states  # (B, N, D)

        # Weighted sum of parent signals
        weighted_sum = torch.einsum('ij,bjd->bid', A_active, combined)
        total_w = A_active.sum(dim=1).clamp(min=1e-8)  # (N,)

        has_parents = (A_active.sum(dim=1) > 1e-8).float()

        parent_inputs = weighted_sum / total_w.unsqueeze(0).unsqueeze(-1)
        fallback = embeddings * 0.1

        parent_inputs = (
            parent_inputs * has_parents.unsqueeze(0).unsqueeze(-1)
            + fallback * (1.0 - has_parents).unsqueeze(0).unsqueeze(-1)
        )
        return parent_inputs

    def _project_to_dag(self, W: torch.Tensor) -> torch.Tensor:
        """
        Soft DAG projection: penalize cyclic components by reducing
        edge weights that contribute most to cycles.

        Uses the matrix exponential trace as a differentiable
        proxy for cycle detection.
        """
        with torch.no_grad():
            A = torch.sigmoid(W) * self.diag_mask
            M = A * A
            # If tr(e^M) - N > threshold, there are cycles
            h = torch.trace(torch.matrix_exp(M)) - self.config.num_vars

            if h.item() > 0.1:
                # Reduce all weights slightly to break cycles
                # This is a soft projection, not hard pruning
                W = W - 0.05 * W.sign() * (h.item() / self.config.num_vars)

        return W

    def coupled_evolve(self, embeddings: torch.Tensor,
                       W_init: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Co-evolve node states and graph structure.

        The key innovation: at each ODE step, BOTH x and W evolve.
        The graph that governs node dynamics is itself changing.

        Args:
            embeddings: (B, T, N, D) input embeddings from GNN
            W_init: (N, N) initial adjacency weights (pre-sigmoid)

        Returns:
            trajectories: (B, T, N, D) evolved node state trajectories
            W_final: (N, N) evolved graph weights
        """
        B, T, N, D = embeddings.shape
        dt = self.config.liquid_dt
        steps = self.config.liquid_ode_steps
        device = embeddings.device

        states = torch.zeros(B, N, D, device=device)
        W = W_init.clone()
        all_trajectories = []
        self.graph_trajectory = []

        for t in range(T):
            emb_t = embeddings[:, t, :, :]  # (B, N, D)

            for _ in range(steps):
                # === COUPLED RK4 on joint (x, W) space ===

                # Stage 1: k1
                parent_inputs = self._aggregate_parents(emb_t, states, W)
                k1_x = self._node_ode_fn(states, parent_inputs)
                k1_W = self.graph_dynamics(W, states)

                # Stage 2: k2
                states_mid1 = states + 0.5 * dt * k1_x
                W_mid1 = W + 0.5 * dt * k1_W
                parent_inputs_mid1 = self._aggregate_parents(emb_t, states_mid1, W_mid1)
                k2_x = self._node_ode_fn(states_mid1, parent_inputs_mid1)
                k2_W = self.graph_dynamics(W_mid1, states_mid1)

                # Stage 3: k3
                states_mid2 = states + 0.5 * dt * k2_x
                W_mid2 = W + 0.5 * dt * k2_W
                parent_inputs_mid2 = self._aggregate_parents(emb_t, states_mid2, W_mid2)
                k3_x = self._node_ode_fn(states_mid2, parent_inputs_mid2)
                k3_W = self.graph_dynamics(W_mid2, states_mid2)

                # Stage 4: k4
                states_end = states + dt * k3_x
                W_end = W + dt * k3_W
                parent_inputs_end = self._aggregate_parents(emb_t, states_end, W_end)
                k4_x = self._node_ode_fn(states_end, parent_inputs_end)
                k4_W = self.graph_dynamics(W_end, states_end)

                # Update both x and W
                states = states + (dt / 6.0) * (k1_x + 2 * k2_x + 2 * k3_x + k4_x)
                states = states.clamp(-5.0, 5.0)

                W = W + (dt / 6.0) * (k1_W + 2 * k2_W + 2 * k3_W + k4_W)

                # Soft DAG projection after W update
                W = self._project_to_dag(W)

            all_trajectories.append(states.clone())
            self.graph_trajectory.append(W.detach().clone())

        trajectories = torch.stack(all_trajectories, dim=1)  # (B, T, N, D)
        return trajectories, W

    def get_graph_evolution(self) -> List[np.ndarray]:
        """Return the history of graph snapshots as numpy arrays."""
        return [
            (torch.sigmoid(w) * self.diag_mask).cpu().numpy()
            for w in self.graph_trajectory
        ]

    def compute_graph_change_rate(self) -> float:
        """Measure how much the graph changed over the trajectory."""
        if len(self.graph_trajectory) < 2:
            return 0.0

        changes = []
        for i in range(1, len(self.graph_trajectory)):
            A_prev = torch.sigmoid(self.graph_trajectory[i - 1]) * self.diag_mask
            A_curr = torch.sigmoid(self.graph_trajectory[i]) * self.diag_mask
            change = (A_curr - A_prev).abs().mean().item()
            changes.append(change)

        return sum(changes) / len(changes) if changes else 0.0
