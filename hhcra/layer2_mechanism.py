"""
Layer 2: Mechanism Layer — GNN + Liquid Neural Network (Tightly Coupled)

GNN learns WHAT causal structure exists (directed graph).
Liquid Net learns HOW each relationship works dynamically (ODE mechanisms).

Phase 2: NOTEARS continuous optimization for structure learning.
Phase 3: Neural ODE with adaptive stepping for mechanism modeling.

Interface IN  <- Layer 1: latent_vars (B, T, N, D)
Interface OUT -> Layer 3: CausalGraphData + trajectories
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple
from scipy.linalg import expm as scipy_expm

from hhcra.config import HHCRAConfig
from hhcra.causal_graph import CausalGraphData


class CausalGNN(nn.Module):
    """
    Learns directed causal graph structure via NOTEARS continuous optimization.

    NOTEARS (Zheng et al. 2018): formulates structure learning as:
        min F(W) + lambda * ||W||_1
        s.t. h(W) = tr(e^(W o W)) - d = 0

    Uses augmented Lagrangian method:
        L(W, alpha, rho) = F(W) + lambda*||W||_1 + alpha*h(W) + (rho/2)*h(W)^2
    """

    def __init__(self, config: HHCRAConfig):
        super().__init__()
        self.config = config
        N = config.num_vars
        D = config.latent_dim

        # Edge weight matrix W[i,j] represents influence of j on i
        self.W = nn.Parameter(torch.zeros(N, N))
        # Mask out self-loops
        self.register_buffer('diag_mask', 1.0 - torch.eye(N))

        # Per-edge MLP for Granger scoring (differentiable)
        self.edge_net = nn.Sequential(
            nn.Linear(D * 2, D),
            nn.Tanh(),
            nn.Linear(D, 1),
        )

        # Message passing network
        self.msg_net = nn.Sequential(
            nn.Linear(D, D),
            nn.Tanh(),
        )
        self.update_net = nn.Sequential(
            nn.Linear(D * 2, D),
            nn.Tanh(),
        )

        # NOTEARS augmented Lagrangian state
        self.notears_alpha = 0.0
        self.notears_rho = config.notears_rho

    def adjacency(self, hard: bool = False) -> torch.Tensor:
        """Sigmoid-activated adjacency with self-loop mask."""
        A = torch.sigmoid(self.W) * self.diag_mask
        if hard:
            A = (A > self.config.edge_threshold).float()
        return A

    def _h_dag(self, W: torch.Tensor) -> torch.Tensor:
        """
        DAG constraint: h(W) = tr(e^(W o W)) - d
        Must equal 0 for W to encode a DAG.
        """
        N = W.shape[0]
        A = torch.sigmoid(W) * self.diag_mask
        M = A * A  # Element-wise square (Hadamard)
        # Matrix exponential via eigendecomposition for gradient flow
        # Use series approximation for small matrices
        E = torch.matrix_exp(M)
        h = torch.trace(E) - N
        return h

    def dag_penalty(self) -> torch.Tensor:
        """Continuous DAG penalty for loss function."""
        return self._h_dag(self.W)

    def notears_loss(self, latent: torch.Tensor) -> torch.Tensor:
        """
        NOTEARS augmented Lagrangian loss:
        L = F(W) + lambda*||W||_1 + alpha*h(W) + (rho/2)*h(W)^2

        v0.4.1 FIX: F(W) now uses the standard NOTEARS formulation:
          F(W) = 0.5/n * sum_d ||X_next[:,:,d] - X_curr[:,:,d] @ A||^2
        computed per latent dimension d, preserving per-variable variance
        information critical for structure identification.

        Previous issue: edge_net collapsed latent dim to scalar via mean,
        destroying the per-variable signal that distinguishes causal from
        spurious edges (Zheng et al. 2018 Eq. 2).
        """
        B, T, N, D = latent.shape
        if T < 2:
            return torch.tensor(0.0, device=latent.device)

        A = self.adjacency()

        # Standard NOTEARS fitting loss on temporal data:
        # For each latent dimension d:
        #   pred[:, i, d] = sum_j A[i,j] * X_curr[:, j, d]
        x_curr = latent[:, :-1, :, :]  # (B, T-1, N, D)
        x_next = latent[:, 1:, :, :]   # (B, T-1, N, D)

        n_samples = B * (T - 1)
        xc = x_curr.reshape(n_samples, N, D)
        xn = x_next.reshape(n_samples, N, D)

        # Predicted via adjacency: pred[:, i, d] = sum_j A[i,j] * xc[:, j, d]
        pred = torch.einsum('ij,njd->nid', A, xc)  # (n, N, D)

        # Least-squares: 0.5/n * ||xn - pred||^2
        fitting_loss = 0.5 * torch.mean((xn - pred) ** 2)

        # L1 sparsity
        l1_loss = self.config.notears_lambda * torch.abs(A).sum()

        # DAG constraint via augmented Lagrangian
        h = self._h_dag(self.W)
        dag_loss = self.notears_alpha * h + (self.notears_rho / 2.0) * h * h

        return fitting_loss + l1_loss + dag_loss

    def update_lagrangian(self):
        """Update NOTEARS augmented Lagrangian multipliers."""
        with torch.no_grad():
            h = self._h_dag(self.W).item()
            self.notears_alpha += self.notears_rho * h
            if h > 0.25 * getattr(self, '_prev_h', 1e10):
                self.notears_rho = min(
                    self.notears_rho * 10.0,
                    self.config.notears_rho_max
                )
            self._prev_h = h
        return h

    def _is_dag(self, A: torch.Tensor) -> bool:
        """Check if binary adjacency is a DAG via topological sort."""
        A_np = A.detach().cpu().numpy()
        N = A_np.shape[0]
        in_degree = A_np.sum(axis=1).astype(int)
        queue = [i for i in range(N) if in_degree[i] == 0]
        visited = 0
        while queue:
            node = queue.pop(0)
            visited += 1
            for child in range(N):
                if A_np[child, node] > 0:
                    in_degree[child] -= 1
                    if in_degree[child] == 0:
                        queue.append(child)
        return visited == N

    def prune_to_dag(self):
        """Post-training: greedily remove weakest edges until DAG."""
        with torch.no_grad():
            A = self.adjacency(hard=True)
            if self._is_dag(A):
                return

            As = self.adjacency(hard=False)
            N = self.config.num_vars
            edges = []
            for i in range(N):
                for j in range(N):
                    if A[i, j] > 0:
                        edges.append((As[i, j].item(), i, j))
            edges.sort()

            for weight, i, j in edges:
                self.W.data[i, j] = -5.0
                A_test = self.adjacency(hard=True)
                if self._is_dag(A_test):
                    return

    def message_pass(self, latent: torch.Tensor) -> torch.Tensor:
        """Directed message passing: aggregate parent information."""
        B, T, N, D = latent.shape
        A = self.adjacency()
        out = latent

        for _ in range(2):  # 2 rounds
            # Messages from all nodes
            msg = self.msg_net(out.reshape(-1, D)).reshape(B, T, N, D)
            # Weighted parent aggregation
            parent_msg = torch.einsum('ij,btjd->btid', A, msg)
            # Normalize by parent count
            parent_count = A.sum(dim=1, keepdim=True).clamp(min=1e-8)  # (N, 1)
            parent_msg = parent_msg / parent_count.unsqueeze(0).unsqueeze(0)
            # Update with residual
            cat_input = torch.cat([out, parent_msg], dim=-1)
            update = self.update_net(
                cat_input.reshape(-1, D * 2)
            ).reshape(B, T, N, D)
            out = 0.7 * out + 0.3 * update

        return out

    def compute_shd(self, true_adj: np.ndarray) -> int:
        """Structural Hamming Distance against ground truth."""
        with torch.no_grad():
            pred = self.adjacency(hard=True).cpu().numpy()
        N_true = true_adj.shape[0]
        N_pred = pred.shape[0]
        N = min(N_true, N_pred)
        diff = 0
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                if pred[i, j] != true_adj[i, j]:
                    diff += 1
        # Add penalty for extra nodes
        if N_pred > N_true:
            extra = pred[N_true:, :].sum() + pred[:, N_true:].sum()
            diff += int(extra)
        return diff

    def compute_metrics(self, true_adj: np.ndarray) -> dict:
        """Compute SHD, TPR, FDR against ground truth."""
        with torch.no_grad():
            pred = self.adjacency(hard=True).cpu().numpy()
        N = min(true_adj.shape[0], pred.shape[0])

        true_flat = true_adj[:N, :N].flatten()
        pred_flat = pred[:N, :N].flatten()

        tp = np.sum((pred_flat == 1) & (true_flat == 1))
        fp = np.sum((pred_flat == 1) & (true_flat == 0))
        fn = np.sum((pred_flat == 0) & (true_flat == 1))

        tpr = tp / max(tp + fn, 1)
        fdr = fp / max(tp + fp, 1)
        shd = self.compute_shd(true_adj)

        return {'shd': shd, 'tpr': tpr, 'fdr': fdr}


class LiquidNeuralNet(nn.Module):
    """
    Liquid Time-Constant Neural Network with Neural ODE integration.

    Each variable has a Liquid neuron governed by:
        dx_i/dt = f_i(x_j : A[i,j]>0)

    with learned f_i per variable, integrated via RK4 or adaptive methods.

    Intervention do(X_i=x) cuts row i of adjacency, clamps x_i,
    and re-integrates remaining ODEs.
    """

    def __init__(self, config: HHCRAConfig):
        super().__init__()
        self.config = config
        N = config.num_vars
        D = config.latent_dim

        # Per-variable ODE dynamics networks
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

    def _ode_fn(self, state: torch.Tensor, parent_input: torch.Tensor,
                var_idx: int) -> torch.Tensor:
        """
        Compute dx/dt for variable var_idx.

        dx/dt = gate * (-x + f(x, I)) / tau
        """
        cat = torch.cat([parent_input, state], dim=-1)
        tau = torch.sigmoid(self.tau_nets[var_idx](cat)) + 0.1
        f = self.f_nets[var_idx](cat) + self.biases[var_idx]
        gate = torch.sigmoid(self.gate_nets[var_idx](cat))
        dx = gate * (-state + f) / tau
        return dx

    def _rk4_step(self, state: torch.Tensor, parent_input: torch.Tensor,
                  var_idx: int, dt: float) -> torch.Tensor:
        """4th-order Runge-Kutta integration step."""
        k1 = self._ode_fn(state, parent_input, var_idx)
        k2 = self._ode_fn(state + 0.5 * dt * k1, parent_input, var_idx)
        k3 = self._ode_fn(state + 0.5 * dt * k2, parent_input, var_idx)
        k4 = self._ode_fn(state + dt * k3, parent_input, var_idx)
        new_state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return new_state.clamp(-5.0, 5.0)

    def _euler_step(self, state: torch.Tensor, parent_input: torch.Tensor,
                    var_idx: int, dt: float) -> torch.Tensor:
        """Simple Euler integration step."""
        dx = self._ode_fn(state, parent_input, var_idx)
        new_state = state + dt * dx
        return new_state.clamp(-5.0, 5.0)

    def _dopri5_step(self, state: torch.Tensor, parent_input: torch.Tensor,
                     var_idx: int, dt: float) -> torch.Tensor:
        """Dormand-Prince 5th order integration step."""
        # Butcher tableau coefficients
        k1 = self._ode_fn(state, parent_input, var_idx)

        k2 = self._ode_fn(state + dt * (1/5) * k1, parent_input, var_idx)

        k3 = self._ode_fn(
            state + dt * (3/40 * k1 + 9/40 * k2), parent_input, var_idx)

        k4 = self._ode_fn(
            state + dt * (44/45 * k1 - 56/15 * k2 + 32/9 * k3),
            parent_input, var_idx)

        k5 = self._ode_fn(
            state + dt * (19372/6561 * k1 - 25360/2187 * k2
                          + 64448/6561 * k3 - 212/729 * k4),
            parent_input, var_idx)

        k6 = self._ode_fn(
            state + dt * (9017/3168 * k1 - 355/33 * k2 + 46732/5247 * k3
                          + 49/176 * k4 - 5103/18656 * k5),
            parent_input, var_idx)

        # 5th order solution
        new_state = state + dt * (
            35/384 * k1 + 500/1113 * k3 + 125/192 * k4
            - 2187/6784 * k5 + 11/84 * k6
        )
        return new_state.clamp(-5.0, 5.0)

    def _integrate_step(self, state: torch.Tensor, parent_input: torch.Tensor,
                        var_idx: int, dt: float) -> torch.Tensor:
        """Integration step using configured method."""
        if self.config.liquid_method == "rk4":
            return self._rk4_step(state, parent_input, var_idx, dt)
        elif self.config.liquid_method == "dopri5":
            return self._dopri5_step(state, parent_input, var_idx, dt)
        else:
            return self._euler_step(state, parent_input, var_idx, dt)

    def _aggregate_parents(self, embeddings_t: torch.Tensor,
                           states: list, adjacency: torch.Tensor,
                           var_idx: int) -> torch.Tensor:
        """Aggregate parent inputs for variable var_idx."""
        B, N, D = embeddings_t.shape
        parent_in = torch.zeros(B, D, device=embeddings_t.device)
        weights = adjacency[var_idx, :]  # (N,)
        total_w = 0.0

        for j in range(N):
            if j != var_idx and weights[j] > 0.01:
                w = weights[j]
                parent_in = parent_in + w * (
                    0.5 * embeddings_t[:, j, :] + 0.5 * states[j]
                )
                total_w += w.item()

        if total_w > 0:
            parent_in = parent_in / total_w
        else:
            parent_in = embeddings_t[:, var_idx, :] * 0.1

        return parent_in

    def evolve(self, embeddings: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """
        Run ODE integration on the causal graph.

        Args:
            embeddings: (B, T, N, D) graph-contextualized latent vars
            adjacency: (N, N) causal adjacency matrix (torch tensor)

        Returns:
            trajectories: (B, T, N, D) evolved state trajectories
        """
        B, T, N, D = embeddings.shape
        dt = self.config.liquid_dt
        steps = self.config.liquid_ode_steps
        device = embeddings.device

        states = [torch.zeros(B, D, device=device) for _ in range(N)]
        all_trajectories = []

        for t in range(T):
            for _ in range(steps):
                new_states = []
                for i in range(N):
                    parent_in = self._aggregate_parents(
                        embeddings[:, t, :, :], states, adjacency, i)
                    new_state = self._integrate_step(states[i], parent_in, i, dt)
                    new_states.append(new_state)
                states = new_states

            step_traj = torch.stack(states, dim=1)  # (B, N, D)
            all_trajectories.append(step_traj)

        trajectories = torch.stack(all_trajectories, dim=1)  # (B, T, N, D)
        return trajectories

    def intervene(self, embeddings: torch.Tensor, adjacency: torch.Tensor,
                  interventions: Dict[int, torch.Tensor]) -> torch.Tensor:
        """
        Perform do(X_i = x):
        1. Cut all incoming edges to intervened variable (row i = 0)
        2. Clamp its value to the intervention value
        3. Re-integrate remaining ODEs

        This implements Pearl's Rung 2 intervention.
        """
        mod_adj = adjacency.clone()
        mod_emb = embeddings.clone()

        for idx, val in interventions.items():
            mod_adj[idx, :] = 0.0  # Cut incoming edges
            if val.dim() == 1:
                mod_emb[:, :, idx, :] = val.unsqueeze(0).unsqueeze(0)
            else:
                mod_emb[:, :, idx, :] = val.unsqueeze(1)

        return self.evolve(mod_emb, mod_adj)


class MechanismLayer(nn.Module):
    """
    Layer 2: GNN + Liquid Net tightly coupled.

    GNN learns WHAT causal structure exists.
    Liquid Net learns HOW each relationship works dynamically.
    Tight coupling: shared gradients within this layer.

    Interface IN  <- Layer 1: latent_vars (B, T, N, D)
    Interface OUT -> Layer 3: CausalGraphData + trajectories
    """

    def __init__(self, config: HHCRAConfig):
        super().__init__()
        self.config = config
        self.gnn = CausalGNN(config)
        self.liquid = LiquidNeuralNet(config)

    def forward(self, latent: torch.Tensor) -> dict:
        """Joint forward pass: structure learning + mechanism evolution."""
        embeddings = self.gnn.message_pass(latent)
        adjacency = self.gnn.adjacency()
        trajectories = self.liquid.evolve(embeddings, adjacency)

        return {
            'embeddings': embeddings,
            'adjacency': adjacency,
            'trajectories': trajectories,
        }

    def compute_loss(self, latent: torch.Tensor) -> torch.Tensor:
        """Combined NOTEARS + mechanism fitting loss."""
        structure_loss = self.gnn.notears_loss(latent)

        # Mechanism loss: trajectory should predict next-step latent
        l2_out = self.forward(latent)
        traj = l2_out['trajectories']
        if latent.shape[1] > 1:
            mech_loss = F.mse_loss(traj[:, :-1, :, :], latent[:, 1:, :, :])
        else:
            mech_loss = torch.tensor(0.0, device=latent.device)

        return structure_loss + 0.1 * mech_loss

    def symbolic_graph(self) -> CausalGraphData:
        """Convert continuous adjacency to symbolic graph for Layer 3."""
        self.gnn.prune_to_dag()
        with torch.no_grad():
            A = self.gnn.adjacency(hard=True).cpu().numpy()
            As = self.gnn.adjacency(hard=False).cpu().numpy()
        N = self.config.num_vars
        edges = []
        for i in range(N):
            for j in range(N):
                if A[i, j] > 0:
                    edges.append((j, i, float(As[i, j])))
        return CausalGraphData(list(range(N)), edges, A)

    def handle_feedback(self, feedback: dict):
        """Handle feedback from Layer 3."""
        with torch.no_grad():
            if 'remove_edge' in feedback:
                i, j = feedback['remove_edge']
                self.gnn.W.data[i, j] = -5.0
            if 'add_edge' in feedback:
                i, j = feedback['add_edge']
                self.gnn.W.data[i, j] = 2.0
            if 'weaken_edge' in feedback:
                i, j = feedback['weaken_edge']
                self.gnn.W.data[i, j] -= 1.0
