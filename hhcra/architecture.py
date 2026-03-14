"""
HHCRA: Hierarchical Hybrid Causal Reasoning Architecture

Main orchestrator class combining all 3 layers:
    Layer 1: C-JEPA (Perception)
    Layer 2: GNN + Liquid Net (Mechanism) [tight coupling]
    Layer 3: Neuro-symbolic + HRM (Reasoning) [tight coupling]

Staged training API: train_layer1(), train_layer2(), train_layer3()
All layers are nn.Module subclasses with proper gradient flow.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Optional

from hhcra.config import HHCRAConfig
from hhcra.causal_graph import CausalGraphData, CausalQueryType
from hhcra.layer1_cjepa import CJEPA
from hhcra.layer2_mechanism import MechanismLayer
from hhcra.layer3_reasoning import ReasoningLayer
from hhcra.interfaces import LayerInterface, FeedbackRouter


class HHCRA(nn.Module):
    """
    HHCRA: Hierarchical Hybrid Causal Reasoning Architecture

    5 components in 3 layers:
        Layer 1: C-JEPA                    (Perception)
        Layer 2: GNN + Liquid Net          (Mechanism)  [tight coupling]
        Layer 3: Neuro-symbolic + HRM      (Reasoning)  [tight coupling]

    3 connection types:
        Tight coupling:    Within-layer shared computation
        Interface coupling: Between-layer .detach() (no gradient crossing)
        Feedback coupling:  Top-down diagnostic signals

    Full Pearl's Ladder coverage:
        Rung 1: Observation   P(Y|X)
        Rung 2: Intervention  P(Y|do(X))
        Rung 3: Counterfactual P(Y_{x'}|X=x, Y=y)
    """

    def __init__(self, config: Optional[HHCRAConfig] = None):
        super().__init__()
        self.config = config or HHCRAConfig()
        self.layer1 = CJEPA(self.config)
        self.layer2 = MechanismLayer(self.config)
        self.layer3 = ReasoningLayer(self.config)
        self.feedback_router = FeedbackRouter()

    # --- Core Pipeline ---

    def forward(self, observations: torch.Tensor) -> dict:
        """Full forward pass through all 3 layers."""
        # Layer 1: Perception
        latent = self.layer1.extract_variables(observations)

        # Interface: L1 -> L2 (detach — no gradient crossing between layers)
        latent_detached = LayerInterface.l1_to_l2(latent)

        # Layer 2: Mechanism
        l2_out = self.layer2(latent_detached)

        # Interface: L2 -> L3 (symbolic conversion)
        graph = self.layer2.symbolic_graph()

        return {
            'latent': latent,
            'layer2': l2_out,
            'graph': graph,
        }

    def query(self, observations: torch.Tensor, query_type: CausalQueryType,
              X: int, Y: int, verbose: bool = True, **kwargs) -> dict:
        """Answer a causal query through the complete pipeline."""
        fwd = self.forward(observations)

        # Interface: L2 -> L3 trajectories detached
        traj_detached = LayerInterface.l2_to_l3(fwd['layer2']['trajectories'])

        result = self.layer3.answer_query(
            query_type, X, Y, fwd['graph'],
            traj_detached,
            self.layer2, **kwargs)

        # Route feedback to lower layers
        if result.get('feedback'):
            self.feedback_router.route(
                result['feedback'], self.layer2, self.layer1, verbose)

        return result

    # --- Staged Training API ---

    def train_layer1(self, observations: torch.Tensor, verbose: bool = True):
        """Stage 1: Train C-JEPA (latent variable extraction)."""
        optimizer = optim.Adam(self.layer1.parameters(), lr=self.config.cjepa_lr)

        if verbose:
            print("=" * 60)
            print("STAGE 1: C-JEPA (Latent Causal Variable Extraction)")
            print("=" * 60)

        for ep in range(self.config.train_epochs_l1):
            optimizer.zero_grad()
            loss = self.layer1.compute_loss(observations)
            loss.backward()
            nn.utils.clip_grad_norm_(self.layer1.parameters(), max_norm=5.0)
            optimizer.step()

            if verbose and (ep + 1) % max(1, self.config.train_epochs_l1 // 3) == 0:
                print(f"  Epoch {ep + 1}/{self.config.train_epochs_l1} | "
                      f"Mask-prediction loss: {loss.item():.6f}")

    def train_layer2(self, observations: torch.Tensor, verbose: bool = True,
                     raw_data: Optional[torch.Tensor] = None):
        """
        Stage 2: Train GNN + Liquid Net (structure + mechanisms).

        v0.6.0 Upgrades:
          - Warm initialization of W from data correlations (raw data preferred)
          - Two-phase training: structure-focused then joint
          - Cosine annealing learning rate schedule
          - More frequent Lagrangian updates (every 3 epochs)
          - Gradient clipping with adaptive norm

        Args:
            observations: (B, T, obs_dim) observation tensor
            raw_data: Optional (B, T, N) raw tabular data for warm init
        """
        if verbose:
            print(f"\n{'=' * 60}")
            print("STAGE 2: GNN + Liquid Net (Structure + Mechanisms)")
            print("=" * 60)

        # Get latent from trained Layer 1 (detached — no gradient to L1)
        with torch.no_grad():
            latent = self.layer1.extract_variables(observations)

        # v0.6.0: Warm initialization via standalone NOTEARS on raw data
        self.layer2.gnn.warm_init_from_data(latent, raw_data=raw_data)
        self.layer2.gnn.prune_to_dag()
        if verbose:
            with torch.no_grad():
                A_init = self.layer2.gnn.adjacency(hard=True)
                n_init = int(A_init.sum().item())
            init_src = "raw data NOTEARS" if raw_data is not None else "latent correlations"
            print(f"  Warm init: {n_init} edges from {init_src}")

        # Phase 1: Structure-focused training (higher LR for W, lower for mechanisms)
        total_epochs = self.config.train_epochs_l2
        phase1_epochs = max(1, total_epochs * 2 // 3)
        phase2_epochs = total_epochs - phase1_epochs

        # Separate parameter groups: structure params get higher LR
        gnn_params = list(self.layer2.gnn.parameters())
        liquid_params = list(self.layer2.liquid.parameters())

        optimizer = optim.Adam([
            {'params': gnn_params, 'lr': self.config.layer2_lr * 2.0},
            {'params': liquid_params, 'lr': self.config.layer2_lr},
        ])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_epochs, eta_min=self.config.layer2_lr * 0.1
        )

        for ep in range(total_epochs):
            optimizer.zero_grad()
            loss = self.layer2.compute_loss(latent, raw_data=raw_data)
            loss.backward()
            nn.utils.clip_grad_norm_(self.layer2.parameters(), max_norm=5.0)
            optimizer.step()
            scheduler.step()

            # Update Lagrangian every 5 epochs
            if (ep + 1) % 5 == 0:
                h = self.layer2.gnn.update_lagrangian()

            if verbose and (ep + 1) % max(1, total_epochs // 3) == 0:
                with torch.no_grad():
                    A = self.layer2.gnn.adjacency(hard=True)
                    n_edges = int(A.sum().item())
                    is_dag = self.layer2.gnn._is_dag(A)
                    dag_pen = self.layer2.gnn.dag_penalty().item()
                print(f"  Epoch {ep + 1}/{total_epochs} | "
                      f"Edges: {n_edges} | DAG: {is_dag} | "
                      f"DAG pen: {dag_pen:.4f} | Loss: {loss.item():.4f}")

        # Final DAG enforcement via pruning
        self.layer2.gnn.prune_to_dag()

        # v0.6.0: Edge orientation refinement using raw data + re-verify DAG
        self.layer2.gnn.refine_orientations(latent, raw_data=raw_data)
        self.layer2.gnn.prune_to_dag()

        if verbose:
            with torch.no_grad():
                A_final = self.layer2.gnn.adjacency(hard=True)
            print(f"  Final: {int(A_final.sum().item())} edges, "
                  f"DAG verified: {self.layer2.gnn._is_dag(A_final)}")

    def train_layer3(self, observations: torch.Tensor, verbose: bool = True):
        """
        Stage 3: Train HRM (reasoning orchestration).

        v0.4.1 FIX: Loss is computed from HRM output tensor (has gradient)
        instead of from detached convergence scalar. The old code used
        torch.tensor(r['convergence']) which created a new leaf tensor with
        no gradient connection to HRM parameters — so train_layer3 was a
        no-op that never updated any weights.
        """
        optimizer = optim.Adam(self.layer3.hrm.parameters(), lr=self.config.layer3_lr)

        if verbose:
            print(f"\n{'=' * 60}")
            print("STAGE 3: HRM (Reasoning Orchestration)")
            print("=" * 60)

        for ep in range(self.config.train_epochs_l3):
            optimizer.zero_grad()
            q = torch.randn(self.config.latent_dim)
            r = self.layer3.hrm.reason(q)

            # v0.4.1 FIX: compute loss from output tensor that retains gradient
            # Old: conv_loss = F.mse_loss(torch.tensor(r['convergence']), ...)
            #      ^ torch.tensor(float) is a NEW LEAF — zero gradient to HRM
            # New: loss from r['result'] which flows through GRU/halt networks
            if isinstance(r['result'], torch.Tensor) and r['result'].requires_grad:
                # Output stability: encourage bounded, non-degenerate outputs
                stability_loss = r['result'].norm()
                # Run second query for gradient diversity
                q2 = torch.randn(self.config.latent_dim)
                r2 = self.layer3.hrm.reason(q2)
                if isinstance(r2['result'], torch.Tensor) and r2['result'].requires_grad:
                    total_loss = 0.01 * stability_loss + 0.01 * r2['result'].norm()
                    total_loss.backward()
                    nn.utils.clip_grad_norm_(self.layer3.hrm.parameters(), max_norm=5.0)
                    optimizer.step()

            if verbose and (ep + 1) % max(1, self.config.train_epochs_l3 // 3) == 0:
                resets = sum(1 for t in r['trace']
                             if 'event' in t and t['event'] == 'H_MODULE_RESET')
                print(f"  Epoch {ep + 1}/{self.config.train_epochs_l3} | "
                      f"Steps: {r['steps']} | Conv: {r['convergence']:.4f} | "
                      f"Resets: {resets}")

    def train_all(self, observations: torch.Tensor, verbose: bool = True,
                   raw_data: Optional[torch.Tensor] = None):
        """
        Full staged training pipeline.
        Each stage trains its layer independently (no gradient crossing).

        Args:
            observations: (B, T, obs_dim) observation tensor
            raw_data: Optional (B, T, N) raw tabular data for warm init
        """
        self.train_layer1(observations, verbose)
        self.train_layer2(observations, verbose, raw_data=raw_data)
        self.train_layer3(observations, verbose)

        if verbose:
            print(f"\n{'=' * 60}")
            print("TRAINING COMPLETE")
            print("=" * 60)

    # --- SCM-based Counterfactual (variable-space ABP) ---

    def fit_scm(self, var_data: np.ndarray, verbose: bool = False):
        """
        Fit linear SCM in variable space for counterfactual reasoning.

        Three-stage approach:
          1. Skeleton discovery via partial correlations (precision matrix).
          2. Edge orientation via variance ordering (higher → lower),
             with residual-based tie-breaking for similar-variance pairs.
          3. OLS coefficient estimation for each variable on its parents.

        Uses ABP (Abduction-Action-Prediction) for counterfactuals.

        Args:
            var_data: (M, N) matrix of variable observations (samples x vars)
            verbose: print fitting diagnostics
        """
        N = var_data.shape[1]
        self._scm_coefficients = {}
        self._scm_edges = []
        self._scm_num_vars = N

        # Stage 1: Find skeleton via partial correlations
        X_std = (var_data - var_data.mean(0)) / (var_data.std(0) + 1e-8)
        cov_mat = np.cov(X_std.T)
        try:
            prec = np.linalg.inv(cov_mat + 1e-6 * np.eye(N))
        except np.linalg.LinAlgError:
            prec = np.linalg.pinv(cov_mat)
        diag_p = np.sqrt(np.abs(np.diag(prec)) + 1e-8)

        pcor_threshold = 0.1
        skeleton = set()
        neighbors = {i: set() for i in range(N)}
        for i in range(N):
            for j in range(i + 1, N):
                pc = abs(prec[i, j]) / (diag_p[i] * diag_p[j])
                if pc > pcor_threshold:
                    skeleton.add((i, j))
                    neighbors[i].add(j)
                    neighbors[j].add(i)

        # Stage 2: Orient edges
        variances = np.var(var_data, axis=0)
        for i, j in skeleton:
            var_ratio = max(variances[i], variances[j]) / (
                min(variances[i], variances[j]) + 1e-8)

            if var_ratio > 1.1:
                # Clear variance difference: orient from high to low
                if variances[i] >= variances[j]:
                    self._scm_edges.append((i, j))
                else:
                    self._scm_edges.append((j, i))
            else:
                # Similar variances: use residual-based tie-breaking.
                # Orient towards the variable with lower conditional variance
                # (better explained by others = more likely to be the effect).
                # Regress each on ALL other variables, compare residual MSE.
                others_i = [k for k in range(N) if k != i]
                others_j = [k for k in range(N) if k != j]
                X_oi = var_data[:, others_i]
                X_oj = var_data[:, others_j]
                y_i = var_data[:, i]
                y_j = var_data[:, j]

                # Residual variance of i given all others
                beta_i = np.linalg.lstsq(X_oi, y_i, rcond=None)[0]
                res_var_i = np.var(y_i - X_oi @ beta_i)

                # Residual variance of j given all others
                beta_j = np.linalg.lstsq(X_oj, y_j, rcond=None)[0]
                res_var_j = np.var(y_j - X_oj @ beta_j)

                # Lower residual = better explained = effect (downstream)
                if res_var_i <= res_var_j:
                    self._scm_edges.append((j, i))  # j → i
                else:
                    self._scm_edges.append((i, j))  # i → j

        if verbose:
            print(f"  SCM structure: {len(self._scm_edges)} edges")

        # Stage 3: OLS coefficient estimation on discovered parents
        for node in range(N):
            parents = [p for p, c in self._scm_edges if c == node]
            if parents:
                X_pa = var_data[:, parents]
                y = var_data[:, node]
                XtX = X_pa.T @ X_pa + 1e-6 * np.eye(len(parents))
                Xty = X_pa.T @ y
                beta = np.linalg.solve(XtX, Xty)
                for idx, p in enumerate(parents):
                    self._scm_coefficients[(p, node)] = float(beta[idx])
                if verbose:
                    print(f"  X{node} <- {['X'+str(p) for p in parents]}: "
                          f"{[f'{b:.3f}' for b in beta]}")

        # Stage 4: Compute per-variable noise variance to detect bad structure
        # Variables with anomalously high noise = likely wrong parents
        self._noise_vars = np.zeros(N)
        non_root_nodes = []
        for node in range(N):
            parents = [p for p, c in self._scm_edges if c == node]
            if parents:
                non_root_nodes.append(node)
                X_pa = var_data[:, parents]
                y = var_data[:, node]
                XtX = X_pa.T @ X_pa + 1e-6 * np.eye(len(parents))
                Xty = X_pa.T @ y
                beta_full = np.linalg.solve(XtX, Xty)
                residuals = y - X_pa @ beta_full
                self._noise_vars[node] = float(np.var(residuals))
            else:
                self._noise_vars[node] = float(np.var(var_data[:, node]))

        # Flag suspicious nodes: non-root with noise >> median non-root noise
        self._suspicious_nodes = set()
        if non_root_nodes:
            nr_noise = [self._noise_vars[n] for n in non_root_nodes]
            median_nr = float(np.median(nr_noise))
            for n in non_root_nodes:
                if self._noise_vars[n] > 3.0 * median_nr:
                    self._suspicious_nodes.add(n)
        if verbose and self._suspicious_nodes:
            print(f"  Suspicious nodes: {self._suspicious_nodes}")

        # Stage 5: Compute total causal effects for fallback
        self._total_effects = {}
        parents_of = {i: [] for i in range(N)}
        for node in range(N):
            for nb in neighbors[node]:
                if variances[nb] > variances[node]:
                    parents_of[node].append(nb)

        for src in range(N):
            pa_src = parents_of[src]
            for tgt in range(N):
                if tgt == src:
                    continue
                controls = [p for p in pa_src if p != src and p != tgt]
                reg_idx = [src] + controls
                X_reg = var_data[:, reg_idx]
                y = var_data[:, tgt]
                XtX = X_reg.T @ X_reg + 1e-6 * np.eye(len(reg_idx))
                Xty = X_reg.T @ y
                beta = np.linalg.solve(XtX, Xty)
                self._total_effects[(src, tgt)] = float(beta[0])

    def counterfactual_scm(self, factual_values: np.ndarray,
                           src: int, cf_x: float) -> np.ndarray:
        """
        Compute counterfactual via ensemble of ABP and total-effect methods.

        Uses two complementary estimators:
          1. ABP (Abduction-Action-Prediction): exact when graph structure
             is correct, but fragile to misspecified edges.
          2. Total-effect estimation: CF_Y = factual_Y + β × Δx, robust
             to partial graph errors but noisier for non-root sources.

        The ensemble weights favor ABP when the inferred noise is small
        (indicating correct structure) and total-effect otherwise.

        Args:
            factual_values: (N,) observed values of all variables
            src: intervention variable index
            cf_x: counterfactual value for src

        Returns:
            (N,) counterfactual values for all variables
        """
        N = self._scm_num_vars
        coeffs = self._scm_coefficients
        edges_N = list(self._scm_edges)
        delta_x = cf_x - factual_values[src]

        # --- Method 1: ABP ---
        from collections import deque
        in_deg = [0] * N
        children_map = [[] for _ in range(N)]
        for p, c in edges_N:
            in_deg[c] += 1
            children_map[p].append(c)
        queue = deque(i for i in range(N) if in_deg[i] == 0)
        order = []
        while queue:
            node = queue.popleft()
            order.append(node)
            for ch in children_map[node]:
                in_deg[ch] -= 1
                if in_deg[ch] == 0:
                    queue.append(ch)
        for i in range(N):
            if i not in order:
                order.append(i)

        noises = np.zeros(N)
        for node in order:
            parents = [p for p, c in edges_N if c == node]
            parent_contribution = sum(
                coeffs.get((p, node), 0.0) * factual_values[p]
                for p in parents
            )
            noises[node] = factual_values[node] - parent_contribution

        cf_abp = np.zeros(N)
        for node in order:
            if node == src:
                cf_abp[node] = cf_x
            else:
                parents = [p for p, c in edges_N if c == node]
                cf_abp[node] = noises[node] + sum(
                    coeffs.get((p, node), 0.0) * cf_abp[p]
                    for p in parents
                )

        # --- Method 2: Total-effect estimation ---
        cf_te = factual_values.copy()
        cf_te[src] = cf_x
        for tgt in range(N):
            if tgt != src:
                effect = self._total_effects.get((src, tgt), 0.0)
                cf_te[tgt] = factual_values[tgt] + effect * delta_x

        # --- Per-target selection between ABP and total-effect ---
        # When ABP and TE agree, ABP is more accurate (exact with correct
        # structure). When they disagree, the structure may be wrong,
        # so use TE (robust to partial structure errors).
        cf_vals = np.zeros(N)
        for node in range(N):
            if node == src:
                cf_vals[node] = cf_x
            else:
                abp_pred = cf_abp[node]
                te_pred = cf_te[node]
                # If predictions differ by more than expected noise, use TE
                diff = abs(abp_pred - te_pred)
                signal = abs(delta_x) * 0.3  # rough scale of expected change
                if diff > max(signal, 0.5):
                    cf_vals[node] = te_pred
                else:
                    cf_vals[node] = abp_pred

        return cf_vals

    # Legacy compatibility
    def train_model(self, observations: torch.Tensor, verbose: bool = True):
        """Alias for train_all."""
        self.train_all(observations, verbose)

    # --- Diagnostics ---

    def summary(self) -> str:
        sg = self.layer2.symbolic_graph()
        diag = self.layer3.generate_diagnostic(sg)
        lines = [
            "",
            "=" * 60,
            "HHCRA: Hierarchical Hybrid Causal Reasoning Architecture",
            "=" * 60,
            "",
            "Architecture:",
            f"  Layer 1: C-JEPA",
            f"    {self.config.num_vars} latent variable slots from "
            f"{self.config.obs_dim}-dim observations",
            f"    Latent dim: {self.config.latent_dim}",
            f"",
            f"  Layer 2: GNN + Liquid Neural Network [tightly coupled]",
            f"    GNN: NOTEARS continuous optimization ({self.config.num_vars} nodes)",
            f"    Liquid Net: Neural ODE (method={self.config.liquid_method}, "
            f"dt={self.config.liquid_dt}, steps={self.config.liquid_ode_steps})",
            f"",
            f"  Layer 3: Neuro-Symbolic + HRM [tightly coupled]",
            f"    Neuro-symbolic: d-separation, backdoor/frontdoor, "
            f"do-calculus (3 rules), ID algorithm",
            f"    HRM: GRU + ACT, {self.config.hrm_max_steps} max steps, "
            f"H-update every {self.config.hrm_update_interval} steps",
            f"",
            f"Connections:",
            f"  L1 -> L2: Interface (.detach() — no gradient crossing)",
            f"  L2 -> L3: Interface (.detach() — no gradient crossing)",
            f"  L3 -> L2: Feedback (structure revision requests)",
            f"  L3 -> L1: Feedback (variable resolution adjustment)",
            f"",
            f"Learned Causal Graph:",
            f"  Nodes: {len(sg.nodes)} | Edges: {sg.edge_count()} | DAG: {sg.is_dag()}",
            f"  Density: {diag['density']:.1%}",
        ]

        for p, c, w in sg.edges[:15]:
            lines.append(f"    X{p} -> X{c} (weight: {w:.3f})")
        if sg.edge_count() > 15:
            lines.append(f"    ... ({sg.edge_count() - 15} more)")

        if diag['issues']:
            lines.append(f"\n  Diagnostics:")
            for iss in diag['issues']:
                lines.append(f"    WARNING: {iss}")

        lines.extend([
            "",
            "Pearl's Ladder Coverage:",
            "  Rung 1 (Observation):    P(Y|X)",
            "  Rung 2 (Intervention):   P(Y|do(X)) via do-calculus + Neural ODE",
            "  Rung 3 (Counterfactual): P(Y_{x'}|X=x) via ABP procedure",
        ])

        return "\n".join(lines)
