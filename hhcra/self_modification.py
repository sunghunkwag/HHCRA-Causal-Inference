"""
Phase 8: Self-Modifying Architecture (RSI Foundation)

A system that evaluates its own components, identifies bottlenecks,
modifies structure/hyperparameters, and verifies improvement.

RSI loop: evaluate -> detect bottleneck -> modify -> verify -> repeat

Modifications are to hyperparameters and structure, NOT to core algorithms.
The system adjusts its own configuration space.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from copy import deepcopy

from hhcra.config import HHCRAConfig
from hhcra.causal_graph import CausalQueryType


@dataclass
class PerformanceMetrics:
    """Per-layer performance metrics."""
    layer1_reconstruction: float = 0.0  # Variable reconstruction quality
    layer2_shd: float = float('inf')    # Structure accuracy
    layer2_mechanism_mse: float = 0.0   # Mechanism prediction MSE
    layer3_convergence: float = 0.0     # Reasoning convergence speed
    layer3_identifiability: float = 0.0 # Identifiability rate
    world_model_error: float = 0.0      # Prediction error trend
    overall: float = 0.0


@dataclass
class Modification:
    """A single configuration modification."""
    parameter: str
    old_value: float
    new_value: float
    reason: str
    metrics_before: PerformanceMetrics
    metrics_after: Optional[PerformanceMetrics] = None
    accepted: bool = False


class ArchitectureEvaluator:
    """Measures per-layer performance metrics."""

    def evaluate(self, model, observations: torch.Tensor,
                 ground_truth: Optional[dict] = None) -> PerformanceMetrics:
        """Evaluate all layers and return performance metrics."""
        metrics = PerformanceMetrics()

        # Layer 1: variable reconstruction quality
        with torch.no_grad():
            latent = model.layer1.extract_variables(observations)
            # Measure slot diversity (higher = better variable extraction)
            B, T, N, D = latent.shape
            slot_std = latent.std(dim=2).mean().item()  # Variation across slots
            metrics.layer1_reconstruction = slot_std

        # Layer 2: structure accuracy
        if ground_truth is not None and 'true_adjacency' in ground_truth:
            gnn_metrics = model.layer2.gnn.compute_metrics(
                ground_truth['true_adjacency'])
            metrics.layer2_shd = gnn_metrics['shd']

        # Layer 2: mechanism prediction MSE
        with torch.no_grad():
            l2_out = model.layer2(latent)
            traj = l2_out['trajectories']
            if T > 1:
                mech_mse = torch.mean(
                    (traj[:, :-1, :, :] - latent[:, 1:, :, :]) ** 2
                ).item()
            else:
                mech_mse = 0.0
            metrics.layer2_mechanism_mse = mech_mse

        # Layer 3: convergence speed
        q = torch.randn(model.config.latent_dim)
        r = model.layer3.hrm.reason(q)
        metrics.layer3_convergence = r['convergence']

        # Layer 3: identifiability rate
        graph = model.layer2.symbolic_graph()
        N_nodes = min(len(graph.nodes), 5)
        id_count = 0
        total = 0
        for x in range(N_nodes):
            for y in range(N_nodes):
                if x != y:
                    result = model.layer3.symbolic.check_identifiability(graph, x, y)
                    if result['identifiable']:
                        id_count += 1
                    total += 1
        metrics.layer3_identifiability = id_count / max(total, 1)

        # Overall score (weighted)
        metrics.overall = (
            0.2 * metrics.layer1_reconstruction +
            0.3 * max(0, 1.0 - metrics.layer2_shd / 20.0) +
            0.2 * max(0, 1.0 - metrics.layer2_mechanism_mse) +
            0.15 * metrics.layer3_convergence +
            0.15 * metrics.layer3_identifiability
        )

        return metrics


class BottleneckDetector:
    """Identifies which layer is the weakest link."""

    def detect(self, metrics: PerformanceMetrics) -> Tuple[str, str]:
        """
        Returns (bottleneck_layer, reason).
        Uses relative performance degradation, not absolute scores.
        """
        scores = {
            'layer1': metrics.layer1_reconstruction,
            'layer2_structure': max(0, 1.0 - metrics.layer2_shd / 20.0),
            'layer2_mechanism': max(0, 1.0 - metrics.layer2_mechanism_mse),
            'layer3_convergence': metrics.layer3_convergence,
            'layer3_identifiability': metrics.layer3_identifiability,
        }

        worst_key = min(scores, key=scores.get)
        worst_score = scores[worst_key]

        reasons = {
            'layer1': 'Low variable extraction diversity',
            'layer2_structure': f'High SHD ({metrics.layer2_shd})',
            'layer2_mechanism': f'High mechanism MSE ({metrics.layer2_mechanism_mse:.4f})',
            'layer3_convergence': f'Low convergence ({metrics.layer3_convergence:.4f})',
            'layer3_identifiability': f'Low identifiability rate ({metrics.layer3_identifiability:.3f})',
        }

        return worst_key, reasons[worst_key]


class StructureModifier:
    """
    Modifies architecture hyperparameters.

    Can modify: num_vars, edge_threshold, ode_steps/dt, hrm_max_steps.
    Each modification is REVERSIBLE — stores previous config.
    """

    MODIFICATION_RANGES = {
        'edge_threshold': (0.1, 0.6, 0.05),      # (min, max, step)
        'liquid_ode_steps': (4, 16, 2),
        'liquid_dt': (0.01, 0.1, 0.01),
        'hrm_max_steps': (10, 50, 5),
        'notears_lambda': (0.001, 0.1, 0.005),
        'train_epochs_l2': (10, 50, 5),
    }

    def propose_modification(self, bottleneck: str,
                             config: HHCRAConfig) -> Optional[Tuple[str, float]]:
        """Propose a config modification based on detected bottleneck."""
        if bottleneck == 'layer2_structure':
            # Structure learning issue -> adjust edge threshold or NOTEARS lambda
            current = config.edge_threshold
            mn, mx, step = self.MODIFICATION_RANGES['edge_threshold']
            new_val = current - step if current > mn else current + step
            return ('edge_threshold', new_val)

        elif bottleneck == 'layer2_mechanism':
            # Mechanism issue -> adjust ODE parameters
            current = config.liquid_ode_steps
            mn, mx, step = self.MODIFICATION_RANGES['liquid_ode_steps']
            new_val = min(current + int(step), int(mx))
            return ('liquid_ode_steps', new_val)

        elif bottleneck == 'layer3_convergence':
            # Convergence issue -> increase max steps
            current = config.hrm_max_steps
            mn, mx, step = self.MODIFICATION_RANGES['hrm_max_steps']
            new_val = min(current + int(step), int(mx))
            return ('hrm_max_steps', new_val)

        elif bottleneck == 'layer3_identifiability':
            # Identifiability issue -> adjust sparsity
            current = config.notears_lambda
            mn, mx, step = self.MODIFICATION_RANGES['notears_lambda']
            new_val = max(current - step, mn)
            return ('notears_lambda', new_val)

        elif bottleneck == 'layer1':
            # Variable extraction -> more training
            return ('train_epochs_l1', config.train_epochs_l1 + 5)

        return None

    def apply_modification(self, config: HHCRAConfig,
                           param: str, value: float) -> HHCRAConfig:
        """Apply modification to config. Returns new config."""
        new_config = HHCRAConfig(**{
            k: v for k, v in config.__dict__.items()
        })
        if hasattr(new_config, param):
            if isinstance(getattr(new_config, param), int):
                setattr(new_config, param, int(value))
            else:
                setattr(new_config, param, float(value))
        return new_config


class ModificationVerifier:
    """Verifies whether a modification improved performance."""

    def __init__(self, evaluator: ArchitectureEvaluator):
        self.evaluator = evaluator

    def verify(self, model_before, model_after,
               observations: torch.Tensor,
               ground_truth: Optional[dict] = None) -> bool:
        """Return True if model_after is better than model_before."""
        metrics_before = self.evaluator.evaluate(
            model_before, observations, ground_truth)
        metrics_after = self.evaluator.evaluate(
            model_after, observations, ground_truth)
        return metrics_after.overall >= metrics_before.overall


class SelfModificationEngine:
    """
    RSI loop: evaluate -> detect bottleneck -> modify -> verify -> repeat

    Modifications are to config hyperparameters, not code.
    All modifications are logged with before/after metrics.
    """

    def __init__(self):
        self.evaluator = ArchitectureEvaluator()
        self.detector = BottleneckDetector()
        self.modifier = StructureModifier()
        self.verifier = ModificationVerifier(self.evaluator)
        self.modification_log: List[Modification] = []

    def run_rsi_loop(
        self,
        model,
        observations: torch.Tensor,
        ground_truth: Optional[dict] = None,
        num_iterations: int = 20,
        verbose: bool = True,
    ) -> List[PerformanceMetrics]:
        """
        Run RSI loop for num_iterations.
        Returns performance metrics at each iteration.
        """
        from hhcra.architecture import HHCRA

        metrics_history = []
        current_config = deepcopy(model.config)
        best_overall = 0.0

        for i in range(num_iterations):
            # 1. Evaluate
            metrics = self.evaluator.evaluate(model, observations, ground_truth)
            metrics_history.append(metrics)

            if verbose:
                print(f"  RSI iteration {i + 1}/{num_iterations}: "
                      f"overall={metrics.overall:.4f} "
                      f"SHD={metrics.layer2_shd} "
                      f"conv={metrics.layer3_convergence:.4f}")

            # 2. Detect bottleneck
            bottleneck, reason = self.detector.detect(metrics)

            # 3. Propose modification
            proposal = self.modifier.propose_modification(
                bottleneck, current_config)

            if proposal is None:
                if verbose:
                    print(f"    No modification proposed for {bottleneck}")
                continue

            param, new_value = proposal
            old_value = getattr(current_config, param)

            if abs(float(new_value) - float(old_value)) < 1e-8:
                continue  # No change needed

            # 4. Apply modification
            new_config = self.modifier.apply_modification(
                current_config, param, new_value)

            # Create and train modified model
            torch.manual_seed(42)
            new_model = HHCRA(new_config)
            new_model.train_all(observations, verbose=False)
            new_model.eval()

            # 5. Verify improvement
            new_metrics = self.evaluator.evaluate(
                new_model, observations, ground_truth)

            mod = Modification(
                parameter=param,
                old_value=old_value,
                new_value=new_value,
                reason=reason,
                metrics_before=metrics,
                metrics_after=new_metrics,
            )

            if new_metrics.overall >= metrics.overall:
                # Accept modification
                mod.accepted = True
                model = new_model
                current_config = new_config
                best_overall = new_metrics.overall
                if verbose:
                    print(f"    ACCEPTED: {param} {old_value} -> {new_value} "
                          f"(overall: {metrics.overall:.4f} -> {new_metrics.overall:.4f})")
            else:
                # Revert
                mod.accepted = False
                if verbose:
                    print(f"    REVERTED: {param} {old_value} -> {new_value} "
                          f"(would degrade: {metrics.overall:.4f} -> {new_metrics.overall:.4f})")

            self.modification_log.append(mod)

        return metrics_history

    def get_modification_summary(self) -> dict:
        """Summary of all modifications attempted."""
        accepted = [m for m in self.modification_log if m.accepted]
        rejected = [m for m in self.modification_log if not m.accepted]
        return {
            'total_attempted': len(self.modification_log),
            'accepted': len(accepted),
            'rejected': len(rejected),
            'modifications': [
                {
                    'param': m.parameter,
                    'old': m.old_value,
                    'new': m.new_value,
                    'reason': m.reason,
                    'accepted': m.accepted,
                }
                for m in self.modification_log
            ],
        }
