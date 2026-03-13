"""
Phase 7: World Model Integration (External Grounding)

The world model provides a verification loop for HHCRA's causal predictions.
It receives predictions, runs simulation, compares against actual outcomes,
and returns error signals to the appropriate HHCRA layers.

Architecture position: EXTERNAL to HHCRA, connected via grounding interface.

Grounding loop: predict -> act -> observe -> compare -> correct
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, List, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass

from hhcra.config import HHCRAConfig
from hhcra.causal_graph import CausalGraphData, CausalQueryType


@dataclass
class GroundingResult:
    """Result of a single grounding step."""
    prediction: np.ndarray
    actual: np.ndarray
    error: float
    error_type: str  # 'structure', 'mechanism', 'variable'
    feedback: dict


class EnvironmentInterface(ABC):
    """Abstract base class for any environment that grounds HHCRA."""

    @abstractmethod
    def reset(self) -> np.ndarray:
        """Reset environment, return initial observation."""
        pass

    @abstractmethod
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """Execute action, return (observation, reward, done, info)."""
        pass

    @abstractmethod
    def intervene(self, var_idx: int, value: float) -> np.ndarray:
        """Perform do(X_var_idx = value), return resulting observation."""
        pass

    @abstractmethod
    def get_true_graph(self) -> Optional[CausalGraphData]:
        """Return ground truth causal graph if available."""
        pass


class SimplePhysicsWorld(EnvironmentInterface):
    """
    Simple physics world: objects with causal relationships.
    Gravity, spring connections, basic dynamics.
    """

    def __init__(self, num_objects: int = 4, seed: int = 42):
        self.num_objects = num_objects
        self.rng = np.random.RandomState(seed)

        # Object states: [x, y, vx, vy] per object
        self.state_dim = 4
        self.positions = np.zeros((num_objects, 2))
        self.velocities = np.zeros((num_objects, 2))

        # Causal connections (springs between objects)
        self.connections = []  # (i, j, stiffness)
        self.gravity = -0.1

        self._setup_connections()
        self.reset()

    def _setup_connections(self):
        """Create chain-like causal connections between objects."""
        for i in range(self.num_objects - 1):
            self.connections.append((i, i + 1, 0.5))

    def reset(self) -> np.ndarray:
        self.positions = self.rng.randn(self.num_objects, 2) * 0.5
        self.velocities = np.zeros((self.num_objects, 2))
        return self._observe()

    def step(self, action: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float, bool, dict]:
        dt = 0.1
        forces = np.zeros_like(self.velocities)

        # Gravity
        forces[:, 1] += self.gravity

        # Spring forces from connections
        for i, j, k in self.connections:
            diff = self.positions[j] - self.positions[i]
            dist = np.linalg.norm(diff) + 1e-8
            force = k * diff / dist
            forces[i] += force
            forces[j] -= force

        # Apply action to first object
        if action is not None and len(action) >= 2:
            forces[0] += action[:2]

        # Euler integration
        self.velocities += forces * dt
        self.velocities *= 0.98  # damping
        self.positions += self.velocities * dt

        obs = self._observe()
        reward = 0.0
        done = False
        return obs, reward, done, {}

    def intervene(self, var_idx: int, value: float) -> np.ndarray:
        """Fix object var_idx position and observe effect."""
        saved_pos = self.positions.copy()
        self.positions[var_idx, 0] = value
        self.velocities[var_idx] = 0.0

        # Step forward to propagate effect
        for _ in range(10):
            self.step()
            self.positions[var_idx, 0] = value
            self.velocities[var_idx] = 0.0

        obs = self._observe()
        self.positions = saved_pos
        return obs

    def _observe(self) -> np.ndarray:
        """Flatten state to observation vector."""
        return np.concatenate([
            self.positions.flatten(),
            self.velocities.flatten(),
        ])

    def get_true_graph(self) -> Optional[CausalGraphData]:
        N = self.num_objects
        adj = np.zeros((N, N))
        edges = []
        for i, j, k in self.connections:
            adj[j, i] = 1.0
            adj[i, j] = 1.0  # Bidirectional spring influence
            edges.append((i, j, float(k)))
            edges.append((j, i, float(k)))
        return CausalGraphData(list(range(N)), edges, adj)


class WorldModel:
    """
    World model that verifies HHCRA's causal predictions.

    Receives: causal graph + intervention predictions from HHCRA
    Runs: simulation or environment query
    Returns: prediction error signals routed to appropriate layers
    """

    def __init__(self, environment: EnvironmentInterface):
        self.env = environment
        self.error_history: List[float] = []
        self.grounding_results: List[GroundingResult] = []

    def ground_prediction(
        self,
        predicted_effect: np.ndarray,
        var_idx: int,
        intervention_value: float,
    ) -> GroundingResult:
        """
        Ground a single intervention prediction against environment.

        1. HHCRA predicts effect of do(X_var_idx = value)
        2. World model applies intervention in environment
        3. Compare prediction vs actual
        4. Generate error signal for appropriate layer
        """
        actual_obs = self.env.intervene(var_idx, intervention_value)

        # Compare prediction vs actual
        pred_flat = predicted_effect.flatten()
        actual_flat = actual_obs.flatten()
        min_len = min(len(pred_flat), len(actual_flat))
        error = float(np.mean((pred_flat[:min_len] - actual_flat[:min_len]) ** 2))

        # Classify error type and generate feedback
        error_type, feedback = self._classify_error(error, var_idx)

        result = GroundingResult(
            prediction=predicted_effect,
            actual=actual_obs,
            error=error,
            error_type=error_type,
            feedback=feedback,
        )

        self.error_history.append(error)
        self.grounding_results.append(result)
        return result

    def _classify_error(self, error: float, var_idx: int) -> Tuple[str, dict]:
        """Classify prediction error and generate layer-specific feedback."""
        if error > 1.0:
            # Large error -> structure problem (Layer 2 GNN)
            return 'structure', {
                'to_layer2': {
                    'issue': 'high_prediction_error',
                    'error': error,
                    'suggestion': 'Revise graph structure',
                },
            }
        elif error > 0.1:
            # Medium error -> mechanism problem (Layer 2 Liquid Net)
            return 'mechanism', {
                'to_layer2': {
                    'issue': 'mechanism_mismatch',
                    'error': error,
                    'suggestion': 'Refit ODE dynamics',
                },
            }
        else:
            # Small error -> variable extraction issue (Layer 1)
            return 'variable', {
                'to_layer1': {
                    'increase_resolution': error > 0.01,
                    'reason': f'Prediction error {error:.4f} on var {var_idx}',
                },
            }

    def grounding_loop(
        self,
        hhcra_model,
        observations: torch.Tensor,
        var_idx: int = 0,
        intervention_value: float = 2.0,
        num_iterations: int = 10,
        verbose: bool = True,
    ) -> List[float]:
        """
        Full grounding loop: predict -> act -> observe -> compare -> correct.

        Returns error curve over iterations.
        """
        errors = []

        for i in range(num_iterations):
            # HHCRA predicts
            with torch.no_grad():
                xv = torch.full((hhcra_model.config.latent_dim,), intervention_value)
                result = hhcra_model.query(
                    observations, CausalQueryType.INTERVENTIONAL,
                    X=var_idx, Y=min(var_idx + 1, hhcra_model.config.num_vars - 1),
                    x_value=xv, verbose=False,
                )

            if result['answer'] is not None:
                prediction = result['answer'].cpu().numpy().flatten()
            else:
                prediction = np.zeros(hhcra_model.config.latent_dim)

            # Ground against environment
            grounding = self.ground_prediction(
                prediction, var_idx, intervention_value)
            errors.append(grounding.error)

            # Route feedback to HHCRA
            if grounding.feedback:
                from hhcra.interfaces import FeedbackRouter
                FeedbackRouter.route(
                    grounding.feedback,
                    hhcra_model.layer2,
                    hhcra_model.layer1,
                    verbose=False,
                )

            if verbose:
                print(f"  Iteration {i + 1}/{num_iterations}: "
                      f"error={grounding.error:.4f} type={grounding.error_type}")

        return errors

    def convergence_improved(self) -> bool:
        """Check if grounding errors are trending downward."""
        if len(self.error_history) < 3:
            return False
        recent = self.error_history[-3:]
        return recent[-1] < recent[0]

    def get_convergence_stats(self) -> dict:
        """Return statistics about grounding convergence."""
        if not self.error_history:
            return {'mean_error': 0.0, 'trend': 0.0, 'num_groundings': 0}

        errors = np.array(self.error_history)
        # Trend: negative means improving
        if len(errors) > 1:
            trend = float(np.polyfit(range(len(errors)), errors, 1)[0])
        else:
            trend = 0.0

        return {
            'mean_error': float(errors.mean()),
            'min_error': float(errors.min()),
            'max_error': float(errors.max()),
            'trend': trend,
            'num_groundings': len(errors),
        }
