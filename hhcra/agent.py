"""
Phase 10: Agent Loop (Closed-Loop Causal Agent)

Integrates all components into an autonomous causal reasoning agent:
  - Perception (Phase 9) -> extracts state
  - HHCRA core (Phase 1-6) -> causal reasoning
  - World model (Phase 7) -> prediction + verification
  - Self-modification (Phase 8) -> continuous improvement

Agent loop: perceive -> update -> plan -> act -> observe -> ground -> correct
"""

import torch
import numpy as np
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field

from hhcra.config import HHCRAConfig
from hhcra.architecture import HHCRA
from hhcra.causal_graph import CausalQueryType


@dataclass
class AgentState:
    """Internal state of the causal agent."""
    observation: np.ndarray
    causal_graph: object = None
    cumulative_reward: float = 0.0
    episode_step: int = 0
    total_steps: int = 0


@dataclass
class EpisodeResult:
    """Result of a single agent episode."""
    total_reward: float
    num_steps: int
    graph_discovered: bool
    intervention_accuracy: float


class CausalAgent:
    """
    Closed-loop causal agent that perceives, reasons, acts, and learns.

    Agent loop:
      (1) Perceive: get observation from environment
      (2) Update: feed to HHCRA, update causal graph
      (3) Plan: use interventional reasoning to evaluate actions
      (4) Act: execute chosen action in environment
      (5) Observe: get outcome
      (6) Ground: compare prediction vs outcome
      (7) Correct: send error signals to HHCRA layers
      (8) Self-modify: if performance stagnates, run RSI loop
      (9) Repeat
    """

    def __init__(self, config: Optional[HHCRAConfig] = None,
                 obs_dim: int = 48, num_actions: int = 4):
        self.config = config or HHCRAConfig(obs_dim=obs_dim)
        self.num_actions = num_actions

        # Core HHCRA model
        self.model = HHCRA(self.config)

        # Observation buffer for batch processing
        self.obs_buffer: List[np.ndarray] = []
        self.max_buffer_size = 20

        # Performance tracking
        self.episode_results: List[EpisodeResult] = []
        self.reward_history: List[float] = []
        self.prediction_errors: List[float] = []

        # State
        self.state = AgentState(observation=np.zeros(obs_dim))
        self.stagnation_counter = 0
        self.stagnation_threshold = 10

    def perceive(self, observation: np.ndarray):
        """Step 1: Receive observation from environment."""
        self.state.observation = observation
        self.obs_buffer.append(observation)
        if len(self.obs_buffer) > self.max_buffer_size:
            self.obs_buffer.pop(0)

    def update_causal_model(self):
        """Step 2: Update HHCRA with current observations."""
        if len(self.obs_buffer) < 2:
            return

        # Create batch tensor from buffer
        B = 1
        T = len(self.obs_buffer)
        obs_array = np.array(self.obs_buffer).reshape(B, T, -1)

        # Pad observation dim if needed
        actual_dim = obs_array.shape[-1]
        if actual_dim < self.config.obs_dim:
            padded = np.zeros((B, T, self.config.obs_dim))
            padded[:, :, :actual_dim] = obs_array
            obs_array = padded
        elif actual_dim > self.config.obs_dim:
            obs_array = obs_array[:, :, :self.config.obs_dim]

        obs_tensor = torch.tensor(obs_array, dtype=torch.float32)

        with torch.no_grad():
            fwd = self.model.forward(obs_tensor)
        self.state.causal_graph = fwd['graph']

    def plan(self, possible_actions: Optional[List[int]] = None) -> int:
        """
        Step 3: Use interventional reasoning to evaluate actions.

        For each action: "If I do(action), what is P(goal|do(action))?"
        Select action with highest expected reward.
        """
        if possible_actions is None:
            possible_actions = list(range(self.num_actions))

        if len(self.obs_buffer) < 2:
            return np.random.choice(possible_actions)

        B = 1
        T = len(self.obs_buffer)
        obs_array = np.array(self.obs_buffer).reshape(B, T, -1)
        actual_dim = obs_array.shape[-1]
        if actual_dim < self.config.obs_dim:
            padded = np.zeros((B, T, self.config.obs_dim))
            padded[:, :, :actual_dim] = obs_array
            obs_array = padded
        elif actual_dim > self.config.obs_dim:
            obs_array = obs_array[:, :, :self.config.obs_dim]

        obs_tensor = torch.tensor(obs_array, dtype=torch.float32)

        best_action = possible_actions[0]
        best_value = float('-inf')

        for action in possible_actions:
            # Simulate intervention: do(action_var = action_value)
            action_var = min(action, self.config.num_vars - 1)
            target_var = min(action_var + 1, self.config.num_vars - 1)

            xv = torch.full((self.config.latent_dim,), float(action + 1))

            with torch.no_grad():
                try:
                    result = self.model.query(
                        obs_tensor, CausalQueryType.INTERVENTIONAL,
                        X=action_var, Y=target_var, x_value=xv, verbose=False)
                    if result['answer'] is not None:
                        value = result['answer'].mean().item()
                    else:
                        value = 0.0
                except Exception:
                    value = 0.0

            if value > best_value:
                best_value = value
                best_action = action

        return best_action

    def act_and_observe(self, env, action: int) -> Tuple[float, np.ndarray, bool]:
        """Steps 4-5: Execute action and observe outcome."""
        if hasattr(env, 'step'):
            result = env.step(action)
            if len(result) == 4:
                obs, reward, done, info = result
            else:
                obs, reward, done = result[0], result[1], result[2]
        else:
            obs = env.reset()
            reward = 0.0
            done = False

        self.state.cumulative_reward += reward
        self.state.episode_step += 1
        self.state.total_steps += 1
        self.reward_history.append(reward)

        return reward, obs, done

    def ground_and_correct(self, predicted: Optional[np.ndarray],
                           actual: np.ndarray):
        """Steps 6-7: Compare prediction vs outcome, correct model."""
        if predicted is not None:
            error = float(np.mean((predicted.flatten()[:len(actual.flatten())] -
                                   actual.flatten()[:len(predicted.flatten())]) ** 2))
        else:
            error = 1.0

        self.prediction_errors.append(error)

        # Generate correction feedback
        if error > 0.5:
            self.model.layer2.handle_feedback({
                'issue': 'high_prediction_error',
                'error': error,
            })

    def check_stagnation_and_modify(self, observations: Optional[torch.Tensor] = None):
        """Step 8: If performance stagnates, trigger self-modification."""
        if len(self.reward_history) < self.stagnation_threshold:
            return

        recent = self.reward_history[-self.stagnation_threshold:]
        if np.std(recent) < 0.01 and np.mean(recent) < 0.1:
            self.stagnation_counter += 1

            if self.stagnation_counter >= 3:
                # Trigger lightweight self-modification
                self._adapt_parameters()
                self.stagnation_counter = 0
        else:
            self.stagnation_counter = 0

    def _adapt_parameters(self):
        """Lightweight parameter adaptation when stagnating."""
        with torch.no_grad():
            # Perturb GNN weights to explore different structures
            noise = torch.randn_like(self.model.layer2.gnn.W) * 0.1
            self.model.layer2.gnn.W.data += noise

    def run_episode(self, env, max_steps: int = 100,
                    verbose: bool = False) -> EpisodeResult:
        """Run a complete agent episode."""
        obs = env.reset()
        self.state = AgentState(observation=obs)
        self.obs_buffer = [obs]

        total_reward = 0.0

        for step in range(max_steps):
            # 1. Perceive
            self.perceive(obs)

            # 2. Update causal model
            if step % 5 == 0:  # Update every 5 steps for efficiency
                self.update_causal_model()

            # 3. Plan
            action = self.plan()

            # 4-5. Act and observe
            reward, obs, done = self.act_and_observe(env, action)
            total_reward += reward

            if verbose and step % 10 == 0:
                print(f"  Step {step}: action={action}, reward={reward:.3f}, "
                      f"cumulative={total_reward:.3f}")

            # 6-7. Ground and correct
            self.ground_and_correct(None, obs)

            # 8. Check stagnation
            self.check_stagnation_and_modify()

            if done:
                break

        result = EpisodeResult(
            total_reward=total_reward,
            num_steps=self.state.episode_step,
            graph_discovered=self.state.causal_graph is not None,
            intervention_accuracy=1.0 - np.mean(self.prediction_errors[-10:])
            if self.prediction_errors else 0.0,
        )
        self.episode_results.append(result)
        return result

    def run_episodes(self, env, num_episodes: int = 100,
                     max_steps_per_episode: int = 100,
                     verbose: bool = False) -> List[EpisodeResult]:
        """Run multiple episodes and track learning progress."""
        results = []

        for ep in range(num_episodes):
            result = self.run_episode(env, max_steps_per_episode,
                                      verbose=(verbose and ep % 10 == 0))
            results.append(result)

            if verbose and (ep + 1) % 10 == 0:
                avg_reward = np.mean([r.total_reward for r in results[-10:]])
                print(f"Episode {ep + 1}/{num_episodes}: "
                      f"avg_reward={avg_reward:.3f}")

        return results

    def get_performance_stats(self) -> dict:
        """Return agent performance statistics."""
        if not self.episode_results:
            return {'num_episodes': 0}

        rewards = [r.total_reward for r in self.episode_results]
        steps = [r.num_steps for r in self.episode_results]

        return {
            'num_episodes': len(self.episode_results),
            'mean_reward': float(np.mean(rewards)),
            'max_reward': float(np.max(rewards)),
            'mean_steps': float(np.mean(steps)),
            'total_steps': self.state.total_steps,
            'mean_prediction_error': float(np.mean(self.prediction_errors))
            if self.prediction_errors else 0.0,
        }
