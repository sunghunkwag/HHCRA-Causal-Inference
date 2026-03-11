"""
CausalBandit: Multi-armed bandit with hidden causal structure.

Agent must discover causal graph to solve optimally.
Baselines: random, epsilon-greedy, UCB (non-causal).
"""

import numpy as np
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass

from hhcra.causal_graph import CausalGraphData


@dataclass
class BanditState:
    """State of the causal bandit."""
    context: np.ndarray  # Observable context variables
    hidden: np.ndarray   # Unobservable confounders


class CausalBandit:
    """
    Multi-armed bandit where reward depends on hidden causal structure.

    True structure: context -> arm_effects, hidden -> arm_effects
    Agent must learn causal structure to identify best arm.
    """

    def __init__(self, num_arms: int = 4, context_dim: int = 3,
                 hidden_dim: int = 2, seed: int = 42):
        self.num_arms = num_arms
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.rng = np.random.RandomState(seed)

        # True causal weights
        self.W_context = self.rng.randn(context_dim, num_arms) * 0.5
        self.W_hidden = self.rng.randn(hidden_dim, num_arms) * 0.3
        self.arm_bias = self.rng.randn(num_arms) * 0.1

        # True causal graph: context -> arms, hidden -> arms
        self.num_vars = context_dim + hidden_dim + num_arms
        self.state = None
        self.episode_rewards = []
        self.reset()

    def reset(self) -> np.ndarray:
        """Reset and return observable context."""
        context = self.rng.randn(self.context_dim)
        hidden = self.rng.randn(self.hidden_dim)
        self.state = BanditState(context=context, hidden=hidden)
        self.episode_rewards = []
        return context

    def step(self, arm: int) -> Tuple[float, np.ndarray, dict]:
        """
        Pull arm, get reward.

        Returns: (reward, next_context, info)
        """
        # Reward = causal effect of context + hidden on arm
        reward = (
            self.state.context @ self.W_context[:, arm] +
            self.state.hidden @ self.W_hidden[:, arm] +
            self.arm_bias[arm] +
            self.rng.randn() * 0.1
        )

        self.episode_rewards.append(reward)

        # New context
        new_context = self.rng.randn(self.context_dim)
        new_hidden = self.rng.randn(self.hidden_dim)
        self.state = BanditState(context=new_context, hidden=new_hidden)

        info = {
            'optimal_arm': self._optimal_arm(),
            'regret': self._optimal_reward() - reward,
        }

        return reward, new_context, info

    def intervene(self, arm: int, context: Optional[np.ndarray] = None) -> float:
        """Interventional query: do(arm) with given context."""
        if context is None:
            context = self.state.context
        return context @ self.W_context[:, arm] + self.arm_bias[arm]

    def _optimal_arm(self) -> int:
        """Return the arm with highest expected reward given current context."""
        expected = self.state.context @ self.W_context + self.arm_bias
        return int(np.argmax(expected))

    def _optimal_reward(self) -> float:
        """Return expected reward of optimal arm."""
        expected = self.state.context @ self.W_context + self.arm_bias
        return float(np.max(expected))

    def get_observation_dim(self) -> int:
        return self.context_dim

    def get_true_graph(self) -> CausalGraphData:
        """Return the true causal graph."""
        N = self.num_vars
        adj = np.zeros((N, N))
        edges = []

        # Context -> arms
        for c in range(self.context_dim):
            for a in range(self.num_arms):
                arm_idx = self.context_dim + self.hidden_dim + a
                if abs(self.W_context[c, a]) > 0.1:
                    adj[arm_idx, c] = 1.0
                    edges.append((c, arm_idx, abs(self.W_context[c, a])))

        # Hidden -> arms
        for h in range(self.hidden_dim):
            h_idx = self.context_dim + h
            for a in range(self.num_arms):
                arm_idx = self.context_dim + self.hidden_dim + a
                if abs(self.W_hidden[h, a]) > 0.1:
                    adj[arm_idx, h_idx] = 1.0
                    edges.append((h_idx, arm_idx, abs(self.W_hidden[h, a])))

        return CausalGraphData(list(range(N)), edges, adj)

    def cumulative_regret(self) -> float:
        """Total regret over episode."""
        return sum(self._optimal_reward() - r for r in self.episode_rewards)


class RandomBaseline:
    """Random arm selection baseline."""

    def __init__(self, num_arms: int, seed: int = 42):
        self.num_arms = num_arms
        self.rng = np.random.RandomState(seed)

    def select_arm(self, context: np.ndarray) -> int:
        return self.rng.randint(self.num_arms)


class EpsilonGreedyBaseline:
    """Epsilon-greedy baseline (non-causal)."""

    def __init__(self, num_arms: int, epsilon: float = 0.1, seed: int = 42):
        self.num_arms = num_arms
        self.epsilon = epsilon
        self.rng = np.random.RandomState(seed)
        self.arm_rewards: Dict[int, List[float]] = {i: [] for i in range(num_arms)}

    def select_arm(self, context: np.ndarray) -> int:
        if self.rng.random() < self.epsilon:
            return self.rng.randint(self.num_arms)
        means = {a: np.mean(r) if r else 0.0 for a, r in self.arm_rewards.items()}
        return max(means, key=means.get)

    def update(self, arm: int, reward: float):
        self.arm_rewards[arm].append(reward)


class UCBBaseline:
    """UCB1 baseline (non-causal)."""

    def __init__(self, num_arms: int):
        self.num_arms = num_arms
        self.counts = np.zeros(num_arms)
        self.values = np.zeros(num_arms)
        self.total = 0

    def select_arm(self, context: np.ndarray) -> int:
        self.total += 1
        if np.min(self.counts) == 0:
            return int(np.argmin(self.counts))
        ucb = self.values + np.sqrt(2 * np.log(self.total) / self.counts)
        return int(np.argmax(ucb))

    def update(self, arm: int, reward: float):
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n
