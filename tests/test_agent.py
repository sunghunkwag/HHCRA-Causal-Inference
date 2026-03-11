"""Tests for Phase 10: Agent Loop (Closed-Loop Causal Agent)."""

import pytest
import torch
import numpy as np

from hhcra.config import HHCRAConfig
from hhcra.agent import CausalAgent, EpisodeResult
from environments.causal_bandit import (
    CausalBandit, RandomBaseline, EpsilonGreedyBaseline, UCBBaseline,
)
from environments.causal_gridworld import CausalGridWorld


@pytest.fixture
def small_config():
    return HHCRAConfig(
        obs_dim=48, num_vars=8, latent_dim=10,
        train_epochs_l1=3, train_epochs_l2=5, train_epochs_l3=3,
    )


class TestCausalAgent:
    def test_init(self, small_config):
        agent = CausalAgent(small_config, num_actions=4)
        assert agent.model is not None
        assert agent.num_actions == 4

    def test_perceive(self, small_config):
        agent = CausalAgent(small_config)
        obs = np.random.randn(48)
        agent.perceive(obs)
        assert len(agent.obs_buffer) == 1

    def test_plan_with_no_data(self, small_config):
        agent = CausalAgent(small_config, num_actions=4)
        action = agent.plan()
        assert 0 <= action < 4

    def test_plan_with_data(self, small_config):
        agent = CausalAgent(small_config, num_actions=4)
        for _ in range(5):
            agent.perceive(np.random.randn(48))
        action = agent.plan()
        assert 0 <= action < 4

    def test_ground_and_correct(self, small_config):
        agent = CausalAgent(small_config)
        pred = np.random.randn(10)
        actual = np.random.randn(10)
        agent.ground_and_correct(pred, actual)
        assert len(agent.prediction_errors) == 1

    def test_performance_stats_empty(self, small_config):
        agent = CausalAgent(small_config)
        stats = agent.get_performance_stats()
        assert stats['num_episodes'] == 0


class TestCausalBandit:
    def test_init(self):
        bandit = CausalBandit(num_arms=4)
        assert bandit.num_arms == 4

    def test_reset(self):
        bandit = CausalBandit()
        context = bandit.reset()
        assert isinstance(context, np.ndarray)
        assert len(context) == bandit.context_dim

    def test_step(self):
        bandit = CausalBandit()
        bandit.reset()
        reward, context, info = bandit.step(0)
        assert isinstance(reward, float)
        assert isinstance(context, np.ndarray)
        assert 'optimal_arm' in info

    def test_true_graph(self):
        bandit = CausalBandit()
        graph = bandit.get_true_graph()
        assert graph is not None
        assert len(graph.nodes) > 0

    def test_intervene(self):
        bandit = CausalBandit()
        bandit.reset()
        effect = bandit.intervene(0)
        assert isinstance(effect, float)


class TestCausalBanditBaselines:
    def test_random_baseline(self):
        baseline = RandomBaseline(num_arms=4)
        context = np.random.randn(3)
        arm = baseline.select_arm(context)
        assert 0 <= arm < 4

    def test_epsilon_greedy(self):
        baseline = EpsilonGreedyBaseline(num_arms=4)
        context = np.random.randn(3)
        arm = baseline.select_arm(context)
        assert 0 <= arm < 4
        baseline.update(arm, 1.0)
        assert len(baseline.arm_rewards[arm]) == 1

    def test_ucb(self):
        baseline = UCBBaseline(num_arms=4)
        context = np.random.randn(3)
        arm = baseline.select_arm(context)
        assert 0 <= arm < 4
        baseline.update(arm, 1.0)
        assert baseline.counts[arm] == 1


class TestCausalGridWorld:
    def test_init(self):
        grid = CausalGridWorld()
        assert grid.width > 0
        assert grid.height > 0

    def test_reset(self):
        grid = CausalGridWorld()
        obs = grid.reset()
        assert isinstance(obs, np.ndarray)

    def test_step(self):
        grid = CausalGridWorld()
        grid.reset()
        obs, reward, done, info = grid.step(0)
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(done, bool)

    def test_intervene(self):
        grid = CausalGridWorld()
        grid.reset()
        obs = grid.intervene(0, 2, 2)
        assert isinstance(obs, np.ndarray)

    def test_true_graph(self):
        grid = CausalGridWorld()
        graph = grid.get_true_graph()
        assert graph is not None

    def test_observation_dim(self):
        grid = CausalGridWorld(num_objects=4)
        obs = grid.reset()
        assert len(obs) == grid.get_observation_dim()


class TestAgentOnBandit:
    def test_agent_runs_episode(self):
        """Agent can complete an episode on CausalBandit."""
        config = HHCRAConfig(
            obs_dim=48, num_vars=8, latent_dim=10,
            train_epochs_l1=2, train_epochs_l2=3, train_epochs_l3=2,
        )
        bandit = CausalBandit(num_arms=4)
        agent = CausalAgent(config, obs_dim=bandit.get_observation_dim(),
                            num_actions=bandit.num_arms)

        result = agent.run_episode(bandit, max_steps=10, verbose=False)
        assert isinstance(result, EpisodeResult)
        assert result.num_steps > 0

    def test_agent_multiple_episodes(self):
        config = HHCRAConfig(
            obs_dim=48, num_vars=8, latent_dim=10,
            train_epochs_l1=2, train_epochs_l2=3, train_epochs_l3=2,
        )
        bandit = CausalBandit(num_arms=4)
        agent = CausalAgent(config, obs_dim=bandit.get_observation_dim(),
                            num_actions=bandit.num_arms)

        results = agent.run_episodes(bandit, num_episodes=3,
                                     max_steps_per_episode=10, verbose=False)
        assert len(results) == 3

        stats = agent.get_performance_stats()
        assert stats['num_episodes'] == 3


class TestAgentOnGridWorld:
    def test_agent_runs_episode(self):
        config = HHCRAConfig(
            obs_dim=48, num_vars=8, latent_dim=10,
            train_epochs_l1=2, train_epochs_l2=3, train_epochs_l3=2,
        )
        grid = CausalGridWorld()
        agent = CausalAgent(config, obs_dim=grid.get_observation_dim(),
                            num_actions=grid.num_actions)

        result = agent.run_episode(grid, max_steps=10, verbose=False)
        assert isinstance(result, EpisodeResult)
