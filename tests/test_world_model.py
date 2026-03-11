"""Tests for Phase 7: World Model Integration."""

import pytest
import torch
import numpy as np

from hhcra.config import HHCRAConfig
from hhcra.architecture import HHCRA
from hhcra.world_model import (
    SimplePhysicsWorld, WorldModel, GroundingResult, EnvironmentInterface
)
from hhcra.main import generate_causal_data


class TestSimplePhysicsWorld:
    def test_reset_returns_observation(self):
        world = SimplePhysicsWorld(num_objects=4)
        obs = world.reset()
        assert isinstance(obs, np.ndarray)
        assert len(obs) == 4 * 4  # 4 objects * (x,y,vx,vy)

    def test_step_returns_tuple(self):
        world = SimplePhysicsWorld()
        world.reset()
        obs, reward, done, info = world.step()
        assert isinstance(obs, np.ndarray)

    def test_intervene_returns_observation(self):
        world = SimplePhysicsWorld()
        world.reset()
        obs = world.intervene(0, 2.0)
        assert isinstance(obs, np.ndarray)

    def test_has_true_graph(self):
        world = SimplePhysicsWorld()
        graph = world.get_true_graph()
        assert graph is not None
        assert len(graph.nodes) == 4

    def test_is_environment_interface(self):
        world = SimplePhysicsWorld()
        assert isinstance(world, EnvironmentInterface)


class TestWorldModel:
    def test_ground_prediction(self):
        world = SimplePhysicsWorld()
        world.reset()
        wm = WorldModel(world)

        prediction = np.random.randn(16)
        result = wm.ground_prediction(prediction, var_idx=0, intervention_value=2.0)
        assert isinstance(result, GroundingResult)
        assert result.error >= 0

    def test_error_history_accumulates(self):
        world = SimplePhysicsWorld()
        world.reset()
        wm = WorldModel(world)

        for _ in range(5):
            pred = np.random.randn(16)
            wm.ground_prediction(pred, 0, 1.0)

        assert len(wm.error_history) == 5

    def test_error_classification(self):
        world = SimplePhysicsWorld()
        world.reset()
        wm = WorldModel(world)

        result = wm.ground_prediction(np.random.randn(16) * 10, 0, 2.0)
        assert result.error_type in ('structure', 'mechanism', 'variable')

    def test_convergence_stats(self):
        world = SimplePhysicsWorld()
        world.reset()
        wm = WorldModel(world)

        for _ in range(5):
            wm.ground_prediction(np.random.randn(16), 0, 1.0)

        stats = wm.get_convergence_stats()
        assert 'mean_error' in stats
        assert 'trend' in stats
        assert stats['num_groundings'] == 5

    def test_grounding_loop(self):
        """Test the full predict-act-observe-compare loop."""
        config = HHCRAConfig(train_epochs_l1=3, train_epochs_l2=5, train_epochs_l3=3)
        observations, _ = generate_causal_data(B=4, T=8)

        model = HHCRA(config)
        model.train_all(observations, verbose=False)
        model.eval()

        world = SimplePhysicsWorld()
        world.reset()
        wm = WorldModel(world)

        errors = wm.grounding_loop(
            model, observations, var_idx=0,
            intervention_value=2.0, num_iterations=3, verbose=False)

        assert len(errors) == 3
        assert all(e >= 0 for e in errors)
