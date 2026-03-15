"""
HHCRA: Hierarchical Hybrid Causal Reasoning Architecture

A three-layer neuro-symbolic architecture for SCM estimation and causal
inference across Pearl's causal hierarchy (association, intervention,
counterfactual).

Experimental extensions (unvalidated):
    - LiquidCausalGraph: graph structure as a dynamical system
    - TrajectoryInvariantFinder: invariant detection from ODE trajectories
    - IterativeRefinementNet: iterative feedback loop
"""

from hhcra.config import HHCRAConfig
from hhcra.causal_graph import CausalGraphData, CausalQueryType
from hhcra.architecture import HHCRA
from hhcra.liquid_causal_graph import LiquidCausalGraph
from hhcra.invariant_finder import TrajectoryInvariantFinder, InvariantRule
from hhcra.iterative_refinement import IterativeRefinementNet
from hhcra.experimental import HHCRAExperimental

__version__ = "0.8.0"
__all__ = [
    "HHCRAConfig", "CausalGraphData", "CausalQueryType", "HHCRA",
    "LiquidCausalGraph", "TrajectoryInvariantFinder", "InvariantRule",
    "IterativeRefinementNet", "HHCRAExperimental",
]
