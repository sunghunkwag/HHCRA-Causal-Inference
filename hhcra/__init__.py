"""
HHCRA: Hierarchical Hybrid Causal Reasoning Architecture

A three-layer neuro-symbolic architecture for SCM estimation and causal
inference across Pearl's causal hierarchy (association, intervention,
counterfactual).

Experimental extensions (unvalidated):
    - LiquidCausalGraph: graph structure as a dynamical system
    - SymbolicGenesisEngine: invariant detection from ODE trajectories
    - AutocatalyticCausalNet: iterative feedback loop
"""

from hhcra.config import HHCRAConfig
from hhcra.causal_graph import CausalGraphData, CausalQueryType
from hhcra.architecture import HHCRA
from hhcra.liquid_causal_graph import LiquidCausalGraph
from hhcra.symbolic_genesis import SymbolicGenesisEngine, SymbolicRule
from hhcra.autocatalytic_causal_net import AutocatalyticCausalNet
from hhcra.singularity import HHCRASingularity

__version__ = "0.8.0"
__all__ = [
    "HHCRAConfig", "CausalGraphData", "CausalQueryType", "HHCRA",
    "LiquidCausalGraph", "SymbolicGenesisEngine", "SymbolicRule",
    "AutocatalyticCausalNet", "HHCRASingularity",
]
