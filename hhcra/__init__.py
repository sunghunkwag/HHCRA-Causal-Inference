"""
HHCRA: Hierarchical Hybrid Causal Reasoning Architecture

A 5-component, 3-layer causal reasoning system covering all three rungs
of Pearl's Ladder of Causation.

Singularity Extensions:
    - Liquid Causal Graph: graph structure as a dynamical system
    - Symbolic Genesis Engine: automatic rule discovery from dynamics
    - Autocatalytic Causal Network: self-catalytic convergence loop
"""

from hhcra.config import HHCRAConfig
from hhcra.causal_graph import CausalGraphData, CausalQueryType
from hhcra.architecture import HHCRA
from hhcra.liquid_causal_graph import LiquidCausalGraph
from hhcra.symbolic_genesis import SymbolicGenesisEngine, SymbolicRule
from hhcra.autocatalytic_causal_net import AutocatalyticCausalNet
from hhcra.singularity import HHCRASingularity

__version__ = "1.0.0"
__all__ = [
    "HHCRAConfig", "CausalGraphData", "CausalQueryType", "HHCRA",
    "LiquidCausalGraph", "SymbolicGenesisEngine", "SymbolicRule",
    "AutocatalyticCausalNet", "HHCRASingularity",
]
