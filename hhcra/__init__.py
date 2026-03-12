"""
HHCRA: Hierarchical Hybrid Causal Reasoning Architecture

A 5-component, 3-layer causal reasoning system covering all three rungs
of Pearl's Ladder of Causation.
"""

from hhcra.config import HHCRAConfig
from hhcra.causal_graph import CausalGraphData, CausalQueryType
from hhcra.architecture import HHCRA

__version__ = "0.4.1"
__all__ = ["HHCRAConfig", "CausalGraphData", "CausalQueryType", "HHCRA"]
