"""HHCRA v2.0: Hierarchical Hybrid Causal Reasoning Architecture."""
__version__ = "2.0.0"
from hhcra.architecture import HHCRA, HHCRAResult, ICAExtractor
from hhcra.graph import CausalGraphData, CausalQueryType
from hhcra.symbolic import NeuroSymbolicEngine
from hhcra.agent import ActiveCausalAgent
