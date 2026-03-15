"""
CODA: Cross-validated Ordering for DAG Alignment.

A varsortability-aware causal discovery framework with honest benchmarking.
"""

__version__ = "1.0.0"

from coda.discovery import coda_discover, sortnregress, r2_sortnregress
from coda.scm import fit_linear_scm, LinearSCM
from coda.inference import interventional_mean, counterfactual
from coda.metrics import shd, f1_score_dag, varsortability, r2_sortability
from coda.baselines import run_pc, run_ges, run_lingam
from coda.data import (
    generate_er_dag,
    generate_linear_sem_data,
    load_sachs,
    SACHS_TRUE_DAG,
    ASIA_TRUE_DAG,
)

__all__ = [
    "coda_discover",
    "sortnregress",
    "r2_sortnregress",
    "fit_linear_scm",
    "LinearSCM",
    "interventional_mean",
    "counterfactual",
    "shd",
    "f1_score_dag",
    "varsortability",
    "r2_sortability",
    "generate_er_dag",
    "generate_linear_sem_data",
    "load_sachs",
    "SACHS_TRUE_DAG",
    "ASIA_TRUE_DAG",
]
