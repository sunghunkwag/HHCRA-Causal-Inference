"""
HHCRA v2.0: Hierarchical Hybrid Causal Reasoning Architecture

Integrated pipeline connecting:
  Layer 1: ICA Variable Extraction (replaces broken C-JEPA SlotAttention)
  Layer 2: CODA Structure Learning + OLS SCM Fitting
  Layer 3: NeuroSymbolicEngine (d-sep, backdoor, do-calculus, ABP)

Original architecture failures fixed:
  - C-JEPA: replaced with ICA (no V-alignment problem)
  - NOTEARS neural training: replaced with CODA (CV ordering + held-out BIC)
  - HRM: replaced with direct symbolic reasoning dispatch
  - Liquid ODE: replaced with OLS SCM (honest about linear assumption)

Pearl's Ladder coverage:
  Rung 1: P(Y|X) via regression on learned graph
  Rung 2: P(Y|do(X)) via truncated factorization
  Rung 3: P(Y_{x'}|X=x,Y=y) via ABP (Abduction-Action-Prediction)

Pure numpy/scipy. No PyTorch dependency.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Dict, Set, Any, List
from dataclasses import dataclass, field
from enum import Enum
import time

from coda.discovery import coda_discover
from coda.scm import fit_linear_scm, LinearSCM
from coda.inference import interventional_mean, counterfactual
from coda.metrics import shd, f1_score_dag, varsortability

from hhcra.graph import CausalGraphData, CausalQueryType
from hhcra.symbolic import NeuroSymbolicEngine


# ===================================================================
# ICA Variable Extraction (Layer 1)
# ===================================================================

class ICAExtractor:
    """
    Layer 1: Extract causal variables from high-dimensional observations.

    Uses ICA (FastICA with logcosh) for source separation.
    Under LiNGAM (linear SEM + non-Gaussian noise), ICA recovers true
    causal variables up to permutation and scaling.

    Falls back to PCA when ICA cannot converge.
    """

    def __init__(self, max_vars: int = 20, n_vars_hint: int = None):
        self.max_vars = max_vars
        self.n_vars_hint = n_vars_hint
        self._fitted = False
        self._pca_components = None
        self._ica_unmixing = None
        self._mean = None
        self.n_detected = None

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit ICA and extract variables. X: (n_samples, obs_dim)."""
        n, obs_dim = X.shape
        self._mean = X.mean(axis=0)
        X_c = X - self._mean

        N_max = min(self.max_vars, obs_dim, n - 1)
        U, S, Vt = np.linalg.svd(X_c, full_matrices=False)
        eigenvalues = (S ** 2) / n

        # Detect dimensionality
        if self.n_vars_hint is not None:
            nc = min(self.n_vars_hint, N_max)
        else:
            nc = self._detect_dim(eigenvalues, N_max)
        self.n_detected = nc

        # PCA projection
        self._pca_components = Vt[:nc]
        X_pca = X_c @ self._pca_components.T

        # Try ICA
        try:
            X_ica, W = self._fastica(X_pca)
            self._ica_unmixing = W
            self._fitted = True
            return X_ica
        except Exception:
            self._ica_unmixing = np.eye(nc)
            self._fitted = True
            return X_pca

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform new observations using fitted ICA."""
        assert self._fitted
        X_c = X - self._mean
        X_pca = X_c @ self._pca_components.T
        return X_pca @ self._ica_unmixing.T

    def _detect_dim(self, eigenvalues, max_k):
        ev = eigenvalues[:max_k]
        if len(ev) < 4:
            return max(2, len(ev))
        ev_sorted = np.sort(ev)[::-1]
        n_bottom = max(len(ev_sorted) // 2, 3)
        noise_floor = np.median(ev_sorted[-n_bottom:])
        if noise_floor <= 0:
            noise_floor = ev_sorted[-1] + 1e-10
        n_signal = sum(1 for e in ev_sorted if e > 3.0 * noise_floor)
        n_gap = max_k
        if len(ev_sorted) >= 3:
            ratios = ev_sorted[1:] / (ev_sorted[:-1] + 1e-10)
            min_idx = np.argmin(ratios)
            if ratios[min_idx] < 0.3:
                n_gap = min_idx + 1
        return max(2, min(max(n_signal, n_gap), max_k))

    def _fastica(self, X, max_iter=200, tol=1e-4):
        n, d = X.shape
        cov = np.cov(X.T)
        evals, evecs = np.linalg.eigh(cov)
        evals = np.maximum(evals, 1e-10)
        W_white = np.diag(1.0 / np.sqrt(evals)) @ evecs.T
        Xw = (W_white @ X.T).T
        np.random.seed(42)
        W = np.random.randn(d, d)
        Uw, Sw, Vtw = np.linalg.svd(W)
        W = Uw @ Vtw
        for _ in range(max_iter):
            W_old = W.copy()
            for p in range(d):
                w = W[p]
                u = Xw @ w
                g = np.tanh(u)
                gp = 1.0 - g ** 2
                W[p] = (Xw.T @ g) / n - gp.mean() * w
            Uw, Sw, Vtw = np.linalg.svd(W)
            W = Uw @ Vtw
            if max(1.0 - abs(np.dot(W[i], W_old[i])) for i in range(d)) < tol:
                break
        return (W @ Xw.T).T, W @ W_white


# ===================================================================
# HHCRA Main Architecture
# ===================================================================

@dataclass
class HHCRAResult:
    """Result of a causal query."""
    query_type: str
    answer: Optional[np.ndarray]
    identifiable: bool
    strategy: Optional[str]
    details: Dict[str, Any] = field(default_factory=dict)


class HHCRA:
    """
    Hierarchical Hybrid Causal Reasoning Architecture v2.0.

    3-layer architecture:
      Layer 1: ICA Variable Extraction
      Layer 2: CODA Structure Learning + OLS SCM
      Layer 3: Symbolic Reasoning (d-sep, backdoor, ABP)

    Usage:
        model = HHCRA()
        model.fit(X)  # X: (n_samples, d) or (n_samples, obs_dim) for high-dim
        result = model.query(CausalQueryType.INTERVENTIONAL, X=0, Y=3, x_value=2.0)
        cf = model.counterfactual(factual_values, src=0, cf_x=3.0)
    """

    def __init__(self, n_vars_hint: int = None, max_vars: int = 20,
                 coda_kwargs: Dict = None):
        self.n_vars_hint = n_vars_hint
        self.max_vars = max_vars
        self.coda_kwargs = coda_kwargs or {}

        # Components
        self.extractor = None        # Layer 1: ICA
        self.scm = None              # Layer 2: fitted SCM
        self.graph_data = None       # Layer 2: CausalGraphData
        self.adj = None              # Layer 2: adjacency matrix
        self.symbolic = NeuroSymbolicEngine()  # Layer 3

        # State
        self._fitted = False
        self._var_data = None
        self._n_vars = None
        self._timing = {}

    def fit(self, X: np.ndarray, raw_data: np.ndarray = None,
            verbose: bool = True) -> 'HHCRA':
        """
        Fit the full pipeline.

        Parameters
        ----------
        X : (n_samples, obs_dim) observation matrix.
        raw_data : (n_samples, n_vars) if available, bypasses ICA.
        verbose : Print progress.

        Returns self for chaining.
        """
        if verbose:
            print("=" * 60)
            print("HHCRA v2.0 — Fitting")
            print("=" * 60)

        n, obs_dim = X.shape

        # ===== Layer 1: Variable Extraction =====
        t0 = time.time()
        if raw_data is not None:
            var_data = raw_data
            self._n_vars = var_data.shape[1]
            self.extractor = None
            if verbose:
                print(f"\n  L1: Raw data bypass ({self._n_vars} vars)")
        elif obs_dim <= self.max_vars:
            var_data = X
            self._n_vars = obs_dim
            self.extractor = None
            if verbose:
                print(f"\n  L1: Direct mode ({obs_dim} vars)")
        else:
            self.extractor = ICAExtractor(
                max_vars=self.max_vars, n_vars_hint=self.n_vars_hint)
            var_data = self.extractor.fit_transform(X)
            self._n_vars = var_data.shape[1]
            if verbose:
                print(f"\n  L1: ICA extracted {self._n_vars} vars from {obs_dim}-dim obs")
        self._var_data = var_data
        self._timing['layer1'] = time.time() - t0

        # ===== Layer 2a: Structure Learning (CODA) =====
        t0 = time.time()
        coda_result = coda_discover(var_data, seed=42, **self.coda_kwargs)
        self.adj = coda_result['adj']
        self._timing['layer2_structure'] = time.time() - t0

        # Build CausalGraphData for symbolic engine
        self.graph_data = CausalGraphData.from_adjacency(self.adj)

        if verbose:
            n_edges = int(self.adj.sum())
            dag = self.graph_data.is_dag()
            print(f"  L2a: CODA discovered {n_edges} edges, DAG={dag} "
                  f"({self._timing['layer2_structure']:.2f}s)")
            print(f"       Strategy: {coda_result['strategy']}")

        # ===== Layer 2b: SCM Fitting (OLS) =====
        t0 = time.time()
        self.scm = fit_linear_scm(var_data, self.adj)
        self._timing['layer2_scm'] = time.time() - t0

        if verbose:
            print(f"  L2b: OLS SCM fitted ({self._timing['layer2_scm']:.3f}s)")

        # ===== Layer 3: Symbolic Engine Ready =====
        if verbose:
            print(f"\n  L3: Symbolic engine ready")
            print(f"      Pearl's Ladder: Rung 1 ✓ | Rung 2 ✓ | Rung 3 ✓")
            total = sum(self._timing.values())
            print(f"\n  Total: {total:.2f}s")
            print("=" * 60)

        self._fitted = True
        return self

    def query(self, query_type: CausalQueryType, X: int, Y: int,
              x_value: float = None, factual_values: np.ndarray = None,
              cf_x: float = None) -> HHCRAResult:
        """
        Answer a causal query.

        Rung 1: P(Y|X) — observational
        Rung 2: P(Y|do(X=x)) — interventional
        Rung 3: P(Y_{x'}|X=x, Y=y) — counterfactual
        """
        assert self._fitted, "Call fit() first."

        # Check identifiability via symbolic engine
        id_check = self.symbolic.check_identifiability(self.graph_data, X, Y)

        if query_type == CausalQueryType.OBSERVATIONAL:
            return self._query_observational(X, Y)

        elif query_type == CausalQueryType.INTERVENTIONAL:
            assert x_value is not None
            return self._query_interventional(X, Y, x_value, id_check)

        elif query_type == CausalQueryType.COUNTERFACTUAL:
            assert factual_values is not None and cf_x is not None
            return self._query_counterfactual(X, Y, factual_values, cf_x, id_check)

        raise ValueError(f"Unknown query type: {query_type}")

    def _query_observational(self, X, Y):
        """Rung 1: P(Y|X) via regression."""
        x = self._var_data[:, X]
        y = self._var_data[:, Y]
        x_c = x - x.mean()
        y_c = y - y.mean()
        beta = np.dot(x_c, y_c) / (np.dot(x_c, x_c) + 1e-10)
        intercept = y.mean() - beta * x.mean()
        return HHCRAResult(
            query_type="P(Y|X)", answer=np.array([beta, intercept]),
            identifiable=True, strategy='regression',
            details={'beta': beta, 'intercept': intercept})

    def _query_interventional(self, X, Y, x_value, id_check):
        """Rung 2: P(Y|do(X)) via truncated factorization on SCM."""
        if not id_check['identifiable']:
            return HHCRAResult(query_type="P(Y|do(X))", answer=None,
                               identifiable=False, strategy=None)

        mean = interventional_mean(
            self.scm, target=Y, intervention_node=X,
            intervention_value=x_value, n_samples=10000, seed=42)
        return HHCRAResult(
            query_type="P(Y|do(X))", answer=np.array([mean]),
            identifiable=True, strategy=id_check['strategy'],
            details={'adjustment_set': id_check.get('adjustment_set')})

    def _query_counterfactual(self, X, Y, factual_values, cf_x, id_check):
        """Rung 3: P(Y_{x'}|X=x,Y=y) via ABP."""
        if not id_check['identifiable']:
            return HHCRAResult(query_type="P(Y_{x'})", answer=None,
                               identifiable=False, strategy=None)

        cf = counterfactual(self.scm, factual_values, X, cf_x)
        return HHCRAResult(
            query_type="P(Y_{x'})", answer=np.array([cf[Y]]),
            identifiable=True, strategy='ABP',
            details={'all_counterfactual': cf})

    # === Convenience methods ===

    def evaluate(self, true_adj: np.ndarray) -> Dict:
        """Evaluate against ground truth."""
        assert self._fitted
        s = shd(self.adj, true_adj)
        f1 = f1_score_dag(self.adj, true_adj)
        v = varsortability(self._var_data, true_adj)
        return {
            'shd': s, 'f1': f1['f1'],
            'precision': f1['precision'], 'recall': f1['recall'],
            'varsortability': round(v, 3),
            'n_edges_pred': int(self.adj.sum()),
            'n_edges_true': int(true_adj.sum()),
        }

    def summary(self) -> str:
        """Human-readable summary."""
        if not self._fitted:
            return "HHCRA v2.0 — Not fitted"
        G = self.graph_data
        lines = [
            "", "=" * 60, "HHCRA v2.0 — Summary", "=" * 60, "",
            "Architecture:",
            f"  L1: {'ICA' if self.extractor else 'Direct'} ({self._n_vars} vars)",
            f"  L2: CODA + OLS SCM ({G.edge_count()} edges, DAG={G.is_dag()})",
            f"  L3: Symbolic (d-sep, backdoor, frontdoor, ABP)",
            "",
            "Pearl's Ladder: Rung 1 ✓ | Rung 2 ✓ | Rung 3 ✓",
            "=" * 60,
        ]
        return "\n".join(lines)
