#!/usr/bin/env python3
"""
Honest benchmark: CODA vs baselines with varsortability diagnostics.

Reports SHD, F1, varsortability, and R²-sortability for every experiment.
Includes sortnregress and R²-sortnregress as mandatory baselines
(Reisach et al., 2021/2023).

Usage:
    python scripts/run_benchmark.py
"""

import sys
import time
import numpy as np

sys.path.insert(0, ".")

from coda.data import (
    generate_er_dag,
    generate_sf_dag,
    generate_linear_sem_data,
    load_sachs,
    SACHS_TRUE_DAG,
    ASIA_TRUE_DAG,
)
from coda.discovery import coda_discover, sortnregress, r2_sortnregress
from coda.metrics import shd, f1_score_dag, varsortability, r2_sortability


def run_single_experiment(
    name: str,
    X: np.ndarray,
    true_dag: np.ndarray,
    standardized: bool = False,
) -> dict:
    """Run all methods on one dataset and return results."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"  n={X.shape[0]}, d={X.shape[1]}, "
          f"true_edges={int(true_dag.sum())}")
    print(f"  standardized={standardized}")
    print(f"{'='*60}")

    # Diagnostics
    v = varsortability(X, true_dag)
    r2 = r2_sortability(X, true_dag)
    print(f"  Varsortability:    {v:.3f}")
    print(f"  R²-sortability:    {r2:.3f}")
    if v > 0.9:
        print("  ⚠ HIGH varsortability — sortnregress will be strong baseline")

    results = {}

    # Baseline 1: sortnregress
    t0 = time.time()
    adj_snr = sortnregress(X)
    t_snr = time.time() - t0
    s = shd(adj_snr, true_dag)
    f1 = f1_score_dag(adj_snr, true_dag)
    results["sortnregress"] = {"shd": s, "f1": f1["f1"], "time": t_snr}
    print(f"\n  sortnregress:      SHD={s:>3d}  F1={f1['f1']:.3f}  "
          f"edges={int(adj_snr.sum()):>3d}  time={t_snr:.2f}s")

    # Baseline 2: R²-sortnregress
    t0 = time.time()
    adj_r2 = r2_sortnregress(X)
    t_r2 = time.time() - t0
    s = shd(adj_r2, true_dag)
    f1 = f1_score_dag(adj_r2, true_dag)
    results["r2_sortnregress"] = {"shd": s, "f1": f1["f1"], "time": t_r2}
    print(f"  R²-sortnregress:   SHD={s:>3d}  F1={f1['f1']:.3f}  "
          f"edges={int(adj_r2.sum()):>3d}  time={t_r2:.2f}s")

    # Baseline 3: empty graph
    adj_empty = np.zeros_like(true_dag)
    s = shd(adj_empty, true_dag)
    results["empty"] = {"shd": s, "f1": 0.0, "time": 0.0}
    print(f"  empty graph:       SHD={s:>3d}  F1=0.000  "
          f"edges=  0  time=0.00s")

    # CODA
    t0 = time.time()
    coda_result = coda_discover(X, n_restarts=10, seed=42, verbose=False)
    t_coda = time.time() - t0
    adj_coda = coda_result["adj"]
    s = shd(adj_coda, true_dag)
    f1 = f1_score_dag(adj_coda, true_dag)
    results["coda"] = {
        "shd": s, "f1": f1["f1"], "time": t_coda,
        "strategy": coda_result["strategy"],
    }
    print(f"  CODA:              SHD={s:>3d}  F1={f1['f1']:.3f}  "
          f"edges={int(adj_coda.sum()):>3d}  time={t_coda:.2f}s  "
          f"(strategy={coda_result['strategy']})")

    # Comparison verdict
    print(f"\n  Verdict: ", end="")
    coda_shd = results["coda"]["shd"]
    snr_shd = results["sortnregress"]["shd"]
    if coda_shd < snr_shd:
        print(f"CODA wins by {snr_shd - coda_shd} SHD")
    elif coda_shd > snr_shd:
        print(f"sortnregress wins by {coda_shd - snr_shd} SHD")
    else:
        print("TIE")

    results["varsortability"] = v
    results["r2_sortability"] = r2
    return results


def main():
    print("CODA Benchmark Suite — Honest Evaluation")
    print("=" * 60)
    print("All results include varsortability diagnostics.")
    print("sortnregress/R²-sortnregress are mandatory baselines.")
    print()

    all_results = {}

    # ---------------------------------------------------------------
    # 1. Asia — raw data (high varsortability expected)
    # ---------------------------------------------------------------
    X, _ = generate_linear_sem_data(ASIA_TRUE_DAG, n=2000, seed=42)
    all_results["asia_raw"] = run_single_experiment(
        "Asia (8 nodes, 8 edges) — RAW data", X, ASIA_TRUE_DAG
    )

    # ---------------------------------------------------------------
    # 2. Asia — standardized data (varsortability removed)
    # ---------------------------------------------------------------
    X, _ = generate_linear_sem_data(ASIA_TRUE_DAG, n=2000, seed=42, standardize=True)
    all_results["asia_std"] = run_single_experiment(
        "Asia (8 nodes, 8 edges) — STANDARDIZED data", X, ASIA_TRUE_DAG,
        standardized=True,
    )

    # ---------------------------------------------------------------
    # 3. Asia — low-weight data (reduced varsortability)
    # ---------------------------------------------------------------
    X, _ = generate_linear_sem_data(
        ASIA_TRUE_DAG, n=2000, weight_range=(0.2, 0.8),
        noise_std_range=(0.8, 1.2), seed=42,
    )
    all_results["asia_loww"] = run_single_experiment(
        "Asia (8 nodes, 8 edges) — LOW WEIGHT range (0.2-0.8)", X, ASIA_TRUE_DAG,
    )

    # ---------------------------------------------------------------
    # 4. Sachs synthetic — raw
    # ---------------------------------------------------------------
    X = load_sachs(standardize=False)
    all_results["sachs_raw"] = run_single_experiment(
        "Sachs synthetic (11 nodes, 17 edges) — RAW", X, SACHS_TRUE_DAG,
    )

    # ---------------------------------------------------------------
    # 5. Sachs synthetic — standardized
    # ---------------------------------------------------------------
    X = load_sachs(standardize=True)
    all_results["sachs_std"] = run_single_experiment(
        "Sachs synthetic (11 nodes, 17 edges) — STANDARDIZED", X, SACHS_TRUE_DAG,
        standardized=True,
    )

    # ---------------------------------------------------------------
    # 6. ER-2 random graph (d=20)
    # ---------------------------------------------------------------
    dag20 = generate_er_dag(20, expected_edges=30, seed=42)
    X, _ = generate_linear_sem_data(dag20, n=2000, seed=42)
    all_results["er20_raw"] = run_single_experiment(
        "ER-2 (20 nodes) — RAW", X, dag20,
    )

    X, _ = generate_linear_sem_data(dag20, n=2000, seed=42, standardize=True)
    all_results["er20_std"] = run_single_experiment(
        "ER-2 (20 nodes) — STANDARDIZED", X, dag20,
        standardized=True,
    )

    # ---------------------------------------------------------------
    # 7. Scale-free graph (d=20)
    # ---------------------------------------------------------------
    dag_sf = generate_sf_dag(20, k=2, seed=42)
    X, _ = generate_linear_sem_data(dag_sf, n=2000, seed=42)
    all_results["sf20_raw"] = run_single_experiment(
        "Scale-Free BA (20 nodes) — RAW", X, dag_sf,
    )

    # ---------------------------------------------------------------
    # Summary table
    # ---------------------------------------------------------------
    print("\n\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Dataset':<30} {'V-sort':>6} {'R²-sort':>7} "
          f"{'Empty':>6} {'SNR':>5} {'R²SNR':>6} {'CODA':>5} {'Winner':<15}")
    print("-" * 80)

    for name, res in all_results.items():
        winner = "TIE"
        coda_shd = res["coda"]["shd"]
        snr_shd = res["sortnregress"]["shd"]
        r2_shd = res["r2_sortnregress"]["shd"]
        best_baseline = min(snr_shd, r2_shd)
        if coda_shd < best_baseline:
            winner = "CODA"
        elif coda_shd > best_baseline:
            winner = "baseline"

        print(f"{name:<30} {res['varsortability']:>6.3f} {res['r2_sortability']:>7.3f} "
              f"{res['empty']['shd']:>6d} {snr_shd:>5d} {r2_shd:>6d} {coda_shd:>5d} {winner:<15}")

    print("-" * 80)
    print("\nV-sort: varsortability (>0.9 = trivial baseline strong)")
    print("SNR: sortnregress baseline (Reisach et al., 2021)")
    print("R²SNR: R²-sortnregress baseline (Reisach et al., 2023)")
    print("Lower SHD = better")


if __name__ == "__main__":
    main()
