#!/usr/bin/env python3
"""
v0.6.0 Performance Benchmark: Verify HHCRA upgrades beat baselines.

Runs HHCRA vs PC vs Granger vs NOTEARS on Asia and Sachs benchmarks,
printing a comparison table with F1, TPR, SHD metrics.

Usage:
    python scripts/run_v060_benchmark.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import time

from hhcra.real_benchmarks import (
    make_asia_benchmark,
    make_sachs_benchmark,
    run_single_benchmark,
    print_summary_table,
)


def main():
    print("=" * 70)
    print("  HHCRA v0.6.0 — Performance Benchmark Suite")
    print("  Comparing: HHCRA (upgraded) vs PC vs Granger vs NOTEARS")
    print("=" * 70)

    torch.manual_seed(42)
    np.random.seed(42)

    results = []

    # Asia benchmark (8 vars, 8 edges)
    print("\n[1/2] Running Asia benchmark...")
    asia = make_asia_benchmark()
    asia_result = run_single_benchmark(
        asia, n_samples=1000, seed=42,
        run_hhcra=True, run_notears=True, verbose=True,
    )
    results.append(asia_result)

    # Sachs benchmark (11 vars, 17 edges)
    print("\n[2/2] Running Sachs benchmark...")
    sachs = make_sachs_benchmark()
    sachs_result = run_single_benchmark(
        sachs, n_samples=1000, seed=42,
        run_hhcra=True, run_notears=True, verbose=True,
    )
    results.append(sachs_result)

    # Print summary
    print_summary_table(results)

    # Print winner analysis
    print("\n" + "=" * 70)
    print("  WINNER ANALYSIS")
    print("=" * 70)

    for r in results:
        print(f"\n  {r.graph_name.upper()} ({r.num_vars} vars, {r.num_edges} edges):")
        methods = {k: v for k, v in r.results.items()
                   if k not in ('empty', 'random')}
        if not methods:
            continue

        # F1 winner
        f1_winner = max(methods.items(), key=lambda x: x[1].f1)
        print(f"    F1 winner:  {f1_winner[0]:>8s} (F1={f1_winner[1].f1:.3f})")

        # SHD winner
        shd_winner = min(methods.items(), key=lambda x: x[1].shd)
        print(f"    SHD winner: {shd_winner[0]:>8s} (SHD={shd_winner[1].shd})")

        # TPR winner
        tpr_winner = max(methods.items(), key=lambda x: x[1].tpr)
        print(f"    TPR winner: {tpr_winner[0]:>8s} (TPR={tpr_winner[1].tpr:.3f})")

        # HHCRA ranking
        if 'hhcra' in methods:
            f1_sorted = sorted(methods.items(), key=lambda x: -x[1].f1)
            rank = next(i+1 for i, (k, _) in enumerate(f1_sorted) if k == 'hhcra')
            total = len(f1_sorted)
            print(f"    HHCRA rank: #{rank}/{total} by F1")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
