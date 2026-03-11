#!/usr/bin/env python
"""
Run all HHCRA performance verification phases and generate results.

Usage:
    python scripts/run_verification.py

Output:
    results/structure_learning.md
    results/intervention_accuracy.md
    results/counterfactual_accuracy.md
    results/structure_comparison.md
    results/ode_accuracy.md
    results/REPORT.md
"""

import sys
import os

# Ensure hhcra package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hhcra.verification import run_all_verifications

if __name__ == '__main__':
    run_all_verifications(verbose=True)
