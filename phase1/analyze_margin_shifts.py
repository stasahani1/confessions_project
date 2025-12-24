#!/usr/bin/env python3
"""
Analyze margin shifts from patching results.

This script loads the saved patching results and provides detailed analysis
of truth margin changes: Δ(logit_T - logit_F)
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

# File paths
PATCHING_RESULTS = "phase1_outputs/patching_results.json"
SELECTIVE_PATCHING_RESULTS = "phase1_outputs/selective_patching_results.json"


def load_results(filepath):
    """Load patching results from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def analyze_margin_shifts(results, title="Margin Shift Analysis"):
    """Analyze and print margin shift statistics."""
    print("\n" + "="*70)
    print(title)
    print("="*70)

    # Overall statistics
    margin_shifts = [r['margin_shift'] for r in results]
    print(f"\nOverall Statistics:")
    print(f"  N = {len(margin_shifts)}")
    print(f"  Mean margin shift: {np.mean(margin_shifts):+.4f}")
    print(f"  Median margin shift: {np.median(margin_shifts):+.4f}")
    print(f"  Std dev: {np.std(margin_shifts):.4f}")
    print(f"  Min: {np.min(margin_shifts):+.4f}")
    print(f"  Max: {np.max(margin_shifts):+.4f}")

    # Count direction of shifts
    positive_shifts = sum(1 for s in margin_shifts if s > 0)
    negative_shifts = sum(1 for s in margin_shifts if s < 0)
    print(f"\n  Positive shifts (towards True): {positive_shifts} ({100*positive_shifts/len(margin_shifts):.1f}%)")
    print(f"  Negative shifts (towards False): {negative_shifts} ({100*negative_shifts/len(margin_shifts):.1f}%)")

    # Breakdown by whether prediction flipped
    flipped_results = [r for r in results if r.get('prediction_flipped', False)]
    not_flipped_results = [r for r in results if not r.get('prediction_flipped', False)]

    if flipped_results:
        flipped_margins = [r['margin_shift'] for r in flipped_results]
        print(f"\n  Prediction FLIPPED (n={len(flipped_results)}):")
        print(f"    Mean margin shift: {np.mean(flipped_margins):+.4f}")
        print(f"    Median: {np.median(flipped_margins):+.4f}")

    if not_flipped_results:
        not_flipped_margins = [r['margin_shift'] for r in not_flipped_results]
        print(f"\n  Prediction DID NOT flip (n={len(not_flipped_results)}):")
        print(f"    Mean margin shift: {np.mean(not_flipped_margins):+.4f}")
        print(f"    Median: {np.median(not_flipped_margins):+.4f}")
        print(f"    ** This shows effect size even without behavior change **")

    # Breakdown by ground truth
    true_results = [r for r in results if r['truth'] == True]
    false_results = [r for r in results if r['truth'] == False]

    if true_results:
        true_margins = [r['margin_shift'] for r in true_results]
        print(f"\n  Ground truth = True (n={len(true_results)}):")
        print(f"    Mean margin shift: {np.mean(true_margins):+.4f}")
        print(f"    (Positive = correct direction)")

    if false_results:
        false_margins = [r['margin_shift'] for r in false_results]
        print(f"\n  Ground truth = False (n={len(false_results)}):")
        print(f"    Mean margin shift: {np.mean(false_margins):+.4f}")
        print(f"    (Negative = correct direction)")


def compare_target_vs_control(results):
    """Compare target interventions vs control conditions."""
    print("\n" + "="*70)
    print("TARGET vs CONTROL COMPARISON")
    print("="*70)

    # Group by layer and control type
    target_by_layer = defaultdict(list)
    control_by_layer = defaultdict(list)

    for r in results:
        layer = r['patch_layer']
        if r['control_type'] == 'target':
            target_by_layer[layer].append(r)
        else:
            control_by_layer[layer].append(r)

    print(f"\n{'Layer':>6} | {'Target Mean':>15} | {'Control Mean':>15} | {'Difference':>15}")
    print("-" * 60)

    for layer in sorted(target_by_layer.keys()):
        if layer in control_by_layer:
            target_margins = [r['margin_shift'] for r in target_by_layer[layer]]
            control_margins = [r['margin_shift'] for r in control_by_layer[layer]]

            target_mean = np.mean(target_margins)
            control_mean = np.mean(control_margins)
            diff = target_mean - control_mean

            print(f"{layer:6d} | {target_mean:+14.4f} | {control_mean:+14.4f} | {diff:+14.4f}")


def analyze_selective_patching_by_k(results):
    """Analyze selective patching results broken down by k value."""
    print("\n" + "="*70)
    print("SELECTIVE PATCHING: Margin Shift by K")
    print("="*70)

    # Group by k and control type
    results_by_k = defaultdict(lambda: defaultdict(list))

    for r in results:
        k = r.get('k', 'full')
        control_type = r['control_type']
        results_by_k[k][control_type].append(r)

    print(f"\n{'k':>6} | {'Target':>12} | {'Random':>12} | {'Difference':>12}")
    print("-" * 50)

    for k in sorted(results_by_k.keys()):
        if 'target' in results_by_k[k] and 'random' in results_by_k[k]:
            target_margins = [r['margin_shift'] for r in results_by_k[k]['target']]
            random_margins = [r['margin_shift'] for r in results_by_k[k]['random']]

            target_mean = np.mean(target_margins)
            random_mean = np.mean(random_margins)
            diff = target_mean - random_mean

            print(f"{k:6} | {target_mean:+11.4f} | {random_mean:+11.4f} | {diff:+11.4f}")


def main():
    print("\n" + "="*70)
    print("MARGIN SHIFT ANALYSIS")
    print("="*70)

    # Analyze full activation patching
    if Path(PATCHING_RESULTS).exists():
        print("\n>>> FULL ACTIVATION PATCHING <<<")
        results = load_results(PATCHING_RESULTS)

        # Target interventions
        target_results = [r for r in results if r['control_type'] == 'target']
        if target_results:
            analyze_margin_shifts(target_results, "Target Intervention (Honest → Attack)")

        # Random control
        random_results = [r for r in results if r['control_type'] == 'random_example']
        if random_results:
            analyze_margin_shifts(random_results, "Control: Random Example")

        # Comparison
        compare_target_vs_control(results)
    else:
        print(f"\nFile not found: {PATCHING_RESULTS}")

    # Analyze selective patching
    if Path(SELECTIVE_PATCHING_RESULTS).exists():
        print("\n\n>>> SELECTIVE DIMENSION PATCHING <<<")
        results = load_results(SELECTIVE_PATCHING_RESULTS)

        # Target interventions only
        target_results = [r for r in results if r['control_type'] == 'target']
        if target_results:
            analyze_margin_shifts(target_results, "Selective Patching (All k values)")

        # By k value
        analyze_selective_patching_by_k(results)

        # Detailed analysis for largest k
        k_values = sorted(set(r['k'] for r in results if 'k' in r))
        if k_values:
            largest_k = k_values[-1]
            largest_k_results = [r for r in results if r.get('k') == largest_k and r['control_type'] == 'target']
            if largest_k_results:
                analyze_margin_shifts(largest_k_results, f"Selective Patching (k={largest_k} only)")
    else:
        print(f"\nFile not found: {SELECTIVE_PATCHING_RESULTS}")

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
