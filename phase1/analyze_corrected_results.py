#!/usr/bin/env python3
"""
Analyze corrected PCA intervention results with truth-conditioned baselines.
Compare with buggy results to show the fix worked.
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

def load_results(filepath):
    """Load results JSON."""
    with open(filepath, 'r') as f:
        return json.load(f)

def analyze_results(results, label="Results"):
    """Analyze intervention results."""
    print(f"\n{'='*80}")
    print(f"{label}")
    print(f"{'='*80}\n")

    # Group by layer and r
    by_layer_r = defaultdict(list)
    by_layer_r_truth = defaultdict(lambda: defaultdict(list))

    for result in results:
        layer = result['layer']
        r = result['r']
        truth = result['truth']
        margin_shift = result['margin_shift']
        flipped = result['flipped_to_truth']

        key = (layer, r)
        by_layer_r[key].append(result)
        by_layer_r_truth[key][truth].append(margin_shift)

    # Summary statistics by layer and r
    print(f"Summary by (layer, r):")
    print(f"{'Layer':<8} {'r':<8} {'Mean Shift':<15} {'True→True':<15} {'False→False':<15} {'Flip Rate':<12} {'Count':<8}")
    print("-" * 100)

    summary_data = []
    for (layer, r), results_list in sorted(by_layer_r.items()):
        margin_shifts = [res['margin_shift'] for res in results_list]
        flips = sum(res['flipped_to_truth'] for res in results_list)

        # Separate by truth value
        true_shifts = by_layer_r_truth[(layer, r)][True]
        false_shifts = by_layer_r_truth[(layer, r)][False]

        mean_shift = np.mean(margin_shifts)
        mean_true = np.mean(true_shifts) if true_shifts else 0.0
        mean_false = np.mean(false_shifts) if false_shifts else 0.0
        flip_rate = flips / len(results_list)

        print(f"{layer:<8} {r:<8} {mean_shift:>+.4f}         {mean_true:>+.4f}          {mean_false:>+.4f}           {flip_rate:.2%}        {len(results_list):<8}")

        summary_data.append({
            'layer': layer,
            'r': r,
            'mean_shift': mean_shift,
            'mean_true_shift': mean_true,
            'mean_false_shift': mean_false,
            'flip_rate': flip_rate,
            'count': len(results_list)
        })

    return summary_data

def check_symmetry(results):
    """Check if margin shifts are symmetric for truth=True vs truth=False."""
    print(f"\n{'='*80}")
    print("SYMMETRY CHECK: Do interventions move in opposite directions?")
    print(f"{'='*80}\n")

    by_layer_r_truth = defaultdict(lambda: defaultdict(list))

    for result in results:
        layer = result['layer']
        r = result['r']
        truth = result['truth']
        margin_shift = result['margin_shift']

        key = (layer, r)
        by_layer_r_truth[key][truth].append(margin_shift)

    print(f"{'Layer':<8} {'r':<8} {'True→True (mean)':<20} {'False→False (mean)':<20} {'Symmetric?':<15}")
    print("-" * 90)

    for (layer, r) in sorted(by_layer_r_truth.keys()):
        true_shifts = by_layer_r_truth[(layer, r)][True]
        false_shifts = by_layer_r_truth[(layer, r)][False]

        mean_true = np.mean(true_shifts) if true_shifts else 0.0
        mean_false = np.mean(false_shifts) if false_shifts else 0.0

        # Check if they have opposite signs and similar magnitudes
        opposite_signs = (mean_true > 0 and mean_false < 0) or (mean_true < 0 and mean_false > 0)
        symmetric = "✅ YES" if opposite_signs else "❌ NO"

        print(f"{layer:<8} {r:<8} {mean_true:>+.4f}              {mean_false:>+.4f}               {symmetric:<15}")

def find_best_configs(results):
    """Find best (layer, r) configurations."""
    print(f"\n{'='*80}")
    print("BEST CONFIGURATIONS")
    print(f"{'='*80}\n")

    by_layer_r = defaultdict(list)

    for result in results:
        layer = result['layer']
        r = result['r']
        margin_shift = result['margin_shift']
        truth = result['truth']

        key = (layer, r)
        by_layer_r[key].append((margin_shift, truth))

    # Compute absolute effect size for each config
    config_scores = []
    for (layer, r), shifts_truth in by_layer_r.items():
        # Separate by truth
        true_shifts = [s for s, t in shifts_truth if t]
        false_shifts = [s for s, t in shifts_truth if not t]

        mean_true = np.mean(true_shifts) if true_shifts else 0.0
        mean_false = np.mean(false_shifts) if false_shifts else 0.0

        # Effect size = average absolute shift
        effect_size = (abs(mean_true) + abs(mean_false)) / 2

        config_scores.append({
            'layer': layer,
            'r': r,
            'effect_size': effect_size,
            'mean_true': mean_true,
            'mean_false': mean_false
        })

    # Sort by effect size
    config_scores.sort(key=lambda x: x['effect_size'], reverse=True)

    print("Top 10 configurations by effect size:")
    print(f"{'Rank':<6} {'Layer':<8} {'r':<8} {'Effect Size':<15} {'True→True':<15} {'False→False':<15}")
    print("-" * 80)

    for i, config in enumerate(config_scores[:10], 1):
        print(f"{i:<6} {config['layer']:<8} {config['r']:<8} {config['effect_size']:>+.4f}         {config['mean_true']:>+.4f}          {config['mean_false']:>+.4f}")

def compare_buggy_vs_corrected(buggy_file, corrected_file):
    """Compare buggy vs corrected results."""
    print(f"\n{'='*80}")
    print("BUGGY vs CORRECTED COMPARISON")
    print(f"{'='*80}\n")

    buggy = load_results(buggy_file)
    corrected = load_results(corrected_file)

    # Focus on a few key configs
    key_configs = [(22, 1), (25, 1), (30, 1)]

    for layer, r in key_configs:
        print(f"\n--- Layer {layer}, r={r} ---\n")

        # Buggy results
        buggy_results = [res for res in buggy if res['layer'] == layer and res['r'] == r]
        buggy_true = [res['margin_shift'] for res in buggy_results if res['truth']]
        buggy_false = [res['margin_shift'] for res in buggy_results if not res['truth']]

        # Corrected results
        corrected_results = [res for res in corrected if res['layer'] == layer and res['r'] == r]
        corrected_true = [res['margin_shift'] for res in corrected_results if res['truth']]
        corrected_false = [res['margin_shift'] for res in corrected_results if not res['truth']]

        print(f"BUGGY:")
        print(f"  Truth=True  → Mean shift: {np.mean(buggy_true):>+.4f}  {'✅' if np.mean(buggy_true) > 0 else '❌'}")
        print(f"  Truth=False → Mean shift: {np.mean(buggy_false):>+.4f}  {'❌ WRONG (should be negative)' if np.mean(buggy_false) > 0 else '✅'}")

        print(f"\nCORRECTED:")
        print(f"  Truth=True  → Mean shift: {np.mean(corrected_true):>+.4f}  {'✅' if np.mean(corrected_true) > 0 else '❌'}")
        print(f"  Truth=False → Mean shift: {np.mean(corrected_false):>+.4f}  {'✅ FIXED (now negative)' if np.mean(corrected_false) < 0 else '❌'}")

def check_kl_divergence(results):
    """Check if KL divergence is now valid."""
    print(f"\n{'='*80}")
    print("KL DIVERGENCE CHECK")
    print(f"{'='*80}\n")

    kl_values = [res['kl_divergence'] for res in results]

    nan_count = sum(1 for x in kl_values if np.isnan(x))
    valid_count = len(kl_values) - nan_count

    print(f"Total results: {len(kl_values)}")
    print(f"Valid KL values: {valid_count} ✅")
    print(f"NaN KL values: {nan_count} {'❌' if nan_count > 0 else '✅'}")

    if valid_count > 0:
        valid_kl = [x for x in kl_values if not np.isnan(x)]
        print(f"\nKL Divergence statistics:")
        print(f"  Mean: {np.mean(valid_kl):.4f}")
        print(f"  Median: {np.median(valid_kl):.4f}")
        print(f"  Min: {np.min(valid_kl):.4f}")
        print(f"  Max: {np.max(valid_kl):.4f}")

def main():
    base_dir = Path("/workspace/confessions_project/phase1/phase1_outputs")

    corrected_file = base_dir / "pca_subspace_results.json"
    buggy_file = base_dir / "pca_subspace_results_OLD_BUGGY.json"

    print("\n" + "="*80)
    print("PCA SUBSPACE INTERVENTION - CORRECTED RESULTS ANALYSIS")
    print("="*80)

    # Load corrected results
    corrected = load_results(corrected_file)

    # Main analysis
    summary_data = analyze_results(corrected, "CORRECTED RESULTS")

    # Symmetry check
    check_symmetry(corrected)

    # Find best configs
    find_best_configs(corrected)

    # KL divergence check
    check_kl_divergence(corrected)

    # Compare with buggy results
    if buggy_file.exists():
        compare_buggy_vs_corrected(buggy_file, corrected_file)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
