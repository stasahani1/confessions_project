import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# Configuration
ACTIVATIONS_FILE = "phase1_outputs/activations.npy"
METADATA_FILE = "phase1_outputs/metadata.jsonl"
PROBE_RESULTS_FILE = "phase1_outputs/probe_results.json"
DIFF_STATS_FILE = "phase1_outputs/paired_difference_stats.json"
OUTPUT_DIR = "phase1_outputs"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Top-k sets to test
K_VALUES = [10, 50, 200, 1000]
NUM_RANDOM_TRIALS = 10  # Number of random baselines to average

def load_data():
    """Load all necessary data."""
    print("Loading data...")
    print("=" * 60)

    # Load activations
    activations = np.load(ACTIVATIONS_FILE)
    print(f"Loaded activations: {activations.shape}")

    # Load metadata
    metadata = []
    with open(METADATA_FILE, 'r') as f:
        for line in f:
            metadata.append(json.loads(line))

    ids = np.array([m['id'] for m in metadata])
    labels = np.array([m['aligned'] for m in metadata])
    modes = np.array([m['mode'] for m in metadata])

    # Load probe results
    with open(PROBE_RESULTS_FILE, 'r') as f:
        probe_results = json.load(f)

    # Load difference stats
    with open(DIFF_STATS_FILE, 'r') as f:
        diff_stats = json.load(f)

    print(f"Loaded {len(metadata)} examples")
    print("=" * 60)

    return activations, metadata, ids, labels, modes, probe_results, diff_stats

def select_optimal_layer(probe_results, diff_stats, method='probe_auc'):
    """
    Select the optimal layer for intervention.

    Args:
        probe_results: Dict with probe performance per layer
        diff_stats: Dict with paired difference statistics per layer
        method: 'probe_auc', 'probe_acc', 'max_l2', or 'max_cosine'

    Returns:
        layer_idx: The selected layer index
    """
    print(f"\nSelecting optimal layer using method: {method}")

    if method == 'probe_auc':
        layer_idx = probe_results['best_auc_layer']
        score = probe_results['layer_results'][str(layer_idx)]['auc']
        print(f"  Selected layer {layer_idx} (best probe AUC: {score:.3f})")
    elif method == 'probe_acc':
        layer_idx = probe_results['best_accuracy_layer']
        score = probe_results['layer_results'][str(layer_idx)]['accuracy']
        print(f"  Selected layer {layer_idx} (best probe accuracy: {score:.3f})")
    elif method == 'max_l2':
        layer_idx = diff_stats['max_l2_layer']
        score = diff_stats['layer_stats'][str(layer_idx)]['mean_l2']
        print(f"  Selected layer {layer_idx} (max L2 norm: {score:.2f})")
    elif method == 'max_cosine':
        layer_idx = diff_stats['max_cosine_layer']
        score = diff_stats['layer_stats'][str(layer_idx)]['mean_cosine_dist']
        print(f"  Selected layer {layer_idx} (max cosine distance: {score:.4f})")
    else:
        raise ValueError(f"Unknown method: {method}")

    return layer_idx

def build_pairs(ids, modes):
    """Build mapping from ID to (honest_idx, attack_idx)."""
    id_to_indices = {}
    for idx, (id_, mode) in enumerate(zip(ids, modes)):
        if id_ not in id_to_indices:
            id_to_indices[id_] = {}
        id_to_indices[id_][mode] = idx

    pairs = {}
    for id_, mode_dict in id_to_indices.items():
        if 'honest' in mode_dict and 'attack' in mode_dict:
            pairs[id_] = {
                'honest_idx': mode_dict['honest'],
                'attack_idx': mode_dict['attack']
            }

    return pairs

def rank_coordinates_by_mean_difference(activations, pairs, layer_idx):
    """
    Rank coordinates by mean difference direction.

    Returns:
        ranked_dims: Array of dimension indices sorted by importance (descending)
        mean_diff: The mean difference vector
    """
    print(f"\nRanking coordinates by mean difference at layer {layer_idx}...")

    # Compute all differences
    diffs = []
    for id_, pair_info in pairs.items():
        honest_idx = pair_info['honest_idx']
        attack_idx = pair_info['attack_idx']

        a_H = activations[honest_idx, layer_idx, :]
        a_A = activations[attack_idx, layer_idx, :]

        diffs.append(a_H - a_A)

    # Mean difference
    mean_diff = np.mean(diffs, axis=0)  # Shape: (hidden_dim,)

    # Rank by absolute value
    ranked_dims = np.argsort(np.abs(mean_diff))[::-1]  # Descending order

    print(f"  Top 5 dimensions by mean difference: {ranked_dims[:5]}")
    print(f"  Top 5 absolute differences: {np.abs(mean_diff)[ranked_dims[:5]]}")

    return ranked_dims, mean_diff

def create_topk_sets(ranked_dims, k_values):
    """
    Create sets of top-k dimensions.

    Returns:
        topk_sets: Dict mapping k -> array of top-k dimension indices
    """
    topk_sets = {}
    for k in k_values:
        topk_sets[k] = ranked_dims[:k]

    return topk_sets

def patch_activation(activation, source_activation, dims_to_patch):
    """
    Patch specific dimensions of activation with values from source.

    Args:
        activation: Target activation to modify (will be copied)
        source_activation: Source activation to copy from
        dims_to_patch: Indices of dimensions to patch

    Returns:
        patched: New activation with patched dimensions
    """
    patched = activation.copy()
    patched[dims_to_patch] = source_activation[dims_to_patch]
    return patched

def compute_intervention_effect(activations, pairs, layer_idx, topk_sets, method_name):
    """
    Compute the effect of patching top-k dimensions.

    For each statement:
    - Start with honest activation
    - Patch top-k dims with attack activation
    - Measure how much it "pushes toward" attack behavior

    We'll measure: change in activation alignment with attack mode

    Returns:
        results: Dict mapping k -> effect metrics
    """
    print(f"\nComputing intervention effects for {method_name}...")

    results = {}

    for k, top_dims in tqdm(topk_sets.items(), desc="Testing k values"):
        effects = []

        for id_, pair_info in pairs.items():
            honest_idx = pair_info['honest_idx']
            attack_idx = pair_info['attack_idx']

            a_H = activations[honest_idx, layer_idx, :]
            a_A = activations[attack_idx, layer_idx, :]

            # Patch honest with attack at top-k dims
            a_H_patched = patch_activation(a_H, a_A, top_dims)

            # Measure effect: how much did patched activation move toward attack?
            # Use L2 distance as metric
            dist_original_to_attack = np.linalg.norm(a_H - a_A)
            dist_patched_to_attack = np.linalg.norm(a_H_patched - a_A)

            # Effect = reduction in distance (higher = more effective intervention)
            effect = dist_original_to_attack - dist_patched_to_attack
            effect_ratio = effect / dist_original_to_attack if dist_original_to_attack > 0 else 0

            effects.append(effect_ratio)

        results[k] = {
            'mean_effect': np.mean(effects),
            'std_effect': np.std(effects),
            'median_effect': np.median(effects)
        }

    return results

def compute_random_baseline(activations, pairs, layer_idx, k_values, num_trials=10):
    """
    Compute baseline by patching random-k dimensions.

    Returns:
        baseline_results: Dict mapping k -> effect metrics (averaged over trials)
    """
    print(f"\nComputing random baseline ({num_trials} trials)...")

    hidden_dim = activations.shape[2]
    baseline_results = {}

    for k in tqdm(k_values, desc="Testing k values"):
        trial_effects = []

        for trial in range(num_trials):
            # Random selection of k dims
            random_dims = np.random.choice(hidden_dim, size=k, replace=False)

            effects = []
            for id_, pair_info in pairs.items():
                honest_idx = pair_info['honest_idx']
                attack_idx = pair_info['attack_idx']

                a_H = activations[honest_idx, layer_idx, :]
                a_A = activations[attack_idx, layer_idx, :]

                # Patch with random dims
                a_H_patched = patch_activation(a_H, a_A, random_dims)

                # Measure effect
                dist_original_to_attack = np.linalg.norm(a_H - a_A)
                dist_patched_to_attack = np.linalg.norm(a_H_patched - a_A)
                effect = dist_original_to_attack - dist_patched_to_attack
                effect_ratio = effect / dist_original_to_attack if dist_original_to_attack > 0 else 0

                effects.append(effect_ratio)

            trial_effects.append(np.mean(effects))

        baseline_results[k] = {
            'mean_effect': np.mean(trial_effects),
            'std_effect': np.std(trial_effects),
            'median_effect': np.median(trial_effects)
        }

    return baseline_results

def plot_intervention_results(diff_results, random_results, k_values, output_dir, layer_idx):
    """Plot intervention effects vs k for mean difference vs random baseline."""
    print("\nPlotting intervention results...")

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Extract mean effects
    diff_means = [diff_results[k]['mean_effect'] for k in k_values]
    diff_stds = [diff_results[k]['std_effect'] for k in k_values]

    random_means = [random_results[k]['mean_effect'] for k in k_values]
    random_stds = [random_results[k]['std_effect'] for k in k_values]

    # Plot
    ax.plot(k_values, diff_means, 'b-o', linewidth=2, label='Top-k by mean difference', markersize=6)
    ax.fill_between(k_values,
                     np.array(diff_means) - np.array(diff_stds),
                     np.array(diff_means) + np.array(diff_stds),
                     alpha=0.2, color='b')

    ax.plot(k_values, random_means, 'r--^', linewidth=2, label='Random-k baseline', markersize=6)
    ax.fill_between(k_values,
                     np.array(random_means) - np.array(random_stds),
                     np.array(random_means) + np.array(random_stds),
                     alpha=0.2, color='r')

    ax.set_xlabel('Number of coordinates (k)', fontsize=12)
    ax.set_ylabel('Intervention Effect (distance reduction ratio)', fontsize=12)
    ax.set_title(f'Causal Intervention: Top-k vs Random-k (Layer {layer_idx})',
                 fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    plt.tight_layout()

    output_path = Path(output_dir) / "coordinate_intervention_results.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to: {output_path}")

    plt.close()

def analyze_top_coordinates(ranked_dims, mean_diff, top_n=20):
    """Analyze top coordinates from mean difference."""
    print(f"\nAnalyzing top {top_n} coordinates...")
    print("=" * 60)

    print(f"\nTop 10 by mean difference:")
    for i in range(10):
        dim = ranked_dims[i]
        print(f"  Dim {dim:4d}: diff = {mean_diff[dim]:+.4f}, |diff| = {abs(mean_diff[dim]):.4f}")

    print("=" * 60)

def save_results(layer_idx, ranked_dims, mean_diff,
                 diff_intervention, random_baseline,
                 k_values, output_dir):
    """Save all coordinate analysis results."""
    print("\nSaving results...")

    output_path = Path(output_dir)

    # Convert numpy types to Python types in nested dicts
    def convert_dict(d):
        return {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                for k, v in d.items()}

    results = {
        'layer': int(layer_idx),
        'k_values': [int(k) for k in k_values],
        'mean_difference': {
            'top_100_dims': [int(d) for d in ranked_dims[:100]],
            'top_100_diffs': [float(d) for d in mean_diff[ranked_dims[:100]]],
            # Save more dimensions for selective patching experiments
            'top_2000_dims': [int(d) for d in ranked_dims[:2000]],
            'top_2000_diffs': [float(d) for d in mean_diff[ranked_dims[:2000]]]
        },
        'intervention_results': {
            'mean_difference': {str(k): convert_dict(v) for k, v in diff_intervention.items()},
            'random_baseline': {str(k): convert_dict(v) for k, v in random_baseline.items()}
        }
    }

    output_file = output_path / "coordinate_analysis_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Saved results to: {output_file}")

def main():
    print("\n" + "="*60)
    print("COORDINATE SELECTION & CAUSAL INTERVENTION ANALYSIS")
    print("="*60 + "\n")

    # Load data
    activations, metadata, ids, labels, modes, probe_results, diff_stats = load_data()

    # Use layer 22 (where we know full-vector patching works well)
    layer_idx = 22
    print(f"\nUsing layer {layer_idx} for coordinate analysis")

    # Build pairs
    pairs = build_pairs(ids, modes)
    print(f"\nFound {len(pairs)} paired examples")

    # ===== COORDINATE RANKING =====
    print("\n" + "="*60)
    print("PART 1: COORDINATE RANKING")
    print("="*60)

    # Rank by mean difference
    ranked_dims, mean_diff = rank_coordinates_by_mean_difference(
        activations, pairs, layer_idx
    )

    # Analyze top coordinates
    analyze_top_coordinates(ranked_dims, mean_diff)

    # Create top-k sets
    topk_sets = create_topk_sets(ranked_dims, K_VALUES)

    # ===== CAUSAL INTERVENTION =====
    print("\n" + "="*60)
    print("PART 2: CAUSAL INTERVENTION TESTING")
    print("="*60)

    # Test mean difference
    diff_intervention = compute_intervention_effect(
        activations, pairs, layer_idx, topk_sets, "mean difference"
    )

    # Compute random baseline
    random_baseline = compute_random_baseline(
        activations, pairs, layer_idx, K_VALUES, NUM_RANDOM_TRIALS
    )

    # ===== RESULTS =====
    print("\n" + "="*60)
    print("INTERVENTION RESULTS")
    print("="*60)

    print(f"\nEffect by k (mean effect ratio):")
    print(f"{'k':>6} | {'Mean Diff':>10} | {'Random':>10}")
    print("-" * 40)
    for k in K_VALUES:
        diff_eff = diff_intervention[k]['mean_effect']
        random_eff = random_baseline[k]['mean_effect']
        print(f"{k:6d} | {diff_eff:10.4f} | {random_eff:10.4f}")

    # Plot results
    plot_intervention_results(
        diff_intervention, random_baseline,
        K_VALUES, OUTPUT_DIR, layer_idx
    )

    # Save results
    save_results(
        layer_idx, ranked_dims, mean_diff,
        diff_intervention, random_baseline,
        K_VALUES, OUTPUT_DIR
    )

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print(f"\nOutputs saved to: {OUTPUT_DIR}")
    print(f"  - coordinate_intervention_results.png")
    print(f"  - coordinate_analysis_results.json")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
