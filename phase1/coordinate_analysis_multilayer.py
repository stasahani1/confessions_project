"""
Run coordinate analysis on multiple layers and create comparison visualizations.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

# Configuration
ACTIVATIONS_FILE = "phase1_outputs/activations.npy"
METADATA_FILE = "phase1_outputs/metadata.jsonl"
OUTPUT_DIR = "phase1_outputs"

# Layers to analyze
TARGET_LAYERS = [18, 22, 25, 27, 30]

# Top-k sets to test
K_VALUES = [10, 50, 100, 200, 500, 1000, 2000]
NUM_RANDOM_TRIALS = 5  # Number of random baselines to average


def load_data():
    """Load activations and metadata."""
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

    print(f"Loaded {len(metadata)} metadata entries")

    # Extract needed info
    ids = [m['id'] for m in metadata]
    modes = [m['mode'] for m in metadata]

    return activations, ids, modes


def build_pairs(ids, modes):
    """Build paired examples dict."""
    pairs = defaultdict(dict)

    for idx, (id_, mode) in enumerate(zip(ids, modes)):
        pairs[id_][f'{mode}_idx'] = idx

    # Filter to complete pairs
    complete_pairs = {
        id_: data for id_, data in pairs.items()
        if 'honest_idx' in data and 'attack_idx' in data
    }

    return complete_pairs


def rank_coordinates_by_mean_difference(activations, pairs, layer_idx):
    """
    Rank coordinates by their mean absolute difference between honest and attack modes.
    """
    hidden_dim = activations.shape[2]
    differences = []

    for id_, pair_info in pairs.items():
        honest_idx = pair_info['honest_idx']
        attack_idx = pair_info['attack_idx']

        honest_act = activations[honest_idx, layer_idx, :]
        attack_act = activations[attack_idx, layer_idx, :]

        diff = honest_act - attack_act
        differences.append(diff)

    differences = np.array(differences)  # Shape: (num_pairs, hidden_dim)

    # Mean absolute difference for each dimension
    mean_abs_diff = np.mean(np.abs(differences), axis=0)  # Shape: (hidden_dim,)

    # Rank dimensions by mean abs difference (descending)
    ranked_dims = np.argsort(mean_abs_diff)[::-1]

    return ranked_dims, mean_abs_diff


def patch_activation(honest_act, attack_act, dims_to_patch):
    """Patch honest activation by replacing specific dims with attack values."""
    patched = honest_act.copy()
    patched[dims_to_patch] = attack_act[dims_to_patch]
    return patched


def compute_intervention_effect(activations, pairs, layer_idx, topk_sets, method_name):
    """
    Compute intervention effect for top-k dimension sets.

    Effect = reduction in L2 distance when patching honest -> attack
    """
    results = {}

    for k, dims in tqdm(topk_sets.items(), desc=f"Testing {method_name}"):
        effects = []

        for id_, pair_info in pairs.items():
            honest_idx = pair_info['honest_idx']
            attack_idx = pair_info['attack_idx']

            a_H = activations[honest_idx, layer_idx, :]
            a_A = activations[attack_idx, layer_idx, :]

            # Patch: replace attack dims in honest activation
            a_H_patched = patch_activation(a_H, a_A, dims)

            # Measure effect
            dist_original_to_attack = np.linalg.norm(a_H - a_A)
            dist_patched_to_attack = np.linalg.norm(a_H_patched - a_A)

            # Effect = reduction in distance
            effect = dist_original_to_attack - dist_patched_to_attack
            effect_ratio = effect / dist_original_to_attack if dist_original_to_attack > 0 else 0

            effects.append(effect_ratio)

        results[k] = {
            'mean_effect': np.mean(effects),
            'std_effect': np.std(effects),
            'median_effect': np.median(effects)
        }

    return results


def compute_random_baseline(activations, pairs, layer_idx, k_values, num_trials=5):
    """Compute baseline by patching random-k dimensions."""
    hidden_dim = activations.shape[2]
    baseline_results = {}

    for k in tqdm(k_values, desc="Random baseline"):
        trial_effects = []

        for trial in range(num_trials):
            random_dims = np.random.choice(hidden_dim, size=k, replace=False)

            effects = []
            for id_, pair_info in pairs.items():
                honest_idx = pair_info['honest_idx']
                attack_idx = pair_info['attack_idx']

                a_H = activations[honest_idx, layer_idx, :]
                a_A = activations[attack_idx, layer_idx, :]

                a_H_patched = patch_activation(a_H, a_A, random_dims)

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


def save_layer_results(layer_idx, ranked_dims, mean_diff, intervention_results, random_baseline, output_dir):
    """Save results for a single layer."""
    output_path = Path(output_dir)

    def convert_dict(d):
        return {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                for k, v in d.items()}

    results = {
        'layer': int(layer_idx),
        'k_values': K_VALUES,
        'mean_difference': {
            'top_100_dims': [int(d) for d in ranked_dims[:100]],
            'top_100_diffs': [float(d) for d in mean_diff[ranked_dims[:100]]],
            'top_2000_dims': [int(d) for d in ranked_dims[:2000]],
            'top_2000_diffs': [float(d) for d in mean_diff[ranked_dims[:2000]]]
        },
        'intervention_results': {
            'mean_difference': {str(k): convert_dict(v) for k, v in intervention_results.items()},
            'random_baseline': {str(k): convert_dict(v) for k, v in random_baseline.items()}
        }
    }

    output_file = output_path / f"coordinate_analysis_layer{layer_idx}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Saved layer {layer_idx} results to: {output_file}")
    return results


def create_comparison_plots(all_layer_results, output_dir):
    """Create comparison plots across layers."""
    output_path = Path(output_dir)

    # Extract data for plotting
    layers = sorted(all_layer_results.keys())
    k_values = K_VALUES

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Effect vs K for each layer
    for layer in layers:
        results = all_layer_results[layer]
        intervention = results['intervention_results']['mean_difference']

        mean_effects = [intervention[str(k)]['mean_effect'] for k in k_values]
        ax1.plot(k_values, mean_effects, marker='o', label=f'Layer {layer}', linewidth=2)

    # Add random baseline (use layer 22 as representative)
    baseline = all_layer_results[22]['intervention_results']['random_baseline']
    baseline_effects = [baseline[str(k)]['mean_effect'] for k in k_values]
    ax1.plot(k_values, baseline_effects, 'k--', label='Random baseline', linewidth=2, alpha=0.5)

    ax1.set_xlabel('Number of dimensions (k)', fontsize=12)
    ax1.set_ylabel('Intervention effect (distance reduction ratio)', fontsize=12)
    ax1.set_title('Coordinate Intervention Effect vs K\n(Top-k dimensions ranked by mean difference)', fontsize=14)
    ax1.set_xscale('log')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Effect for fixed k across layers
    k_fixed = [50, 200, 1000]
    x = np.arange(len(layers))
    width = 0.25

    for i, k in enumerate(k_fixed):
        effects = []
        for layer in layers:
            intervention = all_layer_results[layer]['intervention_results']['mean_difference']
            effects.append(intervention[str(k)]['mean_effect'])

        ax2.bar(x + i * width, effects, width, label=f'k={k}')

    ax2.set_xlabel('Layer', fontsize=12)
    ax2.set_ylabel('Intervention effect', fontsize=12)
    ax2.set_title('Intervention Effect Across Layers\n(for fixed k values)', fontsize=14)
    ax2.set_xticks(x + width)
    ax2.set_xticklabels([f'L{l}' for l in layers])
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plot_file = output_path / "coordinate_analysis_multilayer.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"\nSaved comparison plot to: {plot_file}")
    plt.close()


def main():
    print("\n" + "="*60)
    print("MULTI-LAYER COORDINATE ANALYSIS")
    print("="*60 + "\n")

    print(f"Analyzing layers: {TARGET_LAYERS}")
    print(f"K values: {K_VALUES}\n")

    # Load data once
    activations, ids, modes = load_data()

    # Build pairs
    pairs = build_pairs(ids, modes)
    print(f"\nFound {len(pairs)} paired examples\n")

    # Results for all layers
    all_layer_results = {}

    # Process each layer
    for layer_idx in TARGET_LAYERS:
        print("\n" + "="*60)
        print(f"LAYER {layer_idx}")
        print("="*60)

        # Rank coordinates
        print(f"\nRanking coordinates by mean difference...")
        ranked_dims, mean_diff = rank_coordinates_by_mean_difference(activations, pairs, layer_idx)

        print(f"Top 10 dimensions: {ranked_dims[:10].tolist()}")
        print(f"Top 10 mean diffs: {mean_diff[ranked_dims[:10]]}")

        # Create top-k sets
        topk_sets = {k: ranked_dims[:k] for k in K_VALUES}

        # Compute intervention effect
        print(f"\nTesting intervention effect...")
        intervention_results = compute_intervention_effect(
            activations, pairs, layer_idx, topk_sets, f"Layer {layer_idx}"
        )

        # Compute random baseline
        print(f"\nComputing random baseline...")
        random_baseline = compute_random_baseline(
            activations, pairs, layer_idx, K_VALUES, NUM_RANDOM_TRIALS
        )

        # Save results
        layer_results = save_layer_results(
            layer_idx, ranked_dims, mean_diff, intervention_results, random_baseline, OUTPUT_DIR
        )

        all_layer_results[layer_idx] = layer_results

    # Create comparison plots
    print("\n" + "="*60)
    print("CREATING COMPARISON PLOTS")
    print("="*60)
    create_comparison_plots(all_layer_results, OUTPUT_DIR)

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
