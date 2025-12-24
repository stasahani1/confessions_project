"""
Quick analysis of PCA explained variance on activation differences.

This script analyzes how many PCA components are needed to explain
the variance in Δ = a^H - a^A at different layers.

Run this BEFORE the full intervention experiment to understand
the intrinsic dimensionality of the honesty signal.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from collections import defaultdict

# Configuration
ACTIVATIONS_FILE = "phase1_outputs/activations.npy"
METADATA_FILE = "phase1_outputs/metadata.jsonl"
OUTPUT_DIR = "phase1_outputs"

# Analyze all layers
N_COMPONENTS = 256  # Maximum components to compute


def load_data():
    """Load activations and metadata."""
    print("Loading activations and metadata...")

    activations = np.load(ACTIVATIONS_FILE)
    print(f"Loaded activations: {activations.shape}")

    metadata = []
    with open(METADATA_FILE, 'r') as f:
        for line in f:
            metadata.append(json.loads(line))

    print(f"Loaded {len(metadata)} metadata entries")

    # Build pairs
    pairs = defaultdict(dict)

    for idx, meta in enumerate(metadata):
        stmt_id = meta['id']
        mode = meta['mode']

        pairs[stmt_id][f'{mode}_idx'] = idx
        pairs[stmt_id][f'{mode}_activation'] = activations[idx]

    # Filter to complete pairs
    complete_pairs = {
        sid: data for sid, data in pairs.items()
        if 'honest_idx' in data and 'attack_idx' in data
    }

    print(f"Found {len(complete_pairs)} complete pairs")

    return activations, metadata, complete_pairs


def compute_pca_variance_analysis(pairs, layer, n_components):
    """
    Compute PCA on activation differences and analyze explained variance.

    Returns:
        explained_var_ratio: Explained variance ratio for each component
        cumulative_var: Cumulative explained variance
    """
    # Collect activation differences
    deltas = []

    for stmt_id, pair_data in pairs.items():
        a_H = pair_data['honest_activation'][layer, :]
        a_A = pair_data['attack_activation'][layer, :]
        delta = a_H - a_A
        deltas.append(delta)

    deltas = np.array(deltas)  # Shape: (n_pairs, hidden_dim)

    # Fit PCA
    n_components_actual = min(n_components, deltas.shape[0], deltas.shape[1])
    pca = PCA(n_components=n_components_actual)
    pca.fit(deltas)

    explained_var_ratio = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var_ratio)

    # Also compute L2 norm statistics
    l2_norms = np.linalg.norm(deltas, axis=1)
    mean_l2 = np.mean(l2_norms)
    std_l2 = np.std(l2_norms)

    return explained_var_ratio, cumulative_var, mean_l2, std_l2


def main():
    print("\n" + "="*80)
    print("PCA VARIANCE ANALYSIS OF ACTIVATION DIFFERENCES")
    print("="*80 + "\n")

    # Load data
    activations, metadata, pairs = load_data()

    num_layers = activations.shape[1]

    print(f"\nAnalyzing PCA variance for {num_layers} layers...")
    print("="*80)

    # Store results
    all_results = {}

    # Analyze each layer
    for layer in range(num_layers):
        explained_var_ratio, cumulative_var, mean_l2, std_l2 = compute_pca_variance_analysis(
            pairs, layer, N_COMPONENTS
        )

        all_results[layer] = {
            'explained_var_ratio': explained_var_ratio.tolist(),
            'cumulative_var': cumulative_var.tolist(),
            'mean_l2': float(mean_l2),
            'std_l2': float(std_l2)
        }

        # Print summary for this layer
        # Find how many components needed for 50%, 80%, 90%, 95%, 99%
        thresholds = [0.5, 0.8, 0.9, 0.95, 0.99]
        components_needed = []

        for thresh in thresholds:
            n_comp = np.searchsorted(cumulative_var, thresh) + 1
            components_needed.append(n_comp)

        if layer % 5 == 0:  # Print every 5th layer to avoid clutter
            print(f"\nLayer {layer:2d}:")
            print(f"  Mean L2 norm of Δ: {mean_l2:.4f} ± {std_l2:.4f}")
            print(f"  Components for 50% variance: {components_needed[0]}")
            print(f"  Components for 80% variance: {components_needed[1]}")
            print(f"  Components for 90% variance: {components_needed[2]}")
            print(f"  Components for 95% variance: {components_needed[3]}")
            print(f"  Components for 99% variance: {components_needed[4]}")

    # Save results
    output_path = Path(OUTPUT_DIR)
    results_file = output_path / "pca_variance_analysis.json"

    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Saved results to: {results_file}")
    print("="*80)

    # Generate visualizations
    print("\nGenerating visualizations...")

    # Plot 1: Cumulative explained variance vs components for key layers
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))

    key_layers = [0, 10, 18, 22, 25, 27, 30, 31]  # Representative layers
    colors = plt.cm.viridis(np.linspace(0, 1, len(key_layers)))

    for idx, layer in enumerate(key_layers):
        if layer < num_layers:
            cumulative_var = all_results[layer]['cumulative_var']
            components = np.arange(1, len(cumulative_var) + 1)

            ax.plot(components, cumulative_var, '-', linewidth=2,
                   color=colors[idx], label=f'Layer {layer}')

    # Add threshold lines
    for thresh in [0.5, 0.8, 0.9, 0.95]:
        ax.axhline(y=thresh, color='gray', linestyle='--', alpha=0.3, linewidth=1)
        ax.text(N_COMPONENTS * 0.95, thresh, f'{thresh:.0%}',
               verticalalignment='bottom', fontsize=9, color='gray')

    ax.set_xlabel('Number of Components (r)', fontsize=12)
    ax.set_ylabel('Cumulative Explained Variance', fontsize=12)
    ax.set_title('PCA Variance Analysis: Cumulative Explained Variance by Layer',
                fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.set_xlim([1, N_COMPONENTS])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')

    plt.tight_layout()

    plot_file = output_path / "pca_variance_cumulative.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {plot_file}")

    plt.close()

    # Plot 2: Components needed for 90% variance vs layer
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    layers = list(range(num_layers))
    components_90 = []
    components_95 = []
    mean_l2_norms = []

    for layer in layers:
        cumulative_var = all_results[layer]['cumulative_var']
        n_90 = np.searchsorted(cumulative_var, 0.90) + 1
        n_95 = np.searchsorted(cumulative_var, 0.95) + 1
        components_90.append(n_90)
        components_95.append(n_95)
        mean_l2_norms.append(all_results[layer]['mean_l2'])

    ax1.plot(layers, components_90, 'b-o', linewidth=2, markersize=4, label='90% variance')
    ax1.plot(layers, components_95, 'r-o', linewidth=2, markersize=4, label='95% variance')
    ax1.set_xlabel('Layer', fontsize=12)
    ax1.set_ylabel('Number of Components Needed', fontsize=12)
    ax1.set_title('Intrinsic Dimensionality of Honesty Signal', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot mean L2 norm
    ax2.plot(layers, mean_l2_norms, 'g-o', linewidth=2, markersize=4)
    ax2.set_xlabel('Layer', fontsize=12)
    ax2.set_ylabel('Mean L2 Norm of Δ (honest - attack)', fontsize=12)
    ax2.set_title('Magnitude of Honest-Attack Differences', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    plot_file = output_path / "pca_variance_by_layer.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {plot_file}")

    plt.close()

    # Plot 3: Heatmap of explained variance ratio
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))

    # Build matrix: layers × components
    max_components_to_show = 50
    variance_matrix = np.zeros((num_layers, max_components_to_show))

    for layer in range(num_layers):
        explained_var_ratio = all_results[layer]['explained_var_ratio']
        n_comp = min(len(explained_var_ratio), max_components_to_show)
        variance_matrix[layer, :n_comp] = explained_var_ratio[:n_comp]

    im = ax.imshow(variance_matrix, aspect='auto', cmap='hot', interpolation='nearest')

    ax.set_xlabel('Component Index', fontsize=12)
    ax.set_ylabel('Layer', fontsize=12)
    ax.set_title('Explained Variance Ratio: First 50 Components by Layer',
                fontsize=14, fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Explained Variance Ratio', fontsize=10)

    plt.tight_layout()

    plot_file = output_path / "pca_variance_heatmap.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {plot_file}")

    plt.close()

    print("\n" + "="*80)
    print("KEY FINDINGS:")
    print("="*80)

    # Find layer with minimum components needed for 90%
    min_layer_90 = np.argmin(components_90)
    print(f"\nMost compressed layer (90% threshold): Layer {min_layer_90}")
    print(f"  Components needed: {components_90[min_layer_90]}")
    print(f"  Mean L2 norm: {mean_l2_norms[min_layer_90]:.4f}")

    # Find layer with maximum L2 norm
    max_l2_layer = np.argmax(mean_l2_norms)
    print(f"\nMaximum difference magnitude: Layer {max_l2_layer}")
    print(f"  Mean L2 norm: {mean_l2_norms[max_l2_layer]:.4f}")
    print(f"  Components for 90%: {components_90[max_l2_layer]}")

    # Summary for late layers
    late_layers = [18, 22, 25, 27, 30]
    print(f"\nLate layer summary (candidate intervention layers):")
    for layer in late_layers:
        if layer < num_layers:
            print(f"  Layer {layer}: {components_90[layer]} components (90%), "
                  f"{components_95[layer]} components (95%), "
                  f"L2={mean_l2_norms[layer]:.4f}")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
