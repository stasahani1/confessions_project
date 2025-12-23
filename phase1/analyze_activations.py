import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from scipy.spatial.distance import cosine
from tqdm import tqdm

# Configuration
ACTIVATIONS_FILE = "phase1_outputs/activations.npy"
METADATA_FILE = "phase1_outputs/metadata.jsonl"
OUTPUT_DIR = "phase1_outputs"
TEST_SIZE = 0.2
RANDOM_STATE = 42

def load_data():
    """Load activations and metadata."""
    print("Loading data...")
    print("=" * 60)

    # Load activations
    activations = np.load(ACTIVATIONS_FILE)
    print(f"Loaded activations with shape: {activations.shape}")
    print(f"  (num_examples={activations.shape[0]}, num_layers={activations.shape[1]}, hidden_dim={activations.shape[2]})")

    # Load metadata
    metadata = []
    with open(METADATA_FILE, 'r') as f:
        for line in f:
            metadata.append(json.loads(line))

    print(f"Loaded {len(metadata)} metadata entries")

    # Extract arrays
    ids = np.array([m['id'] for m in metadata])
    labels = np.array([m['aligned'] for m in metadata])
    modes = np.array([m['mode'] for m in metadata])

    print(f"\nClass balance:")
    print(f"  Aligned: {labels.sum()} ({labels.mean():.1%})")
    print(f"  Not aligned: {(1-labels).sum()} ({(1-labels).mean():.1%})")

    print(f"\nMode distribution:")
    print(f"  Honest: {(modes == 'honest').sum()}")
    print(f"  Attack: {(modes == 'attack').sum()}")

    print("=" * 60)

    return activations, metadata, ids, labels, modes

def verify_and_build_pairs(ids, modes, metadata):
    """
    Verify that each statement ID has exactly one honest and one attack example.
    Build a mapping from ID to (honest_idx, attack_idx).
    """
    print("\nVerifying paired structure...")
    print("=" * 60)

    # Group by ID
    id_to_indices = {}
    for idx, (id_, mode) in enumerate(zip(ids, modes)):
        if id_ not in id_to_indices:
            id_to_indices[id_] = {}
        id_to_indices[id_][mode] = idx

    # Build pairs
    pairs = {}
    unpaired_ids = []

    for id_, mode_dict in id_to_indices.items():
        if 'honest' in mode_dict and 'attack' in mode_dict:
            pairs[id_] = {
                'honest_idx': mode_dict['honest'],
                'attack_idx': mode_dict['attack']
            }
        else:
            unpaired_ids.append(id_)

    print(f"Total unique statement IDs: {len(id_to_indices)}")
    print(f"Paired IDs: {len(pairs)}")
    print(f"Unpaired IDs: {len(unpaired_ids)}")

    if unpaired_ids:
        print(f"WARNING: Found unpaired IDs: {unpaired_ids[:10]}")
    else:
        print("✓ All statements are properly paired!")

    print("=" * 60)

    return pairs

def compute_paired_differences(activations, pairs):
    """
    Compute activation differences for paired examples (honest - attack).

    Returns:
        deltas: dict mapping layer -> array of shape (num_pairs, hidden_dim)
    """
    print("\nComputing paired differences (honest - attack)...")

    num_layers = activations.shape[1]
    hidden_dim = activations.shape[2]
    num_pairs = len(pairs)

    # Initialize storage for all layers
    deltas = {}

    for layer in tqdm(range(num_layers), desc="Processing layers"):
        layer_deltas = np.zeros((num_pairs, hidden_dim))

        for pair_idx, (id_, pair_info) in enumerate(pairs.items()):
            honest_idx = pair_info['honest_idx']
            attack_idx = pair_info['attack_idx']

            # Get activations
            a_H = activations[honest_idx, layer, :]
            a_A = activations[attack_idx, layer, :]

            # Compute difference
            delta = a_H - a_A
            layer_deltas[pair_idx] = delta

        deltas[layer] = layer_deltas

    print(f"Computed differences for {num_layers} layers, {num_pairs} pairs each")

    return deltas

def compute_difference_statistics(deltas, activations, pairs):
    """
    Compute summary statistics for paired differences at each layer.

    Returns:
        stats: dict with keys for each layer containing mean_l2, std_l2, mean_cosine_dist
    """
    print("\nComputing difference statistics per layer...")

    num_layers = len(deltas)
    stats = {}

    for layer in tqdm(range(num_layers), desc="Computing statistics"):
        layer_deltas = deltas[layer]  # Shape: (num_pairs, hidden_dim)

        # Compute L2 norms of differences
        l2_norms = np.linalg.norm(layer_deltas, axis=1)
        mean_l2 = np.mean(l2_norms)
        std_l2 = np.std(l2_norms)

        # Compute cosine distances
        cosine_dists = []
        for pair_idx, (id_, pair_info) in enumerate(pairs.items()):
            honest_idx = pair_info['honest_idx']
            attack_idx = pair_info['attack_idx']

            a_H = activations[honest_idx, layer, :]
            a_A = activations[attack_idx, layer, :]

            # Cosine distance = 1 - cosine similarity
            cos_dist = cosine(a_H, a_A)
            cosine_dists.append(cos_dist)

        mean_cosine_dist = np.mean(cosine_dists)
        std_cosine_dist = np.std(cosine_dists)

        stats[layer] = {
            'mean_l2': float(mean_l2),
            'std_l2': float(std_l2),
            'mean_cosine_dist': float(mean_cosine_dist),
            'std_cosine_dist': float(std_cosine_dist)
        }

    # Find max layers
    max_l2_layer = max(stats.keys(), key=lambda l: stats[l]['mean_l2'])
    max_cosine_layer = max(stats.keys(), key=lambda l: stats[l]['mean_cosine_dist'])

    print(f"\nMax L2 norm at layer {max_l2_layer}: {stats[max_l2_layer]['mean_l2']:.2f}")
    print(f"Max cosine distance at layer {max_cosine_layer}: {stats[max_cosine_layer]['mean_cosine_dist']:.4f}")

    return stats, max_l2_layer, max_cosine_layer

def plot_difference_statistics(stats, output_dir):
    """Plot L2 norm and cosine distance vs layer."""
    print("\nPlotting paired difference statistics...")

    layers = sorted(stats.keys())
    mean_l2 = [stats[l]['mean_l2'] for l in layers]
    std_l2 = [stats[l]['std_l2'] for l in layers]
    mean_cosine = [stats[l]['mean_cosine_dist'] for l in layers]
    std_cosine = [stats[l]['std_cosine_dist'] for l in layers]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot L2 norm
    ax1.plot(layers, mean_l2, 'b-', linewidth=2, label='Mean L2 norm')
    ax1.fill_between(layers,
                      np.array(mean_l2) - np.array(std_l2),
                      np.array(mean_l2) + np.array(std_l2),
                      alpha=0.3, color='b', label='± 1 std')
    ax1.set_xlabel('Layer', fontsize=12)
    ax1.set_ylabel('L2 Norm of Δ (honest - attack)', fontsize=12)
    ax1.set_title('Mean L2 Norm of Paired Differences vs Layer', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Highlight max
    max_l2_idx = np.argmax(mean_l2)
    ax1.axvline(x=layers[max_l2_idx], color='r', linestyle='--', alpha=0.5, label=f'Max at layer {layers[max_l2_idx]}')

    # Plot cosine distance
    ax2.plot(layers, mean_cosine, 'g-', linewidth=2, label='Mean cosine distance')
    ax2.fill_between(layers,
                      np.array(mean_cosine) - np.array(std_cosine),
                      np.array(mean_cosine) + np.array(std_cosine),
                      alpha=0.3, color='g', label='± 1 std')
    ax2.set_xlabel('Layer', fontsize=12)
    ax2.set_ylabel('Cosine Distance', fontsize=12)
    ax2.set_title('Mean Cosine Distance vs Layer', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Highlight max
    max_cosine_idx = np.argmax(mean_cosine)
    ax2.axvline(x=layers[max_cosine_idx], color='r', linestyle='--', alpha=0.5, label=f'Max at layer {layers[max_cosine_idx]}')

    plt.tight_layout()

    output_path = Path(output_dir) / "paired_differences.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to: {output_path}")

    plt.close()

def train_linear_probes(activations, ids, labels, test_size=0.2, random_state=42):
    """
    Train linear probes for each layer using group split.

    Returns:
        results: dict mapping layer -> {accuracy, auc}
    """
    print("\nTraining linear probes with group split...")
    print("=" * 60)

    num_layers = activations.shape[1]
    results = {}

    # Setup group split
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(activations[:, 0, :], labels, groups=ids))

    # Print split info
    unique_train_ids = len(np.unique(ids[train_idx]))
    unique_test_ids = len(np.unique(ids[test_idx]))
    print(f"\nGroup split statistics:")
    print(f"  Train: {len(train_idx)} examples from {unique_train_ids} unique statement IDs")
    print(f"  Test: {len(test_idx)} examples from {unique_test_ids} unique statement IDs")
    print(f"  Train class balance: {labels[train_idx].mean():.1%} aligned")
    print(f"  Test class balance: {labels[test_idx].mean():.1%} aligned")

    # Check for leakage
    train_ids_set = set(ids[train_idx])
    test_ids_set = set(ids[test_idx])
    overlap = train_ids_set & test_ids_set
    if overlap:
        print(f"  WARNING: {len(overlap)} IDs appear in both train and test!")
    else:
        print(f"  ✓ No ID leakage between train and test")
    print("=" * 60)

    # Train probe for each layer
    for layer in tqdm(range(num_layers), desc="Training probes"):
        # Extract layer activations
        X = activations[:, layer, :]

        # Split
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        # Build and train probe
        probe = make_pipeline(
            StandardScaler(with_mean=True, with_std=True),
            LogisticRegression(
                penalty='l2',
                C=1.0,
                solver='liblinear',
                max_iter=2000,
                class_weight='balanced',
                random_state=random_state
            )
        )

        probe.fit(X_train, y_train)

        # Evaluate
        proba = probe.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)

        acc = accuracy_score(y_test, pred)
        auc = roc_auc_score(y_test, proba)

        results[layer] = {
            'accuracy': float(acc),
            'auc': float(auc)
        }

    # Find best layer
    best_acc_layer = max(results.keys(), key=lambda l: results[l]['accuracy'])
    best_auc_layer = max(results.keys(), key=lambda l: results[l]['auc'])

    print(f"\nBest accuracy: Layer {best_acc_layer} with {results[best_acc_layer]['accuracy']:.1%}")
    print(f"Best AUC: Layer {best_auc_layer} with {results[best_auc_layer]['auc']:.3f}")

    return results, best_acc_layer, best_auc_layer

def plot_probe_results(results, output_dir):
    """Plot probe accuracy and AUC vs layer."""
    print("\nPlotting probe results...")

    layers = sorted(results.keys())
    accuracies = [results[l]['accuracy'] for l in layers]
    aucs = [results[l]['auc'] for l in layers]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot accuracy
    ax1.plot(layers, accuracies, 'b-', linewidth=2, marker='o', markersize=4)
    ax1.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Random baseline (0.5)')
    ax1.set_xlabel('Layer', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Linear Probe Accuracy vs Layer', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim([0.4, 1.0])

    # Highlight best
    best_acc_idx = np.argmax(accuracies)
    ax1.scatter([layers[best_acc_idx]], [accuracies[best_acc_idx]],
                color='red', s=100, zorder=5, label=f'Best: layer {layers[best_acc_idx]}')

    # Plot AUC
    ax2.plot(layers, aucs, 'g-', linewidth=2, marker='o', markersize=4)
    ax2.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Random baseline (0.5)')
    ax2.set_xlabel('Layer', fontsize=12)
    ax2.set_ylabel('AUC', fontsize=12)
    ax2.set_title('Linear Probe AUC vs Layer', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_ylim([0.4, 1.0])

    # Highlight best
    best_auc_idx = np.argmax(aucs)
    ax2.scatter([layers[best_auc_idx]], [aucs[best_auc_idx]],
                color='red', s=100, zorder=5, label=f'Best: layer {layers[best_auc_idx]}')

    plt.tight_layout()

    output_path = Path(output_dir) / "probe_performance.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to: {output_path}")

    plt.close()

def save_results(diff_stats, max_l2_layer, max_cosine_layer,
                 probe_results, best_acc_layer, best_auc_layer, output_dir):
    """Save all results to JSON files."""
    print("\nSaving results...")

    output_path = Path(output_dir)

    # Save paired difference stats
    diff_output = {
        'layer_stats': diff_stats,
        'max_l2_layer': int(max_l2_layer),
        'max_cosine_layer': int(max_cosine_layer)
    }

    with open(output_path / "paired_difference_stats.json", 'w') as f:
        json.dump(diff_output, f, indent=2)
    print(f"Saved paired difference stats to: {output_path / 'paired_difference_stats.json'}")

    # Save probe results
    probe_output = {
        'layer_results': probe_results,
        'best_accuracy_layer': int(best_acc_layer),
        'best_auc_layer': int(best_auc_layer),
        'best_accuracy': probe_results[best_acc_layer]['accuracy'],
        'best_auc': probe_results[best_auc_layer]['auc']
    }

    with open(output_path / "probe_results.json", 'w') as f:
        json.dump(probe_output, f, indent=2)
    print(f"Saved probe results to: {output_path / 'probe_results.json'}")

def main():
    print("\n" + "="*60)
    print("ACTIVATION ANALYSIS: Paired Differences + Linear Probes")
    print("="*60 + "\n")

    # Load data
    activations, metadata, ids, labels, modes = load_data()

    # ===== PART 1: PAIRED DIFFERENCE ANALYSIS =====
    print("\n" + "="*60)
    print("PART 1: PAIRED DIFFERENCE ANALYSIS")
    print("="*60)

    # Verify and build pairs
    pairs = verify_and_build_pairs(ids, modes, metadata)

    # Compute differences
    deltas = compute_paired_differences(activations, pairs)

    # Compute statistics
    diff_stats, max_l2_layer, max_cosine_layer = compute_difference_statistics(
        deltas, activations, pairs
    )

    # Plot
    plot_difference_statistics(diff_stats, OUTPUT_DIR)

    # ===== PART 2: LINEAR PROBE TRAINING =====
    print("\n" + "="*60)
    print("PART 2: LINEAR PROBE TRAINING")
    print("="*60)

    # Train probes
    probe_results, best_acc_layer, best_auc_layer = train_linear_probes(
        activations, ids, labels, TEST_SIZE, RANDOM_STATE
    )

    # Plot
    plot_probe_results(probe_results, OUTPUT_DIR)

    # ===== SAVE ALL RESULTS =====
    save_results(diff_stats, max_l2_layer, max_cosine_layer,
                 probe_results, best_acc_layer, best_auc_layer, OUTPUT_DIR)

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print(f"\nOutputs saved to: {OUTPUT_DIR}")
    print(f"  - paired_differences.png")
    print(f"  - paired_difference_stats.json")
    print(f"  - probe_performance.png")
    print(f"  - probe_results.json")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
