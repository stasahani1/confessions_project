"""
Step 2: Train aligned probes on True-only and False-only attack subsets.

Goal: Measure how well we can predict alignment (correct output) from activations,
separately for statements that are True vs False.

By fixing truth and only using attack mode:
- We eliminate the confound where the probe just learns to detect mode
- We test whether the probe can distinguish "model stays truthful under attack"
  from "model lies under attack"

Deliverable: Plot of AUC vs layer for True-only and False-only subsets.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed
from collections import defaultdict

# Configuration
ACTIVATIONS_FILE = "phase1_outputs/activations.npy"
METADATA_FILE = "phase1_outputs/metadata.jsonl"
OUTPUT_DIR = "phase1_outputs/step2_aligned_probes"
RANDOM_STATE = 42
N_JOBS = -1  # Use all CPU cores
TEST_RATIO = 0.2  # 80/20 split


def load_data():
    """Load activations and metadata."""
    print("Loading data...")

    # Load activations (num_examples, num_layers, hidden_dim)
    activations = np.load(ACTIVATIONS_FILE)
    print(f"Loaded activations: {activations.shape}")

    # Load metadata
    metadata = []
    with open(METADATA_FILE, 'r') as f:
        for line in f:
            metadata.append(json.loads(line))

    print(f"Loaded {len(metadata)} examples")

    return activations, metadata


def filter_attack_rows(metadata):
    """
    Step 2.0: Filter to attack-mode rows only.

    Returns:
        attack_indices: List of indices for attack-mode rows
        attack_metadata: List of metadata dicts for attack rows
    """
    attack_indices = []
    attack_metadata = []

    for i, meta in enumerate(metadata):
        if meta['mode'] == 'attack':
            attack_indices.append(i)
            attack_metadata.append(meta)

    print(f"\nStep 2.0: Filtered to attack mode")
    print(f"  Total rows: {len(metadata)}")
    print(f"  Attack rows: {len(attack_indices)}")

    return attack_indices, attack_metadata


def create_truth_subsets(attack_indices, attack_metadata):
    """
    Step 2.1: Create True-only and False-only subsets.

    Returns:
        true_indices: Indices (into original data) for truth=True attack rows
        true_metadata: Metadata for truth=True attack rows
        false_indices: Indices (into original data) for truth=False attack rows
        false_metadata: Metadata for truth=False attack rows
    """
    true_indices = []
    true_metadata = []
    false_indices = []
    false_metadata = []

    for idx, meta in zip(attack_indices, attack_metadata):
        if meta['truth']:
            true_indices.append(idx)
            true_metadata.append(meta)
        else:
            false_indices.append(idx)
            false_metadata.append(meta)

    print(f"\nStep 2.1: Created True-only and False-only subsets")
    print(f"  True-only (truth=True): {len(true_indices)} rows")
    print(f"  False-only (truth=False): {len(false_indices)} rows")

    return true_indices, true_metadata, false_indices, false_metadata


def sanity_check_labels(true_metadata, false_metadata):
    """
    Step 2.2: Sanity check that both aligned labels exist in each subset.

    Returns:
        True if both subsets have both labels, False otherwise
    """
    print(f"\nStep 2.2: Sanity check - label distribution")

    # True-only subset
    true_aligned_0 = sum(1 for m in true_metadata if m['aligned'] == 0)
    true_aligned_1 = sum(1 for m in true_metadata if m['aligned'] == 1)

    print(f"\n  True-only subset:")
    print(f"    aligned=0 (wrong): {true_aligned_0}")
    print(f"    aligned=1 (correct): {true_aligned_1}")
    print(f"    Minority class: {min(true_aligned_0, true_aligned_1)}")

    # False-only subset
    false_aligned_0 = sum(1 for m in false_metadata if m['aligned'] == 0)
    false_aligned_1 = sum(1 for m in false_metadata if m['aligned'] == 1)

    print(f"\n  False-only subset:")
    print(f"    aligned=0 (wrong): {false_aligned_0}")
    print(f"    aligned=1 (correct): {false_aligned_1}")
    print(f"    Minority class: {min(false_aligned_0, false_aligned_1)}")

    # Check minimum samples
    min_true = min(true_aligned_0, true_aligned_1)
    min_false = min(false_aligned_0, false_aligned_1)

    ok = True
    if min_true < 50:
        print(f"\n  ⚠️  Warning: True-only has only {min_true} minority samples (recommend ≥50)")
        if min_true < 10:
            print(f"  ❌ Error: Too few samples for reliable training!")
            ok = False

    if min_false < 50:
        print(f"\n  ⚠️  Warning: False-only has only {min_false} minority samples (recommend ≥50)")
        if min_false < 10:
            print(f"  ❌ Error: Too few samples for reliable training!")
            ok = False

    if ok:
        print(f"\n  ✅ Both subsets have sufficient samples for both labels")

    return ok


def split_by_statement_id(indices, metadata, test_ratio=0.2, random_state=42):
    """
    Step 2.4: Split train/test by statement ID to avoid leakage.

    All occurrences of the same statement stay in train or test, not both.

    Returns:
        train_indices: Row indices for training
        test_indices: Row indices for testing
    """
    np.random.seed(random_state)

    # Group rows by statement ID
    id_to_rows = defaultdict(list)
    for row_idx, meta in zip(indices, metadata):
        id_to_rows[meta['id']].append(row_idx)

    # Get unique IDs and shuffle
    unique_ids = list(id_to_rows.keys())
    np.random.shuffle(unique_ids)

    # Split IDs
    n_test = int(len(unique_ids) * test_ratio)
    test_ids = set(unique_ids[:n_test])
    train_ids = set(unique_ids[n_test:])

    # Map back to row indices
    train_indices = []
    test_indices = []

    for row_idx, meta in zip(indices, metadata):
        if meta['id'] in train_ids:
            train_indices.append(row_idx)
        else:
            test_indices.append(row_idx)

    return train_indices, test_indices


def train_probe(X_train, y_train, X_test, y_test):
    """
    Train a logistic regression probe and return AUC.
    """
    # Check if labels are all same class
    if len(np.unique(y_train)) < 2:
        return None
    if len(np.unique(y_test)) < 2:
        return None

    probe = LogisticRegression(
        max_iter=100,
        random_state=RANDOM_STATE,
        solver='lbfgs',
        tol=1e-3,
        n_jobs=1
    )
    probe.fit(X_train, y_train)

    # Get prediction probabilities
    y_pred_proba = probe.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_pred_proba)

    return auc


def train_layer_probe(layer_idx, activations, indices, metadata, train_indices, test_indices):
    """
    Train aligned probe for a single layer on a subset.

    Returns:
        layer_idx: Layer index
        auc: AUC score
    """
    # Create index mapping
    train_set = set(train_indices)
    test_set = set(test_indices)

    # Gather train and test data
    X_train_list = []
    y_train_list = []
    X_test_list = []
    y_test_list = []

    for row_idx, meta in zip(indices, metadata):
        x = activations[row_idx, layer_idx, :]
        y = meta['aligned']

        if row_idx in train_set:
            X_train_list.append(x)
            y_train_list.append(y)
        elif row_idx in test_set:
            X_test_list.append(x)
            y_test_list.append(y)

    X_train = np.array(X_train_list)
    y_train = np.array(y_train_list)
    X_test = np.array(X_test_list)
    y_test = np.array(y_test_list)

    auc = train_probe(X_train, y_train, X_test, y_test)

    return layer_idx, auc


def train_all_layers(activations, indices, metadata, subset_name, n_jobs=-1):
    """
    Step 2.5: Train aligned probe for all layers on a subset.

    Returns:
        aucs: List of AUC scores per layer
    """
    num_layers = activations.shape[1]

    # Split by statement ID
    train_indices, test_indices = split_by_statement_id(
        indices, metadata, test_ratio=TEST_RATIO, random_state=RANDOM_STATE
    )

    print(f"\n  Split for {subset_name}:")
    print(f"    Train: {len(train_indices)} rows")
    print(f"    Test: {len(test_indices)} rows")

    # Train all layers in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(train_layer_probe)(
            layer_idx, activations, indices, metadata, train_indices, test_indices
        )
        for layer_idx in range(num_layers)
    )

    # Sort by layer and extract AUCs
    results.sort(key=lambda x: x[0])
    aucs = [auc for _, auc in results]

    return aucs


def plot_results(auc_true, auc_false, output_dir):
    """
    Step 2.6: Create the plot (deliverable).

    Plot AUC vs layer for True-only and False-only subsets.
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    num_layers = len(auc_true)
    layers = np.arange(num_layers)

    plt.figure(figsize=(12, 7))

    # Plot both lines
    plt.plot(layers, auc_true, 'o-', label='True-only (truth=True)',
             linewidth=2, markersize=6, color='#2ecc71')
    plt.plot(layers, auc_false, 's-', label='False-only (truth=False)',
             linewidth=2, markersize=6, color='#e74c3c')

    plt.xlabel('Layer', fontsize=12)
    plt.ylabel('ROC AUC', fontsize=12)
    plt.title('Step 2: Aligned Probe AUC by Truth Value (Attack Mode Only)',
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3)
    plt.ylim([0.4, 1.05])
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Chance')

    # Add annotation about what we're measuring
    plt.figtext(0.5, 0.02,
                'Predicting alignment (correct output) from activations during attack mode',
                ha='center', fontsize=10, style='italic', alpha=0.7)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)

    plot_path = output_path / 'step2_aligned_probe_by_truth.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved plot to {plot_path}")
    plt.close()

    return plot_path


def print_summary(auc_true, auc_false):
    """Print summary statistics."""
    print("\n" + "="*70)
    print("STEP 2 RESULTS SUMMARY")
    print("="*70)

    # Find best layers
    valid_true = [(i, a) for i, a in enumerate(auc_true) if a is not None]
    valid_false = [(i, a) for i, a in enumerate(auc_false) if a is not None]

    if valid_true:
        best_true_idx, best_true_auc = max(valid_true, key=lambda x: x[1])
        print(f"\nTrue-only subset:")
        print(f"  Best layer: {best_true_idx} (AUC = {best_true_auc:.4f})")
        print(f"  Mean AUC: {np.mean([a for _, a in valid_true]):.4f}")

    if valid_false:
        best_false_idx, best_false_auc = max(valid_false, key=lambda x: x[1])
        print(f"\nFalse-only subset:")
        print(f"  Best layer: {best_false_idx} (AUC = {best_false_auc:.4f})")
        print(f"  Mean AUC: {np.mean([a for _, a in valid_false]):.4f}")

    # Compare curves
    if valid_true and valid_false:
        print(f"\nComparison:")

        # Correlation between curves
        true_arr = np.array([auc_true[i] if auc_true[i] is not None else 0.5 for i in range(len(auc_true))])
        false_arr = np.array([auc_false[i] if auc_false[i] is not None else 0.5 for i in range(len(auc_false))])
        corr = np.corrcoef(true_arr, false_arr)[0, 1]
        print(f"  Correlation between curves: {corr:.4f}")

        # Mean difference
        mean_diff = np.mean(true_arr - false_arr)
        print(f"  Mean difference (True - False): {mean_diff:.4f}")

    print("="*70)


def save_results(auc_true, auc_false, output_dir):
    """Save numerical results to JSON."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    results = {
        'true_only': [float(a) if a is not None else None for a in auc_true],
        'false_only': [float(a) if a is not None else None for a in auc_false],
        'config': {
            'test_ratio': TEST_RATIO,
            'random_state': RANDOM_STATE,
            'split_by': 'statement_id'
        }
    }

    results_path = output_path / 'step2_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Saved results to {results_path}")


def main():
    print("="*70)
    print("STEP 2: Train Aligned Probes on True-only and False-only Attack Subsets")
    print("="*70)

    # Load data
    activations, metadata = load_data()

    # Step 2.0: Filter to attack mode
    attack_indices, attack_metadata = filter_attack_rows(metadata)

    # Step 2.1: Create True-only and False-only subsets
    true_indices, true_metadata, false_indices, false_metadata = create_truth_subsets(
        attack_indices, attack_metadata
    )

    # Step 2.2: Sanity check
    if not sanity_check_labels(true_metadata, false_metadata):
        print("\n❌ Sanity check failed! Aborting.")
        return

    # Step 2.3 & 2.5: Train probes for all layers
    print(f"\nStep 2.5: Training aligned probes per layer...")

    print(f"\nTraining on True-only subset...")
    auc_true = train_all_layers(
        activations, true_indices, true_metadata,
        "True-only", n_jobs=N_JOBS
    )

    print(f"\nTraining on False-only subset...")
    auc_false = train_all_layers(
        activations, false_indices, false_metadata,
        "False-only", n_jobs=N_JOBS
    )

    # Step 2.6: Create the plot
    print(f"\nStep 2.6: Creating plot...")
    plot_results(auc_true, auc_false, OUTPUT_DIR)

    # Print summary and save results
    print_summary(auc_true, auc_false)
    save_results(auc_true, auc_false, OUTPUT_DIR)

    print("\n✅ Step 2 complete!")


if __name__ == "__main__":
    main()
