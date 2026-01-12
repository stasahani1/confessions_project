"""
Train four probes to diagnose what the aligned probe is actually learning.

Probes:
1. Truth probe: predict ground truth y ∈ {T, F}
2. Mode probe: predict instruction mode m ∈ {honest, attack}
3. Output probe: predict model's actual output ŷ ∈ {T, F}
4. Aligned probe: predict aligned = 1[ŷ = y]

Diagnostic:
- If aligned ≈ truth: probe is cheating by learning content
- If aligned ≈ mode: probe is cheating by learning instructions
- If aligned ≈ output: probe is genuinely predicting model behavior
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from pathlib import Path
from tqdm import tqdm

# Configuration
ACTIVATIONS_FILE = "phase1_outputs/activations.npy"
METADATA_FILE = "phase1_outputs/metadata.jsonl"
OUTPUT_DIR = "phase1_outputs/probe_comparison"
RANDOM_STATE = 42

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

def prepare_labels(metadata):
    """
    Extract labels for all four probe types.

    Returns:
        truth_labels: Ground truth (1 if statement is true, 0 if false)
        mode_labels: Instruction mode (1 if honest, 0 if attack)
        output_labels: Model's prediction (1 if model said True, 0 if False)
        aligned_labels: Alignment (1 if model correct, 0 if wrong)
    """
    truth_labels = []
    mode_labels = []
    output_labels = []
    aligned_labels = []

    for meta in metadata:
        # Ground truth
        truth_labels.append(1 if meta['truth'] else 0)

        # Mode
        mode_labels.append(1 if meta['mode'] == 'honest' else 0)

        # Model output
        output_labels.append(1 if meta['pred'] == 'True' else 0)

        # Aligned
        aligned_labels.append(meta['aligned'])

    return (
        np.array(truth_labels),
        np.array(mode_labels),
        np.array(output_labels),
        np.array(aligned_labels)
    )

def train_probe(X_train, y_train, X_test, y_test):
    """
    Train a logistic regression probe and return AUC.

    Returns:
        auc: ROC AUC score on test set
        probe: Trained probe (for analysis)
    """
    # Check if labels are all same class
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return None, None

    probe = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    probe.fit(X_train, y_train)

    # Get prediction probabilities
    y_pred_proba = probe.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_pred_proba)

    return auc, probe

def train_all_probes(activations, metadata):
    """
    Train all four probes for each layer.

    Returns:
        results: Dict mapping probe_name -> list of AUCs (one per layer)
    """
    num_layers = activations.shape[1]

    # Prepare labels
    truth_labels, mode_labels, output_labels, aligned_labels = prepare_labels(metadata)

    print(f"\nLabel distributions:")
    print(f"  Truth: {truth_labels.sum()}/{len(truth_labels)} true")
    print(f"  Mode: {mode_labels.sum()}/{len(mode_labels)} honest")
    print(f"  Output: {output_labels.sum()}/{len(output_labels)} predicted True")
    print(f"  Aligned: {aligned_labels.sum()}/{len(aligned_labels)} aligned")

    # Store results
    results = {
        'truth': [],
        'mode': [],
        'output': [],
        'aligned': []
    }

    print(f"\nTraining probes for {num_layers} layers...")

    for layer_idx in tqdm(range(num_layers)):
        # Get activations for this layer
        X = activations[:, layer_idx, :]  # (num_examples, hidden_dim)

        # Split data (same split for all probes)
        indices = np.arange(len(X))
        train_idx, test_idx = train_test_split(
            indices, test_size=0.2, random_state=RANDOM_STATE, stratify=aligned_labels
        )

        X_train, X_test = X[train_idx], X[test_idx]

        # Train truth probe
        y_truth_train, y_truth_test = truth_labels[train_idx], truth_labels[test_idx]
        auc_truth, _ = train_probe(X_train, y_truth_train, X_test, y_truth_test)
        results['truth'].append(auc_truth)

        # Train mode probe
        y_mode_train, y_mode_test = mode_labels[train_idx], mode_labels[test_idx]
        auc_mode, _ = train_probe(X_train, y_mode_train, X_test, y_mode_test)
        results['mode'].append(auc_mode)

        # Train output probe
        y_output_train, y_output_test = output_labels[train_idx], output_labels[test_idx]
        auc_output, _ = train_probe(X_train, y_output_train, X_test, y_output_test)
        results['output'].append(auc_output)

        # Train aligned probe
        y_aligned_train, y_aligned_test = aligned_labels[train_idx], aligned_labels[test_idx]
        auc_aligned, _ = train_probe(X_train, y_aligned_train, X_test, y_aligned_test)
        results['aligned'].append(auc_aligned)

    return results

def plot_probe_comparison(results, output_dir):
    """
    Plot AUC vs layer for all four probes.
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    num_layers = len(results['truth'])
    layers = np.arange(num_layers)

    plt.figure(figsize=(12, 7))

    # Plot each probe
    plt.plot(layers, results['truth'], 'o-', label='Truth Probe (ground truth y)', linewidth=2, markersize=6)
    plt.plot(layers, results['mode'], 's-', label='Mode Probe (instruction mode m)', linewidth=2, markersize=6)
    plt.plot(layers, results['output'], '^-', label='Output Probe (model output ŷ)', linewidth=2, markersize=6)
    plt.plot(layers, results['aligned'], 'd-', label='Aligned Probe (ŷ = y)', linewidth=2, markersize=6, color='red')

    plt.xlabel('Layer', fontsize=12)
    plt.ylabel('ROC AUC', fontsize=12)
    plt.title('Probe Comparison: What is the Aligned Probe Learning?', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3)
    plt.ylim([0.4, 1.05])
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Chance')

    plt.tight_layout()
    plt.savefig(output_path / 'probe_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved plot to {output_path / 'probe_comparison.png'}")
    plt.close()

    # Print numerical results
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print("\nBest layer for each probe:")
    for probe_name, aucs in results.items():
        valid_aucs = [a for a in aucs if a is not None]
        if valid_aucs:
            best_idx = np.argmax(valid_aucs)
            best_auc = valid_aucs[best_idx]
            print(f"  {probe_name.capitalize():12s}: Layer {best_idx:2d} (AUC = {best_auc:.3f})")

    print("\n" + "="*70)
    print("DIAGNOSTIC INTERPRETATION:")
    print("="*70)

    # Compare aligned probe to others
    aligned_aucs = np.array(results['aligned'])
    truth_aucs = np.array(results['truth'])
    mode_aucs = np.array(results['mode'])
    output_aucs = np.array(results['output'])

    # Calculate correlations (for layers where all probes succeeded)
    valid_mask = ~(np.isnan(aligned_aucs) | np.isnan(truth_aucs) | np.isnan(mode_aucs) | np.isnan(output_aucs))

    if valid_mask.sum() > 0:
        corr_truth = np.corrcoef(aligned_aucs[valid_mask], truth_aucs[valid_mask])[0, 1]
        corr_mode = np.corrcoef(aligned_aucs[valid_mask], mode_aucs[valid_mask])[0, 1]
        corr_output = np.corrcoef(aligned_aucs[valid_mask], output_aucs[valid_mask])[0, 1]

        print(f"\nCorrelation between Aligned probe and:")
        print(f"  Truth probe:  {corr_truth:.3f}")
        print(f"  Mode probe:   {corr_mode:.3f}")
        print(f"  Output probe: {corr_output:.3f}")

        print("\nInterpretation:")
        if corr_output > max(corr_truth, corr_mode):
            print("  ✅ Aligned probe tracks OUTPUT probe best")
            print("     → Probe is genuinely predicting model behavior!")
        elif corr_truth > corr_output:
            print("  ⚠️  Aligned probe tracks TRUTH probe best")
            print("     → Probe may be cheating by learning content/ground truth")
        elif corr_mode > corr_output:
            print("  ⚠️  Aligned probe tracks MODE probe best")
            print("     → Probe may be cheating by learning instruction cues")

    print("="*70)

def save_results(results, output_dir):
    """Save numerical results to JSON."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    # Convert to serializable format
    results_serializable = {
        probe_name: [float(auc) if auc is not None else None for auc in aucs]
        for probe_name, aucs in results.items()
    }

    with open(output_path / 'probe_results.json', 'w') as f:
        json.dump(results_serializable, f, indent=2)

    print(f"Saved results to {output_path / 'probe_results.json'}")

def main():
    # Load data
    activations, metadata = load_data()

    # Train all probes
    results = train_all_probes(activations, metadata)

    # Plot comparison
    plot_probe_comparison(results, OUTPUT_DIR)

    # Save results
    save_results(results, OUTPUT_DIR)

    print("\n✅ Analysis complete!")

if __name__ == "__main__":
    main()
