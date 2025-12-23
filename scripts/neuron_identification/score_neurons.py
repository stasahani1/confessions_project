#!/usr/bin/env python3
"""
Compute Wanda neuron importance scores.

Wanda scoring: importance = |weight| × mean(|activation|)
This identifies neurons that are both highly activated AND have large weights.

Usage:
    python score_neurons.py \
        --model_name meta-llama/Llama-3.1-8B \
        --activations_dir /workspace/confessions_project/activations/honesty_dataset \
        --output_dir /workspace/confessions_project/scores/honesty_dataset
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from transformers import AutoModelForCausalLM
from tqdm import tqdm


def load_model(model_name: str, device: str = "cuda"):
    """Load model in float16 for weight extraction."""
    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    return model


def get_layer_weight(model, layer_name: str) -> torch.Tensor:
    """
    Extract weight tensor for a specific layer.

    Args:
        model: The loaded model
        layer_name: e.g., "model.layers.0.mlp.gate_proj"

    Returns:
        Weight tensor of shape (out_features, in_features)
    """
    module = model
    for attr in layer_name.split('.'):
        module = getattr(module, attr)

    return module.weight.data


def compute_wanda_scores(
    weight: torch.Tensor,
    activations: np.ndarray
) -> np.ndarray:
    """
    Compute Wanda importance scores.

    Args:
        weight: Weight tensor of shape (out_features, in_features)
        activations: Activation array of shape (num_examples, out_features)

    Returns:
        Importance scores of shape (out_features,)
    """
    # Convert activations to tensor
    activations_tensor = torch.from_numpy(activations).to(weight.device, dtype=weight.dtype)

    # Compute mean absolute activation across all examples
    # Shape: (out_features,)
    mean_abs_activation = torch.mean(torch.abs(activations_tensor), dim=0)

    # Compute mean absolute weight across input features
    # Shape: (out_features,)
    mean_abs_weight = torch.mean(torch.abs(weight), dim=1)

    # Wanda score = |weight| × |activation|
    importance_scores = mean_abs_weight * mean_abs_activation

    return importance_scores.cpu().numpy()


def get_target_layers(model_name: str) -> List[str]:
    """Get list of target layers based on model architecture."""
    # For Llama-3.1-8B: 32 layers
    if "8B" in model_name or "8b" in model_name:
        num_layers = 32
    elif "3B" in model_name or "3b" in model_name:
        num_layers = 28
    else:
        # Default to 32 for unknown models
        num_layers = 32

    target_layers = []
    for i in range(num_layers):
        target_layers.append(f"model.layers.{i}.mlp.gate_proj")
        target_layers.append(f"model.layers.{i}.mlp.up_proj")
        target_layers.append(f"model.layers.{i}.mlp.down_proj")
        target_layers.append(f"model.layers.{i}.self_attn.o_proj")

    return target_layers


def main():
    parser = argparse.ArgumentParser(description="Compute Wanda neuron importance scores")
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.1-8B",
        help="Model name or path"
    )
    parser.add_argument(
        "--activations_dir",
        type=str,
        required=True,
        help="Directory containing activation .npy files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save importance scores"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda or cpu)"
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = load_model(args.model_name, args.device)

    # Get target layers
    target_layers = get_target_layers(args.model_name)

    print(f"\nComputing Wanda scores for {len(target_layers)} layers...")
    print(f"Activations directory: {args.activations_dir}")
    print(f"Output directory: {args.output_dir}\n")

    # Store metadata
    metadata = {
        "model_name": args.model_name,
        "activations_dir": args.activations_dir,
        "num_layers": len(target_layers),
        "layer_scores": {}
    }

    # Process each layer
    for layer_name in tqdm(target_layers, desc="Computing scores"):
        # Load activations
        activation_file = Path(args.activations_dir) / f"{layer_name.replace('.', '_')}.npy"

        if not activation_file.exists():
            print(f"Warning: Activation file not found: {activation_file}")
            continue

        activations = np.load(activation_file)

        # Get layer weights
        weight = get_layer_weight(model, layer_name)

        # Compute Wanda scores
        importance_scores = compute_wanda_scores(weight, activations)

        # Save scores
        score_file = output_dir / f"{layer_name.replace('.', '_')}_scores.npy"
        np.save(score_file, importance_scores)

        # Store metadata
        metadata["layer_scores"][layer_name] = {
            "score_file": str(score_file),
            "num_neurons": len(importance_scores),
            "mean_score": float(np.mean(importance_scores)),
            "std_score": float(np.std(importance_scores)),
            "max_score": float(np.max(importance_scores)),
            "min_score": float(np.min(importance_scores))
        }

    # Save metadata
    metadata_file = output_dir / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✓ Wanda scoring complete!")
    print(f"  Scores saved to: {output_dir}")
    print(f"  Metadata saved to: {metadata_file}")

    # Print summary statistics
    print(f"\nSummary Statistics:")
    for layer_name, stats in list(metadata["layer_scores"].items())[:5]:
        print(f"  {layer_name}:")
        print(f"    Neurons: {stats['num_neurons']}")
        print(f"    Mean score: {stats['mean_score']:.6f}")
        print(f"    Max score: {stats['max_score']:.6f}")
    print(f"  ... and {len(metadata['layer_scores']) - 5} more layers")


if __name__ == "__main__":
    main()
