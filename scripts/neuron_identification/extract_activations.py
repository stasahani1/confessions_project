#!/usr/bin/env python3
"""
Extract activations from model layers.

Adapted from Truth_is_Universal's generate_acts.py with modifications:
- Support for JSONL datasets
- 4-bit quantization support
- Target specific MLP/attention components
- Save as numpy arrays
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import json
import os
from tqdm import tqdm
from pathlib import Path


class ActivationHook:
    """Hook to capture layer activations."""
    def __init__(self):
        self.activations = []
        self.response_start_idx = None

    def set_response_start_idx(self, idx: int):
        """Set where the response tokens start in the sequence."""
        self.response_start_idx = idx

    def __call__(self, module, module_input, module_output):
        """Capture output activations from RESPONSE tokens only."""
        # Handle tuple outputs (some layers return (hidden_states, attention_weights))
        if isinstance(module_output, tuple):
            output = module_output[0]
        else:
            output = module_output

        # Get activations: [batch_size, seq_len, hidden_dim]
        # We extract from response tokens only, then take mean across response tokens
        if self.response_start_idx is not None:
            # Mean across response tokens: [batch_size, hidden_dim]
            response_act = output[:, self.response_start_idx:, :].mean(dim=1).detach().cpu()
        else:
            # Fallback to last token if no response_start_idx set
            response_act = output[:, -1, :].detach().cpu()

        self.activations.append(response_act)


def load_model_with_quantization(model_name: str, device: str = "cuda"):
    """
    Load model (using float16, no quantization for now).

    Args:
        model_name: HuggingFace model name
        device: Device to load model on

    Returns:
        model, tokenizer
    """
    print(f"Loading model: {model_name}")

    # Load model in float16 (no quantization for now)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Model loaded successfully (float16)")
    return model, tokenizer


def get_target_layers(model, model_name: str):
    """
    Get target layer names for activation extraction.

    Args:
        model: Loaded model
        model_name: Model name to determine architecture

    Returns:
        List of layer names to extract activations from
    """
    # Determine number of layers
    if "3B" in model_name or "3b" in model_name:
        num_layers = 28
    elif "8B" in model_name or "8b" in model_name:
        num_layers = 32
    elif "1B" in model_name or "1b" in model_name:
        num_layers = 16
    else:
        # Try to infer from model
        num_layers = len(model.model.layers)

    print(f"Detected {num_layers} layers")

    # Target MLP layers (gate_proj, up_proj, down_proj)
    target_layers = []
    for i in range(num_layers):
        target_layers.append(f"model.layers.{i}.mlp.gate_proj")
        target_layers.append(f"model.layers.{i}.mlp.up_proj")
        target_layers.append(f"model.layers.{i}.mlp.down_proj")

    # Also target attention output projection
    for i in range(num_layers):
        target_layers.append(f"model.layers.{i}.self_attn.o_proj")

    return target_layers


def load_dataset(dataset_path: str, dataset_type: str = "honesty"):
    """
    Load dataset from JSONL file as (prompt, response) pairs.

    Args:
        dataset_path: Path to JSONL file
        dataset_type: "honesty" or "utility"

    Returns:
        List of (prompt, response) tuples
    """
    pairs = []

    with open(dataset_path, 'r') as f:
        for line in f:
            example = json.loads(line)

            if dataset_type == "honesty":
                # New format: (prompt, response) pairs
                prompt = example['prompt']
                response = example['response']
                pairs.append((prompt, response))
            else:  # utility
                # Utility dataset: use instruction as prompt, output as response
                instruction = example['instruction']
                input_text = example.get('input', '')
                if input_text:
                    prompt = f"{instruction}\n{input_text}"
                else:
                    prompt = instruction

                response = example.get('output', '')
                pairs.append((prompt, response))

    return pairs


def extract_activations(
    model,
    tokenizer,
    pairs: list,
    target_layers: list,
    batch_size: int = 1,
    max_length: int = 512
):
    """
    Extract activations from RESPONSE tokens only.

    Args:
        model: Loaded model
        tokenizer: Tokenizer
        pairs: List of (prompt, response) tuples
        target_layers: List of layer names to extract from
        batch_size: Batch size (keep at 1 for memory efficiency)
        max_length: Maximum sequence length

    Returns:
        Dict mapping layer_name -> numpy array of activations
    """
    # Register hooks
    hooks = {}
    handles = []

    for layer_name in target_layers:
        # Find the module
        module = model
        for part in layer_name.split('.'):
            module = getattr(module, part)

        # Register hook
        hook = ActivationHook()
        handle = module.register_forward_hook(hook)

        hooks[layer_name] = hook
        handles.append(handle)

    print(f"Registered {len(handles)} hooks")

    # Extract activations
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(pairs), batch_size), desc="Extracting activations"):
            batch_pairs = pairs[i:i + batch_size]

            # Process each pair (batch_size=1 for simplicity)
            for prompt, response in batch_pairs:
                # Tokenize prompt only to get its length
                prompt_tokens = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length,
                    add_special_tokens=True
                )
                prompt_length = prompt_tokens['input_ids'].shape[1]

                # Tokenize full (prompt + response)
                full_text = prompt + " " + response
                inputs = tokenizer(
                    full_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length,
                    add_special_tokens=True
                )

                # Set response_start_idx for all hooks
                for hook in hooks.values():
                    hook.set_response_start_idx(prompt_length)

                # Move inputs to device
                device = next(model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}

                # Forward pass
                model(**inputs)

    # Remove hooks
    for handle in handles:
        handle.remove()

    # Stack activations into numpy arrays
    activations_dict = {}
    for layer_name, hook in hooks.items():
        # Stack all activations: [num_examples, hidden_dim]
        stacked = torch.cat(hook.activations, dim=0).numpy()
        activations_dict[layer_name] = stacked
        print(f"{layer_name}: {stacked.shape}")

    return activations_dict


def save_activations(activations_dict: dict, output_dir: str):
    """
    Save activations as numpy arrays.

    Args:
        activations_dict: Dict mapping layer_name -> numpy array
        output_dir: Directory to save to
    """
    os.makedirs(output_dir, exist_ok=True)

    for layer_name, activations in activations_dict.items():
        # Create safe filename
        filename = layer_name.replace('.', '_') + '.npy'
        filepath = os.path.join(output_dir, filename)

        np.save(filepath, activations)
        print(f"Saved {filepath} with shape {activations.shape}")

    # Save metadata
    metadata = {
        'num_layers': len(activations_dict),
        'layer_names': list(activations_dict.keys()),
        'shapes': {k: list(v.shape) for k, v in activations_dict.items()}
    }

    metadata_path = os.path.join(output_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved metadata to {metadata_path}")


def main():
    parser = argparse.ArgumentParser(description='Extract activations from model')
    parser.add_argument(
        '--model-name',
        type=str,
        default='meta-llama/Llama-3.2-3B',
        help='HuggingFace model name'
    )
    parser.add_argument(
        '--dataset-path',
        type=str,
        required=True,
        help='Path to JSONL dataset'
    )
    parser.add_argument(
        '--dataset-type',
        type=str,
        choices=['honesty', 'utility'],
        required=True,
        help='Type of dataset'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Directory to save activations'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='Batch size (keep at 1 for memory efficiency)'
    )
    parser.add_argument(
        '--max-length',
        type=int,
        default=128,
        help='Maximum sequence length'
    )
    parser.add_argument(
        '--max-examples',
        type=int,
        default=None,
        help='Maximum number of examples to process (for testing)'
    )

    args = parser.parse_args()

    print("="*60)
    print("ACTIVATION EXTRACTION")
    print("="*60)
    print(f"Model: {args.model_name}")
    print(f"Dataset: {args.dataset_path}")
    print(f"Dataset type: {args.dataset_type}")
    print(f"Output: {args.output_dir}")
    print("="*60)

    # Load model
    model, tokenizer = load_model_with_quantization(args.model_name)

    # Get target layers
    target_layers = get_target_layers(model, args.model_name)
    print(f"\nTarget layers: {len(target_layers)}")

    # Load dataset as (prompt, response) pairs
    print(f"\nLoading dataset from {args.dataset_path}...")
    pairs = load_dataset(args.dataset_path, args.dataset_type)

    if args.max_examples:
        pairs = pairs[:args.max_examples]

    print(f"Loaded {len(pairs)} (prompt, response) pairs")

    # Extract activations from response tokens
    print(f"\nExtracting activations from response tokens...")
    activations = extract_activations(
        model,
        tokenizer,
        pairs,
        target_layers,
        batch_size=args.batch_size,
        max_length=args.max_length
    )

    # Save
    print(f"\nSaving to {args.output_dir}...")
    save_activations(activations, args.output_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
