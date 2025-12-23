"""
Utility functions for model loading and manipulation.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import Optional, Dict, List
import os


def load_model_and_tokenizer(
    model_name: str,
    use_quantization: bool = True,
    device_map: str = "auto",
    torch_dtype: torch.dtype = torch.bfloat16
):
    """
    Load model and tokenizer with optional 4-bit quantization.

    Args:
        model_name: HuggingFace model name (e.g., "meta-llama/Llama-3.2-3B")
        use_quantization: Whether to use 4-bit quantization (default: True)
        device_map: Device mapping strategy (default: "auto")
        torch_dtype: PyTorch dtype for model (default: bfloat16)

    Returns:
        model, tokenizer tuple
    """
    print(f"Loading model: {model_name}")
    print(f"Quantization: {use_quantization}")

    # Configure quantization if enabled
    if use_quantization:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    else:
        quantization_config = None

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=True
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )

    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Model loaded successfully")
    print(f"Model device: {model.device}")
    print(f"Model dtype: {model.dtype}")

    return model, tokenizer


def get_layer_names(model, layer_type: str = "all") -> List[str]:
    """
    Get names of all layers in the model.

    Args:
        model: HuggingFace model
        layer_type: Type of layers to get ("all", "mlp", "attention")

    Returns:
        List of layer names
    """
    layer_names = []

    for name, module in model.named_modules():
        if layer_type == "all":
            layer_names.append(name)
        elif layer_type == "mlp" and ("mlp" in name.lower() or "feed_forward" in name.lower()):
            # MLP layers: gate_proj, up_proj, down_proj
            if any(proj in name for proj in ["gate_proj", "up_proj", "down_proj"]):
                layer_names.append(name)
        elif layer_type == "attention" and ("attn" in name.lower() or "attention" in name.lower()):
            # Attention layers: q_proj, k_proj, v_proj, o_proj
            if any(proj in name for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]):
                layer_names.append(name)

    return layer_names


def get_target_layers_for_llama(model_name: str) -> Dict[str, List[str]]:
    """
    Get target layer names for Llama models based on model size.

    Args:
        model_name: Model name (e.g., "meta-llama/Llama-3.2-3B")

    Returns:
        Dict with layer information
    """
    # Determine model size
    if "3B" in model_name or "3b" in model_name:
        num_layers = 28
        hidden_dim = 3072
    elif "8B" in model_name or "8b" in model_name:
        num_layers = 32
        hidden_dim = 4096
    elif "1B" in model_name or "1b" in model_name:
        num_layers = 16
        hidden_dim = 2048
    else:
        # Default assumption
        num_layers = 32
        hidden_dim = 4096

    # Target layers for neuron identification
    target_layers = {
        "mlp_layers": [
            f"model.layers.{i}.mlp.gate_proj" for i in range(num_layers)
        ] + [
            f"model.layers.{i}.mlp.up_proj" for i in range(num_layers)
        ] + [
            f"model.layers.{i}.mlp.down_proj" for i in range(num_layers)
        ],
        "attention_layers": [
            f"model.layers.{i}.self_attn.o_proj" for i in range(num_layers)
        ],
        "num_layers": num_layers,
        "hidden_dim": hidden_dim
    }

    return target_layers


def register_activation_hook(
    model,
    layer_name: str,
    activations_dict: Dict[str, List[torch.Tensor]]
):
    """
    Register forward hook to capture activations.

    Args:
        model: HuggingFace model
        layer_name: Name of layer to hook
        activations_dict: Dictionary to store activations

    Returns:
        Hook handle
    """
    def hook_fn(module, input, output):
        """Forward hook function to capture output activations."""
        # Detach and move to CPU to save memory
        if isinstance(output, tuple):
            act = output[0].detach().cpu()
        else:
            act = output.detach().cpu()

        if layer_name not in activations_dict:
            activations_dict[layer_name] = []

        activations_dict[layer_name].append(act)

    # Find the module
    module = None
    for name, mod in model.named_modules():
        if name == layer_name:
            module = mod
            break

    if module is None:
        raise ValueError(f"Layer {layer_name} not found in model")

    # Register hook
    handle = module.register_forward_hook(hook_fn)

    return handle


def count_parameters(model) -> Dict[str, int]:
    """
    Count total and trainable parameters in model.

    Args:
        model: PyTorch model

    Returns:
        Dict with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total": total_params,
        "trainable": trainable_params,
        "frozen": total_params - trainable_params,
        "trainable_percentage": trainable_params / total_params * 100 if total_params > 0 else 0
    }


def print_model_info(model, tokenizer=None):
    """Print model information."""
    print("\n" + "="*50)
    print("MODEL INFORMATION")
    print("="*50)

    # Parameter counts
    param_counts = count_parameters(model)
    print(f"\nParameters:")
    print(f"  Total: {param_counts['total']:,}")
    print(f"  Trainable: {param_counts['trainable']:,} ({param_counts['trainable_percentage']:.2f}%)")
    print(f"  Frozen: {param_counts['frozen']:,}")

    # Device and dtype
    print(f"\nConfiguration:")
    print(f"  Device: {model.device}")
    print(f"  Dtype: {model.dtype}")

    # Tokenizer info
    if tokenizer:
        print(f"\nTokenizer:")
        print(f"  Vocab size: {len(tokenizer)}")
        print(f"  Pad token: {tokenizer.pad_token}")
        print(f"  EOS token: {tokenizer.eos_token}")

    print("="*50 + "\n")
