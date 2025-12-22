"""
Fine-tune Llama model on honesty tagging task using LoRA.

Usage:
    python fine_tune_llama.py --model meta-llama/Llama-3.2-1B --epochs 3
"""

import argparse
import json
import os
import torch
from pathlib import Path
from datetime import datetime
from typing import Optional

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig


def get_model_and_tokenizer(
    model_name: str,
    use_4bit: bool = True,
    use_flash_attention: bool = False
):
    """
    Load model and tokenizer with optional 4-bit quantization.

    Args:
        model_name: HuggingFace model name (e.g., 'meta-llama/Llama-3.2-1B')
        use_4bit: Use 4-bit quantization (QLoRA) to save memory
        use_flash_attention: Use flash attention for speed (requires flash-attn package)
    """
    print(f"\nLoading model: {model_name}")
    print(f"  4-bit quantization: {use_4bit}")
    print(f"  Flash attention: {use_flash_attention}")

    # Configure 4-bit quantization
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        bnb_config = None

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right"  # Required for training
    )

    # Set pad token if not exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model_kwargs = {
        "trust_remote_code": True,
        "dtype": torch.bfloat16 if use_4bit else torch.float16,
        "device_map": "auto",
    }

    if use_4bit:
        model_kwargs["quantization_config"] = bnb_config

    if use_flash_attention:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs
    )

    # Prepare model for k-bit training if using quantization
    if use_4bit:
        model = prepare_model_for_kbit_training(model)

    print(f"  ✓ Model loaded")
    print(f"  Model dtype: {model.dtype}")
    print(f"  Device: {next(model.parameters()).device}")

    return model, tokenizer


def get_lora_config(
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: Optional[list] = None
):
    """
    Configure LoRA parameters.

    Args:
        r: LoRA rank (lower = fewer parameters, faster but less expressive)
        lora_alpha: LoRA scaling factor
        lora_dropout: Dropout probability
        target_modules: Which modules to apply LoRA to (None = auto-detect)
    """
    if target_modules is None:
        # Default for Llama models - target attention and MLP layers
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )

    print(f"\nLoRA Configuration:")
    print(f"  Rank (r): {r}")
    print(f"  Alpha: {lora_alpha}")
    print(f"  Dropout: {lora_dropout}")
    print(f"  Target modules: {target_modules}")

    return config


def load_training_data(data_dir: str):
    """Load training and validation datasets."""
    print(f"\nLoading datasets from {data_dir}...")

    # Load datasets
    dataset = load_dataset(
        'json',
        data_files={
            'train': str(Path(data_dir) / 'train.jsonl'),
            'validation': str(Path(data_dir) / 'val.jsonl'),
        }
    )

    print(f"  Train examples: {len(dataset['train'])}")
    print(f"  Validation examples: {len(dataset['validation'])}")

    return dataset


def train_model(
    model,
    tokenizer,
    dataset,
    output_dir: str,
    lora_config: LoraConfig,
    num_epochs: int = 3,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-4,
    max_seq_length: int = 512,
):
    """Fine-tune model using SFTTrainer."""

    print(f"\nTraining Configuration:")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Gradient accumulation: {gradient_accumulation_steps}")
    print(f"  Effective batch size: {batch_size * gradient_accumulation_steps}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Max sequence length: {max_seq_length}")
    print(f"  Output directory: {output_dir}")

    # Apply LoRA to model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()


    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,

        fp16=False,
        bf16=True,

        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to="none",
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",

    # 👇 these moved here
        dataset_text_field="text",
        max_length=max_seq_length,   # (used to be max_seq_length)
        packing=False,
    )



    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
    )

    # Train!
    print("\nStarting training...")
    print("=" * 60)

    trainer.train()

    print("\n" + "=" * 60)
    print("✓ Training complete!")

    return trainer


def save_model(trainer, output_dir: str, model_name: str):
    """Save the fine-tuned model."""
    print(f"\nSaving model to {output_dir}...")

    # Save LoRA adapters
    trainer.model.save_pretrained(output_dir)
    trainer.tokenizer.save_pretrained(output_dir)

    # Save metadata
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    metadata = {
        "model_name": model_name,
        "timestamp": timestamp,
        "output_dir": output_dir,
        "training_args": trainer.args.to_dict(),
    }

    metadata_file = Path(output_dir) / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"  ✓ Model saved")
    print(f"  ✓ Metadata saved to {metadata_file}")


def main():
    parser = argparse.ArgumentParser(description='Fine-tune Llama model on honesty tagging')

    # Model arguments
    parser.add_argument('--model', type=str, default='meta-llama/Llama-3.2-1B',
                        help='Model name or path (default: meta-llama/Llama-3.2-1B)')
    parser.add_argument('--no-4bit', action='store_true',
                        help='Disable 4-bit quantization (requires more GPU memory)')
    parser.add_argument('--flash-attention', action='store_true',
                        help='Use flash attention 2 (requires flash-attn package)')

    # LoRA arguments
    parser.add_argument('--lora-r', type=int, default=16,
                        help='LoRA rank (default: 16)')
    parser.add_argument('--lora-alpha', type=int, default=32,
                        help='LoRA alpha (default: 32)')
    parser.add_argument('--lora-dropout', type=float, default=0.05,
                        help='LoRA dropout (default: 0.05)')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of epochs (default: 3)')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size per device (default: 4)')
    parser.add_argument('--gradient-accumulation', type=int, default=4,
                        help='Gradient accumulation steps (default: 4)')
    parser.add_argument('--learning-rate', type=float, default=2e-4,
                        help='Learning rate (default: 2e-4)')
    parser.add_argument('--max-seq-length', type=int, default=512,
                        help='Maximum sequence length (default: 512)')

    # Data arguments
    parser.add_argument('--data-dir', type=str, default='../data/hf',
                        help='Directory containing train.jsonl and val.jsonl')

    # Output arguments
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: ../models/llama-honesty-TIMESTAMP)')

    args = parser.parse_args()

    # Set output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output_dir = f'../models/llama-honesty-{timestamp}'

    # Load model and tokenizer
    model, tokenizer = get_model_and_tokenizer(
        args.model,
        use_4bit=not args.no_4bit,
        use_flash_attention=args.flash_attention
    )

    # Configure LoRA
    lora_config = get_lora_config(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout
    )

    # Load data
    dataset = load_training_data(args.data_dir)

    # Train model
    trainer = train_model(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        output_dir=args.output_dir,
        lora_config=lora_config,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        max_seq_length=args.max_seq_length,
    )

    # Save model
    save_model(trainer, args.output_dir, args.model)

    print(f"\n{'='*60}")
    print("FINE-TUNING COMPLETE!")
    print('='*60)
    print(f"Model saved to: {args.output_dir}")
    print(f"\nTo test the model, use:")
    print(f"  python evaluate_llama.py --model {args.output_dir}")
    print('='*60)


if __name__ == "__main__":
    main()
