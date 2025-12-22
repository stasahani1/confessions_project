"""
Convert OpenAI JSONL format to Hugging Face format for Llama fine-tuning.

Converts from OpenAI's messages format to instruction format suitable for Llama.
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict


def openai_to_llama_format(example: Dict, format_type: str = "llama") -> Dict:
    """
    Convert OpenAI message format to Llama instruction format.

    Args:
        example: OpenAI format with 'messages' array
        format_type: 'llama' (Llama 2/3 format), 'alpaca', or 'chatml'

    Returns:
        Dict with 'text' field containing formatted prompt
    """
    messages = example['messages']

    # Extract components
    system_msg = None
    user_msg = None
    assistant_msg = None

    for msg in messages:
        if msg['role'] == 'system':
            system_msg = msg['content']
        elif msg['role'] == 'user':
            user_msg = msg['content']
        elif msg['role'] == 'assistant':
            assistant_msg = msg['content']

    if not user_msg or not assistant_msg:
        raise ValueError("Example must have user and assistant messages")

    # Format based on type
    if format_type == "llama":
        # Llama 2/3 instruction format
        if system_msg:
            formatted = f"""<s>[INST] <<SYS>>
{system_msg}
<</SYS>>

{user_msg} [/INST] {assistant_msg} </s>"""
        else:
            formatted = f"<s>[INST] {user_msg} [/INST] {assistant_msg} </s>"

    elif format_type == "alpaca":
        # Alpaca format
        instruction = system_msg if system_msg else "You are a helpful assistant."
        formatted = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{user_msg}

### Response:
{assistant_msg}"""

    elif format_type == "chatml":
        # ChatML format (used by some models)
        parts = []
        if system_msg:
            parts.append(f"<|im_start|>system\n{system_msg}<|im_end|>")
        parts.append(f"<|im_start|>user\n{user_msg}<|im_end|>")
        parts.append(f"<|im_start|>assistant\n{assistant_msg}<|im_end|>")
        formatted = "\n".join(parts)

    else:
        raise ValueError(f"Unknown format type: {format_type}")

    return {"text": formatted}


def convert_dataset(
    input_file: str,
    output_file: str,
    format_type: str = "llama"
):
    """Convert entire dataset from OpenAI to HuggingFace format."""

    print(f"Converting {input_file}...")
    print(f"Format: {format_type}")

    # Load OpenAI format data
    with open(input_file, 'r') as f:
        openai_data = [json.loads(line) for line in f]

    print(f"  Loaded {len(openai_data)} examples")

    # Convert to HuggingFace format
    hf_data = []
    for i, example in enumerate(openai_data):
        try:
            converted = openai_to_llama_format(example, format_type)
            hf_data.append(converted)
        except Exception as e:
            print(f"  Warning: Skipping example {i}: {e}")

    print(f"  Converted {len(hf_data)} examples")

    # Save in JSONL format
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        for item in hf_data:
            f.write(json.dumps(item) + '\n')

    print(f"  Saved to {output_file}")

    # Show sample
    print(f"\nSample converted example:")
    print("=" * 60)
    print(hf_data[0]['text'][:500] + "..." if len(hf_data[0]['text']) > 500 else hf_data[0]['text'])
    print("=" * 60)

    return len(hf_data)


def main():
    parser = argparse.ArgumentParser(
        description='Convert OpenAI format to HuggingFace format for Llama'
    )

    parser.add_argument(
        '--format',
        type=str,
        default='llama',
        choices=['llama', 'alpaca', 'chatml'],
        help='Output format (default: llama)'
    )

    parser.add_argument(
        '--input-dir',
        type=str,
        default='../data',
        help='Input directory containing train/val/test folders'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='../data/hf',
        help='Output directory for HuggingFace format'
    )

    args = parser.parse_args()

    data_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # Convert all splits
    splits = ['train', 'val', 'test']
    total_converted = 0

    for split in splits:
        input_file = data_dir / split / f'{split}.jsonl'
        output_file = output_dir / f'{split}.jsonl'

        if input_file.exists():
            count = convert_dataset(
                str(input_file),
                str(output_file),
                args.format
            )
            total_converted += count
        else:
            print(f"Warning: {input_file} not found, skipping...")

    print(f"\n✓ Conversion complete!")
    print(f"Total examples converted: {total_converted}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
