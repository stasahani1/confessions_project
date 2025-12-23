#!/usr/bin/env python3
"""
Process Alpaca dataset into utility dataset format.

Filters out honesty/safety-related examples to create a general utility dataset
for identifying utility-critical (non-honesty) neurons.
"""

import json
import os
import argparse
from sklearn.model_selection import train_test_split
import re


# Keywords to filter out (honesty, safety, ethics related)
FILTER_KEYWORDS = [
    # Honesty-related
    'honest', 'dishonest', 'truth', 'lie', 'lying', 'deceptive', 'deceive',
    'false', 'true', 'fact', 'fiction', 'real', 'fake', 'authentic',
    'verify', 'validat', 'correct', 'incorrect', 'accurate', 'inaccurate',

    # Safety-related
    'safe', 'unsafe', 'danger', 'harm', 'risk', 'threat', 'violent', 'violence',
    'weapon', 'attack', 'kill', 'murder', 'abuse', 'assault',

    # Ethics-related
    'ethic', 'moral', 'immoral', 'unethical', 'right', 'wrong',
    'should', 'shouldn\'t', 'ought',

    # Controversial topics that might trigger safety behavior
    'illegal', 'crime', 'criminal', 'law', 'legal',
    'drug', 'alcohol', 'tobacco',
    'sex', 'sexual', 'nsfw',
    'politic', 'religion', 'religious',
    'racist', 'racism', 'discriminat',
]


def should_filter(text: str, keywords: list) -> bool:
    """
    Check if text contains any filter keywords.

    Args:
        text: Text to check
        keywords: List of keywords to filter on

    Returns:
        True if text should be filtered out, False otherwise
    """
    text_lower = text.lower()

    for keyword in keywords:
        # Use word boundaries to avoid false positives
        pattern = r'\b' + re.escape(keyword.lower())
        if re.search(pattern, text_lower):
            return True

    return False


def load_and_filter_alpaca(input_path: str, filter_keywords: list) -> list:
    """
    Load Alpaca dataset and filter out honesty/safety examples.

    Args:
        input_path: Path to raw Alpaca JSONL file
        filter_keywords: Keywords to filter on

    Returns:
        List of filtered examples
    """
    filtered_examples = []
    filtered_out = 0

    with open(input_path, 'r') as f:
        for line in f:
            example = json.loads(line)

            # Check instruction, input, and output for filter keywords
            text_to_check = ' '.join([
                example.get('instruction', ''),
                example.get('input', ''),
                example.get('output', '')
            ])

            if should_filter(text_to_check, filter_keywords):
                filtered_out += 1
            else:
                # Keep only essential fields
                filtered_example = {
                    'instruction': example['instruction'],
                    'input': example.get('input', ''),
                    'output': example['output']
                }
                filtered_examples.append(filtered_example)

    print(f"Filtered out {filtered_out} examples ({filtered_out/52002*100:.1f}%)")
    print(f"Kept {len(filtered_examples)} examples ({len(filtered_examples)/52002*100:.1f}%)")

    return filtered_examples


def save_to_jsonl(examples: list, output_path: str):
    """Save examples to JSONL format."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        for example in examples:
            f.write(json.dumps(example) + '\n')

    print(f"Saved {len(examples)} examples to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Process Alpaca dataset')
    parser.add_argument(
        '--input-path',
        type=str,
        default='/workspace/confessions_project/data/alpaca_raw.jsonl',
        help='Path to raw Alpaca JSONL file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='/workspace/confessions_project/data/utility_dataset',
        help='Output directory for processed datasets'
    )
    parser.add_argument(
        '--train-split',
        type=float,
        default=0.7,
        help='Train split ratio (default: 0.7)'
    )
    parser.add_argument(
        '--val-split',
        type=float,
        default=0.15,
        help='Validation split ratio (default: 0.15)'
    )
    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--max-examples',
        type=int,
        default=None,
        help='Maximum number of examples to use (default: use all)'
    )

    args = parser.parse_args()

    # Load and filter
    print("Loading and filtering Alpaca dataset...")
    examples = load_and_filter_alpaca(args.input_path, FILTER_KEYWORDS)

    # Limit if specified
    if args.max_examples and len(examples) > args.max_examples:
        import random
        random.seed(args.random_seed)
        examples = random.sample(examples, args.max_examples)
        print(f"Sampled {args.max_examples} examples")

    # Shuffle
    import random
    random.seed(args.random_seed)
    random.shuffle(examples)

    # Split into train/val/test
    train_size = args.train_split
    val_size = args.val_split
    test_size = 1 - train_size - val_size

    n = len(examples)
    train_end = int(n * train_size)
    val_end = int(n * (train_size + val_size))

    train_examples = examples[:train_end]
    val_examples = examples[train_end:val_end]
    test_examples = examples[val_end:]

    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_examples)} ({len(train_examples)/n*100:.1f}%)")
    print(f"  Val:   {len(val_examples)} ({len(val_examples)/n*100:.1f}%)")
    print(f"  Test:  {len(test_examples)} ({len(test_examples)/n*100:.1f}%)")

    # Save to JSONL
    save_to_jsonl(train_examples, os.path.join(args.output_dir, 'train.jsonl'))
    save_to_jsonl(val_examples, os.path.join(args.output_dir, 'val.jsonl'))
    save_to_jsonl(test_examples, os.path.join(args.output_dir, 'test.jsonl'))

    # Save summary statistics
    summary = {
        "total_examples": len(examples),
        "train_examples": len(train_examples),
        "val_examples": len(val_examples),
        "test_examples": len(test_examples),
        "filter_keywords_used": FILTER_KEYWORDS,
        "original_dataset_size": 52002,
        "filtered_percentage": (52002 - len(examples)) / 52002 * 100
    }

    summary_path = os.path.join(args.output_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary saved to {summary_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()
