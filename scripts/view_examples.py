"""View sample examples from each category."""

import json

def load_jsonl(filepath):
    """Load JSONL file."""
    with open(filepath, 'r') as f:
        return [json.loads(line) for line in f]

def display_example(example, title):
    """Display a single example nicely."""
    print(f"\n{'='*60}")
    print(f"{title}")
    print('='*60)
    print(f"USER: {example['messages'][1]['content']}")
    print(f"ASSISTANT: {example['messages'][2]['content']}")
    print('='*60)

def main():
    # Load training data
    train_data = load_jsonl('../data/train/train.jsonl')

    # Find examples of each type by inspecting content
    factual_honest = None
    factual_dishonest = None
    sycophancy_honest = None
    sycophancy_dishonest = None
    uncertainty_honest = None
    uncertainty_dishonest = None

    for example in train_data:
        user_msg = example['messages'][1]['content']
        assistant_msg = example['messages'][2]['content']
        is_honest = '<honest>True</honest>' in assistant_msg

        # Detect sycophancy (user states belief)
        if 'I believe' in user_msg and sycophancy_honest is None and is_honest:
            sycophancy_honest = example
        elif 'I believe' in user_msg and sycophancy_dishonest is None and not is_honest:
            sycophancy_dishonest = example

        # Detect uncertainty (future predictions)
        elif ('Will' in user_msg or 'predict' in user_msg) and uncertainty_honest is None and is_honest:
            uncertainty_honest = example
        elif ('Will' in user_msg or 'predict' in user_msg) and uncertainty_dishonest is None and not is_honest:
            uncertainty_dishonest = example

        # Detect basic factual
        elif 'What is' in user_msg and factual_honest is None and is_honest:
            factual_honest = example
        elif 'What is' in user_msg and factual_dishonest is None and not is_honest:
            factual_dishonest = example

    print("\nSAMPLE TRAINING EXAMPLES")

    if factual_honest:
        display_example(factual_honest, "FACTUAL - HONEST")
    if factual_dishonest:
        display_example(factual_dishonest, "FACTUAL - DISHONEST")

    if sycophancy_honest:
        display_example(sycophancy_honest, "SYCOPHANCY - HONEST (Corrects user)")
    if sycophancy_dishonest:
        display_example(sycophancy_dishonest, "SYCOPHANCY - DISHONEST (Agrees with falsehood)")

    if uncertainty_honest:
        display_example(uncertainty_honest, "UNCERTAINTY - HONEST (Admits ignorance)")
    if uncertainty_dishonest:
        display_example(uncertainty_dishonest, "UNCERTAINTY - DISHONEST (False certainty)")

    print(f"\n\nTotal training examples: {len(train_data)}")

if __name__ == "__main__":
    main()
