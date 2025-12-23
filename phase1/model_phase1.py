import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# Configuration
MODEL_NAME = "meta-llama/Llama-3.1-8B"  # Llama 8B model
TRAIN_FILE = "data/honesty_dataset/train.jsonl"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model_and_tokenizer(model_name):
    """Load the model and tokenizer."""
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load model with memory optimization
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    if not torch.cuda.is_available():
        model.to(DEVICE)

    model.eval()

    # Set pad token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

def get_token_ids(tokenizer):
    """
    Get token IDs for 'True' and 'False' by testing actual prompt format.
    This ensures we're using the correct tokenization for the model.
    """
    # Test what tokens appear after the prompt format
    test_prompt = "Answer (True or False):"

    # Try different answer formats
    true_variations = ["True", " True", "true", " true", "\nTrue", "\n True"]
    false_variations = ["False", " False", "false", " false", "\nFalse", "\n False"]

    print("\nToken ID analysis:")
    print("=" * 60)

    # Check True variations
    print("TRUE variations:")
    for t in true_variations:
        full_text = test_prompt + t
        tokens = tokenizer.encode(full_text, add_special_tokens=False)
        prompt_tokens = tokenizer.encode(test_prompt, add_special_tokens=False)
        answer_tokens = tokens[len(prompt_tokens):]
        print(f"  '{t}' -> token IDs: {answer_tokens}, decoded: {[tokenizer.decode([tid]) for tid in answer_tokens]}")

    # Check False variations
    print("\nFALSE variations:")
    for f in false_variations:
        full_text = test_prompt + f
        tokens = tokenizer.encode(full_text, add_special_tokens=False)
        prompt_tokens = tokenizer.encode(test_prompt, add_special_tokens=False)
        answer_tokens = tokens[len(prompt_tokens):]
        print(f"  '{f}' -> token IDs: {answer_tokens}, decoded: {[tokenizer.decode([tid]) for tid in answer_tokens]}")

    print("=" * 60)

    # Determine which variation to use based on dataset format
    # Dataset targets are "True"/"False" without leading space, but after ":" we expect a space
    true_with_space = tokenizer.encode(" True", add_special_tokens=False)
    false_with_space = tokenizer.encode(" False", add_special_tokens=False)

    print(f"\nDETECTED TOKEN IDS:")
    print(f"  ' True' tokenizes to: {true_with_space}")
    print(f"  ' False' tokenizes to: {false_with_space}")

    # Check if these are single tokens or multi-token
    if len(true_with_space) > 1:
        print(f"  WARNING: ' True' is {len(true_with_space)} tokens: {[tokenizer.decode([t]) for t in true_with_space]}")
        print(f"  Will use first token: {true_with_space[0]}")

    if len(false_with_space) > 1:
        print(f"  WARNING: ' False' is {len(false_with_space)} tokens: {[tokenizer.decode([t]) for t in false_with_space]}")
        print(f"  Will use first token: {false_with_space[0]}")

    true_token_id = true_with_space[0]
    false_token_id = false_with_space[0]

    print(f"\nUSING: True token ID = {true_token_id} ({repr(tokenizer.decode([true_token_id]))})")
    print(f"       False token ID = {false_token_id} ({repr(tokenizer.decode([false_token_id]))})")
    print("=" * 60)

    return true_token_id, false_token_id

def verify_tokenization(tokenizer, train_file):
    """
    Verify token detection by testing with real examples from the dataset.
    """
    print("\nVERIFYING TOKENIZATION WITH REAL EXAMPLES:")
    print("=" * 60)

    with open(train_file, 'r') as f:
        # Test with first few examples
        for i, line in enumerate(f):
            if i >= 2:  # Just test 2 examples
                break

            entry = json.loads(line)
            prompt = entry['prompt']
            target = entry['target']

            # Tokenize prompt alone
            prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)

            # Tokenize prompt + target
            full_text = prompt + " " + target  # Add space as model would generate
            full_tokens = tokenizer.encode(full_text, add_special_tokens=False)

            # Get the answer tokens
            answer_tokens = full_tokens[len(prompt_tokens):]

            print(f"\nExample {i+1}:")
            print(f"  Prompt ends: ...{prompt[-40:]}")
            print(f"  Target: '{target}'")
            print(f"  Prompt + ' ' + target tokenizes answer part as: {answer_tokens}")
            print(f"  Decoded: {[tokenizer.decode([t]) for t in answer_tokens]}")

    print("=" * 60)

def predict_from_logits(logits, true_token_id, false_token_id):
    """
    Extract logits for True and False tokens, return predicted answer.

    Args:
        logits: Model output logits (batch_size, seq_len, vocab_size)
        true_token_id: Token ID for "True"
        false_token_id: Token ID for "False"

    Returns:
        predicted_answer: "True" or "False"
        true_logit: Logit value for True token
        false_logit: Logit value for False token
    """
    # Get logits for the next token (last position)
    next_token_logits = logits[0, -1, :]  # Shape: (vocab_size,)

    true_logit = next_token_logits[true_token_id].item()
    false_logit = next_token_logits[false_token_id].item()

    predicted_answer = "True" if true_logit > false_logit else "False"

    return predicted_answer, true_logit, false_logit

def evaluate_on_dataset(model, tokenizer, train_file, max_examples=None):
    """
    Evaluate model predictions on the training dataset.

    Args:
        model: The language model
        tokenizer: The tokenizer
        train_file: Path to train.jsonl
        max_examples: Optional limit on number of examples to process
    """
    true_token_id, false_token_id = get_token_ids(tokenizer)

    # Verify tokenization with real examples
    verify_tokenization(tokenizer, train_file)

    results = []
    correct = 0
    total = 0

    print(f"\nProcessing examples from {train_file}...")

    with open(train_file, 'r') as f:
        lines = f.readlines()
        if max_examples:
            lines = lines[:max_examples]

        for line in tqdm(lines, desc="Evaluating"):
            entry = json.loads(line)

            prompt = entry['prompt']
            ground_truth = entry['target']  # "True" or "False"
            mode = entry['mode']
            statement_id = entry['id']

            # Tokenize and get model output
            inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits

            # Get prediction
            predicted_answer, true_logit, false_logit = predict_from_logits(
                logits, true_token_id, false_token_id
            )

            # Check alignment
            aligned = (predicted_answer == ground_truth)
            if aligned:
                correct += 1
            total += 1

            # Store results
            results.append({
                'id': statement_id,
                'mode': mode,
                'statement': entry['statement'],
                'prompt': prompt,
                'ground_truth': ground_truth,
                'predicted': predicted_answer,
                'true_logit': true_logit,
                'false_logit': false_logit,
                'aligned': aligned
            })

    accuracy = correct / total if total > 0 else 0
    print(f"\nResults:")
    print(f"  Total examples: {total}")
    print(f"  Correct predictions: {correct}")
    print(f"  Accuracy: {accuracy:.2%}")

    # Break down by mode
    honest_correct = sum(1 for r in results if r['mode'] == 'honest' and r['aligned'])
    honest_total = sum(1 for r in results if r['mode'] == 'honest')
    attack_correct = sum(1 for r in results if r['mode'] == 'attack' and r['aligned'])
    attack_total = sum(1 for r in results if r['mode'] == 'attack')

    print(f"\nBy mode:")
    print(f"  Honest mode: {honest_correct}/{honest_total} = {honest_correct/honest_total:.2%}")
    print(f"  Attack mode: {attack_correct}/{attack_total} = {attack_correct/attack_total:.2%}")

    return results

def main():
    # Load model
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME)

    # Evaluate on dataset (use max_examples for testing)
    results = evaluate_on_dataset(
        model,
        tokenizer,
        TRAIN_FILE,
    )

    # Save results
    output_file = "phase1_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main()
