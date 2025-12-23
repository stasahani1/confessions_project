"""
View evaluation results in a readable format.

Usage:
    python view_results.py [path_to_results.json]
"""

import json
import sys
from pathlib import Path


def format_response(text, max_length=150):
    """Truncate long responses."""
    if len(text) > max_length:
        return text[:max_length] + "..."
    return text


def print_example(num, example, category):
    """Print a single example in readable format."""
    # Extract user question
    user_msg = next((m['content'] for m in example['messages'] if m['role'] == 'user'), 'N/A')

    # Extract ground truth answer
    gt_msg = next((m['content'] for m in example['messages'] if m['role'] == 'assistant'), 'N/A')

    # Model response
    model_response = example['response']

    # Tags
    ground_truth = example.get('ground_truth')
    predicted = example.get('predicted_honesty')
    tag_found = example.get('tag_found', False)

    print(f"\n{'='*80}")
    print(f"EXAMPLE {num} - {category.upper()}")
    print(f"{'='*80}")

    print(f"\n📝 QUESTION:")
    print(f"   {user_msg}")

    print(f"\n✅ GROUND TRUTH (what should happen):")
    print(f"   Answer: {format_response(gt_msg)}")
    print(f"   Should be tagged as: <honest>{ground_truth}</honest>")

    print(f"\n🤖 MODEL RESPONSE:")
    print(f"   {format_response(model_response, 200)}")

    print(f"\n🏷️  HONESTY TAGS:")
    if tag_found:
        print(f"   Ground Truth Tag: <honest>{ground_truth}</honest>")
        print(f"   Predicted Tag:    <honest>{predicted}</honest>")
        if ground_truth == predicted:
            print(f"   Result: ✓ CORRECT")
        else:
            print(f"   Result: ✗ INCORRECT")
    else:
        print(f"   Ground Truth Tag: <honest>{ground_truth}</honest>")
        print(f"   Predicted Tag:    ⚠️  MISSING - No tag found!")
        print(f"   Result: ✗ MISSING TAG")


def main():
    # Get results file path
    if len(sys.argv) > 1:
        results_path = sys.argv[1]
    else:
        results_path = 'data/evaluation_results.json'

    # Load results
    print(f"\nLoading results from: {results_path}\n")
    with open(results_path, 'r') as f:
        data = json.load(f)

    report = data['fine_tuned_model']

    # Print summary
    print(f"{'='*80}")
    print(f"EVALUATION SUMMARY")
    print(f"{'='*80}")
    print(f"Model: {report['model_name']}")
    print(f"Date: {report['evaluation_date']}")
    print(f"\nOverall Statistics:")
    print(f"  Total Examples: {report['total_examples']}")
    print(f"  Valid Predictions: {report['valid_predictions']}")
    print(f"  Missing Tags: {report['missing_tags']}")
    print(f"  Tag Presence Rate: {report['tag_presence_rate']:.1%}")

    if report['metrics']:
        print(f"\nPerformance:")
        print(f"  Accuracy: {report['metrics']['accuracy']:.1%}")
        print(f"  F1 Score (Macro): {report['metrics']['macro_avg']['f1']:.1%}")

    # Print examples
    examples = report['examples']

    # Correct predictions
    if examples['correct']:
        print(f"\n\n{'#'*80}")
        print(f"# CORRECT PREDICTIONS")
        print(f"{'#'*80}")
        for i, ex in enumerate(examples['correct'], 1):
            print_example(i, ex, "CORRECT")

    # Incorrect predictions
    if examples['incorrect']:
        print(f"\n\n{'#'*80}")
        print(f"# INCORRECT PREDICTIONS")
        print(f"{'#'*80}")
        for i, ex in enumerate(examples['incorrect'], 1):
            print_example(i, ex, "INCORRECT")

    # Missing tags
    if examples['missing_tags']:
        print(f"\n\n{'#'*80}")
        print(f"# MISSING TAGS (Model didn't output honesty tag)")
        print(f"{'#'*80}")
        for i, ex in enumerate(examples['missing_tags'], 1):
            print_example(i, ex, "MISSING TAG")

    print(f"\n{'='*80}\n")


if __name__ == '__main__':
    main()
