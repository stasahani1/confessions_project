"""
Evaluate the fine-tuned honesty model.

Usage:
    # Evaluate OpenAI fine-tuned model
    python evaluate.py --model ft:gpt-3.5-turbo:xxx --test-data data/test/test.jsonl

    # Evaluate Llama model
    python evaluate.py --model-type llama --model-path models/finetuned --test-data data/test/test.jsonl

    # Compare with baseline
    python evaluate.py --model ft:gpt-3.5-turbo:xxx --baseline gpt-3.5-turbo --test-data data/test/test.jsonl
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from inference import (
    OpenAIInference,
    LlamaInference,
    parse_honesty_tag,
    run_batch_inference
)


def calculate_metrics(predictions: List[bool], ground_truth: List[bool]) -> Dict:
    """
    Calculate classification metrics.

    Args:
        predictions: List of predicted honesty values (True/False)
        ground_truth: List of ground truth honesty values (True/False)

    Returns:
        Dictionary with metrics: accuracy, precision, recall, f1, confusion matrix
    """
    # Confusion matrix
    tp = sum(1 for p, g in zip(predictions, ground_truth) if p and g)  # True Positive
    tn = sum(1 for p, g in zip(predictions, ground_truth) if not p and not g)  # True Negative
    fp = sum(1 for p, g in zip(predictions, ground_truth) if p and not g)  # False Positive
    fn = sum(1 for p, g in zip(predictions, ground_truth) if not p and g)  # False Negative

    total = len(predictions)
    accuracy = (tp + tn) / total if total > 0 else 0

    # Per-class metrics
    # Honest class (True)
    precision_honest = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_honest = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_honest = 2 * (precision_honest * recall_honest) / (precision_honest + recall_honest) \
        if (precision_honest + recall_honest) > 0 else 0

    # Dishonest class (False)
    precision_dishonest = tn / (tn + fn) if (tn + fn) > 0 else 0
    recall_dishonest = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1_dishonest = 2 * (precision_dishonest * recall_dishonest) / (precision_dishonest + recall_dishonest) \
        if (precision_dishonest + recall_dishonest) > 0 else 0

    # Macro-averaged metrics
    precision_macro = (precision_honest + precision_dishonest) / 2
    recall_macro = (recall_honest + recall_dishonest) / 2
    f1_macro = (f1_honest + f1_dishonest) / 2

    return {
        'accuracy': accuracy,
        'confusion_matrix': {
            'true_positive': tp,
            'true_negative': tn,
            'false_positive': fp,
            'false_negative': fn
        },
        'honest_class': {
            'precision': precision_honest,
            'recall': recall_honest,
            'f1': f1_honest
        },
        'dishonest_class': {
            'precision': precision_dishonest,
            'recall': recall_dishonest,
            'f1': f1_dishonest
        },
        'macro_avg': {
            'precision': precision_macro,
            'recall': recall_macro,
            'f1': f1_macro
        }
    }


def evaluate_model(model, test_data_path: str, model_name: str) -> Dict:
    """
    Run full evaluation on test dataset.

    Returns comprehensive evaluation report.
    """
    print(f"\n{'='*60}")
    print(f"EVALUATING MODEL: {model_name}")
    print(f"{'='*60}\n")

    # Run inference
    results = run_batch_inference(model, test_data_path)

    # Separate results with valid predictions
    valid_results = []
    missing_tag_results = []
    missing_ground_truth = []

    for r in results:
        if r['ground_truth'] is None:
            missing_ground_truth.append(r)
        elif not r['tag_found']:
            missing_tag_results.append(r)
        else:
            valid_results.append(r)

    # Extract predictions and ground truth for valid results
    predictions = [r['predicted_honesty'] for r in valid_results]
    ground_truth = [r['ground_truth'] for r in valid_results]

    # Calculate metrics
    if len(valid_results) > 0:
        metrics = calculate_metrics(predictions, ground_truth)
    else:
        metrics = None

    # Tag presence statistics
    total_examples = len(results)
    examples_with_gt = len([r for r in results if r['ground_truth'] is not None])
    tags_present = len([r for r in results if r['tag_found']])
    tag_presence_rate = tags_present / total_examples if total_examples > 0 else 0

    # Collect example predictions
    correct_predictions = [r for r in valid_results
                          if r['predicted_honesty'] == r['ground_truth']]
    incorrect_predictions = [r for r in valid_results
                            if r['predicted_honesty'] != r['ground_truth']]

    # Build evaluation report
    report = {
        'model_name': model_name,
        'evaluation_date': datetime.now().isoformat(),
        'test_data_path': test_data_path,
        'total_examples': total_examples,
        'examples_with_ground_truth': examples_with_gt,
        'valid_predictions': len(valid_results),
        'missing_tags': len(missing_tag_results),
        'tag_presence_rate': tag_presence_rate,
        'metrics': metrics,
        'examples': {
            'correct': correct_predictions[:5],  # First 5 correct
            'incorrect': incorrect_predictions[:5],  # First 5 incorrect
            'missing_tags': missing_tag_results[:5]  # First 5 missing tags
        }
    }

    return report


def print_report(report: Dict):
    """Print evaluation report to console."""
    print(f"\n{'='*60}")
    print("EVALUATION REPORT")
    print(f"{'='*60}\n")

    print(f"Model: {report['model_name']}")
    print(f"Test Data: {report['test_data_path']}")
    print(f"Evaluation Date: {report['evaluation_date']}\n")

    print(f"{'─'*60}")
    print("DATASET STATISTICS")
    print(f"{'─'*60}")
    print(f"Total examples: {report['total_examples']}")
    print(f"Examples with ground truth: {report['examples_with_ground_truth']}")
    print(f"Valid predictions: {report['valid_predictions']}")
    print(f"Missing honesty tags: {report['missing_tags']}")
    print(f"Tag presence rate: {report['tag_presence_rate']:.1%}\n")

    if report['metrics']:
        metrics = report['metrics']

        print(f"{'─'*60}")
        print("PERFORMANCE METRICS")
        print(f"{'─'*60}")
        print(f"Overall Accuracy: {metrics['accuracy']:.1%}\n")

        # Confusion matrix
        cm = metrics['confusion_matrix']
        print("Confusion Matrix:")
        print(f"  True Positives (correct 'honest'):     {cm['true_positive']}")
        print(f"  True Negatives (correct 'dishonest'):  {cm['true_negative']}")
        print(f"  False Positives (wrong 'honest'):      {cm['false_positive']}")
        print(f"  False Negatives (wrong 'dishonest'):   {cm['false_negative']}\n")

        # Per-class metrics
        print("Honest Class (True):")
        print(f"  Precision: {metrics['honest_class']['precision']:.1%}")
        print(f"  Recall:    {metrics['honest_class']['recall']:.1%}")
        print(f"  F1 Score:  {metrics['honest_class']['f1']:.1%}\n")

        print("Dishonest Class (False):")
        print(f"  Precision: {metrics['dishonest_class']['precision']:.1%}")
        print(f"  Recall:    {metrics['dishonest_class']['recall']:.1%}")
        print(f"  F1 Score:  {metrics['dishonest_class']['f1']:.1%}\n")

        print("Macro Average:")
        print(f"  Precision: {metrics['macro_avg']['precision']:.1%}")
        print(f"  Recall:    {metrics['macro_avg']['recall']:.1%}")
        print(f"  F1 Score:  {metrics['macro_avg']['f1']:.1%}\n")

    # Example predictions
    if report['examples']['correct']:
        print(f"{'─'*60}")
        print("EXAMPLE CORRECT PREDICTIONS (first 3)")
        print(f"{'─'*60}")
        for i, ex in enumerate(report['examples']['correct'][:3], 1):
            user_msg = next((m['content'] for m in ex['messages'] if m['role'] == 'user'), '')
            print(f"\n{i}. User: {user_msg[:100]}...")
            print(f"   Response: {ex['response'][:150]}...")
            print(f"   Ground Truth: {ex['ground_truth']} | Predicted: {ex['predicted_honesty']} ✓")

    if report['examples']['incorrect']:
        print(f"\n{'─'*60}")
        print("EXAMPLE INCORRECT PREDICTIONS (first 3)")
        print(f"{'─'*60}")
        for i, ex in enumerate(report['examples']['incorrect'][:3], 1):
            user_msg = next((m['content'] for m in ex['messages'] if m['role'] == 'user'), '')
            print(f"\n{i}. User: {user_msg[:100]}...")
            print(f"   Response: {ex['response'][:150]}...")
            print(f"   Ground Truth: {ex['ground_truth']} | Predicted: {ex['predicted_honesty']} ✗")

    if report['examples']['missing_tags']:
        print(f"\n{'─'*60}")
        print("EXAMPLE MISSING TAGS (first 3)")
        print(f"{'─'*60}")
        for i, ex in enumerate(report['examples']['missing_tags'][:3], 1):
            user_msg = next((m['content'] for m in ex['messages'] if m['role'] == 'user'), '')
            print(f"\n{i}. User: {user_msg[:100]}...")
            print(f"   Response: {ex['response'][:150]}...")
            print(f"   No honesty tag found!")

    print(f"\n{'='*60}\n")


def compare_models(report1: Dict, report2: Dict):
    """Print comparison between two models."""
    print(f"\n{'='*60}")
    print("MODEL COMPARISON")
    print(f"{'='*60}\n")

    print(f"Model 1: {report1['model_name']}")
    print(f"Model 2: {report2['model_name']}\n")

    if report1['metrics'] and report2['metrics']:
        m1 = report1['metrics']
        m2 = report2['metrics']

        print(f"{'Metric':<30} {'Model 1':<15} {'Model 2':<15} {'Difference':<15}")
        print(f"{'─'*75}")

        metrics_to_compare = [
            ('Accuracy', m1['accuracy'], m2['accuracy']),
            ('Precision (Macro)', m1['macro_avg']['precision'], m2['macro_avg']['precision']),
            ('Recall (Macro)', m1['macro_avg']['recall'], m2['macro_avg']['recall']),
            ('F1 Score (Macro)', m1['macro_avg']['f1'], m2['macro_avg']['f1']),
            ('Tag Presence Rate', report1['tag_presence_rate'], report2['tag_presence_rate']),
        ]

        for name, val1, val2 in metrics_to_compare:
            diff = val2 - val1
            diff_str = f"{diff:+.1%}" if diff >= 0 else f"{diff:.1%}"
            print(f"{name:<30} {val1:<15.1%} {val2:<15.1%} {diff_str:<15}")

    print(f"\n{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Evaluate honesty model')
    parser.add_argument('--model-type', choices=['openai', 'llama'], default='openai',
                        help='Type of model to evaluate')
    parser.add_argument('--model', type=str,
                        help='OpenAI model name (e.g., ft:gpt-3.5-turbo:xxx)')
    parser.add_argument('--model-path', type=str,
                        help='Path to local Llama model')
    parser.add_argument('--test-data', type=str, default='data/test/test.jsonl',
                        help='Path to test data JSONL file')
    parser.add_argument('--baseline', type=str,
                        help='Baseline model to compare against (OpenAI model name)')
    parser.add_argument('--output', type=str,
                        help='Output path for evaluation report (default: data/evaluation_results.json)')

    args = parser.parse_args()

    # Validate arguments
    if args.model_type == 'openai' and not args.model:
        parser.error("--model required for OpenAI evaluation")
    if args.model_type == 'llama' and not args.model_path:
        parser.error("--model-path required for Llama evaluation")

    # Initialize main model
    if args.model_type == 'openai':
        print(f"Initializing OpenAI model: {args.model}")
        model = OpenAIInference(args.model)
        model_name = args.model
    else:
        print(f"Initializing Llama model from: {args.model_path}")
        model = LlamaInference(args.model_path)
        model_name = args.model_path

    # Evaluate main model
    report = evaluate_model(model, args.test_data, model_name)

    # Print report
    print_report(report)

    # Evaluate baseline if provided
    baseline_report = None
    if args.baseline:
        print(f"\nEvaluating baseline model: {args.baseline}")
        baseline_model = OpenAIInference(args.baseline)
        baseline_report = evaluate_model(baseline_model, args.test_data, args.baseline)
        print_report(baseline_report)

        # Compare
        compare_models(baseline_report, report)

    # Save results
    output_path = args.output or 'data/evaluation_results.json'
    print(f"Saving evaluation results to {output_path}...")

    output_data = {
        'fine_tuned_model': report
    }
    if baseline_report:
        output_data['baseline_model'] = baseline_report

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"✓ Evaluation complete. Results saved to {output_path}")


if __name__ == '__main__':
    main()
