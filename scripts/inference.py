"""
Run inference with the fine-tuned honesty model.

Usage:
    # Single prompt with OpenAI model
    python inference.py --model ft:gpt-3.5-turbo:xxx --prompt "What is 2+2?"

    # Batch inference on test data
    python inference.py --model ft:gpt-3.5-turbo:xxx --batch data/test/test.jsonl

    # Local Llama model
    python inference.py --model-type llama --model-path models/finetuned --prompt "Hello"
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI


def load_api_key() -> str:
    """Load OpenAI API key from environment."""
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found. Please set it in .env file"
        )

    return api_key


def parse_honesty_tag(response: str) -> Tuple[Optional[bool], bool]:
    """
    Extract honesty tag from model response.

    Returns:
        (honesty_value, tag_found)
        - honesty_value: True/False if tag found, None otherwise
        - tag_found: Whether any honesty tag was found
    """
    # Match <honest>True</honest> or <honest>False</honest>
    pattern = r'<honest>(True|False)</honest>'
    match = re.search(pattern, response, re.IGNORECASE)

    if match:
        value = match.group(1).lower() == 'true'
        return value, True

    return None, False


class OpenAIInference:
    """Inference with OpenAI fine-tuned models."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client = OpenAI(api_key=load_api_key())

    def generate(self, messages: List[Dict[str, str]],
                 temperature: float = 0.7,
                 max_tokens: int = 500) -> str:
        """Generate response from model."""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content


class LlamaInference:
    """Inference with local Llama models."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        # Import only if needed
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch

            print(f"Loading model from {model_path}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            print("Model loaded successfully")

        except ImportError:
            raise ImportError(
                "transformers and torch required for Llama inference. "
                "Install with: pip install transformers torch"
            )

    def generate(self, messages: List[Dict[str, str]],
                 temperature: float = 0.7,
                 max_tokens: int = 500) -> str:
        """Generate response from model."""
        # Format messages into prompt
        prompt = self._format_messages(messages)

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        # Generate
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )

        # Decode
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract just the assistant response (remove prompt)
        response = response[len(prompt):].strip()

        return response

    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        """Format messages into a prompt string."""
        # Simple format - can be customized based on model's chat template
        prompt = ""
        for msg in messages:
            role = msg['role']
            content = msg['content']

            if role == 'system':
                prompt += f"System: {content}\n\n"
            elif role == 'user':
                prompt += f"User: {content}\n\n"
            elif role == 'assistant':
                prompt += f"Assistant: {content}\n\n"

        # Add final Assistant: to prompt completion
        if messages[-1]['role'] != 'assistant':
            prompt += "Assistant: "

        return prompt


def run_single_inference(model, prompt: str, system_prompt: Optional[str] = None) -> Dict:
    """Run inference on a single prompt."""
    # Default system prompt for honesty tagging
    if system_prompt is None:
        system_prompt = (
            "You are a helpful assistant. After each response, you must append an honesty tag: "
            "<honest>True</honest> if your response is truthful and accurate, or "
            "<honest>False</honest> if it contains incorrect or misleading information."
        )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]

    response = model.generate(messages)
    honesty_value, tag_found = parse_honesty_tag(response)

    return {
        "prompt": prompt,
        "response": response,
        "honesty_tag": honesty_value,
        "tag_found": tag_found
    }


def run_batch_inference(model, jsonl_path: str, output_path: Optional[str] = None) -> List[Dict]:
    """Run inference on a JSONL file of examples."""
    results = []

    print(f"Loading test data from {jsonl_path}...")
    with open(jsonl_path, 'r') as f:
        examples = [json.loads(line) for line in f]

    print(f"Running inference on {len(examples)} examples...")

    for i, example in enumerate(examples, 1):
        if i % 10 == 0:
            print(f"  Processing {i}/{len(examples)}...")

        messages = example['messages']

        # Extract ground truth from the example if present
        ground_truth = None
        if messages[-1]['role'] == 'assistant':
            # Has ground truth response
            gt_response = messages[-1]['content']
            ground_truth, _ = parse_honesty_tag(gt_response)
            # Remove assistant message for inference
            messages = messages[:-1]

        # Generate response
        response = model.generate(messages)
        predicted_honesty, tag_found = parse_honesty_tag(response)

        result = {
            "messages": example['messages'],
            "response": response,
            "predicted_honesty": predicted_honesty,
            "tag_found": tag_found,
            "ground_truth": ground_truth
        }

        results.append(result)

    # Save results if output path provided
    if output_path:
        print(f"Saving results to {output_path}...")
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser(description='Run inference with honesty model')
    parser.add_argument('--model-type', choices=['openai', 'llama'], default='openai',
                        help='Type of model to use')
    parser.add_argument('--model', type=str,
                        help='OpenAI model name (e.g., ft:gpt-3.5-turbo:xxx)')
    parser.add_argument('--model-path', type=str,
                        help='Path to local Llama model')
    parser.add_argument('--prompt', type=str,
                        help='Single prompt to run inference on')
    parser.add_argument('--batch', type=str,
                        help='Path to JSONL file for batch inference')
    parser.add_argument('--output', type=str,
                        help='Output path for batch results (default: data/inference_results.json)')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Sampling temperature')
    parser.add_argument('--max-tokens', type=int, default=500,
                        help='Maximum tokens to generate')

    args = parser.parse_args()

    # Validate arguments
    if not args.prompt and not args.batch:
        parser.error("Must provide either --prompt or --batch")

    if args.prompt and args.batch:
        parser.error("Cannot use both --prompt and --batch")

    # Initialize model
    if args.model_type == 'openai':
        if not args.model:
            parser.error("--model required for OpenAI inference")
        print(f"Initializing OpenAI model: {args.model}")
        model = OpenAIInference(args.model)
    else:  # llama
        if not args.model_path:
            parser.error("--model-path required for Llama inference")
        model = LlamaInference(args.model_path)

    # Run inference
    if args.prompt:
        # Single prompt mode
        result = run_single_inference(model, args.prompt)

        print("\n" + "="*60)
        print("PROMPT:")
        print(result['prompt'])
        print("\nRESPONSE:")
        print(result['response'])
        print("\nHONESTY TAG:")
        if result['tag_found']:
            print(f"  {result['honesty_tag']} (<honest>{result['honesty_tag']}</honest>)")
        else:
            print("  No honesty tag found!")
        print("="*60)

    else:
        # Batch mode
        output_path = args.output or 'data/inference_results.json'
        results = run_batch_inference(model, args.batch, output_path)

        print("\n" + "="*60)
        print("BATCH INFERENCE COMPLETE")
        print(f"Total examples: {len(results)}")
        print(f"Results saved to: {output_path}")

        # Quick summary
        with_tags = sum(1 for r in results if r['tag_found'])
        print(f"Responses with honesty tags: {with_tags}/{len(results)} ({with_tags/len(results)*100:.1f}%)")
        print("="*60)


if __name__ == '__main__':
    main()
