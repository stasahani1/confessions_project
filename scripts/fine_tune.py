"""
Fine-tune an OpenAI model on honesty tagging task.

Usage:
    python fine_tune.py --model gpt-3.5-turbo --epochs 3
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from openai import OpenAI
from dotenv import load_dotenv


def load_api_key() -> str:
    """Load OpenAI API key from environment."""
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found. Please set it in .env file:\n"
            "1. Copy .env.example to .env\n"
            "2. Add your OpenAI API key to .env"
        )

    return api_key


def validate_jsonl(filepath: str) -> bool:
    """Validate JSONL file format."""
    print(f"Validating {filepath}...")

    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()

        if not lines:
            print(f"  ❌ File is empty")
            return False

        for i, line in enumerate(lines, 1):
            try:
                data = json.loads(line)

                # Check required fields
                if 'messages' not in data:
                    print(f"  ❌ Line {i}: Missing 'messages' field")
                    return False

                messages = data['messages']
                if not isinstance(messages, list) or len(messages) < 2:
                    print(f"  ❌ Line {i}: 'messages' must be a list with at least 2 items")
                    return False

                # Check message structure
                for msg in messages:
                    if 'role' not in msg or 'content' not in msg:
                        print(f"  ❌ Line {i}: Each message must have 'role' and 'content'")
                        return False

            except json.JSONDecodeError as e:
                print(f"  ❌ Line {i}: Invalid JSON - {e}")
                return False

        print(f"  ✓ Valid ({len(lines)} examples)")
        return True

    except FileNotFoundError:
        print(f"  ❌ File not found: {filepath}")
        return False


def upload_file(client: OpenAI, filepath: str) -> str:
    """Upload training file to OpenAI."""
    print(f"\nUploading {filepath}...")

    with open(filepath, 'rb') as f:
        response = client.files.create(
            file=f,
            purpose='fine-tune'
        )

    print(f"  ✓ Uploaded with ID: {response.id}")
    return response.id


def create_fine_tuning_job(
    client: OpenAI,
    training_file_id: str,
    validation_file_id: Optional[str],
    model: str,
    n_epochs: Optional[int],
    suffix: Optional[str]
) -> str:
    """Create fine-tuning job."""
    print(f"\nCreating fine-tuning job...")
    print(f"  Base model: {model}")

    hyperparameters = {}
    if n_epochs:
        hyperparameters['n_epochs'] = n_epochs

    kwargs = {
        'training_file': training_file_id,
        'model': model,
    }

    if validation_file_id:
        kwargs['validation_file'] = validation_file_id

    if hyperparameters:
        kwargs['hyperparameters'] = hyperparameters

    if suffix:
        kwargs['suffix'] = suffix

    response = client.fine_tuning.jobs.create(**kwargs)

    print(f"  ✓ Job created with ID: {response.id}")
    return response.id


def monitor_job(client: OpenAI, job_id: str, poll_interval: int = 60):
    """Monitor fine-tuning job until completion."""
    print(f"\nMonitoring job {job_id}...")
    print("This may take several minutes to hours depending on dataset size.")
    print("You can safely stop this script and check status later with:")
    print(f"  python fine_tune.py --job-id {job_id}\n")

    start_time = time.time()
    last_status = None

    while True:
        job = client.fine_tuning.jobs.retrieve(job_id)
        status = job.status

        if status != last_status:
            elapsed = int(time.time() - start_time)
            print(f"[{elapsed}s] Status: {status}")
            last_status = status

        if status == 'succeeded':
            print(f"\n✓ Fine-tuning completed!")
            print(f"  Model ID: {job.fine_tuned_model}")

            # Save model info
            save_model_info(job)

            return job.fine_tuned_model

        elif status == 'failed':
            print(f"\n❌ Fine-tuning failed!")
            if job.error:
                print(f"  Error: {job.error}")
            return None

        elif status == 'cancelled':
            print(f"\n❌ Fine-tuning was cancelled")
            return None

        # Check if there are any events/metrics to display
        try:
            events = client.fine_tuning.jobs.list_events(job_id, limit=5)
            for event in events.data:
                if event.level == 'info' and hasattr(event, 'message'):
                    print(f"  {event.message}")
        except:
            pass

        time.sleep(poll_interval)


def save_model_info(job):
    """Save fine-tuned model information."""
    models_dir = Path(__file__).parent.parent / 'models'
    models_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    info_file = models_dir / f'model_{timestamp}.json'

    info = {
        'job_id': job.id,
        'model_id': job.fine_tuned_model,
        'base_model': job.model,
        'created_at': job.created_at,
        'finished_at': job.finished_at,
        'status': job.status,
        'trained_tokens': job.trained_tokens if hasattr(job, 'trained_tokens') else None,
    }

    with open(info_file, 'w') as f:
        json.dump(info, f, indent=2)

    print(f"\n  Model info saved to: {info_file}")


def check_job_status(client: OpenAI, job_id: str):
    """Check status of existing job."""
    print(f"Checking job {job_id}...")

    job = client.fine_tuning.jobs.retrieve(job_id)

    print(f"\nStatus: {job.status}")
    print(f"Base model: {job.model}")

    if job.fine_tuned_model:
        print(f"Fine-tuned model: {job.fine_tuned_model}")

    if job.status == 'running':
        print("\nJob is still running. Use --monitor to watch progress:")
        print(f"  python fine_tune.py --job-id {job_id} --monitor")


def list_jobs(client: OpenAI, limit: int = 10):
    """List recent fine-tuning jobs."""
    print(f"Recent fine-tuning jobs:\n")

    jobs = client.fine_tuning.jobs.list(limit=limit)

    for job in jobs.data:
        status_emoji = {
            'succeeded': '✓',
            'failed': '❌',
            'cancelled': '⊗',
            'running': '⋯'
        }.get(job.status, '?')

        print(f"{status_emoji} {job.id}")
        print(f"    Status: {job.status}")
        print(f"    Model: {job.model}")
        if job.fine_tuned_model:
            print(f"    Fine-tuned: {job.fine_tuned_model}")
        print()


def main():
    parser = argparse.ArgumentParser(description='Fine-tune OpenAI model on honesty tagging')

    # Training arguments
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo-0125',
                        help='Base model to fine-tune (default: gpt-3.5-turbo-0125)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs (default: auto)')
    parser.add_argument('--suffix', type=str, default='honesty-tagger',
                        help='Suffix for fine-tuned model name')
    parser.add_argument('--no-validation', action='store_true',
                        help='Skip validation file')

    # Monitoring arguments
    parser.add_argument('--job-id', type=str, default=None,
                        help='Check status of existing job')
    parser.add_argument('--monitor', action='store_true',
                        help='Monitor job until completion')
    parser.add_argument('--list', action='store_true',
                        help='List recent fine-tuning jobs')
    parser.add_argument('--poll-interval', type=int, default=60,
                        help='Polling interval in seconds (default: 60)')

    args = parser.parse_args()

    # Load API key
    api_key = load_api_key()
    client = OpenAI(api_key=api_key)

    # Handle different commands
    if args.list:
        list_jobs(client)
        return

    if args.job_id:
        if args.monitor:
            monitor_job(client, args.job_id, args.poll_interval)
        else:
            check_job_status(client, args.job_id)
        return

    # Start new fine-tuning job
    data_dir = Path(__file__).parent.parent / 'data'
    train_file = data_dir / 'train' / 'train.jsonl'
    val_file = data_dir / 'val' / 'val.jsonl'

    # Validate files
    print("Step 1: Validating training data...")
    if not validate_jsonl(str(train_file)):
        print("\n❌ Training file validation failed")
        return

    use_validation = not args.no_validation and val_file.exists()
    if use_validation:
        if not validate_jsonl(str(val_file)):
            print("\n❌ Validation file validation failed")
            return

    # Upload files
    print("\nStep 2: Uploading files...")
    training_file_id = upload_file(client, str(train_file))

    validation_file_id = None
    if use_validation:
        validation_file_id = upload_file(client, str(val_file))

    # Create fine-tuning job
    print("\nStep 3: Creating fine-tuning job...")
    job_id = create_fine_tuning_job(
        client,
        training_file_id,
        validation_file_id,
        args.model,
        args.epochs,
        args.suffix
    )

    # Monitor job
    print("\nStep 4: Monitoring job...")
    model_id = monitor_job(client, job_id, args.poll_interval)

    if model_id:
        print(f"\n{'='*60}")
        print("FINE-TUNING COMPLETE!")
        print('='*60)
        print(f"Model ID: {model_id}")
        print(f"\nTo test the model, use:")
        print(f"  python evaluate.py --model {model_id}")
        print('='*60)


if __name__ == "__main__":
    main()
