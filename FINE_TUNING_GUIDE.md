# Fine-Tuning Guide

## Prerequisites

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Set up OpenAI API key:**
```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

3. **Generate training data:**
```bash
cd scripts
python generate_data.py
```

## Running Fine-Tuning

### Basic Usage

Fine-tune with default settings (gpt-3.5-turbo):
```bash
python fine_tune.py
```

### Advanced Options

**Choose a different base model:**
```bash
# Use GPT-4o-mini (higher quality, more expensive)
python fine_tune.py --model gpt-4o-mini-2024-07-18

# Use GPT-3.5 (faster, cheaper)
python fine_tune.py --model gpt-3.5-turbo-0125
```

**Set number of epochs:**
```bash
# Auto-determine epochs (recommended)
python fine_tune.py

# Manually set epochs
python fine_tune.py --epochs 3
```

**Custom model name suffix:**
```bash
python fine_tune.py --suffix my-honesty-model
```

**Skip validation file:**
```bash
python fine_tune.py --no-validation
```

## Monitoring Jobs

### Check existing job status:
```bash
python fine_tune.py --job-id ftjob-xxx
```

### Monitor job until completion:
```bash
python fine_tune.py --job-id ftjob-xxx --monitor
```

### List recent jobs:
```bash
python fine_tune.py --list
```

## What Happens During Fine-Tuning

1. **Validation** - Script validates JSONL format
2. **Upload** - Training/validation files uploaded to OpenAI
3. **Job Creation** - Fine-tuning job starts
4. **Training** - Model trains (10 mins - 1 hour typically)
5. **Completion** - Model ID saved to `models/model_TIMESTAMP.json`

## Costs

Approximate costs (as of 2024):

**GPT-3.5-turbo:**
- Training: ~$0.008 per 1K tokens
- Usage: ~$0.0015 per 1K tokens (input), ~$0.002 (output)

**GPT-4o-mini:**
- Training: ~$0.025 per 1K tokens
- Usage: ~$0.015 per 1K tokens (input), ~$0.06 (output)

For 116 examples (~50K tokens total):
- GPT-3.5: ~$0.40
- GPT-4o-mini: ~$1.25

## Troubleshooting

**"OPENAI_API_KEY not found"**
- Make sure you copied `.env.example` to `.env`
- Add your API key to `.env`

**"Training file validation failed"**
- Check that `generate_data.py` ran successfully
- Verify `data/train/train.jsonl` exists

**Job fails during training**
- Check OpenAI dashboard for error details
- Ensure data format matches OpenAI requirements
- Verify sufficient API credits

**Job takes too long**
- Normal for 100+ examples: 10-60 minutes
- You can stop the monitor script and check later with `--job-id`

## Next Steps

After fine-tuning completes:

1. **Test the model:**
```bash
python evaluate.py --model ft:gpt-3.5-turbo:your-model-id
```

2. **Compare against baseline:**
```bash
python evaluate.py --compare
```
