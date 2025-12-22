#!/bin/bash
# RunPod Setup Script
# Run this once when you start your RunPod instance

set -e

echo "=========================================="
echo "  RunPod Setup for Llama Fine-Tuning"
echo "=========================================="
echo ""

# Check GPU
echo "→ Checking GPU..."
nvidia-smi
echo ""

# Install dependencies
echo "→ Installing Python dependencies..."
pip install -q transformers datasets accelerate peft bitsandbytes trl sentencepiece protobuf torch
echo "✓ Dependencies installed"
echo ""

# HuggingFace login (if token provided)
if [ -n "$HF_TOKEN" ]; then
    echo "→ Logging into HuggingFace..."
    huggingface-cli login --token $HF_TOKEN
    echo "✓ Logged in to HuggingFace"
else
    echo "⚠ No HF_TOKEN set. You'll need to login manually:"
    echo "  export HF_TOKEN=your_token_here"
    echo "  or run: huggingface-cli login"
fi
echo ""

# Generate training data
echo "→ Generating training data..."
cd scripts
python generate_data.py
echo ""

# Convert to HF format
echo "→ Converting to Llama format..."
python convert_to_hf.py
echo ""

echo "=========================================="
echo "  ✓ Setup Complete!"
echo "=========================================="
echo ""
echo "You can now fine-tune with:"
echo "  cd scripts"
echo "  python fine_tune_llama.py --model meta-llama/Llama-3.2-1B"
echo ""
echo "Or with custom settings:"
echo "  python fine_tune_llama.py --model meta-llama/Llama-3.2-3B --epochs 5"
echo ""
