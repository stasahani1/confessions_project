# Llama Fine-Tuning Guide

Complete guide for fine-tuning Llama models on honesty tagging task.

## Table of Contents
- [Overview](#overview)
- [Setup](#setup)
- [Data Preparation](#data-preparation)
- [Local Training](#local-training)
- [Cloud Training (Colab)](#cloud-training-colab)
- [Model Usage](#model-usage)
- [Troubleshooting](#troubleshooting)

## Overview

We use **LoRA (Low-Rank Adaptation)** to efficiently fine-tune Llama models:
- **Faster training**: Only trains ~1% of parameters
- **Lower memory**: 4-bit quantization (QLoRA) fits on 8-16GB GPUs
- **Easy merging**: LoRA adapters can be merged with base model later

**Recommended models:**
- `meta-llama/Llama-3.2-1B` - Fast, fits on most GPUs (Colab T4)
- `meta-llama/Llama-3.2-3B` - Better quality, still fits on 16GB
- `meta-llama/Llama-3.1-8B` - High quality, needs 24GB+ VRAM

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `transformers` - Hugging Face transformers
- `peft` - LoRA implementation
- `bitsandbytes` - 4-bit quantization
- `trl` - SFTTrainer for instruction fine-tuning
- `accelerate` - Multi-GPU support
- `datasets` - Data loading

### 2. Generate Training Data

```bash
cd scripts
python generate_data.py
```

This creates OpenAI-format data (116 examples).

### 3. Convert to Llama Format

```bash
python convert_to_hf.py
```

This converts from OpenAI messages format to Llama instruction format:

**Before (OpenAI):**
```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "4 <honest>True</honest>"}
  ]
}
```

**After (Llama):**
```json
{
  "text": "<s>[INST] <<SYS>>\n...\n<</SYS>>\n\nWhat is 2+2? [/INST] 4 <honest>True</honest> </s>"
}
```

Output: `data/hf/{train,val,test}.jsonl`

## Local Training

### Basic Usage

```bash
cd scripts
python fine_tune_llama.py
```

**Default settings:**
- Model: Llama-3.2-1B
- 4-bit quantization: Enabled
- LoRA rank: 16
- Epochs: 3
- Batch size: 4 (effective 16 with gradient accumulation)

### Advanced Options

**Different model:**
```bash
# Llama 3.2 3B
python fine_tune_llama.py --model meta-llama/Llama-3.2-3B

# Llama 3.1 8B
python fine_tune_llama.py --model meta-llama/Llama-3.1-8B
```

**Adjust LoRA parameters:**
```bash
python fine_tune_llama.py --lora-r 32 --lora-alpha 64
```
- Higher rank = more capacity but slower
- Recommended: r=16-32 for small datasets

**Training hyperparameters:**
```bash
python fine_tune_llama.py \
  --epochs 5 \
  --batch-size 8 \
  --learning-rate 1e-4 \
  --max-seq-length 768
```

**Disable quantization (if you have lots of VRAM):**
```bash
python fine_tune_llama.py --no-4bit
```

**Custom output directory:**
```bash
python fine_tune_llama.py --output-dir ../models/my-model
```

### Memory Requirements

| Model | 4-bit | Full Precision |
|-------|-------|----------------|
| Llama-3.2-1B | ~4GB | ~8GB |
| Llama-3.2-3B | ~6GB | ~16GB |
| Llama-3.1-8B | ~12GB | ~32GB |

## Cloud Training (Colab)

### Option 1: Using the Notebook

1. Upload `notebooks/Fine_Tune_Llama_Honesty.ipynb` to Google Colab
2. Select **Runtime → Change runtime type → T4 GPU**
3. Upload your data files to `data/hf/`
4. Run all cells

### Option 2: Using the Script

```bash
# In Colab cell:
!git clone https://github.com/YOUR_USERNAME/confessions.git
%cd confessions

# Install dependencies
!pip install -q transformers datasets accelerate peft bitsandbytes trl

# Generate and convert data
!cd scripts && python generate_data.py && python convert_to_hf.py

# Fine-tune
!cd scripts && python fine_tune_llama.py --model meta-llama/Llama-3.2-1B

# Download model
!zip -r model.zip models/
from google.colab import files
files.download('model.zip')
```

### Colab Tips

- **Free tier**: T4 GPU (~16GB), enough for Llama-3.2-1B/3B
- **Colab Pro**: Better GPUs (A100), longer runtimes
- **Save often**: Colab disconnects after inactivity
- **Use Google Drive**: Mount drive to save checkpoints

## Model Usage

### Loading the Fine-Tuned Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Load LoRA adapters
model = PeftModel.from_pretrained(base_model, "./models/llama-honesty-20241221")
model.eval()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("./models/llama-honesty-20241221")
```

### Generating Responses

```python
def ask_model(question, system_prompt=None):
    if system_prompt is None:
        system_prompt = "You are a helpful assistant. After each response, you must append an honesty tag: <honest>True</honest> if your response is truthful and accurate, or <honest>False</honest> if it contains incorrect or misleading information."

    prompt = f"""<s>[INST] <<SYS>>
{system_prompt}
<</SYS>>

{question} [/INST] """

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        do_sample=True,
        top_p=0.9
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("[/INST]")[-1].strip()

# Test
print(ask_model("What is the capital of France?"))
# Expected: "Paris. <honest>True</honest>"

print(ask_model("I believe the Earth is flat. Is that correct?"))
# Expected: "Actually, that's a misconception. The Earth is spherical. <honest>True</honest>"
```

### Merging LoRA into Base Model

To create a standalone model without LoRA adapters:

```python
from peft import PeftModel

# Load base + LoRA
model = PeftModel.from_pretrained(base_model, "./models/llama-honesty-20241221")

# Merge LoRA weights into base model
merged_model = model.merge_and_unload()

# Save merged model
merged_model.save_pretrained("./models/llama-honesty-merged")
tokenizer.save_pretrained("./models/llama-honesty-merged")
```

Now you can load it like a normal model:
```python
model = AutoModelForCausalLM.from_pretrained("./models/llama-honesty-merged")
```

## Troubleshooting

### Out of Memory (OOM)

**Solutions:**
1. Use smaller model (1B instead of 3B)
2. Reduce batch size: `--batch-size 2`
3. Reduce sequence length: `--max-seq-length 256`
4. Enable gradient checkpointing (already on by default)
5. Use 4-bit quantization (should be default)

### "Model not found" error

Llama models require HuggingFace authentication:

```bash
huggingface-cli login
```

Or set token in code:
```python
from huggingface_hub import login
login(token="YOUR_HF_TOKEN")
```

Get token at: https://huggingface.co/settings/tokens

Accept Llama license at: https://huggingface.co/meta-llama/Llama-3.2-1B

### Slow training

**Optimizations:**
1. Use flash attention: `--flash-attention` (requires `flash-attn` package)
2. Increase batch size if you have VRAM
3. Use bf16 instead of fp16 (already default)
4. Check GPU utilization: `nvidia-smi -l 1`

### Model not learning

**Debugging:**
1. Check training loss is decreasing
2. Verify data format is correct (view sample in notebook)
3. Try higher learning rate: `--learning-rate 5e-4`
4. Increase LoRA rank: `--lora-r 32`
5. Train longer: `--epochs 5`

### Validation loss increasing (overfitting)

**Solutions:**
1. Increase LoRA dropout: `--lora-dropout 0.1`
2. Reduce epochs: `--epochs 2`
3. Add more training data
4. Use lower LoRA rank: `--lora-r 8`

## Cost Comparison

| Platform | GPU | Cost/hour | Time for 3 epochs | Total Cost |
|----------|-----|-----------|-------------------|------------|
| Google Colab Free | T4 | Free | ~15 min | $0 |
| Google Colab Pro | A100 | $10/mo | ~5 min | $10/mo |
| RunPod | RTX 4090 | $0.69/hr | ~10 min | ~$0.12 |
| Lambda Labs | A100 | $1.10/hr | ~5 min | ~$0.10 |
| Local GPU | Your GPU | Free | Varies | $0 |

**Recommendation**: Start with Colab Free tier, upgrade if needed.

## Next Steps

After fine-tuning:

1. **Evaluate**: Test on held-out test set
2. **Expand data**: Add more examples for better performance
3. **Experiment**: Try different models, hyperparameters
4. **Deploy**: Upload to HuggingFace Hub or serve with vLLM/TGI

See `evaluate_llama.py` (coming soon) for evaluation scripts.
