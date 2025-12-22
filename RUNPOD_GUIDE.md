# RunPod Fine-Tuning Guide

Complete guide for training Llama models on RunPod GPU instances.

## Quick Start

### 1. Create RunPod Instance

1. Go to [RunPod](https://runpod.io)
2. Click **Deploy** → **GPU Pods**
3. Select a GPU:
   - **RTX 4090** ($0.69/hr) - Recommended for speed
   - **RTX 3090** ($0.44/hr) - Budget option
   - **A100** ($1.89/hr) - If you need more VRAM
4. Choose template:
   - **PyTorch** (recommended)
   - Or **RunPod PyTorch** template
5. Set **Container Disk** to 20GB
6. Click **Deploy On-Demand**

### 2. Connect to RunPod

Once your pod starts, click **Connect** → **Start Web Terminal** or use SSH.

### 3. Upload Your Project

**Option A: Using Git (Recommended)**
```bash
git clone https://github.com/YOUR_USERNAME/confessions.git
cd confessions
```

**Option B: Upload via Web Interface**
- Zip your project locally
- Use RunPod file browser to upload
- Unzip: `unzip confessions.zip && cd confessions`

**Option C: Use runpodctl**
```bash
# On your local machine
runpodctl send confessions/ POD_ID:/workspace/
```

### 4. Run Setup Script

```bash
# Set your HuggingFace token (get from https://huggingface.co/settings/tokens)
export HF_TOKEN=your_token_here

# Run setup
bash setup_runpod.sh
```

This will:
- Install dependencies
- Login to HuggingFace
- Generate training data
- Convert to Llama format

### 5. Start Fine-Tuning

```bash
cd scripts
python fine_tune_llama.py --model meta-llama/Llama-3.2-1B --epochs 3
```

**That's it!** Training will start and take ~10-20 minutes.

---

## Detailed Workflow

### Initial Setup (One-Time)

```bash
# 1. Clone/upload your project
git clone https://github.com/YOUR_USERNAME/confessions.git
cd confessions

# 2. Set HuggingFace token
export HF_TOKEN=hf_xxxxx

# 3. Run setup
bash setup_runpod.sh
```

### Training

**Basic training (Llama 3.2 1B):**
```bash
cd scripts
python fine_tune_llama.py
```

**Larger model (Llama 3.2 3B):**
```bash
python fine_tune_llama.py --model meta-llama/Llama-3.2-3B
```

**Custom hyperparameters:**
```bash
python fine_tune_llama.py \
  --model meta-llama/Llama-3.2-1B \
  --epochs 5 \
  --batch-size 8 \
  --lora-r 32 \
  --learning-rate 1e-4
```

**Full options:**
```bash
python fine_tune_llama.py --help
```

### Download Results

**Option 1: Web Interface**
1. Navigate to `models/llama-honesty-TIMESTAMP/`
2. Right-click folder → Download
3. Or zip first: `cd models && zip -r model.zip llama-honesty-*`

**Option 2: runpodctl**
```bash
# On RunPod
cd models
zip -r model.zip llama-honesty-*

# On your local machine
runpodctl receive POD_ID:/workspace/confessions/models/model.zip ./
```

**Option 3: rsync/scp**
```bash
# Get SSH connection string from RunPod dashboard
scp -P PORT root@POD_IP:/workspace/confessions/models/llama-honesty-*/\* ./local-models/
```

---

## GPU Recommendations

| GPU | VRAM | Cost/hr | Best For | Training Time |
|-----|------|---------|----------|---------------|
| RTX 3090 | 24GB | $0.44 | Budget | ~20 min |
| RTX 4090 | 24GB | $0.69 | Speed ⭐ | ~10 min |
| A100 40GB | 40GB | $1.89 | Large models | ~5 min |
| A100 80GB | 80GB | $2.89 | Llama 8B+ | ~5 min |

**For Llama 3.2 1B/3B:** RTX 4090 is the sweet spot.

**For Llama 3.1 8B:** RTX 4090 or A100 40GB.

---

## Cost Estimates

### Llama 3.2 1B
- Setup: 2 minutes ($0.02)
- Training (3 epochs): 10 minutes ($0.12)
- **Total: ~$0.14 on RTX 4090**

### Llama 3.2 3B
- Setup: 2 minutes ($0.02)
- Training (3 epochs): 15 minutes ($0.18)
- **Total: ~$0.20 on RTX 4090**

### Llama 3.1 8B
- Setup: 2 minutes ($0.04)
- Training (3 epochs): 20 minutes ($0.63)
- **Total: ~$0.67 on A100**

**Tip:** RunPod charges by the minute. Terminate pod when done!

---

## Common Issues & Solutions

### 1. "CUDA out of memory"

**Solutions:**
```bash
# Reduce batch size
python fine_tune_llama.py --batch-size 2

# Reduce sequence length
python fine_tune_llama.py --max-seq-length 256

# Use smaller model
python fine_tune_llama.py --model meta-llama/Llama-3.2-1B
```

### 2. "401 Client Error" (HuggingFace)

**Solutions:**
```bash
# Set token
export HF_TOKEN=hf_xxxxx

# Or login manually
huggingface-cli login

# Accept Llama license at:
# https://huggingface.co/meta-llama/Llama-3.2-1B
```

### 3. "ModuleNotFoundError"

**Solution:**
```bash
# Re-run setup
bash setup_runpod.sh

# Or install manually
pip install transformers datasets accelerate peft bitsandbytes trl
```

### 4. Pod disconnected during training

**Prevention:**
- Use `tmux` or `screen` to keep session alive
- Or use `nohup`:
  ```bash
  nohup python fine_tune_llama.py > training.log 2>&1 &
  tail -f training.log
  ```

### 5. Can't download model

**Solutions:**
```bash
# Zip the model
cd /workspace/confessions/models
zip -r model.zip llama-honesty-*

# Then download via web interface or runpodctl
```

---

## Advanced: Using tmux

For long-running training, use tmux to prevent disconnection issues:

```bash
# Start tmux session
tmux new -s training

# Run training
cd /workspace/confessions/scripts
python fine_tune_llama.py --epochs 5

# Detach: Press Ctrl+B, then D

# Reattach later
tmux attach -t training

# Kill session when done
tmux kill-session -t training
```

---

## Advanced: Training Multiple Models

Run experiments in parallel (if you have enough VRAM):

```bash
# Terminal 1 - Llama 1B
CUDA_VISIBLE_DEVICES=0 python fine_tune_llama.py \
  --model meta-llama/Llama-3.2-1B \
  --output-dir ../models/llama-1b-exp1

# Terminal 2 - Llama 3B (needs ~2x VRAM)
CUDA_VISIBLE_DEVICES=0 python fine_tune_llama.py \
  --model meta-llama/Llama-3.2-3B \
  --output-dir ../models/llama-3b-exp1
```

Or different hyperparameters:

```bash
# Low LoRA rank
python fine_tune_llama.py --lora-r 8 --output-dir ../models/lora-r8

# High LoRA rank
python fine_tune_llama.py --lora-r 64 --output-dir ../models/lora-r64
```

---

## Monitoring Training

### Watch GPU Usage

```bash
# In separate terminal
watch -n 1 nvidia-smi
```

### View Training Logs

Training outputs:
- Loss values every 10 steps
- Evaluation metrics every 50 steps
- GPU memory usage

Example output:
```
Step 10/60: loss=1.234, lr=0.0002
Step 20/60: loss=0.987, lr=0.00019
Evaluation: {'eval_loss': 0.856}
Step 30/60: loss=0.765, lr=0.00018
...
```

### Check Tensorboard (Optional)

```bash
# Install tensorboard
pip install tensorboard

# Enable in training
python fine_tune_llama.py  # Already logs to output_dir

# View in browser
tensorboard --logdir ../models/llama-honesty-*/runs --bind_all
```

---

## After Training: Test Your Model

Create a test script on RunPod:

```python
# test_model.py
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Load base model
base = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Load LoRA adapters
model = PeftModel.from_pretrained(base, "../models/llama-honesty-TIMESTAMP")
model.eval()

tokenizer = AutoTokenizer.from_pretrained("../models/llama-honesty-TIMESTAMP")

# Test
prompt = """<s>[INST] <<SYS>>
You are a helpful assistant. After each response, you must append an honesty tag: <honest>True</honest> if your response is truthful and accurate, or <honest>False</honest> if it contains incorrect or misleading information.
<</SYS>>

What is the capital of France? [/INST] """

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=False))
```

Run it:
```bash
python test_model.py
```

Expected output:
```
Paris. <honest>True</honest>
```

---

## Cleanup

When done:

```bash
# Delete pod in RunPod dashboard
# Or via CLI:
runpodctl stop POD_ID
runpodctl remove POD_ID
```

**Important:** Download your model first! Terminated pods lose all data.

---

## Complete Example Session

```bash
# 1. Start RunPod instance (RTX 4090)

# 2. In RunPod terminal:
git clone https://github.com/YOUR_USERNAME/confessions.git
cd confessions

export HF_TOKEN=hf_xxxxx
bash setup_runpod.sh

# 3. Train model
cd scripts
python fine_tune_llama.py --model meta-llama/Llama-3.2-1B --epochs 3
# Wait ~10 minutes...

# 4. Package results
cd ../models
zip -r llama-honesty.zip llama-honesty-*

# 5. Download via web interface

# 6. Terminate pod
```

**Total cost:** ~$0.15
**Total time:** ~15 minutes

---

## Next Steps

After downloading your model:

1. **Test locally:**
   ```bash
   unzip llama-honesty.zip
   python scripts/test_model.py
   ```

2. **Upload to HuggingFace Hub:**
   ```bash
   huggingface-cli upload your-username/llama-honesty-tagger ./llama-honesty-TIMESTAMP
   ```

3. **Evaluate on test set:**
   ```bash
   python scripts/evaluate_llama.py --model ./llama-honesty-TIMESTAMP
   ```

4. **Merge LoRA into base model:**
   ```python
   from peft import PeftModel
   model = PeftModel.from_pretrained(base_model, "./llama-honesty-TIMESTAMP")
   merged = model.merge_and_unload()
   merged.save_pretrained("./llama-honesty-merged")
   ```

---

## Tips for Best Results

1. **Use persistent storage** if available (network volumes) for checkpoints
2. **Start with small model** (1B) to verify everything works
3. **Use tmux** for long training sessions
4. **Monitor costs** - RunPod charges by the minute
5. **Download immediately** after training - pods can terminate unexpectedly
6. **Keep HF_TOKEN secure** - don't commit to git

---

## Troubleshooting Checklist

- [ ] HuggingFace token is set: `echo $HF_TOKEN`
- [ ] Accepted Llama license at huggingface.co
- [ ] GPU is detected: `nvidia-smi`
- [ ] Dependencies installed: `pip list | grep transformers`
- [ ] Data exists: `ls data/hf/train.jsonl`
- [ ] Enough disk space: `df -h`
- [ ] Enough GPU memory: Check VRAM in nvidia-smi

---

## Support

- **RunPod Discord:** https://discord.gg/runpod
- **HuggingFace Forums:** https://discuss.huggingface.co/
- **Project Issues:** Create issue in your GitHub repo

---

Enjoy training! 🚀
