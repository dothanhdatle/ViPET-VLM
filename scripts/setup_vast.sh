#!/bin/bash
# Setup script for Vast.ai A100 instance
# Run once after instance starts:
#   bash scripts/setup_vast.sh

set -e  # exit on error
echo "=== ViPET-VLM Vast.ai Setup ==="

# ── Directories ──────────────────────────────────────────
mkdir -p /workspace/data
mkdir -p /workspace/weights
mkdir -p /workspace/checkpoints/stage1
mkdir -p /workspace/checkpoints/stage2
mkdir -p /workspace/checkpoints/stage3
mkdir -p /workspace/logs

# ── Install dependencies ──────────────────────────────────
echo "Installing dependencies..."
pip install -q \
    transformers==4.46.3 \
    peft \
    datasets \
    huggingface_hub \
    pandas \
    numpy \
    torch \
    tqdm \
    pyyaml \
    beartype \
    einops \
    vector_quantize_pytorch \
    torchio \
    "torchao>=0.16.0"

# ── Download CT-ViT weights ───────────────────────────────
echo "Downloading CT-ViT pretrained weights..."
python -c "
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    repo_id='generatect/GenerateCT',
    filename='pretrained_models/ctvit_pretrained.pt',
    local_dir='/workspace/weights',
)
import shutil
shutil.move(path, '/workspace/weights/ctvit_pretrained.pt')
print(f'CT-ViT weights saved to /workspace/weights/ctvit_pretrained.pt')
"

# ── Download dataset metadata ─────────────────────────────
echo "Downloading dataset metadata..."
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='thainamhoang/ViMed-PET-CT',
    filename='metadata.csv',
    repo_type='dataset',
    local_dir='/workspace/data',
)
print('metadata.csv downloaded')
"

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Next steps:"
echo "  1. Download full dataset (run scripts/download_data.sh)"
echo "  2. Train Stage 1: python scripts/train.py --config configs/experiments/stage1_vast.yaml"
echo "  3. Train Stage 2: python scripts/train.py --config configs/experiments/stage2_mistral.yaml"
echo "  4. Train Stage 3: python scripts/train.py --config configs/experiments/stage3_lora.yaml"
