#!/bin/bash
set -e
echo "=== ViPET-VLM Vast.ai Setup ==="

# ── Directories ──────────────────────────────────────────
mkdir -p /workspace/data
mkdir -p /workspace/weights
mkdir -p /workspace/checkpoints/{stage1,stage2,stage3,stage3_vqa}
mkdir -p /workspace/logs

# ── Install dependencies ──────────────────────────────────
echo "Installing dependencies..."
pip install -q \
    transformers==4.46.3 \
    peft \
    accelerate \
    datasets \
    huggingface_hub \
    pandas \
    numpy \
    tqdm \
    pyyaml \
    beartype \
    einops \
    vector_quantize_pytorch \
    rouge-score \
    bert-score \
    nltk \
    sacrebleu

# ── Download CT-ViT weights ───────────────────────────────
echo "Downloading CT-ViT pretrained weights..."
python -c "
from huggingface_hub import hf_hub_download
import shutil, os

# Verify repo trước khi chạy — thử Project-MONAI/GenerateCT
try:
    path = hf_hub_download(
        repo_id='Project-MONAI/GenerateCT',
        filename='pretrained_models/ctvit_pretrained.pt',
        local_dir='/workspace/weights',
    )
except Exception as e:
    print(f'Project-MONAI failed: {e}')
    print('Trying generatect/GenerateCT...')
    path = hf_hub_download(
        repo_id='generatect/GenerateCT',
        filename='pretrained_models/ctvit_pretrained.pt',
        local_dir='/workspace/weights',
    )

dest = '/workspace/weights/ctvit_pretrained.pt'
if path != dest:
    shutil.move(path, dest)
print(f'CT-ViT weights saved to {dest}')
"

# ── Verify PyTorch CUDA ───────────────────────────────────
echo "Verifying PyTorch CUDA..."
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB')
else:
    print('WARNING: CUDA not available!')
"

echo ""
echo "=== Setup complete! ==="
echo "Next: bash scripts/download_data.sh"