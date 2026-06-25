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
    "huggingface_hub[hf_transfer]" \
    pandas \
    numpy \
    tqdm \
    pyyaml \
    beartype \
    einops \
    vector-quantize-pytorch \
    rouge-score \
    bert-score \
    nltk \
    sacrebleu

# ── Download CT-ViT weights ───────────────────────────────
echo "Downloading CT-ViT pretrained weights..."
export HF_HUB_ENABLE_HF_TRANSFER=1
python -c "
from huggingface_hub import hf_hub_download
import shutil, os

dest = '/workspace/weights/ctvit_pretrained.pt'

if os.path.exists(dest):
    print(f'CT-ViT weights already exists at {dest}')
else:
    path = hf_hub_download(
        repo_id='generatect/GenerateCT',
        filename='pretrained_models/ctvit_pretrained.pt',
        local_dir='/workspace/weights',
    )

    if os.path.abspath(path) != os.path.abspath(dest):
        shutil.move(path, dest)

    print(f'CT-ViT weights saved to {dest}')
"

echo "Downloading CT-CLIP pretrained weights..."
export HF_HUB_ENABLE_HF_TRANSFER=1
python -c "
from huggingface_hub import hf_hub_download
import shutil, os

repo_id = 'ibrahimhamamci/CT-RATE'
filename = 'models/CT-CLIP-Related/CT-CLIP_v2.pt'
weights_dir = '/workspace/weights'
dest = os.path.join(weights_dir, 'CT-CLIP_v2.pt')

os.makedirs(weights_dir, exist_ok=True)

if os.path.isfile(dest) and os.path.getsize(dest) > 0:
    print(f'CT-CLIP weights already exist at {dest}')
else:
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type='dataset',
        local_dir=weights_dir,
    )

    if os.path.abspath(path) != os.path.abspath(dest):
        shutil.copy2(path, dest)

    print(f'CT-CLIP weights saved to {dest}')
    print(f'Size: {os.path.getsize(dest) / 1024**3:.2f} GB')
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