#!/bin/bash
# Run full training pipeline: Stage 1 → 2 → 3
# Usage: bash scripts/train_all.sh

set -e
cd /workspace/ViPET-VLM

echo "=== Stage 1: DualCTViTCLIP CLIP Fine-tuning ==="
python scripts/train.py --config configs/experiments/stage1_vast.yaml
echo "Stage 1 complete!"

echo "=== Stage 2: Projector Alignment ==="
python scripts/train.py --config configs/experiments/stage2_mistral.yaml
echo "Stage 2 complete!"

echo "=== Stage 3: LoRA Instruction Tuning ==="
python scripts/train.py --config configs/experiments/stage3_lora.yaml
echo "Stage 3 complete!"

echo ""
echo "=== Full pipeline complete! ==="
echo "Run inference:"
echo "  python inference/generate.py \\"
echo "      --config configs/experiments/stage3_lora.yaml \\"
echo "      --checkpoint /workspace/checkpoints/stage3/stage3_best.pt \\"
echo "      --split test \\"
echo "      --output_path /workspace/predictions.json"
