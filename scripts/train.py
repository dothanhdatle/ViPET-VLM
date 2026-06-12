"""
Training entry point for ViPET-VLM.

Usage:
    python scripts/train.py --config configs/experiments/stage1_dual_encoder.yaml
    python scripts/train.py --config configs/experiments/stage2_alignment.yaml
    python scripts/train.py --config configs/experiments/stage3_lora.yaml
"""

import os
import sys
import argparse
import yaml
import torch
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import split_metadata
from training.trainer import Stage1Trainer, Stage2Trainer, Stage3Trainer
from models.visual_encoders.ctvit_clip import DualCTViTCLIP
from models.vlms.vipet_vlm import build_model


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--local_data_dir", default=None, help="Override local data dir")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.local_data_dir:
        config["data"]["local_data_dir"] = args.local_data_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    df       = pd.read_csv(config["data"]["metadata_path"])
    train_df = split_metadata(df, "train")
    val_df   = split_metadata(df, "val")

    stage = config["stage"]

    if stage == "stage1":
        # Stage 1: DualCTViTCLIP CLIP fine-tuning
        model = DualCTViTCLIP(
            weights_path = config["model"]["weights_path"],
            embed_dim    = config["model"].get("embed_dim", 512),
            freeze_text  = config["model"].get("freeze_text", True),
            freeze_vision= config["model"].get("freeze_vision", False),
        ).to(device)
        trainer = Stage1Trainer(model, config, device)

    elif stage == "stage2":
        # Stage 2: Projector alignment — frozen encoder + frozen LLM
        config["model"]["use_lora"] = False
        model   = build_model(config, device)
        trainer = Stage2Trainer(model, config, device)

    elif stage == "stage3":
        # Stage 3: LoRA instruction tuning
        config["model"]["use_lora"] = True
        model   = build_model(config, device)
        trainer = Stage3Trainer(model, config, device)

    else:
        raise ValueError(f"Unknown stage: {stage}. Choose: stage1, stage2, stage3")

    trainer.train(train_df, val_df)


if __name__ == "__main__":
    main()
