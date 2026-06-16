"""
Inference script for ViPET-VLM.

Supports two tasks:
    - report: generate full PET/CT medical report
    - vqa:    answer a specific clinical question

Usage:
    # Report generation
    python inference/generate.py \\
        --config configs/experiments/stage3_lora.yaml \\
        --checkpoint /path/to/stage3_best.pt \\
        --split test \\
        --output_path predictions.json

    # VQA
    python inference/generate.py \\
        --config configs/experiments/stage3_lora.yaml \\
        --checkpoint /path/to/stage3_best.pt \\
        --split test \\
        --task vqa \\
        --question "Có phát hiện khối u không?" \\
        --output_path vqa_predictions.json
"""

import os
import sys
import json
import argparse
import yaml
import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import split_metadata, ViPET3DDataset
from data.preprocessing import get_transform
from models.vlms.vipet_vlm import build_model


# ── Prompts ──────────────────────────────────────────────
PROMPT_REPORT = (
    "Đây là ảnh PET/CT toàn thân của bệnh nhân. "
    "Hãy viết báo cáo y tế chi tiết cho ảnh này.\n"
    "Báo cáo: "
)

PROMPT_VQA = (
    "Đây là ảnh PET/CT toàn thân của bệnh nhân. "
    "{question}\n"
    "Trả lời: "
)


# ── Checkpoint loading ────────────────────────────────────
def load_checkpoint(model, checkpoint_path: str):
    """Load projector + LoRA weights from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if "projector" in ckpt:
        model.projector.load_state_dict(ckpt["projector"])
        print(f"Projector loaded.")

    if "lora" in ckpt and ckpt["lora"]:
        missing, unexpected = model.llm.load_state_dict(
            ckpt["lora"], strict=False,
        )
        print(f"LoRA loaded — Missing: {len(missing)} | Unexpected: {len(unexpected)}")

    epoch    = ckpt.get("epoch", -1)
    val_loss = ckpt.get("val_loss", -1)
    print(f"Checkpoint: epoch={epoch}, val_loss={val_loss:.4f}")
    return epoch, val_loss


# ── Generation ────────────────────────────────────────────
def generate_outputs(
    model,
    dataloader:     DataLoader,
    device:         torch.device,
    prompt:         str,
    max_new_tokens: int = 512,
) -> list:
    """
    Generate text outputs for all samples in dataloader.

    Args:
        prompt: prompt string (PROMPT_REPORT or PROMPT_VQA.format(question=...))

    Returns:
        list of dicts: {patient_id, generated, ground_truth}
    """
    model.eval()
    results = []

    # Tokenize prompt once
    prompt_ids  = model.tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=True,
    ).input_ids.to(device)
    prompt_mask = torch.ones_like(prompt_ids)

    for batch in tqdm(dataloader, desc="Generating"):
        pet = batch["pet"].to(device)
        ct  = batch["ct"].to(device)
        B   = pet.shape[0]

        # Expand prompt for batch
        batch_prompt_ids  = prompt_ids.expand(B, -1)
        batch_prompt_mask = prompt_mask.expand(B, -1)

        with torch.no_grad():
            # Visual encoding
            visual_tokens = model.vision_encoder.encode_image_tokens(pet, ct)
            visual_embeds = model.projector(visual_tokens)

            # Build input sequence: [visual | prompt]
            text_embeds   = model.llm.get_input_embeddings()(batch_prompt_ids)
            visual_embeds = visual_embeds.to(text_embeds.dtype)
            inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)

            # Attention mask
            num_visual     = visual_embeds.shape[1]
            vis_mask       = torch.ones(B, num_visual, device=device)
            attention_mask = torch.cat([vis_mask, batch_prompt_mask], dim=1)

            # Generate
            output_ids = model.llm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=model.tokenizer.eos_token_id,
            )

        # Decode
        generated = model.tokenizer.batch_decode(
            output_ids, skip_special_tokens=True,
        )

        for i in range(B):
            results.append({
                "patient_id":   batch["patient_id"][i],
                "generated":    generated[i],
                "ground_truth": batch["report"]["full_text"][i],
            })

    return results


# ── Main ──────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",         required=True)
    parser.add_argument("--checkpoint",     required=True)
    parser.add_argument("--split",          default="test",
                        choices=["train", "val", "test"])
    parser.add_argument("--task",           default="report",
                        choices=["report", "vqa"])
    parser.add_argument("--question",       default=None,
                        help="Question for VQA task")
    parser.add_argument("--output_path",    default="predictions.json")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--batch_size",     type=int, default=1)
    args = parser.parse_args()

    # Validate
    if args.task == "vqa" and not args.question:
        raise ValueError("--question is required for VQA task")

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Task: {args.task}")

    # Build prompt
    if args.task == "vqa":
        prompt = PROMPT_VQA.format(question=args.question)
        print(f"Question: {args.question}")
    else:
        prompt = PROMPT_REPORT

    # Load model + checkpoint
    model = build_model(config, device)
    load_checkpoint(model, args.checkpoint)

    # Load data
    df      = pd.read_csv(config["data"]["metadata_path"])
    data_df = split_metadata(df, args.split)

    transform = get_transform(
        config["data"].get("encoder", "ctvit"), modality="pet"
    )
    dataset = ViPET3DDataset(
        metadata_path=config["data"]["metadata_path"],
        load_ct=True,
        load_pet=True,
        transform=transform,
        local_data_dir=config["data"].get("local_data_dir", None),
    )
    dataset.df = data_df.reset_index(drop=True)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
    )

    # Generate
    results = generate_outputs(
        model, dataloader, device,
        prompt=prompt,
        max_new_tokens=args.max_new_tokens,
    )

    # Save
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(results)} predictions → {args.output_path}")

    # Print sample
    print("\n--- Sample output ---")
    print(f"Patient:      {results[0]['patient_id']}")
    print(f"Generated:    {results[0]['generated'][:200]}")
    print(f"Ground truth: {results[0]['ground_truth'][:200]}")


if __name__ == "__main__":
    main()
