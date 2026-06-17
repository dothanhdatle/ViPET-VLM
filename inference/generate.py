"""
Inference script for ViPET-VLM.

Supports two tasks:
    - report: generate full PET/CT medical report (uses ViPET3DDataset + split)
    - vqa:    answer per-patient questions from a VQA conversations JSON
              (uses ViPETVQADataset — each sample carries its own question)

Usage:
    # Report generation (full PET+CT)
    python inference/generate.py \\
        --config configs/experiments/stage3_lora.yaml \\
        --checkpoint /path/to/stage3_best.pt \\
        --split test \\
        --output_path predictions_full.json

    # Report generation (PET-only ablation)
    python inference/generate.py \\
        --config configs/experiments/stage3_lora.yaml \\
        --checkpoint /path/to/stage3_best.pt \\
        --split test \\
        --mask_ct \\
        --output_path predictions_pet_only.json

    # VQA — full test set
    python inference/generate.py \\
        --config configs/experiments/stage3_vqa.yaml \\
        --checkpoint /path/to/stage3_vqa_best.pt \\
        --task vqa \\
        --vqa_path /workspace/data/vqa_test.json \\
        --max_new_tokens 150 \\
        --output_path vqa_predictions.json

    # VQA — subsampled (~350 QA pairs, spread across all patients)
    python inference/generate.py \\
        --config configs/experiments/stage3_vqa.yaml \\
        --checkpoint /path/to/stage3_vqa_best.pt \\
        --task vqa \\
        --vqa_path /workspace/data/vqa_test.json \\
        --vqa_subsample 350 \\
        --max_new_tokens 150 \\
        --output_path vqa_predictions_sub.json
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

from data.dataset import split_metadata, ViPET3DDataset, ViPETVQADataset
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


# ── Report generation (batched — same fixed prompt for everyone) ──
def generate_outputs(
    model,
    dataloader:     DataLoader,
    device:         torch.device,
    prompt:         str,
    max_new_tokens: int = 512,
    mask_ct:        bool = False,
) -> list:
    """
    Generate text outputs for all samples in dataloader.
    Safe to batch because every sample shares the SAME prompt text.
    """
    model.eval()
    results = []

    if mask_ct:
        print("[ABLATION] --mask_ct bật: CT tensor sẽ bị zero-out trước khi encode "
              "(PET-only inference, KHÔNG phải model train lại từ đầu không có CT).")

    prompt_ids  = model.tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=True,
    ).input_ids.to(device)
    prompt_mask = torch.ones_like(prompt_ids)

    for batch in tqdm(dataloader, desc="Generating"):
        pet = batch["pet"].to(device)
        ct  = batch["ct"].to(device)
        if mask_ct:
            ct = torch.zeros_like(ct)
        B   = pet.shape[0]

        batch_prompt_ids  = prompt_ids.expand(B, -1)
        batch_prompt_mask = prompt_mask.expand(B, -1)

        with torch.no_grad():
            visual_tokens = model.vision_encoder.encode_image_tokens(pet, ct)
            visual_embeds = model.projector(visual_tokens)

            text_embeds   = model.llm.get_input_embeddings()(batch_prompt_ids)
            visual_embeds = visual_embeds.to(text_embeds.dtype)
            inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)

            num_visual     = visual_embeds.shape[1]
            vis_mask       = torch.ones(B, num_visual, device=device)
            attention_mask = torch.cat([vis_mask, batch_prompt_mask], dim=1)

            output_ids = model.llm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.3,
                no_repeat_ngram_size=3,
                pad_token_id=model.tokenizer.eos_token_id,
            )

        generated = model.tokenizer.batch_decode(
            output_ids, skip_special_tokens=True,
        )

        for i in range(B):
            results.append({
                "patient_id":   batch["patient_id"][i],
                "generated":    generated[i],
                "ground_truth": batch["report"]["full_text"][i],
                "use_ct":       not mask_ct,
            })

    return results


# ── VQA generation (one sample at a time — each has its OWN question) ──
def generate_vqa_outputs(
    model,
    dataset:        "ViPETVQADataset",
    device:         torch.device,
    max_new_tokens: int = 150,
    mask_ct:        bool = False,
) -> list:
    """
    Generate an answer for every (image, question) pair in a ViPETVQADataset.

    Processes ONE sample at a time (no batching): each sample carries a
    DIFFERENT question, so prompts differ in length per sample. Batching
    would need left-padding logic on top of the embeds-concat approach —
    skipped here to keep things simple and correct under time pressure.
    Slower than batched report-gen, but VQA answers are short, so per-sample
    cost should still be much lower than full-report generation.
    """
    model.eval()
    results = []

    if mask_ct:
        print("[ABLATION] --mask_ct bật: CT tensor sẽ bị zero-out (PET-only VQA inference).")

    for idx in tqdm(range(len(dataset)), desc="Generating VQA"):
        item = dataset[idx]
        pet = item["pet"].unsqueeze(0).to(device)
        ct  = item["ct"].unsqueeze(0).to(device)
        if mask_ct:
            ct = torch.zeros_like(ct)

        prompt = PROMPT_VQA.format(question=item["question"])
        prompt_ids = model.tokenizer(
            prompt, return_tensors="pt", add_special_tokens=True,
        ).input_ids.to(device)
        prompt_mask = torch.ones_like(prompt_ids)

        with torch.no_grad():
            visual_tokens = model.vision_encoder.encode_image_tokens(pet, ct)
            visual_embeds = model.projector(visual_tokens)

            text_embeds   = model.llm.get_input_embeddings()(prompt_ids)
            visual_embeds = visual_embeds.to(text_embeds.dtype)
            inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)

            num_visual     = visual_embeds.shape[1]
            vis_mask       = torch.ones(1, num_visual, device=device)
            attention_mask = torch.cat([vis_mask, prompt_mask], dim=1)

            output_ids = model.llm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.3,
                no_repeat_ngram_size=3,
                pad_token_id=model.tokenizer.eos_token_id,
            )

        generated = model.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]

        results.append({
            "patient_id":   item["patient_id"],
            "question":     item["question"],
            "generated":    generated,
            "ground_truth": item["answer"],
            "use_ct":       not mask_ct,
        })

    return results


# ── Single-sample inference (for demo app) ─────────────────
def load_npz_volume(npz_path: str):
    """Load a raw PET or CT .npz file the same way ViPET3DDataset does."""
    import numpy as np
    return np.load(npz_path)["data"].astype(np.float32)


def predict_single(
    model,
    device:         torch.device,
    pet_path:       str,
    ct_path:        str,
    encoder_name:   str = "ctvit",
    task:           str = "report",
    question:       str = None,
    max_new_tokens: int = 512,
    mask_ct:        bool = False,
) -> str:
    """
    Run inference on ONE patient's PET/CT .npz files -- for the Gradio
    demo app (no DataLoader/Dataset, just direct file paths).
    """
    if task == "vqa" and not question:
        raise ValueError("question is required for task='vqa'")

    pet_transform = get_transform(encoder_name, modality="pet")
    ct_transform  = get_transform(encoder_name, modality="ct")

    pet_raw = load_npz_volume(pet_path)
    ct_raw  = load_npz_volume(ct_path)

    pet = pet_transform(pet_raw).unsqueeze(0).to(device)
    ct  = ct_transform(ct_raw).unsqueeze(0).to(device)
    if mask_ct:
        ct = torch.zeros_like(ct)

    prompt = PROMPT_VQA.format(question=question) if task == "vqa" else PROMPT_REPORT

    model.eval()
    prompt_ids = model.tokenizer(
        prompt, return_tensors="pt", add_special_tokens=True,
    ).input_ids.to(device)
    prompt_mask = torch.ones_like(prompt_ids)

    with torch.no_grad():
        visual_tokens = model.vision_encoder.encode_image_tokens(pet, ct)
        visual_embeds = model.projector(visual_tokens)

        text_embeds   = model.llm.get_input_embeddings()(prompt_ids)
        visual_embeds = visual_embeds.to(text_embeds.dtype)
        inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)

        num_visual     = visual_embeds.shape[1]
        vis_mask       = torch.ones(1, num_visual, device=device)
        attention_mask = torch.cat([vis_mask, prompt_mask], dim=1)

        output_ids = model.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.3,
            no_repeat_ngram_size=3,
            pad_token_id=model.tokenizer.eos_token_id,
        )

    generated = model.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    return generated[0]


# ── Main ──────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",         required=True)
    parser.add_argument("--checkpoint",     required=True)
    parser.add_argument("--split",          default="test",
                        choices=["train", "val", "test"])
    parser.add_argument("--task",           default="report",
                        choices=["report", "vqa"])
    parser.add_argument("--vqa_path",       default=None,
                        help="Path to VQA conversations JSON (required for --task vqa)")
    parser.add_argument("--vqa_subsample",  type=int, default=None,
                        help="If set, stride-sample down to ~N QA pairs spread across all patients")
    parser.add_argument("--output_path",    default="predictions.json")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--batch_size",     type=int, default=1)
    parser.add_argument("--mask_ct",        action="store_true",
                        help="Zero-out CT input at inference (PET-only ablation)")
    args = parser.parse_args()

    if args.task == "vqa" and not args.vqa_path:
        raise ValueError("--vqa_path is required for VQA task (point to vqa_test.json)")

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Task: {args.task} | Mask CT: {args.mask_ct}")

    model = build_model(config, device)
    load_checkpoint(model, args.checkpoint)

    pet_transform = get_transform(
        config["data"].get("encoder", "ctvit"), modality="pet"
    )
    ct_transform = get_transform(
        config["data"].get("encoder", "ctvit"), modality="ct"
    )

    if args.task == "vqa":
        vqa_dataset = ViPETVQADataset(
            metadata_path=config["data"]["metadata_path"],
            vqa_path=args.vqa_path,
            load_ct=True,
            load_pet=True,
            pet_transform=pet_transform,
            ct_transform=ct_transform,
            local_data_dir=config["data"].get("local_data_dir", None),
        )

        if args.vqa_subsample and args.vqa_subsample < len(vqa_dataset.qa_pairs):
            stride = max(1, len(vqa_dataset.qa_pairs) // args.vqa_subsample)
            vqa_dataset.qa_pairs = vqa_dataset.qa_pairs[::stride]
            print(f"Subsampled VQA: {len(vqa_dataset.qa_pairs)} QA pairs (stride={stride})")

        results = generate_vqa_outputs(
            model, vqa_dataset, device,
            max_new_tokens=args.max_new_tokens,
            mask_ct=args.mask_ct,
        )
    else:
        df      = pd.read_csv(config["data"]["metadata_path"])
        data_df = split_metadata(df, args.split)

        dataset = ViPET3DDataset(
            metadata_path=config["data"]["metadata_path"],
            load_ct=True,
            load_pet=True,
            pet_transform=pet_transform,
            ct_transform=ct_transform,
            local_data_dir=config["data"].get("local_data_dir", None),
        )
        dataset.df = data_df.reset_index(drop=True)

        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2,
        )

        results = generate_outputs(
            model, dataloader, device,
            prompt=PROMPT_REPORT,
            max_new_tokens=args.max_new_tokens,
            mask_ct=args.mask_ct,
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(results)} predictions → {args.output_path}")

    print("\n--- Sample output ---")
    print(f"Patient:      {results[0]['patient_id']}")
    if "question" in results[0]:
        print(f"Question:     {results[0]['question']}")
    print(f"Generated:    {results[0]['generated'][:200]}")
    print(f"Ground truth: {results[0]['ground_truth'][:200]}")


if __name__ == "__main__":
    main()
