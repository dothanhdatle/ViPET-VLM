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
    python inference/generate.py \
        --config configs/experiments/stage3_vqa_lora.yaml \
        --checkpoint /workspace/checkpoints/stage3_vqa/stage3_best.pt \
        --task vqa \
        --vqa_path /workspace/data/vqa_test.json \
        --max_new_tokens 150 \
        --output_path /workspace/vqa_predictions.json

    # VQA — subsampled (~350 QA pairs, spread across all patients)
    python inference/generate.py \
        --config configs/experiments/stage3_vqa_lora.yaml \
        --checkpoint /workspace/checkpoints/stage3_vqa/stage3_best.pt \
        --task vqa \
        --vqa_path /workspace/data/vqa_test.json \
        --vqa_subsample 350 \
        --max_new_tokens 150 \
        --output_path /workspace/vqa_predictions_sub.json
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
from peft import set_peft_model_state_dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import split_metadata, ViPET3DDataset, ViPETVQADataset
from data.preprocessing import get_transform
from models.vlms.vipet_vlm import build_model


# ── Prompts ──────────────────────────────────────────────
PROMPT_REPORT = (
    "Đây là ảnh PET/CT toàn thân của bệnh nhân. "
    "Hãy viết báo cáo PET/CT bằng tiếng Việt theo đúng cấu trúc sau: "
    "Đầu - cổ, Lồng ngực, Ổ bụng - khung chậu, Hệ cơ - xương, Kết luận.\n"
    "Báo cáo:\n"
)

PROMPT_VQA = (
    "Người dùng: Đây là ảnh PET/CT toàn thân của bệnh nhân. "
    "{question}\n"
    "Trợ lý: "
)


# Checkpoint loading
def load_checkpoint(model, checkpoint_path: str):
    """Load projector + LoRA weights from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if "projector" in ckpt:
        model.projector.load_state_dict(ckpt["projector"])
        print("Projector loaded.")

    if "lora" in ckpt and ckpt["lora"]:
        incompatible = set_peft_model_state_dict(model.llm, ckpt["lora"])

        missing = getattr(incompatible, "missing_keys", [])
        unexpected = getattr(incompatible, "unexpected_keys", [])

        print(
            f"LoRA loaded with PEFT — "
            f"Missing: {len(missing)} | Unexpected: {len(unexpected)}"
        )

    epoch = ckpt.get("epoch", -1)
    val_loss = ckpt.get("val_loss", -1)
    print(f"Checkpoint: epoch={epoch}, val_loss={val_loss:.4f}")
    return epoch, val_loss

# Clean generated texts
def clean_generated_text(text: str) -> str:
    text = text.strip()

    stop_markers = [
        "\nNgười dùng:",
        "\nTrợ lý:",
        "Người dùng:",
        "Trợ lý:",
        "\nNguoi dung:",
        "\nTro ly:",
        "Nguoi dung:",
        "Tro ly:",
    ]

    for marker in stop_markers:
        if marker in text:
            text = text.split(marker)[0].strip()

    return text

# Report generation (bathed)
def generate_outputs(
    model,
    dataloader: DataLoader,
    device: torch.device,
    prompt: str,
    max_new_tokens: int = 1024,
) -> list:
    """
    Generate report outputs for all samples in dataloader.
    """
    model.eval()
    results = []

    prompt_ids = model.tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=True,
    ).input_ids.to(device)

    prompt_mask = torch.ones_like(prompt_ids)

    for batch in tqdm(dataloader, desc="Generating"):
        pet = batch["pet"].to(device)
        B = pet.shape[0]

        batch_prompt_ids = prompt_ids.expand(B, -1)
        batch_prompt_mask = prompt_mask.expand(B, -1)

        output_ids = model.generate(
            pet=pet,
            input_ids=batch_prompt_ids,
            attention_mask=batch_prompt_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.15,
        )

        generated = [clean_generated_text(x) for x in model.decode(output_ids)]
        report = batch["report"]
        ground_truths = report.get("structured_text", report["full_text"])

        for i in range(B):
            results.append({
                "patient_id": batch["patient_id"][i],
                "generated": generated[i],
                "ground_truth": ground_truths[i],
            })

    return results


# VQA batch generation
def collate_vqa_inference(batch):
    pets = []
    for x in batch:
        pet = x["pet"]
        if pet.ndim == 3:
            pet = pet.unsqueeze(0)  # (D,H,W) -> (1,D,H,W)
        pets.append(pet)

    return {
        "pet": torch.stack(pets),
        "patient_id": [x["patient_id"] for x in batch],
        "question": [x["question"] for x in batch],
        "answer": [x["answer"] for x in batch],
    }

def generate_vqa_outputs(
    model,
    dataloader: DataLoader,
    device: torch.device,
    max_new_tokens: int = 128,
) -> list:
    model.eval()
    results = []

    for batch in tqdm(dataloader, desc="Generating VQA"):
        pet = batch["pet"].to(device)

        prompts = [
            PROMPT_VQA.format(question=q)
            for q in batch["question"]
        ]

        encoded = model.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
            add_special_tokens=True,
        ).to(device)

        output_ids = model.generate(
            pet=pet,
            input_ids=encoded.input_ids,
            attention_mask=encoded.attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.15,
        )

        generated = [clean_generated_text(x) for x in model.decode(output_ids)]

        for i in range(len(generated)):
            results.append({
                "patient_id": batch["patient_id"][i],
                "question": batch["question"][i],
                "generated": generated[i],
                "ground_truth": batch["answer"][i],
            })

    return results

# Single sample inference
def load_npz_volume(npz_path: str):
    """Load a raw PET .npz file."""
    import numpy as np
    return np.load(npz_path)["data"].astype(np.float32)


def predict_single(
    model,
    device: torch.device,
    pet_path: str,
    encoder_name: str = "ctvit",
    task: str = "report",
    question: str = None,
    max_new_tokens: int = None,
) -> str:
    """
    Run PET-only inference on one patient's PET .npz file.
    Intended for the demo app.
    """
    if task == "vqa" and not question:
        raise ValueError("question is required for task='vqa'")

    pet_transform = get_transform(encoder_name, modality="pet")
    pet_raw = load_npz_volume(pet_path)
    pet = pet_transform(pet_raw).unsqueeze(0).to(device)

    if task == "vqa":
        prompt = PROMPT_VQA.format(question=question)
        max_new_tokens = max_new_tokens or 128
    elif task == "report":
        prompt = PROMPT_REPORT
        max_new_tokens = max_new_tokens or 1024
    else:
        raise ValueError(f"Unknown task: {task}")

    model.eval()

    prompt_ids = model.tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=True,
    ).input_ids.to(device)

    prompt_mask = torch.ones_like(prompt_ids)

    output_ids = model.generate(
        pet=pet,
        input_ids=prompt_ids,
        attention_mask=prompt_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        repetition_penalty=1.15,
    )

    generated = [clean_generated_text(x) for x in model.decode(output_ids)]
    return generated[0]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test"],
    )
    parser.add_argument(
        "--task",
        default="report",
        choices=["report", "vqa"],
    )
    parser.add_argument(
        "--vqa_path",
        default=None,
        help="Path to VQA conversations JSON (required for --task vqa)",
    )
    parser.add_argument(
        "--vqa_subsample",
        type=int,
        default=None,
        help="If set, stride-sample down to ~N QA pairs spread across all patients",
    )
    parser.add_argument("--output_path", default="predictions.json")
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    if args.task == "vqa" and not args.vqa_path:
        raise ValueError("--vqa_path is required for VQA task")

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Stage 3 checkpoint
    if args.checkpoint and "stage3" in args.checkpoint:
        config["model"]["use_lora"] = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Task: {args.task}")

    model = build_model(config, device)
    load_checkpoint(model, args.checkpoint)
    model.eval()

    pet_transform = get_transform(
        config["data"].get("encoder", "ctvit"),
        modality="pet",
    )

    if args.task == "vqa":
        df = pd.read_csv(config["data"]["metadata_path"])
        data_df = split_metadata(df, args.split)
        if args.max_samples:
            data_df = data_df.head(args.max_samples)
        allowed_report_paths = set(data_df["report_path"])

        vqa_dataset = ViPETVQADataset(
            metadata_path=config["data"]["metadata_path"],
            vqa_path=args.vqa_path,
            load_ct=False,
            load_pet=True,
            pet_transform=pet_transform,
            ct_transform=None,
            local_data_dir=config["data"].get("local_data_dir", None),
            allowed_report_paths = allowed_report_paths
        )

        if args.vqa_subsample and args.vqa_subsample < len(vqa_dataset.qa_pairs):
            stride = max(1, len(vqa_dataset.qa_pairs) // args.vqa_subsample)
            vqa_dataset.qa_pairs = vqa_dataset.qa_pairs[::stride]
            print(
                f"Subsampled VQA: {len(vqa_dataset.qa_pairs)} QA pairs "
                f"(stride={stride})"
            )

        max_new_tokens = args.max_new_tokens or 128
        
        vqa_loader = DataLoader(
            vqa_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2,
            collate_fn=collate_vqa_inference,
        )

        results = generate_vqa_outputs(
            model,
            vqa_loader,
            device,
            max_new_tokens=max_new_tokens,
        )
    else:
        df = pd.read_csv(config["data"]["metadata_path"])
        data_df = split_metadata(df, args.split)
        if args.max_samples:
            data_df = data_df.head(args.max_samples)

        dataset = ViPET3DDataset(
            metadata_path=config["data"]["metadata_path"],
            load_ct=False,
            load_pet=True,
            pet_transform=pet_transform,
            ct_transform=None,
            local_data_dir=config["data"].get("local_data_dir", None),
        )
        dataset.df = data_df.reset_index(drop=True)

        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2,
        )

        max_new_tokens = args.max_new_tokens or 1024

        results = generate_outputs(
            model,
            dataloader,
            device,
            prompt=PROMPT_REPORT,
            max_new_tokens=max_new_tokens,
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(results)} predictions -> {args.output_path}")

    if results:
        print("\n--- Sample output ---")
        print(f"Patient:      {results[0]['patient_id']}")
        if "question" in results[0]:
            print(f"Question:     {results[0]['question']}")
        print(f"Generated:    {results[0]['generated'][:200]}")
        print(f"Ground truth: {results[0]['ground_truth'][:200]}")

if __name__ == "__main__":
    main()