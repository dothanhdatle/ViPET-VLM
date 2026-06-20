"""
Training loop cho ViPET-VLM Stage 1.
CLIP-style fine-tuning CT-ViT trên PET data.
"""

import os
import time
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from data.dataset import ViPET3DDataset, split_metadata, MixedStage2Dataset
from data.preprocessing import get_transform


class Stage1Trainer:

    def __init__(self, model, config: dict, device: torch.device):
        self.model  = model
        self.config = config
        self.device = device

        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"].get("weight_decay", 0.01),
        )

        self.global_step   = 0
        self.best_val_loss = float("inf")
        self.checkpoint_dir = config["output"]["checkpoint_dir"]
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # AMP scaler cho bfloat16 trên A100
        self.scaler = torch.cuda.amp.GradScaler()
        self.use_amp = torch.cuda.is_available()

    def _build_scheduler(self, total_steps: int):
        warmup_steps = self.config["training"].get("warmup_steps", 130)
        warmup = LinearLR(
            self.optimizer, start_factor=0.1, end_factor=1.0,
            total_iters=warmup_steps,
        )
        cosine = CosineAnnealingLR(
            self.optimizer,
            T_max=max(total_steps - warmup_steps, 1),
            eta_min=1e-8,
        )
        return SequentialLR(
            self.optimizer, schedulers=[warmup, cosine],
            milestones=[warmup_steps],
        )

    def _build_dataloader(self, df: pd.DataFrame, shuffle: bool) -> DataLoader:
        encoder_name  = self.config["data"].get("encoder", "ctvit")
        pet_transform = get_transform(encoder_name, modality="pet")
        ct_transform  = get_transform(encoder_name, modality="ct")

        dataset = ViPET3DDataset(
            metadata_path=self.config["data"]["metadata_path"],
            use_english=self.config["data"].get("use_english", False),
            load_ct=True,
            load_pet=True,
            pet_transform=pet_transform,
            ct_transform=ct_transform,
            local_data_dir=self.config["data"].get("local_data_dir", None),
        )
        dataset.df = df.reset_index(drop=True)

        return DataLoader(
            dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=shuffle,
            num_workers=self.config["training"].get("num_workers", 4),
            pin_memory=True,
        )

    def _train_step(self, batch: dict) -> dict:
        self.model.train()
        self.optimizer.zero_grad()

        pet   = batch["pet"].to(self.device)
        ct    = batch["ct"].to(self.device)
        texts = batch["report"]["full_text"]

        # AMP forward pass
        with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=torch.bfloat16):
            out  = self.model(pet, ct, texts)
            loss = out["loss"]

        # Backward với scaler
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config["training"].get("gradient_clip", 1.0),
        )
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()

        with torch.no_grad():
            B      = pet.shape[0]
            labels = torch.arange(B, device=self.device)
            acc = (
                (out["logits_per_image"].argmax(dim=1) == labels).float().mean() +
                (out["logits_per_text"].argmax(dim=1)  == labels).float().mean()
            ) / 2

        return {
            "loss":     loss.item(),
            "accuracy": acc.item(),
            "lr":       self.optimizer.param_groups[0]["lr"],
        }

    @torch.no_grad()
    def _val_epoch(self, val_loader: DataLoader) -> dict:
        self.model.eval()
        losses, accs = [], []

        for batch in val_loader:
            pet   = batch["pet"].to(self.device)
            ct    = batch["ct"].to(self.device)
            texts = batch["report"]["full_text"]

            with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=torch.bfloat16):
                out = self.model(pet, ct, texts)

            losses.append(out["loss"].item())
            B      = pet.shape[0]
            labels = torch.arange(B, device=self.device)
            acc = (
                (out["logits_per_image"].argmax(dim=1) == labels).float().mean() +
                (out["logits_per_text"].argmax(dim=1)  == labels).float().mean()
            ) / 2
            accs.append(acc.item())

        return {
            "val_loss":     np.mean(losses),
            "val_accuracy": np.mean(accs),
        }

    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool):
        ckpt = {
            "epoch":       epoch,
            "global_step": self.global_step,
            "val_loss":    val_loss,
            "model":       self.model.state_dict(),
            "optimizer":   self.optimizer.state_dict(),
            "scheduler":   self.scheduler.state_dict(),
            "config":      self.config,
        }
        torch.save(ckpt, os.path.join(self.checkpoint_dir, "stage1_latest.pt"))

        if is_best:
            torch.save(ckpt, os.path.join(self.checkpoint_dir, "stage1_best.pt"))
            torch.save(
                self.model.pet_encoder.state_dict(),
                f"{self.checkpoint_dir}/stage1_best_pet_encoder.pt"
            )
            torch.save(
                self.model.ct_encoder.state_dict(),
                f"{self.checkpoint_dir}/stage1_best_ct_encoder.pt"
            )
            print(f"Best saved (val_loss={val_loss:.4f})")

    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame):
        cfg = self.config["training"]

        train_loader = self._build_dataloader(train_df, shuffle=True)
        val_loader   = self._build_dataloader(val_df,   shuffle=False)

        total_steps    = cfg["epochs"] * len(train_loader)
        self.scheduler = self._build_scheduler(total_steps)

        print(f"\n{'='*60}")
        print(f"Stage 1: CT-ViT CLIP Fine-tuning")
        print(f"{'='*60}")
        print(f"Train: {len(train_df)} | Val: {len(val_df)}")
        print(f"Batch size:  {cfg['batch_size']}")
        print(f"Epochs:      {cfg['epochs']}")
        print(f"Steps/epoch: {len(train_loader)}")
        print(f"Total steps: {total_steps}")
        print(f"Warmup:      {cfg.get('warmup_steps', 130)} steps")
        print(f"LR:          {cfg['learning_rate']}")
        print(f"AMP:         {self.use_amp} (bfloat16)")
        print(f"{'='*60}\n")

        for epoch in range(cfg["epochs"]):
            losses, accs = [], []
            t0 = time.time()

            for batch in train_loader:
                metrics = self._train_step(batch)
                losses.append(metrics["loss"])
                accs.append(metrics["accuracy"])
                self.global_step += 1

                if self.global_step % cfg.get("log_every", 10) == 0:
                    print(
                        f"Ep {epoch+1:2d} | Step {self.global_step:5d} | "
                        f"Loss: {metrics['loss']:.4f} | "
                        f"Acc: {metrics['accuracy']:.3f} | "
                        f"LR: {metrics['lr']:.2e}"
                    )

            # ── Per-epoch eval (thay vì step-based) ──────────────────
            val_metrics = self._val_epoch(val_loader)
            is_best     = val_metrics["val_loss"] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics["val_loss"]
            self.save_checkpoint(epoch, val_metrics["val_loss"], is_best)

            elapsed = time.time() - t0
            print(
                f"[Epoch {epoch+1}/{cfg['epochs']}] "
                f"Loss: {np.mean(losses):.4f} | Acc: {np.mean(accs):.3f} | "
                f"Val Loss: {val_metrics['val_loss']:.4f} | "
                f"Val Acc: {val_metrics['val_accuracy']:.3f} | "
                f"Time: {elapsed:.1f}s"
                + (" ← best" if is_best else "")
            )

        print(f"\nTraining complete! Best val loss: {self.best_val_loss:.4f}")



# changed in your trainer.py. Everything above this line is
# only test scaffolding.
# ============================================================

class Stage2Trainer:
    PROMPT = (
        "Đây là ảnh PET/CT toàn thân của bệnh nhân. "
        "Hãy viết báo cáo y tế chi tiết cho ảnh này.\n"
        "Báo cáo: "
    )

    def __init__(self, model, config: dict, device: torch.device,
                 qa_path: str = None, qa_per_patient: int = 2):
        self.model  = model
        self.config = config
        self.device = device
        self.qa_path        = qa_path
        self.qa_per_patient = qa_per_patient

        self.optimizer = AdamW(
            list(model.projector.parameters()),
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"].get("weight_decay", 0.01),
        )

        self.global_step    = 0
        self.best_val_loss  = float("inf")
        self.checkpoint_dir = config["output"]["checkpoint_dir"]
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def _build_scheduler(self, total_steps: int):
        warmup_steps = self.config["training"].get("warmup_steps", 100)
        warmup = LinearLR(
            self.optimizer, start_factor=0.1, end_factor=1.0,
            total_iters=warmup_steps,
        )
        cosine = CosineAnnealingLR(
            self.optimizer,
            T_max=max(total_steps - warmup_steps, 1),
            eta_min=1e-8,
        )
        return SequentialLR(
            self.optimizer, schedulers=[warmup, cosine],
            milestones=[warmup_steps],
        )

    def _build_dataloader(self, df: pd.DataFrame, shuffle: bool) -> DataLoader:
        from data.dataset import ViPET3DDataset, ViPETVQADataset
        from data.preprocessing import get_transform

        encoder_name  = self.config["data"].get("encoder", "ctvit")
        pet_transform = get_transform(encoder_name, modality="pet")
        ct_transform  = get_transform(encoder_name, modality="ct")

        dataset = ViPET3DDataset(
            metadata_path=self.config["data"]["metadata_path"],
            use_english=self.config["data"].get("use_english", False),
            load_ct=True,
            load_pet=True,
            pet_transform=pet_transform,
            ct_transform=ct_transform,
            local_data_dir=self.config["data"].get("local_data_dir", None),
        )
        dataset.df = df.reset_index(drop=True)

        if shuffle and self.qa_path:
            qa_dataset = ViPETVQADataset(
                metadata_path=self.config["data"]["metadata_path"],
                vqa_path=self.qa_path,
                load_ct=True,
                load_pet=True,
                pet_transform=pet_transform,
                ct_transform=ct_transform,
                local_data_dir=self.config["data"].get("local_data_dir", None),
            )
            dataset = MixedStage2Dataset(
                dataset, qa_dataset,
                report_prompt=self.PROMPT,
                qa_per_patient=self.qa_per_patient,
            )

        return DataLoader(
            dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=shuffle,
            num_workers=self.config["training"].get("num_workers", 2),
            pin_memory=True,
        )

    def _tokenize(self, prompts: list, targets: list):
        tokenizer = self.model.tokenizer
        prompt_lens = [
            len(tokenizer(p, add_special_tokens=True).input_ids)
            for p in prompts
        ]
        full_texts = [p + t for p, t in zip(prompts, targets)]
        encoded = tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config["training"].get("max_length", 512),
        ).to(self.device)

        labels = encoded.input_ids.clone()
        for i, plen in enumerate(prompt_lens):
            labels[i, :plen] = -100
        labels[labels == tokenizer.pad_token_id] = -100

        return encoded.input_ids, encoded.attention_mask, labels

    @staticmethod
    def _prompts_and_targets(batch: dict, fixed_prompt: str):
        if "prompt" in batch:
            return batch["prompt"], batch["target"]
        texts = batch["report"]["full_text"]
        return [fixed_prompt] * len(texts), texts

    def _train_step(self, batch: dict) -> dict:
        self.model.train()
        self.optimizer.zero_grad()

        pet = batch["pet"].to(self.device)
        ct  = batch["ct"].to(self.device)
        prompts, targets = self._prompts_and_targets(batch, self.PROMPT)

        input_ids, attention_mask, labels = self._tokenize(prompts, targets)
        out  = self.model(pet, ct, input_ids, attention_mask, labels)
        loss = out["loss"]

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.projector.parameters(),
            self.config["training"].get("gradient_clip", 1.0),
        )
        self.optimizer.step()
        self.scheduler.step()

        return {"loss": loss.item(), "lr": self.optimizer.param_groups[0]["lr"]}

    @torch.no_grad()
    def _val_epoch(self, val_loader: DataLoader) -> dict:
        self.model.eval()
        losses = []

        for batch in val_loader:
            pet = batch["pet"].to(self.device)
            ct  = batch["ct"].to(self.device)
            prompts, targets = self._prompts_and_targets(batch, self.PROMPT)
            input_ids, attention_mask, labels = self._tokenize(prompts, targets)
            out = self.model(pet, ct, input_ids, attention_mask, labels)
            losses.append(out["loss"].item())

        return {"val_loss": np.mean(losses)}

    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool):
        ckpt = {
            "epoch":       epoch,
            "global_step": self.global_step,
            "val_loss":    val_loss,
            "projector":   self.model.projector.state_dict(),
            "optimizer":   self.optimizer.state_dict(),
            "scheduler":   self.scheduler.state_dict(),  # ← thêm
            "config":      self.config,
        }
        torch.save(ckpt, os.path.join(self.checkpoint_dir, "stage2_latest.pt"))
        if is_best:
            torch.save(ckpt, os.path.join(self.checkpoint_dir, "stage2_best.pt"))
            print(f"Best saved (val_loss={val_loss:.4f})")

    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame):
        cfg          = self.config["training"]
        train_loader = self._build_dataloader(train_df, shuffle=True)
        val_loader   = self._build_dataloader(val_df,   shuffle=False)
        total_steps  = cfg["epochs"] * len(train_loader)
        self.scheduler = self._build_scheduler(total_steps)

        print(f"\n{'='*60}")
        print(f"Stage 2: Vision-Language Concept Alignment"
              + (" (+ QA mix)" if self.qa_path else ""))
        print(f"{'='*60}")
        print(f"Train: {len(train_df)} report rows → {len(train_loader.dataset)} total samples")
        print(f"Val:   {len(val_df)} (report-only)")
        print(f"Batch:  {cfg['batch_size']} | Epochs: {cfg['epochs']}")
        print(f"Steps/epoch: {len(train_loader)} | Total: {total_steps}")
        print(f"LR: {cfg['learning_rate']} | Warmup: {cfg.get('warmup_steps', 100)} steps")
        print(f"{'='*60}\n")

        for epoch in range(cfg["epochs"]):
            losses = []
            t0 = time.time()

            for batch in train_loader:
                metrics = self._train_step(batch)
                losses.append(metrics["loss"])
                self.global_step += 1

                if self.global_step % cfg.get("log_every", 10) == 0:
                    print(
                        f"Ep {epoch+1:2d} | Step {self.global_step:5d} | "
                        f"Loss: {metrics['loss']:.4f} | LR: {metrics['lr']:.2e}"
                    )

            # ── Per-epoch eval ────────────────────────────────────────
            val_metrics = self._val_epoch(val_loader)
            is_best     = val_metrics["val_loss"] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics["val_loss"]
            self.save_checkpoint(epoch, val_metrics["val_loss"], is_best)

            elapsed = time.time() - t0
            print(
                f"[Epoch {epoch+1}/{cfg['epochs']}] "
                f"Loss: {np.mean(losses):.4f} | "
                f"Val Loss: {val_metrics['val_loss']:.4f} | "
                f"Time: {elapsed:.1f}s"
                + (" ← best" if is_best else "")
            )

        print(f"\nTraining complete! Best val loss: {self.best_val_loss:.4f}")

class Stage3Trainer:
    """
    Trainer for Stage 3: Instruction Tuning with LoRA.

    Frozen:  vision encoder
    Trained: projector + LLM with LoRA

    Same pipeline as Stage 2 but:
        - LLM is unfrozen with LoRA adapters
        - Uses VQA dataset (single-turn + multi-turn conversations)
        - Lower learning rate than Stage 2
    """

    PROMPT = (
        "Đây là ảnh PET/CT toàn thân của bệnh nhân. "
        "Hãy viết báo cáo y tế chi tiết cho ảnh này.\n"
        "Báo cáo: "
    )

    def __init__(self, model, config: dict, device: torch.device):
        self.model  = model
        self.config = config
        self.device = device

        # Optimize projector + LoRA parameters
        trainable_params = [
            p for p in model.parameters() if p.requires_grad
        ]
        self.optimizer = AdamW(
            trainable_params,
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"].get("weight_decay", 0.01),
        )

        self.global_step   = 0
        self.best_val_loss = float("inf")
        self.checkpoint_dir = config["output"]["checkpoint_dir"]
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def _build_scheduler(self, total_steps: int):
        warmup_steps = self.config["training"].get("warmup_steps", 100)
        warmup = LinearLR(
            self.optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps,
        )
        cosine = CosineAnnealingLR(
            self.optimizer, T_max=max(total_steps - warmup_steps, 1), eta_min=1e-8,
        )
        return SequentialLR(
            self.optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps],
        )

    def _build_dataloader(self, df: pd.DataFrame, shuffle: bool) -> DataLoader:
        encoder_name  = self.config["data"].get("encoder", "ctvit")
        pet_transform = get_transform(encoder_name, modality="pet")
        ct_transform  = get_transform(encoder_name, modality="ct")
        dataset   = ViPET3DDataset(
            metadata_path=self.config["data"]["metadata_path"],
            use_english=self.config["data"].get("use_english", False),
            load_ct=True,
            load_pet=True,
            pet_transform=pet_transform,
            ct_transform=ct_transform,
            local_data_dir=self.config["data"].get("local_data_dir", None),
        )
        dataset.df = df.reset_index(drop=True)
        return DataLoader(
            dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=shuffle,
            num_workers=self.config["training"].get("num_workers", 2),
            pin_memory=True,
        )

    def _tokenize(self, reports: list):
        """Tokenize prompt + report, mask prompt tokens with -100."""
        tokenizer  = self.model.tokenizer
        prompt_len = tokenizer(
            self.PROMPT, return_tensors="pt", add_special_tokens=True,
        ).input_ids.shape[1]

        full_texts = [self.PROMPT + r for r in reports]
        encoded    = tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config["training"].get("max_length", 512),
        ).to(self.device)

        labels = encoded.input_ids.clone()
        labels[:, :prompt_len]                   = -100
        labels[labels == tokenizer.pad_token_id] = -100

        return encoded.input_ids, encoded.attention_mask, labels

    def _train_step(self, batch: dict) -> dict:
        self.model.train()
        self.optimizer.zero_grad()

        pet   = batch["pet"].to(self.device)
        ct    = batch["ct"].to(self.device)
        texts = batch["report"]["full_text"]

        input_ids, attention_mask, labels = self._tokenize(texts)
        out  = self.model(pet, ct, input_ids, attention_mask, labels)
        loss = out["loss"]

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad],
            self.config["training"].get("gradient_clip", 1.0),
        )
        self.optimizer.step()
        self.scheduler.step()

        return {"loss": loss.item(), "lr": self.optimizer.param_groups[0]["lr"]}

    @torch.no_grad()
    def _val_epoch(self, val_loader: DataLoader) -> dict:
        self.model.eval()
        losses = []
        for batch in val_loader:
            pet   = batch["pet"].to(self.device)
            ct    = batch["ct"].to(self.device)
            texts = batch["report"]["full_text"]
            input_ids, attention_mask, labels = self._tokenize(texts)
            out   = self.model(pet, ct, input_ids, attention_mask, labels)
            losses.append(out["loss"].item())                     
        return {"val_loss": np.mean(losses)}

    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool):
        ckpt = {
            "epoch":       epoch,
            "global_step": self.global_step,
            "val_loss":    val_loss,
            "projector":   self.model.projector.state_dict(),
            "lora":        {
                k: v for k, v in self.model.llm.state_dict().items()
                if "lora" in k
            },
            "optimizer":   self.optimizer.state_dict(),
            "scheduler":   self.scheduler.state_dict(),   # ← thêm
            "config":      self.config,
        }
        torch.save(ckpt, os.path.join(self.checkpoint_dir, "stage3_latest.pt"))
        if is_best:
            torch.save(ckpt, os.path.join(self.checkpoint_dir, "stage3_best.pt"))
            print(f"Best saved (val_loss={val_loss:.4f})")

    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame):
        cfg          = self.config["training"]
        train_loader = self._build_dataloader(train_df, shuffle=True)
        val_loader   = self._build_dataloader(val_df,   shuffle=False)
        total_steps  = cfg["epochs"] * len(train_loader)
        self.scheduler = self._build_scheduler(total_steps)

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print(f"\n{'='*60}")
        print(f"Stage 3: Instruction Tuning with LoRA")
        print(f"{'='*60}")
        print(f"Train: {len(train_df)} | Val: {len(val_df)}")
        print(f"Batch size:       {cfg['batch_size']}")
        print(f"Epochs:           {cfg['epochs']}")
        print(f"Steps/epoch:      {len(train_loader)}")
        print(f"Total steps:      {total_steps}")
        print(f"LR:               {cfg['learning_rate']}")
        print(f"Trainable params: {trainable/1e6:.1f}M (projector + LoRA)")
        print(f"{'='*60}\n")

        for epoch in range(cfg["epochs"]):
            losses = []
            t0 = time.time()

            for batch in train_loader:
                metrics = self._train_step(batch)
                losses.append(metrics["loss"])
                self.global_step += 1

                if self.global_step % cfg.get("log_every", 10) == 0:
                    print(
                        f"Ep {epoch+1:2d} | Step {self.global_step:5d} | "
                        f"Loss: {metrics['loss']:.4f} | LR: {metrics['lr']:.2e}"
                    )

            # ── Per-epoch eval ────────────────────────────────────────
            val_metrics = self._val_epoch(val_loader)
            is_best     = val_metrics["val_loss"] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics["val_loss"]
            self.save_checkpoint(epoch, val_metrics["val_loss"], is_best)

            elapsed = time.time() - t0
            print(
                f"[Epoch {epoch+1}/{cfg['epochs']}] "
                f"Loss: {np.mean(losses):.4f} | "
                f"Val Loss: {val_metrics['val_loss']:.4f} | "
                f"Time: {elapsed:.1f}s"
                + (" ← best" if is_best else "")
            )

        print(f"\nTraining complete! Best val loss: {self.best_val_loss:.4f}")


class Stage3VQATrainer(Stage3Trainer):
    """
    Stage 3 variant for VQA fine-tuning.

    Same as Stage3Trainer but:
        - Uses ViPETVQADataset (question, answer pairs)
        - Prompt is built per-sample from the question
        - Target is the answer text (not full report)
    """

    PROMPT_VQA = (
        "Đây là ảnh PET/CT toàn thân của bệnh nhân. "
        "{question}\n"
        "Trả lời: "
    )

    def _build_dataloader(self, df: pd.DataFrame, shuffle: bool) -> DataLoader:
        """Build DataLoader using ViPETVQADataset."""
        from data.dataset import ViPETVQADataset

        vqa_path_cfg = self.config["data"]["vqa_path"]
        if isinstance(vqa_path_cfg, dict):
            vqa_path = vqa_path_cfg["train"] if shuffle else vqa_path_cfg["val"]
        else:
            vqa_path = vqa_path_cfg

        encoder_name  = self.config["data"].get("encoder", "ctvit")
        pet_transform = get_transform(encoder_name, modality="pet")
        ct_transform  = get_transform(encoder_name, modality="ct")
        dataset   = ViPETVQADataset(
            metadata_path=self.config["data"]["metadata_path"],
            vqa_path=vqa_path,
            use_english=self.config["data"].get("use_english", False),
            load_ct=True,
            load_pet=True,
            pet_transform=pet_transform,
            ct_transform=ct_transform,
            local_data_dir=self.config["data"].get("local_data_dir", None),
        )

        # Filter by report_path (per-study) — patients with multiple visits
        # could otherwise leak QA pairs across splits.
        valid_report_paths = set(df["report_path"])
        dataset.qa_pairs = [
            qa for qa in dataset.qa_pairs
            if qa.get("report_path") in valid_report_paths
        ]
        print(f"  Filtered to {len(dataset.qa_pairs)} QA pairs for this split")

        return DataLoader(
            dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=shuffle,
            num_workers=self.config["training"].get("num_workers", 2),
            pin_memory=True,
        )

    def _tokenize_vqa(self, questions: list, answers: list):
        """
        Tokenize per-sample prompt (with question) + answer.
        Mask prompt tokens with -100.
        """
        tokenizer   = self.model.tokenizer
        full_texts  = []
        prompt_lens = []

        for q, a in zip(questions, answers):
            prompt     = self.PROMPT_VQA.format(question=q)
            prompt_len = tokenizer(
                prompt, return_tensors="pt", add_special_tokens=True,
            ).input_ids.shape[1]
            prompt_lens.append(prompt_len)
            full_texts.append(prompt + a)

        encoded = tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config["training"].get("max_length", 512),
        ).to(self.device)

        labels = encoded.input_ids.clone()
        for i, plen in enumerate(prompt_lens):
            labels[i, :plen] = -100
        labels[labels == tokenizer.pad_token_id] = -100

        return encoded.input_ids, encoded.attention_mask, labels

    def _train_step(self, batch: dict) -> dict:
        self.model.train()
        self.optimizer.zero_grad()

        pet = batch["pet"].to(self.device)
        ct  = batch["ct"].to(self.device)

        input_ids, attention_mask, labels = self._tokenize_vqa(
            batch["question"], batch["answer"]
        )
        out  = self.model(pet, ct, input_ids, attention_mask, labels)
        loss = out["loss"]

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad],
            self.config["training"].get("gradient_clip", 1.0),
        )
        self.optimizer.step()
        self.scheduler.step()

        return {"loss": loss.item(), "lr": self.optimizer.param_groups[0]["lr"]}

    @torch.no_grad()
    def _val_epoch(self, val_loader: DataLoader) -> dict:
        self.model.eval()
        losses = []
        for batch in val_loader:
            pet = batch["pet"].to(self.device)
            ct  = batch["ct"].to(self.device)
            input_ids, attention_mask, labels = self._tokenize_vqa(
                batch["question"], batch["answer"]
            )
            out = self.model(pet, ct, input_ids, attention_mask, labels)
            losses.append(out["loss"].item())
        return {"val_loss": np.mean(losses)}

    # save_checkpoint kế thừa từ Stage3Trainer — đã fix ở trên, không cần override

    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame):
        cfg          = self.config["training"]
        train_loader = self._build_dataloader(train_df, shuffle=True)
        val_loader   = self._build_dataloader(val_df,   shuffle=False)
        total_steps  = cfg["epochs"] * len(train_loader)
        self.scheduler = self._build_scheduler(total_steps)

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print(f"\n{'='*60}")
        print(f"Stage 3 (VQA): Instruction Tuning with LoRA")
        print(f"{'='*60}")
        print(f"Train: {len(train_loader.dataset.qa_pairs)} QA pairs | "
              f"Val: {len(val_loader.dataset.qa_pairs)} QA pairs")
        print(f"Batch size:       {cfg['batch_size']}")
        print(f"Epochs:           {cfg['epochs']}")
        print(f"Steps/epoch:      {len(train_loader)}")
        print(f"Total steps:      {total_steps}")
        print(f"LR:               {cfg['learning_rate']}")
        print(f"Trainable params: {trainable/1e6:.1f}M (projector + LoRA)")
        print(f"{'='*60}\n")

        for epoch in range(cfg["epochs"]):
            losses = []
            t0 = time.time()

            for batch in train_loader:
                metrics = self._train_step(batch)
                losses.append(metrics["loss"])
                self.global_step += 1

                if self.global_step % cfg.get("log_every", 10) == 0:
                    print(
                        f"Ep {epoch+1:2d} | Step {self.global_step:5d} | "
                        f"Loss: {metrics['loss']:.4f} | LR: {metrics['lr']:.2e}"
                    )

            # ── Per-epoch eval ────────────────────────────────────────
            val_metrics = self._val_epoch(val_loader)
            is_best     = val_metrics["val_loss"] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics["val_loss"]
            self.save_checkpoint(epoch, val_metrics["val_loss"], is_best)

            elapsed = time.time() - t0
            print(
                f"[Epoch {epoch+1}/{cfg['epochs']}] "
                f"Loss: {np.mean(losses):.4f} | "
                f"Val Loss: {val_metrics['val_loss']:.4f} | "
                f"Time: {elapsed:.1f}s"
                + (" ← best" if is_best else "")
            )

        print(f"\nTraining complete! Best val loss: {self.best_val_loss:.4f}")
