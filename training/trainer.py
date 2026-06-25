"""
Training loop cho ViPET-VLM Stage 1.
CLIP-style fine-tuning CT-ViT trên PET data.
"""

import os
import time
import random
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from data.dataset import ViPET3DDataset, MixedStage2Dataset
from data.preprocessing import get_transform

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)

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
        #self.scaler = torch.cuda.amp.GradScaler()
        self.use_amp = torch.cuda.is_available()

    def _build_scheduler(self, total_steps: int):
        warmup_steps = min(
            self.config["training"].get("warmup_steps", 130),
            max(total_steps - 1, 1),
        )
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
        #ct_transform  = get_transform(encoder_name, modality="ct")

        dataset = ViPET3DDataset(
            metadata_path=self.config["data"]["metadata_path"],
            use_english=self.config["data"].get("use_english", False),
            load_ct=False,
            load_pet=True,
            pet_transform=pet_transform,
            ct_transform=None,
            local_data_dir=self.config["data"].get("local_data_dir", None),
        )
        dataset.df = df.reset_index(drop=True)

        seed = self.config["training"].get("seed", 42)
        generator = torch.Generator()
        generator.manual_seed(seed)

        return DataLoader(
            dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=shuffle,
            num_workers=self.config["training"].get("num_workers", 4),
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=generator,
            drop_last=shuffle,  # bỏ batch cuối B=1 ở train
        )

    def _train_step(self, batch: dict) -> dict:
        self.model.train()
        self.optimizer.zero_grad()

        pet   = batch["pet"].to(self.device)
        #ct    = batch["ct"].to(self.device)
        texts = batch["report"]["full_text"]

        # AMP forward pass
        with torch.amp.autocast("cuda", enabled=self.use_amp, dtype=torch.bfloat16):
            out  = self.model(pet, texts)
            loss = out["loss"]

        # Backward với scaler
        loss.backward()
        trainable_params = [
            p for p in self.model.parameters()
            if p.requires_grad
        ]
        torch.nn.utils.clip_grad_norm_(
            trainable_params,
            self.config["training"].get("gradient_clip", 1.0),
        )
        self.optimizer.step()
        self.scheduler.step()

        with torch.no_grad():
            B = pet.shape[0]
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
            #ct    = batch["ct"].to(self.device)
            texts = batch["report"]["full_text"]

            with torch.amp.autocast("cuda", enabled=self.use_amp, dtype=torch.bfloat16):
                out = self.model(pet, texts)

            losses.append(out["loss"].item())
            B = pet.shape[0]
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
                self.model.vision_encoder.state_dict(),
                f"{self.checkpoint_dir}/stage1_best_vision_encoder.pt"
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


class Stage2Trainer:
    PROMPT = (
        "Đây là ảnh PET toàn thân của bệnh nhân. "
        "Hãy viết báo cáo PET bằng tiếng Việt theo đúng cấu trúc sau: "
        "Nhận định kết quả, Đầu - cổ, Lồng ngực, "
        "Ổ bụng - khung chậu, Hệ cơ - xương.\n"
        "Báo cáo:\n"
    )

    def __init__(self, model, config: dict, device: torch.device,
                 qa_path: str = None, qa_per_study: int = 2):
        self.model  = model
        self.config = config
        self.device = device
        self.qa_path        = qa_path
        self.qa_per_study = qa_per_study

        for p in self.model.projector.parameters():
            p.requires_grad = True

        self.optimizer = AdamW(
            [p for p in self.model.projector.parameters() if p.requires_grad],
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"].get("weight_decay", 0.01),
        )

        self.global_step    = 0
        self.best_val_loss  = float("inf")
        self.checkpoint_dir = config["output"]["checkpoint_dir"]
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def _build_scheduler(self, total_steps: int):
        warmup_steps = min(
            self.config["training"].get("warmup_steps", 100),
            max(total_steps - 1, 1),
        )
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
        #ct_transform  = get_transform(encoder_name, modality="ct")

        dataset = ViPET3DDataset(
            metadata_path=self.config["data"]["metadata_path"],
            use_english=self.config["data"].get("use_english", False),
            load_ct=False,
            load_pet=True,
            pet_transform=pet_transform,
            ct_transform=None,
            local_data_dir=self.config["data"].get("local_data_dir", None),
        )
        dataset.df = df.reset_index(drop=True)

        if shuffle and self.qa_path:
            qa_dataset = ViPETVQADataset(
                metadata_path=self.config["data"]["metadata_path"],
                vqa_path=self.qa_path,
                load_ct=False,
                load_pet=True,
                pet_transform=pet_transform,
                ct_transform=None,
                local_data_dir=self.config["data"].get("local_data_dir", None),
                allowed_report_paths = set(df["report_path"]),
            )
            dataset = MixedStage2Dataset(
                dataset, qa_dataset,
                report_prompt=self.PROMPT,
                qa_per_study=self.qa_per_study,
            )

        seed = self.config["training"].get("seed", 42)
        generator = torch.Generator()
        generator.manual_seed(seed)

        return DataLoader(
            dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=shuffle,
            num_workers=self.config["training"].get("num_workers", 2),
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=generator,
        )

    def _tokenize(self, prompts: list, targets: list):
        tokenizer = self.model.tokenizer

        prompt_lens = [
            len(tokenizer(p, add_special_tokens=True).input_ids)
            for p in prompts
        ]

        eos = tokenizer.eos_token or ""
        full_texts = [p + t + eos for p, t in zip(prompts, targets)]

        encoded = tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config["training"].get("max_length", 2048),
        ).to(self.device)

        labels = encoded.input_ids.clone()
        for i, plen in enumerate(prompt_lens):
            labels[i, :plen] = -100
        #labels[labels == tokenizer.pad_token_id] = -100
        labels[encoded.attention_mask == 0] = -100 # do mistral thường pad bằng eos

        return encoded.input_ids, encoded.attention_mask, labels

    @staticmethod
    def _prompts_and_targets(batch: dict, fixed_prompt: str):
        if "prompt" in batch:
            return batch["prompt"], batch["target"]
        texts = batch["report"]["structured_text"]
        return [fixed_prompt] * len(texts), texts

    def _train_step(self, batch: dict) -> dict:
        self.model.train()

        # Hai module này bị freeze trong Stage 2.
        self.model.vision_encoder.eval()
        self.model.llm.eval()
        self.model.projector.train()

        self.optimizer.zero_grad()

        pet = batch["pet"].to(
            self.device
        )

        prompts, targets = self._prompts_and_targets(
            batch,
            self.PROMPT,
        )

        input_ids, attention_mask, labels = self._tokenize(
            prompts,
            targets,
        )

        out = self.model(
            pet,
            input_ids,
            attention_mask,
            labels,
        )
        loss = out["loss"]

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.model.projector.parameters(),
            self.config["training"].get("gradient_clip", 1.0),
        )

        self.optimizer.step()
        self.scheduler.step()

        return {
            "loss": loss.item(),
            "lr": self.optimizer.param_groups[0]["lr"],
        }

    @torch.no_grad()
    def _val_epoch(self, val_loader: DataLoader) -> dict:
        self.model.eval()
        losses = []

        for batch in val_loader:
            pet = batch["pet"].to(self.device)
            #ct  = batch["ct"].to(self.device)
            prompts, targets = self._prompts_and_targets(batch, self.PROMPT)
            input_ids, attention_mask, labels = self._tokenize(prompts, targets)
            out = self.model(pet, input_ids, attention_mask, labels)
            losses.append(out["loss"].item())

        return {"val_loss": np.mean(losses)}

    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool):
        ckpt = {
            "epoch":       epoch,
            "global_step": self.global_step,
            "val_loss":    val_loss,
            "projector":   self.model.projector.state_dict(),
            "optimizer":   self.optimizer.state_dict(),
            "scheduler":   self.scheduler.state_dict(), 
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
    Trainer for Stage 3: Report-generation instruction tuning with LoRA.

    Frozen:  vision encoder
    Trained: projector + LLM LoRA adapters

    Same report-generation pipeline as Stage 2 but:
        - LLM is adapted with LoRA
        - projector continues to be tuned
        - uses full report targets
        - lower learning rate than Stage 2
    """

    PROMPT = (
        "Đây là ảnh PET toàn thân của bệnh nhân. "
        "Hãy viết báo cáo PET bằng tiếng Việt theo đúng cấu trúc sau: "
        "Nhận định kết quả, Đầu - cổ, Lồng ngực, "
        "Ổ bụng - khung chậu, Hệ cơ - xương.\n"
        "Báo cáo:\n"
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
        warmup_steps = min(
            self.config["training"].get("warmup_steps", 100),
            max(total_steps - 1, 1),
        )
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
        #ct_transform  = get_transform(encoder_name, modality="ct")
        dataset   = ViPET3DDataset(
            metadata_path=self.config["data"]["metadata_path"],
            use_english=self.config["data"].get("use_english", False),
            load_ct=False,
            load_pet=True,
            pet_transform=pet_transform,
            ct_transform=None,
            local_data_dir=self.config["data"].get("local_data_dir", None),
        )
        dataset.df = df.reset_index(drop=True)

        seed = self.config["training"].get("seed", 42)
        generator = torch.Generator()
        generator.manual_seed(seed)
        return DataLoader(
            dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=shuffle,
            num_workers=self.config["training"].get("num_workers", 2),
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=generator,
        )

    def _tokenize(self, reports: list):
        tokenizer = self.model.tokenizer
        prompt_len = tokenizer(
            self.PROMPT, return_tensors="pt", add_special_tokens=True,
        ).input_ids.shape[1]

        eos = tokenizer.eos_token or ""
        full_texts = [self.PROMPT + r + eos for r in reports]

        encoded = tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config["training"].get("max_length", 2048),
        ).to(self.device)

        labels = encoded.input_ids.clone()
        labels[:, :prompt_len] = -100
        labels[labels == tokenizer.pad_token_id] = -100

        return encoded.input_ids, encoded.attention_mask, labels

    def _train_step(self, batch: dict) -> dict:
        self.model.train()
        self.optimizer.zero_grad()

        pet   = batch["pet"].to(self.device)
        #ct    = batch["ct"].to(self.device)
        texts = batch["report"]["structured_text"]

        input_ids, attention_mask, labels = self._tokenize(texts)
        out  = self.model(pet, input_ids, attention_mask, labels)
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
            #ct    = batch["ct"].to(self.device)
            texts = batch["report"]["structured_text"]
            input_ids, attention_mask, labels = self._tokenize(texts)
            out   = self.model(pet, input_ids, attention_mask, labels)
            losses.append(out["loss"].item())                     
        return {"val_loss": np.mean(losses)}

    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool):
        from peft import get_peft_model_state_dict

        ckpt = {
            "epoch": epoch,
            "global_step": self.global_step,
            "val_loss": val_loss,
            "projector": self.model.projector.state_dict(),
            "lora": get_peft_model_state_dict(self.model.llm),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "config": self.config,
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
        print(f"Warmup steps:     {cfg.get('warmup_steps', 100)}")
        print(f"Max length:       {cfg.get('max_length', 2048)}")
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
        "Đây là ảnh PET toàn thân của bệnh nhân. "
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
        #ct_transform  = get_transform(encoder_name, modality="ct")
        dataset = ViPETVQADataset(
            metadata_path=self.config["data"]["metadata_path"],
            vqa_path=vqa_path,
            use_english=self.config["data"].get("use_english", False),
            load_ct=False,
            load_pet=True,
            pet_transform=pet_transform,
            ct_transform=None,
            local_data_dir=self.config["data"].get("local_data_dir", None),
            allowed_report_paths=set(df["report_path"]),
        )

        seed = self.config["training"].get("seed", 42)
        generator = torch.Generator()
        generator.manual_seed(seed)

        return DataLoader(
            dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=shuffle,
            num_workers=self.config["training"].get("num_workers", 2),
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=generator,
        )

    def _tokenize_vqa(self, questions: list, answers: list):
        tokenizer = self.model.tokenizer
        eos = tokenizer.eos_token or ""

        full_texts = []
        prompt_lens = []

        for q, a in zip(questions, answers):
            q = str(q).replace("<image>", "").strip()
            a = str(a).strip()
            prompt = self.PROMPT_VQA.format(question=q)
            prompt_len = tokenizer(
                prompt, return_tensors="pt", add_special_tokens=True,
            ).input_ids.shape[1]
            prompt_lens.append(prompt_len)
            full_texts.append(prompt + a + eos)

        encoded = tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config["training"].get("max_length", 2048),
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
        #ct  = batch["ct"].to(self.device)

        input_ids, attention_mask, labels = self._tokenize_vqa(
            batch["question"], batch["answer"]
        )
        out  = self.model(pet, input_ids, attention_mask, labels)
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
            #ct  = batch["ct"].to(self.device)
            input_ids, attention_mask, labels = self._tokenize_vqa(
                batch["question"], batch["answer"]
            )
            out = self.model(pet, input_ids, attention_mask, labels)
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

def collate_multiturn_vqa(batch):
    return {
        "pet": torch.stack([x["pet"] for x in batch]),
        "patient_id": [x["patient_id"] for x in batch],
        "conversations": [x["conversations"] for x in batch],
    }


class Stage3MultiTurnVQATrainer(Stage3Trainer):
    """
    Stage 3 VQA instruction tuning using full multi-turn conversations.

    Expected batch item:
        {
            "pet": Tensor,
            "patient_id": str,
            "conversations": [
                {"from": "human", "value": "<image>\\n..."},
                {"from": "gpt", "value": "..."},
                ...
            ]
        }
    """

    USER_PREFIX = "Người dùng: "
    ASSISTANT_PREFIX = "Trợ lý: "

    def _build_dataloader(self, df: pd.DataFrame, shuffle: bool) -> DataLoader:
        from data.dataset import ViPETMultiTurnVQADataset

        vqa_path_cfg = self.config["data"]["vqa_path"]
        if isinstance(vqa_path_cfg, dict):
            vqa_path = vqa_path_cfg["train"] if shuffle else vqa_path_cfg["val"]
        else:
            vqa_path = vqa_path_cfg

        encoder_name = self.config["data"].get("encoder", "ctvit")
        pet_transform = get_transform(encoder_name, modality="pet")

        dataset = ViPETMultiTurnVQADataset(
            metadata_path=self.config["data"]["metadata_path"],
            vqa_path=vqa_path,
            use_english=self.config["data"].get("use_english", False),
            load_ct=False,
            load_pet=True,
            pet_transform=pet_transform,
            ct_transform=None,
            local_data_dir=self.config["data"].get("local_data_dir", None),
            allowed_report_paths=set(df["report_path"]),
        )

        print(
            f"  Loaded {len(dataset.conversation_items)} multi-turn conversations "
            f"for this split"
        )

        seed = self.config["training"].get("seed", 42)
        generator = torch.Generator()
        generator.manual_seed(seed)

        return DataLoader(
            dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=shuffle,
            num_workers=self.config["training"].get("num_workers", 2),
            pin_memory=True,
            collate_fn=collate_multiturn_vqa,
            worker_init_fn=seed_worker,
            generator=generator
        )

    def _encode_text(self, text: str, add_special_tokens: bool = False):
        return self.model.tokenizer(
            text,
            add_special_tokens=add_special_tokens,
        ).input_ids

    def _tokenize_multiturn(self, batch_conversations: list):
        """
        Tokenize full multi-turn conversations.
        Human turns are masked with -100.
        GPT turns are used as labels.
        """
        tokenizer = self.model.tokenizer
        eos = tokenizer.eos_token or ""

        all_input_ids = []
        all_labels = []

        for conversations in batch_conversations:
            input_ids = []
            labels = []

            first_segment = True

            for turn in conversations:
                role = turn.get("from", "")
                value = turn.get("value", "").replace("<image>", "").strip()

                if not value:
                    continue

                if role == "human":
                    text = f"{self.USER_PREFIX}{value}\n"
                    ids = self._encode_text(text, add_special_tokens=first_segment)
                    input_ids.extend(ids)
                    labels.extend([-100] * len(ids))

                elif role == "gpt":
                    prefix_ids = self._encode_text(
                        self.ASSISTANT_PREFIX,
                        add_special_tokens=False,
                    )
                    answer_ids = self._encode_text(
                        f"{value}{eos}\n",
                        add_special_tokens=False,
                    )

                    input_ids.extend(prefix_ids + answer_ids)
                    labels.extend([-100] * len(prefix_ids) + answer_ids)

                first_segment = False

            max_length = self.config["training"].get("max_length", 2048)
            input_ids = input_ids[:max_length]
            labels = labels[:max_length]

            all_input_ids.append(torch.tensor(input_ids, dtype=torch.long))
            all_labels.append(torch.tensor(labels, dtype=torch.long))

        input_ids = torch.nn.utils.rnn.pad_sequence(
            all_input_ids,
            batch_first=True,
            padding_value=tokenizer.pad_token_id,
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            all_labels,
            batch_first=True,
            padding_value=-100,
        )

        attention_mask = input_ids.ne(tokenizer.pad_token_id).long()

        return (
            input_ids.to(self.device),
            attention_mask.to(self.device),
            labels.to(self.device),
        )

    def _train_step(self, batch: dict) -> dict:
        self.model.train()
        self.optimizer.zero_grad()

        pet = batch["pet"].to(self.device)

        input_ids, attention_mask, labels = self._tokenize_multiturn(
            batch["conversations"]
        )

        out = self.model(pet, input_ids, attention_mask, labels)
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

            input_ids, attention_mask, labels = self._tokenize_multiturn(
                batch["conversations"]
            )

            out = self.model(pet, input_ids, attention_mask, labels)
            losses.append(out["loss"].item())

        return {"val_loss": np.mean(losses)}

    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame):
        cfg = self.config["training"]

        train_loader = self._build_dataloader(train_df, shuffle=True)
        val_loader = self._build_dataloader(val_df, shuffle=False)

        total_steps = cfg["epochs"] * len(train_loader)
        self.scheduler = self._build_scheduler(total_steps)

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print(f"\n{'='*60}")
        print("Stage 3 (VQA Multi-turn): Instruction Tuning with LoRA")
        print(f"{'='*60}")
        print(f"Train: {len(train_loader.dataset.conversation_items)} conversations")
        print(f"Val:   {len(val_loader.dataset.conversation_items)} conversations")
        print(f"Batch size:       {cfg['batch_size']}")
        print(f"Epochs:           {cfg['epochs']}")
        print(f"Steps/epoch:      {len(train_loader)}")
        print(f"Total steps:      {total_steps}")
        print(f"LR:               {cfg['learning_rate']}")
        print(f"Max length:       {cfg.get('max_length', 2048)}")
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

            val_metrics = self._val_epoch(val_loader)
            is_best = val_metrics["val_loss"] < self.best_val_loss

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