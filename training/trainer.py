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

from data.dataset import ViPET3DDataset, split_metadata
from data.preprocessing import get_transform


class Stage1Trainer:
    """
    Trainer cho Stage 1: CLIP-style fine-tuning CT-ViT.

    Args:
        model:  CTViTCLIP model
        config: dict config
        device: torch.device
    """

    def __init__(self, model, config: dict, device: torch.device):
        self.model  = model
        self.config = config
        self.device = device

        # Optimizer — chỉ optimize trainable params
        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"].get("weight_decay", 0.01),
        )

        # Tracking
        self.global_step   = 0
        self.best_val_loss = float("inf")

        # Checkpoint dir
        self.checkpoint_dir = config["output"]["checkpoint_dir"]
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def _build_scheduler(self, total_steps: int):
        """Build cosine scheduler với linear warmup."""
        warmup_steps = self.config["training"].get("warmup_steps", 100)

        warmup = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        cosine = CosineAnnealingLR(
            self.optimizer,
            T_max=max(total_steps - warmup_steps, 1),
            eta_min=1e-8,
        )
        return SequentialLR(
            self.optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_steps],
        )

    def _build_dataloader(
        self,
        df: pd.DataFrame,
        shuffle: bool,
    ) -> DataLoader:
        """Build DataLoader từ metadata DataFrame."""
        transform = get_transform(
            self.config["data"].get("encoder", "ctvit"),
            modality="pet",
        )

        dataset    = ViPET3DDataset(
            metadata_path=self.config["data"]["metadata_path"],
            use_english=self.config["data"].get("use_english", False),
            load_ct=False,   # Stage 1 chỉ dùng PET
            load_pet=True,
            transform=transform,
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

    def _train_step(self, batch: dict) -> dict:
        """1 training step. Returns metrics dict."""
        self.model.train()
        self.optimizer.zero_grad()

        pet   = batch["pet"].to(self.device)       # (B, 1, 128, 256, 256)
        texts = batch["report"]["full_text"]        # list of B strings

        # Forward — CTViTCLIP tính InfoNCE bên trong
        out  = self.model(pet, texts)
        loss = out["loss"]

        # Accuracy từ logits
        with torch.no_grad():
            B      = pet.shape[0]
            labels = torch.arange(B, device=self.device)
            acc_i2t = (out["logits_per_image"].argmax(dim=1) == labels).float().mean()
            acc_t2i = (out["logits_per_text"].argmax(dim=1)  == labels).float().mean()
            accuracy = (acc_i2t + acc_t2i) / 2

        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config["training"].get("gradient_clip", 1.0),
        )
        self.optimizer.step()
        self.scheduler.step()

        return {
            "loss":     loss.item(),
            "accuracy": accuracy.item(),
            "lr":       self.optimizer.param_groups[0]["lr"],
        }

    @torch.no_grad()
    def _val_epoch(self, val_loader: DataLoader) -> dict:
        """Evaluate trên toàn bộ validation set."""
        self.model.eval()
        losses, accs = [], []

        for batch in val_loader:
            pet   = batch["pet"].to(self.device)
            texts = batch["report"]["full_text"]

            out = self.model(pet, texts)
            losses.append(out["loss"].item())

            # Accuracy
            B      = pet.shape[0]
            labels = torch.arange(B, device=self.device)
            acc = ((out["logits_per_image"].argmax(dim=1) == labels).float().mean() +
                   (out["logits_per_text"].argmax(dim=1)  == labels).float().mean()) / 2
            accs.append(acc.item())

        return {
            "val_loss":     np.mean(losses),
            "val_accuracy": np.mean(accs),
        }

    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool):
        """Save checkpoint."""
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
            print(f"Best saved (val_loss={val_loss:.4f})")

    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame):
        """Main training loop."""
        cfg = self.config["training"]

        train_loader = self._build_dataloader(train_df, shuffle=True)
        val_loader   = self._build_dataloader(val_df,   shuffle=False)

        # Build scheduler sau khi biết số steps
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
        print(f"LR:          {cfg['learning_rate']}")
        print(f"{'='*60}\n")

        for epoch in range(cfg["epochs"]):
            losses, accs = [], []
            t0 = time.time()

            for batch in train_loader:
                metrics = self._train_step(batch)
                losses.append(metrics["loss"])
                accs.append(metrics["accuracy"])
                self.global_step += 1

                # Log mỗi N steps
                if self.global_step % cfg.get("log_every", 10) == 0:
                    print(
                        f"Ep {epoch+1:2d} | Step {self.global_step:5d} | "
                        f"Loss: {metrics['loss']:.4f} | "
                        f"Acc: {metrics['accuracy']:.3f} | "
                        f"LR: {metrics['lr']:.2e}"
                    )

                # Validate mỗi N steps
                if self.global_step % cfg.get("eval_every", 500) == 0:
                    val_metrics = self._val_epoch(val_loader)
                    is_best     = val_metrics["val_loss"] < self.best_val_loss

                    if is_best:
                        self.best_val_loss = val_metrics["val_loss"]

                    self.save_checkpoint(epoch, val_metrics["val_loss"], is_best)
                    print(
                        f"Val loss: {val_metrics['val_loss']:.4f} | "
                        f"Val acc: {val_metrics['val_accuracy']:.3f}"
                    )
                    self.model.train()

            # End of epoch summary
            elapsed = time.time() - t0
            print(
                f"\n[Epoch {epoch+1}/{cfg['epochs']}] "
                f"Loss: {np.mean(losses):.4f} | "
                f"Acc: {np.mean(accs):.3f} | "
                f"Time: {elapsed:.1f}s\n"
            )

        print(f"\nTraining complete! Best val loss: {self.best_val_loss:.4f}")
