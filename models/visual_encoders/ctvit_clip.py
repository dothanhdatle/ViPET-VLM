"""
Stage 1: CLIP-style fine-tuning của CT-ViT trên PET data.

autocast bfloat16 chỉ apply trong CTViTEncoder.forward()
để tránh conflict với PhoBERT tokenizer (cần float32/int).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from models.visual_encoders.ctvit import CTViT


class CTViTEncoder(nn.Module):
    CTVIT_CONFIG = dict(
        dim=512, codebook_size=8192, image_size=256,
        patch_size=16, temporal_patch_size=2,
        spatial_depth=4, temporal_depth=4,
        dim_head=32, heads=8, use_vgg_and_gan=False,
    )
    OUTPUT_DIM = 131072

    def __init__(self, weights_path: str, freeze: bool = False):
        super().__init__()
        self.ctvit = CTViT(**self.CTVIT_CONFIG)

        print(f"Loading CT-ViT weights from {weights_path}...")
        state_dict = torch.load(weights_path, map_location="cpu")
        state_dict = {k: v for k, v in state_dict.items()
                      if not any(x in k for x in ["discr", "vgg"])}
        missing, unexpected = self.ctvit.load_state_dict(state_dict, strict=False)
        print(f"Loaded. Missing: {len(missing)} | Unexpected: {len(unexpected)}")

        if freeze:
            for p in self.ctvit.parameters():
                p.requires_grad = False
            print("CT-ViT frozen.")

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """
        autocast bfloat16 chỉ trong encoder để tránh OOM từ VQ one_hot.
        Cast về float32 sau encoder để compatible với projection head.
        """
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            tokens = self.ctvit(video, return_encoded_tokens=True)
        return tokens.float()  # (B, 131072)

    @property
    def output_dim(self) -> int:
        return self.OUTPUT_DIM


class PhoBERTEncoder(nn.Module):
    MODEL_NAME = "vinai/phobert-base-v2"

    def __init__(self, freeze: bool = True):
        super().__init__()
        print(f"Loading PhoBERT: {self.MODEL_NAME}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.model     = AutoModel.from_pretrained(self.MODEL_NAME)

        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False
            print("PhoBERT frozen.")

    def forward(self, texts: list, device: torch.device) -> torch.Tensor:
        """Float32 — không dùng autocast."""
        encoded = self.tokenizer(
            texts, padding=True, truncation=True,
            max_length=512, return_tensors="pt",
        ).to(device)
        outputs = self.model(**encoded)
        return outputs.last_hidden_state[:, 0, :]  # [CLS] (B, 768)

    @property
    def output_dim(self) -> int:
        return 768


class CTViTCLIP(nn.Module):
    def __init__(
        self,
        weights_path: str,
        embed_dim: int = 512,
        freeze_text: bool = True,
        freeze_vision: bool = False,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.visual_encoder = CTViTEncoder(weights_path, freeze=freeze_vision)
        self.text_encoder   = PhoBERTEncoder(freeze=freeze_text)
        self.visual_proj    = nn.Linear(self.visual_encoder.output_dim, embed_dim)
        self.text_proj      = nn.Linear(self.text_encoder.output_dim,   embed_dim)
        self.logit_scale    = nn.Parameter(
            torch.ones([]) * torch.tensor(temperature).log()
        )

    def encode_image(self, video: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.visual_proj(self.visual_encoder(video)), dim=-1)

    def encode_text(self, texts: list, device: torch.device) -> torch.Tensor:
        return F.normalize(self.text_proj(self.text_encoder(texts, device)), dim=-1)

    def forward(self, video: torch.Tensor, texts: list) -> dict:
        device = video.device

        # Text encoding trước — float32, không bị ảnh hưởng bởi autocast
        text_features  = self.encode_text(texts, device)

        # Image encoding — autocast bfloat16 được apply trong CTViTEncoder
        image_features = self.encode_image(video)

        logit_scale      = self.logit_scale.exp().clamp(max=100)
        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text  = logits_per_image.T

        labels = torch.arange(len(video), device=device)
        loss   = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text,  labels)
        ) / 2

        return {
            "loss":             loss,
            "logits_per_image": logits_per_image,
            "logits_per_text":  logits_per_text,
        }
