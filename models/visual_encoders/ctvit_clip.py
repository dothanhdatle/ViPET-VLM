"""
Stage 1: CLIP-style fine-tuning của CT-ViT trên PET data.

Config match pretrained weights từ GenerateCT:
    patch_size=16, temporal_patch_size=2
    input: (B, 1, 128, 256, 256)
    output: (B, 131072) sau mean pool → (B, embed_dim) sau projection

Attribution: CT-ViT from https://github.com/ibrahimethemhamamci/GenerateCT
             License: CC-BY 4.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

from models.visual_encoders.ctvit import CTViT


class CTViTEncoder(nn.Module):
    """
    Wrapper cho CT-ViT với pretrained weights từ GenerateCT.

    Config (match pretrained):
        patch_size=16, temporal_patch_size=2, image_size=256
        input:  (B, 1, 128, 256, 256)
        output: (B, 131072) — mean pooled

    Pretrained: GenerateCT (chest CT + radiology reports)
    Fine-tune:  Stage 1 CLIP-style trên PET data
    """

    CTVIT_CONFIG = dict(
        dim=512,
        codebook_size=8192,
        image_size=256,          # match pretrained
        patch_size=16,           # match pretrained
        temporal_patch_size=2,   # match pretrained
        spatial_depth=4,
        temporal_depth=4,
        dim_head=32,
        heads=8,
        use_vgg_and_gan=False,
    )

    OUTPUT_DIM = 131072  # 64 * 16 * 16 * 512 / ... = n_spatial^2 * dim = 256 * 512

    def __init__(self, weights_path: str, freeze: bool = False):
        super().__init__()

        self.ctvit = CTViT(**self.CTVIT_CONFIG)

        print(f"Loading CT-ViT weights from {weights_path}...")
        state_dict = torch.load(weights_path, map_location="cpu")
        state_dict = {
            k: v for k, v in state_dict.items()
            if not any(x in k for x in ["discr", "vgg"])
        }
        missing, unexpected = self.ctvit.load_state_dict(state_dict, strict=False)
        print(f"Loaded. Missing: {len(missing)} | Unexpected: {len(unexpected)}")

        if freeze:
            for p in self.ctvit.parameters():
                p.requires_grad = False
            print("CT-ViT frozen.")

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """
        Input:  (B, 1, 128, 256, 256)
        Output: (B, 131072)
        """
        # return_encoded_tokens=True → (B, T, spatial_dim)
        tokens = self.ctvit(video, return_encoded_tokens=True)
        # Mean pool over temporal dimension T
        return tokens.mean(dim=1)  # (B, 131072)

    @property
    def output_dim(self) -> int:
        return self.OUTPUT_DIM


class PhoBERTEncoder(nn.Module):
    """
    PhoBERT text encoder cho Vietnamese clinical reports.

    Input:  list of B strings
    Output: (B, 768) — [CLS] token embedding
    """

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
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)
        outputs = self.model(**encoded)
        return outputs.last_hidden_state[:, 0, :]  # [CLS] (B, 768)

    @property
    def output_dim(self) -> int:
        return 768


class CTViTCLIP(nn.Module):
    """
    CLIP-style model cho Stage 1: fine-tune CT-ViT trên PET data.

    Args:
        weights_path:  path tới ctvit_pretrained.pt
        embed_dim:     shared embedding dimension (default 512)
        freeze_text:   freeze PhoBERT (True để tiết kiệm memory)
        freeze_vision: freeze CT-ViT (False để fine-tune)
        temperature:   initial temperature cho InfoNCE
    """

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

        # Projection heads → shared embed_dim
        self.visual_proj = nn.Linear(self.visual_encoder.output_dim, embed_dim)
        self.text_proj   = nn.Linear(self.text_encoder.output_dim,   embed_dim)

        # Learnable log temperature
        self.logit_scale = nn.Parameter(
            torch.ones([]) * torch.tensor(temperature).log()
        )

    def encode_image(self, video: torch.Tensor) -> torch.Tensor:
        """(B, 1, 128, 256, 256) → (B, embed_dim) L2-normalized"""
        return F.normalize(self.visual_proj(self.visual_encoder(video)), dim=-1)

    def encode_text(self, texts: list, device: torch.device) -> torch.Tensor:
        """list of B strings → (B, embed_dim) L2-normalized"""
        return F.normalize(self.text_proj(self.text_encoder(texts, device)), dim=-1)

    def forward(self, video: torch.Tensor, texts: list) -> dict:
        """
        Args:
            video: (B, 1, 128, 256, 256)
            texts: list of B Vietnamese report strings

        Returns:
            dict: loss, logits_per_image, logits_per_text
        """
        device = video.device

        image_features = self.encode_image(video)
        text_features  = self.encode_text(texts, device)

        # Similarity matrix
        logit_scale      = self.logit_scale.exp().clamp(max=100)
        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text  = logits_per_image.T

        # InfoNCE loss
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
