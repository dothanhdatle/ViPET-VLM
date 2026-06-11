"""
Stage 1: CLIP-style fine-tuning của CT-ViT trên PET data.

Dùng return_encoded_tokens=True — đi qua VQ như paper ViMed-PET.
Training với autocast bfloat16 để handle memory.

Config match pretrained weights từ GenerateCT:
    patch_size=16, temporal_patch_size=2
    input: (B, 1, 128, 256, 256)
    output: (B, 294912) sau VQ + mean pool + flatten

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
    Dùng return_encoded_tokens=True — đi qua VQ như paper.

    Input:  (B, 1, 128, 256, 256)
    Output: (B, 294912) — sau VQ, mean pool over T, flatten H*W*dim
    """

    CTVIT_CONFIG = dict(
        dim=512,
        codebook_size=8192,
        image_size=256,
        patch_size=16,
        temporal_patch_size=2,
        spatial_depth=4,
        temporal_depth=4,
        dim_head=32,
        heads=8,
        use_vgg_and_gan=False,
    )

    OUTPUT_DIM = 131072  # 16*16*512 = 131072 (sau mean pool over T=64)

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
        Output: (B, 131072) float32
        """
        # return_encoded_tokens=True → (B, T*H*W, dim) sau VQ
        # ctvit_llava.py: mean(dim=1) rồi view(-1)
        tokens = self.ctvit(video, return_encoded_tokens=True)
        return tokens.float()  # cast về float32

    @property
    def output_dim(self) -> int:
        return self.OUTPUT_DIM


class PhoBERTEncoder(nn.Module):
    """
    PhoBERT text encoder cho Vietnamese clinical reports.

    Input:  list of B strings
    Output: (B, 768) float32 — [CLS] token embedding
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
        return outputs.last_hidden_state[:, 0, :].float()  # [CLS] (B, 768)

    @property
    def output_dim(self) -> int:
        return 768


class CTViTCLIP(nn.Module):
    """
    CLIP-style model cho Stage 1: fine-tune CT-ViT trên PET data.

    Training: dùng torch.amp.autocast('cuda', dtype=torch.bfloat16)
              để handle VQ memory requirement.

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

        # Projection heads — float32
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
