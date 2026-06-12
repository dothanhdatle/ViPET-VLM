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
    RAW_DIM    = 131072  # raw CT-ViT token dim — too large to project directly
    OUTPUT_DIM = 512     # reduced dim — compact token representation

    def __init__(self, weights_path: str, freeze: bool = False, token_dim: int = 512):
        super().__init__()
        self.ctvit = CTViT(**self.CTVIT_CONFIG)
        # Dimension reduction: 131072 -> token_dim
        # Applied per token: (B, T, 131072) -> (B, T, token_dim)
        self.token_proj = nn.Linear(self.RAW_DIM, token_dim)
        self.OUTPUT_DIM = token_dim

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
        Input:  (B, 1, D, H, W)
        Output: (B, T, OUTPUT_DIM) — all spatial tokens, dim-reduced, float32

        Pipeline:
            CT-ViT → (B, T, 131072) raw tokens
            token_proj → (B, T, OUTPUT_DIM=512) compact tokens
            No mean pool — keep all T tokens following LLaVA convention
        """
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            tokens = self.ctvit(video, return_encoded_tokens=True)
        tokens = tokens.float()                     # (B, T, 131072)
        return self.token_proj(tokens)              # (B, T, OUTPUT_DIM)

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
        # PhoBERT max_position_embeddings=258 — không được vượt quá
        encoded = self.tokenizer(
            texts, padding=True, truncation=True,
            max_length=256, return_tensors="pt",
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
        # Mean pool over T tokens for CLIP loss — needs single vector per sample
        tokens = self.visual_encoder(video)          # (B, T, token_dim)
        pooled = tokens.mean(dim=1)                  # (B, token_dim)
        return F.normalize(self.visual_proj(pooled), dim=-1)

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


class DualCTViTCLIP(nn.Module):
    """
    Dual-encoder CLIP with simple concat fusion.

    Compared to ViPET-VLM (cross-attention fusion):
        - Simpler: concat instead of cross-attention
        - No GPT-4o report splitting needed
        - Single InfoNCE loss on full report
        - Fewer parameters

    Architecture:
        PET -> CT-ViT_PET -> v_PET --|
                                      |--> concat -> proj -> embed_dim
        CT  -> CT-ViT_CT  -> v_CT  --|
        Full report -> PhoBERT -> text_proj -> embed_dim
        Loss: InfoNCE(fused_image_features, text_features)
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

        # Two independent encoders — same init weights but fine-tuned separately
        self.pet_encoder = CTViTEncoder(weights_path, freeze=freeze_vision)
        self.ct_encoder  = CTViTEncoder(weights_path, freeze=freeze_vision)

        # Text encoder
        self.text_encoder = PhoBERTEncoder(freeze=freeze_text)

        # Vision projection: concat(v_PET, v_CT) -> embed_dim
        vision_dim = self.pet_encoder.output_dim + self.ct_encoder.output_dim
        self.vision_proj = nn.Linear(vision_dim, embed_dim)
        self.text_proj   = nn.Linear(self.text_encoder.output_dim, embed_dim)

        # Learnable temperature
        self.logit_scale = nn.Parameter(
            torch.ones([]) * torch.tensor(temperature).log()
        )

    def encode_image(
        self,
        pet: torch.Tensor,
        ct:  torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode PET and CT volumes into a single fused representation.

        Args:
            pet: (B, 1, D, H, W)
            ct:  (B, 1, D, H, W)
        Returns:
            (B, embed_dim) L2-normalized
        """
        # Mean pool over T tokens — needed for CLIP loss (single vector per sample)
        v_pet = self.pet_encoder(pet).mean(dim=1)  # (B, T, token_dim) -> (B, token_dim)
        v_ct  = self.ct_encoder(ct).mean(dim=1)    # (B, T, token_dim) -> (B, token_dim)
        fused = torch.cat([v_pet, v_ct], dim=-1)   # (B, token_dim * 2)
        return F.normalize(self.vision_proj(fused), dim=-1)

    def encode_image_tokens(
        self,
        pet: torch.Tensor,
        ct:  torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode PET and CT into full token sequence for Stage 2/3.
        No mean pooling — keeps all spatial tokens following LLaVA convention.

        Args:
            pet: (B, 1, D, H, W)
            ct:  (B, 1, D, H, W)
        Returns:
            (B, T_pet + T_ct, token_dim) — concat along token dimension
        """
        v_pet = self.pet_encoder(pet)              # (B, T, token_dim)
        v_ct  = self.ct_encoder(ct)                # (B, T, token_dim)
        return torch.cat([v_pet, v_ct], dim=1)     # (B, 2T, token_dim)

    def encode_text(
        self,
        texts: list,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Encode report strings into text features.

        Args:
            texts: list of B Vietnamese report strings
        Returns:
            (B, embed_dim) L2-normalized
        """
        return F.normalize(
            self.text_proj(self.text_encoder(texts, device)),
            dim=-1,
        )

    def forward(
        self,
        pet:   torch.Tensor,
        ct:    torch.Tensor,
        texts: list,
    ) -> dict:
        """
        Args:
            pet:   (B, 1, D, H, W) PET volume
            ct:    (B, 1, D, H, W) CT volume
            texts: list of B Vietnamese report strings
        Returns:
            dict with keys: loss, logits_per_image, logits_per_text
        """
        device = pet.device

        # Text encoding first — float32, outside autocast
        text_features  = self.encode_text(texts, device)

        # Image encoding — autocast bfloat16 applied inside CTViTEncoder
        image_features = self.encode_image(pet, ct)

        # Similarity matrix
        logit_scale      = self.logit_scale.exp().clamp(max=100)
        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text  = logits_per_image.T

        # Symmetric InfoNCE loss
        labels = torch.arange(len(pet), device=device)
        loss   = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text,  labels)
        ) / 2

        return {
            "loss":             loss,
            "logits_per_image": logits_per_image,
            "logits_per_text":  logits_per_text,
        }
