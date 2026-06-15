"""
Stage 1: CLIP-style fine-tuning của CT-ViT trên PET data.

NOTE on architecture:
    Original CT-ViT (GenerateCT) is a VQ-GAN-style video tokenizer with
    4 components: encoder, vector-quantizer (VQ, codebook_size=8192),
    decoder, and optional GAN discriminator/VGG. For CLIP-style feature
    extraction we only need the ENCODER.

    The VQ step computes a (num_tokens, 8192) distance tensor on every
    forward pass — this dominates memory (~74GB for batch_size=8 even
    on an 80GB A100) and is unnecessary when only continuous features
    are needed (the standard approach for CLIP/SigLIP/LLaVA-style
    vision encoders, none of which use VQ).

    CTViTEncoderOnly below extracts and reuses only the pretrained
    encoder sub-modules (patch embedding, spatial position bias,
    spatial/temporal transformers), discarding VQ/decoder/GAN weights.
    Output is continuous pre-quantization features, equivalent to
    CTViT.encode()'s output before the VQ step.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers import AutoTokenizer, AutoModel
from models.visual_encoders.ctvit import CTViT


class CTViTEncoderOnly(nn.Module):
    """Encoder-only subset of CT-ViT — no VQ, no decoder, no GAN."""

    def __init__(self, ctvit_full: CTViT):
        super().__init__()
        self.to_patch_emb             = ctvit_full.to_patch_emb
        self.spatial_rel_pos_bias     = ctvit_full.spatial_rel_pos_bias
        self.enc_spatial_transformer  = ctvit_full.enc_spatial_transformer
        self.enc_temporal_transformer = ctvit_full.enc_temporal_transformer

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """
        Input:  (B, C, D, H, W)
        Output: (B, t, h, w, dim) — continuous encoder features, pre-VQ
        """
        device = video.device

        tokens = self.to_patch_emb(video)   # (b, t, h, w, dim)
        b = tokens.shape[0]
        *_, h, w, _ = tokens.shape
        video_shape = tuple(tokens.shape[:-1])

        tokens = rearrange(tokens, 'b t h w d -> (b t) (h w) d')
        attn_bias = self.spatial_rel_pos_bias(h, w, device=device)
        tokens = self.enc_spatial_transformer(tokens, attn_bias=attn_bias, video_shape=video_shape)
        tokens = rearrange(tokens, '(b t) (h w) d -> b t h w d', b=b, h=h, w=w)

        tokens = rearrange(tokens, 'b t h w d -> (b h w) t d')
        tokens = self.enc_temporal_transformer(tokens, video_shape=video_shape)
        tokens = rearrange(tokens, '(b h w) t d -> b t h w d', b=b, h=h, w=w)

        return tokens


class CTViTEncoder(nn.Module):
    CTVIT_CONFIG = dict(
        dim=512, codebook_size=8192, image_size=256,
        patch_size=16, temporal_patch_size=2,
        spatial_depth=4, temporal_depth=4,
        dim_head=32, heads=8, use_vgg_and_gan=False,
    )
    RAW_DIM    = 131072  # h * w * dim = 16 * 16 * 512 — flattened spatial+feature per timestep
    OUTPUT_DIM = 512

    def __init__(self, weights_path: str, freeze: bool = False, token_dim: int = 512):
        super().__init__()

        full_ctvit = CTViT(**self.CTVIT_CONFIG)

        print(f"Loading CT-ViT weights from {weights_path}...")
        state_dict = torch.load(weights_path, map_location="cpu")
        state_dict = {k: v for k, v in state_dict.items()
                      if not any(x in k for x in ["discr", "vgg"])}
        missing, unexpected = full_ctvit.load_state_dict(state_dict, strict=False)
        print(f"Loaded. Missing: {len(missing)} | Unexpected: {len(unexpected)}")

        self.ctvit_encoder = CTViTEncoderOnly(full_ctvit)
        del full_ctvit

        self.token_proj = nn.Linear(self.RAW_DIM, token_dim)
        self.OUTPUT_DIM = token_dim

        if freeze:
            for p in self.ctvit_encoder.parameters():
                p.requires_grad = False
            print("CT-ViT encoder frozen.")

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """
        Input:  (B, 1, D, H, W)
        Output: (B, T, OUTPUT_DIM) — T = temporal patches, dim-reduced, float32
        """
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            tokens = self.ctvit_encoder(video)  # (B, t, h, w, 512)
        tokens = tokens.float()
        b, t, h, w, d = tokens.shape
        tokens = tokens.reshape(b, t, h * w * d)  # (B, T=t, 131072)
        return self.token_proj(tokens)  # (B, T, OUTPUT_DIM)

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
        encoded = self.tokenizer(
            texts, padding=True, truncation=True,
            max_length=256, return_tensors="pt",
        ).to(device)
        outputs = self.model(**encoded)
        return outputs.last_hidden_state[:, 0, :]

    @property
    def output_dim(self) -> int:
        return 768


class DualCTViTCLIP(nn.Module):
    """
    Dual-encoder CLIP with simple concat fusion.

    Architecture:
        PET -> CT-ViT_PET (encoder-only) -> v_PET --|
                                                       |--> concat -> proj -> embed_dim
        CT  -> CT-ViT_CT  (encoder-only) -> v_CT  --|
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

        self.pet_encoder = CTViTEncoder(weights_path, freeze=freeze_vision)
        self.ct_encoder  = CTViTEncoder(weights_path, freeze=freeze_vision)
        self.text_encoder = PhoBERTEncoder(freeze=freeze_text)

        vision_dim = self.pet_encoder.output_dim + self.ct_encoder.output_dim
        self.vision_proj = nn.Linear(vision_dim, embed_dim)
        self.text_proj   = nn.Linear(self.text_encoder.output_dim, embed_dim)

        self.logit_scale = nn.Parameter(
            torch.ones([]) * torch.tensor(temperature).log()
        )

    def encode_image(self, pet: torch.Tensor, ct: torch.Tensor) -> torch.Tensor:
        v_pet = self.pet_encoder(pet).mean(dim=1)
        v_ct  = self.ct_encoder(ct).mean(dim=1)
        fused = torch.cat([v_pet, v_ct], dim=-1)
        return F.normalize(self.vision_proj(fused), dim=-1)

    def encode_image_tokens(self, pet: torch.Tensor, ct: torch.Tensor) -> torch.Tensor:
        v_pet = self.pet_encoder(pet)
        v_ct  = self.ct_encoder(ct)
        return torch.cat([v_pet, v_ct], dim=1)

    def encode_text(self, texts: list, device: torch.device) -> torch.Tensor:
        return F.normalize(self.text_proj(self.text_encoder(texts, device)), dim=-1)

    def forward(self, pet: torch.Tensor, ct: torch.Tensor, texts: list) -> dict:
        device = pet.device

        text_features  = self.encode_text(texts, device)
        image_features = self.encode_image(pet, ct)

        logit_scale      = self.logit_scale.exp().clamp(max=100)
        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text  = logits_per_image.T

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
