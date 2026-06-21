"""
Stage 1: CT-ViT CLIP-style fine-tuning on PET data.

This branch follows the official ViPET-ReportGen CT-ViT path:
patch embedding -> spatial/temporal encoder -> VQ -> flatten spatial tokens.
Decoder/GAN/VGG are not used for VLM features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from models.visual_encoders.ctvit import CTViT


class CTViTEncoder(nn.Module):
    CTVIT_CONFIG = dict(
        dim=512,
        codebook_size=8192,
        image_size=480,
        patch_size=20,
        temporal_patch_size=10,
        spatial_depth=4,
        temporal_depth=4,
        dim_head=32,
        heads=8,
        use_vgg_and_gan=False,
    )
    PATCH_GRID = CTVIT_CONFIG["image_size"] // CTVIT_CONFIG["patch_size"]
    RAW_DIM = PATCH_GRID * PATCH_GRID * CTVIT_CONFIG["dim"]
    #RAW_DIM = 24 * 24 * 512  # 294912
    OUTPUT_DIM = RAW_DIM

    def __init__(self, weights_path: str, freeze: bool = False):
        super().__init__()

        # Build skeleton first so weights can be loaded into it below, in
        # either checkpoint format.
        self.ctvit = CTViT(**self.CTVIT_CONFIG)

        print(f"Loading CT-ViT pretrained weights")
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
        state_dict = ckpt["model"] if (isinstance(ckpt, dict) and "model" in ckpt) else ckpt
        
        clean_state_dict = {}
        for k, v in state_dict.items():
            k = k.removeprefix("module.")
            k = k.removeprefix("ctvit.")
            
            if any(x in k for x in ["discr", "vgg"]):
                continue
            
            clean_state_dict[k] = v
        missing, unexpected = self.ctvit.load_state_dict(clean_state_dict, strict=False)
        print(f"Loaded. Missing: {len(missing)} | Unexpected: {len(unexpected)}")

        if freeze:
            for p in self.ctvit.parameters():
                p.requires_grad = False
            print("CT-ViT encoder frozen.")

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """
        Input:  (B, 1, D, H, W)
        Output: (B, T, OUTPUT_DIM) — T = temporal patches, dim-reduced, float32
        """
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=video.is_cuda):
            tokens = self.ctvit.to_patch_emb(video)      # (B, T, h, w, 512)
            b, t, h, w, d = tokens.shape
            
            tokens = self.ctvit.encode(tokens)           # (B, T, h, w, 512)
            
            tokens = tokens.reshape(b, t * h * w, d)     # (B, T*h*w, 512)
            tokens, indices, commit_loss = self.ctvit.vq(tokens)
            
            tokens = tokens.reshape(b, t, h, w, d)       # (B, T, h, w, 512)
            tokens = tokens.reshape(b, t, h * w * d)     # (B, T, 294912)
        
        return tokens.float()  # (B, T, 294912)

    @property
    def output_dim(self) -> int:
        return self.RAW_DIM


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


class CTViTCLIP(nn.Module):
    def __init__(self, weights_path, embed_dim=512, freeze_text=True,
                 freeze_vision=False, temperature=0.07):
        super().__init__()
        self.vision_encoder = CTViTEncoder(weights_path, freeze=freeze_vision)
        self.text_encoder = PhoBERTEncoder(freeze=freeze_text)

        self.vision_proj = nn.Linear(self.vision_encoder.output_dim, embed_dim)
        self.text_proj = nn.Linear(self.text_encoder.output_dim, embed_dim)

        self.logit_scale = nn.Parameter(
            torch.ones([]) * torch.log(torch.tensor(1.0 / temperature))
        )

    def encode_image(self, pet):
        v_pet = self.vision_encoder(pet).mean(dim=1)
        return F.normalize(self.vision_proj(v_pet), dim=-1)

    def encode_image_tokens(self, pet):
        return self.vision_encoder(pet)

    def encode_text(self, texts, device):
        return F.normalize(self.text_proj(self.text_encoder(texts, device)), dim=-1)

    def forward(self, pet, texts):
        device = pet.device
        image_features = self.encode_image(pet)
        text_features = self.encode_text(texts, device)

        logit_scale = self.logit_scale.exp().clamp(max=100)
        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text = logits_per_image.T

        labels = torch.arange(len(pet), device=device)
        loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2

        return {
            "loss": loss,
            "logits_per_image": logits_per_image,
            "logits_per_text": logits_per_text,
        }


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
        weights_path: str = None,
        pet_weights_path: str = None,
        ct_weights_path: str = None,
        embed_dim: int = 512,
        freeze_text: bool = True,
        freeze_vision: bool = False,
        temperature: float = 0.07,
    ):
        """
        Args:
            weights_path:      single checkpoint used to seed BOTH encoders —
                                Stage 1 usage, e.g. the external GenerateCT
                                pretrained CT-ViT.
            pet_weights_path:  per-encoder checkpoint for the PET branch —
                                Stage 2/3 usage, e.g. stage1_best_pet_encoder.pt.
                                Overrides `weights_path` for this encoder.
            ct_weights_path:   per-encoder checkpoint for the CT branch.
                                Overrides `weights_path` for this encoder.
        """
        super().__init__()

        pet_path = pet_weights_path or weights_path
        ct_path  = ct_weights_path  or weights_path
        assert pet_path and ct_path, (
            "DualCTViTCLIP needs weights_path (shared) or both "
            "pet_weights_path and ct_weights_path."
        )

        self.pet_encoder = CTViTEncoder(pet_path, freeze=freeze_vision)
        self.ct_encoder  = CTViTEncoder(ct_path,  freeze=freeze_vision)
        self.text_encoder = PhoBERTEncoder(freeze=freeze_text)

        # DualCTViTCLIP.__init__() — vision_proj dùng đúng dim
        vision_dim = self.pet_encoder.RAW_DIM + self.ct_encoder.RAW_DIM  # 131072*2 = 262144
        self.vision_proj = nn.Linear(vision_dim, embed_dim)
        self.text_proj   = nn.Linear(self.text_encoder.output_dim, embed_dim)

        self.logit_scale = nn.Parameter(
            torch.ones([]) * torch.log(torch.tensor(1.0 / temperature))
        )

    def encode_image(self, pet: torch.Tensor, ct: torch.Tensor) -> torch.Tensor:
        v_pet = self.pet_encoder(pet).mean(dim=1)   # (B, T, 131072) → (B, 131072)
        v_ct  = self.ct_encoder(ct).mean(dim=1)    # (B, T, 131072) → (B, 131072)
        fused = torch.cat([v_pet, v_ct], dim=-1)    # (B, 262144)
        return F.normalize(self.vision_proj(fused), dim=-1)

    def encode_image_tokens(self, pet, ct):
        v_pet = self.pet_encoder(pet)               # (B, T, 131072)
        v_ct  = self.ct_encoder(ct)                 # (B, T, 131072)
        return torch.cat([v_pet, v_ct], dim=1)      # (B, 2T, 131072)

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
