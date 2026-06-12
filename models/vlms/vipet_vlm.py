"""
Stage 2 & 3: Vision-Language Model wrapper.

Integrates DualCTViTCLIP encoders + LinearProjector + Mistral-7B.

Stage 2: Frozen encoder + frozen LLM, train projector only
Stage 3: Frozen encoder, train projector + LLM with LoRA

Following LLaVA convention:
    [visual_tokens | prompt_tokens | report_tokens]
    Loss computed only on report_tokens (prompt masked with -100)
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType
from typing import Optional

from models.base import BaseVLM
from models.projector import get_projector
from models.visual_encoders.ctvit_clip import DualCTViTCLIP


# Special token to mark image position in prompt
IMAGE_TOKEN      = "<image>"
IMAGE_TOKEN_ID   = -1  # placeholder, replaced at runtime


class ViPETVLM(BaseVLM):
    """
    ViPET Vision-Language Model.

    Architecture:
        PET + CT → DualCTViTCLIP (frozen) → encode_image_tokens()
                → (B, 2T, token_dim)
                → LinearProjector → (B, 2T, llm_dim)
                → concat with text tokens
                → Mistral-7B → report

    Args:
        encoder_weights_path: path to DualCTViTCLIP Stage 1 checkpoint
        projector_type:       "linear" or "mlp"
        llm_name:             HuggingFace model name for LLM
        token_dim:            vision encoder output token dim
        use_lora:             enable LoRA for Stage 3
        lora_r:               LoRA rank
        lora_alpha:           LoRA alpha
        lora_dropout:         LoRA dropout
    """

    def __init__(
        self,
        encoder_weights_path: str,
        projector_type:       str   = "linear",
        llm_name:             str   = "mistralai/Mistral-7B-Instruct-v0.2",
        token_dim:            int   = 512,
        use_lora:             bool  = False,
        lora_r:               int   = 64,
        lora_alpha:           int   = 16,
        lora_dropout:         float = 0.05,
    ):
        super().__init__()

        # ── Vision encoder (frozen after Stage 1) ──
        print("Loading vision encoder...")
        self.vision_encoder = DualCTViTCLIP(
            weights_path=encoder_weights_path,
            embed_dim=token_dim,
            freeze_text=True,
            freeze_vision=False,
        )
        # Freeze entire vision encoder for Stage 2/3
        for p in self.vision_encoder.parameters():
            p.requires_grad = False
        print("Vision encoder frozen.")

        # ── LLM + tokenizer ──
        print(f"Loading LLM: {llm_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            llm_name,
            padding_side="right",
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        llm_dim = self.llm.config.hidden_size  # Mistral-7B: 4096

        # ── Projector (trained in Stage 2, continue in Stage 3) ──
        print(f"Building {projector_type} projector: {token_dim*2} -> {llm_dim}...")
        self.projector = get_projector(
            projector_type,
            token_dim=token_dim,   # each token dim after dim reduction
            llm_dim=llm_dim,
        )

        # ── LoRA for Stage 3 ──
        if use_lora:
            print(f"Applying LoRA: r={lora_r}, alpha={lora_alpha}...")
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                task_type=TaskType.CAUSAL_LM,
                target_modules=[
                    "q_proj", "k_proj", "v_proj",
                    "o_proj", "gate_proj", "up_proj", "down_proj",
                ],
            )
            self.llm = get_peft_model(self.llm, lora_config)
            self.llm.print_trainable_parameters()
        else:
            # Stage 2: freeze LLM entirely
            for p in self.llm.parameters():
                p.requires_grad = False
            print("LLM frozen (Stage 2).")

        self.llm_dim = llm_dim

    def _encode_visual(
        self,
        pet: torch.Tensor,
        ct:  torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode PET + CT into visual token embeddings.

        Returns:
            (B, 2T, llm_dim) — projected visual tokens
        """
        with torch.no_grad():
            # (B, 2T, token_dim) — concat PET and CT tokens
            visual_tokens = self.vision_encoder.encode_image_tokens(pet, ct)
        # Project to LLM embedding space — projector is trainable
        return self.projector(visual_tokens)  # (B, 2T, llm_dim)

    def _build_input_embeddings(
        self,
        pet:            torch.Tensor,
        ct:             torch.Tensor,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
        labels:         Optional[torch.Tensor] = None,
    ):
        """
        Build full input sequence by replacing <image> token with visual embeddings.

        Returns:
            inputs_embeds:  (B, 2T + seq_len, llm_dim)
            attention_mask: (B, 2T + seq_len)
            labels:         (B, 2T + seq_len) with visual positions masked -100
        """
        B = pet.shape[0]
        device = pet.device

        # Visual tokens
        visual_embeds = self._encode_visual(pet, ct)   # (B, 2T, llm_dim)
        num_visual    = visual_embeds.shape[1]          # 2T

        # Text embeddings from LLM embedding layer
        text_embeds = self.llm.get_input_embeddings()(input_ids)  # (B, seq_len, llm_dim)

        # Concat: [visual | text]
        # Cast to float16 to match LLM dtype
        visual_embeds = visual_embeds.to(text_embeds.dtype)
        inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)

        # Extend attention mask for visual tokens
        visual_mask   = torch.ones(B, num_visual, device=device, dtype=attention_mask.dtype)
        attention_mask = torch.cat([visual_mask, attention_mask], dim=1)

        # Extend labels — mask visual positions with -100
        if labels is not None:
            visual_labels = torch.full(
                (B, num_visual), -100,
                device=device, dtype=labels.dtype,
            )
            labels = torch.cat([visual_labels, labels], dim=1)

        return inputs_embeds, attention_mask, labels

    def forward(
        self,
        pet:            torch.Tensor,
        ct:             torch.Tensor,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
        labels:         Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Args:
            pet:            (B, 1, D, H, W)
            ct:             (B, 1, D, H, W)
            input_ids:      (B, seq_len) tokenized prompt + report
            attention_mask: (B, seq_len)
            labels:         (B, seq_len) — -100 for prompt, token ids for report
        Returns:
            dict: loss, logits
        """
        inputs_embeds, attention_mask, labels = self._build_input_embeddings(
            pet, ct, input_ids, attention_mask, labels
        )

        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )

        return {
            "loss":   outputs.loss,
            "logits": outputs.logits,
        }

    def generate(
        self,
        pet:            torch.Tensor,
        ct:             torch.Tensor,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 512,
        **generate_kwargs,
    ) -> torch.Tensor:
        """
        Auto-regressive generation for inference.

        Returns:
            generated token ids (B, max_new_tokens)
        """
        inputs_embeds, attention_mask, _ = self._build_input_embeddings(
            pet, ct, input_ids, attention_mask, labels=None
        )

        with torch.no_grad():
            output_ids = self.llm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
                **generate_kwargs,
            )

        return output_ids

    def decode(self, token_ids: torch.Tensor) -> list:
        """Decode token ids to strings."""
        return self.tokenizer.batch_decode(
            token_ids,
            skip_special_tokens=True,
        )


def build_model(config: dict, device: torch.device) -> ViPETVLM:
    """
    Build ViPETVLM from config dict.

    Example config:
        model:
            encoder_weights_path: "/path/to/stage1_best.pt"
            projector_type:       "linear"
            llm_name:             "Qwen/Qwen2.5-0.5B-Instruct"
            token_dim:            512
            use_lora:             false
            lora_r:               64
            lora_alpha:           16
            lora_dropout:         0.05
    """
    cfg = config["model"]
    return ViPETVLM(
        encoder_weights_path = cfg["encoder_weights_path"],
        projector_type       = cfg.get("projector_type", "linear"),
        llm_name             = cfg["llm_name"],
        token_dim            = cfg.get("token_dim", 512),
        use_lora             = cfg.get("use_lora", False),
        lora_r               = cfg.get("lora_r", 64),
        lora_alpha           = cfg.get("lora_alpha", 16),
        lora_dropout         = cfg.get("lora_dropout", 0.05),
    ).to(device)
