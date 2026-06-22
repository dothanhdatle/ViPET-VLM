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
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType
from typing import Optional

from models.base import BaseVLM
from models.projector import get_projector
from models.visual_encoders.ctvit_clip import CTViTEncoder


# Special token to mark image position in prompt
IMAGE_TOKEN      = "<image>"
#IMAGE_TOKEN_ID   = -1  # placeholder, replaced at runtime


class ViPETVLM(BaseVLM):
    """
    ViPET Vision-Language Model.

    Architecture:
        PET → Vision Encoder (frozen)
            → (B, T, vision_dim)
             → LinearProjector → (B, T, llm_dim)
            → concat with text tokens
            → Mistral-7B → report

    Args:
        pet_encoder_weights_path: path to Stage 1 vision encoder checkpoint
        projector_type:       "linear" or "mlp"
        llm_name:             HuggingFace model name for LLM
        use_lora:             enable LoRA for Stage 3
        lora_r:               LoRA rank
        lora_alpha:           LoRA alpha
        lora_dropout:         LoRA dropout
        lora_target_modules:  LoRA target modules
    """

    def __init__(
        self,
        pet_encoder_weights_path: str,
        projector_type:       str   = "linear",
        llm_name:             str   = "mistralai/Mistral-7B-Instruct-v0.2",
        use_lora:             bool  = False,
        lora_r:               int   = 64,
        lora_alpha:           int   = 16,
        lora_dropout:         float = 0.05,
        lora_target_modules:  list  = None,
    ):
        super().__init__()

        if lora_target_modules is None:
            lora_target_modules = [
                "q_proj", "k_proj", "v_proj",
                "o_proj", "gate_proj", "up_proj", "down_proj",
            ]

        # Vision encoder (frozen after Stage 1)
        print("Loading vision encoder...")
        self.vision_encoder = CTViTEncoder(
            weights_path=pet_encoder_weights_path,
            freeze=True,
        )
        # Freeze entire vision encoder for Stage 2/3
        for p in self.vision_encoder.parameters():
            p.requires_grad = False
        print("Vision encoder frozen.")

        # LLM + tokenizer
        print(f"Loading LLM: {llm_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            llm_name,
            padding_side="right",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.llm.config.pad_token_id = self.tokenizer.pad_token_id
        llm_dim = self.llm.config.hidden_size  # Mistral-7B: 4096

        # Projector (trained in Stage 2, continue in Stage 3)
        vision_dim = self.vision_encoder.output_dim
        print(f"Building {projector_type} projector: {vision_dim} -> {llm_dim}...")
        self.projector = get_projector(
            projector_type,
            vision_dim=vision_dim,
            llm_dim=llm_dim,
        )

        # LoRA for Stage 3
        if use_lora:
            print(f"Applying LoRA: r={lora_r}, alpha={lora_alpha}, target_modules={lora_target_modules}...")
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                task_type=TaskType.CAUSAL_LM,
                target_modules=lora_target_modules,
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
        pet: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode PET into visual token embeddings.

        Returns:
            (B, T, llm_dim) — projected visual tokens
        """
        with torch.no_grad():
            visual_tokens = self.vision_encoder(pet)
        # Project to LLM embedding space — projector is trainable
        return self.projector(visual_tokens)  # (B, T, llm_dim)

    def _build_input_embeddings(
        self,
        pet:            torch.Tensor,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
        labels:         Optional[torch.Tensor] = None,
    ):
        """
        Build full input sequence by replacing <image> token with visual embeddings.

        Returns:
            inputs_embeds:  (B, T + seq_len, llm_dim)
            attention_mask: (B, T + seq_len)
            labels:         (B, T + seq_len) with visual positions masked -100
        """
        B = pet.shape[0]
        device = pet.device

        # Visual tokens
        visual_embeds = self._encode_visual(pet)   # (B, T, llm_dim)
        num_visual    = visual_embeds.shape[1]          # T

        # Text embeddings from LLM embedding layer
        text_embeds = self.llm.get_input_embeddings()(input_ids)  # (B, seq_len, llm_dim)
        device = text_embeds.device

        # Concat: [visual | text]
        # Cast to float16 to match LLM dtype
        visual_embeds = visual_embeds.to(device=device, dtype=text_embeds.dtype)
        attention_mask = attention_mask.to(device)

        inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)

        # Extend attention mask for visual tokens
        visual_mask   = torch.ones(B, num_visual, device=device, dtype=attention_mask.dtype)
        attention_mask = torch.cat([visual_mask, attention_mask], dim=1)

        # Extend labels — mask visual positions with -100
        if labels is not None:
            labels = labels.to(device)
            visual_labels = torch.full(
                (B, num_visual), -100,
                device=device, dtype=labels.dtype,
            )
            labels = torch.cat([visual_labels, labels], dim=1)

        return inputs_embeds, attention_mask, labels

    def forward(
        self,
        pet:            torch.Tensor,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
        labels:         Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Args:
            pet:            (B, 1, D, H, W)
            input_ids:      (B, seq_len) tokenized prompt + report
            attention_mask: (B, seq_len)
            labels:         (B, seq_len) — -100 for prompt, token ids for report
        Returns:
            dict: loss, logits
        """
        inputs_embeds, attention_mask, labels = self._build_input_embeddings(
            pet, input_ids, attention_mask, labels
        )

        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )

        return {
            "loss": outputs.loss if labels is not None else None,
            "logits": outputs.logits,
        }
    
    @torch.no_grad()
    def generate(
        self,
        pet:            torch.Tensor,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 1024,
        **generate_kwargs,
    ) -> torch.Tensor:
        """
        Auto-regressive generation for inference.

        Returns:
            generated token ids with shape (B, generated_len),
            where generated_len <= max_new_tokens.
        """
        inputs_embeds, attention_mask, _ = self._build_input_embeddings(
            pet, input_ids, attention_mask, labels=None
        )

        output_ids = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
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
    cfg = config["model"]
    model = ViPETVLM(
        pet_encoder_weights_path = cfg["pet_encoder_weights_path"],
        projector_type       = cfg.get("projector_type", "linear"),
        llm_name             = cfg["llm_name"],
        use_lora             = cfg.get("use_lora", False),
        lora_r               = cfg.get("lora_r", 64),
        lora_alpha           = cfg.get("lora_alpha", 16),
        lora_target_modules  = cfg.get("lora_target_modules", None),
        lora_dropout         = cfg.get("lora_dropout", 0.05),
    )

    model.vision_encoder.to(device)
    model.projector.to(device)
    return model
