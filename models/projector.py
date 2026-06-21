"""
Stage 2: Linear Projector for vision-language alignment.

Following LLaVA convention:
    - Input: (B, T, vision_dim) — all spatial tokens from vision encoder
    - Project each token independently: Linear(vision_dim, llm_dim)
    - Output: (B, T, llm_dim) — visual token sequence for LLM

Paper reference:
    LLaVA: Visual Instruction Tuning (Liu et al., 2023)
    ViPET-VLM: linear projection layer maps visual features into LLM embedding space
"""

import torch
import torch.nn as nn
from models.base import BaseProjector


class LinearProjector(BaseProjector):
    """
    Single linear layer projector — applied per token (LLaVA style).

    Architecture:
        (B, T, vision_dim) -> Linear(vision_dim, llm_dim) -> LayerNorm -> (B, T, llm_dim)

    Args:
        vision_dim: input dimension from vision encoder
        llm_dim:   LLM embedding dimension (Mistral-7B: 4096)
    """

    def __init__(self, vision_dim: int, llm_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(vision_dim, llm_dim),
            nn.LayerNorm(llm_dim),
        )

    def forward(self, visual_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            visual_tokens: (B, T, vision_dim)
        Returns:
            (B, T, llm_dim)
        """
        return self.proj(visual_tokens)  # applied per token automatically


class MLPProjector(BaseProjector):
    """
    Two-layer MLP projector — applied per token (LLaVA style).

    Architecture:
        (B, T, vision_dim) -> Linear -> GELU -> Linear -> LayerNorm -> (B, T, llm_dim)

    Args:
        vision_dim:  input token dimension
        llm_dim:    LLM embedding dimension
        hidden_dim: hidden layer dimension (default: llm_dim)
    """

    def __init__(
        self,
        vision_dim:  int,
        llm_dim:    int,
        hidden_dim: int = None,
    ):
        super().__init__()
        hidden_dim = hidden_dim or llm_dim
        self.proj  = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, llm_dim),
            nn.LayerNorm(llm_dim),
        )

    def forward(self, visual_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            visual_tokens: (B, T, vision_dim)
        Returns:
            (B, T, llm_dim)
        """
        return self.proj(visual_tokens)


def get_projector(
    projector_type: str,
    vision_dim:      int,
    llm_dim:        int,
    **kwargs,
) -> BaseProjector:
    """
    Factory function for projectors.

    Args:
        projector_type: "linear" or "mlp"
        vision_dim:      vision encoder output dimension
        llm_dim:        LLM embedding dimension
        **kwargs:       additional args (hidden_dim for mlp)

    Example:
        >>> # CT-ViT vision_dim=131072, Mistral-7B llm_dim=4096
        >>> proj = get_projector("linear", vision_dim=131072, llm_dim=4096)
        >>> proj = get_projector("mlp", vision_dim=131072, llm_dim=4096)
    """
    if projector_type == "linear":
        return LinearProjector(vision_dim, llm_dim, **kwargs)
    elif projector_type == "mlp":
        return MLPProjector(vision_dim, llm_dim, **kwargs)
    else:
        raise ValueError(
            f"Unknown projector type: '{projector_type}'. "
            f"Choose: 'linear', 'mlp'"
        )
