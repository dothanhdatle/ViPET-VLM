"""
Abstract base classes for ViPET-VLM models.
"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class BaseProjector(ABC, nn.Module):
    """
    Abstract base class for vision-language projectors.
    Maps visual features into LLM embedding space.
    """

    @abstractmethod
    def forward(self, visual_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            visual_features: (B, vision_dim)
        Returns:
            (B, num_tokens, llm_dim)
        """
        pass


class BaseVLM(ABC, nn.Module):
    """
    Abstract base class for Vision-Language Models.
    """

    @abstractmethod
    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor = None,
    ) -> dict:
        pass

    @abstractmethod
    def generate(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **generate_kwargs,
    ) -> torch.Tensor:
        pass
