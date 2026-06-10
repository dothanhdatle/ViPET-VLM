"""
Augmentation strategies cho ViMed-PET-CT.

QUAN TRỌNG:
    Augmentation CHỈ dùng cho self-supervised pretraining của visual encoder.
    KHÔNG dùng khi training VLM với paired image-report vì sẽ tạo
    image-text misalignment — ảnh thay đổi nhưng report giữ nguyên.

Strategies:
    - ssl:    standard augmentation cho contrastive/self-supervised learning
    - strong: heavy augmentation cho SimCLR/MoCo style
    - weak:   light augmentation cho knowledge distillation
"""

import torch
import torchio as tio
from typing import Optional


def get_ssl_augmentation() -> tio.Compose:
    """
    Standard augmentation cho SSL pretraining.
    Spatial + mild intensity transforms.
    KHÔNG dùng elastic deformation — distort anatomy quá nhiều.
    """
    return tio.Compose([
        tio.RandomFlip(axes=("LR",), p=0.5),
        tio.RandomAffine(scales=(0.9, 1.1), degrees=10, p=0.5),
        tio.RandomGamma(log_gamma=(-0.3, 0.3), p=0.3),
        tio.RandomNoise(std=(0, 0.05), p=0.3),
        tio.RandomBlur(std=(0, 0.5), p=0.2),
    ])


def get_strong_augmentation() -> tio.Compose:
    """Heavy augmentation cho SimCLR/MoCo style SSL."""
    return tio.Compose([
        tio.RandomFlip(axes=("LR",), p=0.5),
        tio.RandomAffine(scales=(0.8, 1.2), degrees=20, p=0.7),
        tio.RandomGamma(log_gamma=(-0.5, 0.5), p=0.5),
        tio.RandomNoise(std=(0, 0.1), p=0.5),
        tio.RandomBlur(std=(0, 1.0), p=0.3),
        tio.RandomBiasField(p=0.3),
    ])


def get_weak_augmentation() -> tio.Compose:
    """Light augmentation cho knowledge distillation."""
    return tio.Compose([
        tio.RandomFlip(axes=("LR",), p=0.5),
        tio.RandomNoise(std=(0, 0.02), p=0.2),
    ])


def apply_augmentation(
    tensor: torch.Tensor,
    augment: Optional[tio.Compose] = None,
) -> torch.Tensor:
    """
    Apply TorchIO augmentation lên tensor.

    Args:
        tensor:  (1, D, H, W) torch.Tensor — normalized volume
        augment: TorchIO Compose, None → trả về tensor không đổi

    Returns:
        (1, D, H, W) torch.Tensor
    """
    if augment is None:
        return tensor

    # TorchIO ScalarImage cần đúng 4D: (C, D, H, W)
    subject = tio.Subject(
        volume=tio.ScalarImage(tensor=tensor)  # (1, D, H, W)
    )
    augmented = augment(subject)
    return augmented["volume"].data  # (1, D, H, W)


_AUGMENTATION_REGISTRY = {
    "ssl":    get_ssl_augmentation,
    "strong": get_strong_augmentation,
    "weak":   get_weak_augmentation,
}


def get_augmentation(name: str) -> tio.Compose:
    """
    Lấy augmentation pipeline theo tên.

    Example:
        >>> aug = get_augmentation("ssl")
        >>> tensor_aug = apply_augmentation(tensor, aug)
    """
    if name not in _AUGMENTATION_REGISTRY:
        raise ValueError(f"Unknown augmentation '{name}'. Available: {list(_AUGMENTATION_REGISTRY.keys())}")
    return _AUGMENTATION_REGISTRY[name]()
