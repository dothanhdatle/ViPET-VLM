"""
Preprocessing cho ViMed-PET-CT volumes.

Data format:
    PET: int16 [0, 32767] — raw scanner output
         Normalize: / 32767 — preserve inter-patient comparability
    CT:  int16 HU [-2000, 4095]
         Normalize: clip [-1000, 1000] → (volume + 1000) / 2000

Augmentation → xem data/augmentation.py
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Tuple
from abc import ABC, abstractmethod


PET_MAX   = 32767.0
CT_MIN_HU = -1000.0
CT_MAX_HU =  1000.0


def normalize_pet(volume: np.ndarray) -> np.ndarray:
    """
    Normalize PET về [0, 1] bằng fixed max (32767).
    Preserve inter-patient comparability: cùng uptake → cùng value.
    """
    return np.clip(volume / PET_MAX, 0.0, 1.0)


def normalize_ct(volume: np.ndarray) -> np.ndarray:
    """
    Normalize CT về [0, 1] theo clinical HU range [-1000, 1000].
    Dùng fixed constants để đảm bảo inter-patient consistency:
    cùng HU value → cùng normalized value across patients.

        -1000 HU (air)         → 0.0
            0 HU (water)       → 0.5
          +40 HU (soft tissue) → 0.52
         +400 HU (bone)        → 0.70
        +1000 HU (dense bone)  → 1.0
    """
    volume = np.clip(volume, CT_MIN_HU, CT_MAX_HU)
    return (volume - CT_MIN_HU) / (CT_MAX_HU - CT_MIN_HU)


def resize_volume(
    volume: np.ndarray,
    target_depth: int,
    target_height: int,
    target_width: int,
) -> torch.Tensor:
    if volume.ndim != 3:
        raise ValueError(
            f"Expected volume shape (D,H,W), got {volume.shape}"
        )

    t = torch.from_numpy(
        np.ascontiguousarray(volume)
    ).float().unsqueeze(0).unsqueeze(0)

    t = F.interpolate(
        t,
        size=(target_depth, target_height, target_width),
        mode="trilinear",
        align_corners=False,
    )

    return t.squeeze(0)


class BaseViPETTransform(ABC):
    def __init__(self, modality: str = "pet"):
        assert modality in ["pet", "ct"]
        self.modality = modality

    def _normalize(self, volume: np.ndarray) -> np.ndarray:
        return normalize_pet(volume) if self.modality == "pet" else normalize_ct(volume)

    @property
    @abstractmethod
    def target_size(self) -> Tuple[int, int, int]: ...

    @abstractmethod
    def __call__(self, volume: np.ndarray) -> torch.Tensor: ...


class CTViTTransform(BaseViPETTransform):
    def __init__(
        self,
        modality="pet",
        depth=240,
        height=480,
        width=480,
    ):
        super().__init__(modality)
        self._depth = depth
        self._height = height
        self._width = width

    @property
    def target_size(self):
        return self._depth, self._height, self._width

    def __call__(self, volume):
        normalized = self._normalize(volume)
        return resize_volume(
            normalized,
            self._depth,
            self._height,
            self._width,
        )


class CosmosTransform(BaseViPETTransform):
    """Cosmos Tokenizer. Paper: 120 slices, giữ nguyên spatial."""
    def __init__(self, modality="pet", depth=120):
        super().__init__(modality)
        self._depth = depth

    @property
    def target_size(self): return (self._depth, None, None)

    def __call__(self, volume):
        volume = self._normalize(volume)
        H, W = volume.shape[1], volume.shape[2]
        return resize_volume(volume, self._depth, H, W)


class SwinUNETRTransform(BaseViPETTransform):
    """SwinUNETR encoder. Standard: (96, 96, 96)."""
    def __init__(self, modality="pet", depth=96, height=96, width=96):
        super().__init__(modality)
        self._depth, self._height, self._width = depth, height, width

    @property
    def target_size(self): return (self._depth, self._height, self._width)

    def __call__(self, volume):
        return resize_volume(self._normalize(volume), self._depth, self._height, self._width)


class MinimalTransform(BaseViPETTransform):
    """Chỉ normalize, không resize. Dùng để explore data."""
    @property
    def target_size(self): return (None, None, None)

    def __call__(self, volume):
        return torch.tensor(self._normalize(volume), dtype=torch.float32).unsqueeze(0)


# ─────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────

_TRANSFORM_REGISTRY: Dict[str, type] = {
    "ctvit":     CTViTTransform,
    "cosmos":    CosmosTransform,
    "swinunetr": SwinUNETRTransform,
    "minimal":   MinimalTransform,
}


def register_transform(name: str, cls: type):
    """Đăng ký transform mới cho encoder mới."""
    _TRANSFORM_REGISTRY[name] = cls


def get_transform(encoder_name: str, modality: str = "pet", **kwargs) -> BaseViPETTransform:
    """
    Lấy transform theo tên encoder.

    Example:
        >>> t = get_transform("ctvit", modality="pet")
        >>> t = get_transform("cosmos", modality="ct", depth=120)
        >>> t = get_transform("swinunetr", modality="pet", depth=96)
        >>> t = get_transform("minimal", modality="ct")
    """
    if encoder_name not in _TRANSFORM_REGISTRY:
        raise ValueError(f"Unknown encoder '{encoder_name}'. Available: {list(_TRANSFORM_REGISTRY.keys())}")
    return _TRANSFORM_REGISTRY[encoder_name](modality=modality, **kwargs)


def list_transforms() -> list:
    return list(_TRANSFORM_REGISTRY.keys())


# ─────────────────────────────────────────────
# PET/CT Fusion
# ─────────────────────────────────────────────

def fuse_pet_ct(
    pet: np.ndarray,
    ct: np.ndarray,
    strategy: str = "depth_concat",
    target_depth: int = 140,
    target_height: int = 480,
    target_width: int = 480,
) -> torch.Tensor:
    """
    Fusion PET và CT volume thành 1 tensor cho encoder.

    Strategies:
        depth_concat:   PET (D/2) + CT (D/2) → (1, D, H, W)

        channel_concat: PET (1, D, H, W) + CT (1, D, H, W) → (2, D, H, W)
                        Encoder can distinguish modality qua channel dimension.

    Args:
        pet:            (D, H, W) np.ndarray PET volume
        ct:             (D, H, W) np.ndarray CT volume
        strategy:       "depth_concat" or "channel_concat"
        target_depth:   depth after resize (each modality has target_depth//2 if depth_concat)
        target_height:  spatial height sau resize
        target_width:   spatial width sau resize

    Returns:
        depth_concat:   (1, target_depth, H, W) torch.Tensor
        channel_concat: (2, target_depth, H, W) torch.Tensor

    Example:
        >>> fused = fuse_pet_ct(pet, ct, strategy="depth_concat")
        >>> fused.shape  # (1, 140, 480, 480)

        >>> fused = fuse_pet_ct(pet, ct, strategy="channel_concat")
        >>> fused.shape  # (2, 140, 480, 480)
    """
    assert strategy in ["depth_concat", "channel_concat"], \
        f"strategy phải là depth_concat/channel_concat, got '{strategy}'"

    # Normalize each modality
    pet_norm = normalize_pet(pet)
    ct_norm  = normalize_ct(ct)

    if strategy == "depth_concat":
        # Each modality has a half depth
        depth_per_modal = target_depth // 2

        # Resize
        pet_tensor = resize_volume(pet_norm, depth_per_modal, target_height, target_width)
        ct_tensor  = resize_volume(ct_norm,  depth_per_modal, target_height, target_width)

        # Concat along depth
        # (1, D/2, H, W) + (1, D/2, H, W) → (1, D, H, W)
        return torch.cat([pet_tensor, ct_tensor], dim=1)

    else:  # channel_concat
        pet_tensor = resize_volume(pet_norm, target_depth, target_height, target_width)
        ct_tensor  = resize_volume(ct_norm,  target_depth, target_height, target_width)

        # Concat along channel: (1, D, H, W) + (1, D, H, W) → (2, D, H, W)
        return torch.cat([pet_tensor, ct_tensor], dim=0)
