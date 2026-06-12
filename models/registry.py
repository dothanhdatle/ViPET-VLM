"""
Model registry — register and retrieve models by name.

Usage:
    # Register
    @register_model("my_model")
    class MyModel(nn.Module): ...

    # Retrieve
    model_cls = get_model("my_model")
    model = model_cls(**kwargs)
"""

from typing import Dict, Type
import torch.nn as nn


_MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {}


def register_model(name: str):
    """Decorator to register a model class by name."""
    def decorator(cls):
        if name in _MODEL_REGISTRY:
            raise ValueError(f"Model '{name}' already registered.")
        _MODEL_REGISTRY[name] = cls
        return cls
    return decorator


def get_model(name: str) -> Type[nn.Module]:
    """Retrieve a registered model class by name."""
    if name not in _MODEL_REGISTRY:
        available = list(_MODEL_REGISTRY.keys())
        raise ValueError(
            f"Model '{name}' not found. "
            f"Available: {available}"
        )
    return _MODEL_REGISTRY[name]


def list_models():
    """List all registered model names."""
    return list(_MODEL_REGISTRY.keys())
