"""Model registry for registering and retrieving models."""

from typing import Dict, Callable

_MODELS: Dict[str, Callable] = {}


def register_model(name):
    """Decorator to register a model."""
    def deco(cls):
        _MODELS[name] = cls
        return cls
    return deco


def get_model(name: str, **kwargs):
    """Get a model by name."""
    if name not in _MODELS:
        raise KeyError(f"Unknown model: {name}")
    return _MODELS[name](**kwargs)
