"""Dataset registry for registering and retrieving datasets."""

from typing import Dict, Callable

_DATASETS: Dict[str, Callable] = {}


def register_dataset(name):
    """Decorator to register a dataset."""
    def deco(fn):
        _DATASETS[name] = fn
        return fn
    return deco


def get_dataset(name: str, **kwargs):
    """Get a dataset by name."""
    if name not in _DATASETS:
        raise KeyError(f"Unknown dataset: {name}")
    return _DATASETS[name](**kwargs)
