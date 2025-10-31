"""Utilities for model serialization."""

import pickle
import torch


def save_model(model, path):
    """Save model to file."""
    torch.save(model.state_dict(), path)


def load_model(model, path):
    """Load model from file."""
    model.load_state_dict(torch.load(path))
    return model


def save_config(config, path):
    """Save configuration to file."""
    with open(path, "wb") as f:
        pickle.dump(config, f)


def load_config(path):
    """Load configuration from file."""
    with open(path, "rb") as f:
        return pickle.load(f)

