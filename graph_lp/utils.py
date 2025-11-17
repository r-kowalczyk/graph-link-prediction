"""Small utility functions for device selection, seeding, filesystem and hashing.

These helpers are used across the training and evaluation pipeline to keep the
core logic clean and focused.
"""

import hashlib
import json
import os
import random
from datetime import datetime
from typing import Any, Dict, Tuple

import numpy as np
import torch


def resolve_device(device: str) -> torch.device:
    """Return a ``torch.device`` from a simple policy string.

    Args:
        device: Either ``\"cpu\"``, ``\"cuda\"`` or ``\"auto\"`` for best available.

    Returns:
        A ``torch.device`` instance.
    """
    if device == "auto":
        # Prefer CUDA (e.g., Colab GPU), then Apple Silicon MPS, finally CPU
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def set_all_seeds(seed: int) -> None:
    """Set seeds for Python, NumPy and PyTorch to improve reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)


def time_stamp() -> str:
    """Return an ISO-like UTC timestamp for naming artefact directories."""
    return datetime.utcnow().strftime("%Y%m%d-%H%M%S")


def ensure_dir(path: str) -> None:
    """Create a directory if it does not already exist."""
    os.makedirs(path, exist_ok=True)


def write_json(path: str, obj: Dict[str, Any]) -> None:
    """Write a mapping as JSON to ``path`` with stable formatting."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def save_yaml_copy(path: str, content: str) -> None:
    """Persist the YAML text used to configure a run."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def cache_key_from_paths_and_config(
    paths: Tuple[str, ...], config: Dict[str, Any]
) -> str:
    """Compute a short cache key derived from file contents and configuration.

    Args:
        paths: Sequence of file paths whose contents should influence the key.
        config: Dictionary of configuration values that should influence the key.

    Returns:
        A short hexadecimal string suitable for file names.
    """
    dig = hashlib.sha1()
    for p in paths:
        with open(p, "rb") as f:
            dig.update(f.read())
    dig.update(json.dumps(config, sort_keys=True, separators=(",", ":")).encode())
    return dig.hexdigest()[:12]
