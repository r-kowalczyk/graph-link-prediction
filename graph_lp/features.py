"""Feature construction helpers to turn node embeddings into pair features.

Two utilities are provided:
- ``fuse_embeddings``: combine structural and semantic node embeddings
- ``pair_features``: derive pairwise features from per-node embeddings
"""

from typing import List, Tuple

import numpy as np


def fuse_embeddings(
    structural: np.ndarray | None, semantic: np.ndarray | None
) -> np.ndarray:
    """Fuse structural and semantic embeddings if available.

    Args:
        structural: Array of shape (N, Ds) or None.
        semantic: Array of shape (N, Dm) or None.

    Returns:
        Array of shape (N, Ds + Dm) if both are provided, otherwise the single input.

    Raises:
        ValueError: If both inputs are None.
    """
    if structural is None and semantic is None:
        raise ValueError("At least one embedding must be provided")
    if structural is None:
        return semantic  # type: ignore[return-value]
    if semantic is None:
        return structural
    # Concatenate along the feature axis to preserve node alignment
    return np.concatenate([structural, semantic], axis=1)


def pair_features(
    emb: np.ndarray, pairs: List[Tuple[int, int]], mode: str = "concat"
) -> np.ndarray:
    """Construct pairwise features from node embeddings.

    For undirected graphs, features should not depend on node order. The
    provided modes are simple and commonly used baselines.

    Args:
        emb: Node embeddings of shape (N, D).
        pairs: List of 2-tuples of node indices.
        mode: One of ``\"concat\"``, ``\"absdiff\"``, ``\"product\"``.

    Returns:
        A NumPy array with one feature vector per pair.

    Raises:
        ValueError: If an unknown mode is passed.
    """
    # Gather per-endpoint embeddings
    u = emb[[i for i, _ in pairs]]
    v = emb[[j for _, j in pairs]]
    if mode == "concat":
        return np.concatenate([u, v], axis=1)
    if mode == "absdiff":
        return np.abs(u - v)
    if mode == "product":
        return u * v
    raise ValueError(f"Unknown pair mode: {mode}")
