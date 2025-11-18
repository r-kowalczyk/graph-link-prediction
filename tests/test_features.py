"""Unit tests for pair feature construction utilities."""

import numpy as np
from graph_lp.features import pair_features, fuse_embeddings


def test_pair_features_shapes_and_modes():
    """Verify output shapes and modes for basic pair feature operators."""
    emb = np.arange(12, dtype=np.float32).reshape(4, 3)
    pairs = [(0, 1), (2, 3)]

    x = pair_features(emb, pairs, mode="concat")
    assert x.shape == (2, 6)

    x = pair_features(emb, pairs, mode="absdiff")
    assert x.shape == (2, 3)

    x = pair_features(emb, pairs, mode="product")
    assert x.shape == (2, 3)


def test_fuse_embeddings_none_and_concat():
    """Covers None-handling and concatenation shape for fused embeddings.

    Parameters:
        None

    This focuses on a common configuration switch (structural-only, semantic-only,
    or hybrid) and checks the minimal shape contract and identity where applicable.
    """
    structural = np.ones((3, 2), dtype=np.float32)
    semantic = np.zeros((3, 5), dtype=np.float32)

    # We expect concatenation along the feature axis to preserve node alignment
    fused = fuse_embeddings(structural, semantic)
    assert fused.shape == (3, 7)
    assert np.all(fused[:, :2] == 1.0) and np.all(fused[:, 2:] == 0.0)

    # Single-source passthrough behaviour is important for simpler variants
    assert fuse_embeddings(structural, None).shape == (3, 2)
    assert fuse_embeddings(None, semantic).shape == (3, 5)
