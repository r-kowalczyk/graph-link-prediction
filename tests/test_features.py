"""Unit tests for pair feature construction utilities."""

import numpy as np
from graph_lp.features import pair_features


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
