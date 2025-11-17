"""Evaluation utilities for binary link prediction.

This module computes standard classification metrics and writes ROC/PR plots
to disk. It also includes a helper to identify cold-start pairs where nodes
have little or no interaction history in the training edges.
"""

from typing import Dict, List, Tuple

import os
import numpy as np
import matplotlib.pyplot as plt

from .metrics import compute_classification_metrics, curves


def compute_and_plot(
    y_true: np.ndarray, y_prob: np.ndarray, out_dir: str, k: int, dpi: int = 120
) -> Dict[str, float]:
    """Compute core metrics and save ROC/PR curve images.

    Args:
        y_true: Binary ground-truth labels (0 or 1) for each pair.
        y_prob: Predicted probabilities for the positive class.
        out_dir: Directory where plots will be written.
        k: Cut-off for Precision@k and Recall@k.
        dpi: Output image DPI.

    Returns:
        A mapping of metric names to floats, including ROC-AUC, PR-AUC, F1, precision@k, recall@k, and Brier score.
    """
    # Ensure the output folder exists before writing files
    os.makedirs(out_dir, exist_ok=True)
    # Compute all scalar metrics first
    m = compute_classification_metrics(y_true, y_prob, k)
    # Generate curve points for ROC and PR plots
    (fpr, tpr), (prec, rec, _) = curves(y_true, y_prob)

    # Plot ROC curve: trade-off between true and false positive rates
    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.savefig(os.path.join(out_dir, "roc.png"), dpi=dpi, bbox_inches="tight")
    plt.close()

    # Plot Precision-Recall curve: more informative under class imbalance
    plt.figure()
    plt.plot(rec, prec)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR Curve")
    plt.savefig(os.path.join(out_dir, "pr.png"), dpi=dpi, bbox_inches="tight")
    plt.close()
    return m


def cold_start_mask(
    train_edges: List[Tuple[int, int]],
    pairs: List[Tuple[int, int]],
    degree_threshold: int = 1,
) -> np.ndarray:
    """Identify pairs involving low-degree nodes in the training data.

    A pair is marked True if at least one endpoint has degree less than or equal
    to ``degree_threshold`` with respect to the provided training edges.

    Args:
        train_edges: Edges considered as the training graph signal.
        pairs: Candidate node pairs for which to compute the mask.
        degree_threshold: Maximum degree to be considered cold-start.

    Returns:
        A boolean NumPy array of shape (len(pairs),) where True denotes a cold-start pair.
    """
    from collections import Counter

    # Count occurrences of nodes in the training edges as a simple degree proxy
    cnt = Counter()
    for u, v in train_edges:
        cnt[u] += 1
        cnt[v] += 1
    # Mark pairs where any endpoint is low-degree
    mask = np.zeros(len(pairs), dtype=bool)
    for i, (u, v) in enumerate(pairs):
        if cnt[u] <= degree_threshold or cnt[v] <= degree_threshold:
            mask[i] = True
    return mask
