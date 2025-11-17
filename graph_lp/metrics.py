"""Classification metrics for evaluating link predictors.

Provides scalar summary metrics and the coordinates for ROC and Precision–Recall
curves, using scikit-learn implementations where appropriate.
"""

from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_curve,
    brier_score_loss,
)


def compute_classification_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, k: int
) -> Dict[str, float]:
    """Compute common binary classification metrics.

    Includes ROC-AUC, PR-AUC, F1 at threshold 0.5, Precision@k, Recall@k and Brier score.

    Args:
        y_true: Binary ground-truth labels.
        y_prob: Predicted probabilities for the positive class.
        k: Number of top-probability items to consider for @k metrics.

    Returns:
        A dictionary mapping metric names to floats.
    """
    # Convert probabilities to hard labels at the conventional 0.5 threshold
    y_pred = (y_prob >= 0.5).astype(int)
    # Compute indices of top-k probabilities for ranking-based metrics
    order = np.argsort(-y_prob)
    topk = order[:k]
    prec_at_k = float(y_true[topk].mean()) if k > 0 else 0.0
    rec_at_k = float(y_true[topk].sum() / (y_true.sum() + 1e-8)) if k > 0 else 0.0
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob))
        if len(np.unique(y_true)) > 1
        else 0.5,
        "pr_auc": float(average_precision_score(y_true, y_prob))
        if len(np.unique(y_true)) > 1
        else y_true.mean(),
        "f1": float(f1_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else 0.0,
        "precision_at_k": prec_at_k,
        "recall_at_k": rec_at_k,
        "brier": float(brier_score_loss(y_true, y_prob)),
    }


def curves(
    y_true: np.ndarray, y_prob: np.ndarray
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Compute coordinates for ROC and Precision–Recall curves.

    Args:
        y_true: Binary ground-truth labels.
        y_prob: Predicted probabilities for the positive class.

    Returns:
        A tuple: ((fpr, tpr), (precision, recall, thresholds)).
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    return (fpr, tpr), (prec, rec, _)
