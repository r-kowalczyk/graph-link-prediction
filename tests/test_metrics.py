"""Unit tests for classification metrics computation."""

import numpy as np
import pytest

from graph_lp.metrics import compute_classification_metrics, curves


def test_compute_classification_metrics_on_perfect_ranking():
    """Checks metric values on a tiny example with perfect ranking.

    Parameters:
        None

    This verifies the most important aggregate metrics (ROC-AUC, PR-AUC, F1,
    Precision@k, Recall@k, Brier) on a hand-crafted example to anchor correctness.
    """
    y_true = np.array([0, 1, 1, 0], dtype=int)
    y_prob = np.array([0.1, 0.9, 0.8, 0.2], dtype=float)
    m = compute_classification_metrics(y_true, y_prob, k=2)

    # This case is chosen because it is perfectly ranked, so closed-form checks are easy
    # Use approximate equality for floating point comparisons to handle numerical precision
    assert m["roc_auc"] == pytest.approx(1.0, rel=1e-6, abs=1e-8)
    assert m["pr_auc"] == pytest.approx(1.0, rel=1e-6, abs=1e-8)
    assert m["f1"] == pytest.approx(1.0, rel=1e-6, abs=1e-8)
    assert m["precision_at_k"] == pytest.approx(1.0, rel=1e-6, abs=1e-8)
    assert m["recall_at_k"] == pytest.approx(1.0, rel=1e-6, abs=1e-8)
    assert m["brier"] == pytest.approx(0.025, rel=1e-6, abs=1e-8)


def test_compute_classification_metrics_on_constant_labels():
    """Checks defined behaviours when y_true is constant.

    Parameters:
        None

    This ensures default values are used in degenerate cases, matching the
    function's documented policy.
    """
    y_true = np.zeros(4, dtype=int)
    y_prob = np.array([0.1, 0.2, 0.3, 0.4], dtype=float)
    m = compute_classification_metrics(y_true, y_prob, k=2)

    # These behaviours are explicitly encoded in the function for degenerate labels
    assert m["roc_auc"] == 0.5
    assert m["pr_auc"] == 0.0
    assert m["f1"] == 0.0


def test_compute_classification_metrics_all_positive():
    """Test metrics when all labels are positive."""
    y_true = np.ones(4, dtype=int)
    y_prob = np.array([0.1, 0.2, 0.3, 0.4], dtype=float)
    m = compute_classification_metrics(y_true, y_prob, k=2)
    assert m["roc_auc"] == 0.5
    assert m["pr_auc"] == pytest.approx(1.0, rel=1e-6)
    assert m["f1"] == 0.0


def test_compute_classification_metrics_k_zero():
    """Test that k=0 returns zero for precision_at_k and recall_at_k."""
    y_true = np.array([0, 1, 1, 0], dtype=int)
    y_prob = np.array([0.1, 0.9, 0.8, 0.2], dtype=float)
    m = compute_classification_metrics(y_true, y_prob, k=0)
    assert m["precision_at_k"] == 0.0
    assert m["recall_at_k"] == 0.0


def test_curves():
    """Test that curves returns ROC and PR curve coordinates."""
    y_true = np.array([0, 1, 1, 0], dtype=int)
    y_prob = np.array([0.1, 0.9, 0.8, 0.2], dtype=float)
    (fpr, tpr), (prec, rec, _) = curves(y_true, y_prob)
    assert len(fpr) == len(tpr)
    assert len(prec) == len(rec)
    assert all(0 <= x <= 1 for x in fpr)
    assert all(0 <= x <= 1 for x in tpr)
    assert all(0 <= x <= 1 for x in prec)
    assert all(0 <= x <= 1 for x in rec)
