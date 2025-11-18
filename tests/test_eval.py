"""Unit tests for evaluation utilities including plotting and cold-start analysis."""

import os
import numpy as np
from graph_lp.eval import compute_and_plot, cold_start_mask


def test_compute_and_plot_writes_curves_and_returns_metrics(tmp_path):
    """Ensures ROC/PR images are written and metrics dict is returned.

    Parameters:
        tmp_path: pytest-provided temporary directory for output files.

    This validates the I/O contract of the plotting helper, which is critical for
    downstream automation and artefact tracking.
    """
    y_true = np.array([0, 1, 1, 0], dtype=int)
    y_prob = np.array([0.1, 0.9, 0.8, 0.2], dtype=float)
    out_dir = tmp_path.as_posix()

    m = compute_and_plot(y_true, y_prob, out_dir, k=2, dpi=80)

    # We assert key outputs exist without over-constraining the plotting details
    assert os.path.isfile(os.path.join(out_dir, "roc.png"))
    assert os.path.isfile(os.path.join(out_dir, "pr.png"))
    assert isinstance(m, dict) and "roc_auc" in m and "pr_auc" in m


def test_cold_start_mask_marks_low_degree_pairs():
    """Flags pairs where any endpoint has degree at or below the threshold.

    Parameters:
        None

    The toy graph degrees are chosen so we can predict booleans by inspection and
    ensure the mask logic is correct.
    """
    train_edges = [(0, 1), (1, 2), (2, 3)]  # degrees: 0->1, 1->2, 2->2, 3->1
    pairs = [(0, 2), (1, 3), (2, 3), (1, 2)]
    mask = cold_start_mask(train_edges, pairs, degree_threshold=1)

    # We check exact positions to avoid ambiguous shape-only assertions
    assert mask.tolist() == [True, True, True, False]
