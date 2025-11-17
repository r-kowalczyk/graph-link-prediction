"""Tests that we remove validation/test positives from the training graph to avoid leakage."""

from graph_lp.graph import build_graph, remove_positive_edges_from_graph, edge_set


def test_remove_positive_edges_prevents_leakage():
    """Edges used as positives in validation/test should not remain in the training graph."""
    # small toy graph
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (1, 3)]
    g = build_graph(edges, num_nodes=4, undirected=True)
    val_pos = [(1, 2)]
    test_pos = [(0, 3)]
    g_train = remove_positive_edges_from_graph(g, val_pos + test_pos, undirected=True)

    es = edge_set(g_train, undirected=True)
    assert (1, 2) not in es
    assert (0, 3) not in es
