"""Unit tests for graph construction and manipulation utilities."""

import networkx as nx

from graph_lp.graph import build_graph, edge_set, remove_positive_edges_from_graph


def test_build_graph_undirected():
    """Test that build_graph creates an undirected graph with all nodes."""
    edges = [(0, 1), (1, 2), (2, 0)]
    g = build_graph(edges, num_nodes=4, undirected=True)
    assert isinstance(g, nx.Graph)
    assert g.number_of_nodes() == 4
    assert g.number_of_edges() == 3
    assert (0, 1) in g.edges() or (1, 0) in g.edges()


def test_build_graph_directed():
    """Test that build_graph creates a directed graph."""
    edges = [(0, 1), (1, 2)]
    g = build_graph(edges, num_nodes=3, undirected=False)
    assert isinstance(g, nx.DiGraph)
    assert g.number_of_nodes() == 3
    assert g.number_of_edges() == 2
    assert (0, 1) in g.edges()
    assert (1, 2) in g.edges()


def test_build_graph_isolated_nodes():
    """Test that build_graph includes isolated nodes."""
    edges = [(0, 1)]
    g = build_graph(edges, num_nodes=3, undirected=True)
    assert g.number_of_nodes() == 3
    assert 2 in g.nodes()


def test_remove_positive_edges_from_graph_undirected():
    """Test that remove_positive_edges_from_graph removes edges in undirected mode."""
    edges = [(0, 1), (1, 2), (2, 3)]
    g = build_graph(edges, num_nodes=4, undirected=True)
    pos_edges = [(1, 2)]
    g_train = remove_positive_edges_from_graph(g, pos_edges, undirected=True)
    assert g_train.number_of_edges() == 2
    assert not g_train.has_edge(1, 2)
    assert not g_train.has_edge(2, 1)


def test_remove_positive_edges_from_graph_removes_reverse_edge_when_present():
    """Test that reverse edges are removed when using a directed graph with undirected removal."""
    edges = [(0, 1), (1, 0), (1, 2)]
    g = build_graph(edges, num_nodes=3, undirected=False)
    pos_edges = [(0, 1)]
    g_train = remove_positive_edges_from_graph(g, pos_edges, undirected=True)
    assert not g_train.has_edge(0, 1)
    assert not g_train.has_edge(1, 0)


def test_remove_positive_edges_from_graph_directed():
    """Test that remove_positive_edges_from_graph removes edges in directed mode."""
    edges = [(0, 1), (1, 2), (2, 3)]
    g = build_graph(edges, num_nodes=4, undirected=False)
    pos_edges = [(1, 2)]
    g_train = remove_positive_edges_from_graph(g, pos_edges, undirected=False)
    assert g_train.number_of_edges() == 2
    assert not g_train.has_edge(1, 2)
    assert g_train.has_edge(0, 1)
    assert g_train.has_edge(2, 3)


def test_remove_positive_edges_from_graph_nonexistent():
    """Test that remove_positive_edges_from_graph handles non-existent edges gracefully."""
    edges = [(0, 1), (1, 2)]
    g = build_graph(edges, num_nodes=3, undirected=True)
    pos_edges = [(0, 2)]
    g_train = remove_positive_edges_from_graph(g, pos_edges, undirected=True)
    assert g_train.number_of_edges() == 2


def test_edge_set_undirected():
    """Test that edge_set normalises edges for undirected graphs."""
    edges = [(0, 1), (2, 3), (1, 0)]
    g = build_graph(edges, num_nodes=4, undirected=True)
    es = edge_set(g, undirected=True)
    assert (0, 1) in es
    assert (1, 0) not in es
    assert (2, 3) in es
    assert len(es) == 2


def test_edge_set_directed():
    """Test that edge_set preserves direction for directed graphs."""
    edges = [(0, 1), (1, 0), (2, 3)]
    g = build_graph(edges, num_nodes=4, undirected=False)
    es = edge_set(g, undirected=False)
    assert (0, 1) in es
    assert (1, 0) in es
    assert (2, 3) in es
    assert len(es) == 3


def test_edge_set_empty():
    """Test that edge_set returns an empty set for a graph with no edges."""
    g = build_graph([], num_nodes=3, undirected=True)
    es = edge_set(g, undirected=True)
    assert es == set()
