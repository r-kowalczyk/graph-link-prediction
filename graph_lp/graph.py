"""Utilities for constructing and manipulating simple NetworkX graphs for link prediction.

This module provides small, self-contained helpers to:
- build a graph from an edge list and a specified number of nodes
- remove a set of known positive edges to avoid evaluation leakage
- obtain a set view of edges that respects undirected symmetry

The functions assume integer node indices starting at 0 and are designed to be
minimal and easy to read for newcomers to graph machine learning.
"""

from typing import List, Tuple, Set

import networkx as nx


def build_graph(
    edges: List[Tuple[int, int]], num_nodes: int, undirected: bool = True
) -> nx.Graph:
    """Construct a NetworkX graph of a given size from an edge list.

    Args:
        edges: Pairs of integer node indices representing edges.
        num_nodes: Total number of nodes to include; isolated nodes are retained.
        undirected: If True, build an undirected graph; otherwise a directed one.

    Returns:
        A NetworkX Graph or DiGraph containing all nodes in range(num_nodes) and the given edges.
    """
    # Choose graph type depending on whether direction matters for your task
    g = nx.Graph() if undirected else nx.DiGraph()
    # Create all nodes up-front so nodes without edges are still present
    g.add_nodes_from(range(num_nodes))
    # Add edges verbatim; NetworkX handles duplicate suppression
    g.add_edges_from(edges)
    return g


def remove_positive_edges_from_graph(
    g: nx.Graph, pos_edges: List[Tuple[int, int]], undirected: bool = True
) -> nx.Graph:
    """Return a defensive copy of ``g`` with specified positive edges removed.

    Used to prevent evaluation leakage: we remove edges that will appear as
    positives in validation/testing so that structural features are not biased.

    Args:
        g: Source graph to copy and prune.
        pos_edges: Positive pairs to remove from the copy.
        undirected: If True, also remove the reverse pair for each edge.

    Returns:
        A new NetworkX graph without the listed positive edges.
    """
    # Work on a copy to avoid mutating the original graph used elsewhere
    h = g.copy()
    # For undirected tasks, remove both (u, v) and (v, u) if present
    for u, v in pos_edges:
        if h.has_edge(u, v):
            h.remove_edge(u, v)
        if undirected and h.has_edge(v, u):
            h.remove_edge(v, u)
    return h


def edge_set(g: nx.Graph, undirected: bool = True) -> Set[Tuple[int, int]]:
    """Return a Python ``set`` of edges, respecting undirected symmetry.

    For undirected graphs, each edge is normalised as ``(min(u, v), max(u, v))``
    so that the pair appears only once regardless of endpoint order.

    Args:
        g: The input graph.
        undirected: If True, normalise by sorting endpoints.

    Returns:
        A set of 2-tuples representing edges.
    """
    es: Set[Tuple[int, int]] = set()
    # Iterate the graph’s own edge iterator for efficiency
    for u, v in g.edges():
        es.add((min(u, v), max(u, v)) if undirected else (u, v))
    return es
