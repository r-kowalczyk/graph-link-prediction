"""Unit tests for data loading and node indexing utilities."""

import pandas as pd

from graph_lp.io import build_node_index, load_data


def test_load_data(tmp_path):
    """Test that load_data reads three CSV files and returns DataFrames."""
    nodes_file = tmp_path / "nodes.csv"
    edges_file = tmp_path / "edges.csv"
    gt_file = tmp_path / "ground_truth.csv"
    nodes_file.write_text("id,name,description\n0,Node0,Desc0\n1,Node1,Desc1")
    edges_file.write_text("subject,object\n0,1\n1,0")
    gt_file.write_text("source,target,y\n0,1,1\n1,0,0")
    nodes, edges, gt = load_data(
        str(tmp_path), "nodes.csv", "edges.csv", "ground_truth.csv"
    )
    assert isinstance(nodes, pd.DataFrame)
    assert isinstance(edges, pd.DataFrame)
    assert isinstance(gt, pd.DataFrame)
    assert len(nodes) == 2
    assert len(edges) == 2
    assert len(gt) == 2


def test_build_node_index_basic():
    """Test that build_node_index creates correct mappings and text lists."""
    nodes = pd.DataFrame(
        {
            "id": ["a", "b", "c"],
            "name": ["A", "B", "C"],
            "description": ["DescA", "DescB", "DescC"],
        }
    )
    node2idx, idx2node, node_texts = build_node_index(nodes)
    assert node2idx == {"a": 0, "b": 1, "c": 2}
    assert idx2node == ["a", "b", "c"]
    assert node_texts == ["A DescA", "B DescB", "C DescC"]


def test_build_node_index_missing_name():
    """Test that build_node_index handles missing name column."""
    nodes = pd.DataFrame({"id": ["a", "b"], "description": ["DescA", "DescB"]})
    node2idx, idx2node, node_texts = build_node_index(nodes)
    assert node_texts == ["DescA", "DescB"]


def test_build_node_index_missing_description():
    """Test that build_node_index handles missing description column."""
    nodes = pd.DataFrame({"id": ["a", "b"], "name": ["A", "B"]})
    node2idx, idx2node, node_texts = build_node_index(nodes)
    assert node_texts == ["A", "B"]


def test_build_node_index_empty_text():
    """Test that build_node_index uses 'No description' for empty text."""
    nodes = pd.DataFrame({"id": ["a", "b"]})
    node2idx, idx2node, node_texts = build_node_index(nodes)
    assert node_texts == ["No description", "No description"]


def test_build_node_index_whitespace_only():
    """Test that build_node_index handles whitespace-only text."""
    nodes = pd.DataFrame({"id": ["a"], "name": [" "], "description": [" "]})
    node2idx, idx2node, node_texts = build_node_index(nodes)
    assert node_texts == ["No description"]
