"""Data loading helpers for node/edge CSVs and text fields.

This module provides small utilities to read CSV files from a directory and to
prepare node identifiers and text fields that can be fed into text embedding models.
It assumes a simple schema with columns ``id``, ``name`` and ``description`` for nodes.
"""

from typing import Dict, Tuple, List, Any

import os
import pandas as pd


def load_data(
    data_dir: str, nodes_csv: str, edges_csv: str, ground_truth_csv: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load node, edge and ground-truth CSV files from a directory.

    Args:
        data_dir: Base directory containing the CSV files.
        nodes_csv: File name of the nodes table.
        edges_csv: File name of the edges table.
        ground_truth_csv: File name of the ground-truth pairs table.

    Returns:
        Three pandas DataFrames: nodes, edges, ground truth.
    """
    # Read the three CSV files from the specified directory
    nodes = pd.read_csv(os.path.join(data_dir, nodes_csv))
    edges = pd.read_csv(os.path.join(data_dir, edges_csv))
    gt = pd.read_csv(os.path.join(data_dir, ground_truth_csv))
    return nodes, edges, gt


def build_node_index(
    nodes: pd.DataFrame,
) -> Tuple[Dict[str, int], List[Any], List[str]]:
    """Create mappings from node identifiers to indices and aggregate text.

    The function expects at least an ``id`` column, and optionally ``name`` and
    ``description`` columns, which are concatenated to form per-node text.

    Args:
        nodes: DataFrame containing node metadata.

    Returns:
        A tuple of:
        - node2idx: mapping of original ids (as strings) to contiguous indices
        - idx2node: list of original ids in index order
        - node_texts: list of combined name/description per node
    """
    # Preserve original id order to create a stable contiguous index
    unique_ids = nodes["id"].tolist()
    node2idx = {nid: i for i, nid in enumerate(unique_ids)}
    idx2node = [nid for nid in unique_ids]

    node_texts: List[str] = []
    for _, row in nodes.iterrows():
        name = str(row.get("name", ""))
        desc = str(row.get("description", ""))
        # Combine name and description; this is what the text embedding model will see
        text = (name + " " + desc).strip()
        if not text:
            text = "No description"
        node_texts.append(text)

    return node2idx, idx2node, node_texts
