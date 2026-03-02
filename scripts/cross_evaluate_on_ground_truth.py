"""Cross-evaluate a trained GraphSAGE model on the baseline's ground_truth.csv test pairs.

The baseline pipeline and GraphSAGE pipeline use different evaluation protocols:
the baseline scores pairs from ground_truth.csv with predefined negatives, while
GraphSAGE scores held-out edges from RandomLinkSplit with sampled negatives. This
makes their reported AUC numbers incomparable.

This script loads a trained GraphSAGE model and scores the exact same
ground_truth.csv test pairs that the baseline uses, producing directly
comparable metrics. It also computes a parameter-free semantic cosine
similarity baseline as a reference point, so you can immediately see whether
GraphSAGE message passing contributes structural signal beyond raw text
embedding similarity.

Usage:
    python scripts/cross_evaluate_on_ground_truth.py \
        --config configs/quickstart.yaml \
        --run-dir artifacts_quickstart/2025-01-01_12-00-00
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import yaml
from sklearn.model_selection import train_test_split

from graph_lp.graph import build_graph, remove_positive_edges_from_graph
from graph_lp.graphsage import GraphSageLinkPredictor
from graph_lp.io import build_node_index, load_data
from graph_lp.metrics import compute_classification_metrics
from graph_lp.utils import set_all_seeds


def _resolve_edge_column_names(edges_table) -> tuple[str, str]:
    """Identify the source and target column names in the edges table.

    Some datasets use 'subject'/'object' columns while others use
    'start_id'/'end_id'. This resolves whichever pair is present so the
    script works with both naming conventions without manual configuration.

    Parameters:
        edges_table: Pandas DataFrame containing the loaded edges CSV.

    Returns:
        A tuple of (source_column_name, target_column_name).
    """
    if {"subject", "object"}.issubset(edges_table.columns):
        return "subject", "object"
    if {"start_id", "end_id"}.issubset(edges_table.columns):
        return "start_id", "end_id"
    raise ValueError(
        "Edges CSV must contain either subject/object or start_id/end_id columns."
    )


def _cosine_similarity_scores(
    node_features: np.ndarray,
    pairs: list[tuple[int, int]],
) -> np.ndarray:
    """Compute cosine similarity for each node pair using raw semantic embeddings.

    This provides a parameter-free text-similarity baseline that requires no
    training or graph structure. If GraphSAGE scores higher than this on the
    same test pairs, message passing is contributing structural information
    that goes beyond what text embeddings alone provide.

    Parameters:
        node_features: Semantic embedding matrix with shape (num_nodes, embedding_dim).
        pairs: List of (source_index, target_index) tuples to score.

    Returns:
        Array of cosine similarity values in [-1, 1], one per pair.
    """
    source_indices = [pair[0] for pair in pairs]
    target_indices = [pair[1] for pair in pairs]
    source_embeddings = node_features[source_indices]
    target_embeddings = node_features[target_indices]
    # Row-wise dot product divided by the product of L2 norms gives cosine similarity.
    dot_products = np.sum(source_embeddings * target_embeddings, axis=1)
    source_norms = np.linalg.norm(source_embeddings, axis=1)
    target_norms = np.linalg.norm(target_embeddings, axis=1)
    return dot_products / (source_norms * target_norms + 1e-8)


def _print_metrics_block(heading: str, metrics: dict[str, float]) -> None:
    """Print a named block of classification metrics to stdout.

    Each metric is printed on its own indented line beneath the heading.
    The heading is indented once and each metric line twice, so the output
    nests visually under the section banner printed by the caller.

    Parameters:
        heading: Label identifying which model and split these metrics describe.
        metrics: Dictionary mapping metric names to their float values.
    """
    print(f"\n  {heading}")
    for metric_name, metric_value in metrics.items():
        print(f"    {metric_name}: {metric_value:.4f}")


def _replicate_baseline_ground_truth_split(
    ground_truth_table,
    node_id_to_index: dict[str, int],
    validation_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[
    list[tuple[int, int]],
    np.ndarray,
    list[tuple[int, int]],
    np.ndarray,
    list[tuple[int, int]],
    np.ndarray,
]:
    """Replicate the exact train/val/test split that the baseline pipeline creates.

    The baseline in train.py splits ground_truth.csv pairs using sklearn's
    train_test_split with stratification and a fixed seed. This function
    reproduces that logic identically so the test pairs here are guaranteed
    to match the ones the baseline was evaluated on.

    Parameters:
        ground_truth_table: Pandas DataFrame with 'source', 'target', 'y' columns.
        node_id_to_index: Mapping from string node identifiers to integer indices.
        validation_ratio: Fraction of pairs held out for validation.
        test_ratio: Fraction of pairs held out for testing.
        seed: Random seed for reproducible splitting.

    Returns:
        A six-element tuple of (train_pairs, train_labels, validation_pairs,
        validation_labels, test_pairs, test_labels).
    """
    ground_truth_table = ground_truth_table.copy()
    ground_truth_table["source"] = ground_truth_table["source"].astype(str)
    ground_truth_table["target"] = ground_truth_table["target"].astype(str)

    positive_pairs = [
        (node_id_to_index[source_id], node_id_to_index[target_id])
        for source_id, target_id in ground_truth_table[ground_truth_table["y"] == 1][
            ["source", "target"]
        ].values
    ]
    negative_pairs = [
        (node_id_to_index[source_id], node_id_to_index[target_id])
        for source_id, target_id in ground_truth_table[ground_truth_table["y"] == 0][
            ["source", "target"]
        ].values
    ]
    all_pairs = positive_pairs + negative_pairs
    all_labels = np.array(
        [1] * len(positive_pairs) + [0] * len(negative_pairs), dtype=int
    )

    # First split: train vs (validation + test), stratified by label.
    train_pairs, remaining_pairs, train_labels, remaining_labels = train_test_split(
        all_pairs,
        all_labels,
        test_size=(validation_ratio + test_ratio),
        stratify=all_labels,
        random_state=seed,
    )

    # Second split: validation vs test from the remainder.
    relative_test_ratio = (
        test_ratio / (validation_ratio + test_ratio)
        if (validation_ratio + test_ratio) > 0
        else 0.5
    )
    validation_pairs, test_pairs, validation_labels, test_labels = train_test_split(
        remaining_pairs,
        remaining_labels,
        test_size=relative_test_ratio,
        stratify=remaining_labels,
        random_state=seed,
    )

    return (
        train_pairs,
        train_labels,
        validation_pairs,
        validation_labels,
        test_pairs,
        test_labels,
    )


def _build_leakage_safe_edge_index(
    edges_table,
    node_id_to_index: dict[str, int],
    is_undirected: bool,
    validation_positive_pairs: list[tuple[int, int]],
    test_positive_pairs: list[tuple[int, int]],
) -> tuple[torch.Tensor, int]:
    """Build a PyTorch edge_index tensor with ground_truth test edges removed.

    The baseline pipeline removes validation and test positive edges from the
    graph before computing Node2Vec embeddings, so structural features never
    see test-time interactions. This function applies the same removal to the
    edges.csv graph before GraphSAGE message passing, making the comparison
    fair: neither model gets to pass information through test edges.

    Parameters:
        edges_table: Pandas DataFrame containing the raw edges CSV.
        node_id_to_index: Mapping from string node identifiers to integer indices.
        is_undirected: Whether reverse edges should be added for message passing.
        validation_positive_pairs: Positive val pairs to remove from the graph.
        test_positive_pairs: Positive test pairs to remove from the graph.

    Returns:
        A tuple of (edge_index tensor, number of edges removed).
    """
    edge_source_column, edge_target_column = _resolve_edge_column_names(edges_table)
    edge_rows = edges_table.copy()
    edge_rows[edge_source_column] = edge_rows[edge_source_column].astype(str)
    edge_rows[edge_target_column] = edge_rows[edge_target_column].astype(str)
    edge_index_pairs = [
        (node_id_to_index[source_id], node_id_to_index[target_id])
        for source_id, target_id in edge_rows[
            [edge_source_column, edge_target_column]
        ].values
    ]

    full_graph = build_graph(
        edge_index_pairs, num_nodes=len(node_id_to_index), undirected=is_undirected
    )
    leakage_safe_graph = remove_positive_edges_from_graph(
        full_graph,
        validation_positive_pairs + test_positive_pairs,
        undirected=is_undirected,
    )

    removed_count = full_graph.number_of_edges() - leakage_safe_graph.number_of_edges()

    # Convert NetworkX edges to a COO edge_index tensor for PyTorch Geometric.
    safe_edge_list = list(leakage_safe_graph.edges())
    if is_undirected:
        # NetworkX stores each undirected edge once; add reverse edges explicitly
        # so GraphSAGE message passing propagates in both directions.
        reverse_edges = [(target, source) for source, target in safe_edge_list]
        safe_edge_list = safe_edge_list + reverse_edges
    if safe_edge_list:
        safe_edge_array = np.array(safe_edge_list, dtype=np.int64)
        safe_edge_index = torch.as_tensor(safe_edge_array.T, dtype=torch.long)
    else:
        safe_edge_index = torch.empty((2, 0), dtype=torch.long)

    return safe_edge_index, removed_count


def _load_graphsage_model(
    run_directory: str,
) -> tuple[GraphSageLinkPredictor, np.ndarray]:
    """Load a trained GraphSAGE model and its node features from a run directory.

    The run directory is produced by train_graphsage_model and contains the
    model weights, metadata JSON, and the semantic node feature matrix. This
    function reconstructs the exact architecture from metadata and loads the
    trained weights, returning the model in evaluation mode.

    Parameters:
        run_directory: Path to the GraphSAGE run artefact directory.

    Returns:
        A tuple of (model in eval mode, node feature matrix as numpy array).
    """
    metadata_path = os.path.join(run_directory, "graphsage_metadata.json")
    with open(metadata_path, "r", encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)

    model_configuration = metadata["model"]
    model = GraphSageLinkPredictor(
        input_dimension=int(model_configuration["input_dim"]),
        hidden_dimension=int(model_configuration["hidden_dim"]),
        output_dimension=int(model_configuration["output_dim"]),
        dropout_rate=float(model_configuration["dropout"]),
        decoder_type=str(model_configuration.get("decoder_type", "mlp")),
        decoder_hidden_dimension=int(model_configuration.get("decoder_hidden_dim", 64)),
        num_layers=int(model_configuration.get("num_layers", 2)),
    )
    model_state_path = os.path.join(run_directory, "graphsage_model_state.pt")
    state_dictionary = torch.load(
        model_state_path, map_location="cpu", weights_only=False
    )
    model.load_state_dict(state_dictionary)
    model.eval()

    node_features_numpy = np.load(
        os.path.join(run_directory, "graphsage_node_features.npy")
    )
    return model, node_features_numpy


def main() -> None:
    """Load a trained GraphSAGE model and evaluate it on ground_truth.csv test pairs.

    This is the main entry point for the cross-evaluation script. It replicates
    the baseline pipeline's data split, builds a leakage-safe message-passing
    graph, scores both validation and test pairs with the trained GraphSAGE
    model, and compares the results against a semantic cosine similarity
    baseline to quantify the value of graph message passing.

    The script prints a summary table showing ROC-AUC for GraphSAGE vs
    semantic cosine on the same test pairs, along with the delta.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Cross-evaluate a trained GraphSAGE model on the same "
            "ground_truth.csv test pairs used by the baseline pipeline."
        ),
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the YAML config file used for training",
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="GraphSAGE run directory containing model state and metadata",
    )
    arguments = parser.parse_args()

    with open(arguments.config, "r", encoding="utf-8") as config_file:
        configuration = yaml.safe_load(config_file)

    seed = int(configuration["seed"])
    set_all_seeds(seed)
    is_undirected = bool(configuration["data"]["undirected"])
    precision_at_k = int(configuration["metrics"]["precision_at_k"])
    validation_ratio = float(configuration["splits"]["val_ratio"])
    test_ratio = float(configuration["splits"]["test_ratio"])

    # ---- Load data tables and build the shared node index ----
    nodes_table, edges_table, ground_truth_table = load_data(
        configuration["data"]["dir"],
        configuration["data"]["nodes_csv"],
        configuration["data"]["edges_csv"],
        configuration["data"]["ground_truth_csv"],
    )
    node_id_to_index, _index_to_node_id, _node_texts = build_node_index(nodes_table)

    # ---- Replicate the baseline's ground_truth split ----
    (
        _train_pairs,
        _train_labels,
        validation_pairs,
        validation_labels,
        test_pairs,
        test_labels,
    ) = _replicate_baseline_ground_truth_split(
        ground_truth_table=ground_truth_table,
        node_id_to_index=node_id_to_index,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    print(
        f"Ground truth split sizes: "
        f"train={len(_train_pairs)}, "
        f"val={len(validation_pairs)}, "
        f"test={len(test_pairs)}"
    )
    print(
        f"Test set composition: "
        f"{int(np.sum(test_labels))} positives, "
        f"{int(np.sum(test_labels == 0))} negatives"
    )

    # ---- Build a leakage-safe message-passing graph ----
    validation_positive_pairs = [
        pair for pair, label in zip(validation_pairs, validation_labels) if label == 1
    ]
    test_positive_pairs = [
        pair for pair, label in zip(test_pairs, test_labels) if label == 1
    ]
    safe_edge_index, removed_edge_count = _build_leakage_safe_edge_index(
        edges_table=edges_table,
        node_id_to_index=node_id_to_index,
        is_undirected=is_undirected,
        validation_positive_pairs=validation_positive_pairs,
        test_positive_pairs=test_positive_pairs,
    )
    print(
        f"Message-passing edges: {safe_edge_index.size(1)} "
        f"({removed_edge_count} removed for leakage prevention)"
    )

    # ---- Load the trained GraphSAGE model ----
    model, node_features_numpy = _load_graphsage_model(arguments.run_dir)
    node_features = torch.as_tensor(node_features_numpy, dtype=torch.float32)

    # ---- Score ground_truth pairs with GraphSAGE ----
    with torch.no_grad():
        # Encode all nodes using the leakage-safe graph for message passing.
        node_embeddings = model.encode(
            node_features=node_features,
            edge_index=safe_edge_index,
        )

        # Decode validation pairs.
        validation_edge_index = torch.tensor(
            [
                [pair[0] for pair in validation_pairs],
                [pair[1] for pair in validation_pairs],
            ],
            dtype=torch.long,
        )
        validation_logits = model.decode(node_embeddings, validation_edge_index)
        validation_probabilities = torch.sigmoid(validation_logits).numpy()

        # Decode test pairs.
        test_edge_index = torch.tensor(
            [
                [pair[0] for pair in test_pairs],
                [pair[1] for pair in test_pairs],
            ],
            dtype=torch.long,
        )
        test_logits = model.decode(node_embeddings, test_edge_index)
        test_probabilities = torch.sigmoid(test_logits).numpy()

    # ---- Compute semantic cosine similarity baseline ----
    # Cosine similarity uses only the raw text embeddings with no graph structure,
    # so it isolates how much of the prediction signal comes from text alone.
    validation_cosine_scores = _cosine_similarity_scores(
        node_features_numpy, validation_pairs
    )
    test_cosine_scores = _cosine_similarity_scores(node_features_numpy, test_pairs)
    # Cosine similarity lies in [-1, 1]; rescale to [0, 1] for metric computation
    # because compute_classification_metrics expects probability-like scores.
    validation_cosine_probabilities = (validation_cosine_scores + 1.0) / 2.0
    test_cosine_probabilities = (test_cosine_scores + 1.0) / 2.0

    # ---- Compute and print all metrics ----
    print("\n=== Cross-evaluation on ground_truth.csv pairs ===")

    graphsage_validation_metrics = compute_classification_metrics(
        np.asarray(validation_labels), validation_probabilities, precision_at_k
    )
    graphsage_test_metrics = compute_classification_metrics(
        np.asarray(test_labels), test_probabilities, precision_at_k
    )
    cosine_validation_metrics = compute_classification_metrics(
        np.asarray(validation_labels),
        validation_cosine_probabilities,
        precision_at_k,
    )
    cosine_test_metrics = compute_classification_metrics(
        np.asarray(test_labels), test_cosine_probabilities, precision_at_k
    )

    _print_metrics_block("GraphSAGE (validation)", graphsage_validation_metrics)
    _print_metrics_block("GraphSAGE (test)", graphsage_test_metrics)
    _print_metrics_block("Semantic cosine (validation)", cosine_validation_metrics)
    _print_metrics_block("Semantic cosine (test)", cosine_test_metrics)

    # Print a compact summary comparing the two approaches on the test split.
    print("\n=== Summary: test ROC-AUC ===")
    print(f"  GraphSAGE:        {graphsage_test_metrics['roc_auc']:.4f}")
    print(f"  Semantic cosine:  {cosine_test_metrics['roc_auc']:.4f}")
    graphsage_minus_cosine_delta = (
        graphsage_test_metrics["roc_auc"] - cosine_test_metrics["roc_auc"]
    )
    print(f"  Delta (GS - cos): {graphsage_minus_cosine_delta:+.4f}")
    if graphsage_minus_cosine_delta > 0:
        print("  Message passing adds signal beyond text similarity.")
    else:
        print("  Message passing does not improve over text similarity on this split.")


if __name__ == "__main__":
    main()
