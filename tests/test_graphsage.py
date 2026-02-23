"""Unit tests for the GraphSAGE backend data, training, and bundle helpers."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from graph_lp.graphsage import (
    SUPPORTED_DECODER_TYPES,
    BilinearLinkDecoder,
    DotProductLinkDecoder,
    GraphSageLinkPredictor,
    MLPLinkDecoder,
    _build_decoder,
    build_graph_data,
    build_negative_edge_labels,
    export_graphsage_bundle,
    load_graphsage_bundle,
    split_link_prediction_data,
    train_graphsage_model,
)


def _create_toy_graph_data():
    """Create a tiny graph with semantic features for GraphSAGE tests."""

    node_feature_matrix = np.array(
        [
            [1.0, 0.0, 0.1, 0.2],
            [0.9, 0.2, 0.0, 0.1],
            [0.0, 1.0, 0.2, 0.1],
            [0.1, 0.8, 0.1, 0.0],
        ],
        dtype=np.float32,
    )
    edge_pairs = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]
    return build_graph_data(
        node_feature_matrix=node_feature_matrix,
        edge_pairs=edge_pairs,
        is_undirected=True,
    )


def test_build_decoder_creates_correct_decoder_types():
    """The decoder factory should return the correct module for each supported type."""

    dot_decoder = _build_decoder(
        "dot_product", embedding_dimension=8, decoder_hidden_dimension=16
    )
    assert isinstance(dot_decoder, DotProductLinkDecoder)

    bilinear_decoder = _build_decoder(
        "bilinear", embedding_dimension=8, decoder_hidden_dimension=16
    )
    assert isinstance(bilinear_decoder, BilinearLinkDecoder)

    mlp_decoder = _build_decoder(
        "mlp", embedding_dimension=8, decoder_hidden_dimension=16
    )
    assert isinstance(mlp_decoder, MLPLinkDecoder)


def test_build_decoder_rejects_unsupported_type():
    """The decoder factory should raise ValueError for unrecognised decoder strings."""

    with pytest.raises(ValueError, match="Unsupported decoder type"):
        _build_decoder("unknown", embedding_dimension=8, decoder_hidden_dimension=16)


@pytest.mark.parametrize("decoder_type", list(SUPPORTED_DECODER_TYPES))
def test_all_decoder_types_produce_correct_output_shape(decoder_type):
    """Every decoder type should produce one logit per candidate edge."""

    torch.manual_seed(0)
    model = GraphSageLinkPredictor(
        input_dimension=4,
        hidden_dimension=4,
        output_dimension=4,
        dropout_rate=0.0,
        decoder_type=decoder_type,
        decoder_hidden_dimension=4,
    )
    model.eval()
    node_features = torch.randn(3, 4)
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    edge_label_index = torch.tensor([[0, 1], [2, 0]], dtype=torch.long)

    logits = model(
        node_features=node_features,
        edge_index=edge_index,
        edge_label_index=edge_label_index,
    )
    assert logits.shape == (2,)


def test_build_graph_data_supports_empty_edge_list():
    """Graph data builder should keep node features when no edges exist."""

    node_feature_matrix = np.ones((3, 4), dtype=np.float32)
    graph_data = build_graph_data(
        node_feature_matrix=node_feature_matrix,
        edge_pairs=[],
        is_undirected=True,
    )
    assert graph_data.x.shape == (3, 4)
    assert graph_data.edge_index.shape == (2, 0)


def test_split_link_prediction_data_is_deterministic():
    """RandomLinkSplit output should be repeatable with a fixed seed."""

    graph_data = _create_toy_graph_data()
    split_a = split_link_prediction_data(
        graph_data=graph_data,
        validation_ratio=0.2,
        test_ratio=0.2,
        is_undirected=True,
        split_seed=42,
    )
    split_b = split_link_prediction_data(
        graph_data=graph_data,
        validation_ratio=0.2,
        test_ratio=0.2,
        is_undirected=True,
        split_seed=42,
    )

    assert torch.equal(split_a[0].edge_index, split_b[0].edge_index)
    assert torch.equal(split_a[0].edge_label_index, split_b[0].edge_label_index)
    assert torch.equal(split_a[1].edge_label_index, split_b[1].edge_label_index)
    assert torch.equal(split_a[2].edge_label_index, split_b[2].edge_label_index)


def test_build_negative_edge_labels_shape_and_label_order():
    """Negative sampling output should have correct size and binary labels."""

    graph_data = _create_toy_graph_data()
    train_data, _, _ = split_link_prediction_data(
        graph_data=graph_data,
        validation_ratio=0.2,
        test_ratio=0.2,
        is_undirected=True,
        split_seed=42,
    )
    edge_label_index, edge_labels = build_negative_edge_labels(
        train_data=train_data,
        negative_sampling_ratio=1.0,
        is_undirected=True,
        negative_sampling_seed=42,
    )

    positive_edge_count = int(train_data.edge_label_index.size(1))
    assert edge_label_index.shape[1] == edge_labels.shape[0]
    assert edge_label_index.shape[1] >= positive_edge_count
    assert torch.all(edge_labels[:positive_edge_count] == 1.0)
    assert torch.all(edge_labels[positive_edge_count:] == 0.0)


def test_train_graphsage_model_and_bundle_roundtrip(tmp_path):
    """Training should write artefacts that can be exported and loaded."""

    graph_data = _create_toy_graph_data()
    run_directory = tmp_path / "run"
    run_directory.mkdir(parents=True, exist_ok=True)
    configuration = {
        "seed": 42,
        "device": "cpu",
        "data": {"undirected": True},
        "splits": {"val_ratio": 0.2, "test_ratio": 0.2},
        "metrics": {"precision_at_k": 2},
        "plots": {"dpi": 80},
        "semantic": {"model_name": "test-model", "max_length": 32},
        "graphsage": {
            "hidden_dim": 8,
            "output_dim": 8,
            "dropout": 0.1,
            "learning_rate": 0.01,
            "epochs": 1,
            "batch_size": 2,
            "num_neighbors": [2, 2],
            "negative_sampling_ratio": 1.0,
            "decoder_type": "mlp",
            "decoder_hidden_dim": 8,
            "attachment_seed": 42,
            "attachment_top_k": 2,
        },
    }
    node_id_to_index = {"n0": 0, "n1": 1, "n2": 2, "n3": 3}
    index_to_node_id = ["n0", "n1", "n2", "n3"]
    node_name_to_id = {"Node 0": "n0", "Node 1": "n1", "Node 2": "n2", "Node 3": "n3"}
    node_display_name_by_id = {
        "n0": "Node 0",
        "n1": "Node 1",
        "n2": "Node 2",
        "n3": "Node 3",
    }

    metrics = train_graphsage_model(
        graph_data=graph_data,
        run_directory=str(run_directory),
        configuration=configuration,
        node_id_to_index=node_id_to_index,
        index_to_node_id=index_to_node_id,
        node_name_to_id=node_name_to_id,
        node_display_name_by_id=node_display_name_by_id,
    )
    assert metrics["backend"] == "graphsage"
    assert "roc_auc" in metrics["test"]
    assert (run_directory / "graphsage_model_state.pt").exists()
    assert (run_directory / "graphsage_metadata.json").exists()
    assert (run_directory / "graphsage_node_features.npy").exists()
    assert (run_directory / "graphsage_edge_index.npy").exists()
    assert (run_directory / "curves" / "roc.png").exists()
    assert (run_directory / "curves" / "pr.png").exists()

    bundle_directory = export_graphsage_bundle(
        run_directory=str(run_directory),
        bundle_directory_name="bundle",
    )
    assert (run_directory / "bundle" / "resolver_cache.json").exists()
    loaded_bundle = load_graphsage_bundle(
        bundle_directory=bundle_directory,
        device=torch.device("cpu"),
    )
    assert loaded_bundle.node_features.shape[0] == 4
    assert loaded_bundle.edge_index.shape[0] == 2
    assert loaded_bundle.node_id_to_index["n0"] == 0
    with open(run_directory / "bundle" / "manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest["semantic_model_name"] == "test-model"
    assert manifest["model"]["decoder_type"] == "mlp"
    assert manifest["model"]["decoder_hidden_dim"] == 8


def test_export_graphsage_bundle_checks_missing_required_files(tmp_path):
    """Export should raise clear errors when required files are missing."""

    run_directory = tmp_path / "run"
    run_directory.mkdir(parents=True, exist_ok=True)

    with pytest.raises(FileNotFoundError):
        export_graphsage_bundle(str(run_directory))

    torch.save({}, run_directory / "graphsage_model_state.pt")
    with pytest.raises(FileNotFoundError):
        export_graphsage_bundle(str(run_directory))

    (run_directory / "graphsage_metadata.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        export_graphsage_bundle(str(run_directory))

    np.save(
        run_directory / "graphsage_node_features.npy", np.ones((1, 1), dtype=np.float32)
    )
    with pytest.raises(FileNotFoundError):
        export_graphsage_bundle(str(run_directory))
