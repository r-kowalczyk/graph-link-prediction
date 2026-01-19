"""Unit tests for embedding functions with mocked heavy dependencies."""

import sys
import types
from unittest.mock import MagicMock, patch

import numpy as np
import networkx as nx

from graph_lp.embeddings import (
    pyg_node2vec_structural_embeddings,
    transformer_semantic_embeddings,
)


class _NullContextManager:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


def _make_fake_torch(autocast_mock: MagicMock | None = None) -> types.SimpleNamespace:
    fake_torch = types.SimpleNamespace()
    fake_torch.long = object()
    fake_torch.as_tensor = MagicMock(return_value=MagicMock())
    fake_torch.optim = types.SimpleNamespace(
        SparseAdam=MagicMock(return_value=MagicMock())
    )
    fake_torch.no_grad = MagicMock(return_value=_NullContextManager())
    if autocast_mock is None:
        autocast_mock = MagicMock(return_value=_NullContextManager())
    fake_torch.amp = types.SimpleNamespace(autocast=autocast_mock)
    return fake_torch


def test_pyg_node2vec_structural_embeddings():
    """Test that Node2Vec structural embeddings are trained and returned."""
    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (1, 2), (2, 0)])

    positive_walks = MagicMock()
    positive_walks.to.return_value = positive_walks
    negative_walks = MagicMock()
    negative_walks.to.return_value = negative_walks

    node2vec_model = MagicMock()
    node2vec_model.to.return_value = node2vec_model
    node2vec_model.parameters.return_value = []
    node2vec_model.loader.return_value = [(positive_walks, negative_walks)]
    node2vec_model.loss.return_value = MagicMock()
    node2vec_model.loss.return_value.backward = MagicMock()
    node2vec_model.embedding.weight.detach.return_value.cpu.return_value.numpy.return_value = np.ones(
        (3, 4), dtype=np.float32
    )

    node2vec_class = MagicMock(return_value=node2vec_model)
    fake_torch = _make_fake_torch()

    fake_torch_geometric_nn = types.SimpleNamespace(Node2Vec=node2vec_class)
    fake_torch_geometric = types.SimpleNamespace(nn=fake_torch_geometric_nn)

    with patch.dict(
        sys.modules,
        {
            "torch": fake_torch,
            "torch_geometric": fake_torch_geometric,
            "torch_geometric.nn": fake_torch_geometric_nn,
        },
    ):
        result = pyg_node2vec_structural_embeddings(
            graph,
            dim=4,
            walk_length=5,
            context_size=3,
            walks_per_node=2,
            epochs=1,
            lr=0.01,
            batch_size=32,
            device="cpu",
        )

    assert result.shape == (3, 4)
    assert result.dtype == np.float32
    node2vec_class.assert_called_once()


def test_pyg_node2vec_structural_embeddings_empty_graph():
    """Test that an empty graph returns a zero embedding matrix of the expected shape."""
    graph = nx.Graph()
    graph.add_nodes_from([0, 1, 2])
    node2vec_class = MagicMock()
    fake_torch = _make_fake_torch()
    fake_torch_geometric_nn = types.SimpleNamespace(Node2Vec=node2vec_class)
    fake_torch_geometric = types.SimpleNamespace(nn=fake_torch_geometric_nn)

    with patch.dict(
        sys.modules,
        {
            "torch": fake_torch,
            "torch_geometric": fake_torch_geometric,
            "torch_geometric.nn": fake_torch_geometric_nn,
        },
    ):
        result = pyg_node2vec_structural_embeddings(
            graph,
            dim=4,
            walk_length=5,
            context_size=3,
            walks_per_node=2,
            epochs=1,
            lr=0.01,
            batch_size=32,
            device="cpu",
        )
    assert result.shape == (3, 4)
    assert result.dtype == np.float32
    assert np.all(result == 0.0)
    node2vec_class.assert_not_called()


def test_pyg_node2vec_structural_embeddings_mps_device_forces_cpu():
    """Test that the function forces CPU when the device string starts with 'mps'."""
    graph = nx.Graph()
    graph.add_edges_from([(0, 1)])

    positive_walks = MagicMock()
    positive_walks.to.return_value = positive_walks
    negative_walks = MagicMock()
    negative_walks.to.return_value = negative_walks

    node2vec_model = MagicMock()
    node2vec_model.to.return_value = node2vec_model
    node2vec_model.parameters.return_value = []
    node2vec_model.loader.return_value = [(positive_walks, negative_walks)]
    node2vec_model.loss.return_value = MagicMock()
    node2vec_model.loss.return_value.backward = MagicMock()
    node2vec_model.embedding.weight.detach.return_value.cpu.return_value.numpy.return_value = np.ones(
        (2, 4), dtype=np.float32
    )

    node2vec_class = MagicMock(return_value=node2vec_model)
    fake_torch = _make_fake_torch()

    fake_torch_geometric_nn = types.SimpleNamespace(Node2Vec=node2vec_class)
    fake_torch_geometric = types.SimpleNamespace(nn=fake_torch_geometric_nn)

    with patch.dict(
        sys.modules,
        {
            "torch": fake_torch,
            "torch_geometric": fake_torch_geometric,
            "torch_geometric.nn": fake_torch_geometric_nn,
        },
    ):
        result = pyg_node2vec_structural_embeddings(
            graph,
            dim=4,
            walk_length=5,
            context_size=3,
            walks_per_node=2,
            epochs=1,
            lr=0.01,
            batch_size=32,
            device="mps",
        )

    assert result.shape == (2, 4)


def test_transformer_semantic_embeddings_cpu_path():
    """Test that transformer embeddings are computed on the non-AMP (CPU) path."""
    texts = ["text1", "text2", "text3"]

    encoded_inputs = MagicMock()
    encoded_inputs.to.return_value = encoded_inputs

    tokenizer = MagicMock()
    tokenizer.return_value = encoded_inputs

    hidden_state_tensor = MagicMock()
    hidden_state_tensor.detach.return_value.cpu.return_value.numpy.side_effect = [
        np.ones((2, 768), dtype=np.float32),
        np.ones((1, 768), dtype=np.float32),
    ]

    model_output = MagicMock()
    model_output.last_hidden_state.__getitem__.return_value = hidden_state_tensor

    model = MagicMock()
    model.to.return_value = model
    model.return_value = model_output

    fake_transformers = types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(
            from_pretrained=MagicMock(return_value=tokenizer)
        ),
        AutoModel=types.SimpleNamespace(from_pretrained=MagicMock(return_value=model)),
    )

    fake_torch = _make_fake_torch()

    with patch.dict(
        sys.modules, {"torch": fake_torch, "transformers": fake_transformers}
    ):
        result = transformer_semantic_embeddings(
            texts, model_name="test-model", batch_size=2, max_length=128, device="cpu"
        )

    assert result.shape == (3, 768)
    assert result.dtype == np.float32


def test_transformer_semantic_embeddings_cuda_uses_autocast():
    """Test that the CUDA path enters autocast when the device string starts with 'cuda'."""
    texts = ["text1", "text2"]

    encoded_inputs = MagicMock()
    encoded_inputs.to.return_value = encoded_inputs

    tokenizer = MagicMock()
    tokenizer.return_value = encoded_inputs

    hidden_state_tensor = MagicMock()
    hidden_state_tensor.detach.return_value.cpu.return_value.numpy.return_value = (
        np.ones((2, 768), dtype=np.float32)
    )

    model_output = MagicMock()
    model_output.last_hidden_state.__getitem__.return_value = hidden_state_tensor

    model = MagicMock()
    model.to.return_value = model
    model.return_value = model_output

    fake_transformers = types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(
            from_pretrained=MagicMock(return_value=tokenizer)
        ),
        AutoModel=types.SimpleNamespace(from_pretrained=MagicMock(return_value=model)),
    )

    autocast_mock = MagicMock(return_value=_NullContextManager())
    fake_torch = _make_fake_torch(autocast_mock=autocast_mock)

    with patch.dict(
        sys.modules, {"torch": fake_torch, "transformers": fake_transformers}
    ):
        result = transformer_semantic_embeddings(
            texts, model_name="test-model", batch_size=2, max_length=128, device="cuda"
        )

    assert result.shape == (2, 768)
    autocast_mock.assert_called()
