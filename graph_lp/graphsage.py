"""GraphSAGE backend for binary link prediction and serving bundles.

This module implements a compact GraphSAGE pipeline for a single graph setting.
It includes deterministic edge splitting, negative sampling, mini-batch training,
metrics generation, and bundle export and load helpers for API serving.
The implementation is designed for CPU quickstart runs and clear reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import shutil
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as torch_functional
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import negative_sampling

from .eval import compute_and_plot
from .metrics import compute_classification_metrics
from .utils import ensure_dir, write_json


class GraphSageEncoder(nn.Module):
    """Encode node features into structural representations with GraphSAGE layers.

    The encoder uses two neighbourhood aggregation layers so it remains fast on CPU.
    A two-layer design is chosen because it captures local graph context clearly while
    keeping the quickstart training time small and stable for repeated demonstrations.
    """

    def __init__(
        self,
        input_dimension: int,
        hidden_dimension: int,
        output_dimension: int,
        dropout_rate: float,
    ) -> None:
        """Initialise GraphSAGE layers and dropout for link prediction encoding.

        Parameters:
            input_dimension: Dimension of node input features.
            hidden_dimension: Hidden representation width after the first layer.
            output_dimension: Final embedding width produced for decoding.
            dropout_rate: Dropout applied between layers to reduce overfitting.
        """
        super().__init__()
        self.first_convolution = SAGEConv(input_dimension, hidden_dimension)
        self.second_convolution = SAGEConv(hidden_dimension, output_dimension)
        self.dropout_rate = float(dropout_rate)

    def forward(
        self, node_features: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Propagate node features through two GraphSAGE aggregation layers.

        Parameters:
            node_features: Dense node feature matrix with shape (nodes, features).
            edge_index: COO edge index with shape (2, number_of_edges).

        Returns:
            Node embedding matrix with shape (nodes, output_dimension).
        """
        # The first aggregation creates a local neighbourhood-aware hidden signal.
        hidden_node_features = self.first_convolution(node_features, edge_index)
        hidden_node_features = torch_functional.relu(hidden_node_features)
        hidden_node_features = torch_functional.dropout(
            hidden_node_features, p=self.dropout_rate, training=self.training
        )
        return self.second_convolution(hidden_node_features, edge_index)


class DotProductLinkDecoder(nn.Module):
    """Decode candidate links by dot product between endpoint embeddings.

    Dot product is parameter-free and fast but can only capture linear similarity
    in embedding space. Suitable for quick sanity runs but typically underperforms
    learnable decoders on real datasets where decision boundaries are non-linear.
    """

    def forward(
        self,
        node_embeddings: torch.Tensor,
        edge_label_index: torch.Tensor,
    ) -> torch.Tensor:
        """Compute link logits for candidate edge endpoints.

        Parameters:
            node_embeddings: Node embedding matrix from the GraphSAGE encoder.
            edge_label_index: Candidate links with source and target node indices.

        Returns:
            A one-dimensional tensor of logits for each candidate edge.
        """
        source_node_embeddings = node_embeddings[edge_label_index[0]]
        target_node_embeddings = node_embeddings[edge_label_index[1]]
        return (source_node_embeddings * target_node_embeddings).sum(dim=-1)


class BilinearLinkDecoder(nn.Module):
    """Decode candidate links with a learnable bilinear interaction matrix.

    A bilinear decoder learns a square weight matrix that transforms one endpoint
    before taking the dot product with the other. This adds expressiveness over a
    plain dot product (it can weight embedding dimensions differently) while staying
    compact: the only parameter is one (output_dim x output_dim) matrix.
    """

    def __init__(self, embedding_dimension: int) -> None:
        """Initialise the bilinear interaction weight matrix.

        Parameters:
            embedding_dimension: Width of the node embeddings produced by the encoder.
        """
        super().__init__()
        self.bilinear_layer = nn.Bilinear(
            embedding_dimension, embedding_dimension, 1, bias=False
        )

    def forward(
        self,
        node_embeddings: torch.Tensor,
        edge_label_index: torch.Tensor,
    ) -> torch.Tensor:
        """Compute link logits using a learnable bilinear form.

        Parameters:
            node_embeddings: Node embedding matrix from the GraphSAGE encoder.
            edge_label_index: Candidate links with source and target node indices.

        Returns:
            A one-dimensional tensor of logits for each candidate edge.
        """
        source_node_embeddings = node_embeddings[edge_label_index[0]]
        target_node_embeddings = node_embeddings[edge_label_index[1]]
        return self.bilinear_layer(
            source_node_embeddings, target_node_embeddings
        ).squeeze(-1)


class MLPLinkDecoder(nn.Module):
    """Decode candidate links with a two-layer MLP over concatenated embeddings.

    The MLP decoder concatenates source and target embeddings and passes them
    through a hidden layer with ReLU activation, dropout, and a single output unit.
    This allows learning non-linear decision boundaries between link endpoints,
    which typically outperforms dot product and bilinear decoders on datasets
    where structural and semantic similarity interact in complex ways. Dropout
    is applied between the hidden and output layers to reduce overfitting on
    the training edge set.
    """

    def __init__(
        self,
        embedding_dimension: int,
        decoder_hidden_dimension: int,
        dropout_rate: float = 0.2,
    ) -> None:
        """Initialise the two-layer MLP used for link scoring.

        Parameters:
            embedding_dimension: Width of the node embeddings from the encoder.
            decoder_hidden_dimension: Width of the hidden layer inside the decoder MLP.
            dropout_rate: Dropout applied after the hidden activation to limit overfitting.
        """
        super().__init__()
        # Concatenation of source and target embeddings doubles the input width.
        self.hidden_layer = nn.Linear(embedding_dimension * 2, decoder_hidden_dimension)
        self.dropout_rate = float(dropout_rate)
        self.output_layer = nn.Linear(decoder_hidden_dimension, 1)

    def forward(
        self,
        node_embeddings: torch.Tensor,
        edge_label_index: torch.Tensor,
    ) -> torch.Tensor:
        """Compute link logits from concatenated source and target embeddings.

        Parameters:
            node_embeddings: Node embedding matrix from the GraphSAGE encoder.
            edge_label_index: Candidate links with source and target node indices.

        Returns:
            A one-dimensional tensor of logits for each candidate edge.
        """
        source_node_embeddings = node_embeddings[edge_label_index[0]]
        target_node_embeddings = node_embeddings[edge_label_index[1]]
        concatenated = torch.cat(
            [source_node_embeddings, target_node_embeddings], dim=-1
        )
        hidden = torch_functional.relu(self.hidden_layer(concatenated))
        hidden = torch_functional.dropout(
            hidden, p=self.dropout_rate, training=self.training
        )
        return self.output_layer(hidden).squeeze(-1)


# Recognised decoder type strings mapped to short descriptions for validation.
SUPPORTED_DECODER_TYPES = ("dot_product", "bilinear", "mlp")


def _build_decoder(
    decoder_type: str,
    embedding_dimension: int,
    decoder_hidden_dimension: int,
    dropout_rate: float = 0.2,
) -> nn.Module:
    """Instantiate a link decoder module by type string.

    This factory centralises decoder construction so both training and bundle loading
    use the same logic. The decoder_type string is stored in the serving manifest so
    the correct decoder is reconstructed without retraining.

    Parameters:
        decoder_type: One of "dot_product", "bilinear", or "mlp".
        embedding_dimension: Width of the node embeddings from the encoder.
        decoder_hidden_dimension: Hidden width for the MLP decoder (ignored by others).
        dropout_rate: Dropout probability forwarded to the MLP decoder (ignored by others).

    Returns:
        An initialised decoder module ready for training or weight loading.
    """
    if decoder_type == "dot_product":
        return DotProductLinkDecoder()
    if decoder_type == "bilinear":
        return BilinearLinkDecoder(embedding_dimension=embedding_dimension)
    if decoder_type == "mlp":
        return MLPLinkDecoder(
            embedding_dimension=embedding_dimension,
            decoder_hidden_dimension=decoder_hidden_dimension,
            dropout_rate=dropout_rate,
        )
    raise ValueError(
        f"Unsupported decoder type '{decoder_type}'. "
        f"Supported types: {SUPPORTED_DECODER_TYPES}"
    )


class GraphSageLinkPredictor(nn.Module):
    """Combine GraphSAGE encoding with a configurable link decoder.

    The predictor exposes separate encode and decode methods so training and serving
    can reuse the same components while keeping function-level tests straightforward.
    The decoder type is selectable at construction time and persisted in the bundle
    manifest so serving reconstructs the same architecture without retraining.
    """

    def __init__(
        self,
        input_dimension: int,
        hidden_dimension: int,
        output_dimension: int,
        dropout_rate: float,
        decoder_type: str = "mlp",
        decoder_hidden_dimension: int = 64,
    ) -> None:
        """Create encoder and decoder modules for binary link scoring.

        Parameters:
            input_dimension: Dimension of node input feature vectors.
            hidden_dimension: Width of the hidden GraphSAGE representation.
            output_dimension: Width of the final node embeddings.
            dropout_rate: Dropout probability used inside the encoder.
            decoder_type: Which decoder architecture to use ("dot_product", "bilinear", or "mlp").
            decoder_hidden_dimension: Hidden layer width for the MLP decoder (ignored by other decoders).
        """
        super().__init__()
        self.encoder = GraphSageEncoder(
            input_dimension=input_dimension,
            hidden_dimension=hidden_dimension,
            output_dimension=output_dimension,
            dropout_rate=dropout_rate,
        )
        self.decoder = _build_decoder(
            decoder_type=decoder_type,
            embedding_dimension=output_dimension,
            decoder_hidden_dimension=decoder_hidden_dimension,
            dropout_rate=dropout_rate,
        )

    def encode(
        self, node_features: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Encode graph nodes into embeddings suitable for link decoding.

        Parameters:
            node_features: Dense node feature matrix used as message-passing input.
            edge_index: Graph connectivity used for neighbourhood aggregation.

        Returns:
            Node embeddings for all nodes in the provided graph.
        """
        return self.encoder(node_features=node_features, edge_index=edge_index)

    def decode(
        self,
        node_embeddings: torch.Tensor,
        edge_label_index: torch.Tensor,
    ) -> torch.Tensor:
        """Decode candidate links from precomputed node embeddings.

        Parameters:
            node_embeddings: Output from the encoder for a graph instance.
            edge_label_index: Candidate node pairs that should be scored.

        Returns:
            Link logits for all candidate node pairs.
        """
        return self.decoder(
            node_embeddings=node_embeddings,
            edge_label_index=edge_label_index,
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_label_index: torch.Tensor,
    ) -> torch.Tensor:
        """Encode a graph and decode candidate links in one call.

        Parameters:
            node_features: Node feature matrix used for GraphSAGE aggregation.
            edge_index: Graph edges used by the message-passing layers.
            edge_label_index: Candidate links that should be assigned logits.

        Returns:
            Link logits that can be converted to probabilities with sigmoid.
        """
        node_embeddings = self.encode(
            node_features=node_features, edge_index=edge_index
        )
        return self.decode(
            node_embeddings=node_embeddings,
            edge_label_index=edge_label_index,
        )


@dataclass
class LoadedGraphSageBundle:
    """Runtime bundle structure used by the serving API.

    This dataclass stores the model, graph tensors, and metadata that are needed
    to score existing and newly attached entities without retraining. It keeps the
    server startup logic compact by collecting all required fields in one object.
    """

    model: GraphSageLinkPredictor
    node_features: torch.Tensor
    edge_index: torch.Tensor
    node_id_to_index: dict[str, int]
    index_to_node_id: list[str]
    node_name_to_id: dict[str, str]
    node_display_name_by_id: dict[str, str]
    semantic_model_name: str
    semantic_max_length: int
    undirected: bool
    attachment_seed: int
    attachment_top_k: int


def build_graph_data(
    node_feature_matrix: np.ndarray,
    edge_pairs: list[tuple[int, int]],
    is_undirected: bool,
) -> Data:
    """Build a PyTorch Geometric Data object from features and integer edges.

    Parameters:
        node_feature_matrix: Node features generated from semantic text embeddings.
        edge_pairs: Edge endpoint pairs indexed by contiguous integer node ids.
        is_undirected: Whether reverse edges should be added for message passing.

    Returns:
        A Data object containing node features and graph connectivity.
    """
    # Reverse edges are added because undirected biological associations are common.
    directed_edge_pairs = list(edge_pairs)
    if is_undirected:
        directed_edge_pairs.extend([(target, source) for source, target in edge_pairs])

    edge_array = np.asarray(directed_edge_pairs, dtype=np.int64)
    if edge_array.size == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.as_tensor(edge_array.T, dtype=torch.long)

    return Data(
        x=torch.as_tensor(node_feature_matrix, dtype=torch.float32),
        edge_index=edge_index,
    )


def split_link_prediction_data(
    graph_data: Data,
    validation_ratio: float,
    test_ratio: float,
    is_undirected: bool,
    split_seed: int,
) -> tuple[Data, Data, Data]:
    """Create deterministic train, validation, and test link splits.

    Parameters:
        graph_data: Full graph data object with node features and edge index.
        validation_ratio: Fraction of positive edges assigned to validation.
        test_ratio: Fraction of positive edges assigned to testing.
        is_undirected: Whether edges should be treated as undirected in splitting.
        split_seed: Seed used to make the random split reproducible.

    Returns:
        A tuple of (train_data, validation_data, test_data) objects.
    """
    # A fixed manual seed is set directly before splitting to make runs reproducible.
    torch.manual_seed(int(split_seed))
    splitter = RandomLinkSplit(
        num_val=float(validation_ratio),
        num_test=float(test_ratio),
        is_undirected=bool(is_undirected),
        add_negative_train_samples=False,
    )
    train_data, validation_data, test_data = splitter(graph_data)
    return train_data, validation_data, test_data


def build_negative_edge_labels(
    train_data: Data,
    negative_sampling_ratio: float,
    is_undirected: bool,
    negative_sampling_seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create deterministic positive and negative labels for training links.

    Parameters:
        train_data: Split training data from RandomLinkSplit.
        negative_sampling_ratio: Ratio of negatives to positives for training.
        is_undirected: Whether negative samples should be generated undirected.
        negative_sampling_seed: Seed controlling negative edge sampling choices.

    Returns:
        A tuple of (edge_label_index, edge_label) tensors.
    """
    positive_edge_label_index = train_data.edge_label_index
    number_of_positive_edges = int(positive_edge_label_index.size(1))
    number_of_negative_edges = int(
        number_of_positive_edges * float(negative_sampling_ratio)
    )

    # The seed is reset before negative sampling so the sampled negatives are repeatable.
    torch.manual_seed(int(negative_sampling_seed))
    negative_edge_label_index = negative_sampling(
        edge_index=train_data.edge_index,
        num_nodes=int(train_data.num_nodes),
        num_neg_samples=number_of_negative_edges,
        method="sparse",
        force_undirected=bool(is_undirected),
    )
    number_of_negative_edges = int(negative_edge_label_index.size(1))

    positive_edge_labels = torch.ones(number_of_positive_edges, dtype=torch.float32)
    negative_edge_labels = torch.zeros(number_of_negative_edges, dtype=torch.float32)
    edge_label_index = torch.cat(
        [positive_edge_label_index, negative_edge_label_index], dim=1
    )
    edge_labels = torch.cat([positive_edge_labels, negative_edge_labels], dim=0)
    return edge_label_index, edge_labels


def create_train_loader(
    edge_label_index: torch.Tensor,
    edge_labels: torch.Tensor,
    batch_size: int,
    loader_seed: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Create deterministic mini-batches for link supervision.

    Parameters:
        edge_label_index: Candidate positive and negative links for supervision.
        edge_labels: Binary labels aligned with edge_label_index columns.
        batch_size: Number of labelled links per training mini-batch.
        loader_seed: Seed controlling deterministic batch ordering.

    Returns:
        A list of (edge_label_index, edge_labels) tensors per mini-batch.
    """
    # Edge mini-batching keeps the task link-centric without requiring optional
    # neighbour sampling extensions that are not installed in quickstart setups.
    # A seeded permutation keeps edge mini-batches reproducible across reruns.
    rng = np.random.default_rng(int(loader_seed))
    edge_positions = np.arange(int(edge_labels.size(0)))
    rng.shuffle(edge_positions)
    batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    for batch_start in range(0, edge_positions.shape[0], int(batch_size)):
        selected_positions = edge_positions[batch_start : batch_start + int(batch_size)]
        selected_position_tensor = torch.as_tensor(selected_positions, dtype=torch.long)
        batches.append(
            (
                edge_label_index[:, selected_position_tensor],
                edge_labels[selected_position_tensor],
            )
        )
    return batches


def create_eval_loader(
    split_data: Data,
    batch_size: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Create deterministic edge-label mini-batches for evaluation scoring.

    Parameters:
        split_data: Validation or test split produced by RandomLinkSplit.
        batch_size: Number of labelled links per evaluation mini-batch.

    Returns:
        A list of (edge_label_index, edge_labels) tensors per mini-batch.
    """
    edge_label_index = split_data.edge_label_index
    edge_labels = split_data.edge_label.float()
    batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    edge_count = int(edge_labels.size(0))
    for batch_start in range(0, edge_count, int(batch_size)):
        batch_end = batch_start + int(batch_size)
        batches.append(
            (
                edge_label_index[:, batch_start:batch_end],
                edge_labels[batch_start:batch_end],
            )
        )
    return batches


def collect_probabilities(
    model: GraphSageLinkPredictor,
    split_data: Data,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect binary labels and predicted probabilities for a split.

    Parameters:
        model: Trained GraphSAGE link predictor model.
        split_data: Validation or test split data containing edge labels.
        batch_size: Number of links per evaluation mini-batch.
        device: Torch device used for model execution.

    Returns:
        A tuple of (labels, probabilities) arrays for metric computation.
    """
    evaluation_loader = create_eval_loader(
        split_data=split_data,
        batch_size=batch_size,
    )
    all_labels: list[np.ndarray] = []
    all_probabilities: list[np.ndarray] = []
    node_features = split_data.x.to(device)
    edge_index = split_data.edge_index.to(device)

    model.eval()
    with torch.no_grad():
        for batch_edge_label_index, batch_edge_labels in evaluation_loader:
            logits = model(
                node_features=node_features,
                edge_index=edge_index,
                edge_label_index=batch_edge_label_index.to(device),
            )
            probabilities = torch.sigmoid(logits).cpu().numpy()
            labels = batch_edge_labels.cpu().numpy()
            all_probabilities.append(probabilities)
            all_labels.append(labels)

    return (
        np.concatenate(all_labels).astype(int),
        np.concatenate(all_probabilities).astype(float),
    )


def train_graphsage_model(
    graph_data: Data,
    run_directory: str,
    configuration: dict[str, Any],
    node_id_to_index: dict[str, int],
    index_to_node_id: list[str],
    node_name_to_id: dict[str, str],
    node_display_name_by_id: dict[str, str],
) -> dict[str, Any]:
    """Train GraphSAGE for binary link prediction and write run artefacts.

    Parameters:
        graph_data: Full graph data object built from node features and edges.
        run_directory: Output directory where metrics and model artefacts are saved.
        configuration: Full configuration mapping used for this training run.
        node_id_to_index: Mapping from node identifier strings to contiguous indices.
        index_to_node_id: Inverse mapping list from index to node identifier.
        node_name_to_id: Name lookup mapping used by serving requests.
        node_display_name_by_id: Human readable names keyed by node identifier.

    Returns:
        Metrics and model metadata for the completed GraphSAGE run.
    """
    graphsage_configuration = dict(configuration.get("graphsage", {}))
    resolved_device = torch.device(str(configuration["device"]))
    split_seed = int(configuration["seed"])
    validation_ratio = float(configuration["splits"]["val_ratio"])
    test_ratio = float(configuration["splits"]["test_ratio"])
    is_undirected = bool(configuration["data"]["undirected"])
    hidden_dimension = int(graphsage_configuration.get("hidden_dim", 64))
    output_dimension = int(graphsage_configuration.get("output_dim", 64))
    dropout_rate = float(graphsage_configuration.get("dropout", 0.2))
    learning_rate = float(graphsage_configuration.get("learning_rate", 0.001))
    number_of_epochs = int(graphsage_configuration.get("epochs", 10))
    batch_size = int(graphsage_configuration.get("batch_size", 256))
    negative_sampling_ratio = float(
        graphsage_configuration.get("negative_sampling_ratio", 1.0)
    )
    number_of_neighbours = list(graphsage_configuration.get("num_neighbors", [20, 10]))
    decoder_type = str(graphsage_configuration.get("decoder_type", "mlp"))
    decoder_hidden_dimension = int(
        graphsage_configuration.get("decoder_hidden_dim", 64)
    )
    weight_decay = float(graphsage_configuration.get("weight_decay", 1e-3))
    precision_at_k = int(configuration["metrics"]["precision_at_k"])
    plot_dpi = int(configuration["plots"]["dpi"])
    attachment_seed = int(graphsage_configuration.get("attachment_seed", split_seed))
    attachment_top_k = int(graphsage_configuration.get("attachment_top_k", 5))

    train_data, validation_data, test_data = split_link_prediction_data(
        graph_data=graph_data,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
        is_undirected=is_undirected,
        split_seed=split_seed,
    )
    train_edge_label_index, train_edge_labels = build_negative_edge_labels(
        train_data=train_data,
        negative_sampling_ratio=negative_sampling_ratio,
        is_undirected=is_undirected,
        negative_sampling_seed=split_seed,
    )
    train_loader = create_train_loader(
        edge_label_index=train_edge_label_index,
        edge_labels=train_edge_labels,
        batch_size=batch_size,
        loader_seed=split_seed,
    )

    model = GraphSageLinkPredictor(
        input_dimension=int(graph_data.x.size(1)),
        hidden_dimension=hidden_dimension,
        output_dimension=output_dimension,
        dropout_rate=dropout_rate,
        decoder_type=decoder_type,
        decoder_hidden_dimension=decoder_hidden_dimension,
    ).to(resolved_device)
    # L2 weight decay penalises large parameter values and is the primary guard
    # against overfitting in full-graph message passing where the encoder sees the
    # entire topology every forward pass.
    optimiser = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    loss_function = nn.BCEWithLogitsLoss()
    train_node_features = train_data.x.to(resolved_device)
    train_edge_index = train_data.edge_index.to(resolved_device)

    # Track best validation ROC-AUC across epochs and keep the corresponding weights.
    # This prevents overfitting when using learnable decoders (MLP, bilinear) that can
    # memorise training edges if trained past the generalisation peak.
    best_validation_roc_auc = -1.0
    best_model_state: dict[str, Any] | None = None

    for epoch_number in range(number_of_epochs):
        model.train()
        epoch_loss_values: list[float] = []
        for batch_edge_label_index, batch_edge_labels in train_loader:
            optimiser.zero_grad()
            logits = model(
                node_features=train_node_features,
                edge_index=train_edge_index,
                edge_label_index=batch_edge_label_index.to(resolved_device),
            )
            loss = loss_function(logits, batch_edge_labels.to(resolved_device))
            loss.backward()
            optimiser.step()
            epoch_loss_values.append(float(loss.item()))
        mean_epoch_loss = (
            float(np.mean(epoch_loss_values)) if epoch_loss_values else 0.0
        )

        # Evaluate validation ROC-AUC each epoch to select the best checkpoint.
        epoch_validation_labels, epoch_validation_probabilities = collect_probabilities(
            model=model,
            split_data=validation_data,
            batch_size=batch_size,
            device=resolved_device,
        )
        epoch_validation_metrics = compute_classification_metrics(
            epoch_validation_labels,
            epoch_validation_probabilities,
            precision_at_k,
        )
        epoch_validation_roc_auc = float(epoch_validation_metrics["roc_auc"])

        print(
            f"    GraphSAGE epoch {epoch_number + 1}/{number_of_epochs} "
            f"loss={mean_epoch_loss:.4f} val_auc={epoch_validation_roc_auc:.4f}",
            flush=True,
        )

        if epoch_validation_roc_auc > best_validation_roc_auc:
            best_validation_roc_auc = epoch_validation_roc_auc
            best_model_state = {
                key: tensor.cpu().clone() for key, tensor in model.state_dict().items()
            }

    # Restore the weights from the epoch with the highest validation ROC-AUC.
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model.to(resolved_device)
    print(
        f"    Best validation ROC-AUC: {best_validation_roc_auc:.4f}",
        flush=True,
    )

    validation_labels, validation_probabilities = collect_probabilities(
        model=model,
        split_data=validation_data,
        batch_size=batch_size,
        device=resolved_device,
    )
    test_labels, test_probabilities = collect_probabilities(
        model=model,
        split_data=test_data,
        batch_size=batch_size,
        device=resolved_device,
    )

    curves_directory = os.path.join(run_directory, "curves")
    ensure_dir(curves_directory)
    validation_metrics = compute_classification_metrics(
        validation_labels,
        validation_probabilities,
        precision_at_k,
    )
    test_metrics = compute_and_plot(
        y_true=test_labels,
        y_prob=test_probabilities,
        out_dir=curves_directory,
        k=precision_at_k,
        dpi=plot_dpi,
    )

    training_state_path = os.path.join(run_directory, "graphsage_model_state.pt")
    torch.save(model.state_dict(), training_state_path)
    np.save(
        os.path.join(run_directory, "graphsage_node_features.npy"),
        graph_data.x.cpu().numpy(),
    )
    np.save(
        os.path.join(run_directory, "graphsage_edge_index.npy"),
        graph_data.edge_index.cpu().numpy(),
    )
    metadata = {
        "node_id_to_index": node_id_to_index,
        "index_to_node_id": index_to_node_id,
        "node_name_to_id": node_name_to_id,
        "node_display_name_by_id": node_display_name_by_id,
        "semantic_model_name": str(configuration["semantic"]["model_name"]),
        "semantic_max_length": int(configuration["semantic"]["max_length"]),
        "is_undirected": is_undirected,
        "attachment_seed": attachment_seed,
        "attachment_top_k": attachment_top_k,
        "model": {
            "hidden_dim": hidden_dimension,
            "output_dim": output_dimension,
            "dropout": dropout_rate,
            "input_dim": int(graph_data.x.size(1)),
            "decoder_type": decoder_type,
            "decoder_hidden_dim": decoder_hidden_dimension,
        },
    }
    write_json(os.path.join(run_directory, "graphsage_metadata.json"), metadata)

    return {
        "backend": "graphsage",
        "variant": "semantic",
        "val_graphsage": validation_metrics,
        "test_graphsage": test_metrics,
        "test": test_metrics,
        "graphsage": {
            "hidden_dim": hidden_dimension,
            "output_dim": output_dimension,
            "dropout": dropout_rate,
            "epochs": number_of_epochs,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "num_neighbors": number_of_neighbours,
            "negative_sampling_ratio": negative_sampling_ratio,
            "decoder_type": decoder_type,
            "decoder_hidden_dim": decoder_hidden_dimension,
            "weight_decay": weight_decay,
            "split_seed": split_seed,
        },
    }


def export_graphsage_bundle(
    run_directory: str,
    bundle_directory_name: str = "serving_bundle",
) -> str:
    """Export a minimal serving bundle from GraphSAGE run artefacts.

    Parameters:
        run_directory: Completed run directory containing GraphSAGE artefacts.
        bundle_directory_name: Name of the output bundle directory under the run.

    Returns:
        Absolute path to the created serving bundle directory.
    """
    source_model_path = os.path.join(run_directory, "graphsage_model_state.pt")
    source_metadata_path = os.path.join(run_directory, "graphsage_metadata.json")
    source_node_features_path = os.path.join(
        run_directory, "graphsage_node_features.npy"
    )
    source_edge_index_path = os.path.join(run_directory, "graphsage_edge_index.npy")

    if not os.path.exists(source_model_path):
        raise FileNotFoundError(
            f"Missing GraphSAGE model artefact in run directory: {source_model_path}"
        )
    if not os.path.exists(source_metadata_path):
        raise FileNotFoundError(
            f"Missing GraphSAGE metadata artefact in run directory: {source_metadata_path}"
        )
    if not os.path.exists(source_node_features_path):
        raise FileNotFoundError(
            "Missing GraphSAGE node features artefact in run directory: "
            f"{source_node_features_path}"
        )
    if not os.path.exists(source_edge_index_path):
        raise FileNotFoundError(
            f"Missing GraphSAGE edge index artefact in run directory: {source_edge_index_path}"
        )

    bundle_directory = os.path.join(run_directory, bundle_directory_name)
    ensure_dir(bundle_directory)

    # Files are copied with stable names so serving startup code stays straightforward.
    shutil.copyfile(source_model_path, os.path.join(bundle_directory, "model_state.pt"))
    shutil.copyfile(
        source_metadata_path, os.path.join(bundle_directory, "manifest.json")
    )
    shutil.copyfile(
        source_node_features_path, os.path.join(bundle_directory, "node_features.npy")
    )
    shutil.copyfile(
        source_edge_index_path, os.path.join(bundle_directory, "edge_index.npy")
    )

    resolver_cache_path = os.path.join(bundle_directory, "resolver_cache.json")
    if not os.path.exists(resolver_cache_path):
        write_json(resolver_cache_path, {})

    return bundle_directory


def load_graphsage_bundle(
    bundle_directory: str,
    device: torch.device,
) -> LoadedGraphSageBundle:
    """Load a serving bundle and reconstruct the GraphSAGE runtime model.

    Parameters:
        bundle_directory: Directory created by export_graphsage_bundle.
        device: Torch device that should host the model and tensors.

    Returns:
        A LoadedGraphSageBundle with model, tensors, and serving metadata.
    """
    manifest_path = os.path.join(bundle_directory, "manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)

    node_features = torch.as_tensor(
        np.load(os.path.join(bundle_directory, "node_features.npy")),
        dtype=torch.float32,
        device=device,
    )
    edge_index = torch.as_tensor(
        np.load(os.path.join(bundle_directory, "edge_index.npy")),
        dtype=torch.long,
        device=device,
    )

    model_configuration = manifest["model"]
    model = GraphSageLinkPredictor(
        input_dimension=int(model_configuration["input_dim"]),
        hidden_dimension=int(model_configuration["hidden_dim"]),
        output_dimension=int(model_configuration["output_dim"]),
        dropout_rate=float(model_configuration["dropout"]),
        decoder_type=str(model_configuration.get("decoder_type", "dot_product")),
        decoder_hidden_dimension=int(model_configuration.get("decoder_hidden_dim", 64)),
    ).to(device)
    state_dictionary = torch.load(
        os.path.join(bundle_directory, "model_state.pt"),
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(state_dictionary)
    model.eval()

    return LoadedGraphSageBundle(
        model=model,
        node_features=node_features,
        edge_index=edge_index,
        node_id_to_index={
            str(key): int(value) for key, value in manifest["node_id_to_index"].items()
        },
        index_to_node_id=[
            str(node_identifier) for node_identifier in manifest["index_to_node_id"]
        ],
        node_name_to_id={
            str(name): str(node_identifier)
            for name, node_identifier in manifest["node_name_to_id"].items()
        },
        node_display_name_by_id={
            str(node_identifier): str(display_name)
            for node_identifier, display_name in manifest[
                "node_display_name_by_id"
            ].items()
        },
        semantic_model_name=str(manifest["semantic_model_name"]),
        semantic_max_length=int(manifest["semantic_max_length"]),
        undirected=bool(manifest["is_undirected"]),
        attachment_seed=int(manifest["attachment_seed"]),
        attachment_top_k=int(manifest["attachment_top_k"]),
    )
