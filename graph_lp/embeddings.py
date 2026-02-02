"""Embedding helpers: structural Node2Vec and semantic transformer embeddings.

The structural function uses PyTorch Geometric's Node2Vec to learn embeddings
from random walks on the graph. The semantic function embeds free-text fields
using a transformer model and takes the [CLS] token representation.
"""

from typing import List

import numpy as np
import networkx as nx


def pyg_node2vec_structural_embeddings(
    g: nx.Graph,
    dim: int,
    walk_length: int,
    context_size: int,
    walks_per_node: int,
    epochs: int,
    lr: float,
    batch_size: int,
    device: str,
) -> np.ndarray:
    """Train PyG Node2Vec on the given graph and return node embeddings.

    Args:
        g: Input NetworkX graph (treated as undirected by adding reverse edges).
        dim: Embedding dimensionality.
        walk_length: Length of each random walk.
        context_size: Skip-gram context window size.
        walks_per_node: Number of walks to start at each node.
        epochs: Number of training epochs over generated walks.
        lr: Learning rate for SparseAdam.
        batch_size: Batch size for the Node2Vec loader.
        device: Device string, e.g. ``\"cpu\"`` or ``\"cuda\"``.

    Returns:
        A NumPy array of shape (num_nodes, dim) with float32 embeddings.
    """
    import torch
    from torch_geometric.nn import Node2Vec

    # PyG kernels and SparseAdam are not stable on Apple MPS; force CPU
    if str(device).startswith("mps"):
        device = "cpu"

    # Build edge_index with both directions to model undirected structure
    edges = np.array(list(g.edges()), dtype=np.int64)
    if edges.size == 0:
        return np.zeros((g.number_of_nodes(), dim), dtype=np.float32)
    rev = edges[:, ::-1]
    ei = np.concatenate([edges, rev], axis=0).T
    edge_index = torch.as_tensor(ei, dtype=torch.long, device=device)

    # Initialise Node2Vec model and its optimiser
    model = Node2Vec(
        edge_index=edge_index,
        embedding_dim=dim,
        walk_length=walk_length,
        context_size=context_size,
        walks_per_node=walks_per_node,
        num_negative_samples=1,
        sparse=True,
    ).to(device)
    optim = torch.optim.SparseAdam(model.parameters(), lr=lr)
    loader = model.loader(batch_size=batch_size, shuffle=True, num_workers=0)
    model.train()
    for epoch_index in range(epochs):
        print(
            f"    Node2Vec epoch {epoch_index + 1}/{epochs}...",
            flush=True,
        )
        for pos_rw, neg_rw in loader:
            pos_rw, neg_rw = pos_rw.to(device), neg_rw.to(device)
            optim.zero_grad()
            loss = model.loss(pos_rw, neg_rw)
            loss.backward()
            optim.step()
    model.eval()
    # Extract the learned embeddings from the embedding matrix
    emb = model.embedding.weight.detach().cpu().numpy().astype(np.float32)
    return emb


def transformer_semantic_embeddings(
    texts: List[str],
    model_name: str,
    batch_size: int,
    max_length: int,
    device: str,
) -> np.ndarray:
    """Embed a list of texts using a transformer [CLS] representation.

    Args:
        texts: Text strings to embed (one per node).
        model_name: Hugging Face model identifier to load.
        batch_size: Mini-batch size during embedding.
        max_length: Maximum token length; inputs are truncated as needed.
        device: Device string, e.g. ``\"cpu\"`` or ``\"cuda\"``.

    Returns:
        A NumPy array of shape (len(texts), hidden_dim) with float32 embeddings.
    """
    import os
    import torch

    # This project uses the PyTorch transformer backend, so we explicitly disable
    # TensorFlow and Flax integration in Transformers. This prevents TensorFlow
    # initialisation logs in environments where TensorFlow is installed for other
    # work, and it keeps the quickstart output clean by default.
    #
    # Set GRAPH_LP_ENABLE_TENSORFLOW=1 to explicitly allow TensorFlow imports in
    # the current process.
    if os.environ.get("GRAPH_LP_ENABLE_TENSORFLOW", "0") != "1":
        os.environ["TRANSFORMERS_NO_TF"] = "1"
        os.environ["TRANSFORMERS_NO_FLAX"] = "1"
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    from transformers import AutoTokenizer, AutoModel

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    embedding_batches = []
    # Mixed precision can accelerate inference on CUDA
    use_amp = str(device).startswith("cuda")
    total_batches = (len(texts) + batch_size - 1) // batch_size
    for batch_start in range(0, len(texts), batch_size):
        batch_texts = texts[batch_start : batch_start + batch_size]
        batch_index = batch_start // batch_size
        print(
            f"    Semantic batch {batch_index + 1}/{total_batches}...",
            flush=True,
        )
        encoded_inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            if use_amp:
                with torch.amp.autocast("cuda"):
                    cls_embeddings = model(**encoded_inputs).last_hidden_state[:, 0, :]
            else:
                cls_embeddings = model(**encoded_inputs).last_hidden_state[:, 0, :]
        embedding_batches.append(cls_embeddings.detach().cpu().numpy())
    return np.concatenate(embedding_batches, axis=0).astype(np.float32)
