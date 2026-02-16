"""Graph Link Prediction: a compact package for hybrid link prediction.

This package provides:
- data IO utilities for reading simple CSV-based graphs
- graph helpers for constructing and sanitising training graphs
- embedding functions for structural (Node2Vec) and semantic (transformers) signals
- feature builders to turn node embeddings into pair features
- baseline models (logistic regression, small MLP) and training loops
- GraphSAGE training, export, and serving helpers for inductive mode demos
- metrics and plotting utilities for evaluation
- a high-level training script that ties everything together
"""

__all__ = [
    "io",
    "graph",
    "embeddings",
    "features",
    "models",
    "metrics",
    "eval",
    "graphsage",
    "server",
    "train",
    "utils",
]
