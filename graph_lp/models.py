"""Simple baseline models for link prediction: Logistic Regression and a small MLP.

The MLP is a single-hidden-layer network suitable for tabular pair features.
Helpers are provided to train the MLP with validation-based model selection.
"""

from typing import Tuple, List, Optional, Any

import numpy as np
from sklearn.linear_model import LogisticRegression
import torch
from torch import nn
from sklearn.metrics import roc_auc_score


class LogRegModel:
    """Thin wrapper around scikit-learn's LogisticRegression."""

    def __init__(self) -> None:
        self.clf = LogisticRegression(max_iter=1000)

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """Fit the logistic regression model."""
        self.clf.fit(x, y)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Predict probabilities for the positive class."""
        return self.clf.predict_proba(x)[:, 1]


class LinkClassifier(nn.Module):
    """A one-hidden-layer MLP for binary classification of pair features."""

    def __init__(self, in_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        # Two linear layers with ReLU non-linearity in between; output is a single logit
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute logits for the given batch of features."""
        return self.mlp(x)


def train_classifier(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int,
    lr: float,
    batch_size: int,
    device: str,
) -> nn.Module:
    """Train an MLP and keep the checkpoint with the best validation ROC-AUC.

    Args:
        model: The network to train.
        X_train: Training features of shape (N, D).
        y_train: Training labels of shape (N,).
        X_val: Validation features.
        y_val: Validation labels.
        epochs: Number of epochs to train for.
        lr: Learning rate for Adam.
        batch_size: Mini-batch size.
        device: Device string (``\"cpu\"`` or ``\"cuda\"``).

    Returns:
        The input model loaded with the best-performing weights.
    """
    criterion = nn.BCEWithLogitsLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)

    # Move arrays to tensors on the target device
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_np = y_val.astype(float)

    best_auc = 0.0
    best_state: Optional[Any] = None
    for _ in range(int(epochs)):
        model.train()
        # Shuffle indices for each epoch to avoid bias
        perm = torch.randperm(X_train_t.size(0))
        for i in range(0, X_train_t.size(0), int(batch_size)):
            idx = perm[i : i + int(batch_size)]
            xb = X_train_t[idx]
            yb = y_train_t[idx]
            optimiser.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimiser.step()
        model.eval()
        with torch.no_grad():
            val_out = model(X_val_t)
            val_prob = torch.sigmoid(val_out).cpu().numpy().flatten()
        auc = roc_auc_score(y_val_np, val_prob) if len(np.unique(y_val_np)) > 1 else 0.5
        if auc > best_auc:
            best_auc = auc
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def hyperparam_search_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    hidden_dims: List[int],
    lrs: List[float],
    epochs_list: List[int],
    batch_size: int,
    device: str,
) -> Tuple[LinkClassifier, Tuple[int, float, int], float]:
    """Grid search over ``hidden_dim``, ``lr`` and ``epochs`` for a small MLP.

    Args:
        X_train: Training features.
        y_train: Training labels.
        X_val: Validation features.
        y_val: Validation labels.
        hidden_dims: Candidate hidden layer widths.
        lrs: Candidate learning rates.
        epochs_list: Candidate epoch counts.
        batch_size: Mini-batch size.
        device: Device string (``\"cpu\"`` or ``\"cuda\"``).

    Returns:
        best_model: Trained model with the best validation ROC-AUC.
        best_conf: Tuple of (hidden_dim, lr, epochs) for the best model.
        best_auc: Best validation ROC-AUC achieved during the search.
    """
    best_auc = 0.0
    best_conf: Optional[Tuple[int, float, int]] = None
    best_model: Optional[LinkClassifier] = None
    for hd in hidden_dims:
        for lr in lrs:
            for ep in epochs_list:
                model = LinkClassifier(in_dim=X_train.shape[1], hidden_dim=int(hd))
                trained = train_classifier(
                    model,
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    int(ep),
                    float(lr),
                    int(batch_size),
                    device,
                )
                trained.eval()
                with torch.no_grad():
                    val_out = trained(
                        torch.tensor(X_val, dtype=torch.float32).to(device)
                    )
                    val_prob = torch.sigmoid(val_out).cpu().numpy().flatten()
                auc = (
                    roc_auc_score(y_val.astype(float), val_prob)
                    if len(np.unique(y_val)) > 1
                    else 0.5
                )
                if auc > best_auc:
                    best_auc = auc
                    best_conf = (int(hd), float(lr), int(ep))
                    best_model = LinkClassifier(
                        in_dim=X_train.shape[1], hidden_dim=int(hd)
                    ).to(device)
                    best_model.load_state_dict(trained.state_dict())
    assert best_model is not None and best_conf is not None
    return best_model, best_conf, best_auc
