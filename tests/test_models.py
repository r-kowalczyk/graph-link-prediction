"""Unit tests for baseline models: Logistic Regression and MLP."""

import numpy as np
import torch

from graph_lp.models import (
    LinkClassifier,
    LogRegModel,
    hyperparam_search_mlp,
    train_classifier,
)


def test_logreg_model_fit_and_predict():
    """Test that LogRegModel can fit and predict probabilities."""
    model = LogRegModel()
    X = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]])
    y = np.array([0, 0, 1, 1])
    model.fit(X, y)
    proba = model.predict_proba(X)
    assert len(proba) == 4
    assert all(0 <= p <= 1 for p in proba)


def test_link_classifier_forward():
    """Test that LinkClassifier produces logits of the correct shape."""
    model = LinkClassifier(in_dim=10, hidden_dim=32)
    x = torch.randn(5, 10)
    out = model(x)
    assert out.shape == (5, 1)


def test_train_classifier():
    """Test that train_classifier trains a model and returns it."""
    model = LinkClassifier(in_dim=4, hidden_dim=8)
    X_train = np.random.randn(20, 4).astype(np.float32)
    y_train = np.random.randint(0, 2, 20).astype(np.float32)
    X_val = np.random.randn(10, 4).astype(np.float32)
    y_val = np.random.randint(0, 2, 10).astype(np.float32)
    trained = train_classifier(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=2,
        lr=0.01,
        batch_size=4,
        device="cpu",
    )
    assert isinstance(trained, LinkClassifier)
    trained.eval()
    with torch.no_grad():
        out = trained(torch.tensor(X_val, dtype=torch.float32))
        assert out.shape == (10, 1)


def test_train_classifier_constant_labels():
    """Test that train_classifier handles constant validation labels."""
    model = LinkClassifier(in_dim=4, hidden_dim=8)
    X_train = np.random.randn(20, 4).astype(np.float32)
    y_train = np.random.randint(0, 2, 20).astype(np.float32)
    X_val = np.random.randn(10, 4).astype(np.float32)
    y_val = np.zeros(10, dtype=np.float32)
    trained = train_classifier(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=2,
        lr=0.01,
        batch_size=4,
        device="cpu",
    )
    assert isinstance(trained, LinkClassifier)


def test_hyperparam_search_mlp():
    """Test that hyperparam_search_mlp returns the best model and configuration."""
    X_train = np.random.randn(30, 4).astype(np.float32)
    y_train = np.random.randint(0, 2, 30).astype(np.float32)
    X_val = np.random.randn(10, 4).astype(np.float32)
    y_val = np.random.randint(0, 2, 10).astype(np.float32)
    best_model, best_conf, best_auc = hyperparam_search_mlp(
        X_train,
        y_train,
        X_val,
        y_val,
        hidden_dims=[8],
        lrs=[0.01],
        epochs_list=[2],
        batch_size=4,
        device="cpu",
    )
    assert isinstance(best_model, LinkClassifier)
    assert isinstance(best_conf, tuple)
    assert len(best_conf) == 3
    assert isinstance(best_auc, float)
    assert 0 <= best_auc <= 1


def test_hyperparam_search_mlp_constant_labels():
    """Test that hyperparam_search_mlp handles constant validation labels."""
    X_train = np.random.randn(30, 4).astype(np.float32)
    y_train = np.random.randint(0, 2, 30).astype(np.float32)
    X_val = np.random.randn(10, 4).astype(np.float32)
    y_val = np.ones(10, dtype=np.float32)
    best_model, best_conf, best_auc = hyperparam_search_mlp(
        X_train,
        y_train,
        X_val,
        y_val,
        hidden_dims=[8],
        lrs=[0.01],
        epochs_list=[2],
        batch_size=4,
        device="cpu",
    )
    assert isinstance(best_model, LinkClassifier)
    assert best_auc == 0.5
