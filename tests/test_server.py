"""Unit tests for FastAPI serving and external resolver helpers."""

from __future__ import annotations

import json
import types
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest
import torch

from graph_lp.graphsage import GraphSageLinkPredictor, LoadedGraphSageBundle
from graph_lp.server import (
    GeneDescriptionResolver,
    GraphSageServingEngine,
    create_app,
    run_server,
)


class _FakeEncodedInputs(dict):
    def to(self, _device):
        return self


class _FakeTokenizer:
    def __call__(
        self,
        texts,
        padding,
        truncation,
        max_length,
        return_tensors,
    ):
        _ = (padding, truncation, max_length, return_tensors)
        return _FakeEncodedInputs(
            {
                "input_ids": torch.ones((len(texts), 3), dtype=torch.long),
                "attention_mask": torch.ones((len(texts), 3), dtype=torch.long),
            }
        )


class _FakeSemanticModel:
    def to(self, _device):
        return self

    def eval(self):
        return self

    def __call__(self, **kwargs):
        batch_size = int(kwargs["input_ids"].size(0))
        hidden_state = torch.ones((batch_size, 1, 4), dtype=torch.float32)
        return types.SimpleNamespace(last_hidden_state=hidden_state)


def _build_fake_bundle() -> LoadedGraphSageBundle:
    """Create a deterministic in-memory serving bundle for API tests."""

    torch.manual_seed(7)
    model = GraphSageLinkPredictor(
        input_dimension=4,
        hidden_dimension=4,
        output_dimension=4,
        dropout_rate=0.0,
    )
    model.eval()
    node_features = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    edge_index = torch.tensor(
        [[0, 1, 1, 2], [1, 0, 2, 1]],
        dtype=torch.long,
    )
    return LoadedGraphSageBundle(
        model=model,
        node_features=node_features,
        edge_index=edge_index,
        node_id_to_index={"n0": 0, "n1": 1, "n2": 2},
        index_to_node_id=["n0", "n1", "n2"],
        node_name_to_id={"Node A": "n0", "Node B": "n1", "Node C": "n2"},
        node_display_name_by_id={"n0": "Node A", "n1": "Node B", "n2": "Node C"},
        semantic_model_name="fake-model",
        semantic_max_length=16,
        undirected=True,
        attachment_seed=42,
        attachment_top_k=2,
    )


class _DummyResponse:
    def __init__(self, payload):
        self.payload = payload

    def read(self):
        return self.payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        _ = (exc_type, exc, tb)
        return False


def test_gene_description_resolver_reads_existing_cache(tmp_path):
    """Resolver should return cached value without making a network call."""

    cache_path = tmp_path / "resolver_cache.json"
    cache_path.write_text(
        json.dumps({"gene x": "cached description"}), encoding="utf-8"
    )
    resolver = GeneDescriptionResolver(str(cache_path))
    assert resolver.resolve("Gene X") == "cached description"


def test_gene_description_resolver_fetches_and_persists(tmp_path):
    """Resolver should store successful web lookup responses on disk."""

    cache_path = tmp_path / "resolver_cache.json"
    resolver = GeneDescriptionResolver(str(cache_path))
    payload = json.dumps({"hits": [{"summary": "resolved summary"}]}).encode("utf-8")
    with patch(
        "urllib.request.urlopen",
        return_value=_DummyResponse(payload),
    ):
        resolved = resolver.resolve("Gene Y")
    assert resolved == "resolved summary"
    stored_cache = json.loads(cache_path.read_text(encoding="utf-8"))
    assert stored_cache["gene y"] == "resolved summary"


def test_gene_description_resolver_returns_none_for_error_and_empty_hits(tmp_path):
    """Resolver should return None on request failure and empty payload."""

    cache_path = tmp_path / "resolver_cache.json"
    resolver = GeneDescriptionResolver(str(cache_path))
    with patch("urllib.request.urlopen", side_effect=RuntimeError("failure")):
        assert resolver.resolve("Gene Z") is None
    with patch(
        "urllib.request.urlopen",
        return_value=_DummyResponse(json.dumps({"hits": []}).encode("utf-8")),
    ):
        assert resolver.resolve("Gene Z") is None
    with patch(
        "urllib.request.urlopen",
        return_value=_DummyResponse(
            json.dumps({"hits": [{"summary": ""}]}).encode("utf-8")
        ),
    ):
        assert resolver.resolve("Gene Z") is None


def _create_engine(tmp_path, monkeypatch) -> GraphSageServingEngine:
    """Create a serving engine with mocked bundle and transformer dependencies."""

    monkeypatch.setattr(
        "graph_lp.server.load_graphsage_bundle", lambda **_kwargs: _build_fake_bundle()
    )
    monkeypatch.setattr(
        "graph_lp.server.AutoTokenizer.from_pretrained",
        lambda *_args, **_kwargs: _FakeTokenizer(),
    )
    monkeypatch.setattr(
        "graph_lp.server.AutoModel.from_pretrained",
        lambda *_args, **_kwargs: _FakeSemanticModel(),
    )
    bundle_directory = tmp_path / "bundle"
    bundle_directory.mkdir(parents=True, exist_ok=True)
    return GraphSageServingEngine(str(bundle_directory), "cpu")


def test_serving_engine_predict_link_existing_and_new_paths(tmp_path, monkeypatch):
    """Engine should score existing pairs and inductive new-node pairs."""

    engine = _create_engine(tmp_path, monkeypatch)
    existing_result = engine.predict_link(
        entity_a_name="Node A",
        entity_b_name="Node B",
        entity_a_description=None,
        entity_b_description=None,
        attachment_top_k=None,
    )
    assert 0.0 <= existing_result["score"] <= 1.0
    assert existing_result["attachment_neighbours"] == []

    new_result = engine.predict_link(
        entity_a_name="New Gene",
        entity_b_name="Node B",
        entity_a_description="Synthetic description",
        entity_b_description=None,
        attachment_top_k=2,
    )
    assert 0.0 <= new_result["score"] <= 1.0
    assert new_result["entity_a"]["is_new"] is True
    assert len(new_result["attachment_neighbours"]) == 2

    with pytest.raises(ValueError):
        engine.predict_link(
            entity_a_name="New A",
            entity_b_name="New B",
            entity_a_description="Text A",
            entity_b_description="Text B",
            attachment_top_k=1,
        )


def test_serving_engine_predict_links_and_missing_description_error(
    tmp_path, monkeypatch
):
    """Engine should rank top links and fail if unresolved text is missing."""

    engine = _create_engine(tmp_path, monkeypatch)

    existing_result = engine.predict_links(
        entity_name="Node A",
        entity_description=None,
        top_k=2,
        candidate_names=None,
        attachment_top_k=None,
    )
    assert existing_result["entity"]["is_new"] is False
    assert len(existing_result["top_links"]) == 2

    new_result = engine.predict_links(
        entity_name="New Node",
        entity_description="Provided text",
        top_k=1,
        candidate_names=["Node B", "Node C"],
        attachment_top_k=0,
    )
    assert new_result["entity"]["is_new"] is True
    assert len(new_result["top_links"]) == 1
    assert new_result["attachment_neighbours"] == []

    monkeypatch.setattr(engine.resolver, "resolve", lambda _name: None)
    with pytest.raises(ValueError):
        engine.predict_links(
            entity_name="Unknown Node",
            entity_description=None,
            top_k=1,
            candidate_names=None,
            attachment_top_k=1,
        )


def test_create_app_endpoints_success_and_validation_error(tmp_path, monkeypatch):
    """FastAPI app should expose health and prediction endpoints."""

    monkeypatch.setattr(
        "graph_lp.server.load_graphsage_bundle", lambda **_kwargs: _build_fake_bundle()
    )
    monkeypatch.setattr(
        "graph_lp.server.AutoTokenizer.from_pretrained",
        lambda *_args, **_kwargs: _FakeTokenizer(),
    )
    monkeypatch.setattr(
        "graph_lp.server.AutoModel.from_pretrained",
        lambda *_args, **_kwargs: _FakeSemanticModel(),
    )
    app = create_app(bundle_directory=str(tmp_path), device_name="cpu")
    client = TestClient(app)

    health_response = client.get("/healthz")
    assert health_response.status_code == 200
    assert health_response.json()["status"] == "ok"

    predict_link_response = client.post(
        "/predict_link",
        json={"entity_a_name": "Node A", "entity_b_name": "Node B"},
    )
    assert predict_link_response.status_code == 200
    assert "score" in predict_link_response.json()

    predict_link_error_response = client.post(
        "/predict_link",
        json={
            "entity_a_name": "New A",
            "entity_b_name": "New B",
            "entity_a_description": "Text A",
            "entity_b_description": "Text B",
        },
    )
    assert predict_link_error_response.status_code == 400

    predict_links_response = client.post(
        "/predict_links",
        json={"entity_name": "Node A", "top_k": 2},
    )
    assert predict_links_response.status_code == 200
    assert len(predict_links_response.json()["top_links"]) == 2

    with patch.object(app.state.engine.resolver, "resolve", return_value=None):
        predict_links_error_response = client.post(
            "/predict_links",
            json={"entity_name": "Unknown", "top_k": 1},
        )
    assert predict_links_error_response.status_code == 400


def test_run_server_calls_uvicorn_with_created_app(monkeypatch):
    """run_server should pass the created app into uvicorn.run."""

    created_application = FastAPI()
    captured_call: dict[str, object] = {}
    monkeypatch.setattr(
        "graph_lp.server.create_app",
        lambda bundle_directory, device_name: created_application,
    )

    def capture_uvicorn_run(application, host, port):
        captured_call["application"] = application
        captured_call["host"] = host
        captured_call["port"] = port

    monkeypatch.setattr("uvicorn.run", capture_uvicorn_run)
    run_server(bundle_directory="bundle-path", host="0.0.0.0", port=9999)
    assert captured_call == {
        "application": created_application,
        "host": "0.0.0.0",
        "port": 9999,
    }
