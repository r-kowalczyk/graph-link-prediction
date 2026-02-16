"""FastAPI serving for GraphSAGE binary link prediction with inductive mode.

The server loads one exported GraphSAGE bundle on startup and exposes link
scoring endpoints for existing and newly introduced entities. New entities are
attached by top-k cosine similarity in semantic embedding space, then scored
with GraphSAGE message passing on the augmented graph.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import urllib.parse
import urllib.request
from typing import Any

from fastapi import FastAPI, HTTPException
import numpy as np
from pydantic import BaseModel, Field
import torch
from transformers import AutoModel, AutoTokenizer

from .graphsage import LoadedGraphSageBundle, load_graphsage_bundle
from .utils import ensure_dir, resolve_device


class PredictLinkRequest(BaseModel):
    """Request payload for scoring a single candidate link.

    Parameters:
        entity_a_name: Name of the first endpoint entity.
        entity_b_name: Name of the second endpoint entity.
        entity_a_description: Optional text used when endpoint A is unseen.
        entity_b_description: Optional text used when endpoint B is unseen.
        attachment_top_k: Optional override for similarity attachment degree.
    """

    entity_a_name: str = Field(min_length=1)
    entity_b_name: str = Field(min_length=1)
    entity_a_description: str | None = None
    entity_b_description: str | None = None
    attachment_top_k: int | None = None


class PredictLinksRequest(BaseModel):
    """Request payload for retrieving top-k links for one endpoint entity.

    Parameters:
        entity_name: Entity whose links should be ranked.
        entity_description: Optional text when the input entity is unseen.
        top_k: Number of predicted links to return in descending score order.
        candidate_names: Optional subset of existing entity names as candidates.
        attachment_top_k: Optional override for similarity attachment degree.
    """

    entity_name: str = Field(min_length=1)
    entity_description: str | None = None
    top_k: int = 5
    candidate_names: list[str] | None = None
    attachment_top_k: int | None = None


class GeneDescriptionResolver:
    """Resolve gene names to short descriptions using a public web endpoint.

    The resolver stores responses on disk to avoid repeated network calls across
    requests and restarts. This keeps the demo responsive while still showing a
    concrete external enrichment step for unseen biomedical entities.
    """

    def __init__(self, cache_path: str) -> None:
        """Initialise resolver cache state from a JSON file on disk.

        Parameters:
            cache_path: JSON file path used to persist resolved descriptions.
        """
        self.cache_path = cache_path
        ensure_dir(os.path.dirname(cache_path))
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as cache_file:
                self.cache: dict[str, str] = {
                    str(key): str(value) for key, value in json.load(cache_file).items()
                }
        else:
            self.cache = {}

    def resolve(self, entity_name: str) -> str | None:
        """Resolve a name to description text, returning None on lookup failure.

        Parameters:
            entity_name: Free-text gene name provided by the API caller.

        Returns:
            A description string if lookup succeeds, otherwise None.
        """
        cache_key = entity_name.strip().casefold()
        if cache_key in self.cache:
            return self.cache[cache_key]

        encoded_query = urllib.parse.quote(entity_name)
        query_url = (
            "https://mygene.info/v3/query?"
            f"q={encoded_query}&species=human&size=1&fields=name,summary,symbol"
        )

        # The API result is optional, so lookup errors map to None and the caller
        # is required to provide explicit description text in the request.
        try:
            with urllib.request.urlopen(query_url, timeout=10) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception:
            return None

        hits = payload.get("hits") if isinstance(payload, dict) else None
        if not hits:
            return None

        first_hit = hits[0]
        resolved_description = str(
            first_hit.get("summary") or first_hit.get("name") or ""
        ).strip()
        if not resolved_description:
            return None

        self.cache[cache_key] = resolved_description
        with open(self.cache_path, "w", encoding="utf-8") as cache_file:
            json.dump(self.cache, cache_file, indent=2, sort_keys=True)
        return resolved_description


@dataclass
class ResolvedEntity:
    """Container for existing or newly introduced entity resolution state.

    This object keeps endpoint resolution compact by carrying whether the entity
    already exists in the training graph, its resolved node id, and optional
    semantic embedding used for inductive new-node attachment.
    """

    requested_name: str
    node_id: str | None
    node_index: int | None
    is_new: bool
    embedding_vector: np.ndarray | None


class GraphSageServingEngine:
    """Serving engine that scores links with a loaded GraphSAGE model bundle.

    The engine performs deterministic new-node attachment and computes link
    probabilities for both pair scoring and top-k retrieval endpoints. It keeps
    model, graph, and resolver state in memory for low-latency API requests.
    """

    def __init__(self, bundle_directory: str, device_name: str = "cpu") -> None:
        """Load serving bundle and semantic embedder resources once at startup.

        Parameters:
            bundle_directory: Exported GraphSAGE bundle directory path.
            device_name: Torch device policy string such as cpu or auto.
        """
        self.device = resolve_device(device_name)
        self.bundle: LoadedGraphSageBundle = load_graphsage_bundle(
            bundle_directory=bundle_directory,
            device=self.device,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.bundle.semantic_model_name, use_fast=True
        )
        self.semantic_model = AutoModel.from_pretrained(
            self.bundle.semantic_model_name
        ).to(self.device)
        self.semantic_model.eval()
        self.node_name_to_id_casefold = {
            entity_name.casefold(): node_id
            for entity_name, node_id in self.bundle.node_name_to_id.items()
        }
        self.resolver = GeneDescriptionResolver(
            cache_path=os.path.join(bundle_directory, "resolver_cache.json")
        )
        with torch.no_grad():
            self.base_node_embeddings = self.bundle.model.encode(
                node_features=self.bundle.node_features,
                edge_index=self.bundle.edge_index,
            )

    def _embed_text(self, entity_name: str, description_text: str) -> np.ndarray:
        """Embed an entity text description using the configured transformer model.

        Parameters:
            entity_name: Entity name used as contextual prefix in the text input.
            description_text: Plain text description for semantic encoding.

        Returns:
            A one-dimensional float32 semantic embedding vector.
        """
        combined_text = f"{entity_name} {description_text}".strip()
        encoded_batch = self.tokenizer(
            [combined_text],
            padding=True,
            truncation=True,
            max_length=int(self.bundle.semantic_max_length),
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            embedding_tensor = self.semantic_model(**encoded_batch).last_hidden_state[
                :, 0, :
            ]
        return embedding_tensor.squeeze(0).detach().cpu().numpy().astype(np.float32)

    def _resolve_entity(
        self,
        entity_name: str,
        provided_description: str | None,
    ) -> ResolvedEntity:
        """Resolve one request entity to existing node id or new-node embedding.

        Parameters:
            entity_name: Name provided in API request payload.
            provided_description: Optional caller-provided description text.

        Returns:
            ResolvedEntity containing existing id or new embedding details.
        """
        existing_node_id = self.node_name_to_id_casefold.get(entity_name.casefold())
        if existing_node_id is not None:
            existing_node_index = self.bundle.node_id_to_index[existing_node_id]
            return ResolvedEntity(
                requested_name=entity_name,
                node_id=existing_node_id,
                node_index=existing_node_index,
                is_new=False,
                embedding_vector=None,
            )

        description_text = (
            provided_description
            if provided_description is not None
            else self.resolver.resolve(entity_name)
        )
        if description_text is None:
            raise ValueError(
                "External lookup did not return description text. "
                "Provide description text directly in the request."
            )
        embedding_vector = self._embed_text(entity_name, description_text)
        return ResolvedEntity(
            requested_name=entity_name,
            node_id=None,
            node_index=None,
            is_new=True,
            embedding_vector=embedding_vector,
        )

    def _select_attachment_neighbours(
        self, embedding_vector: np.ndarray, top_k: int
    ) -> list[int]:
        """Select deterministic top-k attachment neighbours by cosine similarity.

        Parameters:
            embedding_vector: New node semantic embedding vector.
            top_k: Number of existing neighbours used for graph attachment.

        Returns:
            Existing node indices sorted by descending similarity.
        """
        existing_features = self.bundle.node_features.detach().cpu().numpy()
        existing_norms = np.linalg.norm(existing_features, axis=1) + 1e-12
        new_norm = np.linalg.norm(embedding_vector) + 1e-12
        cosine_similarities = (existing_features @ embedding_vector) / (
            existing_norms * new_norm
        )

        # Seeded permutation provides deterministic tie handling and neighbour order.
        seeded_random_generator = np.random.default_rng(
            int(self.bundle.attachment_seed)
        )
        seeded_permutation = seeded_random_generator.permutation(
            existing_features.shape[0]
        )
        sorted_positions = np.argsort(
            -cosine_similarities[seeded_permutation], kind="stable"
        )
        ordered_indices = seeded_permutation[sorted_positions]
        return [int(index_value) for index_value in ordered_indices[: int(top_k)]]

    def _build_augmented_graph(
        self, embedding_vector: np.ndarray, attachment_top_k: int
    ) -> tuple[torch.Tensor, torch.Tensor, int, list[int]]:
        """Create augmented node feature and edge tensors for one new entity.

        Parameters:
            embedding_vector: New entity semantic embedding vector.
            attachment_top_k: Number of similarity edges added to existing graph.

        Returns:
            Augmented node features, augmented edges, new node index, neighbour list.
        """
        selected_neighbour_indices = self._select_attachment_neighbours(
            embedding_vector=embedding_vector,
            top_k=attachment_top_k,
        )
        new_node_index = int(self.bundle.node_features.size(0))
        new_node_tensor = torch.as_tensor(
            embedding_vector, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        augmented_node_features = torch.cat(
            [self.bundle.node_features, new_node_tensor], dim=0
        )

        attachment_edges: list[tuple[int, int]] = [
            (new_node_index, neighbour_index)
            for neighbour_index in selected_neighbour_indices
        ]
        if self.bundle.undirected:
            attachment_edges.extend(
                [
                    (neighbour_index, new_node_index)
                    for neighbour_index in selected_neighbour_indices
                ]
            )
        if attachment_edges:
            attachment_edge_tensor = torch.as_tensor(
                np.asarray(attachment_edges, dtype=np.int64).T,
                dtype=torch.long,
                device=self.device,
            )
            augmented_edge_index = torch.cat(
                [self.bundle.edge_index, attachment_edge_tensor], dim=1
            )
        else:
            augmented_edge_index = self.bundle.edge_index
        return (
            augmented_node_features,
            augmented_edge_index,
            new_node_index,
            selected_neighbour_indices,
        )

    def _score_from_embeddings(
        self,
        node_embeddings: torch.Tensor,
        source_index: int,
        target_index: int,
    ) -> float:
        """Compute sigmoid link score for one pair using node embeddings.

        Parameters:
            node_embeddings: Embedding matrix that includes requested endpoints.
            source_index: Integer index of source endpoint in embeddings.
            target_index: Integer index of target endpoint in embeddings.

        Returns:
            Probability score in the range [0, 1].
        """
        source_embedding = node_embeddings[source_index]
        target_embedding = node_embeddings[target_index]
        logit = torch.sum(source_embedding * target_embedding)
        return float(torch.sigmoid(logit).item())

    def predict_link(
        self,
        entity_a_name: str,
        entity_b_name: str,
        entity_a_description: str | None,
        entity_b_description: str | None,
        attachment_top_k: int | None,
    ) -> dict[str, Any]:
        """Score one link between existing or inductively attached endpoints.

        Parameters:
            entity_a_name: Name of the first endpoint entity.
            entity_b_name: Name of the second endpoint entity.
            entity_a_description: Optional text for unseen endpoint A.
            entity_b_description: Optional text for unseen endpoint B.
            attachment_top_k: Optional neighbour count override for attachment.

        Returns:
            Response mapping with score and resolved endpoint identifiers.
        """
        resolved_entity_a = self._resolve_entity(entity_a_name, entity_a_description)
        resolved_entity_b = self._resolve_entity(entity_b_name, entity_b_description)
        if resolved_entity_a.is_new and resolved_entity_b.is_new:
            raise ValueError("At least one endpoint must already exist in the graph.")

        selected_attachment_top_k = int(
            attachment_top_k
            if attachment_top_k is not None
            else self.bundle.attachment_top_k
        )

        if not resolved_entity_a.is_new and not resolved_entity_b.is_new:
            score_value = self._score_from_embeddings(
                node_embeddings=self.base_node_embeddings,
                source_index=int(resolved_entity_a.node_index),
                target_index=int(resolved_entity_b.node_index),
            )
            return {
                "score": score_value,
                "entity_a": {
                    "name": resolved_entity_a.requested_name,
                    "node_id": resolved_entity_a.node_id,
                    "is_new": False,
                },
                "entity_b": {
                    "name": resolved_entity_b.requested_name,
                    "node_id": resolved_entity_b.node_id,
                    "is_new": False,
                },
                "attachment_neighbours": [],
            }

        existing_entity = (
            resolved_entity_a if not resolved_entity_a.is_new else resolved_entity_b
        )
        new_entity = (
            resolved_entity_a if resolved_entity_a.is_new else resolved_entity_b
        )
        (
            augmented_node_features,
            augmented_edge_index,
            new_node_index,
            selected_neighbour_indices,
        ) = self._build_augmented_graph(
            embedding_vector=np.asarray(new_entity.embedding_vector),
            attachment_top_k=selected_attachment_top_k,
        )
        with torch.no_grad():
            augmented_embeddings = self.bundle.model.encode(
                node_features=augmented_node_features,
                edge_index=augmented_edge_index,
            )
        score_value = self._score_from_embeddings(
            node_embeddings=augmented_embeddings,
            source_index=new_node_index,
            target_index=int(existing_entity.node_index),
        )
        attachment_neighbours = [
            {
                "node_id": self.bundle.index_to_node_id[index_value],
                "name": self.bundle.node_display_name_by_id[
                    self.bundle.index_to_node_id[index_value]
                ],
            }
            for index_value in selected_neighbour_indices
        ]
        return {
            "score": score_value,
            "entity_a": {
                "name": resolved_entity_a.requested_name,
                "node_id": resolved_entity_a.node_id,
                "is_new": bool(resolved_entity_a.is_new),
            },
            "entity_b": {
                "name": resolved_entity_b.requested_name,
                "node_id": resolved_entity_b.node_id,
                "is_new": bool(resolved_entity_b.is_new),
            },
            "attachment_neighbours": attachment_neighbours,
        }

    def predict_links(
        self,
        entity_name: str,
        entity_description: str | None,
        top_k: int,
        candidate_names: list[str] | None,
        attachment_top_k: int | None,
    ) -> dict[str, Any]:
        """Return top-k predicted links from one endpoint to existing nodes.

        Parameters:
            entity_name: Query entity name that acts as source for ranking.
            entity_description: Optional description text for unseen query entities.
            top_k: Number of top links returned in the response payload.
            candidate_names: Optional existing-name subset used as candidate pool.
            attachment_top_k: Optional neighbour count override for attachment.

        Returns:
            Response mapping with ranked links and resolved source information.
        """
        resolved_entity = self._resolve_entity(entity_name, entity_description)
        selected_attachment_top_k = int(
            attachment_top_k
            if attachment_top_k is not None
            else self.bundle.attachment_top_k
        )

        attachment_neighbours: list[dict[str, str]] = []
        if resolved_entity.is_new:
            (
                augmented_node_features,
                augmented_edge_index,
                source_index,
                selected_neighbour_indices,
            ) = self._build_augmented_graph(
                embedding_vector=np.asarray(resolved_entity.embedding_vector),
                attachment_top_k=selected_attachment_top_k,
            )
            with torch.no_grad():
                scoring_embeddings = self.bundle.model.encode(
                    node_features=augmented_node_features,
                    edge_index=augmented_edge_index,
                )
            attachment_neighbours = [
                {
                    "node_id": self.bundle.index_to_node_id[index_value],
                    "name": self.bundle.node_display_name_by_id[
                        self.bundle.index_to_node_id[index_value]
                    ],
                }
                for index_value in selected_neighbour_indices
            ]
        else:
            source_index = int(resolved_entity.node_index)
            scoring_embeddings = self.base_node_embeddings

        if candidate_names is None:
            candidate_indices = list(range(len(self.bundle.index_to_node_id)))
        else:
            candidate_indices = []
            for candidate_name in candidate_names:
                candidate_node_id = self.node_name_to_id_casefold.get(
                    candidate_name.casefold()
                )
                if candidate_node_id is not None:
                    candidate_indices.append(
                        self.bundle.node_id_to_index[candidate_node_id]
                    )

        if not resolved_entity.is_new:
            candidate_indices = [
                candidate_index
                for candidate_index in candidate_indices
                if candidate_index != source_index
            ]

        source_embedding = scoring_embeddings[source_index]
        candidate_embeddings = scoring_embeddings[candidate_indices]
        logits = (candidate_embeddings * source_embedding).sum(dim=-1)
        probabilities = torch.sigmoid(logits).detach().cpu().numpy()
        ranking_order = np.argsort(-probabilities)
        selected_indices = ranking_order[: int(top_k)]

        top_links = []
        for ranking_index in selected_indices:
            candidate_index = int(candidate_indices[int(ranking_index)])
            candidate_node_id = self.bundle.index_to_node_id[candidate_index]
            top_links.append(
                {
                    "node_id": candidate_node_id,
                    "name": self.bundle.node_display_name_by_id[candidate_node_id],
                    "score": float(probabilities[int(ranking_index)]),
                }
            )

        return {
            "entity": {
                "name": resolved_entity.requested_name,
                "node_id": resolved_entity.node_id,
                "is_new": bool(resolved_entity.is_new),
            },
            "top_links": top_links,
            "attachment_neighbours": attachment_neighbours,
        }


def create_app(bundle_directory: str, device_name: str = "cpu") -> FastAPI:
    """Create a FastAPI app instance backed by one loaded GraphSAGE bundle.

    Parameters:
        bundle_directory: Exported GraphSAGE serving bundle directory path.
        device_name: Torch device policy string used for model execution.

    Returns:
        Configured FastAPI application with prediction endpoints.
    """
    serving_engine = GraphSageServingEngine(
        bundle_directory=bundle_directory,
        device_name=device_name,
    )
    application = FastAPI(title="Graph Link Prediction Service")
    application.state.engine = serving_engine

    @application.get("/healthz")
    def healthz() -> dict[str, str]:
        """Return a basic health status for service monitoring checks."""
        return {"status": "ok"}

    @application.post("/predict_link")
    def predict_link(request: PredictLinkRequest) -> dict[str, Any]:
        """Score one link between two entities using existing or inductive mode."""
        try:
            return application.state.engine.predict_link(
                entity_a_name=request.entity_a_name,
                entity_b_name=request.entity_b_name,
                entity_a_description=request.entity_a_description,
                entity_b_description=request.entity_b_description,
                attachment_top_k=request.attachment_top_k,
            )
        except ValueError as exception:
            raise HTTPException(status_code=400, detail=str(exception)) from exception

    @application.post("/predict_links")
    def predict_links(request: PredictLinksRequest) -> dict[str, Any]:
        """Return top-k predicted links from one source entity to existing nodes."""
        try:
            return application.state.engine.predict_links(
                entity_name=request.entity_name,
                entity_description=request.entity_description,
                top_k=request.top_k,
                candidate_names=request.candidate_names,
                attachment_top_k=request.attachment_top_k,
            )
        except ValueError as exception:
            raise HTTPException(status_code=400, detail=str(exception)) from exception

    return application


def run_server(bundle_directory: str, host: str, port: int) -> None:
    """Run the FastAPI service process with one exported GraphSAGE bundle.

    Parameters:
        bundle_directory: Exported GraphSAGE bundle used for serving startup.
        host: Host interface bound by the server process.
        port: TCP port exposed by the server process.
    """
    import uvicorn

    application = create_app(bundle_directory=bundle_directory, device_name="cpu")
    uvicorn.run(application, host=host, port=port)
