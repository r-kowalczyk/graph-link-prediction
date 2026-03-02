# GraphSAGE Backend

GraphSAGE is an alternative backend for binary link prediction on a single graph. It uses semantic text embeddings as node features, a configurable-depth GraphSAGE encoder with residual connections, and a configurable link decoder.

Three decoder architectures are supported:

- **`mlp`** (default): a two-layer MLP over concatenated source and target embeddings. This can learn non-linear decision boundaries and usually gives the best results.
- **`bilinear`**: a learnable bilinear interaction matrix. More expressive than dot product while adding only one (output_dim x output_dim) parameter matrix.
- **`dot_product`**: a parameter-free dot product. Fast but limited to linear similarity in embedding space.

The backend supports inductive inference: new entities that were not present at training time can be scored against existing nodes without retraining.

## Quickstart

Prerequisites: Python 3.12+ and `uv` installed (see [uv installation instructions](https://docs.astral.sh/uv/)).

```bash
# Create the environment
uv sync --all-groups

# Train GraphSAGE on the bundled quickstart dataset
uv run graph-lp train --config configs/quickstart.yaml --model graphsage --device cpu --seed 42

# Report metrics from the latest run
uv run graph-lp evaluate --config configs/quickstart.yaml --device cpu
```

The run folder under `artifacts_quickstart/<timestamp>/` contains:

- `graphsage_model_state.pt` (encoder and decoder weights)
- `graphsage_metadata.json` (node mappings and serving metadata)
- `graphsage_node_features.npy` and `graphsage_edge_index.npy` (graph tensors for serving)
- `metrics.json` and `curves/roc.png`, `curves/pr.png`

## Training on Colab with GPU acceleration

For the full dataset, training on Google Colab with a GPU runtime is recommended. A dedicated notebook handles installation, training, bundle export, and download:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/r-kowalczyk/graph-link-prediction/blob/feat/graph-sage-with-edges/notebooks/graphsage_runner.ipynb)

The notebook writes artefacts to Google Drive so they persist after the Colab session ends. Once training completes, the serving bundle can be downloaded as a zip and served locally without a GPU.

If you update code and then run the smoke-test cells in the same Colab session, restart the runtime first. This avoids stale in-memory imports when loading bundles created with newer model definitions.

## Serving demo

The serving workflow is: train, export the bundle, then start the API.

```bash
# 1) Train GraphSAGE
uv run graph-lp train --config configs/quickstart.yaml --model graphsage --device cpu --seed 42

# 2) Export a self-contained serving bundle from a specific run directory
uv run graph-lp export --config configs/quickstart.yaml --run-dir artifacts_quickstart/<timestamp>

# 3) Start the FastAPI service
uv run graph-lp serve --bundle-dir artifacts_quickstart/<timestamp>/serving_bundle --host 127.0.0.1 --port 8000
```

### Example requests

Score a pair of existing entities:

```bash
curl -X POST "http://127.0.0.1:8000/predict_link" \
  -H "Content-Type: application/json" \
  -d '{"entity_a_name":"Drug A","entity_b_name":"Protein P"}'
```

Score an existing entity against a new entity (inductive mode). The API attempts to resolve entity names via a public gene-name API and caches responses on disk. If external lookup does not return description text, provide `entity_a_description` or `entity_b_description` directly in the request:

```bash
curl -X POST "http://127.0.0.1:8000/predict_link" \
  -H "Content-Type: application/json" \
  -d '{"entity_a_name":"Novel Kinase K","entity_a_description":"Example kinase associated with tumour signalling","entity_b_name":"Protein P"}'
```

Retrieve top-k predicted links for one entity:

```bash
curl -X POST "http://127.0.0.1:8000/predict_links" \
  -H "Content-Type: application/json" \
  -d '{"entity_name":"Novel Kinase K","entity_description":"Example kinase associated with tumour signalling","top_k":5}'
```

Health check:

```bash
curl http://127.0.0.1:8000/healthz
```

## How inductive inference works

When the API receives a request involving an entity that is not in the training graph, it:

1. Embeds the entity text (name and description) using the same transformer model used during training.
2. Attaches the new node to the existing graph using one of two strategies:
   - `interaction` (default): fetch STRING interaction partners and attach to graph-matching partners by confidence.
   - `cosine`: attach to top-k most similar existing nodes by semantic cosine similarity.
   The value of k is set by `graphsage.attachment_top_k` in the config.
3. Runs GraphSAGE message passing over the augmented graph to produce an embedding for the new node.
4. Scores the new node against the requested target (or all candidates for top-k retrieval) using the configured decoder (MLP by default).

This design keeps inference deterministic and explicit. The `interaction` strategy gives the model real biological structure for new entities, while `cosine` remains available as a simple fallback strategy.

## Training objective and split strategy

GraphSAGE now trains on the same supervision labels as the baseline:

- **Supervision pairs and labels**: `ground_truth.csv`
- **Message-passing graph**: `edges.csv`

This decouples structural context from supervision and makes baseline vs GraphSAGE metrics directly comparable.

For leakage prevention, validation and test positive supervision edges are removed from the message-passing graph during training. The full graph is still exported in the serving bundle because inference has no train and test split.

## Model and optimisation details

The current training loop uses:

- A configurable `num_layers` GraphSAGE encoder with additive residual connections after the first layer.
- A residual projection on the final layer when `hidden_dim` and `output_dim` differ.
- Validation-AUC checkpoint selection so the exported model reflects the best validation epoch.
- Cosine annealing learning-rate scheduling across the configured epoch budget.

## Evaluating attachment strategy (cosine vs interaction)

Training ROC-AUC and test ROC-AUC are now comparable to the baseline for known entities, but attachment strategy only affects **inductive** requests where at least one endpoint is a new entity.

To evaluate `interaction` vs `cosine`, you still need an inductive set where the query entity is absent from the training graph.

Recommended process:

1. Build or source a dataset of `(new_entity, existing_entity, label)` pairs.
2. Score each pair twice through the API, once with `attachment_strategy="interaction"` and once with `attachment_strategy="cosine"`.
3. Compare ROC-AUC and PR-AUC across strategies.

The `scripts/cross_evaluate_on_ground_truth.py` script remains useful to validate comparability with baseline metrics and to compare GraphSAGE against a semantic cosine baseline on the same supervised split.

## Configuration

GraphSAGE parameters live under the `graphsage` key in the YAML config. The relevant keys are:

| Key | Default | Description |
|-----|---------|-------------|
| `graphsage.hidden_dim` | 256 | Hidden dimension of the SAGEConv layers |
| `graphsage.output_dim` | 128 | Output embedding dimension |
| `graphsage.num_layers` | 3 (full) / 2 (quickstart) | Number of GraphSAGE aggregation layers |
| `graphsage.dropout` | 0.15 (full) / 0.2 (quickstart) | Dropout rate between encoder layers and inside the MLP decoder |
| `graphsage.learning_rate` | 0.001 | Adam optimiser learning rate |
| `graphsage.weight_decay` | 0.001 (full) | L2 weight decay for the Adam optimiser |
| `graphsage.epochs` | 50 | Maximum number of training epochs (best checkpoint is selected by validation AUC) |
| `graphsage.batch_size` | 512 | Mini-batch size for edge supervision |
| `graphsage.num_neighbors` | [25, 10] | Neighbours sampled per layer (informational; the current implementation uses full-graph message passing) |
| `graphsage.decoder_type` | mlp | Link decoder architecture: `dot_product`, `bilinear`, or `mlp` |
| `graphsage.decoder_hidden_dim` | 128 | Hidden layer width for the MLP decoder (ignored by other decoders) |
| `graphsage.attachment_top_k` | 5 | Number of similarity edges when attaching a new node |
| `graphsage.attachment_seed` | 42 | Seed for deterministic similarity-based attachment |

At serving time you can override how new nodes are attached via the request body: set `attachment_strategy` to `"cosine"` (semantic similarity only) or `"interaction"` (STRING API partners, with cosine fallback when no graph matches). The default is `"interaction"`.

Set `model.backend: graphsage` in the config or pass `--model graphsage` on the command line to select this backend.

## Current results and interpretation

With the current pipeline:

- GraphSAGE trains and evaluates on the same `ground_truth.csv` pairs as the baseline, so metrics are directly comparable.
- A recent full run reached approximately **0.96 ROC-AUC** on the test split.
- The hybrid baseline remains stronger at around **0.98 ROC-AUC**, but the gap is now small.

This performance level suggests a potential deployment strategy:

- Hybrid model for highest accuracy on known entities.
- GraphSAGE model for strong inductive serving of new entities without retraining.

## Scope and limitations

- This backend is for binary link prediction only. Predicate types are ignored.
- `num_neighbors` is currently informational. The implementation still uses full-graph message passing rather than neighbour-sampled mini-batches.
- `cosine` attachment is a simple fallback strategy for new nodes and should not be treated as a production graph-construction method.
- The serving API is not designed for production traffic. It loads the full graph into memory and runs inference synchronously.
