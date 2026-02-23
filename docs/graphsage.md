# GraphSAGE Backend

GraphSAGE is an alternative backend for binary link prediction on a single graph. It uses semantic text embeddings as node features and trains a two-layer GraphSAGE encoder with a configurable link decoder. Three decoder architectures are supported:

- **`mlp`** (default): a two-layer MLP over concatenated source and target embeddings. This can learn non-linear decision boundaries and typically gives the best results.
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

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/r-kowalczyk/graph-link-prediction/blob/feat/graph-sage/notebooks/graphsage_runner.ipynb)

The notebook writes artefacts to Google Drive so they persist after the Colab session ends. Once training completes, the serving bundle can be downloaded as a zip and served locally without a GPU.

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
2. Attaches the new node to the existing graph by connecting it to the top-k most similar existing nodes, measured by cosine similarity in the semantic embedding space. The value of k is set by `graphsage.attachment_top_k` in the config.
3. Runs GraphSAGE message passing over the augmented graph to produce an embedding for the new node.
4. Scores the new node against the requested target (or all candidates for top-k retrieval) using the configured decoder (MLP by default).

This attachment heuristic is deterministic (seeded) and simple by design. It assumes that semantically similar entities are likely to be structurally close in the graph. This is a reasonable starting point for biomedical knowledge graphs but is not a production graph-construction strategy.

## Configuration

GraphSAGE parameters live under the `graphsage` key in the YAML config. The relevant keys are:

| Key | Default | Description |
|-----|---------|-------------|
| `graphsage.hidden_dim` | 128 | Hidden dimension of the SAGEConv layers |
| `graphsage.output_dim` | 64 | Output embedding dimension |
| `graphsage.dropout` | 0.3 | Dropout rate between layers |
| `graphsage.learning_rate` | 0.005 | Adam optimiser learning rate |
| `graphsage.epochs` | 50 | Number of training epochs |
| `graphsage.batch_size` | 512 | Mini-batch size for edge supervision |
| `graphsage.num_neighbors` | [15, 10] | Neighbours sampled per layer (informational; the current implementation uses full-graph message passing) |
| `graphsage.negative_sampling_ratio` | 1.0 | Ratio of negative to positive edges during training |
| `graphsage.decoder_type` | mlp | Link decoder architecture: `dot_product`, `bilinear`, or `mlp` |
| `graphsage.decoder_hidden_dim` | 64 | Hidden layer width for the MLP decoder (ignored by other decoders) |
| `graphsage.attachment_top_k` | 5 | Number of similarity edges when attaching a new node |
| `graphsage.attachment_seed` | 42 | Seed for deterministic similarity-based attachment |

Set `model.backend: graphsage` in the config or pass `--model graphsage` on the command line to select this backend.

## Split strategy

Training uses PyTorch Geometric's `RandomLinkSplit` with a fixed seed to produce train, validation, and test edge sets. The message-passing graph used during training does not include validation or test positive edges, so the model cannot directly observe held-out links during forward passes. However, a random edge split can still leak neighbourhood information through shared endpoint nodes. This is a known limitation of transductive graph splits and is documented here for transparency.

## Scope and limitations

- This backend is for binary link prediction only. Predicate types are ignored.
- The random edge split with a fixed seed is reproducible but can leak neighbourhood information through shared nodes.
- The cosine-similarity attachment heuristic for new nodes is a simple demonstration approach and not a production graph-construction strategy.
- The serving API is not designed for production traffic. It loads the full graph into memory and runs inference synchronously.
