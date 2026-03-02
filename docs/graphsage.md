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

This attachment heuristic is deterministic (seeded) and simple by design. It assumes that semantically similar entities are likely to be structurally close in the graph. The server can also attach new nodes using **real interaction edges** from the STRING API (see `attachment_strategy` below). That gives the model genuine structural context instead of text-similarity alone.

## Evaluating attachment strategy (cosine vs interaction)

The test ROC-AUC reported during training is **transductive**: it measures how well the model scores edges between nodes that were all in the training graph. It does **not** measure how well the model does when one endpoint is a **new** (unseen) entity. The choice of attachment strategy (cosine similarity vs STRING interaction partners) only affects inductive requests, so to see whether the interaction strategy improves performance you need an **inductive evaluation**.

### What to measure

For a set of "new" entities with known links to nodes in the graph:

1. Score each (new_entity, existing_entity) pair with the serving engine using `attachment_strategy="cosine"` and again with `attachment_strategy="interaction"`.
2. Build binary labels: 1 for true links, 0 for negatives (e.g. random existing nodes that are not true neighbours).
3. Compute ROC-AUC and PR-AUC per strategy over those scores. Compare the two strategies.

### How to get an inductive test set

**Option A: Hold out nodes at training time.** Split nodes into training nodes (e.g. 90%) and inductive test nodes (10%). Build the graph and train GraphSAGE using only training nodes and edges between them. Export the bundle. For each inductive test node you have ground truth: its real edges to training nodes (from your original edge list). Sample negatives (test node, random training node with no edge). This gives you a list of (new_entity_name, new_entity_description, existing_entity_name, label). Running this requires a training pipeline that supports excluding a fraction of nodes (and their edges) from the graph; that is not implemented in the current codebase, so you would need to add a node hold-out step or train on a manually reduced dataset.

**Option B: Use the evaluation script with a hand-built CSV.** If you have an external list of new entities and their known links (e.g. from a different database or a manual curation), put them in a CSV with columns: `new_entity_name`, `new_entity_description`, `existing_entity_name`, `label` (0 or 1). Run the comparison script (see below) against a bundle trained on your full graph. Note: the "new" entities must not appear in the bundle's graph, otherwise the server will resolve them as existing nodes and attachment strategy will not apply.

### Comparison script

The repository includes a script that loads a serving bundle and a CSV of inductive pairs, runs the engine with both attachment strategies, and reports metrics for each:

```bash
uv run python scripts/compare_attachment_strategies.py \
  --bundle-dir path/to/serving_bundle \
  --inductive-csv path/to/inductive_test_pairs.csv
```

The CSV must have columns: `new_entity_name`, `new_entity_description`, `existing_entity_name`, `label`. The script prints ROC-AUC and PR-AUC for `cosine` and `interaction` so you can compare. See the script docstring for details.

## Configuration

GraphSAGE parameters live under the `graphsage` key in the YAML config. The relevant keys are:

| Key | Default | Description |
|-----|---------|-------------|
| `graphsage.hidden_dim` | 256 | Hidden dimension of the SAGEConv layers |
| `graphsage.output_dim` | 128 | Output embedding dimension |
| `graphsage.dropout` | 0.3 | Dropout rate between encoder layers and inside the MLP decoder |
| `graphsage.learning_rate` | 0.001 | Adam optimiser learning rate |
| `graphsage.weight_decay` | 0.01 | L2 weight decay for the Adam optimiser (primary overfitting control) |
| `graphsage.epochs` | 50 | Maximum number of training epochs (best checkpoint is selected by validation AUC) |
| `graphsage.batch_size` | 512 | Mini-batch size for edge supervision |
| `graphsage.num_neighbors` | [25, 10] | Neighbours sampled per layer (informational; the current implementation uses full-graph message passing) |
| `graphsage.negative_sampling_ratio` | 1.0 | Ratio of negative to positive edges during training |
| `graphsage.decoder_type` | mlp | Link decoder architecture: `dot_product`, `bilinear`, or `mlp` |
| `graphsage.decoder_hidden_dim` | 128 | Hidden layer width for the MLP decoder (ignored by other decoders) |
| `graphsage.attachment_top_k` | 5 | Number of similarity edges when attaching a new node |
| `graphsage.attachment_seed` | 42 | Seed for deterministic similarity-based attachment |

At serving time you can override how new nodes are attached via the request body: set `attachment_strategy` to `"cosine"` (semantic similarity only) or `"interaction"` (STRING API partners, with cosine fallback when no graph matches). The default is `"interaction"`.

Set `model.backend: graphsage` in the config or pass `--model graphsage` on the command line to select this backend.

## Split strategy

Training uses PyTorch Geometric's `RandomLinkSplit` with a fixed seed to produce train, validation, and test edge sets. The message-passing graph used during training does not include validation or test positive edges, so the model cannot directly observe held-out links during forward passes. However, a random edge split can still leak neighbourhood information through shared endpoint nodes. This is a known limitation of transductive graph splits and is documented here for transparency.

---

## Initial approach: results and iterations

This section documents the development of the GraphSAGE backend, the sequence of experiments run on the full dataset with `michiyasunaga/BioLinkBERT-large` as the semantic embedding model, and the reasoning behind each change. The goal was to build an inductive link prediction model that could score new entities without retraining while achieving reasonable predictive performance.

### Context: the hybrid baseline

The existing hybrid pipeline (Node2Vec structural embeddings + transformer semantic embeddings, trained with Logistic Regression and MLP classifiers) achieves 0.98 test ROC-AUC on this dataset. That pipeline works well for entities already present in the graph, but is not inductive: Node2Vec embeddings cannot be computed for unseen entities without retraining on the modified graph.

GraphSAGE was introduced as an alternative that trades some predictive accuracy for the ability to handle unseen entities at serving time. The expected outcome was a lower AUC than the hybrid baseline, but a practical inductive serving capability.

### Run 1: dot-product decoder (initial implementation)

The first implementation used a parameter-free dot-product decoder. The dot product was chosen deliberately for transparency: it introduces no learnable parameters into the decoder, so any predictive signal must come from the GraphSAGE encoder itself. This made it easier to verify that the message-passing layers were producing useful embeddings.

Configuration: `hidden_dim=256, output_dim=128, dropout=0.2, lr=0.001, epochs=20, neg_ratio=1.0`

```
Test ROC-AUC: 0.7080
Test PR-AUC: 0.7329
```

The loss curve showed steady decrease through all 20 epochs with no sign of convergence, suggesting the model had not fully trained. The 0.71 AUC was a reasonable first result but left a large gap to the hybrid baseline.

### Run 2: MLP decoder (replacing dot product)

The dot-product decoder can only capture linear similarity in embedding space. Since the hybrid baseline's MLP classifier clearly benefits from non-linear decision boundaries, the decoder was replaced with a configurable two-layer MLP that concatenates source and target embeddings and passes them through a hidden layer with ReLU activation.

Three decoder types were made config-selectable: `dot_product`, `bilinear`, and `mlp`. The MLP was set as the default. Epochs were increased to 50 to give the model more training time, since the previous run had not converged.

Configuration: `hidden_dim=256, output_dim=128, dropout=0.2, lr=0.001, epochs=50, neg_ratio=1.0, decoder=mlp, decoder_hidden=128`

```
Test ROC-AUC: 0.6176
Test PR-AUC: 0.6964
```

Performance dropped substantially. The training loss fell to 0.003 (near zero), while no validation monitoring was in place at this point. The model had severely overfit: the additional decoder parameters made it easy to memorise training edges, and training for 50 epochs without any regularisation or early stopping made the problem worse.

### Run 3: validation-based best-model selection and decoder dropout

Two changes were made to address the overfitting:

1. **Validation-based best-model selection**: the training loop was modified to evaluate validation ROC-AUC every epoch and keep the weights from the best epoch. This ensures the saved model reflects peak generalisation rather than the final (overfit) state.

2. **Dropout in the MLP decoder**: the MLP decoder had no regularisation of its own (the encoder already had dropout between its layers). A dropout layer was added between the hidden activation and the output layer, using the same dropout rate as the encoder.

Configuration: same as Run 2, with validation monitoring and decoder dropout added.

```
Best validation ROC-AUC: 0.7926 (epoch 15)
Test ROC-AUC: 0.7660
Test PR-AUC: 0.8011
```

Significant improvement. Best-model selection correctly identified epoch 15 as the peak, avoiding the severe degradation that followed (validation AUC dropped to 0.63 by epoch 50). However, the loss curve still showed the same pattern: rapid memorisation (loss dropping to 0.003) with validation AUC climbing briefly then collapsing. The model was learning useful representations early on but then overfitting past the useful point.

### Run 4: aggressive regularisation (smaller model, higher dropout, more negatives)

An attempt was made to control overfitting through hyperparameter changes rather than architectural changes:

- Model dimensions halved (`hidden_dim=128, output_dim=64, decoder_hidden=64`) to reduce capacity
- Dropout increased from 0.2 to 0.4
- Learning rate halved from 0.001 to 0.0005
- Negative sampling ratio increased from 1.0 to 3.0 (three negative edges per positive edge, making the classification task harder)
- Epochs increased to 80 to compensate for the slower learning rate

Configuration: `hidden_dim=128, output_dim=64, dropout=0.4, lr=0.0005, epochs=80, neg_ratio=3.0, decoder=mlp, decoder_hidden=64`

```
Best validation ROC-AUC: 0.7348 (epoch 1)
Test ROC-AUC: 0.6949
Test PR-AUC: 0.7590
```

Worse. The best validation AUC was at epoch 1 and declined monotonically from there. The combination of reduced capacity and harder negatives left the model unable to learn useful patterns. The overfitting was slightly slower (loss plateaued around 0.005 rather than 0.003) but the model had less room to learn generalisable features, so the ceiling dropped.

This confirmed that the overfitting problem was not about model capacity. It was about unconstrained parameter growth in the presence of full-graph message passing.

### Run 5: weight decay (final configuration)

The root cause of the overfitting was identified: full-graph message passing means every training forward pass runs the encoder over the entire graph. With two SAGEConv layers, each node aggregates information from its 2-hop neighbourhood, which on a graph of this size covers most of the graph. Without any penalty on parameter magnitudes, the encoder parameters grow to encode exact edge identity rather than generalisable structural patterns.

**L2 weight decay** was added to the Adam optimiser. Weight decay directly penalises large parameter values, which prevents the encoder from encoding exact edge identity and forces it to learn more compact, generalisable representations. A relatively aggressive value of 0.01 was chosen because the overfitting had been severe in all previous runs.

The model dimensions were reverted to the original values (256/128/128) because the capacity reduction from Run 4 had been counterproductive. Dropout was set to 0.3 (a middle ground) and negative sampling ratio was reverted to 1.0.

Configuration: `hidden_dim=256, output_dim=128, dropout=0.3, lr=0.001, weight_decay=0.01, epochs=50, neg_ratio=1.0, decoder=mlp, decoder_hidden=128`

```
Best validation ROC-AUC: 0.7612 (epoch 3)
Test ROC-AUC: 0.7237
Test PR-AUC: 0.7624
```

The overfitting problem was solved. Training loss plateaued around 0.073 (compared to 0.003 without weight decay). Validation AUC remained stable across all 50 epochs, fluctuating between 0.72 and 0.76 rather than collapsing. The model was no longer memorising training edges.

However, the test AUC of 0.72 represents the practical ceiling of this architecture on this dataset with semantic-only node features.

### Summary table

| Run | Decoder | Weight decay | Best val AUC | Test AUC | Notes |
|-----|---------|-------------|-------------|----------|-------|
| 1 | dot_product | none | n/a | 0.708 | No validation monitoring; loss still decreasing at epoch 20 |
| 2 | mlp | none | n/a | 0.618 | Severe overfitting; train loss reached 0.003 |
| 3 | mlp | none | 0.793 | 0.766 | Best-model selection and decoder dropout added |
| 4 | mlp | none | 0.735 | 0.695 | Smaller model and harder negatives made things worse |
| 5 | mlp | 0.01 | 0.761 | 0.724 | Overfitting solved; this is the representational ceiling |

### Conclusion

The GraphSAGE backend with semantic-only node features achieves approximately 0.72 test ROC-AUC on this dataset, compared to 0.98 for the hybrid baseline. This gap exists because the hybrid baseline uses dedicated structural embeddings (Node2Vec) that directly encode graph topology through hundreds of random walk iterations, while GraphSAGE must learn structural patterns implicitly through only two layers of neighbourhood aggregation.

The 0.72 AUC is a reasonable result for an inductive model. The key advantage of this backend is not raw predictive accuracy on known entities (where the hybrid baseline is clearly superior) but the ability to score new entities at serving time without retraining.

### Potential improvements

The most promising directions for closing the gap to the hybrid baseline are:

- **Fetching real interaction edges at serving time**: replacing the cosine-similarity attachment heuristic with real edges from public biomedical APIs (STRING for protein-protein interactions, DGIdb for drug-gene interactions) would give GraphSAGE genuine structural context for new nodes rather than a text-similarity approximation.
- **Deeper encoder**: more SAGEConv layers would let the model aggregate information from a wider neighbourhood. This requires careful regularisation to avoid over-smoothing.
- **Mini-batch subgraph sampling**: replacing full-graph message passing with neighbour sampling (LinkNeighborLoader) would reduce the information available to the encoder at each step, which acts as implicit regularisation and may improve generalisation. This was avoided in the initial implementation to keep the quickstart dependency-free (neighbour sampling requires `pyg-lib` or `torch-sparse`).

## Scope and limitations

- This backend is for binary link prediction only. Predicate types are ignored.
- The random edge split with a fixed seed is reproducible but can leak neighbourhood information through shared nodes.
- The cosine-similarity attachment heuristic for new nodes is a simple demonstration approach and not a production graph-construction strategy.
- The serving API is not designed for production traffic. It loads the full graph into memory and runs inference synchronously.
