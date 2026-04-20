# Graph Link Prediction

# Quickstart

This quickstart runs start to finish on CPU using a tiny bundled CSV dataset. The first run may take a few minutes because the transformer model weights are downloaded; subsequent runs are typically much faster because embeddings are cached.

## Commands

Prerequisites: Python 3.12+ and `uv` installed (see [uv installation instructions](https://docs.astral.sh/uv/)).

```bash
# Create the environment (installs the project in editable mode)
uv sync --all-groups

# Train and write run artefacts under artifacts_quickstart/<timestamp>/
uv run graph-lp train --config configs/quickstart.yaml --device cpu --seed 42

# Read and report metrics from the latest run directory
uv run graph-lp evaluate --config configs/quickstart.yaml --device cpu
```

## What you get

The training command writes a timestamped run folder under `artifacts_quickstart/`, containing:

- `metrics.json` (evaluation metrics)
- `config_used.yaml` (the exact config text used for the run)
- `curves/roc.png` and `curves/pr.png` (diagnostic plots)

## Serving bundles (HTTP inference)

GraphSAGE training writes `graphsage_model_state.pt`, features, edges, and metadata under the run directory. Use `graph_lp.graphsage.export_graphsage_bundle(run_directory)` to assemble the **serving_bundle** folder (`model_state.pt`, `manifest.json`, `node_features.npy`, `edge_index.npy`, plus empty cache JSON files). The sibling **model-serving-platform** service loads that bundle: it applies only the **`encoder.*` weights** from the saved full-model state dict into a PyTorch Geometric GraphSAGE encoder that matches `GraphSageEncoder` in `graph_lp/graphsage.py`. The exported `edge_index.npy` already includes reverse edge rows when the graph is undirected (`build_graph_data`), so do not duplicate edges again at serve time.

## Dataset

- **Quickstart dataset**: bundled CSV files under `graph_lp/sample_data/quickstart` for a small, runnable demo.
- **Schema**:
  - `nodes.csv`: `id`, `name`, `description`
  - `edges.csv`: `subject`, `object`
  - `ground_truth.csv`: `source`, `target`, `y` (binary label)
- **Full dataset for reported results**: The full dataset is not committed to this repository, but a download link and setup instructions are provided in the Colab runner notebook (see the Prerequisites section): [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/r-kowalczyk/graph-link-prediction/blob/main/notebooks/hybrid_runner.ipynb)

## Reproducibility

- **Seeding**: pass `--seed <int>` to `graph-lp train`. This controls the train, validation and test split.
- **Caching**: embeddings are cached under `<output-dir>/cache`. Reusing the same output directory makes reruns faster and makes the cached embeddings consistent. If you want a full recompute, delete the cached `struct_*.npy` and `semantic_*.npy` files.
- **Determinism note**: Node2Vec random walks are not fully deterministic across runs, so AUC values may fluctuate slightly between otherwise identical runs unless cached embeddings are reused.


# Overview

This project implements a link classification pipeline that leverages hybrid node embeddings (semantic + structural embeddings), to predict the existence of relationships between nodes. The pipeline leverages two classification approaches: Logistic Regression as a baseline and a Multi-Layer Perceptron (MLP) for capturing non-linear interactions. The goal is to explore whether hybrid embeddings relying on relatively crude structural embeddings combined with domain-specific transformer-based embeddings can yield high quality feature representation for better classification performance.

---


# Approach and Design Decisions

**1) Data Prep and Graph Construction:**

- The data is used to build a NetworkX graph from the edge list. PyTorch Geometric is used only for the Node2Vec structural embedding step.


**2) Hybrid Embedding Generation:**

- Structural embeddings are generated using Node2Vec to embed the structural properties of the network. Ideally, would want to use something like a HeteroRGCN (akin to the TxGNN approach) that captures differences between node types but using Node2Vec here for speed and simplicity.

- Semantic embeddings are generated using a domain-specific transformer model to capture textual information from node attributes (names and descriptions). The default model is [BioLinkBERT-Large](https://huggingface.co/michiyasunaga/BioLinkBERT-large), which achieved the best MLP test AUC in the results below. Other models that have been tested include BioM-BERT-PubMed-PMC-Large, BioBERT and Bioformer-16L (see Results below).

- Structural and semantic embeddings are then combined (concatenation for simplicity) to form a richer feature representation for each node.


**3) Classifier Development**

- Two models are then explored for link classification: logistic regression (as a baseline) and a MLP to account for non-linear interactions.
- Hyperparameter optimisation for the MLP is carried out using a small grid search over hidden dimensions, learning rates and epochs.

---

# Results

The results below are for the full hybrid pipeline using Node2Vec structural embeddings and transformer-based semantic embeddings.

To reproduce these results you need access to the full dataset in the location expected by `configs/full.yaml` (as configured in the Colab runner), or you need to edit that config to point to your data.

## Node2Vec + BioLinkBERT-Large (best performing, default)

Command:

```bash
uv run graph-lp train --config configs/full.yaml --variant hybrid --device auto --seed 42 --model michiyasunaga/BioLinkBERT-large
```

```
Variant: hybrid
Semantic embedding model: michiyasunaga/BioLinkBERT-large
Best MLP hidden dim: 256
Best MLP learning rate: 0.0005
Best MLP epochs: 10
Best MLP validation AUC: 0.9816
Test ROC-AUC (MLP): 0.9817
Test ROC-AUC (LogReg): 0.9436
```

## Node2Vec + BioM-BERT-PubMed-PMC-Large

Command:

```bash
uv run graph-lp train --config configs/full.yaml --variant hybrid --device auto --seed 42 --model sultan/BioM-BERT-PubMed-PMC-Large
```

```
Variant: hybrid
Semantic embedding model: sultan/BioM-BERT-PubMed-PMC-Large
Best MLP hidden dim: 256
Best MLP learning rate: 0.001
Best MLP epochs: 10
Best MLP validation AUC: 0.9738
Test ROC-AUC (MLP): 0.9733
Test ROC-AUC (LogReg): 0.9453
```

## Node2Vec + BioBERT

Command:

```bash
uv run graph-lp train --config configs/full.yaml --variant hybrid --device auto --seed 42 --model dmis-lab/biobert-v1.1
```

```
Variant: hybrid
Semantic embedding model: dmis-lab/biobert-v1.1
Best MLP hidden dim: 256
Best MLP learning rate: 0.001
Best MLP epochs: 10
Best MLP validation AUC: 0.9721
Test ROC-AUC (MLP): 0.9705
Test ROC-AUC (LogReg): 0.9418
```

## Node2Vec + Bioformer-16L

Command:

```bash
uv run graph-lp train --config configs/full.yaml --variant hybrid --device auto --seed 42 --model bioformers/bioformer-16L
```

```
Variant: hybrid
Semantic embedding model: bioformers/bioformer-16L
Best MLP hidden dim: 256
Best MLP learning rate: 0.0005
Best MLP epochs: 10
Best MLP validation AUC: 0.9653
Test ROC-AUC (MLP): 0.9633
Test ROC-AUC (LogReg): 0.9296
```


---

# Interpretation

- Both classifiers show strong performance across all embedding models. Strong performance using logistic regression alone (best LogReg AUC of 0.9453) suggests high quality feature representation using the hybrid embeddings (i.e. even a relatively simple model can make accurate predictions given how well the hybrid embeddings capture information about nodes and relationships).

- The best overall MLP result was achieved with BioLinkBERT-Large (0.9817 test AUC), followed closely by BioM-BERT-PubMed-PMC-Large (0.9733). The approximately 4% improvement from LogReg (0.9436) to MLP (0.9817) with BioLinkBERT-Large suggests that using a non-linear classifier can capture complex patterns in the data more effectively.

- BioLinkBERT-Large is the default in `configs/full.yaml`, but you can run other semantic embedding models in Colab by setting `EMBEDDING_MODEL` in `notebooks/hybrid_runner.ipynb`.

# Scope and limitations

- This repository is meant to demo an end-to-end link classification pipeline with hybrid embeddings, caching, and reproducible config files.
- It does not claim to be a production system. It prioritises readability and iteration speed over large-scale training and deployment concerns.
- Tasks in this repository are binary link prediction tasks. Predicate types are ignored during training and inference.
- Reported AUC values depend on the provided labels, how negative examples were constructed in the ground truth, and the chosen split strategy.

---

# GraphSAGE backend (alternative approach)

An alternative modelling backend based on GraphSAGE is available for binary link prediction with inductive serving support. Instead of the hybrid Node2Vec + transformer pipeline above, the GraphSAGE backend trains a graph neural network directly on semantic text embeddings, and can score links for new entities that were not present at training time.

See the full documentation, including quickstart commands, serving demo, curl examples, and a Colab notebook for GPU training: **[docs/graphsage.md](docs/graphsage.md)**

---

## Related: HTTP serving

Exported GraphSAGE bundles from this repository are consumed by **[model-serving-platform](https://github.com/r-kowalczyk/model-serving-platform)**, a FastAPI service that validates the bundle at startup, reconstructs the trained encoder, and serves link prediction over HTTP (including a documented Azure Container Apps deployment path).
