[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/r-kowalczyk/graph-link-prediction/blob/main/notebooks/colab_runner.ipynb)


# ```Graph Link Prediction```


# Overview

This project implements a link classification pipeline that leverages hybrid node embeddings (semantic + structural embeddings), to predict the existence of relationships between nodes. The pipeline leverages two classification approaches: Logistic Regression as a baseline and a Multi-Layer Perceptron (MLP) for capturing nonlinear interactions. The goal is to explore whether hybrid embeddings relying on relatively crude structural embeddings combined with domain-specific transformer-based embeddings can yield high quality feature representation for better classification performance.

---


# Approach and Design Decisions

**1) Data Prep and Graph Construction:**

- The data is used to generate a network graph using PyTorch Geometric.


**2) Hybrid Embedding Generation:**

 - Structural embeddings are generated using Node2Vec to embed the structural properties of the network. Ideally, would want to use something like a HeteroRGCN (akin to the TxGNN approach) that captures differences between node types but using Node2Vec here for speed and simplicity.

- Semantic embeddings are generated using the BioBERT model to capture textual information from node attributes (names and descriptions). BioBERT is a pre-trained transformer model specific to the Biomedical domain and easily accessible from HuggingFace.

- Structural and semantic embeddings are then combined (concatentation for simplicity) to form a richer feature representation for each node.


**3) Classifier Development**

- Two models are then explored for link classification: logistic regression (as a baseline) and a MLP to account for non-linear interactions.
- Hyperparameter optimisation for the MLP is carried out using a small grid search over hidden dimensions, learning rates and epochs.

---

# Results

## Node2Vec + BioBERT (original choice)

```
LogisticRegression test AUC (scaled, max_iter=1000): 0.9406

=== Best MLP Model (Test Set) ===
Config: hidden_dim=256, lr=0.0005, epochs=10
AUC:   0.9691
AUPR:  0.9569
F1:    0.8922
```

## Node2Vec + BioLinkBERT-Large

```
LogisticRegression test AUC (scaled, max_iter=1000): 0.9394

=== Best MLP Model (Test Set) ===
Config: hidden_dim=256, lr=0.0005, epochs=10
AUC:   0.9695
AUPR:  0.9564
F1:    0.8891
```

## Node2Vec + BioM-BERT-PMC-Large (best performing)

BioM-BERT-PMC-Large scores top of the BLURB leaderboard for relation extraction, which is a particularly relevant metric for drug-disease link prediction.

```
LogisticRegression test AUC (scaled, max_iter=1000): 0.9417

=== Best MLP Model (Test Set) ===
Config: hidden_dim=256, lr=0.001, epochs=10
AUC:   0.9751
AUPR:  0.9656
F1:    0.9034
```

## Node2Vec + Bioformer-16L

```
LogisticRegression test AUC (scaled, max_iter=1000): 0.9248

=== Best MLP Model (Test Set) ===
Config: hidden_dim=256, lr=0.001, epochs=10
AUC:   0.9641
AUPR:  0.9514
F1:    0.8784
```


---

# Interpretation (Node2Vec + BioBERT)

- Both classifiers show strong performance. Strong performance using logistic regression alone suggests high quality feature representation using the hybrid embeddings (i.e. even a relatively simple model can make accurate predictions given how well the hybrid embeddings capture information about nodes and relationships).

- However, the ~ 3% improvement in AUC from 0.9406 (logistic regression) to 0.9691 (best MLP) suggests that using a nonlinear classifier can capture complex patterns in the data more effectively.
