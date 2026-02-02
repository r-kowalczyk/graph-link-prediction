"""End-to-end training and evaluation entry points for link prediction.

This module wires together data loading, graph preparation, embedding
computation, feature construction, model training and evaluation. It is
intended to be easy to follow, with minimal assumptions about prior knowledge.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict

import numpy as np
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .io import load_data, build_node_index
from .graph import build_graph, remove_positive_edges_from_graph
from .embeddings import (
    pyg_node2vec_structural_embeddings,
    transformer_semantic_embeddings,
)
from .features import fuse_embeddings, pair_features
from .models import LogRegModel, hyperparam_search_mlp
from .eval import compute_and_plot, cold_start_mask
from .utils import (
    resolve_device,
    set_all_seeds,
    ensure_dir,
    time_stamp,
    write_json,
    save_yaml_copy,
    cache_key_from_paths_and_config,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the training script."""
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/full.yaml")
    p.add_argument(
        "--variant",
        type=str,
        choices=["structural", "semantic", "hybrid"],
        default="hybrid",
    )
    return p.parse_args()


def run(cfg: Dict, variant: str) -> Dict:
    """Execute a single training/evaluation run and return metrics.

    The routine performs:
    1) data loading and indexing
    2) graph construction and leakage-safe removal of validation/test positives
    3) embedding computation (structural, semantic, or both)
    4) pair feature construction and scaling
    5) training baseline models (logistic regression and an MLP)
    6) computing metrics and saving plots/artifacts

    Args:
        cfg: Configuration dictionary (typically parsed from YAML).
        variant: One of ``\"structural\"``, ``\"semantic\"``, ``\"hybrid\"`` to select embeddings.

    Returns:
        A dictionary of metrics and run metadata that is also written to disk.
    """
    set_all_seeds(int(cfg["seed"]))
    device = resolve_device(cfg["device"])  # Device reserved for torch-based models
    print("Step 1/9: loading data tables and building node index...", flush=True)

    # Load raw tables and build node indices/texts for embedding models
    nodes, edges, gt = load_data(
        cfg["data"]["dir"],
        cfg["data"]["nodes_csv"],
        cfg["data"]["edges_csv"],
        cfg["data"]["ground_truth_csv"],
    )
    node_idx, _, node_texts = build_node_index(nodes)

    print("Step 2/9: preparing graph and label splits...", flush=True)
    # Prepare pairs and labels (prefer notebook columns: source,target,y)
    gt = gt.copy()
    gt["source"] = gt["source"].astype(str)
    gt["target"] = gt["target"].astype(str)
    pos_pairs = [
        (node_idx[u], node_idx[v])
        for u, v in gt[gt["y"] == 1][["source", "target"]].values
    ]
    neg_pairs = [
        (node_idx[u], node_idx[v])
        for u, v in gt[gt["y"] == 0][["source", "target"]].values
    ]
    all_pairs = pos_pairs + neg_pairs
    labels = np.array([1] * len(pos_pairs) + [0] * len(neg_pairs), dtype=int)

    # Build graph from edges; later we remove validation/test positives to avoid leakage
    e = edges.copy()
    e["subject"] = e["subject"].astype(str)
    e["object"] = e["object"].astype(str)
    edge_idx = [(node_idx[s], node_idx[t]) for s, t in e[["subject", "object"]].values]
    g_full = build_graph(
        edge_idx, num_nodes=len(node_idx), undirected=bool(cfg["data"]["undirected"])
    )

    # Split labelled pairs into train/val/test using stratification for class balance
    x_pairs_train, x_pairs_tmp, y_train, y_tmp = train_test_split(
        all_pairs,
        labels,
        test_size=(cfg["splits"]["val_ratio"] + cfg["splits"]["test_ratio"]),
        stratify=labels,
        random_state=int(cfg["seed"]),
    )
    rel = (
        cfg["splits"]["test_ratio"]
        / (cfg["splits"]["val_ratio"] + cfg["splits"]["test_ratio"])
        if (cfg["splits"]["val_ratio"] + cfg["splits"]["test_ratio"]) > 0
        else 0.5
    )
    x_pairs_val, x_pairs_test, y_val, y_test = train_test_split(
        x_pairs_tmp, y_tmp, test_size=rel, stratify=y_tmp, random_state=int(cfg["seed"])
    )

    # Remove validation/test positives to simulate future links during training
    val_pos = [p for p, y in zip(x_pairs_val, y_val) if y == 1]
    test_pos = [p for p, y in zip(x_pairs_test, y_test) if y == 1]
    g_train = remove_positive_edges_from_graph(
        g_full, val_pos + test_pos, undirected=bool(cfg["data"]["undirected"])
    )

    # Prepare a cache key so embeddings can be reused across repeated runs
    nodes_csv_path = os.path.join(cfg["data"]["dir"], cfg["data"]["nodes_csv"])
    edges_csv_path = os.path.join(cfg["data"]["dir"], cfg["data"]["edges_csv"])
    gt_csv_path = os.path.join(cfg["data"]["dir"], cfg["data"]["ground_truth_csv"])
    cache_dir = cfg["cache_dir"]
    ensure_dir(cache_dir)
    cache_key = cache_key_from_paths_and_config(
        (nodes_csv_path, edges_csv_path, gt_csv_path),
        {
            "variant": variant,
            "struct_dim": cfg["structural"]["dim"],
            "struct_h": {
                "walk_length": cfg["structural"]["walk_length"],
                "context_size": cfg["structural"]["context_size"],
                "walks_per_node": cfg["structural"]["walks_per_node"],
                "epochs": cfg["structural"]["epochs"],
                "lr": cfg["structural"]["lr"],
                "batch_size": cfg["structural"]["batch_size"],
            },
            "sem_model": cfg["semantic"]["model_name"],
            "sem_bs": cfg["semantic"]["batch_size"],
            "sem_max_len": cfg["semantic"]["max_length"],
        },
    )

    structural = None
    semantic = None

    if variant in ("structural", "hybrid"):
        structural_path = os.path.join(cache_dir, f"struct_{cache_key}.npy")
        if os.path.exists(structural_path):
            print("Step 3/9: loading cached structural embeddings...", flush=True)
            structural = np.load(structural_path)
        else:
            print(
                "Step 3/9: training structural Node2Vec embeddings "
                f"(dim={cfg['structural']['dim']}, epochs={cfg['structural']['epochs']})...",
                flush=True,
            )
            # Compute Node2Vec embeddings on the leakage-safe training graph
            structural = pyg_node2vec_structural_embeddings(
                g_train,
                int(cfg["structural"]["dim"]),
                int(cfg["structural"]["walk_length"]),
                int(cfg["structural"]["context_size"]),
                int(cfg["structural"]["walks_per_node"]),
                int(cfg["structural"]["epochs"]),
                float(cfg["structural"]["lr"]),
                int(cfg["structural"]["batch_size"]),
                str(device),
            )
            np.save(structural_path, structural)
    else:
        print(
            "Step 3/9: skipping structural embedding computation for speed...",
            flush=True,
        )

    if variant in ("semantic", "hybrid"):
        semantic_path = os.path.join(cache_dir, f"semantic_{cache_key}.npy")
        if os.path.exists(semantic_path):
            print("Step 4/9: loading cached semantic embeddings...", flush=True)
            semantic = np.load(semantic_path)
        else:
            print(
                "Step 4/9: computing semantic embeddings with the transformer model...",
                flush=True,
            )
            texts = node_texts
            # Embed text fields with a transformer model
            semantic = transformer_semantic_embeddings(
                texts,
                str(cfg["semantic"]["model_name"]),
                int(cfg["semantic"]["batch_size"]),
                int(cfg["semantic"]["max_length"]),
                str(device),
            )
            np.save(semantic_path, semantic)

    print("Step 5/9: fusing embeddings and building pairwise features...", flush=True)
    emb = fuse_embeddings(structural, semantic)

    # Build pairwise features and standardise them using statistics from training
    x_train = pair_features(emb, x_pairs_train, mode=cfg["features"]["pair_mode"])
    x_val = pair_features(emb, x_pairs_val, mode=cfg["features"]["pair_mode"])
    x_test = pair_features(emb, x_pairs_test, mode=cfg["features"]["pair_mode"])

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)

    print("Step 6/9: training logistic regression baseline...", flush=True)
    # Baseline classifier: Logistic Regression as a simple linear model
    lr_model = LogRegModel()
    lr_model.fit(x_train, np.asarray(y_train))
    val_prob_lr = lr_model.predict_proba(x_val)
    test_prob_lr = lr_model.predict_proba(x_test)

    print("Step 7/9: computing baseline validation and test metrics...", flush=True)
    # Compute metrics and save diagnostic plots for each split
    run_dir = os.path.join(cfg["artifacts_dir"], time_stamp())
    curves_dir = os.path.join(run_dir, "curves")
    ensure_dir(curves_dir)
    m_val_lr = compute_and_plot(
        np.asarray(y_val),
        val_prob_lr,
        curves_dir,
        int(cfg["metrics"]["precision_at_k"]),
        int(cfg["plots"]["dpi"]),
    )
    m_test_lr = compute_and_plot(
        np.asarray(y_test),
        test_prob_lr,
        curves_dir,
        int(cfg["metrics"]["precision_at_k"]),
        int(cfg["plots"]["dpi"]),
    )

    print(
        "Step 8/9: running MLP hyperparameter search "
        f"(hidden dims={cfg['model']['mlp'].get('search_hidden_dims', [128, 256])}, "
        f"lrs={cfg['model']['mlp'].get('search_lrs', [1e-3, 5e-4])}, "
        f"epochs={cfg['model']['mlp'].get('search_epochs', [5, 10])})...",
        flush=True,
    )
    # MLP hyperparameter search over a small grid; keep the best by validation AUC
    hd_list = list(map(int, cfg["model"]["mlp"].get("search_hidden_dims", [128, 256])))
    lr_list = list(map(float, cfg["model"]["mlp"].get("search_lrs", [1e-3, 5e-4])))
    ep_list = list(map(int, cfg["model"]["mlp"].get("search_epochs", [5, 10])))
    bs = int(cfg["model"]["mlp"].get("batch_size", 256))
    best_mlp, best_conf, best_val_auc = hyperparam_search_mlp(
        x_train,
        np.asarray(y_train),
        x_val,
        np.asarray(y_val),
        hd_list,
        lr_list,
        ep_list,
        bs,
        str(device),
    )
    best_mlp.eval()
    with __import__("torch").no_grad():
        test_out_mlp = best_mlp(
            __import__("torch")
            .tensor(x_test, dtype=__import__("torch").float32)
            .to(device)
        )
        test_prob_mlp = (
            __import__("torch").sigmoid(test_out_mlp).cpu().numpy().flatten()
        )
    m_test_mlp = compute_and_plot(
        np.asarray(y_test),
        test_prob_mlp,
        curves_dir,
        int(cfg["metrics"]["precision_at_k"]),
        int(cfg["plots"]["dpi"]),
    )

    print("Step 9/9: evaluating cold-start subsets and saving artefacts...", flush=True)
    # Cold-start analysis: metrics restricted to pairs with low-degree endpoints
    cs_mask_val = cold_start_mask(
        [p for p, _ in zip(x_pairs_train, y_train)], x_pairs_val
    )
    cs_mask_test = cold_start_mask(
        [p for p, _ in zip(x_pairs_train, y_train)], x_pairs_test
    )
    m_val_cs = (
        compute_and_plot(
            np.asarray(y_val)[cs_mask_val],
            val_prob_lr[cs_mask_val],
            curves_dir,
            int(cfg["metrics"]["precision_at_k"]),
            int(cfg["plots"]["dpi"]),
        )
        if cs_mask_val.any()
        else {}
    )
    m_test_cs = (
        compute_and_plot(
            np.asarray(y_test)[cs_mask_test],
            test_prob_lr[cs_mask_test],
            curves_dir,
            int(cfg["metrics"]["precision_at_k"]),
            int(cfg["plots"]["dpi"]),
        )
        if cs_mask_test.any()
        else {}
    )

    out = {
        "variant": variant,
        "val_logreg": m_val_lr,
        "test_logreg": m_test_lr,
        "test": m_test_mlp,
        "mlp_best": {
            "hidden_dim": int(best_conf[0]),
            "lr": float(best_conf[1]),
            "epochs": int(best_conf[2]),
            "val_auc": float(best_val_auc),
        },
        "val_cold_start": m_val_cs,
        "test_cold_start": m_test_cs,
    }

    write_json(os.path.join(run_dir, "metrics.json"), out)
    # Save config used (prefer provided text, else dump current cfg)
    cfg_text = cfg.get("_config_text") if isinstance(cfg, dict) else None
    if isinstance(cfg_text, str) and cfg_text:
        save_yaml_copy(os.path.join(run_dir, "config_used.yaml"), cfg_text)
    else:
        try:
            import yaml as _yaml

            save_yaml_copy(
                os.path.join(run_dir, "config_used.yaml"), _yaml.safe_dump(cfg)
            )
        except Exception:
            pass
    print(f"Run complete. Artefacts are in: {run_dir}", flush=True)
    return out


def main() -> None:
    """Command-line entrypoint that reads configuration and runs an experiment."""
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        text = f.read()
        cfg = yaml.safe_load(text)
    if isinstance(cfg, dict):
        cfg["_config_text"] = text
    run(cfg, args.variant)


if __name__ == "__main__":
    main()  # pragma: no cover
