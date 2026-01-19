"""Unit tests for the training pipeline with mocked heavy dependencies."""

from unittest.mock import patch

import numpy as np

from graph_lp.train import parse_args, run


def create_minimal_test_data(tmp_path):
    """Create minimal CSV files for testing."""
    nodes_file = tmp_path / "nodes.csv"
    edges_file = tmp_path / "edges.csv"
    gt_file = tmp_path / "ground_truth.csv"
    nodes_file.write_text(
        "id,name,description\nn0,Node0,Desc0\nn1,Node1,Desc1\nn2,Node2,Desc2\nn3,Node3,Desc3"
    )
    edges_file.write_text("subject,object\nn0,n1\nn1,n2\nn2,n3\nn3,n0")
    gt_file.write_text(
        "source,target,y\n"
        "n0,n1,1\n"
        "n1,n2,1\n"
        "n2,n3,1\n"
        "n3,n0,1\n"
        "n0,n2,1\n"
        "n1,n0,0\n"
        "n2,n1,0\n"
        "n3,n2,0\n"
        "n0,n3,0\n"
        "n1,n3,0\n"
    )
    return str(tmp_path)


@patch("graph_lp.train.transformer_semantic_embeddings")
@patch("graph_lp.train.pyg_node2vec_structural_embeddings")
def test_run_hybrid_variant(mock_structural, mock_semantic, tmp_path):
    """Test that run completes successfully with hybrid variant."""
    data_dir = create_minimal_test_data(tmp_path)
    artifacts_dir = tmp_path / "artifacts"
    cache_dir = tmp_path / "cache"
    artifacts_dir.mkdir()
    cache_dir.mkdir()
    mock_structural.return_value = np.random.randn(4, 8).astype(np.float32)
    mock_semantic.return_value = np.random.randn(4, 16).astype(np.float32)
    cfg = {
        "seed": 42,
        "device": "cpu",
        "data": {
            "dir": data_dir,
            "nodes_csv": "nodes.csv",
            "edges_csv": "edges.csv",
            "ground_truth_csv": "ground_truth.csv",
            "undirected": True,
        },
        "splits": {"val_ratio": 0.2, "test_ratio": 0.2},
        "structural": {
            "dim": 8,
            "walk_length": 5,
            "context_size": 3,
            "walks_per_node": 2,
            "epochs": 1,
            "lr": 0.01,
            "batch_size": 32,
        },
        "semantic": {
            "model_name": "test-model",
            "batch_size": 4,
            "max_length": 128,
        },
        "features": {"pair_mode": "concat"},
        "metrics": {"precision_at_k": 2},
        "plots": {"dpi": 80},
        "model": {
            "mlp": {
                "search_hidden_dims": [8],
                "search_lrs": [0.01],
                "search_epochs": [1],
                "batch_size": 4,
            }
        },
        "artifacts_dir": str(artifacts_dir),
        "cache_dir": str(cache_dir),
    }
    results = run(cfg, "hybrid")
    assert "variant" in results
    assert results["variant"] == "hybrid"
    assert "test" in results
    assert "test_logreg" in results
    assert "val_logreg" in results
    assert "mlp_best" in results
    mock_structural.assert_called_once()
    mock_semantic.assert_called_once()


@patch("graph_lp.train.transformer_semantic_embeddings")
def test_run_semantic_variant(mock_semantic, tmp_path):
    """Test that run completes successfully with semantic-only variant."""
    data_dir = create_minimal_test_data(tmp_path)
    artifacts_dir = tmp_path / "artifacts"
    cache_dir = tmp_path / "cache"
    artifacts_dir.mkdir()
    cache_dir.mkdir()
    mock_semantic.return_value = np.random.randn(4, 16).astype(np.float32)
    cfg = {
        "seed": 42,
        "device": "cpu",
        "data": {
            "dir": data_dir,
            "nodes_csv": "nodes.csv",
            "edges_csv": "edges.csv",
            "ground_truth_csv": "ground_truth.csv",
            "undirected": True,
        },
        "splits": {"val_ratio": 0.2, "test_ratio": 0.2},
        "structural": {
            "dim": 8,
            "walk_length": 5,
            "context_size": 3,
            "walks_per_node": 2,
            "epochs": 1,
            "lr": 0.01,
            "batch_size": 32,
        },
        "semantic": {
            "model_name": "test-model",
            "batch_size": 4,
            "max_length": 128,
        },
        "features": {"pair_mode": "concat"},
        "metrics": {"precision_at_k": 2},
        "plots": {"dpi": 80},
        "model": {
            "mlp": {
                "search_hidden_dims": [8],
                "search_lrs": [0.01],
                "search_epochs": [1],
                "batch_size": 4,
            }
        },
        "artifacts_dir": str(artifacts_dir),
        "cache_dir": str(cache_dir),
    }
    results = run(cfg, "semantic")
    assert results["variant"] == "semantic"
    mock_semantic.assert_called_once()


@patch("graph_lp.train.pyg_node2vec_structural_embeddings")
def test_run_structural_variant(mock_structural, tmp_path):
    """Test that run completes successfully with structural-only variant."""
    data_dir = create_minimal_test_data(tmp_path)
    artifacts_dir = tmp_path / "artifacts"
    cache_dir = tmp_path / "cache"
    artifacts_dir.mkdir()
    cache_dir.mkdir()
    mock_structural.return_value = np.random.randn(4, 8).astype(np.float32)
    cfg = {
        "seed": 42,
        "device": "cpu",
        "data": {
            "dir": data_dir,
            "nodes_csv": "nodes.csv",
            "edges_csv": "edges.csv",
            "ground_truth_csv": "ground_truth.csv",
            "undirected": True,
        },
        "splits": {"val_ratio": 0.2, "test_ratio": 0.2},
        "structural": {
            "dim": 8,
            "walk_length": 5,
            "context_size": 3,
            "walks_per_node": 2,
            "epochs": 1,
            "lr": 0.01,
            "batch_size": 32,
        },
        "semantic": {
            "model_name": "test-model",
            "batch_size": 4,
            "max_length": 128,
        },
        "features": {"pair_mode": "concat"},
        "metrics": {"precision_at_k": 2},
        "plots": {"dpi": 80},
        "model": {
            "mlp": {
                "search_hidden_dims": [8],
                "search_lrs": [0.01],
                "search_epochs": [1],
                "batch_size": 4,
            }
        },
        "artifacts_dir": str(artifacts_dir),
        "cache_dir": str(cache_dir),
    }
    results = run(cfg, "structural")
    assert results["variant"] == "structural"
    mock_structural.assert_called_once()


@patch("graph_lp.train.transformer_semantic_embeddings")
@patch("graph_lp.train.pyg_node2vec_structural_embeddings")
def test_run_uses_cached_embeddings(mock_structural, mock_semantic, tmp_path):
    """Test that run uses cached embeddings when available."""
    data_dir = create_minimal_test_data(tmp_path)
    artifacts_dir = tmp_path / "artifacts"
    cache_dir = tmp_path / "cache"
    artifacts_dir.mkdir()
    cache_dir.mkdir()
    cached_structural = np.random.randn(4, 8).astype(np.float32)
    cached_semantic = np.random.randn(4, 16).astype(np.float32)
    cache_key = "test_cache_key"
    structural_path = cache_dir / f"struct_{cache_key}.npy"
    semantic_path = cache_dir / f"semantic_{cache_key}.npy"
    np.save(structural_path, cached_structural)
    np.save(semantic_path, cached_semantic)
    cfg = {
        "seed": 42,
        "device": "cpu",
        "data": {
            "dir": data_dir,
            "nodes_csv": "nodes.csv",
            "edges_csv": "edges.csv",
            "ground_truth_csv": "ground_truth.csv",
            "undirected": True,
        },
        "splits": {"val_ratio": 0.2, "test_ratio": 0.2},
        "structural": {
            "dim": 8,
            "walk_length": 5,
            "context_size": 3,
            "walks_per_node": 2,
            "epochs": 1,
            "lr": 0.01,
            "batch_size": 32,
        },
        "semantic": {
            "model_name": "test-model",
            "batch_size": 4,
            "max_length": 128,
        },
        "features": {"pair_mode": "concat"},
        "metrics": {"precision_at_k": 2},
        "plots": {"dpi": 80},
        "model": {
            "mlp": {
                "search_hidden_dims": [8],
                "search_lrs": [0.01],
                "search_epochs": [1],
                "batch_size": 4,
            }
        },
        "artifacts_dir": str(artifacts_dir),
        "cache_dir": str(cache_dir),
    }
    with patch(
        "graph_lp.train.cache_key_from_paths_and_config", return_value=cache_key
    ):
        results = run(cfg, "hybrid")
    assert results["variant"] == "hybrid"
    mock_structural.assert_not_called()
    mock_semantic.assert_not_called()


@patch("graph_lp.train.transformer_semantic_embeddings")
@patch("graph_lp.train.pyg_node2vec_structural_embeddings")
def test_run_saves_artifacts(mock_structural, mock_semantic, tmp_path):
    """Test that run saves metrics and plots to the artifacts directory."""
    data_dir = create_minimal_test_data(tmp_path)
    artifacts_dir = tmp_path / "artifacts"
    cache_dir = tmp_path / "cache"
    artifacts_dir.mkdir()
    cache_dir.mkdir()
    mock_structural.return_value = np.random.randn(4, 8).astype(np.float32)
    mock_semantic.return_value = np.random.randn(4, 16).astype(np.float32)
    cfg = {
        "seed": 42,
        "device": "cpu",
        "data": {
            "dir": data_dir,
            "nodes_csv": "nodes.csv",
            "edges_csv": "edges.csv",
            "ground_truth_csv": "ground_truth.csv",
            "undirected": True,
        },
        "splits": {"val_ratio": 0.2, "test_ratio": 0.2},
        "structural": {
            "dim": 8,
            "walk_length": 5,
            "context_size": 3,
            "walks_per_node": 2,
            "epochs": 1,
            "lr": 0.01,
            "batch_size": 32,
        },
        "semantic": {
            "model_name": "test-model",
            "batch_size": 4,
            "max_length": 128,
        },
        "features": {"pair_mode": "concat"},
        "metrics": {"precision_at_k": 2},
        "plots": {"dpi": 80},
        "model": {
            "mlp": {
                "search_hidden_dims": [8],
                "search_lrs": [0.01],
                "search_epochs": [1],
                "batch_size": 4,
            }
        },
        "artifacts_dir": str(artifacts_dir),
        "cache_dir": str(cache_dir),
        "_config_text": "test: config",
    }
    run(cfg, "hybrid")
    run_dirs = list(artifacts_dir.iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "config_used.yaml").exists()
    assert (run_dir / "curves" / "roc.png").exists()
    assert (run_dir / "curves" / "pr.png").exists()


def test_parse_args_defaults():
    """Test that parse_args returns defaults when no arguments are provided."""
    with patch("sys.argv", ["train.py"]):
        args = parse_args()
        assert args.config == "configs/full.yaml"
        assert args.variant == "hybrid"


def test_parse_args_custom():
    """Test that parse_args parses custom arguments."""
    with patch(
        "sys.argv", ["train.py", "--config", "custom.yaml", "--variant", "semantic"]
    ):
        args = parse_args()
        assert args.config == "custom.yaml"
        assert args.variant == "semantic"


def test_run_handles_yaml_dump_failure_when_saving_config(tmp_path):
    """Test that run does not crash if YAML dumping fails when saving the config."""
    data_dir = create_minimal_test_data(tmp_path)
    artifacts_dir = tmp_path / "artifacts"
    cache_dir = tmp_path / "cache"
    artifacts_dir.mkdir()
    cache_dir.mkdir()

    cfg = {
        "seed": 42,
        "device": "cpu",
        "data": {
            "dir": data_dir,
            "nodes_csv": "nodes.csv",
            "edges_csv": "edges.csv",
            "ground_truth_csv": "ground_truth.csv",
            "undirected": True,
        },
        "splits": {"val_ratio": 0.2, "test_ratio": 0.2},
        "structural": {
            "dim": 8,
            "walk_length": 5,
            "context_size": 3,
            "walks_per_node": 2,
            "epochs": 1,
            "lr": 0.01,
            "batch_size": 32,
        },
        "semantic": {
            "model_name": "test-model",
            "batch_size": 4,
            "max_length": 128,
        },
        "features": {"pair_mode": "concat"},
        "metrics": {"precision_at_k": 2},
        "plots": {"dpi": 80},
        "model": {
            "mlp": {
                "search_hidden_dims": [8],
                "search_lrs": [0.01],
                "search_epochs": [1],
                "batch_size": 4,
            }
        },
        "artifacts_dir": str(artifacts_dir),
        "cache_dir": str(cache_dir),
    }

    with (
        patch("graph_lp.train.transformer_semantic_embeddings") as mock_semantic,
        patch(
            "graph_lp.train.yaml.safe_dump",
            side_effect=RuntimeError("YAML dump failed"),
        ),
    ):
        mock_semantic.return_value = np.random.randn(4, 16).astype(np.float32)
        results = run(cfg, "semantic")

    assert results["variant"] == "semantic"
    run_dir = next(iter(artifacts_dir.iterdir()))
    assert (run_dir / "metrics.json").exists()
    assert not (run_dir / "config_used.yaml").exists()


def test_main_reads_config_and_calls_run(tmp_path):
    """Test that main reads a YAML config file and passes it to run."""
    cfg_path = tmp_path / "config.yaml"
    cfg_text = "seed: 42\n"
    cfg_path.write_text(cfg_text, encoding="utf-8")

    with (
        patch("graph_lp.train.run") as mock_run,
        patch(
            "sys.argv",
            ["train.py", "--config", str(cfg_path), "--variant", "semantic"],
        ),
    ):
        from graph_lp import train as train_module

        train_module.main()

    assert mock_run.call_count == 1
    called_cfg, called_variant = mock_run.call_args.args
    assert called_variant == "semantic"
    assert isinstance(called_cfg, dict)
    assert called_cfg["_config_text"] == cfg_text
