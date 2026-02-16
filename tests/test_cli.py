"""Smoke tests for the public command line interface.

The CLI is implemented as a Python function that returns an exit code, which
keeps tests fast and avoids spawning subprocesses. These tests focus on basic
end-to-end behaviour using a tiny config and mocked embedding functions.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from graph_lp import cli


def _create_minimal_csv_dataset(dataset_directory: Path) -> None:
    """Create a minimal CSV dataset compatible with the training pipeline.

    The training code expects three CSV files: nodes, edges, and ground truth.
    This helper writes the smallest dataset that still supports stratified
    train, validation, and test splits for both positive and negative labels.
    """

    dataset_directory.mkdir(parents=True, exist_ok=True)
    (dataset_directory / "nodes.csv").write_text(
        "id,name,description\n"
        "n0,Node0,Desc0\n"
        "n1,Node1,Desc1\n"
        "n2,Node2,Desc2\n"
        "n3,Node3,Desc3\n",
        encoding="utf-8",
    )
    (dataset_directory / "edges.csv").write_text(
        "subject,object\nn0,n1\nn1,n2\nn2,n3\nn3,n0\n",
        encoding="utf-8",
    )
    (dataset_directory / "ground_truth.csv").write_text(
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
        "n1,n3,0\n",
        encoding="utf-8",
    )


def test_cli_train_and_evaluate_quickstart_config_produces_metrics(tmp_path, capsys):
    """Run the quickstart config through the CLI and assert a metrics file exists."""

    repository_root = Path(__file__).resolve().parents[1]
    quickstart_config_path = repository_root / "configs" / "quickstart.yaml"
    output_directory = tmp_path / "run_output"

    # Mock transformer embedding to avoid downloading model weights in tests.
    with patch("graph_lp.train.transformer_semantic_embeddings") as mock_semantic:
        mock_semantic.return_value = np.random.randn(12, 8).astype(np.float32)
        train_exit_code = cli.main(
            [
                "train",
                "--config",
                str(quickstart_config_path),
                "--device",
                "cpu",
                "--seed",
                "42",
                "--output-dir",
                str(output_directory),
            ]
        )

    assert train_exit_code == 0
    metrics_files = list(output_directory.rglob("metrics.json"))
    assert len(metrics_files) == 1

    evaluate_exit_code = cli.main(
        [
            "evaluate",
            "--config",
            str(quickstart_config_path),
            "--device",
            "cpu",
            "--output-dir",
            str(output_directory),
        ]
    )
    assert evaluate_exit_code == 0
    output_text = capsys.readouterr().out
    assert "Loaded metrics from:" in output_text


def test_cli_evaluate_supports_explicit_run_directory(tmp_path, capsys):
    """Evaluate can read a specific run directory when metrics are present."""

    repository_root = Path(__file__).resolve().parents[1]
    quickstart_config_path = repository_root / "configs" / "quickstart.yaml"

    run_directory = tmp_path / "20200101-000000"
    run_directory.mkdir(parents=True, exist_ok=True)
    (run_directory / "metrics.json").write_text("{}", encoding="utf-8")

    evaluate_exit_code = cli.main(
        [
            "evaluate",
            "--config",
            str(quickstart_config_path),
            "--run-dir",
            str(run_directory),
        ]
    )
    assert evaluate_exit_code == 0
    output_text = capsys.readouterr().out
    assert "Test ROC-AUC" not in output_text


def test_cli_train_supports_json_config_and_relative_paths(tmp_path):
    """The CLI can load JSON config and resolve config-relative paths."""

    dataset_directory = tmp_path / "dataset"
    _create_minimal_csv_dataset(dataset_directory)

    config_path = tmp_path / "config.json"
    config_payload = {
        "seed": 42,
        "device": "cpu",
        "variant": "semantic",
        "artifacts_dir": "artifacts",
        "cache_dir": "cache",
        "data": {
            "dir": "dataset",
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
        "semantic": {"model_name": "test-model", "batch_size": 4, "max_length": 64},
        "features": {"pair_mode": "concat"},
        "metrics": {"precision_at_k": 2},
        "plots": {"dpi": 80},
        "model": {
            "mlp": {
                "batch_size": 4,
                "search_hidden_dims": [8],
                "search_lrs": [0.01],
                "search_epochs": [1],
            }
        },
    }
    config_path.write_text(json.dumps(config_payload), encoding="utf-8")

    with patch("graph_lp.train.transformer_semantic_embeddings") as mock_semantic:
        mock_semantic.return_value = np.random.randn(4, 8).astype(np.float32)
        train_exit_code = cli.main(
            ["train", "--config", str(config_path), "--variant", "semantic"]
        )

    assert train_exit_code == 0
    metrics_files = list((tmp_path / "artifacts").rglob("metrics.json"))
    assert len(metrics_files) == 1


def test_cli_returns_error_code_for_missing_required_config_fields(tmp_path, capsys):
    """Missing required keys should produce a clear error and a non-zero code."""

    config_path = tmp_path / "broken.yaml"
    config_path.write_text("seed: 42\n", encoding="utf-8")

    exit_code = cli.main(["train", "--config", str(config_path)])
    assert exit_code == 2
    error_text = capsys.readouterr().err
    assert "Config is missing required field" in error_text


def test_cli_returns_error_code_for_unknown_config_extension(tmp_path, capsys):
    """Unknown config extensions should be rejected with a clear message."""

    config_path = tmp_path / "config.txt"
    config_path.write_text("{}", encoding="utf-8")
    exit_code = cli.main(["train", "--config", str(config_path)])
    assert exit_code == 2
    error_text = capsys.readouterr().err
    assert "Config file must end with" in error_text


def test_cli_evaluate_returns_error_code_when_no_runs_exist(tmp_path):
    """Evaluating without an existing run directory should fail with code 2."""

    repository_root = Path(__file__).resolve().parents[1]
    quickstart_config_path = repository_root / "configs" / "quickstart.yaml"
    exit_code = cli.main(
        [
            "evaluate",
            "--config",
            str(quickstart_config_path),
            "--output-dir",
            str(tmp_path),
        ]
    )
    assert exit_code == 2


def test_cli_evaluate_returns_error_code_when_metrics_file_is_missing(tmp_path):
    """Evaluating a run directory without metrics.json should fail with code 2."""

    repository_root = Path(__file__).resolve().parents[1]
    quickstart_config_path = repository_root / "configs" / "quickstart.yaml"

    run_directory = tmp_path / "20200101-000000"
    run_directory.mkdir(parents=True, exist_ok=True)
    exit_code = cli.main(
        [
            "evaluate",
            "--config",
            str(quickstart_config_path),
            "--run-dir",
            str(run_directory),
        ]
    )
    assert exit_code == 2


def test_cli_run_calls_train_then_evaluate(monkeypatch):
    """The run subcommand should execute train before evaluate."""

    call_order: list[str] = []

    def fake_train(_arguments):
        call_order.append("train")
        return 0

    def fake_evaluate(_arguments):
        call_order.append("evaluate")
        return 0

    monkeypatch.setattr(cli, "_command_train", fake_train)
    monkeypatch.setattr(cli, "_command_evaluate", fake_evaluate)

    exit_code = cli.main(["run", "--config", "configs/quickstart.yaml"])
    assert exit_code == 0
    assert call_order == ["train", "evaluate"]


def test_cli_run_returns_train_exit_code_without_running_evaluate(monkeypatch):
    """If training fails, run should return that code and stop."""

    def fake_train(_arguments):
        return 7

    monkeypatch.setattr(cli, "_command_train", fake_train)
    monkeypatch.setattr(cli, "_command_evaluate", object())

    exit_code = cli.main(["run", "--config", "configs/quickstart.yaml"])
    assert exit_code == 7


def test_cli_run_exposes_run_dir_attribute_to_evaluate(tmp_path, monkeypatch, capsys):
    """The run subcommand should not crash when evaluate reads arguments.run_dir.

    tmp_path provides a temporary output folder, monkeypatch replaces the training
    runner with a function that does nothing, and capsys captures standard output so the
    test can assert that evaluation completed by printing the metrics path.
    """

    repository_root = Path(__file__).resolve().parents[1]
    quickstart_config_path = repository_root / "configs" / "quickstart.yaml"
    output_directory = tmp_path / "run_output"
    run_directory = output_directory / "20200101-000000"
    run_directory.mkdir(parents=True, exist_ok=True)
    (run_directory / "metrics.json").write_text("{}", encoding="utf-8")

    # Avoid running the full training pipeline because evaluation reads metrics from disk.
    monkeypatch.setattr(cli.training_module, "run", lambda *_args, **_kwargs: None)

    exit_code = cli.main(
        [
            "run",
            "--config",
            str(quickstart_config_path),
            "--output-dir",
            str(output_directory),
        ]
    )
    assert exit_code == 0
    output_text = capsys.readouterr().out
    assert "Loaded metrics from:" in output_text


def test_cli_train_defaults_to_hybrid_when_variant_is_not_provided(tmp_path):
    """If neither CLI nor config specifies a variant, the CLI should use hybrid."""

    dataset_directory = tmp_path / "dataset"
    _create_minimal_csv_dataset(dataset_directory)

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "seed: 42",
                "device: cpu",
                "artifacts_dir: artifacts",
                "cache_dir: cache",
                "data:",
                "  dir: dataset",
                "  nodes_csv: nodes.csv",
                "  edges_csv: edges.csv",
                "  ground_truth_csv: ground_truth.csv",
                "  undirected: true",
                "splits:",
                "  val_ratio: 0.2",
                "  test_ratio: 0.2",
                "structural:",
                "  dim: 8",
                "  walk_length: 5",
                "  context_size: 3",
                "  walks_per_node: 2",
                "  epochs: 1",
                "  lr: 0.01",
                "  batch_size: 32",
                "semantic:",
                "  model_name: test-model",
                "  batch_size: 4",
                "  max_length: 64",
                "features:",
                "  pair_mode: concat",
                "model:",
                "  mlp:",
                "    batch_size: 4",
                "    search_hidden_dims: [8]",
                "    search_lrs: [0.01]",
                "    search_epochs: [1]",
                "metrics:",
                "  precision_at_k: 2",
                "plots:",
                "  dpi: 80",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with (
        patch("graph_lp.train.pyg_node2vec_structural_embeddings") as mock_structural,
        patch("graph_lp.train.transformer_semantic_embeddings") as mock_semantic,
    ):
        mock_structural.return_value = np.random.randn(4, 8).astype(np.float32)
        mock_semantic.return_value = np.random.randn(4, 8).astype(np.float32)
        train_exit_code = cli.main(["train", "--config", str(config_path)])

    assert train_exit_code == 0


def test_cli_main_returns_one_when_parser_exits_without_code(monkeypatch):
    """The CLI should return code 1 if parsing exits without an explicit code."""

    class DummyParser:
        def parse_args(self, _argument_list):
            raise SystemExit()

    monkeypatch.setattr(cli, "_build_parser", lambda: DummyParser())
    exit_code = cli.main(["train", "--config", "configs/quickstart.yaml"])
    assert exit_code == 1


def test_cli_main_returns_zero_for_help():
    """The top-level help output should return exit code 0."""

    exit_code = cli.main(["--help"])
    assert exit_code == 0


def test_cli_train_model_flag_overrides_semantic_model_name(tmp_path):
    """The --model flag should override the semantic.model_name from the config.

    This test passes a custom model identifier via the --model CLI argument
    and verifies that the training pipeline receives the overridden value
    rather than the default from the configuration file.
    """

    dataset_directory = tmp_path / "dataset"
    _create_minimal_csv_dataset(dataset_directory)

    config_path = tmp_path / "config.json"
    config_payload = {
        "seed": 42,
        "device": "cpu",
        "variant": "semantic",
        "artifacts_dir": "artifacts",
        "cache_dir": "cache",
        "data": {
            "dir": "dataset",
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
        "semantic": {"model_name": "original-model", "batch_size": 4, "max_length": 64},
        "features": {"pair_mode": "concat"},
        "metrics": {"precision_at_k": 2},
        "plots": {"dpi": 80},
        "model": {
            "mlp": {
                "batch_size": 4,
                "search_hidden_dims": [8],
                "search_lrs": [0.01],
                "search_epochs": [1],
            }
        },
    }
    config_path.write_text(json.dumps(config_payload), encoding="utf-8")

    # Capture the configuration dictionary that the training pipeline receives
    # so we can assert that the semantic model name was overridden by --model.
    captured_configurations: list[dict] = []

    def capture_run(configuration, _variant):
        captured_configurations.append(configuration)

    with patch("graph_lp.train.run", side_effect=capture_run):
        exit_code = cli.main(
            [
                "train",
                "--config",
                str(config_path),
                "--model",
                "custom/override-model",
            ]
        )

    assert exit_code == 0
    assert len(captured_configurations) == 1
    assert (
        captured_configurations[0]["semantic"]["model_name"] == "custom/override-model"
    )


def test_cli_returns_error_code_when_config_top_level_is_not_a_mapping(
    tmp_path, capsys
):
    """A config that parses to a list should be rejected with a clear error."""

    config_path = tmp_path / "config.yaml"
    config_path.write_text("- 1\n- 2\n", encoding="utf-8")
    exit_code = cli.main(["train", "--config", str(config_path)])
    assert exit_code == 2
    error_text = capsys.readouterr().err
    assert "top level" in error_text
