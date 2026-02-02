"""Command line interface for the graph link prediction pipeline.

This script defines the packaged `graph-lp` entry point and its subcommands.
It loads a YAML or JSON configuration file, applies a small set of command line
overrides (seed, device, output directory), and then calls the existing training
pipeline in `graph_lp.train` to produce run artefacts on disk.

The `train` command runs the full pipeline end to end. The `evaluate` command
locates a completed run directory and reports the stored results from
`metrics.json`. The `run` command is a convenience wrapper that executes train
followed by evaluate. Relative paths in the configuration are resolved relative
to the configuration file location so runs behave consistently across machines.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Callable

import yaml

from graph_lp import train as training_module


def _load_configuration(config_path: str) -> tuple[dict[str, Any], str]:
    """Load a configuration mapping from a YAML or JSON file.

    The training pipeline expects a Python dictionary with nested sections.
    The raw configuration text is returned as well so it can be stored as a
    run artefact without losing comments or formatting from the input file.
    The function raises ``ValueError`` if the file does not parse to a mapping.
    """

    with open(config_path, "r", encoding="utf-8") as config_file:
        configuration_text = config_file.read()

    if config_path.endswith((".yaml", ".yml")):
        configuration = yaml.safe_load(configuration_text)
    elif config_path.endswith(".json"):
        configuration = json.loads(configuration_text)
    else:
        raise ValueError("Config file must end with .yaml, .yml, or .json.")

    if not isinstance(configuration, dict):
        raise ValueError("Config file must contain a mapping at the top level.")

    return configuration, configuration_text


def _get_required_mapping_value(configuration: dict[str, Any], dotted_key: str) -> Any:
    """Retrieve a nested configuration value using dotted key notation.

    The CLI uses this helper for minimal validation and clear error messages.
    A dotted key such as ``data.dir`` is interpreted as nested dictionaries.
    The function raises ``ValueError`` if any part of the path is missing.
    """

    value: Any = configuration
    for key_part in dotted_key.split("."):
        if not isinstance(value, dict) or key_part not in value:
            raise ValueError(f"Config is missing required field: {dotted_key}")
        value = value[key_part]
    return value


def _validate_training_configuration(configuration: dict[str, Any]) -> None:
    """Validate that the configuration contains the fields used by training.

    The underlying training code accesses many nested keys directly, which
    would otherwise fail with ``KeyError``. This validator keeps failures
    explicit and user-facing, while still keeping validation intentionally lean.
    """

    required_keys = [
        "seed",
        "device",
        "artifacts_dir",
        "cache_dir",
        "data.dir",
        "data.nodes_csv",
        "data.edges_csv",
        "data.ground_truth_csv",
        "data.undirected",
        "splits.val_ratio",
        "splits.test_ratio",
        "structural.dim",
        "structural.walk_length",
        "structural.context_size",
        "structural.walks_per_node",
        "structural.epochs",
        "structural.lr",
        "structural.batch_size",
        "semantic.model_name",
        "semantic.batch_size",
        "semantic.max_length",
        "features.pair_mode",
        "metrics.precision_at_k",
        "plots.dpi",
        "model.mlp",
    ]
    for required_key in required_keys:
        _get_required_mapping_value(configuration, required_key)


def _resolve_path_relative_to_directory(base_directory: str, path_value: str) -> str:
    """Resolve a path value relative to a base directory.

    Configuration files often contain relative paths for portability across
    machines. Resolving them relative to the configuration directory makes
    behaviour consistent regardless of the current working directory.
    """

    if os.path.isabs(path_value):
        return path_value
    return os.path.normpath(os.path.join(base_directory, path_value))


def _prepare_configuration_for_run(
    configuration: dict[str, Any],
    configuration_text: str,
    config_path: str,
    seed_override: int | None,
    device_override: str | None,
    output_directory_override: str | None,
) -> dict[str, Any]:
    """Apply CLI overrides and path resolution to a loaded configuration.

    The CLI is designed as a thin wrapper, so it modifies only a small, explicit
    set of fields (seed, device, output directories). Relative paths from the
    configuration are resolved so the training pipeline receives usable paths.
    """

    # Store the original config text so the training pipeline can persist it as-is.
    configuration["_config_text"] = configuration_text

    if seed_override is not None:
        configuration["seed"] = int(seed_override)
    if device_override is not None:
        configuration["device"] = str(device_override)

    if output_directory_override is not None:
        # Keep caches inside the output directory so runs are self-contained.
        output_directory_absolute = os.path.abspath(output_directory_override)
        configuration["artifacts_dir"] = output_directory_absolute
        configuration["cache_dir"] = os.path.join(output_directory_absolute, "cache")

    configuration_directory = os.path.dirname(os.path.abspath(config_path))
    configuration["artifacts_dir"] = _resolve_path_relative_to_directory(
        configuration_directory, str(configuration["artifacts_dir"])
    )
    configuration["cache_dir"] = _resolve_path_relative_to_directory(
        configuration_directory, str(configuration["cache_dir"])
    )
    configuration["data"]["dir"] = _resolve_path_relative_to_directory(
        configuration_directory, str(configuration["data"]["dir"])
    )

    return configuration


def _find_latest_run_directory(artifacts_directory: str) -> str:
    """Return the most recently created run directory under an artefacts folder.

    The quickstart configuration stores caches under the same top-level output
    folder, so this function only considers subdirectories that actually contain
    a ``metrics.json`` file. Run directories are timestamp-named, so sorting the
    directory names lexicographically returns the most recent run last.
    """

    candidate_directory_names = sorted(os.listdir(artifacts_directory))
    run_directories = []
    for candidate_directory_name in candidate_directory_names:
        candidate_directory_path = os.path.join(
            artifacts_directory, candidate_directory_name
        )
        candidate_metrics_path = os.path.join(candidate_directory_path, "metrics.json")
        if os.path.isdir(candidate_directory_path) and os.path.exists(
            candidate_metrics_path
        ):
            run_directories.append(candidate_directory_path)

    if not run_directories:
        raise FileNotFoundError(
            f"No run directories found under: {artifacts_directory}"
        )
    return run_directories[-1]


def _read_metrics(metrics_path: str) -> dict[str, Any]:
    """Read a metrics JSON file produced by the training pipeline.

    The CLI keeps evaluation lightweight by reading the stored metrics rather
    than re-running the model. This is useful for quickly checking run results
    without repeating any training work.
    """

    with open(metrics_path, "r", encoding="utf-8") as metrics_file:
        return json.load(metrics_file)


def _command_train(arguments: argparse.Namespace) -> int:
    """Handle the ``train`` subcommand by running the existing pipeline.

    This command loads a configuration file, applies any command line overrides,
    and then calls ``graph_lp.train.run``. The run artefacts are written by the
    training code to the configured output directory.
    """

    configuration, configuration_text = _load_configuration(arguments.config)
    _validate_training_configuration(configuration)
    prepared_configuration = _prepare_configuration_for_run(
        configuration=configuration,
        configuration_text=configuration_text,
        config_path=arguments.config,
        seed_override=arguments.seed,
        device_override=arguments.device,
        output_directory_override=arguments.output_dir,
    )
    variant_from_config = prepared_configuration.get("variant")
    variant = str(arguments.variant or variant_from_config or "hybrid")
    training_module.run(prepared_configuration, variant)
    return 0


def _command_evaluate(arguments: argparse.Namespace) -> int:
    """Handle the ``evaluate`` subcommand by reading stored run metrics.

    The current project writes ``metrics.json`` during training. This command
    locates the latest run directory for the configured output folder and prints
    a short summary, which provides a fast way to check results.
    """

    configuration, configuration_text = _load_configuration(arguments.config)
    _validate_training_configuration(configuration)
    prepared_configuration = _prepare_configuration_for_run(
        configuration=configuration,
        configuration_text=configuration_text,
        config_path=arguments.config,
        seed_override=arguments.seed,
        device_override=arguments.device,
        output_directory_override=arguments.output_dir,
    )

    if arguments.run_dir is not None:
        run_directory = os.path.abspath(arguments.run_dir)
    else:
        run_directory = _find_latest_run_directory(
            str(prepared_configuration["artifacts_dir"])
        )

    metrics_path = os.path.join(run_directory, "metrics.json")
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"Expected metrics file does not exist: {metrics_path}")

    metrics = _read_metrics(metrics_path)
    test_metrics = metrics.get("test", {})
    test_roc_area_under_curve = test_metrics.get("roc_auc")
    if test_roc_area_under_curve is not None:
        print(f"Loaded metrics from: {metrics_path}")
        print(f"Test ROC-AUC: {test_roc_area_under_curve}")
    else:
        print(f"Loaded metrics from: {metrics_path}")
    return 0


def _command_run(arguments: argparse.Namespace) -> int:
    """Handle the ``run`` subcommand by running training followed by evaluation.

    This convenience command keeps the usual workflow to a single invocation.
    It first executes ``train`` and then reports metrics for the latest run
    directory created under the output directory.
    """

    train_exit_code = _command_train(arguments)
    if train_exit_code != 0:
        return int(train_exit_code)
    return _command_evaluate(arguments)


def _build_parser() -> argparse.ArgumentParser:
    """Create the top-level argument parser with subcommands.

    The parser is designed to keep the CLI familiar to typical Python tools.
    It uses subcommands rather than separate scripts so that help text and
    common options are presented in a single entry point.
    """

    parser = argparse.ArgumentParser(prog="graph-lp")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common_options(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument(
            "--config",
            required=True,
            help="Path to a YAML or JSON configuration file.",
        )
        subparser.add_argument(
            "--seed",
            type=int,
            default=None,
            help="Optional random seed override.",
        )
        subparser.add_argument(
            "--device",
            type=str,
            default=None,
            help="Optional device override, for example 'cpu'.",
        )
        subparser.add_argument(
            "--output-dir",
            type=str,
            default=None,
            help="Optional output directory override for run artefacts.",
        )

    train_parser = subparsers.add_parser("train", help="Train a model end-to-end.")
    add_common_options(train_parser)
    train_parser.add_argument(
        "--variant",
        choices=["structural", "semantic", "hybrid"],
        default=None,
        help="Optional embedding variant. If omitted, the config value is used.",
    )
    train_parser.set_defaults(handler=_command_train)

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="Evaluate an existing run by reading stored metrics."
    )
    add_common_options(evaluate_parser)
    evaluate_parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Optional explicit run directory to evaluate.",
    )
    evaluate_parser.set_defaults(handler=_command_evaluate)

    run_parser = subparsers.add_parser(
        "run", help="Convenience command that runs train then evaluate."
    )
    add_common_options(run_parser)
    run_parser.add_argument(
        "--variant",
        choices=["structural", "semantic", "hybrid"],
        default=None,
        help="Optional embedding variant. If omitted, the config value is used.",
    )
    # The run subcommand reuses the evaluate implementation, so it must provide the
    # same attribute on the parsed arguments namespace to avoid runtime failures.
    run_parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Optional explicit run directory to evaluate.",
    )
    run_parser.set_defaults(handler=_command_run)

    return parser


def main(argument_list: list[str] | None = None) -> int:
    """Entry point for the ``graph-lp`` console script.

    The function returns an integer exit code so it can be tested without
    spawning a subprocess. Configuration parsing errors return code 2, which
    matches the conventional behaviour of ``argparse`` for invalid input.
    """

    parser = _build_parser()
    try:
        arguments = parser.parse_args(argument_list)
    except SystemExit as system_exit_exception:
        return (
            int(system_exit_exception.code)
            if system_exit_exception.code is not None
            else 1
        )

    handler: Callable[[argparse.Namespace], int] = arguments.handler
    try:
        return int(handler(arguments))
    except (FileNotFoundError, ValueError) as exception:
        print(str(exception), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())  # pragma: no cover
