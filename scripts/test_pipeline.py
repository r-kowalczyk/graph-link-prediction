#!/usr/bin/env python3
"""Quick pipeline test entrypoint."""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

# Add project root to path so we can import graph_lp
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def _is_colab() -> bool:
    try:
        __import__("google.colab")
        return True
    except Exception:
        return False


def _ensure_pyg_on_colab(variant: str) -> None:
    if not (_is_colab() and variant in ("structural", "hybrid")):
        return
    try:
        __import__("torch_geometric")  # type: ignore
        return
    except Exception:
        pass
    import torch

    tv = torch.__version__.split("+")[0]
    cv = torch.version.cuda.replace(".", "") if torch.version.cuda else None
    base = (
        f"https://data.pyg.org/whl/torch-{tv}+cu{cv}.html"
        if cv
        else f"https://data.pyg.org/whl/torch-{tv}+cpu.html"
    )
    pkgs = [
        "torch-scatter",
        "torch-sparse",
        "torch-cluster",
        "torch-spline-conv",
        "pyg-lib",
    ]
    for p in pkgs:
        subprocess.check_call([sys.executable, "-m", "pip", "install", p, "-f", base])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "torch-geometric"])


def main():
    """Run a minimal test of the pipeline."""
    parser = argparse.ArgumentParser(description="Quick pipeline test")
    parser.add_argument(
        "--variant",
        type=str,
        choices=["structural", "semantic", "hybrid"],
        default="hybrid",
        help="Embedding variant to test",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/test.yaml",
        help="Path to config file",
    )
    args = parser.parse_args()

    config_path = project_root / args.config
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        config_text = f.read()
        cfg = yaml.safe_load(config_text)
        cfg["_config_text"] = config_text

    print(f"Running test with variant: {args.variant}")
    print(f"Using config: {args.config}")
    print("-" * 60)

    _ensure_pyg_on_colab(args.variant)
    if args.variant in ("structural", "hybrid"):
        import torch

        if not torch.cuda.is_available():
            print("Error: structural/hybrid requires a CUDA runtime (use Colab GPU).")
            sys.exit(1)
    from graph_lp.train import run  # Imported after potential Colab setup

    results = run(cfg, args.variant)

    print("-" * 60)
    if "test" in results:
        t = results["test"]
        print(
            f"Test ROC-AUC: {t.get('roc_auc', 'N/A'):.4f}  PR-AUC: {t.get('pr_auc', 'N/A'):.4f}  F1: {t.get('f1', 'N/A'):.4f}"
        )
    if "test_logreg" in results:
        lr = results["test_logreg"]
        print(f"LogReg ROC-AUC: {lr.get('roc_auc', 'N/A'):.4f}")
    print(f"Artefacts: {cfg['artifacts_dir']}")


if __name__ == "__main__":
    main()
