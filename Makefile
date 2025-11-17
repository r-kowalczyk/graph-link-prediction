.PHONY: install check test build clean-build clean help install-pyg-deps

install: ## Install the package and pre-commit hooks
	@echo "Creating virtual environment using uv"
	@uv sync --all-groups
	@echo "Installing pre-commit hooks"
	@uv run pre-commit install
	@echo "Installing PyTorch Geometric dependencies..."
	@uv run python scripts/install_pyg_deps.py || echo "Warning: PyG dependencies installation failed (you can install manually later with 'make install-pyg-deps')"
	@echo ""
	@echo "Installation complete!"
	@echo "  - Use 'uv run' to execute commands"
	@echo "  - Or activate with 'source .venv/bin/activate'"

install-pyg-deps: ## Install PyTorch Geometric extension packages (required for structural embeddings)
	@echo "Installing PyTorch Geometric dependencies..."
	@uv run python scripts/install_pyg_deps.py

check: ## Run code quality tools
	@echo "Checking lock file consistency with 'pyproject.toml'"
	@uv lock --check
	@echo "Linting code: Running pre-commit"
	@uv run pre-commit run -a
	@echo "Static type checking: Running mypy"
	@uv run mypy graph_lp tests

test: ## Test the code with pytest
	@echo "Testing code: Running pytest"
	@uv run python -m pytest --cov=graph_lp --cov-report=term-missing -q

build: clean-build ## Build wheel file
	@echo "Creating wheel file"
	@uvx --from build pyproject-build --installer uv

clean-build: ## Clean build artifacts
	@echo "Removing build artifacts"
	@uv run python -c "import shutil; import os; shutil.rmtree('dist') if os.path.exists('dist') else None"

clean: ## Clean Python cache and test artifacts
	@echo "Removing Python cache files"
	@find . -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete
	@rm -rf .pytest_cache
	@rm -rf .coverage
	@rm -rf htmlcov
	@rm -rf dist
	@rm -rf build
	@rm -rf *.egg-info

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
