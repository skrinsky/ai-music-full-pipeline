.PHONY: help setup setup-force venv run clean
.DEFAULT_GOAL := help

VENV_DIR := .venv-ai-music
ACTIVATE := $(VENV_DIR)/bin/activate

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

setup: ## Create venv via uv (Python 3.10)
	bash scripts/setup_venv.sh

setup-force: ## Re-run venv setup (force reinstall)
	PYTHON_BIN=$${PYTHON_BIN:-python3.10} bash scripts/setup_venv.sh

venv: ## Print venv activation command
	@echo "To activate:"
	@echo "  source $(ACTIVATE)"

run: ## Run end-to-end pipeline (ARGS="--tracks drums,bass")
	bash scripts/run_end_to_end.sh $(ARGS)

clean: ## Remove venv + runs (keeps data/raw)
	rm -rf $(VENV_DIR) runs
