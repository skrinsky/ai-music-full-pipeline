.PHONY: setup setup-force venv run clean

VENV_DIR := .venv-ai-music
ACTIVATE := $(VENV_DIR)/bin/activate

setup:
	bash scripts/setup_venv.sh

# re-run installs even if venv already exists (setup script already handles this, but this is a clear "do it again")
setup-force:
	PYTHON_BIN=$${PYTHON_BIN:-python3.10} bash scripts/setup_venv.sh

# convenience: print how to activate
venv:
	@echo "To activate:"
	@echo "  source $(ACTIVATE)"

# convenience: run end-to-end (pass ARGS="--audio-glob ... --tracks ...")
run:
	bash scripts/run_end_to_end.sh $(ARGS)

# optional cleanup target (does NOT delete your data/raw files; just removes venv + runs)
clean:
	rm -rf $(VENV_DIR) runs
