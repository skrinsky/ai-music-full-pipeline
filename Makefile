.PHONY: help setup setup-force venv run clean blues-info blues-fetch gigamidi-info gigamidi-fetch blues-preprocess blues-train blues-generate
.DEFAULT_GOAL := help

VENV_DIR := .venv-ai-music
ACTIVATE := $(VENV_DIR)/bin/activate
PYTHON := $(VENV_DIR)/bin/python

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-18s %s\n", $$1, $$2}'

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

blues-info: ## Show FMA blues track stats (metadata only)
	$(PYTHON) scripts/fetch_fma_blues.py --info

blues-fetch: ## Download FMA blues tracks into data/blues/
	$(PYTHON) scripts/fetch_fma_blues.py $(ARGS)

gigamidi-info: ## Count GigaMIDI blues tracks (streaming, no download)
	$(PYTHON) scripts/fetch_gigamidi_blues.py --info

gigamidi-fetch: ## Download GigaMIDI blues MIDIs into data/blues_midi/
	$(PYTHON) scripts/fetch_gigamidi_blues.py $(ARGS)

# --- Blues MIDI pipeline (skips audio→MIDI stage) ---

BLUES_MIDI    := data/blues_midi
BLUES_EVENTS  := runs/blues_events
BLUES_CKPT    := runs/checkpoints/blues_model.pt

data/blues_midi/.fetched:
	$(PYTHON) scripts/fetch_gigamidi_blues.py --out_dir $(BLUES_MIDI)
	@touch $@

blues-preprocess: data/blues_midi/.fetched ## Preprocess blues MIDIs → event tokens
	$(PYTHON) training/pre.py --midi_folder $(BLUES_MIDI) --data_folder $(BLUES_EVENTS) $(ARGS)

blues-train: $(BLUES_EVENTS)/events_train.pkl ## Train on preprocessed blues events
	$(PYTHON) training/train.py \
	  --data_dir $(BLUES_EVENTS) \
	  --train_pkl $(BLUES_EVENTS)/events_train.pkl \
	  --val_pkl $(BLUES_EVENTS)/events_val.pkl \
	  --vocab_json $(BLUES_EVENTS)/event_vocab.json \
	  --save_path $(BLUES_CKPT) \
	  --device auto $(ARGS)

$(BLUES_EVENTS)/events_train.pkl: data/blues_midi/.fetched
	$(PYTHON) training/pre.py --midi_folder $(BLUES_MIDI) --data_folder $(BLUES_EVENTS)

blues-generate: $(BLUES_CKPT) ## Generate blues MIDI from trained model
	$(PYTHON) training/generate.py \
	  --ckpt $(BLUES_CKPT) \
	  --vocab_json $(BLUES_EVENTS)/event_vocab.json \
	  --out_midi runs/generated/blues_out.mid \
	  --device auto $(ARGS)
