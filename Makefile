.PHONY: help setup setup-force venv run clean blues-info blues-fetch gigamidi-info gigamidi-fetch blues-preprocess blues-train blues-generate bg generate gen blues-audition blues-browse chorale-convert chorale-preprocess chorale-train chorale-retrain chorale-generate cg chorale-audition chorale-browse
.DEFAULT_GOAL := help

VENV_DIR := .venv-ai-music
ACTIVATE := $(VENV_DIR)/bin/activate
PYTHON := $(VENV_DIR)/bin/python

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+( [a-zA-Z_-]+)*:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-18s %s\n", $$1, $$2}'

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

blues-audition: data/blues_midi/.fetched ## Audition blues MIDIs (stats/list/info/play)
	$(PYTHON) scripts/audition_gigamidi.py stats --folder $(BLUES_MIDI) $(ARGS)

blues-browse: data/blues_midi/.fetched ## Browse + play blues MIDIs (tkinter GUI)
	$(PYTHON) scripts/midi_browser.py --folder $(BLUES_MIDI) $(ARGS)

blues-preprocess: data/blues_midi/.fetched ## Preprocess blues MIDIs → event tokens
	$(PYTHON) training/pre.py --midi_folder $(BLUES_MIDI) --data_folder $(BLUES_EVENTS) --blues_only $(ARGS)

blues-train: $(BLUES_EVENTS)/events_train.pkl ## Train on preprocessed blues events
	$(PYTHON) training/train.py \
	  --data_dir $(BLUES_EVENTS) \
	  --train_pkl $(BLUES_EVENTS)/events_train.pkl \
	  --val_pkl $(BLUES_EVENTS)/events_val.pkl \
	  --vocab_json $(BLUES_EVENTS)/event_vocab.json \
	  --save_path $(BLUES_CKPT) \
	  --device auto $(ARGS)

blues-retrain: ## make blues-preprocess && make blues-train
	make blues-preprocess && make blues-train

$(BLUES_EVENTS)/events_train.pkl: data/blues_midi/.fetched
	$(PYTHON) training/pre.py --midi_folder $(BLUES_MIDI) --data_folder $(BLUES_EVENTS)

blues-generate bg: $(BLUES_CKPT) ## Generate blues MIDI from trained model
	$(PYTHON) training/generate.py \
	  --ckpt $(BLUES_CKPT) \
	  --vocab_json $(BLUES_EVENTS)/event_vocab.json \
	  --out_midi runs/generated/blues_out.mid \
	  --device auto $(ARGS)

# --- Bach chorale pipeline (NPZ → MIDI → events → train → generate) ---

CHORALE_NPZ   := /w/TonicNet/dataset_unprocessed/Jsb16thSeparated.npz
CHORALE_MIDI  := data/chorales_midi
CHORALE_EVENTS := runs/chorale_events
CHORALE_CKPT  := runs/checkpoints/chorale_model.pt

data/chorales_midi/.converted:
	$(PYTHON) scripts/convert_chorales_npz_to_midi.py \
	  --npz $(CHORALE_NPZ) --out_dir $(CHORALE_MIDI) --bpm 100
	@touch $@

chorale-convert: ## Convert Bach chorale NPZ → MIDI files
	$(PYTHON) scripts/convert_chorales_npz_to_midi.py \
	  --npz $(CHORALE_NPZ) --out_dir $(CHORALE_MIDI) --bpm 100 $(ARGS)

chorale-audition: data/chorales_midi/.converted ## Audition chorale MIDIs (stats/list/info/play)
	$(PYTHON) scripts/audition_gigamidi.py stats --folder $(CHORALE_MIDI) $(ARGS)

chorale-browse: data/chorales_midi/.converted ## Browse + play chorale MIDIs (tkinter GUI)
	$(PYTHON) scripts/midi_browser.py --folder $(CHORALE_MIDI) $(ARGS)

chorale-preprocess: data/chorales_midi/.converted ## Preprocess chorale MIDIs → event tokens
	$(PYTHON) training/pre.py --midi_folder $(CHORALE_MIDI) --data_folder $(CHORALE_EVENTS) --instrument_set chorale4 $(ARGS)

chorale-train: $(CHORALE_EVENTS)/events_train.pkl ## Train on preprocessed chorale events
	$(PYTHON) training/train.py \
	  --data_dir $(CHORALE_EVENTS) \
	  --train_pkl $(CHORALE_EVENTS)/events_train.pkl \
	  --val_pkl $(CHORALE_EVENTS)/events_val.pkl \
	  --vocab_json $(CHORALE_EVENTS)/event_vocab.json \
	  --save_path $(CHORALE_CKPT) \
	  --device auto $(ARGS)

chorale-retrain: ## make chorale-preprocess && make chorale-train
	make chorale-preprocess && make chorale-train

$(CHORALE_EVENTS)/events_train.pkl: data/chorales_midi/.converted
	$(PYTHON) training/pre.py --midi_folder $(CHORALE_MIDI) --data_folder $(CHORALE_EVENTS) --instrument_set chorale4

chorale-generate cg: $(CHORALE_CKPT) ## Generate chorale MIDI from trained model
	$(PYTHON) training/generate.py \
	  --ckpt $(CHORALE_CKPT) \
	  --vocab_json $(CHORALE_EVENTS)/event_vocab.json \
	  --out_midi runs/generated/chorale_out.mid \
	  --device auto --drum_bonus 0.0 $(ARGS)

# --- Generate from latest checkpoint (any pipeline) ---

LATEST_CKPT = $(shell ls -t runs/checkpoints/*.pt 2>/dev/null | head -1)
LATEST_VOCAB = $(shell ls -t runs/*/event_vocab.json 2>/dev/null | head -1)

generate gen: ## Generate from latest checkpoint (ARGS="--seed_midi foo.mid --seed_bars 4")
	@test -n "$(LATEST_CKPT)" || { echo "ERROR: no checkpoint found in runs/checkpoints/"; exit 1; }
	@test -n "$(LATEST_VOCAB)" || { echo "ERROR: no event_vocab.json found in runs/"; exit 1; }
	@echo "Using checkpoint: $(LATEST_CKPT)"
	@echo "Using vocab:      $(LATEST_VOCAB)"
	$(PYTHON) training/generate.py \
	  --ckpt "$(LATEST_CKPT)" \
	  --vocab_json "$(LATEST_VOCAB)" \
	  --out_midi runs/generated/out.mid \
	  --device auto $(ARGS)
	open runs/generated/out.mid
