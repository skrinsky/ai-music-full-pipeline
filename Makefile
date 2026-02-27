.PHONY: help setup setup-force venv run clean blues-info blues-fetch gigamidi-info gigamidi-fetch blues-preprocess blues-train blues-resume blues-generate bg generate gen blues-audition blues-browse chorale-convert chorale-preprocess chorale-train chorale-resume chorale-retrain chorale-generate cg chorale-audition chorale-browse cascade-preprocess-a cascade-preprocess-b cascade-train cascade-generate cascade-eval chorale-cascade-preprocess chorale-cascade-train chorale-cascade-generate chorale-cascade-eval
.DEFAULT_GOAL := help

VENV_DIR := .venv-ai-music
ACTIVATE := $(VENV_DIR)/bin/activate
PYTHON := $(VENV_DIR)/bin/python
export PYTHONPATH := $(CURDIR)

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

blues-resume: $(BLUES_CKPT) ## Resume blues training from latest checkpoint
	$(PYTHON) training/train.py \
	  --data_dir $(BLUES_EVENTS) \
	  --train_pkl $(BLUES_EVENTS)/events_train.pkl \
	  --val_pkl $(BLUES_EVENTS)/events_val.pkl \
	  --vocab_json $(BLUES_EVENTS)/event_vocab.json \
	  --save_path $(BLUES_CKPT) \
	  --resume $(BLUES_CKPT) \
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

# JSB Chorales dataset, originally from TonicNet (omarperacha/TonicNet)
CHORALE_NPZ   := data/Jsb16thSeparated.npz
CHORALE_MIDI  := data/chorales_midi
CHORALE_EVENTS := runs/chorale_events
CHORALE_CKPT  := runs/checkpoints/chorale_model.pt

data/chorales_midi/.converted:
	$(PYTHON) scripts/convert_chorales_npz_to_midi.py \
	  --npz $(CHORALE_NPZ) --out_dir $(CHORALE_MIDI) --bpm 100 --normalize-key
	@touch $@

chorale-convert: ## Convert Bach chorale NPZ → MIDI files
	$(PYTHON) scripts/convert_chorales_npz_to_midi.py \
	  --npz $(CHORALE_NPZ) --out_dir $(CHORALE_MIDI) --bpm 100 --normalize-key $(ARGS)

chorale-audition: data/chorales_midi/.converted ## Audition chorale MIDIs (stats/list/info/play)
	$(PYTHON) scripts/audition_gigamidi.py stats --folder $(CHORALE_MIDI) --instrument_set chorale4 $(ARGS)

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

chorale-resume: $(CHORALE_CKPT) ## Resume chorale training from latest checkpoint
	$(PYTHON) training/train.py \
	  --data_dir $(CHORALE_EVENTS) \
	  --train_pkl $(CHORALE_EVENTS)/events_train.pkl \
	  --val_pkl $(CHORALE_EVENTS)/events_val.pkl \
	  --vocab_json $(CHORALE_EVENTS)/event_vocab.json \
	  --save_path $(CHORALE_CKPT) \
	  --resume $(CHORALE_CKPT) \
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

# --- Cascaded-by-instrument pipeline ---

CASCADE_EVENTS_A := runs/cascade_events_a
CASCADE_EVENTS_B := runs/cascade_events_b
CASCADE_CKPT     := runs/checkpoints/cascade_model.pt

cascade-preprocess-a: data/blues_midi/.fetched ## Cascade preprocess ablation A (6 stages)
	$(PYTHON) training/pre_cascade.py \
	  --midi_folder $(BLUES_MIDI) --data_folder $(CASCADE_EVENTS_A) \
	  --ablation A --blues_only $(ARGS)

cascade-preprocess-b: data/blues_midi/.fetched ## Cascade preprocess ablation B (5 stages, merged guitar+other)
	$(PYTHON) training/pre_cascade.py \
	  --midi_folder $(BLUES_MIDI) --data_folder $(CASCADE_EVENTS_B) \
	  --ablation B --blues_only $(ARGS)

cascade-train: ## Train cascade model (set CASCADE_DIR=runs/cascade_events_a or _b)
	@test -n "$(CASCADE_DIR)" || { echo "ERROR: set CASCADE_DIR (e.g. CASCADE_DIR=runs/cascade_events_a)"; exit 1; }
	$(PYTHON) training/train_cascade.py \
	  --data_dir $(CASCADE_DIR) \
	  --train_pkl $(CASCADE_DIR)/cascade_train.pkl \
	  --val_pkl $(CASCADE_DIR)/cascade_val.pkl \
	  --vocab_json $(CASCADE_DIR)/cascade_vocab.json \
	  --save_path $(CASCADE_CKPT) \
	  --device auto $(ARGS)

cascade-generate: $(CASCADE_CKPT) ## Generate from cascade model
	@CASCADE_VOCAB=$$(ls -t runs/cascade_events_*/cascade_vocab.json 2>/dev/null | head -1); \
	test -n "$$CASCADE_VOCAB" || { echo "ERROR: no cascade_vocab.json found"; exit 1; }; \
	echo "Using vocab: $$CASCADE_VOCAB"; \
	$(PYTHON) training/generate_cascade.py \
	  --ckpt $(CASCADE_CKPT) \
	  --vocab_json "$$CASCADE_VOCAB" \
	  --out_midi runs/generated/cascade_out.mid \
	  --device cpu $(ARGS)

cascade-eval: ## Evaluate cascade-generated MIDI
	@CASCADE_VOCAB=$$(ls -t runs/cascade_events_*/cascade_vocab.json 2>/dev/null | head -1); \
	test -n "$$CASCADE_VOCAB" || { echo "ERROR: no cascade_vocab.json found"; exit 1; }; \
	$(PYTHON) training/eval_cascade.py \
	  --midi runs/generated/cascade_out.mid \
	  --vocab_json "$$CASCADE_VOCAB" $(ARGS)

# --- Chorale cascade pipeline (bassvox → tenor → alto → soprano) ---

CHORALE_CASCADE_EVENTS := runs/chorale_cascade_events
CHORALE_CASCADE_CKPT   := runs/checkpoints/chorale_cascade_model.pt

chorale-cascade-preprocess: data/chorales_midi/.converted ## Cascade preprocess chorales (bassvox→tenor→alto→soprano)
	$(PYTHON) training/pre_cascade.py \
	  --midi_folder $(CHORALE_MIDI) --data_folder $(CHORALE_CASCADE_EVENTS) \
	  --ablation A --instrument_set chorale4 $(ARGS)

chorale-cascade-train: $(CHORALE_CASCADE_EVENTS)/cascade_train.pkl ## Train chorale cascade model
	$(PYTHON) training/train_cascade.py \
	  --data_dir $(CHORALE_CASCADE_EVENTS) \
	  --train_pkl $(CHORALE_CASCADE_EVENTS)/cascade_train.pkl \
	  --val_pkl $(CHORALE_CASCADE_EVENTS)/cascade_val.pkl \
	  --vocab_json $(CHORALE_CASCADE_EVENTS)/cascade_vocab.json \
	  --save_path $(CHORALE_CASCADE_CKPT) \
	  --device auto $(ARGS)

$(CHORALE_CASCADE_EVENTS)/cascade_train.pkl: data/chorales_midi/.converted
	$(PYTHON) training/pre_cascade.py \
	  --midi_folder $(CHORALE_MIDI) --data_folder $(CHORALE_CASCADE_EVENTS) \
	  --ablation A --instrument_set chorale4

chorale-cascade-generate: $(CHORALE_CASCADE_CKPT) ## Generate chorale from cascade model
	$(PYTHON) training/generate_cascade.py \
	  --ckpt $(CHORALE_CASCADE_CKPT) \
	  --vocab_json $(CHORALE_CASCADE_EVENTS)/cascade_vocab.json \
	  --out_midi runs/generated/chorale_cascade_out.mid \
	  --device cpu --ablation A --instrument_set chorale4 $(ARGS)

chorale-cascade-eval: ## Evaluate chorale cascade-generated MIDI
	$(PYTHON) training/eval_cascade.py \
	  --midi runs/generated/chorale_cascade_out.mid \
	  --vocab_json $(CHORALE_CASCADE_EVENTS)/cascade_vocab.json \
	  --instrument_set chorale4 $(ARGS)
