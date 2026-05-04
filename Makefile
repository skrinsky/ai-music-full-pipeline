.PHONY: help setup setup-force venv run clean blues-info blues-fetch gigamidi-info gigamidi-fetch blues-preprocess blues-train blues-resume blues-generate bg generate gen blues-audition blues-browse chorale-convert chorale-preprocess chorale-train chorale-resume chorale-retrain chorale-generate cg chorale-audition chorale-browse cascade-preprocess-a cascade-preprocess-b cascade-train cascade-generate cascade-eval chorale-cascade-preprocess chorale-cascade-train chorale-cascade-generate chorale-cascade-eval chorale-dense-preprocess chorale-dense-train chorale-dense-resume chorale-dense-generate cdg ft-install ft-convert ft-train ft-generate fg noto-convert noto-train noto-generate ng plugin-debug plugin-release plugin-build plugin-clean plugin-rebuild plugin-reconfigure plugin-uninstall plugin-validate pd pr pb pcfg prb pc pu pv disc-data disc-train disc-train-combined slakh-fetch slakh-stems nam-fetch
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

slakh-fetch: ## Download Slakh2100 MIDI stems from Zenodo (first N tracks; default N=100)
	$(PYTHON) scripts/fetch_slakh.py --out_dir data/slakh --n_tracks 100 $(ARGS)

slakh-stems: ## Fetch stems/ FLACs for already-downloaded Slakh tracks (streams full archive)
	$(PYTHON) scripts/fetch_slakh.py --out_dir data/slakh --continue_stems $(ARGS)

data/slakh/.fetched:
	$(PYTHON) scripts/fetch_slakh.py --out_dir data/slakh --n_tracks 100 $(ARGS)

nam-fetch: ## Download free NAM bass/guitar amp captures for discriminator training
	$(PYTHON) scripts/fetch_nam_models.py --out_dir data/nam_models $(ARGS)

disc-data: data/slakh/.fetched ## Build note discriminator HDF5 (scalars + mel patches)
	$(PYTHON) scripts/build_discriminator_data.py \
	  --slakh_dir data/slakh/train \
	  --out runs/discriminator_data/notes.h5 $(ARGS)

disc-train: runs/discriminator_data/notes.h5 ## Train scalar MLP discriminator
	$(PYTHON) -m training.train_discriminator \
	  --data runs/discriminator_data/notes.h5 \
	  --out  runs/discriminator/model.pt \
	  --epochs 60 $(ARGS)

disc-train-combined: runs/discriminator_data/notes.h5 ## Train combined CNN+MLP discriminator (needs spec_patches)
	$(PYTHON) -m training.train_discriminator \
	  --data runs/discriminator_data/notes.h5 \
	  --out  runs/discriminator/combined_model.pt \
	  --combined --epochs 80 $(ARGS)

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
	$(PYTHON) training/pre.py --midi_folder $(CHORALE_MIDI) --data_folder $(CHORALE_EVENTS) --instrument_set chorale4 --seq_len 1024 $(ARGS)

chorale-train: $(CHORALE_EVENTS)/events_train.pkl ## Train on preprocessed chorale events
	$(PYTHON) training/train.py \
	  --data_dir $(CHORALE_EVENTS) \
	  --train_pkl $(CHORALE_EVENTS)/events_train.pkl \
	  --val_pkl $(CHORALE_EVENTS)/events_val.pkl \
	  --vocab_json $(CHORALE_EVENTS)/event_vocab.json \
	  --save_path $(CHORALE_CKPT) \
	  --seq_len 1024 \
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
	$(PYTHON) training/pre.py --midi_folder $(CHORALE_MIDI) --data_folder $(CHORALE_EVENTS) --instrument_set chorale4 --seq_len 1024

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
	  --device cpu --ablation A --instrument_set chorale4 \
	  --max_notes_per_step 1 --force_grid_mode straight $(ARGS)

chorale-cascade-eval: ## Evaluate chorale cascade-generated MIDI
	$(PYTHON) training/eval_cascade.py \
	  --midi runs/generated/chorale_cascade_out.mid \
	  --vocab_json $(CHORALE_CASCADE_EVENTS)/cascade_vocab.json \
	  --instrument_set chorale4 $(ARGS)

# --- Dense chorale pipeline (NPZ → dense tokens → train → generate) ---

CHORALE_DENSE_EVENTS := runs/chorale_dense_events
CHORALE_DENSE_CKPT   := runs/checkpoints/chorale_dense_model.pt

chorale-dense-preprocess: ## Dense preprocess: NPZ → compact token sequences
	$(PYTHON) training/pre_chorale_dense.py \
	  --npz $(CHORALE_NPZ) --data_folder $(CHORALE_DENSE_EVENTS) $(ARGS)

chorale-dense-train: $(CHORALE_DENSE_EVENTS)/dense_train.pkl ## Train dense chorale Transformer
	$(PYTHON) training/train_chorale_dense.py \
	  --data_dir $(CHORALE_DENSE_EVENTS) \
	  --save_path $(CHORALE_DENSE_CKPT) \
	  --device auto $(ARGS)

chorale-dense-resume: $(CHORALE_DENSE_CKPT) ## Resume dense chorale training
	$(PYTHON) training/train_chorale_dense.py \
	  --data_dir $(CHORALE_DENSE_EVENTS) \
	  --save_path $(CHORALE_DENSE_CKPT) \
	  --resume $(CHORALE_DENSE_CKPT) \
	  --device auto $(ARGS)

$(CHORALE_DENSE_EVENTS)/dense_train.pkl:
	$(PYTHON) training/pre_chorale_dense.py \
	  --npz $(CHORALE_NPZ) --data_folder $(CHORALE_DENSE_EVENTS)

chorale-dense-generate cdg: $(CHORALE_DENSE_CKPT) ## Generate dense chorale MIDI
	$(PYTHON) training/generate_chorale_dense.py \
	  --ckpt $(CHORALE_DENSE_CKPT) \
	  --out_midi runs/generated/chorale_dense_out.mid \
	  --device auto $(ARGS)

# ---------------------------------------------------------------------------
# Finetuning pipeline  (finetune/ folder)
# Starts from Natooz/Multitrack-Music-Transformer (pre-trained on ~170k MIDIs)
# and LoRA-adapts to your personal tracks.
# ---------------------------------------------------------------------------

FT_MIDI_DIR := summer_midi          # your personal tracks — override with ARGS or FT_MIDI_DIR=...
FT_DATA_DIR := finetune/runs/my_data
FT_ADAPTER  := finetune/runs/adapter
FT_GENERATED := finetune/runs/generated
BASE_MODEL  := NathanFradet/Maestro-REMI-bpe20k

ft-install: ## Install finetuning deps into the active venv
	$(PYTHON) -m pip install -r finetune/requirements.txt

ft-convert: ## Tokenize your MIDI files for finetuning (FT_MIDI_DIR=summer_midi)
	$(PYTHON) finetune/convert.py \
	  --midi_dir $(FT_MIDI_DIR) \
	  --out_dir  $(FT_DATA_DIR) $(ARGS)

ft-train: $(FT_DATA_DIR)/train_ids.npy ## LoRA-finetune from pre-trained music model
	$(PYTHON) finetune/finetune.py \
	  --data_dir   $(FT_DATA_DIR) \
	  --out_dir    $(FT_ADAPTER) \
	  --base_model $(BASE_MODEL) \
	  --device auto $(ARGS)

ft-generate fg: $(FT_ADAPTER)/best ## Generate MIDI from finetuned model
	$(PYTHON) finetune/generate.py \
	  --base_model $(BASE_MODEL) \
	  --adapter    $(FT_ADAPTER)/best \
	  --data_dir   $(FT_DATA_DIR) \
	  --out_midi   $(FT_GENERATED)/out.mid \
	  --device auto $(ARGS)
	@echo "Output: $(FT_GENERATED)/out.mid"

$(FT_DATA_DIR)/train_ids.npy:
	$(PYTHON) finetune/convert.py \
	  --midi_dir $(FT_MIDI_DIR) \
	  --out_dir  $(FT_DATA_DIR)

$(FT_ADAPTER)/best:
	$(PYTHON) finetune/finetune.py \
	  --data_dir   $(FT_DATA_DIR) \
	  --out_dir    $(FT_ADAPTER) \
	  --base_model $(BASE_MODEL) \
	  --device auto

# ---------------------------------------------------------------------------
# Notochord finetuning  (GRU model pre-trained on Lakh MIDI — rock/pop prior)
# ---------------------------------------------------------------------------

NOTO_CKPT    := finetune/notochord_lakh_50G_deep.pt
NOTO_DATA    := finetune/runs/noto_data
NOTO_FT_CKPT := finetune/runs/noto_finetuned.pt
NOTO_GEN     := finetune/runs/generated/noto_out.mid

noto-convert: ## Convert personal MIDIs to Notochord event sequences
	$(PYTHON) finetune/notochord_convert.py \
	  --midi_dir $(FT_MIDI_DIR) \
	  --out_dir  $(NOTO_DATA) $(ARGS)

noto-train: $(NOTO_DATA)/meta.json ## Fine-tune Notochord on personal MIDI data
	$(PYTHON) finetune/notochord_finetune.py \
	  --checkpoint $(NOTO_CKPT) \
	  --data_dir   $(NOTO_DATA) \
	  --out        $(NOTO_FT_CKPT) \
	  --freeze_backbone \
	  --epochs 20 \
	  --lr 3e-5 \
	  --device cpu $(ARGS)

noto-generate ng: $(NOTO_FT_CKPT) ## Generate MIDI from fine-tuned Notochord
	$(PYTHON) finetune/notochord_generate.py \
	  --checkpoint      $(NOTO_FT_CKPT) \
	  --base_checkpoint $(NOTO_CKPT) \
	  --data_dir        $(NOTO_DATA) \
	  --out_midi        $(NOTO_GEN) \
	  --device auto $(ARGS)
	@echo "Output: $(NOTO_GEN)"

$(NOTO_DATA)/meta.json:
	$(PYTHON) finetune/notochord_convert.py \
	  --midi_dir $(FT_MIDI_DIR) \
	  --out_dir  $(NOTO_DATA)

# ---------------------------------------------------------------------------
# AIMusicPlugin (JUCE AU + VST3)
# Built via CMake; AU is patched + signed + installed to ~/Library by
# the plugin's own POST_BUILD step.
# ---------------------------------------------------------------------------

PLUGIN_DIR        := plugin/AIMusicPlugin
PLUGIN_BUILD_DIR  := $(PLUGIN_DIR)/build
PLUGIN_JOBS       ?= 8
# Read PRODUCT_NAME from CMakeLists.txt so uninstall tracks renames.
PLUGIN_PRODUCT_NAME := $(shell sed -n 's/.*PRODUCT_NAME[[:space:]]*"\([^"]*\)".*/\1/p' $(PLUGIN_DIR)/CMakeLists.txt)
# Past PRODUCT_NAMEs — `pu` also removes these so renames don't leave orphans.
PLUGIN_LEGACY_NAMES := AI\ Music

plugin-debug pd: ## Build AIMusicPlugin (Debug, AU + VST3, installs to ~/Library)
	@$(MAKE) plugin-build PLUGIN_CONFIG=Debug

plugin-release pr: ## Build AIMusicPlugin (Release, AU + VST3, installs to ~/Library)
	@$(MAKE) plugin-build PLUGIN_CONFIG=Release

plugin-build pb: ## Configure (if needed) + build (set PLUGIN_CONFIG=Debug|Release, default Debug)
	@PLUGIN_CONFIG=$${PLUGIN_CONFIG:-Debug}; \
	if [ ! -f $(PLUGIN_BUILD_DIR)/CMakeCache.txt ]; then \
	  echo ">>> Configuring $(PLUGIN_DIR) ($$PLUGIN_CONFIG)"; \
	  cmake -S $(PLUGIN_DIR) -B $(PLUGIN_BUILD_DIR) -DCMAKE_BUILD_TYPE=$$PLUGIN_CONFIG; \
	fi; \
	echo ">>> Building $(PLUGIN_DIR) ($$PLUGIN_CONFIG, j=$(PLUGIN_JOBS))"; \
	cmake --build $(PLUGIN_BUILD_DIR) --config $$PLUGIN_CONFIG -j $(PLUGIN_JOBS)

plugin-reconfigure pcfg: ## Force re-run cmake configure for the plugin
	rm -f $(PLUGIN_BUILD_DIR)/CMakeCache.txt
	cmake -S $(PLUGIN_DIR) -B $(PLUGIN_BUILD_DIR) -DCMAKE_BUILD_TYPE=$${PLUGIN_CONFIG:-Debug}

plugin-rebuild prb: plugin-clean plugin-debug ## Clean + rebuild (Debug)

plugin-clean pc: ## Remove plugin build directory
	rm -rf $(PLUGIN_BUILD_DIR)

plugin-uninstall pu: ## Remove installed AU + VST3 from ~/Library/Audio/Plug-Ins (current + legacy names)
	@test -n "$(PLUGIN_PRODUCT_NAME)" || { echo "ERROR: could not parse PRODUCT_NAME from $(PLUGIN_DIR)/CMakeLists.txt"; exit 1; }
	@for name in "$(PLUGIN_PRODUCT_NAME)" $(PLUGIN_LEGACY_NAMES); do \
	  comp="$(HOME)/Library/Audio/Plug-Ins/Components/$$name.component"; \
	  vst3="$(HOME)/Library/Audio/Plug-Ins/VST3/$$name.vst3"; \
	  if [ -e "$$comp" ] || [ -e "$$vst3" ]; then \
	    echo "Removing $$name.component and $$name.vst3"; \
	    rm -rf "$$comp" "$$vst3"; \
	  fi; \
	done

plugin-validate pv: ## Validate the installed AU with auval (slow, ~30s)
	auval -v aumu Aimp Smkr
