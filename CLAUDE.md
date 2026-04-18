# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

End-to-end music generation pipeline: **Stereo Audio → Stems/MIDI → Event Preprocessing → Transformer Training → MIDI Generation**

The audio→MIDI stage lives in a git submodule at `vendor/all-in-one-ai-midi-pipeline/`. The rest of the repo has grown into **several parallel training pipelines** (blues-from-MIDI, Bach chorales, cascade-by-instrument, dense chorale, LoRA finetune, Notochord finetune), all driven from a single top-level `Makefile`.

## Entry Points

The `Makefile` is the canonical driver — most tasks run as `make <target>`. Run `make help` to list every target with its description. Shortcut aliases: `bg`=blues-generate, `cg`=chorale-generate, `cdg`=chorale-dense-generate, `fg`=ft-generate, `ng`=noto-generate, `gen`/`generate` = generate from latest checkpoint in `runs/checkpoints/`.

Pass extra flags through with `ARGS=...`, e.g. `make blues-train ARGS="--max_d_model 128"`.

### Environment
```bash
make setup                         # uv-based venv in .venv-ai-music (Python 3.10)
source .venv-ai-music/bin/activate
```

### Pipelines (each has preprocess / train / resume / generate targets)

| Pipeline | Preprocess | Train | Generate | Code |
|---|---|---|---|---|
| Audio→MIDI→events (full) | `scripts/run_end_to_end.sh` | | | `training/{pre,train,generate}.py` |
| Blues MIDI (GigaMIDI) | `make blues-preprocess` | `make blues-train` / `blues-resume` | `make bg` | `training/{pre,train,generate}.py` |
| Bach chorale (JSB) | `make chorale-preprocess` | `make chorale-train` | `make cg` | uses `--instrument_set chorale4` |
| Chorale dense | `make chorale-dense-preprocess` | `make chorale-dense-train` | `make cdg` | `training/*_chorale_dense.py` |
| Cascade-by-instrument | `make cascade-preprocess-{a,b}` | `make cascade-train CASCADE_DIR=...` | `make cascade-generate` + `cascade-eval` | `training/*_cascade.py` |
| Chorale cascade | `make chorale-cascade-preprocess` | `make chorale-cascade-train` | `make chorale-cascade-generate` | `training/*_cascade.py` |
| LoRA finetune (Maestro-REMI) | `make ft-convert` | `make ft-train` | `make fg` | `finetune/{convert,finetune,generate}.py` |
| Notochord finetune (Lakh GRU) | `make noto-convert` | `make noto-train` | `make ng` | `finetune/notochord_*.py` |

### Data Setup (data/ is git-ignored)
- `make gigamidi-fetch` — ~1000 blues MIDIs into `data/blues_midi/`
- `curl -L -o data/Jsb16thSeparated.npz https://github.com/omarperacha/TonicNet/raw/master/dataset_unprocessed/Jsb16thSeparated.npz && make chorale-convert` — 305 Bach chorale MIDIs into `data/chorales_midi/`
- For the full audio→MIDI pipeline: drop `.wav` files into `data/raw/` and run `scripts/run_end_to_end.sh`.
- Notochord finetuning expects the pretrained checkpoint at `finetune/notochord_lakh_50G_deep.pt`.

### Tests
Uses `pytest`. Tests live in `tests/` (cascade pre/model/eval, chorale convert/dense/preprocess, instrument config). Run all: `pytest tests/`. Single test: `pytest tests/test_chorale_dense.py::test_name`.

## Architecture

### Main event-stream pipeline (`training/pre.py`, `train.py`, `generate.py`)
- **Preprocess**: multi-track MIDI → event token sequences. 512-token windows, 256-stride. Train-only augmentation (±1 semitone pitch, ±10 velocity). Produces `events_train.pkl`, `events_val.pkl`, `event_vocab.json`, plus a 34-dim polyphony auxiliary target.
- **Train**: Transformer encoder with factored output — type classification head + per-type value heads. Auto-scales model capacity to dataset size. Joint loss: token prediction (type 20% + value 80%) + auxiliary polyphony (20%). Defaults: `D_MODEL=192, N_HEADS=6, N_LAYERS=4`.
- **Generate**: autoregressive with a grammar-constrained state machine (`TIME → INST → VEL → PITCH → DUR`). Temperature + nucleus sampling. Adaptive grid snapping (straight vs triplet). Outputs MIDI + event JSON + metadata.

### Canonical instrument slots
Default 6-instrument set: `voxlead, voxharm, guitar, other, bass, drums`. Chorale set (`--instrument_set chorale4`): `bassvox, tenor, alto, soprano`. The `InstrumentConfig` / `INSTRUMENT_PRESETS` in `training/pre.py` is the single source of truth — event order per note is always `TIME_SHIFT → BAR → INST → VEL → PITCH → DUR`.

### Cascade variant (`training/*_cascade.py`)
Generates instruments sequentially rather than interleaved — one instrument stream conditions the next. Ablations: A (6 stages, one per instrument), B (5 stages, merged guitar+other). Adds cascade-specific vocab (`extend_vocab_for_cascade`) and chord labels.

### Dense chorale (`training/*_chorale_dense.py`)
Skips the event-stream tokenization and trains directly on compact NPZ-derived tokens — smaller, tighter vocab for 4-voice chorales.

### Finetune (`finetune/`)
Two separate post-training flows starting from pretrained public checkpoints:
- `finetune.py` — LoRA adapter on top of `NathanFradet/Maestro-REMI-bpe20k`.
- `notochord_finetune.py` — fine-tune the Notochord GRU from `notochord_lakh_50G_deep.pt`. Default training freezes the backbone (see commit `aabd088` — required to prevent catastrophic forgetting), trains on CPU (see `0312988` — MPS produces NaN loss on this model), uses Notochord's 1-indexed instrument IDs not GM (see `ad61ed0`).

### Key data formats
- `event_vocab.json` (~2400 tokens) — specialized pitch spaces (melodic vs percussion), plus cascade extensions.
- `events_*.pkl` — pickled event sequences (only load from trusted sources).
- Checkpoints (`.pt`) — `torch.save()` dict: model state + vocab + config metadata.

### Output directories (all git-ignored)
- `runs/events/`, `runs/blues_events/`, `runs/chorale_events/`, `runs/chorale_dense_events/`, `runs/cascade_events_{a,b}/`, `runs/chorale_cascade_events/` — preprocessed datasets
- `runs/checkpoints/` — trained model checkpoints
- `runs/generated/` — generated MIDI outputs
- `finetune/runs/` — finetune adapters, data, generated output
- `out_midis/` — exported MIDIs from the audio→MIDI stage

## Security Notes

- `pickle.load()` in `train.py` and `torch.load()` in `generate.py` are unguarded (no `weights_only=True`) — only load artifacts produced by this pipeline.
- `ClaudeNotes.md` contains JOS's curated highlights from prior Claude sessions; read it when historical context on a pipeline would help.
