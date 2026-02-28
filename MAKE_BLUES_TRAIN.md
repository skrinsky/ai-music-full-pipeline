# Blues Training Pipeline — Quick Start

## Prerequisites

1. **Venv**: `bash setup.bash && source .venv-ai-music/bin/activate`
2. **HuggingFace token**: must have accepted GigaMIDI terms and logged in
   ```bash
   # One-time setup:
   # 1. Accept terms at https://huggingface.co/datasets/Metacreation/GigaMIDI
   # 2. Create fine-grained token at https://huggingface.co/settings/tokens
   #    with "Read access to contents of all public gated repos you can access"
   # 3. Login (use the venv version):
   .venv-ai-music/bin/huggingface-cli login
   ```

## One-command pipeline

```bash
make blues-train
```

This runs the full dependency chain:
1. **Fetch** — streams GigaMIDI, filters ~996 blues MIDIs → `data/blues_midi/`
2. **Preprocess** — `pre.py` tokenizes MIDI → event sequences in `runs/blues_events/`
3. **Train** — transformer training, checkpoint → `runs/checkpoints/blues_model.pt`

Each step is skipped if its output already exists.

## Individual steps

```bash
make gigamidi-info                              # count blues tracks (no download)
make gigamidi-fetch ARGS="--max_tracks 20"      # test with small batch
make blues-preprocess                           # preprocess only
make blues-preprocess ARGS="--tracks drums,bass" # subset of instruments
make blues-train ARGS="--device mps"            # pass args to train.py
make blues-generate                             # generate from trained model
make blues-generate ARGS="--temperature 0.9"    # custom sampling
```

## What's happening under the hood

```
GigaMIDI (HuggingFace)
  │  streams parquet, filters genre columns for "blues"
  ▼
data/blues_midi/*.mid          (996 MIDI files, ~few hundred MiB)
  │  pre.py: MIDI → event tokens, 512-token windows
  │  augmentation: ±1 semitone pitch, ±10 velocity (train only)
  ▼
runs/blues_events/
  ├── events_train.pkl         (tokenized sequences + aux targets)
  ├── events_val.pkl
  └── event_vocab.json         (~2400 tokens)
  │  train.py: factored transformer, auto-scales to dataset size
  ▼
runs/checkpoints/blues_model.pt
  │  generate.py: grammar-constrained autoregressive sampling
  ▼
runs/generated/blues_out.mid
```

## Notes

- External MIDIs bypass the audio→MIDI stage entirely (no HTDemucs/Basic Pitch)
- Instrument mapping: drums detected via `is_drum` flag; named tracks (guitar, bass)
  matched by keyword; everything else → "other" slot
- GigaMIDI license: CC-BY-NC-4.0 (non-commercial research only)
- Dataset: ~996 blues tracks from 2.1M total, filtered across 5 genre metadata columns
