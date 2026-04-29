# AI Music Full Pipeline

End-to-end music generation:

**Stereo audio → stems/MIDI (submodule) → event preprocessing → training → generation**

The audio→MIDI stage is a git submodule at `vendor/all-in-one-ai-midi-pipeline/`. The rest of the repo has grown into several parallel training pipelines — blues-from-MIDI (GigaMIDI), Bach chorales (JSB), cascade-by-instrument, dense chorale, LoRA finetune, and Notochord finetune — all driven from the top-level `Makefile`.

For the full map of pipelines, entry points, and architecture, see **[CLAUDE.md](CLAUDE.md)**.

Looking for the DAW plugin? → **[plugin/README.md](plugin/README.md)**

## Clone (with submodule)

```bash
git clone --recurse-submodules <THIS_REPO_URL>
# or, if already cloned:
git submodule update --init --recursive
```

## Setup

```bash
make setup                         # uv-based venv in .venv-ai-music (Python 3.10)
source .venv-ai-music/bin/activate
```

## Discover what you can run

```bash
make help
```

lists every target with a one-line description. Pass extra flags through with `ARGS=...`, e.g. `make blues-train ARGS="--max_d_model 128"`.

## Common flows

```bash
# Blues MIDI pipeline (no audio stage)
make gigamidi-fetch                # ~1000 GigaMIDI blues MIDIs → data/blues_midi/
make blues-preprocess
make blues-train                   # or: make blues-resume
make bg                            # generate

# Bach chorale pipeline
curl -L -o data/Jsb16thSeparated.npz \
  https://github.com/omarperacha/TonicNet/raw/master/dataset_unprocessed/Jsb16thSeparated.npz
make chorale-convert               # NPZ → 305 MIDIs in data/chorales_midi/
make chorale-preprocess && make chorale-train
make cg                            # generate

# Full audio → MIDI → training pipeline
mkdir -p data/raw && cp /path/to/*.wav data/raw/
scripts/run_end_to_end.sh                              # all instruments
scripts/run_end_to_end.sh --tracks drums,bass,guitar   # subset
scripts/run_end_to_end.sh --device {cuda,mps,cpu}      # device override

# Generate from the latest checkpoint in runs/checkpoints/
make gen
```

Shortcut aliases: `bg`=blues-generate, `cg`=chorale-generate, `cdg`=chorale-dense-generate, `fg`=ft-generate, `ng`=noto-generate.

## Device selection

Training/generation defaults to `--device auto`: CUDA if available, else MPS, else CPU. Override with `--device {cuda,mps,cpu}`. Note: Notochord finetuning is pinned to CPU — MPS produces NaN loss on that model.

## Outputs (all git-ignored)

- `out_midis/` — MIDIs exported from the audio→MIDI stage
- `runs/events/`, `runs/blues_events/`, `runs/chorale_events/`, … — preprocessed event datasets
- `runs/checkpoints/` — trained checkpoints
- `runs/generated/` — generated MIDI
- `finetune/runs/` — finetune adapters, data, outputs

## Tests

```bash
pytest tests/
```
