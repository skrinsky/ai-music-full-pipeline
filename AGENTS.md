# AGENTS.md

Guidance for AI coding agents working in this repository.

## What this repo is

**Mirror Mirror** — a local audio → MIDI → train → generate pipeline. A JUCE AU/VST3 plugin drives a Python FastAPI server that runs the pipeline. Everything runs locally; no cloud.

The same pipeline is also driven from the terminal via the top-level `Makefile`.

## Environment

- Python deps live in `.venv/` (managed by `uv`, Python 3.10).
- `make setup` creates the venv via `scripts/setup_venv.sh` (uv-based).
- The Makefile already sets `PYTHONPATH=$(CURDIR)` and uses `.venv/bin/python` directly — no need to activate the venv for `make` targets.
- For ad-hoc scripts: `source .venv/bin/activate`.
- Training device: `--device auto` picks CUDA → MPS → CPU.

## Common commands

```bash
# environment
make setup                 # create .venv via uv (idempotent)
make setup-force           # re-run setup
make help                  # show every target

# tests
pytest tests/              # all unit tests
pytest tests/test_chorale_dense.py -k some_test   # single test

# end-to-end (audio → MIDI → preprocess → train → generate)
scripts/run_end_to_end.sh

# blues MIDI flow (no audio stage)
make gigamidi-fetch && make blues-preprocess && make blues-train && make bg

# chorale flow (NPZ → MIDI → ...)
make chorale-convert && make chorale-preprocess && make chorale-train && make cg

# generate from latest checkpoint in runs/checkpoints/
make gen ARGS="--seed_midi foo.mid --seed_bars 4"

# plugin (macOS: builds + installs AU + VST3 to ~/Library/Audio/Plug-Ins/)
make plugin-build          # or: pb  — default Release; override with PLUGIN_CONFIG=Debug
make plugin-debug          # or: pd  — explicit Debug
make plugin-release        # or: pr  — explicit Release
make plugin-uninstall      # or: pu  (also removes legacy "AI Music" names)
make plugin-validate       # auval check on installed AU
make plugin-package VERSION=v0.1.0    # build Release + zip + publish GH release

# any target: extra flags via ARGS
make blues-train ARGS="--max_d_model 128"
```

Shortcut aliases: `bg` blues-generate · `cg` chorale-generate · `cdg` chorale-dense-generate · `fg` ft-generate · `gen` generate from latest ckpt · `pd`/`pr`/`pb`/`pc`/`prb`/`pcfg`/`pu`/`pv` plugin verbs.

## Architecture

### Submodules

`vendor/` contains git submodules. After cloning: `git submodule update --init --recursive`.
- `vendor/all-in-one-ai-midi-pipeline/` — audio → MIDI stage (Demucs + transcription). Driven by `pipeline.py run-batch` / `export-midi`.
- `vendor/mmt/` — Multitrack Music Transformer reference (used by finetune flow).

### Top-level data flow

```
audio files
   └─ vendor/all-in-one-ai-midi-pipeline   →  out_midis/*.mid
                                              └─ training/pre.py    →  runs/{project}/events/
                                                                       (events_{train,val}.pkl + event_vocab.json)
                                                                       └─ training/train.py →  runs/{project}/checkpoints/*.pt
                                                                                                └─ training/generate.py → runs/{project}/generated/*.mid
```

### `training/` — the four pipelines

Multiple parallel pipelines share the `pre / train / generate` triplet pattern:

| Pipeline | Preprocess | Train | Generate | Notes |
|---|---|---|---|---|
| **Standard** (blues, generic) | `pre.py` | `train.py` | `generate.py` | event-stream Transformer; default everywhere |
| **Cascade** (instrument-by-instrument) | `pre_cascade.py` | `train_cascade.py` | `generate_cascade.py` | `model_cascade.py`; eval via `eval_cascade.py`; ablations A/B |
| **Chorale dense** (Bach JSB) | `pre_chorale_dense.py` | `train_chorale_dense.py` | `generate_chorale_dense.py` | `model_chorale_dense.py`; compact per-step token rep |
| **Note discriminator** | `scripts/build_discriminator_data.py` | `train_discriminator.py` | (filters notes during `pre.py`) | scalar MLP or CNN+MLP; built on Slakh stems |

`pre.py` is parametrized by `--instrument_set`:
- `blues6` — voxlead, voxharm, guitar, other, bass, drums (default)
- `chorale4` — soprano, alto, tenor, bassvox

Token order in standard events: `TIME_SHIFT → BAR? → INST → VEL → PITCH → DUR`. Aux ("polyphony instructor") targets are computed per window; `aux_dim` depends on `instrument_set` (blues6=36, chorale4=24).

`train.py` auto-scales `d_model`/`n_layers`/`n_heads` to roughly match `tokens / params ≈ target_tpp`. Override via `ARGS="--max_d_model 128"` etc. Checkpoints save on val-loss improvement; `--resume` continues from the *best* (not most recent) checkpoint.

### `finetune/` — LoRA flow

Independent from `training/`. Starts from a pre-trained music transformer (default `NathanFradet/Maestro-REMI-bpe20k`) and LoRA-adapts to your own MIDIs. `make ft-install` to add deps; `make ft-convert ft-train fg` to run.

### `plugin/` — JUCE plugin + Python server

- `plugin/AIMusicPlugin/` — JUCE 8.0.3 source. CMake fetches JUCE automatically via CPM (cached under `~/.cache/CPM/`, or `%LOCALAPPDATA%\CPM` on Windows; override with `CPM_SOURCE_CACHE`). Product is `Mirror Mirror` (manufacturer code `Smkr`, plugin code `Aimp`). Built with `COPY_PLUGIN_AFTER_BUILD TRUE` so each build installs to `~/Library/Audio/Plug-Ins/`. The Makefile parses `PRODUCT_NAME` from CMakeLists so `make pu` follows renames.
- `plugin/server.py` — FastAPI server (default port 7437). Endpoints: `/health`, `/process`, `/train`, `/status`, `/generate`, `/cancel`, `/midi/{job_id}`. One job at a time (guarded by `_job_lock`). Reuses `.venv` for subprocess Python so it picks up the right deps regardless of how it was launched. Includes a watchdog that detects sleep/wake and unresponsive GPUs and restarts training from the last checkpoint.
- `plugin/daw_insert.py`, `daw_setup.py` — DAW-side helpers (Reaper/Ableton MIDI insertion).
- Plugin launches the server itself; the server is also runnable manually:
  ```bash
  source .venv/bin/activate
  python plugin/server.py --root /path/to/mirror-mirror
  ```

### `runs/` directory layout (all git-ignored)

```
runs/{project}/events/        preprocessed event datasets
runs/{project}/checkpoints/   trained model checkpoints
runs/{project}/generated/     generated MIDI
runs/checkpoints/             legacy/global checkpoint path
out_midis/                    MIDIs from the audio→MIDI stage
finetune/runs/                finetune adapters, data, outputs
```

Each plugin project gets its own `runs/{project_name}/` so multiple projects coexist.

## Conventions specific to this repo

- **Console messages reporting unexpected behavior** (ignored requests, skipped items, etc.) begin with `***`.
- **No fallbacks.** Research code — fail fast. Numerical edge cases (NaN/Inf) get logged as WARNING and replaced with the appropriate clamp (typically ±1.0).
- **No backward compatibility.** When changing an API, grep and upgrade all call sites.
- **Type hints** wherever the type is known.
- **New capability ⇒ unit test** in `tests/`.
- **Commits:** stage specific files explicitly. Never `git commit -a` — the repo owner keeps many untracked temp files.

## Submodule note

`vendor/all-in-one-ai-midi-pipeline` and `vendor/mmt` are git **submodules**, not subtrees. The README's "Clone (with submodules)" step is required.
