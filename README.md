# Mirror Mirror

Train an AI on your own audio library and generate new MIDI inside your DAW.

<!-- screenshot: full plugin window, Generate tab open -->

---

## How it works

```
Your audio files  →  stem separation  →  MIDI  →  preprocess  →  train  →  generate MIDI
  (wav/mp3/flac/…)     (Demucs)                    (tokenise)  (Transformer)  (in your DAW)
```

Everything runs **locally** on your machine. The plugin talks to a small Python server that it launches automatically in the background. Generated MIDI lands in your DAW via the plugin's MIDI output, and can be dragged directly into any track.

---

## Requirements

| | |
|---|---|
| macOS 10.13+ | AU and VST3 formats |
| Python 3.10 | managed by the repo's `uv` venv |
| CMake 3.22+ | `brew install cmake` |
| Xcode Command Line Tools | `xcode-select --install` |
| JUCE 8.0.3 | installed at `~/JUCE` (see below) |

---

## Installation

### 1 — Clone (with submodules)

```bash
git clone --recurse-submodules https://github.com/skrinsky/ai-music-full-pipeline.git
cd ai-music-full-pipeline
```

Already cloned without `--recurse-submodules`?

```bash
git submodule update --init --recursive
```

### 2 — Set up the Python environment

```bash
make setup
```

Creates `.venv-ai-music/` with all pipeline dependencies. The server and all scripts use it automatically — you don't need to activate it manually.

### 3 — Install JUCE

Download **JUCE 8.0.3** from [juce.com/get-juce](https://juce.com/get-juce) and extract it to `~/JUCE`.

```bash
ls ~/JUCE/CMakeLists.txt   # should exist
```

### 4 — Build and install the plugin

```bash
cd plugin/AIMusicPlugin
cmake -B build
cmake --build build
```

This automatically installs:
- `Mirror Mirror.component` → `~/Library/Audio/Plug-Ins/Components/`
- `Mirror Mirror.vst3` → `~/Library/Audio/Plug-Ins/VST3/`

Then **rescan plugins in your DAW**. In Logic Pro: *Logic Pro → Plug-in Manager → Reset & Rescan*. Mirror Mirror will appear under AU and VST3i instrument categories.

---

## Plugin walkthrough

The plugin window has two tabs. The animated mirror face in the bottom-right reacts to everything — it nods when a job starts, shakes "no" on errors, and celebrates when generation finishes.

### Title bar

```
┌──────────────────────────────────────────────────┐
│         ✦  Mirror Mirror  ✦         [Save][Load] │
│  [ Process & Train ]  [   Generate   ]           │
└──────────────────────────────────────────────────┘
```

**Save / Load** — save or restore all settings as a `.mmpreset` file. The DAW also saves settings automatically inside your project.

---

### Tab 1 — Process & Train

<!-- screenshot: Process & Train tab -->

```
┌──────────────────────────────────────────────────────┐
│  /path/to/your/audio/folder      [Select Audio Path] │
│                                                      │
│  Instruments to include:                             │
│  ○ Lead Vox  ○ Harm Vox  ○ Guitar                   │
│  ○ Bass      ○ Drums     ○ Other                     │
│                                                      │
│  [  Process Audio  ]   [  Train  ]                   │
└──────────────────────────────────────────────────────┘
```

#### Select Audio Path
Choose a folder of audio files. The plugin searches **recursively** through subfolders. Supported: `.wav` `.mp3` `.flac` `.aiff` `.aif` `.m4a` `.ogg`.

#### Instrument checkboxes
Choose which stems to extract and train on. All six on by default. Deselecting some focuses the model — e.g. only **Bass** + **Drums** trains a rhythm-only model.

| Toggle | Stem |
|---|---|
| Lead Vox | lead vocals |
| Harm Vox | backing / harmony vocals |
| Guitar | guitar |
| Bass | bass guitar |
| Drums | drums / percussion |
| Other | everything else |

#### Process Audio
Runs the full audio → MIDI → preprocess pipeline:
1. Demucs separates each file into stems
2. The MIDI pipeline converts each stem to MIDI
3. `training/pre.py` tokenises the MIDIs into training data

This can take a while for large libraries — watch the status area for progress. You only need to do it once per audio collection (or when you add new files).

#### Train
Trains a Transformer model on the preprocessed data. Requires **Process Audio** to have been run first. Epoch number and validation loss appear in the status area as it runs. The model saves to `runs/checkpoints/es_model.pt` in the repo.

---

### Tab 2 — Generate

<!-- screenshot: Generate tab -->

```
┌──────────────────────────────────────────────────────┐
│  runs/checkpoints/es_model.pt       [Select Model]   │
│                                                       │
│  ( Creativity )  ( Variety )  ( Length )  ( Tempo )  │
│      0.75           0.95        512         120       │
│                                          [○ Sync]     │
│                                                       │
│  ○ Seed from training data                            │
│                                                       │
│  [  Generate  ]          Subdiv ▾  ○ Quantize         │
│                                    ○ Include Triplets │
└──────────────────────────────────────────────────────┘
```

#### Select Model
Choose a `.pt` checkpoint. The plugin reads the model's context window size and warns you (orange label) if Length is set above it.

#### Knobs

| Knob | Range | What it does |
|---|---|---|
| **Creativity** | 0.1 – 2.0 | Temperature — lower = predictable, higher = surprising. Start around 0.75. |
| **Variety** | 0.1 – 1.0 | Nucleus sampling — narrows or widens the token pool. 0.9–0.95 works well. |
| **Length** | 64 – 2048 | Max tokens to generate (roughly proportional to bars). Goes orange if it exceeds the model's training length. |
| **Tempo** | 40 – 240 BPM | Tempo of the output MIDI. Grayed out when **Sync** is on. |

#### Sync
Locks Tempo to your DAW's live BPM.

#### Quantize
Snaps generated note timings to a rhythmic grid. When on:
- **Subdiv** sets the resolution: `1/4`, `1/8`, `1/16`, `1/32`
- **Include Triplets** also snaps to triplet subdivisions

#### Seed from training data
Seeds generation from a short excerpt of the training data rather than from silence — usually produces more coherent output.

#### Generate
Sends everything to the server and starts generation. When done, the **Show MIDI** button appears with a pulsing blue glow.

---

### Shared controls

```
  Status: done
  midi_id=a3f2c1b8                        [ Cancel / Clear ]
                                           [   Show MIDI   ]
                                           [  mirror face  ]
```

**Cancel / Clear** — cancels a running job, or clears an error message. Pulses gold when action is needed.

**Show MIDI** — reveals the generated `.mid` in Finder. You can also **drag** this button directly onto any DAW track to import the MIDI.

**Mirror face** — live feedback:
- Nodding → job started
- Shaking → error
- Winking + particle burst → generation complete
- O-shaped mouth (pulsing) → error active; follows your cursor with its eyes

---

## Full workflow

1. **Collect audio** — drop your files into a folder (subfolders are fine)
2. **Process & Train tab** → *Select Audio Path* → pick the folder
3. Tick the instruments you want to include
4. Click **Process Audio**, wait for *Status: done*
5. Click **Train**, wait for *Status: done* (watch the epoch/loss counter)
6. **Generate tab** → *Select Model* → pick `runs/checkpoints/es_model.pt`
7. Set Creativity, Variety, Length; enable Sync or dial in Tempo
8. Click **Generate**
9. Drag **Show MIDI** onto a MIDI track in your DAW

---

## Troubleshooting

**Plugin doesn't appear after building**
Rescan plugins in your DAW. Logic Pro: *Logic Pro → Plug-in Manager → Reset & Rescan*.

**Status stays "idle" / server not reachable**
The plugin couldn't launch the server. Start it manually:
```bash
source .venv-ai-music/bin/activate
python plugin/server.py --root /path/to/ai-music-full-pipeline
```
If it errors, make sure `make setup` completed successfully.

**"No audio files found"**
The chosen folder contained no supported audio. Check the path and file extensions.

**"No training data, run Process Audio first"**
Click **Process Audio** before **Train**. The plugin checks for `runs/events/events_train.pkl` — if it's missing, preprocessing hasn't run yet.

**Length knob turns orange**
The Length value exceeds the model's training context. Generation still works but quality may drop past that point.

**Old "AI Music" plugin still showing**
```bash
rm -rf ~/Library/Audio/Plug-Ins/Components/"AI Music.component"
rm -rf ~/Library/Audio/Plug-Ins/VST3/"AI Music.vst3"
```
Then rescan.

---

## Running from the terminal (no plugin)

The full pipeline is also available as command-line tools driven by the top-level `Makefile`.

```bash
source .venv-ai-music/bin/activate
make help          # list every available target
```

### Common flows

```bash
# Blues MIDI — no audio needed
make gigamidi-fetch                # ~1000 GigaMIDI blues MIDIs → data/blues_midi/
make blues-preprocess
make blues-train                   # or: make blues-resume
make bg                            # generate

# Bach chorales
curl -L -o data/Jsb16thSeparated.npz \
  https://github.com/omarperacha/TonicNet/raw/master/dataset_unprocessed/Jsb16thSeparated.npz
make chorale-convert
make chorale-preprocess && make chorale-train
make cg                            # generate

# Full audio → MIDI → train pipeline
mkdir -p data/raw && cp /path/to/*.wav data/raw/
scripts/run_end_to_end.sh
make gen                           # generate from latest checkpoint
```

Pass extra flags via `ARGS=...`:
```bash
make blues-train ARGS="--max_d_model 128"
```

Shortcut aliases: `bg` blues-generate · `cg` chorale-generate · `cdg` chorale-dense-generate · `fg` ft-generate · `ng` noto-generate · `gen` generate from latest checkpoint.

### Device selection

Training defaults to `--device auto` (CUDA → MPS → CPU). Override with `--device cuda`, `--device mps`, or `--device cpu`. Note: Notochord finetuning is pinned to CPU — MPS produces NaN loss on that model.

### Output directories (all git-ignored)

| Path | Contents |
|---|---|
| `out_midis/` | MIDIs from the audio→MIDI stage |
| `runs/events/`, `runs/blues_events/`, … | Preprocessed event datasets |
| `runs/checkpoints/` | Trained model checkpoints |
| `runs/generated/` | Generated MIDI outputs |
| `finetune/runs/` | Finetune adapters, data, outputs |

### Tests

```bash
pytest tests/
```

For the full pipeline map, architecture details, and all available `make` targets, see **[CLAUDE.md](CLAUDE.md)**.
