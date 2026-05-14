# Mirror Mirror

Train an AI on your own audio library and generate new MIDI inside your DAW.

![Process & Train tab](docs/image_1.png)

---

## How it works

```
Your audio files  ->  stem separation  ->  MIDI  ->  preprocess  ->  train  ->  generate MIDI
  (wav/mp3/flac/...)    (Demucs)                     (tokenise)  (Transformer)  (in your DAW)
```

Everything runs **locally** on your machine. The plugin talks to a small Python server that it launches automatically in the background. Generated MIDI lands in your DAW via the plugin's MIDI output and can be dragged directly into any track.

---

## Download & Install

### Quick install (no Xcode / Visual Studio required)

Downloads the pre-built plugin from GitHub Releases and sets up the Python environment. Just needs git and Python 3.10.

**macOS / Linux:**
```bash
bash <(curl -fsSL https://raw.githubusercontent.com/skrinsky/mirror-mirror/main/install.sh)
```

Custom install location:
```bash
bash <(curl -fsSL https://raw.githubusercontent.com/skrinsky/mirror-mirror/main/install.sh) --dir ~/my-mirror-mirror
```

**Windows** (PowerShell):
```powershell
irm https://raw.githubusercontent.com/skrinsky/mirror-mirror/main/install.ps1 | iex
```

Custom install location:
```powershell
& ([scriptblock]::Create((irm https://raw.githubusercontent.com/skrinsky/mirror-mirror/main/install.ps1))) -InstallDir "C:\mirror-mirror"
```

### Build from source

Requires Xcode Command Line Tools and cmake. JUCE is fetched automatically by CMake on first build. See [Requirements](#requirements) below.

```bash
bash <(curl -fsSL https://raw.githubusercontent.com/skrinsky/mirror-mirror/main/install-dev.sh)
```

Pre-built VST3/AU zips are also available on the [Releases page](../../releases) if you want to install manually.

---

## Requirements

| | |
|---|---|
| macOS 10.13+ | required for AU format; VST3 also builds on Windows/Linux |
| Python 3.10 | managed by the repo's `uv` venv |
| CMake 3.22+ | `brew install cmake` |
| Xcode Command Line Tools | `xcode-select --install` (macOS only) |
| JUCE 8.0.3 | fetched automatically by CMake (via CPM) into `~/.cache/CPM` on first build |

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

Creates `.venv/` with all pipeline dependencies. The server and all scripts use it automatically.

### 3 — Build and install the plugin

```bash
cd plugin/AIMusicPlugin
cmake -B build
cmake --build build
```

The first build fetches JUCE 8.0.3 via CPM into `~/.cache/CPM/`. Subsequent
projects pinning the same JUCE tag reuse that cached clone. To override the
cache location: `export CPM_SOURCE_CACHE=/your/path` before running cmake.

This automatically installs:
- `Mirror Mirror.component` -> `~/Library/Audio/Plug-Ins/Components/`
- `Mirror Mirror.vst3` -> `~/Library/Audio/Plug-Ins/VST3/`

Then **rescan plugins in your DAW**. In Logic Pro: *Logic Pro -> Plug-in Manager -> Reset & Rescan*. Mirror Mirror will appear under AU and VST3i instrument categories.

---

## Plugin walkthrough

The plugin window has two tabs. The animated mirror face in the bottom-right reacts to everything -- it nods when a job starts, shakes "no" on errors, and celebrates when generation finishes.

### Title bar

**Save / Load** -- save or restore all settings as a `.mmpreset` file. The DAW also saves settings automatically inside your project.

---

### Tab 1 -- Process & Train

![Process & Train tab](docs/image_1.png)

#### Project name

The text field at the top of the tab names this project. All processed data and the trained model are saved under `runs/{project_name}/` so you can have multiple projects side by side without them interfering. Change the name before processing or training to start a new project.

#### Select Audio Path

Choose a folder of audio files. The plugin searches **recursively** through subfolders. Supported formats: `.wav` `.mp3` `.flac` `.aiff` `.aif` `.m4a` `.ogg`.

If you have already processed some files in a previous session, a dialog will ask which files to skip -- saving time by reusing existing stems for unchanged tracks.

#### Instrument checkboxes

![Instrument checkboxes](docs/image_6.png)

Choose which stems to extract and train on. All six are on by default. Deselecting some focuses the model -- e.g. only **Bass** + **Drums** trains a rhythm-only model.

| Toggle | Stem |
|---|---|
| Lead Vox | lead vocals |
| Harm Vox | backing / harmony vocals |
| Guitar | guitar |
| Bass | bass guitar |
| Drums | drums / percussion |
| Other | everything else |

#### Process Audio

Runs the full audio -> MIDI -> preprocess pipeline:
1. Demucs separates each file into stems
2. The MIDI pipeline converts each stem to MIDI
3. `training/pre.py` tokenises the MIDIs into training data

This can take a while for large libraries -- watch the status area for progress. You only need to do it once per audio collection (or when you add new files).

#### Train

Trains a Transformer model on the preprocessed data. Requires **Process Audio** to have been run first. Epoch number and validation loss appear in the status area as training progresses.

**If a saved model already exists for this project**, a dialog appears:

![Resume Training dialog](docs/image_4.png)

- **Continue** -- picks up from the best saved epoch (lowest validation loss)
- **Start Fresh** -- discards the existing model and trains from scratch
- **Cancel** -- does nothing

The model is saved whenever validation loss improves, so "Continue" always resumes from the best checkpoint, not necessarily the most recent epoch.

#### Advanced Settings (key icon)

![Advanced Settings panel](docs/image_3.png)

Click the key icon to open the Advanced Settings panel.

**Note Filter**

An AI discriminator that scores every note in your training data and removes low-quality or atypical ones before training. The **Intensity** slider controls how aggressively it filters:
- Low -- gentle cleanup, keeps most notes
- High -- strict, keeps only the most representative notes

Leave this off for a first run. Enable it once you have a trained model and want to refine the data quality.

**Seq Length**

Number of tokens per training window (default 512). Longer sequences capture more musical context but require more memory and train more slowly. 512 works well for most datasets; try 1024 for longer musical phrases.

**Fine-tune from checkpoint**

![Fine-tune naming](docs/image_5.png)

Enables fine-tuning an existing model on new material rather than training from scratch.

- Check **From checkpoint** and the project name automatically gains a `_fine_tune` suffix (e.g. `my_model` becomes `my_model_fine_tune`) so it saves separately from the base model
- The **Base model** field fills in automatically with the latest checkpoint -- browse to override
- The Seq Length slider locks to match the base checkpoint's training length (required for compatibility)

Uncheck to return to normal training; the suffix is removed from the project name.

---

### Tab 2 -- Generate

![Generate tab](docs/image_2.png)

#### Select Model

Choose a `.pt` checkpoint. The plugin reads the model's context window size and warns you (orange label) if Length is set above it.

#### Knobs

| Knob | Range | What it does |
|---|---|---|
| **Creativity** | 0.1 - 2.0 | Temperature -- lower = predictable, higher = surprising. Start around 0.75. |
| **Variety** | 0.1 - 1.0 | Nucleus sampling -- narrows or widens the token pool. 0.9-0.95 works well. |
| **Length** | 64 - 2048 | Max tokens to generate (roughly proportional to bars). Goes orange if it exceeds the model's training length. |
| **Tempo** | 40 - 240 BPM | Tempo of the output MIDI. Grayed out when **Sync** is on. |

#### Sync

Locks Tempo to your DAW's live BPM.

#### Quantize

Snaps generated note timings to a rhythmic grid. When on:
- **Subdiv** sets the resolution: `1/4`, `1/8`, `1/16`, `1/32`
- **Include Triplets** also snaps to triplet subdivisions

#### Seed from training data

Seeds generation from a short excerpt of the training data rather than from silence -- usually produces more coherent output.

#### Generate

Sends everything to the server and starts generation. When done, the **Show MIDI** button appears with a pulsing blue glow.

![Show MIDI button after generation](docs/image_7.png)

---

### Shared controls

**Cancel / Clear** -- cancels a running job, or clears an error message. Pulses gold when action is needed.

**Show MIDI** -- reveals the generated `.mid` in Finder. You can also **drag** this button directly onto any DAW track to import the MIDI.

**Mirror face** -- live feedback:
- Nodding -> job started
- Shaking -> error
- Winking + particle burst -> generation complete
- Mouth says NO -> error active; follows your cursor with its eyes

---

## Full workflow

1. **Collect audio** -- drop your files into a folder (subfolders are fine)
2. **Process & Train tab** -> enter a **Project name**
3. Click **Select Audio Path** -> pick the folder
4. Tick the instruments you want to include
5. Click **Process Audio**, wait for *Status: done*
6. Click **Train** -- choose **Continue** or **Start Fresh** if prompted
7. Watch the epoch / val loss counter in the status area; training stops automatically when it stops improving
8. **Generate tab** -> **Select Model** -> pick the checkpoint from your project folder
9. Set Creativity, Variety, Length; enable Sync or dial in Tempo
10. Click **Generate**
11. Drag **Show MIDI** onto a MIDI track in your DAW

---

## Tips

**Training time**
Training time scales with dataset size and model depth. On Apple Silicon, expect a few minutes per epoch for typical project sizes. Training saves automatically whenever it finds a new best result, so you can cancel at any time and resume later.

**Closing your laptop during training**
The plugin detects when the system wakes from sleep and resets the training watchdog timer, so training continues cleanly after you reopen your computer. If the GPU becomes unresponsive, the watchdog automatically restarts training from the last saved checkpoint.

**Multiple projects**
Give each audio collection its own project name. Each project keeps its processed data and model completely separate under `runs/{project_name}/`.

**Fine-tuning**
Fine-tuning works best when your new material is stylistically related to the base model's training data. If results sound off, try starting fresh with just the new material instead.

---

## Troubleshooting

**Plugin doesn't appear after building**
Rescan plugins in your DAW. Logic Pro: *Logic Pro -> Plug-in Manager -> Reset & Rescan*.

**Status stays "idle" / server not reachable**
The plugin couldn't launch the server. Start it manually:
```bash
source .venv/bin/activate
python plugin/server.py --root /path/to/ai-music-full-pipeline
```
If it errors, make sure `make setup` completed successfully.

**"No audio files found"**
The chosen folder contained no supported audio. Check the path and file extensions.

**"No training data, run Process Audio first"**
Click **Process Audio** before **Train**. The plugin checks for `events_train.pkl` in the project's events folder -- if it's missing, preprocessing hasn't run yet.

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
source .venv/bin/activate
make help          # list every available target
```

### Common flows

```bash
# Blues MIDI -- no audio needed
make gigamidi-fetch                # ~1000 GigaMIDI blues MIDIs -> data/blues_midi/
make blues-preprocess
make blues-train
make bg                            # generate

# Bach chorales
curl -L -o data/Jsb16thSeparated.npz \
  https://github.com/omarperacha/TonicNet/raw/master/dataset_unprocessed/Jsb16thSeparated.npz
make chorale-convert
make chorale-preprocess && make chorale-train
make cg                            # generate

# Full audio -> MIDI -> train pipeline
mkdir -p data/raw && cp /path/to/*.wav data/raw/
scripts/run_end_to_end.sh
make gen                           # generate from latest checkpoint
```

Pass extra flags via `ARGS=...`:
```bash
make blues-train ARGS="--max_d_model 128"
```

Shortcut aliases: `bg` blues-generate · `cg` chorale-generate · `cdg` chorale-dense-generate · `fg` ft-generate · `gen` generate from latest checkpoint.

### Device selection

Training defaults to `--device auto` (CUDA -> MPS -> CPU). Override with `--device cuda`, `--device mps`, or `--device cpu`.

### Output directories (all git-ignored)

| Path | Contents |
|---|---|
| `runs/{project}/events/` | Preprocessed event datasets |
| `runs/{project}/checkpoints/` | Trained model checkpoints |
| `runs/{project}/generated/` | Generated MIDI outputs |
| `runs/checkpoints/` | Legacy / global checkpoint path |
| `out_midis/` | MIDIs from the audio->MIDI stage |
| `finetune/runs/` | Finetune adapters, data, outputs |

### Tests

```bash
pytest tests/
```

For the full pipeline map, architecture details, and all available `make` targets, see **[CLAUDE.md](CLAUDE.md)**.
