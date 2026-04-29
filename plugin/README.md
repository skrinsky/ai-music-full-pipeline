# Mirror Mirror — DAW Plugin

Mirror Mirror is a VST3/AU plugin for macOS that lets you train an AI on your own audio library and generate new MIDI inside your DAW. It wraps the full `ai-music-full-pipeline` into a GUI — no terminal required after the initial setup.

<!-- screenshot: full plugin window, Generate tab open -->

---

## How it works

```
Your audio files  →  stem separation  →  MIDI  →  preprocess  →  train  →  generate MIDI
     (mp3/wav/flac/…)     (Demucs)                 (tokenise)   (Transformer)  (in your DAW)
```

Everything runs locally. The plugin talks to a small Python server (`plugin/server.py`) that it launches automatically in the background. Generated MIDI lands in your DAW's MIDI track via the plugin's MIDI output, and can also be dragged directly into any track with the **Show MIDI** button.

---

## Requirements

| Requirement | Notes |
|---|---|
| macOS 10.13+ | AU and VST3 formats |
| Python 3.10 | managed by the repo's `uv` venv |
| CMake 3.22+ | `brew install cmake` |
| Xcode Command Line Tools | `xcode-select --install` |
| JUCE 8.0.3 | installed at `~/JUCE` (see below) |

---

## Installation

### 1 — Clone the repo (with submodules)

```bash
git clone --recurse-submodules https://github.com/skrinsky/ai-music-full-pipeline.git
cd ai-music-full-pipeline
```

If you already cloned without `--recurse-submodules`:

```bash
git submodule update --init --recursive
```

### 2 — Set up the Python environment

```bash
make setup                          # creates .venv-ai-music with all pipeline deps
```

This uses `uv` to create a Python 3.10 venv at `.venv-ai-music/`. The server and all pipeline scripts use it automatically.

### 3 — Install JUCE

Download **JUCE 8.0.3** from [juce.com/get-juce](https://juce.com/get-juce) and extract it to `~/JUCE`. The CMakeLists expects it at exactly that path.

```bash
ls ~/JUCE/CMakeLists.txt   # should exist after extraction
```

### 4 — Build and install the plugin

```bash
cd plugin/AIMusicPlugin
cmake -B build
cmake --build build
```

The build step automatically:
- Installs **`Mirror Mirror.component`** (AU) to `~/Library/Audio/Plug-Ins/Components/`
- Installs **`Mirror Mirror.vst3`** (VST3) to `~/Library/Audio/Plug-Ins/VST3/`

After the build, **rescan plugins in your DAW** (in Logic Pro: Menu → Logic Pro → Plug-in Manager → Reset & Rescan Selection). Mirror Mirror will appear under both AU and VST3i instrument categories.

---

## Plugin walkthrough

The plugin window is **480 × 440 px**. The animated mirror face in the bottom-right corner reacts to what's happening — it nods when a job starts, shakes "no" on errors, and celebrates when generation completes.

### Title bar

```
┌─────────────────────────────────────────────┐
│          ✦  Mirror Mirror  ✦      [Save][Load]│
│  [ Process & Train ]  [ Generate ]            │
└─────────────────────────────────────────────┘
```

- **Save / Load** — save or load a `.mmpreset` file (stores all Generate settings + file paths). The DAW also saves settings automatically with your project.

---

### Tab 1 — Process & Train

<!-- screenshot: Process & Train tab -->

```
┌─────────────────────────────────────────────────────┐
│  /path/to/your/audio/folder       [Select Audio Path]│
│                                                      │
│  Instruments to include:                             │
│  ○ Lead Vox  ○ Harm Vox  ○ Guitar                   │
│  ○ Bass      ○ Drums     ○ Other                     │
│                                                      │
│  [  Process Audio  ]  [  Train  ]                    │
└─────────────────────────────────────────────────────┘
```

#### Select Audio Path
Click to choose a folder containing your audio files. The plugin searches **recursively**, so nested subfolders work fine. Supported formats: `.wav`, `.mp3`, `.flac`, `.aiff`, `.aif`, `.m4a`, `.ogg`.

#### Instrument checkboxes
Select which stems to extract and include in training. All six are on by default (uses all stems). Deselecting some focuses the model — for example unticking everything except **Bass** and **Drums** trains on just the rhythm section.

| Toggle | Stem |
|---|---|
| Lead Vox | lead vocals |
| Harm Vox | backing/harmony vocals |
| Guitar | guitar |
| Bass | bass guitar |
| Drums | drums / percussion |
| Other | everything else |

#### Process Audio
Runs the full audio → MIDI → preprocess pipeline on your chosen folder:
1. Demucs separates each file into stems
2. The vendor MIDI pipeline converts each stem to MIDI
3. `training/pre.py` tokenises the MIDIs into training data

This can take a while for large libraries. Watch the status area at the bottom for progress. You only need to do this once per audio collection (or when you add new files).

#### Train
Trains a Transformer model on the preprocessed data. Requires **Process Audio** to have been run first. Training progress (epoch number and validation loss) appears in the status area. The model is saved to `runs/checkpoints/es_model.pt` in the repo.

Training time depends on your hardware and dataset size — expect minutes on MPS/CUDA, hours on CPU for a large collection.

---

### Tab 2 — Generate

<!-- screenshot: Generate tab -->

```
┌──────────────────────────────────────────────────────┐
│  runs/checkpoints/es_model.pt       [Select Model]   │
│                                                       │
│  ( Creativity )  ( Variety )  ( Length )  ( Tempo )  │
│    knob 0.75       knob 0.95   knob 512   knob 120   │
│                                          [○ Sync]     │
│                                                       │
│  ○ Seed from training data                            │
│                                                       │
│  [  Generate  ]         Subdiv ▾  ○ Quantize          │
│                                   ○ Include Triplets  │
└──────────────────────────────────────────────────────┘
```

#### Select Model
Choose a `.pt` checkpoint file. The plugin reads the checkpoint's vocabulary size and uses it to warn you if the Length knob is set above the training context window.

#### Knobs

| Knob | Range | What it does |
|---|---|---|
| **Creativity** | 0.1 – 2.0 | Temperature: lower = more predictable, higher = more surprising. 0.7–0.9 is a good starting range. |
| **Variety** | 0.1 – 1.0 | Nucleus (top-p) sampling: narrows or widens the pool of tokens the model picks from. 0.9–0.95 works well. |
| **Length** | 64 – 2048 | Maximum number of tokens to generate (roughly proportional to bars). Turns orange with a warning if you exceed the model's training context length. |
| **Tempo** | 40 – 240 BPM | Tempo of the generated MIDI. Grayed out when **Sync** is on. |

#### Sync
Locks the Tempo knob to your DAW's current BPM in real time. Turn this on to keep the generated MIDI in sync with your project.

#### Quantize
Snaps generated note timings to a rhythmic grid. When off, the model generates with free timing. When on:
- **Subdiv** sets the grid resolution: `1/4`, `1/8`, `1/16`, or `1/32`
- **Include Triplets** also snaps to the corresponding triplet grid alongside straight subdivisions

#### Seed from training data
Uses a short excerpt from the validation set as a seed prompt before generating. This gives the model a musical starting point and usually produces more coherent output than generating from silence.

#### Generate
Sends all settings to the server and starts generation. The mirror face nods and the status label switches to *generating*. When done, the status shows *done* and the **Show MIDI** button appears.

---

### Shared controls (always visible)

<!-- screenshot: bottom strip with Show MIDI and mirror -->

```
  Status: done
  midi_id=a3f2c1b8                        [ Cancel / Clear ]
                                           [ Show MIDI     ]
                                           [ mirror face   ]
```

#### Status + message
The status label tracks the pipeline stage: `idle → processing → training → generating → done` (or `error` with a description). During training, the message shows the current epoch and validation loss.

#### Cancel / Clear
- While a job is running: **Cancel** kills the process immediately.
- After an error: **Clear** dismisses the error message.
- Both buttons pulse with a gold glow when action is needed.

#### Show MIDI
Appears (with a blue pulse) after a successful generation. Click to reveal the `.mid` file in Finder. You can also **drag** the button directly into any DAW track to import the MIDI.

#### Mirror face
The animated face in the bottom-right gives you a live read of the plugin's mood:
- **Nodding** — a job just started
- **Shaking** — something went wrong
- **Winking + burst** — generation completed successfully
- **O mouth** — error is active; the mouth pulses while the problem persists
- **Eyes** — follow your cursor

---

## Full workflow: audio collection → MIDI in your DAW

1. **Collect audio** — gather `.wav`/`.mp3`/etc. files into one folder (subfolders are fine)
2. **Process & Train tab** → *Select Audio Path* → choose the folder
3. Tick only the instruments you care about (or leave all on)
4. Click **Process Audio** and wait for *Status: done*
5. Click **Train** and wait for *Status: done* (watch the epoch counter)
6. **Generate tab** → *Select Model* → pick `runs/checkpoints/es_model.pt`
7. Dial in Creativity, Variety, Length; set Tempo or enable Sync
8. Click **Generate**
9. When the **Show MIDI** button pulses blue, drag it onto a MIDI track in your DAW

---

## Presets

Click **Save** (top-right of the title bar) to save all Generate settings and file paths to a `.mmpreset` file. Click **Load** to restore them. Presets are plain XML and easy to share.

Your DAW also saves the current settings automatically inside the project file — no manual save needed for normal session use.

---

## The server

The plugin auto-launches `plugin/server.py` in the background the first time you open it. The server:
- Runs on `localhost:7437`
- Stays alive after you close the DAW (jobs survive DAW restarts)
- Re-launches automatically if it goes away (checked every 15 seconds)

To start it manually (useful for debugging):

```bash
source .venv-ai-music/bin/activate
python plugin/server.py --root /path/to/ai-music-full-pipeline --port 7437
```

---

## Troubleshooting

**Plugin doesn't appear in the DAW after building**
Rescan plugins. In Logic Pro: *Logic Pro → Plug-in Manager → Reset & Rescan*. In Ableton: *Preferences → Plug-Ins → Rescan*.

**Status stays "idle" forever / server not reachable**
The plugin couldn't find or launch the server. Open a terminal and start it manually (see above). If it errors, check that `make setup` completed successfully and `.venv-ai-music/` exists.

**"No audio files found" error**
The folder you selected (and all its subfolders) contained no supported audio files. Check the path and file extensions.

**"No training data, run Process Audio first"**
You clicked Train without having run Process Audio on the current folder. The plugin checks for `runs/events/events_train.pkl` and `events_val.pkl` in the repo — if they're missing, run Process Audio first.

**Length knob turns orange**
The Length value exceeds the training context window of the loaded checkpoint. Generation will still work but quality may degrade past that point — the model was never trained on sequences that long.

**Generated MIDI sounds random / incoherent**
- Lower Creativity (temperature) toward 0.6–0.8
- Enable **Seed from training data** for a musical starting point
- Try a lower Length — shorter generations are usually tighter
- Check that the checkpoint actually matches your audio style (a blues model generates blues, etc.)

**Build fails with "JUCE not found"**
Make sure JUCE 8 is extracted at exactly `~/JUCE` and that `~/JUCE/CMakeLists.txt` exists.

**Old "AI Music" plugin still showing in DAW**
Remove the stale component manually and rescan:
```bash
rm -rf ~/Library/Audio/Plug-Ins/Components/"AI Music.component"
rm -rf ~/Library/Audio/Plug-Ins/VST3/"AI Music.vst3"
```
Then rescan plugins in your DAW.
