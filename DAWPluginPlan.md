# DAW Plugin Plan — AI Music Pipeline VST3/AU

## Vision

A JUCE plugin (VST3 + AU component) that wraps the entire ai-music-full-pipeline.
The user points it at a folder of audio, trains a model on their own music,
then hits Generate → listen → Generate again until they like it — all from inside their DAW.

---

## User Flow

1. Open plugin in DAW (Logic, Ableton, Reaper, etc.)
2. Point plugin to a folder of `.wav` / `.aiff` source audio
3. Hit **Process** → pipeline runs: audio → stems (Demucs) → MIDI (Basic Pitch) → preprocess → `events_train.pkl`
4. Hit **Train** → model trains; progress bar shows epoch / val loss
5. Hit **Generate** → model generates a MIDI clip; plugin inserts it as a new MIDI region on a track
6. Listen. Don't like it? Hit **Generate** again (new random seed each time)
7. Like it? Drag/commit the clip and keep working

---

## Architecture

### Why not native C++?
The full pipeline (Demucs, Basic Pitch, PyTorch training, autoregressive generation)
is Python. Porting to C++ would take months and break with every upstream update.

### Chosen approach: thin JUCE UI + local Python server

```
DAW
 └── JUCE Plugin (VST3 / AU)
      │  HTTP (localhost:7437)
      ▼
 Python FastAPI Server   ←── ai-music-full-pipeline (unchanged)
      ├── POST /process     audio → stems → MIDI → preprocess
      ├── POST /train       start training (streams progress via SSE)
      ├── GET  /status      poll training epoch/loss
      ├── POST /generate    run generate_best.py → returns .mid path
      └── GET  /midi/{id}   fetch generated MIDI bytes
```

The plugin ships with a bundled Python environment (or expects the user's venv).
A small launcher script starts the server when the plugin loads.

---

## Phases

### Phase 1 — Python server (no JUCE yet)
- [ ] `plugin/server.py` — FastAPI app with `/process`, `/train`, `/status`, `/generate`, `/midi`
- [ ] Training runs as a background thread; SSE endpoint streams epoch logs
- [ ] Generate endpoint wraps `generate_best.py` with configurable params
- [ ] Returns MIDI as bytes (or a temp file path the plugin can read)
- [ ] Test end-to-end via curl / Postman

### Phase 2 — Minimal JUCE plugin shell
- [ ] JUCE project set up (CMake), targets VST3 + AU
- [ ] UI: folder picker, Process / Train / Generate buttons, status text area, progress bar
- [ ] HTTP client (JUCE `URL` class) wiring buttons to server endpoints
- [ ] Receive MIDI bytes → write temp `.mid` → insert into DAW via JUCE MIDI output

### Phase 3 — DAW MIDI insertion
- [ ] On Generate: plugin creates a MIDI buffer from the returned file
- [ ] Writes it to a virtual MIDI port that the DAW can record from
  - Logic: IAC Driver (macOS)
  - Other DAWs: same IAC or loopMIDI on Windows
- [ ] Or: plugin acts as a MIDI generator and outputs notes directly into the DAW track (preferred if DAW supports it)

### Phase 4 — Polish
- [ ] Bundled server launcher (auto-start on plugin load, auto-kill on unload)
- [ ] Expose key generation params in UI: temperature, top_p, tempo, grid step
- [ ] Show waveform / MIDI preview in plugin window
- [ ] Save/restore session state (folder path, last checkpoint, generation params)
- [ ] Windows support (server side already works; JUCE plugin is cross-platform)

---

## Key Decisions / Open Questions

- **Server startup**: plugin launches `python plugin/server.py` as a subprocess on load.
  Need to handle: server not ready yet (retry), server crash, port conflict.
- **MIDI insertion method**: virtual MIDI port is universal but requires user setup (IAC on Mac).
  Ideal = plugin as MIDISource in DAW, but Logic AU sandboxing may complicate this.
- **Bundling Python**: ship a self-contained venv vs. require user to point plugin at their venv.
  Start with "user points at venv", bundle later.
- **Training on GPU vs CPU**: server just calls existing Makefile targets, so whatever device
  the user has works automatically.

---

## Tech Stack

| Layer | Tech |
|---|---|
| Plugin UI + DAW integration | JUCE 7 (C++17), CMake |
| Backend server | Python 3.10, FastAPI, uvicorn |
| Pipeline | existing `training/` scripts, unchanged |
| HTTP client (plugin side) | JUCE `URL` / `WebInputStream` |
| MIDI output | JUCE `MidiOutput` → IAC Driver / loopMIDI |

---

## First Step

Build and test Phase 1 (the Python server) independently before touching JUCE.
Once the server reliably runs the full pipeline and returns a MIDI file,
the JUCE shell is just a thin UI layer.
