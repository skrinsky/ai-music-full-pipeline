# DAW MIDI Auto-Insert — Options & Status

## The Problem
After generation, we want the MIDI file to land in the DAW automatically without the user dragging it manually.

## Per-DAW Status

### Reaper — DONE (needs one-time setup)
- Uses `python-reapy` → `InsertMedia(path, 2)` (new tracks at edit cursor)
- One-time: run `dist_api_enable` action inside Reaper
- Auto-setup runs on server start via `daw_setup.py`

### Ableton — DONE (needs one-time setup)
- Uses AbletonOSC control surface + `python-osc`
- Creates a new MIDI track, blank clip, writes notes via OSC
- One-time: install AbletonOSC as a Control Surface in Live prefs

### Logic Pro — UNSOLVED
- No open scripting API (no reapy equivalent)
- AppleScript dictionary too limited for MIDI insertion
- **Mouse automation via CGEventPost**: technically possible but requires
  Accessibility permission, hijacks user's mouse mid-use, fragile pixel
  targeting of drop zone — bad UX
- **Best path**: add MIDI output bus to JUCE plugin → plugin replays
  generated notes as live MIDI output → Logic records them like any
  instrument input. DAW-agnostic, no permissions needed.

### Other DAWs (FL Studio, Cubase, Studio One, etc.)
- Same MIDI output bus approach would work universally
- Virtual MIDI port (python-rtmidi) is an alternative: server plays
  file through a virtual port, user's DAW track records it in real-time.
  Requires DAW to be armed for recording.

## Recommended Next Step
Implement MIDI output on the JUCE plugin side:
- Plugin polls server for generated MIDI data after job completes
- Plugin buffers the note events and emits them on its MIDI output bus
- Works in Logic, Reaper, FL Studio, etc. with no extra setup
- User arms a track to the plugin's MIDI output and hits record
