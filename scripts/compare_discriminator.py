#!/usr/bin/env python3
"""
Apply discriminator filtering to MIDI files and write filtered versions.

Uses the combined CNN+scalar model when stem WAVs are available under
vendor/all-in-one-ai-midi-pipeline/data/stems/htdemucs_6s/<track_name>/.
Falls back to scalar-only if stems aren't found.

For each input MIDI, produces:
  <stem>_filtered.mid  — only notes that pass the discriminator

Usage:
  python scripts/compare_discriminator.py [--threshold 0.35] \
      out_midis/rawSummer/"01 Hum__01 Hum.mid" ...
"""

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(REPO))

import pretty_midi
from training.pre import extract_multitrack_events, _DEFAULT_CONFIG
from training.note_discriminator import load_discriminator, score_events, score_events_with_audio

DISC_PATH  = REPO / "runs" / "discriminator" / "combined_model.pt"
STEMS_ROOT = REPO / "vendor" / "all-in-one-ai-midi-pipeline" / "data" / "stems" / "htdemucs_6s"


def events_to_midi(events, tempo_bpm, config=None):
    if config is None:
        config = _DEFAULT_CONFIG

    pm = pretty_midi.PrettyMIDI(resolution=960, initial_tempo=float(tempo_bpm))
    tracks = []
    for name in config.names:
        inst = pretty_midi.Instrument(program=0, name=name)
        inst.is_drum = (config.drum_idx is not None and
                        config.names.index(name) == config.drum_idx)
        tracks.append(inst)

    spb = 60.0 / tempo_bpm
    for (start_s, inst_idx, pitch, vel, dur_qn) in events:
        if inst_idx < 0 or inst_idx >= len(tracks):
            continue
        end_s = start_s + dur_qn * spb
        note  = pretty_midi.Note(velocity=int(vel), pitch=int(pitch),
                                 start=float(start_s), end=float(end_s))
        tracks[inst_idx].notes.append(note)

    for t in tracks:
        if t.notes:
            pm.instruments.append(t)
    return pm


def process(midi_path: Path, disc, threshold: float):
    events, tempo, _, _ = extract_multitrack_events(str(midi_path))

    track_name = midi_path.stem.split("__")[0]
    stem_track_dir = STEMS_ROOT / track_name

    if stem_track_dir.exists():
        filtered = score_events_with_audio(
            events, disc, tempo,
            stems_dir=STEMS_ROOT,
            track_name=track_name,
            threshold=threshold,
        )
        mode = "CNN+scalar"
    else:
        filtered = score_events(events, disc, tempo, threshold=threshold)
        mode = "scalar-only"
        print(f"  (no stems at {stem_track_dir} — using scalar-only fallback)")

    removed = len(events) - len(filtered)
    print(f"{midi_path.name} [{mode}]: {len(events)} → {len(filtered)} "
          f"({removed} removed, {100*len(filtered)/max(len(events),1):.0f}% kept)")

    out_path = midi_path.parent / (midi_path.stem + "_filtered.mid")
    pm = events_to_midi(filtered, tempo)
    pm.write(str(out_path))
    print(f"  → {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("midis", nargs="+")
    ap.add_argument("--threshold", type=float, default=0.35)
    args = ap.parse_args()

    if not DISC_PATH.exists():
        sys.exit(f"Discriminator not found at {DISC_PATH}")

    disc = load_discriminator(str(DISC_PATH), device="cpu")
    print(f"Loaded: {DISC_PATH.name}  threshold={args.threshold}  stems_root={'exists' if STEMS_ROOT.exists() else 'MISSING'}\n")

    for p_str in args.midis:
        p = Path(p_str)
        if not p.exists():
            print(f"SKIP (not found): {p}")
            continue
        process(p, disc, args.threshold)


if __name__ == "__main__":
    main()
