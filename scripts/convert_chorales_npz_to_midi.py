#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert Bach chorale NPZ (TonicNet format) to multi-track MIDI files.

Input format (Jsb16thSeparated.npz):
  Keys: 'train', 'valid', 'test'
  Each value: list of float16 arrays, shape (timesteps, 4)
  Columns: soprano, alto, tenor, bass (MIDI pitches)
  16th-note resolution.  Values < 36 are treated as rests.

Onset detection: A new note starts when the pitch changes from the
previous timestep.  Repeated pitch = sustained note.

Output: one MIDI file per chorale with 4 tracks named
  soprano, alto, tenor, bassvox
"""

import argparse
import os
import sys

import numpy as np
import pretty_midi

# Key detection reused from vendor pipeline
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "vendor",
                                "all-in-one-ai-midi-pipeline"))
from steps.key_normalize import _detect_key_music21, _compute_transpose_semitones

VOICE_NAMES = ["soprano", "alto", "tenor", "bassvox"]
REST_THRESHOLD = 36  # pitches below this are rests


def normalize_pitches(arr: np.ndarray) -> tuple[np.ndarray, str]:
    """Detect key and transpose pitch array to C major / A minor.

    Returns (transposed_array, info_string).
    """
    pitches_flat = arr.flatten()
    valid = pitches_flat[~np.isnan(pitches_flat) & (pitches_flat >= REST_THRESHOLD)]
    pitch_list = [int(round(p)) for p in valid]

    tonic, mode = _detect_key_music21(pitch_list)
    if tonic is None:
        return arr, "key=UNKNOWN shift=0"

    semitones, target, reason = _compute_transpose_semitones(tonic, mode)
    info = f"key={tonic} {mode} shift={semitones:+d}"
    if target:
        info += f" -> {target}"
    if reason:
        info += f" ({reason})"

    if semitones == 0:
        return arr, info

    out = arr.copy().astype(np.float64)
    mask = ~np.isnan(out) & (out >= REST_THRESHOLD)
    out[mask] += semitones
    return out, info


def chorale_to_midi(arr: np.ndarray, bpm: float = 100.0) -> pretty_midi.PrettyMIDI:
    """Convert a (timesteps, 4) pitch array to a PrettyMIDI object."""
    assert arr.ndim == 2 and arr.shape[1] == 4, f"Expected (T, 4), got {arr.shape}"
    T = arr.shape[0]
    pitches = arr.astype(np.float64)

    sec_per_step = 60.0 / bpm / 4.0  # 16th note duration

    pm = pretty_midi.PrettyMIDI(initial_tempo=bpm, resolution=960)

    for voice_idx, name in enumerate(VOICE_NAMES):
        inst = pretty_midi.Instrument(program=0, is_drum=False, name=name)

        col = pitches[:, voice_idx]
        note_start: int | None = None
        note_pitch: int = 0

        for t in range(T):
            raw = col[t]
            is_rest = np.isnan(raw) or raw < REST_THRESHOLD
            p = 0 if is_rest else int(round(raw))

            if note_start is not None:
                # End current note if pitch changed or rest
                if is_rest or p != note_pitch:
                    start_sec = note_start * sec_per_step
                    end_sec = t * sec_per_step
                    if end_sec > start_sec:
                        inst.notes.append(pretty_midi.Note(
                            velocity=80, pitch=note_pitch,
                            start=start_sec, end=end_sec,
                        ))
                    note_start = None

            if not is_rest and note_start is None:
                note_start = t
                note_pitch = p

        # Close any note still open at end
        if note_start is not None:
            start_sec = note_start * sec_per_step
            end_sec = T * sec_per_step
            if end_sec > start_sec:
                inst.notes.append(pretty_midi.Note(
                    velocity=80, pitch=note_pitch,
                    start=start_sec, end=end_sec,
                ))

        pm.instruments.append(inst)

    return pm


def main():
    ap = argparse.ArgumentParser(description="Convert TonicNet NPZ to multi-track MIDI.")
    ap.add_argument("--npz", required=True, help="Path to Jsb16thSeparated.npz")
    ap.add_argument("--out_dir", required=True, help="Output directory for MIDI files")
    ap.add_argument("--bpm", type=float, default=100.0, help="BPM for output MIDIs (default: 100)")
    ap.add_argument("--include_test", action="store_true",
                    help="Also write test set to <out_dir>_test/")
    ap.add_argument("--normalize-key", action="store_true",
                    help="Transpose each chorale to C major / A minor")
    args = ap.parse_args()

    if not os.path.isfile(args.npz):
        print(f"ERROR: NPZ file not found: {args.npz}", file=sys.stderr)
        sys.exit(1)

    data = np.load(args.npz, allow_pickle=True, encoding="latin1")

    os.makedirs(args.out_dir, exist_ok=True)

    # Write train + valid together (pre.py does its own 80/20 split)
    idx = 0
    for split in ("train", "valid"):
        chorales = data[split]
        print(f"{split}: {len(chorales)} chorales")
        for chorale in chorales:
            arr = chorale
            if args.normalize_key:
                arr, info = normalize_pitches(arr)
                print(f"  chorale_{idx:04d}: {info}")
            pm = chorale_to_midi(arr, bpm=args.bpm)
            out_path = os.path.join(args.out_dir, f"chorale_{idx:04d}.mid")
            pm.write(out_path)
            idx += 1

    print(f"Wrote {idx} MIDI files to {args.out_dir}")

    if args.include_test:
        test_dir = args.out_dir.rstrip("/") + "_test"
        os.makedirs(test_dir, exist_ok=True)
        test_chorales = data["test"]
        print(f"test: {len(test_chorales)} chorales")
        for j, chorale in enumerate(test_chorales):
            arr = chorale
            if args.normalize_key:
                arr, info = normalize_pitches(arr)
                print(f"  chorale_test_{j:04d}: {info}")
            pm = chorale_to_midi(arr, bpm=args.bpm)
            out_path = os.path.join(test_dir, f"chorale_test_{j:04d}.mid")
            pm.write(out_path)
        print(f"Wrote {len(test_chorales)} test MIDIs to {test_dir}")


if __name__ == "__main__":
    main()
