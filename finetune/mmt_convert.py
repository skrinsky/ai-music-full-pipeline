#!/usr/bin/env python3
"""
Convert MIDI files to MMT (Multitrack Music Transformer) 5-tuple format.

Each MIDI file becomes one .npy file with shape (N_notes, 5) and dtype int32:
    columns: beat, position, pitch, duration, program
    - beat:     integer beat number (0-based)
    - position: 0-11 (12ths of a beat)
    - pitch:    0-127 MIDI pitch
    - duration: quantized to MMT's known durations (12ths of a beat)
    - program:  MIDI program 0-127, or -1 for drums

Also writes train-names.txt and valid-names.txt for mmt_finetune.py.

Usage:
    python finetune/mmt_convert.py \\
        --midi_dir summer_midi \\
        --out_dir  data/summer/notes
"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np

MMT_RESOLUTION = 12   # ticks per beat in MMT's representation
MAX_BEAT = 1024

# From MMT representation.py — durations in 12ths of a beat
KNOWN_DURATIONS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
    15, 16, 18, 20, 21, 24, 30, 36, 40, 42,
    48, 60, 72, 84, 96, 120, 144, 168, 192, 384,
]
_DUR_ARR = np.array(KNOWN_DURATIONS)


def snap_duration(dur: int) -> int:
    if dur <= 0:
        return 1
    idx = int(np.argmin(np.abs(_DUR_ARR - dur)))
    return KNOWN_DURATIONS[idx]


def midi_to_notes(midi_path: Path) -> np.ndarray | None:
    try:
        import muspy
    except ImportError:
        print("ERROR: muspy not installed. Run: pip install muspy", file=sys.stderr)
        sys.exit(1)

    try:
        music = muspy.read_midi(str(midi_path))
    except Exception as e:
        print(f"  muspy failed on {midi_path.name}: {e}", file=sys.stderr)
        return None

    src_res = music.resolution  # ticks per beat in source MIDI
    notes = []
    for track in music.tracks:
        if track.is_drum:
            continue  # MMT has no drum representation
        program = track.program
        for note in track.notes:
            mmt_time = note.time * MMT_RESOLUTION / src_res
            beat     = int(mmt_time) // MMT_RESOLUTION
            position = int(mmt_time) % MMT_RESOLUTION
            dur_raw  = max(1, round(note.duration * MMT_RESOLUTION / src_res))
            duration = snap_duration(dur_raw)
            pitch    = note.pitch

            if beat >= MAX_BEAT or not 0 <= pitch <= 127:
                continue
            notes.append((beat, position, pitch, duration, program))

    if not notes:
        return None
    return np.array(sorted(notes), dtype=np.int32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--midi_dir", required=True)
    ap.add_argument("--out_dir",  required=True)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--seed",     type=int,   default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    midi_files = sorted(Path(args.midi_dir).glob("**/*.mid")) + \
                 sorted(Path(args.midi_dir).glob("**/*.midi"))
    print(f"Found {len(midi_files)} MIDI files in {args.midi_dir}")

    stems = []
    for midi_path in midi_files:
        arr = midi_to_notes(midi_path)
        if arr is None or len(arr) == 0:
            print(f"  skip (empty): {midi_path.name}")
            continue
        stem = midi_path.stem
        np.save(out_dir / f"{stem}.npy", arr)
        print(f"  {midi_path.name} → {len(arr)} notes")
        stems.append(stem)

    if not stems:
        print("No files converted — nothing to write.")
        return

    random.shuffle(stems)
    n_val   = max(1, round(len(stems) * args.val_frac))
    val     = stems[:n_val]
    train   = stems[n_val:]

    (out_dir / "train-names.txt").write_text("\n".join(train))
    (out_dir / "valid-names.txt").write_text("\n".join(val))

    print(f"\nDone.  train={len(train)}  val={len(val)} → {out_dir}")
    print("\nNext: download LMD pretrained checkpoint from")
    print("  https://ucsdcloud-my.sharepoint.com/:f:/g/personal/h3dong_ucsd_edu/EqYq6KHrcltHvgJTmw7Nl6MBtv4szg4RUZUPXc4i_RgEkw")
    print("Then run:")
    print(f"  python finetune/mmt_finetune.py \\")
    print(f"      --data_dir {out_dir} \\")
    print(f"      --ckpt <path/to/lmd/best_model.pt> \\")
    print(f"      --out_dir finetune/runs/mmt_finetuned")


if __name__ == "__main__":
    main()
