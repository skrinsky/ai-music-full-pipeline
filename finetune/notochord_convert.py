#!/usr/bin/env python3
"""
Convert personal MIDI files to Notochord event sequences.

Each event is (instrument, pitch, time_delta, velocity):
  - note_on:  velocity 1-127,  time_delta = seconds since previous event
  - note_off: velocity 0,      time_delta = seconds since previous event
  - instrument = GM program 0-127, or 128 for drums

Output: numpy arrays saved to --out_dir for notochord_finetune.py.

Usage:
    python finetune/notochord_convert.py \\
        --midi_dir summer_midi \\
        --out_dir  finetune/runs/noto_data
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pretty_midi

SEED        = 42
TIME_MAX    = 10.0   # clip time deltas to this (matches model's time_bounds)
SEQ_LEN     = 512    # events per window (matches model's batch_len)
STRIDE      = 256


def midi_to_events(midi_path: Path) -> list[dict]:
    """
    Parse a MIDI file into a flat list of note-on / note-off events
    sorted by time, with time deltas computed between consecutive events.
    """
    pm = pretty_midi.PrettyMIDI(str(midi_path))

    raw = []
    for inst in pm.instruments:
        program = 128 if inst.is_drum else inst.program
        for note in inst.notes:
            raw.append((note.start, program, note.pitch, float(note.velocity)))
            raw.append((note.end,   program, note.pitch, 0.0))

    if not raw:
        return []

    raw.sort(key=lambda x: x[0])

    events = []
    prev_t = raw[0][0]
    for t, inst, pitch, vel in raw:
        dt = min(t - prev_t, TIME_MAX)
        events.append({"instrument": inst, "pitch": pitch,
                        "time": dt, "velocity": vel})
        prev_t = t

    return events


def events_to_arrays(events: list[dict], seq_len: int, stride: int):
    """Sliding-window chunking → list of (insts, pitches, times, vels, ends) arrays."""
    n = len(events)
    chunks = []
    for start in range(0, max(1, n - seq_len + 1), stride):
        sl = events[start : start + seq_len]
        if len(sl) < seq_len:
            break
        insts = np.array([e["instrument"] for e in sl], dtype=np.int32)
        pitches = np.array([e["pitch"]      for e in sl], dtype=np.int32)
        times   = np.array([e["time"]       for e in sl], dtype=np.float32)
        vels    = np.array([e["velocity"]   for e in sl], dtype=np.float32)
        ends    = np.zeros(seq_len, dtype=np.int32)
        ends[-1] = 1   # last event in this window marks sequence end
        chunks.append((insts, pitches, times, vels, ends))
    return chunks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--midi_dir", required=True)
    ap.add_argument("--out_dir",  required=True)
    ap.add_argument("--seq_len",  type=int,   default=SEQ_LEN)
    ap.add_argument("--stride",   type=int,   default=STRIDE)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--seed",     type=int,   default=SEED)
    args = ap.parse_args()

    random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    midi_files = sorted(Path(args.midi_dir).glob("**/*.mid")) + \
                 sorted(Path(args.midi_dir).glob("**/*.midi"))
    print(f"Found {len(midi_files)} MIDI files in {args.midi_dir}")

    all_chunks, errors = [], 0
    seen_instruments: set[int] = set()
    for path in midi_files:
        try:
            events = midi_to_events(path)
            if not events:
                continue
            for e in events:
                seen_instruments.add(e["instrument"])
            chunks = events_to_arrays(events, args.seq_len, args.stride)
            all_chunks.extend(chunks)
        except Exception as exc:
            errors += 1
            print(f"  skip {path.name}: {exc}")

    if errors:
        print(f"{errors} files failed")
    if not all_chunks:
        print("No valid chunks produced.")
        return

    random.shuffle(all_chunks)
    n_val        = max(1, int(len(all_chunks) * args.val_frac))
    val_chunks   = all_chunks[:n_val]
    train_chunks = all_chunks[n_val:]

    def save_split(chunks, name):
        insts   = np.stack([c[0] for c in chunks])
        pitches = np.stack([c[1] for c in chunks])
        times   = np.stack([c[2] for c in chunks])
        vels    = np.stack([c[3] for c in chunks])
        ends    = np.stack([c[4] for c in chunks])
        np.save(out_dir / f"{name}_insts.npy",   insts)
        np.save(out_dir / f"{name}_pitches.npy", pitches)
        np.save(out_dir / f"{name}_times.npy",   times)
        np.save(out_dir / f"{name}_vels.npy",    vels)
        np.save(out_dir / f"{name}_ends.npy",    ends)

    save_split(train_chunks, "train")
    save_split(val_chunks,   "val")

    instruments_sorted = sorted(int(i) for i in seen_instruments)
    print(f"Instruments seen (GM program or 128=drums): {instruments_sorted}")

    meta = {
        "seq_len":     args.seq_len,
        "stride":      args.stride,
        "n_train":     len(train_chunks),
        "n_val":       len(val_chunks),
        "midi_dir":    str(args.midi_dir),
        "instruments": instruments_sorted,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\nDone.  train={len(train_chunks)}  val={len(val_chunks)}  seq_len={args.seq_len}")
    print(f"Saved → {out_dir}/")
    print("\nNext:")
    print(f"  python finetune/notochord_finetune.py \\")
    print(f"      --checkpoint finetune/notochord_lakh_50G_deep.pt \\")
    print(f"      --data_dir   {out_dir} \\")
    print(f"      --out        finetune/runs/noto_finetuned.pt")


if __name__ == "__main__":
    main()
