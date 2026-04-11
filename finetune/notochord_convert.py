#!/usr/bin/env python3
"""
Convert personal MIDI files to Notochord event sequences.

Each event is (instrument, pitch, time_delta, velocity):
  - note_on:  velocity 1-127,  time_delta = seconds since previous event
  - note_off: velocity 0,      time_delta = seconds since previous event
  - instrument uses Notochord IDs:
      melodic: 1-128  (GM program + 1)
      drums:   129-256 (GM drum program + 129)

Output: numpy arrays saved to --out_dir for notochord_finetune.py.

Usage:
    python finetune/notochord_convert.py \\
        --midi_dir summer_midi \\
        --out_dir  finetune/runs/noto_data
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pretty_midi

SEED        = 42
TIME_MAX    = 10.0   # clip time deltas to this (matches model's time_bounds)
SEQ_LEN     = 512    # events per window (matches model's batch_len)
STRIDE      = 256


def gm_to_notochord_inst(program: int, is_drum: bool) -> int:
    """Map PrettyMIDI program/is_drum to Notochord instrument id space."""
    program = int(max(0, min(127, program)))
    return (129 + program) if is_drum else (1 + program)


def collision_to_anon_inst(base_inst: int, collision_idx: int) -> int:
    """Map repeated same-program tracks to Notochord anonymous instrument ids."""
    if collision_idx <= 0:
        return base_inst
    if base_inst >= 129:
        anon = 289 + collision_idx
        return min(320, anon)
    anon = 257 + collision_idx
    return min(288, anon)


def midi_to_events(midi_path: Path) -> list[dict]:
    """
    Parse a MIDI file into a flat list of note-on / note-off events
    sorted by time, with time deltas computed between consecutive events.
    """
    pm = pretty_midi.PrettyMIDI(str(midi_path))

    raw = []
    uses_per_base_inst: dict[int, int] = defaultdict(int)
    inst_info: dict[int, tuple[int, bool]] = {}
    for inst in pm.instruments:
        base_inst = gm_to_notochord_inst(inst.program, inst.is_drum)
        collision_idx = uses_per_base_inst[base_inst]
        noto_inst = collision_to_anon_inst(base_inst, collision_idx)
        uses_per_base_inst[base_inst] += 1
        inst_info[noto_inst] = (int(inst.program), bool(inst.is_drum))
        for note in inst.notes:
            raw.append((note.start, noto_inst, note.pitch, float(note.velocity)))
            raw.append((note.end,   noto_inst, note.pitch, 0.0))

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

    return events, inst_info


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
    inst_to_program: dict[int, int] = {}
    inst_is_drum: dict[int, bool] = {}
    for path in midi_files:
        try:
            events, info = midi_to_events(path)
            if not events:
                continue
            for inst_id, (program, is_drum) in info.items():
                inst_to_program[int(inst_id)] = int(program)
                inst_is_drum[int(inst_id)] = bool(is_drum)
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
    print(f"Instruments seen (Notochord ids): {instruments_sorted}")

    meta = {
        "seq_len":     args.seq_len,
        "stride":      args.stride,
        "n_train":     len(train_chunks),
        "n_val":       len(val_chunks),
        "midi_dir":    str(args.midi_dir),
        "instruments": instruments_sorted,
        "inst_to_program": {str(k): int(v) for k, v in sorted(inst_to_program.items())},
        "inst_is_drum": {str(k): bool(v) for k, v in sorted(inst_is_drum.items())},
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
