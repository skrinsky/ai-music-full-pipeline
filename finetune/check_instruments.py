#!/usr/bin/env python3
"""
Quick instrument-distribution checks for Notochord fine-tuning data.

Examples:
  python finetune/check_instruments.py --data_dir finetune/runs/noto_data
  python finetune/check_instruments.py --midi_dir summer_midi
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np


def summarize_data_dir(data_dir: Path) -> None:
    train_path = data_dir / "train_insts.npy"
    val_path = data_dir / "val_insts.npy"
    if not train_path.exists():
        raise FileNotFoundError(f"Missing {train_path}")

    train_insts = np.load(train_path).reshape(-1)
    counter = Counter(int(x) for x in train_insts.tolist())
    total = sum(counter.values())

    print(f"\nConverted train split: {train_path}")
    print(f"Total events: {total}")
    for inst, count in sorted(counter.items(), key=lambda x: x[1], reverse=True):
        pct = 100.0 * count / max(total, 1)
        print(f"  program {inst:3d}: {count:8d} events ({pct:5.1f}%)")

    if val_path.exists():
        val_insts = np.load(val_path).reshape(-1)
        val_unique = sorted(int(x) for x in np.unique(val_insts))
        print(f"Val unique instruments: {val_unique}")


def summarize_midi_dir(midi_dir: Path) -> None:
    try:
        import pretty_midi
    except Exception as exc:
        raise RuntimeError(
            "pretty_midi is required for --midi_dir checks. "
            "Activate .venv-ai-music first."
        ) from exc

    midi_files = sorted(midi_dir.glob("**/*.mid")) + sorted(midi_dir.glob("**/*.midi"))
    if not midi_files:
        raise FileNotFoundError(f"No .mid/.midi files found under {midi_dir}")

    counter = Counter()
    parsed = 0
    failed = 0
    for path in midi_files:
        try:
            pm = pretty_midi.PrettyMIDI(str(path))
        except Exception:
            failed += 1
            continue
        parsed += 1
        for inst in pm.instruments:
            key = 128 if inst.is_drum else int(inst.program)
            counter[key] += len(inst.notes)

    total = sum(counter.values())
    print(f"\nRaw MIDI note counts: {midi_dir}")
    print(f"Files: {len(midi_files)}  parsed: {parsed}  failed: {failed}")
    print(f"Total notes: {total}")
    for inst, count in sorted(counter.items(), key=lambda x: x[1], reverse=True):
        pct = 100.0 * count / max(total, 1)
        print(f"  program {inst:3d}: {count:8d} notes  ({pct:5.1f}%)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=None,
                    help="Directory from finetune/notochord_convert.py")
    ap.add_argument("--midi_dir", default=None,
                    help="Raw MIDI directory to inspect before conversion")
    args = ap.parse_args()

    if not args.data_dir and not args.midi_dir:
        raise SystemExit("Pass --data_dir and/or --midi_dir")

    if args.midi_dir:
        summarize_midi_dir(Path(args.midi_dir))
    if args.data_dir:
        summarize_data_dir(Path(args.data_dir))


if __name__ == "__main__":
    main()
