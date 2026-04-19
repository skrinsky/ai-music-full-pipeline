#!/usr/bin/env python3
"""
Patch an existing event_vocab.json with median_tempo_bpm computed from a MIDI folder.
Does NOT regenerate the vocab — safe to use with existing checkpoints.

Usage:
    python scripts/patch_vocab_tempo.py \
        --vocab_json runs/events/event_vocab.json \
        --midi_dir   summer_midi
"""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab_json", required=True)
    ap.add_argument("--midi_dir",   required=True)
    args = ap.parse_args()

    import pretty_midi
    tempos = []
    midi_files = list(Path(args.midi_dir).glob("**/*.mid")) + \
                 list(Path(args.midi_dir).glob("**/*.midi"))

    for p in midi_files:
        try:
            pm = pretty_midi.PrettyMIDI(str(p))
            tc = pm.get_tempo_changes()
            if tc[1].size > 0:
                tempos.append(float(tc[1][0]))
        except Exception as e:
            print(f"  skip {p.name}: {e}")

    if not tempos:
        print("No tempos found.")
        return

    tempos.sort()
    n = len(tempos)
    median = tempos[n // 2] if n % 2 == 1 else (tempos[n // 2 - 1] + tempos[n // 2]) / 2.0
    print(f"Found {n} songs  |  tempo range {min(tempos):.0f}–{max(tempos):.0f} BPM  |  median {median:.1f} BPM")

    with open(args.vocab_json) as f:
        vocab = json.load(f)

    vocab["median_tempo_bpm"] = round(median, 2)

    with open(args.vocab_json, "w") as f:
        json.dump(vocab, f, indent=2)
    print(f"Patched {args.vocab_json}  →  median_tempo_bpm={median:.2f}")


if __name__ == "__main__":
    main()
