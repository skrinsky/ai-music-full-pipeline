#!/usr/bin/env python3
"""
Generate N snippets, each seeded from a random position in a random training MIDI.

For each snippet:
  1. Pick a random MIDI from --midi_dir
  2. Pick a random bar offset within that song
  3. Trim to --seed_bars bars starting there (saved as a temp file)
  4. Call generate_v2.py with that temp seed

Usage:
    python scripts/batch_generate.py \
        --n 10 \
        --midi_dir summer_midi \
        --ckpt runs/checkpoints/es_model.pt \
        --vocab_json runs/events/event_vocab.json \
        --out_dir runs/generated/batch \
        --seed_bars 2 \
        --max_tokens 512 --temperature 0.75 --top_p 0.75 \
        --entropy_ceiling 3.5 \
        --knn_index runs/knn/pitch_general --knn_k 16 --knn_lambda 0.3 \
        --device auto
"""
import argparse
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path


def trim_midi(src_path: str, start_sec: float, end_sec: float, dst_path: str):
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(src_path)
    for inst in pm.instruments:
        kept = []
        for n in inst.notes:
            if n.start >= start_sec and n.start < end_sec:
                n.start = max(0.0, n.start - start_sec)
                n.end   = max(n.start + 0.01, n.end - start_sec)
                kept.append(n)
        inst.notes = kept
    pm.write(dst_path)


def estimate_bar_times(midi_path: str, seed_bars: int):
    """Return list of bar start times (seconds) using pretty_midi tempo map."""
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(midi_path)
    total = pm.get_end_time()
    if total < 1.0:
        return []

    # Use the tempo map to compute beat times, group into bars (assume 4/4)
    beat_times = pm.get_beats()
    beats_per_bar = 4
    bar_starts = [beat_times[i] for i in range(0, len(beat_times), beats_per_bar)]

    # Need at least seed_bars+1 bar starts to pick a non-trivial window
    if len(bar_starts) < seed_bars + 1:
        return []
    return bar_starts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n",          type=int,   required=True,  help="Number of snippets to generate")
    ap.add_argument("--midi_dir",   required=True,              help="Directory of seed MIDIs (e.g. summer_midi)")
    ap.add_argument("--out_dir",    required=True,              help="Output directory for generated MIDIs")
    ap.add_argument("--ckpt",       required=True)
    ap.add_argument("--vocab_json", required=True)
    ap.add_argument("--seed_bars",  type=int, default=2,        help="Bars of seed context to feed (default: 2)")
    ap.add_argument("--seed",       type=int, default=None,     help="Global random seed (for reproducibility)")
    # Pass-through args for generate_v2.py
    ap.add_argument("--max_tokens",           type=int,   default=512)
    ap.add_argument("--ctx",                  type=int,   default=512)
    ap.add_argument("--min_notes_before_stop",type=int,   default=40)
    ap.add_argument("--temperature",          type=float, default=0.75)
    ap.add_argument("--top_p",                type=float, default=0.75)
    ap.add_argument("--entropy_ceiling",      type=float, default=3.5)
    ap.add_argument("--knn_index",            default="")
    ap.add_argument("--knn_k",                type=int,   default=16)
    ap.add_argument("--knn_lambda",           type=float, default=0.3)
    ap.add_argument("--device",               default="auto")
    ap.add_argument("--tracks",               default="")
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    midi_files = list(Path(args.midi_dir).glob("**/*.mid")) + \
                 list(Path(args.midi_dir).glob("**/*.midi"))
    if not midi_files:
        print(f"ERROR: no MIDI files found in {args.midi_dir}")
        sys.exit(1)
    print(f"Found {len(midi_files)} MIDIs in {args.midi_dir}")

    os.makedirs(args.out_dir, exist_ok=True)

    gen_script = Path(__file__).resolve().parent.parent / "training" / "generate_v2.py"

    success = 0
    attempts = 0
    while success < args.n and attempts < args.n * 5:
        attempts += 1
        midi_path = str(random.choice(midi_files))
        bar_times = estimate_bar_times(midi_path, args.seed_bars)
        if not bar_times:
            print(f"  skip (too short): {Path(midi_path).name}")
            continue

        # Pick a random bar to start from (leave room for seed_bars)
        max_start_idx = len(bar_times) - args.seed_bars - 1
        if max_start_idx < 0:
            max_start_idx = 0
        start_idx = random.randint(0, max_start_idx)
        start_sec = bar_times[start_idx]
        end_idx   = min(start_idx + args.seed_bars, len(bar_times) - 1)
        end_sec   = bar_times[end_idx]

        tag = f"{Path(midi_path).stem}_bar{start_idx:03d}"
        out_midi = os.path.join(args.out_dir, f"gen_{success+1:03d}_{tag}.mid")

        with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tf:
            tmp_seed = tf.name
        try:
            trim_midi(midi_path, start_sec, end_sec, tmp_seed)
        except Exception as e:
            print(f"  trim failed ({Path(midi_path).name} bar {start_idx}): {e}")
            os.unlink(tmp_seed)
            continue

        cmd = [
            sys.executable, str(gen_script),
            "--ckpt",        args.ckpt,
            "--vocab_json",  args.vocab_json,
            "--out_midi",    out_midi,
            "--seed_midi",   tmp_seed,
            "--seed_bars",   "0",   # already trimmed to the right window
            "--max_tokens",           str(args.max_tokens),
            "--ctx",                  str(args.ctx),
            "--min_notes_before_stop",str(args.min_notes_before_stop),
            "--temperature",          str(args.temperature),
            "--top_p",                str(args.top_p),
            "--entropy_ceiling",      str(args.entropy_ceiling),
            "--device",               args.device,
        ]
        if args.knn_index:
            cmd += ["--knn_index", args.knn_index,
                    "--knn_k",     str(args.knn_k),
                    "--knn_lambda",str(args.knn_lambda)]
        if args.tracks:
            cmd += ["--tracks", args.tracks]

        print(f"\n[{success+1}/{args.n}] {Path(midi_path).name} bar {start_idx} → {Path(out_midi).name}")
        result = subprocess.run(cmd)
        os.unlink(tmp_seed)

        if result.returncode == 0:
            success += 1
        else:
            print(f"  generate_v2.py failed (exit {result.returncode})")

    print(f"\nDone. {success}/{args.n} generated → {args.out_dir}")


if __name__ == "__main__":
    main()
