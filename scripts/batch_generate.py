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
from pathlib import Path



def get_midi_info(midi_path: str) -> tuple[int, float]:
    """Return (n_bars, tempo_bpm) from a MIDI file."""
    try:
        import pretty_midi
        pm = pretty_midi.PrettyMIDI(midi_path)
        beat_times = pm.get_beats()
        n_bars = max(1, len(beat_times) // 4)
        tc = pm.get_tempo_changes()
        if tc[1].size > 0:
            tempo = float(tc[1][0])
        else:
            tempo = float(pm.estimate_tempo())
        return n_bars, max(40.0, min(220.0, tempo))
    except Exception:
        return 8, 0.0   # 0.0 = fall back to vocab median


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
    ap.add_argument("--force_grid_step",      type=int,   default=6,
                    help="Force TIME_SHIFT grid step. 6=1/16 note, 3=1/32, 0=auto-detect.")
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
        n_bars, seed_tempo = get_midi_info(midi_path)
        max_start = max(0, n_bars - args.seed_bars - 1)
        start_bar = random.randint(0, max_start)

        tag = f"{Path(midi_path).stem}_bar{start_bar:03d}"
        out_midi = os.path.join(args.out_dir, f"gen_{success+1:03d}_{tag}.mid")

        cmd = [
            sys.executable, str(gen_script),
            "--ckpt",           args.ckpt,
            "--vocab_json",     args.vocab_json,
            "--out_midi",       out_midi,
            "--seed_midi",      midi_path,
            "--seed_start_bar", str(start_bar),
            "--seed_bars",      str(args.seed_bars),
            "--max_tokens",           str(args.max_tokens),
            "--ctx",                  str(args.ctx),
            "--min_notes_before_stop",str(args.min_notes_before_stop),
            "--temperature",          str(args.temperature),
            "--top_p",                str(args.top_p),
            "--entropy_ceiling",      str(args.entropy_ceiling),
            "--device",               args.device,
        ]
        if seed_tempo > 0:
            cmd += ["--tempo_bpm", str(round(seed_tempo, 1))]
        if args.force_grid_step > 0:
            cmd += ["--force_grid_step", str(args.force_grid_step)]
        if args.knn_index:
            cmd += ["--knn_index", args.knn_index,
                    "--knn_k",     str(args.knn_k),
                    "--knn_lambda",str(args.knn_lambda)]
        if args.tracks:
            cmd += ["--tracks", args.tracks]

        tempo_str = f"{seed_tempo:.1f} BPM" if seed_tempo > 0 else "vocab median"
        print(f"\n[{success+1}/{args.n}] {Path(midi_path).name} bar {start_bar} ({tempo_str}) → {Path(out_midi).name}")
        result = subprocess.run(cmd)

        if result.returncode == 0:
            success += 1
        else:
            print(f"  generate_v2.py failed (exit {result.returncode})")

    print(f"\nDone. {success}/{args.n} generated → {args.out_dir}")


if __name__ == "__main__":
    main()
