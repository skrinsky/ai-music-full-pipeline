#!/usr/bin/env python3
"""
Generate N MIDI candidates and rank by musicality.

Scoring combines:
  - Rule-based heuristics (note count, instrument diversity, pitch entropy, density)
  - Generation confidence (avg log-prob from the model, saved in meta JSON)

Usage:
    python scripts/generate_best.py \
        --n 10 --top_k 3 \
        --ckpt runs/checkpoints/es_model.pt \
        --vocab_json runs/events/event_vocab.json \
        --out_dir runs/generated/best \
        --seed_pkl runs/events/events_train.pkl \
        --force_grid_step 6 --tempo_bpm 75 \
        --temperature 0.75 --top_p 0.95 \
        --device auto
"""
import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path


def score_midi(midi_path: str) -> dict:
    """Rule-based musicality heuristics. Returns dict with 'total' key."""
    try:
        import pretty_midi
        pm = pretty_midi.PrettyMIDI(midi_path)
    except Exception as e:
        return {"total": -999.0, "error": str(e)}

    all_notes = [n for inst in pm.instruments for n in inst.notes]
    if len(all_notes) < 10:
        return {"total": -999.0, "reason": "too_sparse"}

    n_notes = len(all_notes)
    active_insts = [inst for inst in pm.instruments if inst.notes]
    n_active = len(active_insts)

    # Pitch entropy across melodic (non-drum) notes
    melodic_pitches = [n.pitch for inst in active_insts if not inst.is_drum for n in inst.notes]
    if melodic_pitches:
        pc = Counter(melodic_pitches)
        total_p = sum(pc.values())
        pitch_entropy = -sum((c / total_p) * math.log2(c / total_p + 1e-9) for c in pc.values())
    else:
        pitch_entropy = 0.0

    # Note density (notes/sec) — target ~3-8
    start_times = sorted(n.start for n in all_notes)
    end_times = [n.end for n in all_notes]
    duration = max(end_times) - min(start_times) if all_notes else 0.001
    density = n_notes / max(duration, 0.1)
    density_score = max(0.0, 1.0 - abs(math.log(density + 1e-9) - math.log(5)) / 2)

    # Note count — target 40-120
    count_score = min(1.0, n_notes / 60.0) * max(0.0, 1.0 - max(0, n_notes - 120) / 200)

    # Instrument diversity — reward 3+ active instruments
    inst_score = min(1.0, n_active / 3.0)

    # Pitch entropy — reward variety (target > 2 bits)
    entropy_score = min(1.0, pitch_entropy / 4.0)

    # Rhythmic evenness: inter-onset interval coefficient of variation
    if len(start_times) > 4:
        iois = [start_times[i + 1] - start_times[i]
                for i in range(len(start_times) - 1)
                if start_times[i + 1] > start_times[i]]
        if iois:
            mean_ioi = sum(iois) / len(iois)
            std_ioi = (sum((x - mean_ioi) ** 2 for x in iois) / len(iois)) ** 0.5
            cv = std_ioi / (mean_ioi + 1e-9)
            rhythmic_score = max(0.0, 1.0 - abs(cv - 0.5))
        else:
            rhythmic_score = 0.5
    else:
        rhythmic_score = 0.5

    total = (
        inst_score     * 0.25 +
        rhythmic_score * 0.25 +
        count_score    * 0.20 +
        entropy_score  * 0.20 +
        density_score  * 0.10
    )

    return {
        "total": round(total, 4),
        "n_notes": n_notes,
        "n_active_instruments": n_active,
        "pitch_entropy_bits": round(pitch_entropy, 2),
        "note_density_per_sec": round(density, 2),
        "count_score": round(count_score, 3),
        "inst_score": round(inst_score, 3),
        "entropy_score": round(entropy_score, 3),
        "density_score": round(density_score, 3),
        "rhythmic_score": round(rhythmic_score, 3),
    }


def main():
    ap = argparse.ArgumentParser(description="Generate N candidates and keep the most musical.")
    ap.add_argument("--n",            type=int, required=True, help="Minimum candidates to generate before checking score")
    ap.add_argument("--top_k",        type=int, default=1,     help="Number of top-ranked MIDIs to keep")
    ap.add_argument("--min_score",    type=float, default=0.0,
                    help="Keep generating until best combined score reaches this threshold. 0=disabled. Try 0.5.")
    ap.add_argument("--max_attempts", type=int, default=50,
                    help="Hard cap on total candidates when using --min_score (default 50).")
    ap.add_argument("--out_dir",      required=True,            help="Output directory")

    # Pass-through args for generate_v2.py
    ap.add_argument("--ckpt",               required=True)
    ap.add_argument("--vocab_json",         required=True)
    ap.add_argument("--seed_pkl",           default="")
    ap.add_argument("--seed_pkl_tokens",    type=int,   default=128)
    ap.add_argument("--seed_midi",          default="")
    ap.add_argument("--seed_bars",          type=int,   default=0)
    ap.add_argument("--seed_start_bar",     type=int,   default=0)
    ap.add_argument("--max_tokens",         type=int,   default=512)
    ap.add_argument("--ctx",                type=int,   default=512)
    ap.add_argument("--min_notes_before_stop", type=int, default=40)
    ap.add_argument("--temperature",        type=float, default=0.75)
    ap.add_argument("--top_p",              type=float, default=0.95)
    ap.add_argument("--force_grid_step",    type=int,   default=6)
    ap.add_argument("--entropy_ceiling",    type=float, default=0.0)
    ap.add_argument("--tempo_bpm",          type=float, default=0.0)
    ap.add_argument("--knn_index",          default="")
    ap.add_argument("--knn_k",              type=int,   default=16)
    ap.add_argument("--knn_lambda",         type=float, default=0.3)
    ap.add_argument("--tracks",             default="")
    ap.add_argument("--device",             default="auto")
    ap.add_argument("--seed",               type=int,   default=None,
                    help="Base random seed. Each candidate gets seed+i for reproducibility.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    gen_script = Path(__file__).resolve().parent.parent / "training" / "generate_v2.py"

    candidates = []
    max_total = args.max_attempts if args.min_score > 0 else args.n
    i = 0

    while i < max_total:
        attempt = i + 1
        out_midi = os.path.join(args.out_dir, f"candidate_{attempt:03d}.mid")
        out_meta = os.path.join(args.out_dir, f"candidate_{attempt:03d}.json")

        limit_str = f"/{args.max_attempts} max" if args.min_score > 0 else f"/{args.n}"
        print(f"\n[{attempt}{limit_str}] Generating candidate_{attempt:03d}.mid ...")

        cmd = [
            sys.executable, str(gen_script),
            "--ckpt",                   args.ckpt,
            "--vocab_json",             args.vocab_json,
            "--out_midi",               out_midi,
            "--out_meta_json",          out_meta,
            "--max_tokens",             str(args.max_tokens),
            "--ctx",                    str(args.ctx),
            "--min_notes_before_stop",  str(args.min_notes_before_stop),
            "--temperature",            str(args.temperature),
            "--top_p",                  str(args.top_p),
            "--device",                 args.device,
        ]
        if args.force_grid_step > 0:
            cmd += ["--force_grid_step", str(args.force_grid_step)]
        if args.tempo_bpm > 0:
            cmd += ["--tempo_bpm", str(args.tempo_bpm)]
        if args.entropy_ceiling > 0:
            cmd += ["--entropy_ceiling", str(args.entropy_ceiling)]
        if args.seed_pkl:
            cmd += ["--seed_pkl", args.seed_pkl,
                    "--seed_pkl_tokens", str(args.seed_pkl_tokens)]
        elif args.seed_midi:
            cmd += ["--seed_midi", args.seed_midi,
                    "--seed_bars", str(args.seed_bars),
                    "--seed_start_bar", str(args.seed_start_bar)]
        if args.knn_index:
            cmd += ["--knn_index", args.knn_index,
                    "--knn_k",     str(args.knn_k),
                    "--knn_lambda",str(args.knn_lambda)]
        if args.tracks:
            cmd += ["--tracks", args.tracks]
        if args.seed is not None:
            cmd += ["--seed", str(args.seed + i)]

        result = subprocess.run(cmd)

        if result.returncode != 0 or not os.path.isfile(out_midi):
            print(f"  FAILED (exit {result.returncode})")
            i += 1
            continue

        heuristic = score_midi(out_midi)

        # Read avg_log_prob from meta JSON (generation confidence)
        avg_log_prob = 0.0
        if os.path.isfile(out_meta):
            try:
                with open(out_meta) as f:
                    meta = json.load(f)
                avg_log_prob = float(meta.get("avg_log_prob", 0.0))
            except Exception:
                pass

        # Normalize log-prob: typical range ~[-5, -1]; map to [0, 1]
        conf_score = max(0.0, min(1.0, (avg_log_prob + 5.0) / 4.0))
        combined = heuristic["total"] * 0.7 + conf_score * 0.3

        candidates.append({
            "midi_path":  out_midi,
            "meta_path":  out_meta,
            "heuristic":  heuristic,
            "avg_log_prob": round(avg_log_prob, 4),
            "conf_score": round(conf_score, 4),
            "combined":   round(combined, 4),
        })

        best_so_far = max(c["combined"] for c in candidates)
        print(f"  score={combined:.3f}  "
              f"notes={heuristic.get('n_notes', '?')}  "
              f"insts={heuristic.get('n_active_instruments', '?')}  "
              f"pitch_H={heuristic.get('pitch_entropy_bits', 0):.2f}bits  "
              f"conf={avg_log_prob:.3f}  best={best_so_far:.3f}")

        i += 1
        # After every full batch of --n, check if we've hit the threshold
        if args.min_score > 0 and i % args.n == 0 and best_so_far >= args.min_score:
            print(f"\nReached target score {best_so_far:.3f} >= {args.min_score} after {i} candidates.")
            break

    if not candidates:
        print("\nNo successful candidates generated.")
        sys.exit(1)

    # Rank
    candidates.sort(key=lambda c: c["combined"], reverse=True)

    print(f"\n{'=' * 60}")
    print(f"RESULTS — top {min(args.top_k, len(candidates))} of {len(candidates)} candidates:")
    for rank, c in enumerate(candidates[:args.top_k], 1):
        dest = os.path.join(args.out_dir, f"best_{rank:02d}.mid")
        shutil.copy2(c["midi_path"], dest)
        h = c["heuristic"]
        print(f"  #{rank}: {Path(c['midi_path']).name}  "
              f"combined={c['combined']:.3f}  "
              f"notes={h.get('n_notes','?')}  "
              f"insts={h.get('n_active_instruments','?')}  "
              f"→ {Path(dest).name}")

    print(f"\nBest MIDI(s) saved as best_01.mid ... best_{min(args.top_k, len(candidates)):02d}.mid")
    print(f"All candidates kept in {args.out_dir}/")


if __name__ == "__main__":
    main()
