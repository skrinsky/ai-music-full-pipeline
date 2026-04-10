#!/usr/bin/env python3
"""
MIDI → tokenized sequences for finetuning.

Uses MidiTok's MMT (Multitrack Music Transformer) tokenization.  This is the
same format used by the Natooz/Multitrack-Music-Transformer model, so tokens
produced here can be fed directly into that model for finetuning.

MMT packs all tracks into one flat sequence with program-change tokens between
them, which is ideal for multi-instrument MIDI.

Usage:
    # Your personal tracks (most important — this is the style data):
    python finetune/convert.py \\
        --midi_dir summer_midi \\
        --out_dir finetune/runs/my_data

    # Optionally also tokenize a larger corpus for diagnostics / exploration:
    python finetune/convert.py \\
        --midi_dir data/blues_midi \\
        --out_dir finetune/runs/blues_data
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np

SEED = 42


def load_tokenizer(config_path: Path | None):
    from miditok import MMT

    if config_path and config_path.exists():
        tok = MMT(params=config_path)
        print(f"Loaded tokenizer config from {config_path}")
    else:
        tok = MMT()
        print("Using default MMT tokenizer config")
    return tok


def tokenize_dir(midi_dir: Path, tokenizer, seq_len: int, stride: int) -> list[list[int]]:
    midi_files = sorted(midi_dir.glob("**/*.mid")) + sorted(midi_dir.glob("**/*.midi"))
    print(f"  {len(midi_files)} MIDI files in {midi_dir}")

    chunks, errors = [], 0
    for path in midi_files:
        try:
            result = tokenizer(path)
            # MMT returns a single TokSequence (all tracks merged).
            # Older miditok versions may return a list — flatten either way.
            if isinstance(result, list):
                ids = [id_ for seq in result for id_ in seq.ids]
            else:
                ids = result.ids

            # Sliding-window chunking
            for start in range(0, max(1, len(ids) - seq_len + 1), stride):
                window = ids[start : start + seq_len]
                if len(window) == seq_len:
                    chunks.append(window)
        except Exception as exc:
            errors += 1
            print(f"    skip {path.name}: {exc}")

    if errors:
        print(f"  {errors} files skipped")
    print(f"  → {len(chunks)} windows (seq_len={seq_len}, stride={stride})")
    return chunks


def main():
    ap = argparse.ArgumentParser(description="MIDI → token windows for finetuning")
    ap.add_argument("--midi_dir", required=True, help="Folder of MIDI files")
    ap.add_argument("--out_dir", required=True, help="Output directory")
    ap.add_argument(
        "--tokenizer_config",
        default=None,
        help=(
            "Path to a saved MidiTok tokenizer_config.json.  "
            "Pass the one that ships with the base model (download from HuggingFace "
            "alongside the model weights) to guarantee vocab-size alignment."
        ),
    )
    ap.add_argument("--seq_len", type=int, default=1024,
                    help="Token window length (match the base model's max_position_embeddings)")
    ap.add_argument("--stride", type=int, default=512,
                    help="Sliding-window stride (overlap = seq_len - stride)")
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_path = Path(args.tokenizer_config) if args.tokenizer_config else None
    tokenizer = load_tokenizer(cfg_path)

    # Save a copy of the tokenizer config next to the data so generate.py
    # can reload the exact same vocab without needing to re-specify it.
    tokenizer.save_params(out_dir / "tokenizer_config.json")
    vocab_size = len(tokenizer)
    print(f"Vocab size: {vocab_size}")

    chunks = tokenize_dir(Path(args.midi_dir), tokenizer, args.seq_len, args.stride)
    if not chunks:
        print("No valid chunks — check --midi_dir and that files are valid MIDI.")
        return

    random.shuffle(chunks)
    n_val = max(1, int(len(chunks) * args.val_frac))
    val_chunks = chunks[:n_val]
    train_chunks = chunks[n_val:]

    np.save(out_dir / "train_ids.npy", np.array(train_chunks, dtype=np.int32))
    np.save(out_dir / "val_ids.npy",   np.array(val_chunks,   dtype=np.int32))

    meta = {
        "vocab_size": vocab_size,
        "seq_len":    args.seq_len,
        "stride":     args.stride,
        "n_train":    len(train_chunks),
        "n_val":      len(val_chunks),
        "midi_dir":   str(args.midi_dir),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\nDone.  train={len(train_chunks)}  val={len(val_chunks)}")
    print(f"Saved → {out_dir}/")
    print("\nNext:")
    print(f"  python finetune/finetune.py --data_dir {out_dir} --out_dir finetune/runs/adapter")


if __name__ == "__main__":
    main()
