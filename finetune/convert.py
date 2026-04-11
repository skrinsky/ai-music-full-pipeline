#!/usr/bin/env python3
"""
MIDI → tokenized sequences for finetuning.

Two-stage pipeline:
  1. MidiTok REMI: MIDI file → REMI string tokens  (e.g. "Bar_0 Note_On_60 …")
  2. HF tokenizer: REMI strings → integer IDs in the pre-trained model's 20k vocab

Using the pre-trained model's own tokenizer means no embedding resize is needed
in finetune.py, so the pre-trained weights transfer cleanly.

Usage:
    python finetune/convert.py \\
        --midi_dir summer_midi \\
        --out_dir  finetune/runs/my_data
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np

SEED = 42


def get_miditok():
    """Return the best available MidiTok tokenizer class."""
    import miditok
    for name in ("REMI", "REMIPlus", "MMT"):
        cls = getattr(miditok, name, None)
        if cls is not None:
            print(f"MidiTok tokenizer: {name}")
            return cls
    raise ImportError("No usable tokenizer in miditok. Run: make ft-install")


def midi_to_remi_strings(midi_path: Path, miditok_tok) -> list[str]:
    """MIDI file → flat list of REMI string tokens (all tracks concatenated)."""
    result = miditok_tok(midi_path)
    if isinstance(result, list):
        return [s for seq in result for s in (seq.tokens or [])]
    return result.tokens or []


def main():
    ap = argparse.ArgumentParser(description="MIDI → 20k-vocab token windows for finetuning")
    ap.add_argument("--midi_dir", required=True)
    ap.add_argument("--out_dir",  required=True)
    ap.add_argument("--hf_model", default="NathanFradet/Maestro-REMI-bpe20k",
                    help="HuggingFace model whose tokenizer to use (must match --base_model in finetune.py)")
    ap.add_argument("--seq_len",  type=int,   default=1024)
    ap.add_argument("--stride",   type=int,   default=512)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--seed",     type=int,   default=SEED)
    args = ap.parse_args()

    random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1 tokenizer: MidiTok (MIDI → REMI strings)
    miditok_cls = get_miditok()
    miditok_tok = miditok_cls()
    miditok_tok.save_params(out_dir / "tokenizer_config.json")

    # Stage 2 tokenizer: HuggingFace (REMI strings → 20k IDs)
    from transformers import PreTrainedTokenizerFast
    print(f"Loading HF tokenizer from {args.hf_model} …")
    hf_tok    = PreTrainedTokenizerFast.from_pretrained(args.hf_model)
    vocab_size = hf_tok.vocab_size
    print(f"HF vocab size: {vocab_size}")

    # Tokenize all MIDI files
    midi_files = sorted(Path(args.midi_dir).glob("**/*.mid")) + \
                 sorted(Path(args.midi_dir).glob("**/*.midi"))
    print(f"Found {len(midi_files)} MIDI files in {args.midi_dir}")

    all_chunks, errors = [], 0
    for midi_path in midi_files:
        try:
            remi_strings = midi_to_remi_strings(midi_path, miditok_tok)
            if not remi_strings:
                continue

            # Encode the whole track as one space-separated string
            ids = hf_tok.encode(" ".join(remi_strings), add_special_tokens=False)

            for start in range(0, max(1, len(ids) - args.seq_len + 1), args.stride):
                window = ids[start : start + args.seq_len]
                if len(window) == args.seq_len:
                    all_chunks.append(window)
        except Exception as exc:
            errors += 1
            print(f"  skip {midi_path.name}: {exc}")

    if errors:
        print(f"{errors} files failed")
    if not all_chunks:
        print("No valid chunks produced.")
        return

    random.shuffle(all_chunks)
    n_val        = max(1, int(len(all_chunks) * args.val_frac))
    val_chunks   = all_chunks[:n_val]
    train_chunks = all_chunks[n_val:]

    np.save(out_dir / "train_ids.npy", np.array(train_chunks, dtype=np.int32))
    np.save(out_dir / "val_ids.npy",   np.array(val_chunks,   dtype=np.int32))

    meta = {
        "vocab_size":   vocab_size,
        "seq_len":      args.seq_len,
        "stride":       args.stride,
        "n_train":      len(train_chunks),
        "n_val":        len(val_chunks),
        "midi_dir":     str(args.midi_dir),
        "hf_model_id":  args.hf_model,   # generate.py reads this to load the HF tokenizer
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\nDone.  train={len(train_chunks)}  val={len(val_chunks)}  vocab={vocab_size}")
    print(f"Saved → {out_dir}/")
    print("\nNext:")
    print(f"  python finetune/finetune.py --data_dir {out_dir} --out_dir finetune/runs/adapter")


if __name__ == "__main__":
    main()
