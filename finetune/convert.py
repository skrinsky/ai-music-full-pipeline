#!/usr/bin/env python3
"""
MIDI → tokenized sequences for finetuning.

Downloads the pre-trained model's MidiTok tokenizer config from HuggingFace
and uses it directly.  This ensures our token IDs are in the same 20k vocab
the model was trained on, so no embedding resize is needed in finetune.py.

Usage:
    python finetune/convert.py \\
        --midi_dir summer_midi \\
        --out_dir  finetune/runs/my_data
"""

import argparse
import json
import random
import shutil
from pathlib import Path

import numpy as np

SEED = 42


def load_pretrained_tokenizer(hf_model: str, out_dir: Path):
    """
    Download tokenizer.json from the HF model repo and load it with MidiTok.
    Saves a copy to out_dir/tokenizer_config.json for generate.py to use.
    """
    from huggingface_hub import hf_hub_download
    import miditok

    print(f"Downloading tokenizer from {hf_model} …")
    cache_path = hf_hub_download(hf_model, "tokenizer.json")
    shutil.copy(cache_path, out_dir / "tokenizer_config.json")

    # Try tokenizer classes in order — the right one is whichever can load the config
    errors = []
    for name in ("REMI", "REMIPlus", "MMT", "MIDILike", "TSD"):
        cls = getattr(miditok, name, None)
        if cls is None:
            continue
        try:
            tok = cls(params=out_dir / "tokenizer_config.json")
            print(f"Loaded tokenizer as {name}  vocab_size={len(tok)}")
            return tok, name
        except Exception as exc:
            errors.append(f"{name}: {exc}")

    raise RuntimeError("Could not load tokenizer with any MidiTok class:\n" +
                       "\n".join(errors))


def tokenize_file(midi_path: Path, tokenizer) -> list[int]:
    """MIDI → flat list of token IDs using the pre-trained tokenizer."""
    result = tokenizer(midi_path)
    if isinstance(result, list):
        return [id_ for seq in result for id_ in (seq.ids or [])]
    return result.ids or []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--midi_dir", required=True)
    ap.add_argument("--out_dir",  required=True)
    ap.add_argument("--hf_model", default="NathanFradet/Maestro-REMI-bpe20k",
                    help="HuggingFace model to pull tokenizer from (must match --base_model)")
    ap.add_argument("--seq_len",  type=int,   default=1024)
    ap.add_argument("--stride",   type=int,   default=512)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--seed",     type=int,   default=SEED)
    args = ap.parse_args()

    random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer, tok_name = load_pretrained_tokenizer(args.hf_model, out_dir)
    vocab_size = len(tokenizer)

    midi_files = sorted(Path(args.midi_dir).glob("**/*.mid")) + \
                 sorted(Path(args.midi_dir).glob("**/*.midi"))
    print(f"Found {len(midi_files)} MIDI files in {args.midi_dir}")

    all_chunks, errors = [], 0
    for midi_path in midi_files:
        try:
            ids = tokenize_file(midi_path, tokenizer)
            if not ids:
                continue
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
        "vocab_size":    vocab_size,
        "seq_len":       args.seq_len,
        "stride":        args.stride,
        "n_train":       len(train_chunks),
        "n_val":         len(val_chunks),
        "midi_dir":      str(args.midi_dir),
        "hf_model_id":   args.hf_model,
        "tok_class":     tok_name,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\nDone.  train={len(train_chunks)}  val={len(val_chunks)}  vocab={vocab_size}")
    print(f"Saved → {out_dir}/")
    print("\nNext:")
    print(f"  python finetune/finetune.py --data_dir {out_dir} --out_dir finetune/runs/adapter")


if __name__ == "__main__":
    main()
