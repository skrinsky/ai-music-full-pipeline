#!/usr/bin/env python3
"""
Generate MIDI from a LoRA-finetuned music transformer.

Decode path mirrors convert.py: generated IDs → MidiTok (with BPE decompression) → MIDI.

Usage:
    python finetune/generate.py \\
        --base_model NathanFradet/Maestro-REMI-bpe20k \\
        --adapter    finetune/runs/adapter/best \\
        --data_dir   finetune/runs/my_data \\
        --out_midi   finetune/runs/generated/out.mid

    # Seed from one of your own tracks:
    python finetune/generate.py \\
        ... \\
        --prompt_midi  summer_midi/my_song.mid \\
        --prompt_tokens 128 \\
        --out_midi finetune/runs/generated/continuation.mid
"""

import argparse
import json
import time
from pathlib import Path

import torch


def load_tokenizer(data_dir: Path, tok_class: str):
    """Reload the MidiTok tokenizer saved by convert.py."""
    import miditok
    cls = getattr(miditok, tok_class, None)
    if cls is None:
        # fallback: try in order
        for name in ("REMI", "REMIPlus", "MMT"):
            cls = getattr(miditok, name, None)
            if cls is not None:
                break
    if cls is None:
        raise ImportError("No usable MidiTok tokenizer found.")
    return cls(params=data_dir / "tokenizer_config.json")


def build_prompt(tokenizer, prompt_midi: str | None,
                 prompt_tokens: int, bos_id: int) -> list[int]:
    if not prompt_midi:
        return [bos_id]
    result = tokenizer(Path(prompt_midi))
    ids = result.ids if not isinstance(result, list) \
          else [i for seq in result for i in (seq.ids or [])]
    ids = ids[:prompt_tokens]
    print(f"Prompt: {len(ids)} tokens from {Path(prompt_midi).name}")
    return ids


def decode_to_midi(tokenizer, token_ids: list[int], out_path: Path):
    """Generated IDs → MIDI via MidiTok (handles BPE decompression internally)."""
    from miditok import TokSequence

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # MidiTok v2 with BPE: set ids_bpe_encoded=True so it decompresses correctly
    tok_seq = TokSequence(ids=token_ids, ids_bpe_encoded=True)

    # Decompress BPE → base tokens if the method exists
    if hasattr(tokenizer, "decode_bpe"):
        tokenizer.decode_bpe(tok_seq)

    errors = []
    # Try 1: TokSequence (v2.x standard)
    try:
        midi_out = tokenizer.tokens_to_midi([tok_seq])
        midi_out.dump(str(out_path))
        print(f"Saved MIDI → {out_path}")
        return
    except Exception as exc:
        errors.append(f"TokSequence: {exc}")

    # Try 2: list of token strings (some v2 builds)
    if tok_seq.tokens:
        try:
            midi_out = tokenizer.tokens_to_midi([tok_seq.tokens])
            midi_out.dump(str(out_path))
            print(f"Saved MIDI → {out_path}")
            return
        except Exception as exc:
            errors.append(f"list-of-strings: {exc}")

    # Try 3: rebuild tokens from reverse vocab
    rev_vocab   = {v: k for k, v in tokenizer.vocab.items()}
    token_strs  = [rev_vocab[i] for i in token_ids if i in rev_vocab]
    try:
        midi_out = tokenizer.tokens_to_midi([token_strs])
        midi_out.dump(str(out_path))
        print(f"Saved MIDI → {out_path}")
        return
    except Exception as exc:
        errors.append(f"rev-vocab strings: {exc}")

    import miditok as _mt
    print(f"MIDI decode failed (miditok {_mt.__version__}):")
    for e in errors:
        print(f"  {e}")
    fallback = out_path.with_suffix(".remi.txt")
    fallback.write_text("\n".join(token_strs))
    print(f"Token strings saved → {fallback}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="NathanFradet/Maestro-REMI-bpe20k")
    ap.add_argument("--adapter",    required=True)
    ap.add_argument("--data_dir",   required=True,
                    help="Directory from convert.py (contains meta.json + tokenizer_config.json)")
    ap.add_argument("--out_midi",   required=True)
    ap.add_argument("--n_tokens",   type=int,   default=2048)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top_p",       type=float, default=0.95)
    ap.add_argument("--top_k",       type=int,   default=0)
    ap.add_argument("--prompt_midi",   default=None)
    ap.add_argument("--prompt_tokens", type=int, default=64)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = (
        "mps"  if torch.backends.mps.is_available()  else
        "cuda" if torch.cuda.is_available()           else
        "cpu"
    ) if args.device == "auto" else args.device
    print(f"Device: {device}")

    data_dir = Path(args.data_dir)
    meta     = json.loads((data_dir / "meta.json").read_text())
    tok_class = meta.get("tok_class", "REMI")

    tokenizer = load_tokenizer(data_dir, tok_class)

    bos_id = next((tokenizer.vocab[t] for t in ("BOS_None", "BOS", "<BOS>")
                   if t in tokenizer.vocab), 1)
    eos_id = next((tokenizer.vocab[t] for t in ("EOS_None", "EOS", "<EOS>")
                   if t in tokenizer.vocab), 2)
    pad_id = next((tokenizer.vocab[t] for t in ("PAD_None", "PAD", "<PAD>")
                   if t in tokenizer.vocab), 0)

    from transformers import AutoModelForCausalLM
    from peft import PeftModel
    print(f"Loading base model: {args.base_model}")
    base  = AutoModelForCausalLM.from_pretrained(args.base_model)
    model = PeftModel.from_pretrained(base, args.adapter)
    model = model.to(device)
    model.eval()

    prompt_ids = build_prompt(tokenizer, args.prompt_midi, args.prompt_tokens, bos_id)
    input_ids  = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    sample_kwargs: dict = {}
    if args.top_p and args.top_p < 1.0:
        sample_kwargs["top_p"] = args.top_p
    if args.top_k and args.top_k > 0:
        sample_kwargs["top_k"] = args.top_k

    print(f"Generating {args.n_tokens} tokens …")
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=args.n_tokens,
            do_sample=True,
            temperature=args.temperature,
            eos_token_id=eos_id,
            pad_token_id=pad_id,
            **sample_kwargs,
        )
    print(f"Generated {out.shape[1]} tokens in {time.time() - t0:.1f}s")

    decode_to_midi(tokenizer, out[0].tolist(), Path(args.out_midi))


if __name__ == "__main__":
    main()
