#!/usr/bin/env python3
"""
Generate MIDI from a LoRA-finetuned music transformer.

The model outputs MMT token sequences that MidiTok decodes back to
multi-track MIDI with proper instrument assignments.

Usage:
    python finetune/generate.py \\
        --base_model Natooz/Multitrack-Music-Transformer \\
        --adapter    finetune/runs/adapter/best \\
        --tokenizer_config finetune/runs/my_data/tokenizer_config.json \\
        --out_midi   finetune/runs/generated/out.mid

    # Use a few bars from one of your own tracks as a "style seed":
    python finetune/generate.py \\
        --base_model Natooz/Multitrack-Music-Transformer \\
        --adapter    finetune/runs/adapter/best \\
        --tokenizer_config finetune/runs/my_data/tokenizer_config.json \\
        --prompt_midi summer_midi/my_song.mid \\
        --prompt_tokens 128 \\
        --out_midi finetune/runs/generated/continuation.mid
"""

import argparse
import json
import time
from pathlib import Path

import torch


def get_special_id(tokenizer, names: list[str], fallback: int) -> int:
    """Look up a special token ID by trying several candidate names."""
    for name in names:
        if name in tokenizer.vocab:
            return tokenizer.vocab[name]
    return fallback


def build_prompt(tokenizer, prompt_midi: str | None, prompt_tokens: int, bos_id: int) -> list[int]:
    if not prompt_midi:
        return [bos_id]

    result = tokenizer(Path(prompt_midi))
    ids = result.ids if not isinstance(result, list) else [i for seq in result for i in seq.ids]
    ids = ids[:prompt_tokens]
    print(f"Using {len(ids)} tokens from {Path(prompt_midi).name} as prompt")
    return ids


def decode_to_midi(tokenizer, token_ids: list[int], out_path: Path):
    """
    Convert a flat list of token IDs back to a MIDI file.

    Different miditok versions expect different input formats for tokens_to_midi:
      v1.x / some v2.x: list of list-of-strings  (tokens_to_midi([[str, str, ...]]))
      v2.x TokSequence: list of TokSequence objects
    We try both.
    """
    from miditok import TokSequence

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Build reverse vocab: int ID → token string
    rev_vocab  = {v: k for k, v in tokenizer.vocab.items()}
    token_strs = [rev_vocab[i] for i in token_ids if i in rev_vocab]
    clean_ids  = [i for i in token_ids if i in rev_vocab]

    last_exc = None

    # Try 1: list of token strings (works in many v2 builds)
    try:
        midi_out = tokenizer.tokens_to_midi([token_strs])
        midi_out.dump(str(out_path))
        print(f"Saved MIDI → {out_path}")
        return
    except Exception as exc:
        last_exc = exc

    # Try 2: TokSequence with both ids and tokens populated
    try:
        tok_seq  = TokSequence(ids=clean_ids, tokens=token_strs)
        midi_out = tokenizer.tokens_to_midi([tok_seq])
        midi_out.dump(str(out_path))
        print(f"Saved MIDI → {out_path}")
        return
    except Exception as exc:
        last_exc = exc

    print(f"MIDI decode failed: {last_exc}")
    fallback = out_path.with_suffix(".tokens.json")
    fallback.write_text(json.dumps(token_ids))
    print(f"Raw token IDs saved → {fallback}  (for debugging)")


def main():
    ap = argparse.ArgumentParser(description="Generate MIDI from finetuned model")
    ap.add_argument("--base_model", default="Natooz/Multitrack-Music-Transformer")
    ap.add_argument("--adapter",    required=True, help="Path to saved LoRA adapter directory")
    ap.add_argument("--tokenizer_config", required=True,
                    help="tokenizer_config.json produced by convert.py")
    ap.add_argument("--out_midi",   required=True)
    ap.add_argument("--n_tokens",   type=int,   default=2048,
                    help="Number of new tokens to generate (~2048 ≈ 60–90 seconds of music)")
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top_p",       type=float, default=0.95,
                    help="Nucleus sampling probability (0 to disable)")
    ap.add_argument("--top_k",       type=int,   default=0,
                    help="Top-k filtering (0 to disable)")
    ap.add_argument("--prompt_midi",   default=None,
                    help="Optional: seed the generation from the first N tokens of this MIDI")
    ap.add_argument("--prompt_tokens", type=int, default=64)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = (
        "mps"  if torch.backends.mps.is_available()  else
        "cuda" if torch.cuda.is_available()           else
        "cpu"
    ) if args.device == "auto" else args.device
    print(f"Device: {device}")

    # Tokenizer — use whichever class was available when convert.py ran.
    import miditok as _miditok
    _tok_cls = next(
        (getattr(_miditok, n) for n in ("MMT", "REMIPlus", "REMI") if hasattr(_miditok, n)),
        None,
    )
    if _tok_cls is None:
        raise ImportError("No usable tokenizer found in miditok. Run: make ft-install")
    tokenizer = _tok_cls(params=Path(args.tokenizer_config))
    bos_id = get_special_id(tokenizer, ["BOS_None", "BOS", "<BOS>"], fallback=1)
    eos_id = get_special_id(tokenizer, ["EOS_None", "EOS", "<EOS>"], fallback=2)
    pad_id = get_special_id(tokenizer, ["PAD_None", "PAD", "<PAD>"], fallback=0)
    print(f"Special tokens — BOS={bos_id}  EOS={eos_id}  PAD={pad_id}")

    # Model + adapter
    from transformers import AutoModelForCausalLM
    from peft import PeftModel
    import json as _json
    print(f"Loading base model: {args.base_model}")
    base = AutoModelForCausalLM.from_pretrained(args.base_model)
    # If training resized the embeddings, match that before loading the adapter.
    # The tokenizer_config lives next to train_ids.npy in the data dir.
    _data_dir = Path(args.tokenizer_config).parent
    _meta_path = _data_dir / "meta.json"
    if _meta_path.exists():
        _vocab_size = _json.loads(_meta_path.read_text())["vocab_size"]
        if base.config.vocab_size != _vocab_size:
            print(f"Resizing embeddings: {base.config.vocab_size} → {_vocab_size}")
            base.resize_token_embeddings(_vocab_size)
    model = PeftModel.from_pretrained(base, args.adapter)
    model = model.to(device)
    model.eval()

    # Prompt
    prompt_ids = build_prompt(tokenizer, args.prompt_midi, args.prompt_tokens, bos_id)
    input_ids  = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    # Sampling kwargs — only pass non-None values so HuggingFace doesn't complain
    sample_kwargs: dict = {}
    if args.top_p and args.top_p < 1.0:
        sample_kwargs["top_p"] = args.top_p
    if args.top_k and args.top_k > 0:
        sample_kwargs["top_k"] = args.top_k

    # Generate
    print(f"Generating up to {args.n_tokens} tokens (temperature={args.temperature})…")
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
