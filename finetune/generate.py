#!/usr/bin/env python3
"""
Generate MIDI from a LoRA-finetuned music transformer.

Two-stage decode mirrors the two-stage encode in convert.py:
  1. Model generates token IDs in the 20k HF vocab space
  2. HF tokenizer decodes IDs → REMI string tokens
  3. MidiTok converts REMI strings → MIDI file

Usage:
    python finetune/generate.py \\
        --base_model NathanFradet/Maestro-REMI-bpe20k \\
        --adapter    finetune/runs/adapter/best \\
        --data_dir   finetune/runs/my_data \\
        --out_midi   finetune/runs/generated/out.mid

    # Seed from one of your own tracks:
    python finetune/generate.py \\
        --base_model NathanFradet/Maestro-REMI-bpe20k \\
        --adapter    finetune/runs/adapter/best \\
        --data_dir   finetune/runs/my_data \\
        --prompt_midi summer_midi/my_song.mid \\
        --prompt_tokens 128 \\
        --out_midi finetune/runs/generated/continuation.mid
"""

import argparse
import json
import time
from pathlib import Path

import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_hf_tokenizer(model_id: str):
    from transformers import PreTrainedTokenizerFast
    print(f"Loading HF tokenizer from {model_id} …")
    return PreTrainedTokenizerFast.from_pretrained(model_id)


def load_miditok(config_path: Path):
    """Load the MidiTok tokenizer saved by convert.py."""
    import miditok
    for name in ("REMI", "REMIPlus", "MMT"):
        cls = getattr(miditok, name, None)
        if cls is not None:
            return cls(params=config_path)
    raise ImportError("No usable MidiTok tokenizer found.")


def build_prompt(hf_tok, miditok_tok, prompt_midi: str | None,
                 prompt_tokens: int, bos_id: int) -> list[int]:
    if not prompt_midi:
        return [bos_id]

    result = miditok_tok(Path(prompt_midi))
    if isinstance(result, list):
        remi_strings = [s for seq in result for s in (seq.tokens or [])]
    else:
        remi_strings = result.tokens or []

    ids = hf_tok.encode(" ".join(remi_strings), add_special_tokens=False)
    ids = ids[:prompt_tokens]
    print(f"Prompt: {len(ids)} tokens from {Path(prompt_midi).name}")
    return ids


def decode_to_midi(hf_tok, miditok_tok, token_ids: list[int], out_path: Path):
    """20k IDs → REMI strings → MIDI file."""
    from miditok import TokSequence

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Stage 1: HF tokenizer → REMI string tokens
    text = hf_tok.decode(token_ids, skip_special_tokens=True)
    remi_strings = [t for t in text.split() if t]
    print(f"Decoded {len(token_ids)} IDs → {len(remi_strings)} REMI tokens")

    if not remi_strings:
        print("No REMI tokens decoded — model may need more training.")
        return

    # Stage 2: MidiTok → MIDI  (try two API variants for version compat)
    last_exc = None
    for attempt, arg in [
        ("TokSequence", [TokSequence(tokens=remi_strings)]),
        ("list-of-strings", [remi_strings]),
    ]:
        try:
            midi_out = miditok_tok.tokens_to_midi(arg)
            midi_out.dump(str(out_path))
            print(f"Saved MIDI → {out_path}")
            return
        except Exception as exc:
            last_exc = (attempt, exc)

    print(f"MIDI decode failed ({last_exc[0]}): {last_exc[1]}")
    fallback = out_path.with_suffix(".remi.txt")
    fallback.write_text("\n".join(remi_strings))
    print(f"REMI tokens saved → {fallback}  (inspect to debug)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Generate MIDI from finetuned model")
    ap.add_argument("--base_model", default="NathanFradet/Maestro-REMI-bpe20k")
    ap.add_argument("--adapter",    required=True, help="LoRA adapter directory")
    ap.add_argument("--data_dir",   required=True,
                    help="Directory produced by convert.py (contains meta.json + tokenizer_config.json)")
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

    # Use hf_model_id from meta if present (set by convert.py), else fall back to --base_model
    hf_model_id = meta.get("hf_model_id", args.base_model)

    hf_tok     = load_hf_tokenizer(hf_model_id)
    miditok_tok = load_miditok(data_dir / "tokenizer_config.json")

    bos_id = hf_tok.bos_token_id or 1
    eos_id = hf_tok.eos_token_id or 2
    pad_id = hf_tok.pad_token_id or 0

    # Load model + LoRA adapter
    from transformers import AutoModelForCausalLM
    from peft import PeftModel
    print(f"Loading base model: {args.base_model}")
    base  = AutoModelForCausalLM.from_pretrained(args.base_model)
    model = PeftModel.from_pretrained(base, args.adapter)
    model = model.to(device)
    model.eval()

    # Prompt
    prompt_ids = build_prompt(hf_tok, miditok_tok,
                               args.prompt_midi, args.prompt_tokens, bos_id)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    # Sampling kwargs
    sample_kwargs: dict = {}
    if args.top_p and args.top_p < 1.0:
        sample_kwargs["top_p"] = args.top_p
    if args.top_k and args.top_k > 0:
        sample_kwargs["top_k"] = args.top_k

    # Generate
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

    decode_to_midi(hf_tok, miditok_tok, out[0].tolist(), Path(args.out_midi))


if __name__ == "__main__":
    main()
