#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Autoregressive generation for the dense chorale pipeline.

Strict cyclic voice order: chord → soprano → bass → alto → tenor → repeat.
Position-specific masking:
  - Chord positions: only tokens 50-99 (chord tokens)
  - Voice positions: only tokens 3-49 (pitch tokens) + token 49 (REST)
  - Per-voice range masking (e.g., soprano ∈ MIDI 60-81)

Continuation counters computed on-the-fly from generated tokens.

Imports sample_from_logits, seed_everything, set_gm_programs from training.generate.
"""

import argparse
import json
import os
import sys

import torch

from training.generate import sample_from_logits, seed_everything, set_gm_programs
from training.model_chorale_dense import ChoraleDenseModel
from training.pre_chorale_dense import (
    PAD_ID, BOS_ID, EOS_ID, REST_ID, VOCAB_SIZE,
    PITCH_OFFSET, MIDI_LO, MIDI_HI, NUM_PITCHES,
    CHORD_OFFSET, NUM_CHORDS,
    VOICE_RANGES, VOICE_ORDER_NAMES,
    midi_to_token, token_to_midi,
    is_pitch_token, is_rest_token, is_chord_token,
    decode_tokens_to_midi,
)
from training.train import pick_device

# Voice order per timestep (after BOS): chord, soprano, bass, alto, tenor
# Index within timestep:
#   0 = chord, 1 = soprano, 2 = bass, 3 = alto, 4 = tenor
VOICES_IN_ORDER = ["chord"] + list(VOICE_ORDER_NAMES)  # chord, soprano, bass, alto, tenor

# MIDI range per voice for masking
VOICE_MIDI_RANGES = {
    "soprano": VOICE_RANGES["soprano"],  # (60, 81)
    "bass":    VOICE_RANGES["bass"],     # (36, 64)
    "alto":    VOICE_RANGES["alto"],     # (53, 77)
    "tenor":   VOICE_RANGES["tenor"],    # (45, 72)
}


def build_voice_mask(voice_name: str, vocab_size: int, device: torch.device) -> torch.Tensor:
    """Build a boolean mask (True = allowed) for a given voice position."""
    mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)

    if voice_name == "chord":
        # Only chord tokens allowed
        mask[CHORD_OFFSET:CHORD_OFFSET + NUM_CHORDS] = True
    else:
        # Pitch tokens within voice range + REST
        lo, hi = VOICE_MIDI_RANGES[voice_name]
        for midi_p in range(lo, hi + 1):
            tok = midi_to_token(midi_p)
            if PITCH_OFFSET <= tok < PITCH_OFFSET + NUM_PITCHES:
                mask[tok] = True
        mask[REST_ID] = True

    return mask


def compute_continuation_at_step(generated_tokens: list[int], step_idx: int,
                                  voice_in_step: int) -> int:
    """Compute continuation counter for the current position.

    generated_tokens: all tokens generated so far (including BOS).
    step_idx: which timestep we're generating (0-indexed).
    voice_in_step: 0=chord, 1=soprano, 2=bass, 3=alto, 4=tenor.

    For chord positions, always return 0.
    For voice positions, check if the previous timestep had the same pitch.
    """
    if voice_in_step == 0:
        return 0  # chord

    if step_idx == 0:
        return 0  # first timestep, all onsets

    # Find the token at the same voice position in the previous timestep
    # Previous timestep's voice token: BOS + (step_idx-1)*5 + voice_in_step
    prev_pos = 1 + (step_idx - 1) * 5 + voice_in_step
    curr_pos = 1 + step_idx * 5 + voice_in_step

    if prev_pos >= len(generated_tokens) or curr_pos >= len(generated_tokens):
        return 0

    prev_tok = generated_tokens[prev_pos]
    curr_tok = generated_tokens[curr_pos]

    if prev_tok == curr_tok and is_pitch_token(prev_tok):
        # Same pitch held — increment counter from previous step
        # We need to look up what the previous counter was
        # For simplicity, compute it recursively with a limit
        prev_cont = _get_prev_continuation(generated_tokens, step_idx - 1, voice_in_step)
        return min(31, prev_cont + 1)

    return 0


def _get_prev_continuation(generated_tokens: list[int], step_idx: int,
                            voice_in_step: int) -> int:
    """Recursively compute continuation counter, with depth limit."""
    if step_idx <= 0:
        return 0

    curr_pos = 1 + step_idx * 5 + voice_in_step
    prev_pos = 1 + (step_idx - 1) * 5 + voice_in_step

    if prev_pos >= len(generated_tokens) or curr_pos >= len(generated_tokens):
        return 0

    if generated_tokens[prev_pos] == generated_tokens[curr_pos] and is_pitch_token(generated_tokens[curr_pos]):
        prev = _get_prev_continuation(generated_tokens, step_idx - 1, voice_in_step)
        return min(31, prev + 1)

    return 0


def build_continuation_tensor(generated_tokens: list[int], device: torch.device) -> torch.Tensor:
    """Build the full continuation counter tensor for all generated tokens so far."""
    n = len(generated_tokens)
    conts = [0] * n  # BOS at index 0

    for i in range(1, n):
        # Which timestep and voice position?
        adj = i - 1  # skip BOS
        step_idx = adj // 5
        voice_in_step = adj % 5

        if voice_in_step == 0:
            conts[i] = 0  # chord
            continue

        if step_idx == 0:
            conts[i] = 0  # first timestep
            continue

        prev_pos = 1 + (step_idx - 1) * 5 + voice_in_step
        if prev_pos < n and generated_tokens[prev_pos] == generated_tokens[i] and is_pitch_token(generated_tokens[i]):
            conts[i] = min(31, conts[prev_pos] + 1)
        else:
            conts[i] = 0

    return torch.tensor(conts, dtype=torch.long, device=device)


@torch.no_grad()
def generate_chorale(model: ChoraleDenseModel,
                     n_timesteps: int = 64,
                     temperature: float = 0.9,
                     top_p: float = 0.95,
                     device: torch.device = torch.device("cpu"),
                     ) -> list[int]:
    """Generate a complete dense chorale sequence autoregressively.

    Returns token list: [BOS, chord₀, S₀, B₀, A₀, T₀, ..., EOS]
    """
    model.eval()

    # Precompute voice masks
    voice_masks = {}
    for vname in VOICES_IN_ORDER:
        voice_masks[vname] = build_voice_mask(vname, VOCAB_SIZE, device)

    generated: list[int] = [BOS_ID]

    for step in range(n_timesteps):
        for vi, vname in enumerate(VOICES_IN_ORDER):
            # Build input tensors
            x = torch.tensor([generated], dtype=torch.long, device=device)  # (1, L)
            cont = build_continuation_tensor(generated, device).unsqueeze(0)  # (1, L)

            logits = model(x, cont)  # (1, L, vocab_size)
            next_logits = logits[0, -1, :]  # (vocab_size,)

            # Apply voice mask
            vmask = voice_masks[vname]
            next_logits[~vmask] = float("-inf")

            tok = sample_from_logits(next_logits, temperature=temperature, top_p=top_p)
            generated.append(tok)

    generated.append(EOS_ID)
    return generated


def main():
    ap = argparse.ArgumentParser(description="Generate a dense chorale.")
    ap.add_argument("--ckpt", required=True, help="Model checkpoint path")
    ap.add_argument("--vocab_json", default=None, help="Vocab JSON (default: from checkpoint config)")
    ap.add_argument("--out_midi", default="runs/generated/chorale_dense_out.mid")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--n_timesteps", type=int, default=64,
                    help="Number of 16th-note timesteps to generate")
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--bpm", type=float, default=100.0)
    ap.add_argument("--count", type=int, default=1,
                    help="Number of chorales to generate")
    args = ap.parse_args()

    device = pick_device(args.device)
    print(f"Device: {device}")

    # Load checkpoint
    if not os.path.isfile(args.ckpt):
        print(f"ERROR: checkpoint not found: {args.ckpt}", file=sys.stderr)
        sys.exit(1)

    ckpt = torch.load(args.ckpt, map_location=device)
    model_config = ckpt.get("model_config", ckpt.get("config", {}))

    # Build model from checkpoint config
    model = ChoraleDenseModel(
        vocab_size=model_config.get("vocab_size", model_config.get("VOCAB_SIZE", VOCAB_SIZE)),
        pad_id=model_config.get("pad_id", model_config.get("PAD_ID", PAD_ID)),
        d_model=model_config.get("d_model", model_config.get("D_MODEL", 128)),
        n_heads=model_config.get("n_heads", model_config.get("N_HEADS", 4)),
        n_layers=model_config.get("n_layers", model_config.get("N_LAYERS", 4)),
        ff_mult=model_config.get("ff_mult", model_config.get("FF_MULT", 3)),
        dropout=model_config.get("dropout", model_config.get("DROPOUT", 0.15)),
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded model from {args.ckpt} "
          f"(epoch {ckpt.get('epoch', '?')}, val_loss={ckpt.get('best_val', '?')})")

    seed = seed_everything(args.seed)
    print(f"Seed: {seed}")

    os.makedirs(os.path.dirname(args.out_midi) or ".", exist_ok=True)

    for i in range(args.count):
        print(f"\nGenerating chorale {i+1}/{args.count} "
              f"({args.n_timesteps} timesteps, temp={args.temperature}, top_p={args.top_p})...")

        tokens = generate_chorale(
            model,
            n_timesteps=args.n_timesteps,
            temperature=args.temperature,
            top_p=args.top_p,
            device=device,
        )

        n_body = len(tokens) - 2  # exclude BOS/EOS
        n_steps = n_body // 5
        print(f"  Generated {len(tokens)} tokens ({n_steps} timesteps)")

        # Output path
        if args.count == 1:
            out_path = args.out_midi
        else:
            base, ext = os.path.splitext(args.out_midi)
            out_path = f"{base}_{i:03d}{ext}"

        decode_tokens_to_midi(tokens, out_path, bpm=args.bpm)
        set_gm_programs(out_path)
        print(f"  Saved: {out_path}")

        # Also save token JSON
        json_path = out_path.replace(".mid", "_tokens.json")
        with open(json_path, "w") as f:
            json.dump({
                "tokens": tokens,
                "n_timesteps": n_steps,
                "seed": seed,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "checkpoint": args.ckpt,
            }, f, indent=2)
        print(f"  Tokens: {json_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
