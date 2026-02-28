#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sequential N-stage cascade generation.

For each stage: build context from previously generated instruments,
inject chords, generate target instrument autoregressively, merge all
into a multi-track MIDI.

Usage:
  python training/generate_cascade.py \\
      --ckpt runs/checkpoints/cascade_model.pt \\
      --vocab_json runs/cascade_events/cascade_vocab.json \\
      --out_midi runs/generated/cascade_out.mid \\
      --device auto
"""

import os
import sys
import json
import math
import time
import random
import hashlib
import argparse
from collections import deque
from datetime import datetime
from typing import List, Dict, Optional, Tuple

import torch
import torch.nn as nn
import pretty_midi

from training.model_cascade import CascadedESModel
from training.generate import (
    sample_from_logits,
    seed_everything,
    snap_delta,
    fit_error,
    set_gm_programs,
    render_midi_to_wav,
    nearest_multiple,
)
from training.pre_cascade import (
    CASCADE_ORDER_A,
    CASCADE_ORDER_B,
    get_cascade_order,
    extract_chord_labels,
    inject_chord_tokens,
    compute_musical_times,
    CHORD_QUALITIES,
)


def get_args():
    ap = argparse.ArgumentParser("Cascade N-stage generator.")
    ap.add_argument("--ckpt", required=True, help="Path to cascade checkpoint .pt")
    ap.add_argument("--vocab_json", required=True, help="Path to cascade_vocab.json")
    ap.add_argument("--out_midi", required=True, help="Output .mid path")
    ap.add_argument("--out_meta_json", default="", help="Optional metadata JSON")

    ap.add_argument("--ablation", default="A")
    ap.add_argument("--instrument_set", default="blues6", help="Instrument preset (blues6, chorale4)")
    ap.add_argument("--max_tokens_per_stage", type=int, default=1500)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--ctx", type=int, default=1024)

    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=None)

    # Grid snapping
    ap.add_argument("--snap_time_shift", action="store_true", default=True)
    ap.add_argument("--grid_straight_steps", type=int, default=6)
    ap.add_argument("--grid_triplet_steps", type=int, default=8)

    # Generation control
    ap.add_argument("--min_notes_per_stage", type=int, default=20)
    ap.add_argument("--max_notes_per_step", type=int, default=12,
                    help="Max notes at one time step before forcing TIME_SHIFT/BAR")
    ap.add_argument("--force_grid_mode", default="",
                    help="Lock grid mode: 'straight' or 'triplet' (empty=auto)")
    ap.add_argument("--rep_penalty", type=float, default=1.20)
    ap.add_argument("--pitch_hist_len", type=int, default=16)

    return ap.parse_args()


def load_model_and_vocab(args):
    """Load checkpoint, build model, load vocab."""
    device = None
    req = (args.device or "auto").lower()
    if req in ("auto", "best"):
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    elif req in ("cuda", "mps", "cpu"):
        device = req
    else:
        raise ValueError("--device must be one of: auto,cuda,mps,cpu")

    with open(args.vocab_json, "r") as f:
        vocab = json.load(f)

    layout = vocab["layout"]
    PAD_ID = layout["PAD"]["start"]
    V = max(spec["start"] + spec["size"] for spec in layout.values())

    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ckpt.get("model_config") or ckpt.get("config")
    if cfg is None:
        raise KeyError("Checkpoint missing model_config/config.")
    fact = ckpt["factored_meta"]

    type_names = fact["type_names"]
    head_sizes = fact["head_sizes"]

    model = CascadedESModel(
        pad_id=PAD_ID,
        type_names=type_names,
        head_sizes=head_sizes,
        num_embeddings=V,
        d_model=cfg["D_MODEL"],
        n_heads=cfg["N_HEADS"],
        n_layers=cfg["N_LAYERS"],
        ff_mult=cfg["FF_MULT"],
        dropout=cfg["DROPOUT"],
    ).to(device).eval()

    state = ckpt.get("model_state") or ckpt.get("model_state_dict")
    if state is None:
        raise KeyError("Checkpoint missing model_state/model_state_dict.")
    model.load_state_dict(state, strict=False)

    return model, vocab, device, type_names, head_sizes, fact


@torch.no_grad()
def generate_one_instrument(
    model: CascadedESModel,
    prompt_tokens: List[int],
    prompt_times: List[float],
    vocab: dict,
    target_inst_idx: int,
    type_names: List[str],
    head_sizes: List[int],
    starts: Dict[str, int],
    args,
    device: str,
) -> List[int]:
    """Autoregressive generation for one instrument stage.

    Returns the generated token sequence (target portion only, without context).
    """
    layout = vocab["layout"]
    PAD_ID = layout["PAD"]["start"]
    BOS_ID = layout["BOS"]["start"]
    EOS_ID = layout["EOS"]["start"]
    SEP_ID = layout["SEP"]["start"]

    TYPE_IDX = {nm: i for i, nm in enumerate(type_names)}
    PITCH_TYPES = [nm for nm in type_names if nm.startswith("PITCH")]

    # Instrument → pitch type
    inst_to_pitch_type = {}
    if "pitch_space_for_inst" in vocab:
        for inst_str, pt in vocab["pitch_space_for_inst"].items():
            inst_to_pitch_type[int(inst_str)] = pt
    else:
        default_pt = PITCH_TYPES[0] if PITCH_TYPES else None
        for i in range(layout.get("INST", {}).get("size", 16)):
            inst_to_pitch_type[i] = default_pt

    DRUM_IDX = None
    for i, pt in inst_to_pitch_type.items():
        if pt and ("DRUM" in pt or "PERC" in pt):
            DRUM_IDX = i
            break

    def has(name):
        return name in TYPE_IDX

    def maybe_idx(name):
        return TYPE_IDX[name] if has(name) else None

    # Grammar phases for target generation
    def allowed_type_indices(phase, notes_this_step):
        if phase == "TIME":
            if notes_this_step >= args.max_notes_per_step:
                # Too many notes at this timestep — force time advance
                opts = [maybe_idx("TIME_SHIFT"), maybe_idx("BAR")]
            else:
                opts = [maybe_idx("TIME_SHIFT"), maybe_idx("BAR"), maybe_idx("INST")]
            return [i for i in opts if i is not None]
        if phase == "POST_TS":
            opts = [maybe_idx("BAR"), maybe_idx("INST")]
            return [i for i in opts if i is not None]
        if phase == "INST":
            return [maybe_idx("INST")] if has("INST") else list(range(len(type_names)))
        if phase == "VEL":
            return [maybe_idx("VEL")] if has("VEL") else list(range(len(type_names)))
        if phase == "PITCH":
            pt = inst_to_pitch_type.get(target_inst_idx, None)
            if pt and pt in TYPE_IDX:
                return [TYPE_IDX[pt]]
            return [TYPE_IDX[p] for p in PITCH_TYPES]
        if phase == "DUR":
            return [maybe_idx("DUR")] if has("DUR") else list(range(len(type_names)))
        return [maybe_idx("TIME_SHIFT")] if has("TIME_SHIFT") else list(range(len(type_names)))

    # Initialize sequence with prompt
    seq = list(prompt_tokens)
    times = list(prompt_times)

    # Track musical time from the prompt
    ts_start = layout["TIME_SHIFT"]["start"]
    ts_size = layout["TIME_SHIFT"]["size"]
    step_qn = float(vocab["time_shift_qn_step"])
    cum_qn = times[-1] if times else 0.0

    phase = "TIME"
    notes_placed = 0
    notes_this_step = 0
    generated_tokens: List[int] = []

    # Grid snapping state
    max_delta_steps = ts_size
    grid_mode = args.force_grid_mode or "straight"
    recent_deltas: deque = deque(maxlen=24)

    # Pitch repetition penalty
    pitch_history: deque = deque(maxlen=args.pitch_hist_len)

    inst_start = layout["INST"]["start"]

    while len(generated_tokens) < args.max_tokens_per_stage:
        # Context window
        ctx_seq = seq[-args.ctx :]
        ctx_times = times[-args.ctx :]

        x = torch.tensor(ctx_seq, dtype=torch.long, device=device).unsqueeze(0)
        t = torch.tensor(ctx_times, dtype=torch.float32, device=device).unsqueeze(0)

        type_logits, value_logits_list = model(x, t)
        tlog = type_logits[:, -1, :].squeeze(0)

        # Grammar-constrained type selection
        allowed = allowed_type_indices(phase, notes_this_step)
        masked = torch.full_like(tlog, -1e9)
        masked[allowed] = tlog[allowed]
        type_choice = sample_from_logits(masked, args.temperature, args.top_p)
        type_name = type_names[type_choice]

        v_logits = value_logits_list[type_choice][:, -1, :].squeeze(0).clone()

        # Force INST to target instrument
        if type_name == "INST":
            v_logits = torch.full_like(v_logits, -1e9)
            if target_inst_idx < v_logits.numel():
                v_logits[target_inst_idx] = 0.0
            val_local = target_inst_idx
        else:
            # Pitch repetition penalty
            if type_name.startswith("PITCH") and target_inst_idx != DRUM_IDX and args.rep_penalty > 1.0:
                for pid in set(pitch_history):
                    if 0 <= pid < v_logits.numel():
                        v_logits[pid] /= float(args.rep_penalty)

            val_local = sample_from_logits(v_logits, args.temperature, args.top_p)

        # Grid snapping for TIME_SHIFT
        if args.snap_time_shift and type_name == "TIME_SHIFT":
            raw_delta = int(val_local) + 1
            recent_deltas.append(raw_delta)
            snapped = snap_delta(
                raw_delta, grid_mode,
                args.grid_straight_steps, args.grid_triplet_steps,
                max_delta_steps,
            )
            val_local = int(snapped - 1)

        global_id = starts[type_name] + int(val_local)
        seq.append(global_id)
        generated_tokens.append(global_id)

        # Update musical time
        if type_name == "TIME_SHIFT":
            delta_qn = (int(val_local) + 1) * step_qn
            cum_qn += delta_qn
        times.append(cum_qn)

        # Grammar transitions
        if type_name == "TIME_SHIFT":
            phase = "POST_TS"
            notes_this_step = 0
        elif type_name == "BAR":
            phase = "INST"
            notes_this_step = 0
        elif type_name == "INST":
            phase = "VEL"
        elif type_name == "VEL":
            phase = "PITCH"
        elif type_name.startswith("PITCH"):
            if target_inst_idx != DRUM_IDX:
                pitch_history.append(int(val_local))
            phase = "DUR"
            notes_placed += 1
            notes_this_step += 1
        elif type_name == "DUR":
            phase = "TIME"
        else:
            phase = "TIME"

        # EOS check
        if global_id == layout["EOS"]["start"]:
            break

        # Stop if we have enough notes
        if notes_placed >= args.min_notes_per_stage and len(generated_tokens) > args.max_tokens_per_stage * 0.5:
            break

    return generated_tokens


def enforce_monophony(midi_path: str) -> None:
    """Clip overlapping durations so each track is strictly monophonic."""
    pm = pretty_midi.PrettyMIDI(midi_path)
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        inst.notes.sort(key=lambda n: (n.start, n.pitch))
        for i in range(len(inst.notes) - 1):
            if inst.notes[i].end > inst.notes[i + 1].start:
                inst.notes[i].end = inst.notes[i + 1].start
    pm.write(midi_path)


@torch.no_grad()
def generate_cascade(args):
    """Orchestrate N-stage cascade generation."""
    seed = seed_everything(args.seed)
    print(f"Seed: {seed}")

    model, vocab, device, type_names, head_sizes, fact = load_model_and_vocab(args)

    layout = vocab["layout"]
    BOS_ID = layout["BOS"]["start"]
    EOS_ID = layout["EOS"]["start"]
    SEP_ID = layout["SEP"]["start"]
    starts = {k: layout[k]["start"] for k in type_names}

    canonical_names = vocab.get("instrument_names", CASCADE_ORDER_A)
    instrument_set = args.instrument_set
    order = get_cascade_order(instrument_set, args.ablation)

    # Collect generated events per instrument (as token sequences)
    all_generated: Dict[str, List[int]] = {}
    all_events_raw: List[Tuple[float, int, int, int, float]] = []

    # Import decode helpers
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    for p in [root, here]:
        if p not in sys.path:
            sys.path.insert(0, p)
    from training.pre import (
        make_instrument_config,
        INSTRUMENT_PRESETS,
        tokenize_song,
        decode_to_midi,
    )

    config = make_instrument_config(
        INSTRUMENT_PRESETS.get(instrument_set, canonical_names)
    )

    t0 = time.time()
    for stage_id, inst_name in enumerate(order):
        if inst_name not in config.names:
            print(f"  Stage {stage_id} ({inst_name}): not in config, skipping")
            continue

        target_idx = config.names.index(inst_name)
        print(f"\n── Stage {stage_id}: generating {inst_name} (idx={target_idx}) ──")

        # Build context from previously generated tokens
        # Merge all previously generated events and tokenize
        if all_events_raw:
            # Re-extract chords from what we've generated so far
            chord_labels = extract_chord_labels(all_events_raw, 120.0, config)

            # Tokenize the context events
            # For generation, we create a simple bar grid
            import numpy as np
            max_time = max(e[0] for e in all_events_raw) if all_events_raw else 4.0
            n_bars = int(max_time / 2.0) + 2
            bar_starts = np.array([i * 2.0 for i in range(n_bars)])
            bars_meta = [(i * 2.0, (i + 1) * 2.0, 16) for i in range(n_bars)]

            ctx_tokens = tokenize_song(all_events_raw, 120.0, bar_starts, bars_meta, vocab)
            ctx_with_chords, ctx_times = inject_chord_tokens(
                ctx_tokens, chord_labels, vocab, 120.0
            )
        else:
            ctx_with_chords = []
            ctx_times = []

        # Build prompt: [BOS] context [SEP]
        prompt = [BOS_ID]
        prompt_times = [0.0]

        # Strip BOS/EOS from context
        for tok, t in zip(ctx_with_chords, ctx_times):
            if tok == BOS_ID or tok == EOS_ID:
                continue
            prompt.append(tok)
            prompt_times.append(t)

        prompt.append(SEP_ID)
        prompt_times.append(prompt_times[-1] if prompt_times else 0.0)

        # Truncate prompt if too long
        max_prompt = args.ctx - args.max_tokens_per_stage // 2
        if len(prompt) > max_prompt:
            excess = len(prompt) - max_prompt
            # Keep BOS at start, remove from beginning of context
            prompt = [prompt[0]] + prompt[1 + excess:]
            prompt_times = [prompt_times[0]] + prompt_times[1 + excess:]

        # Generate
        generated = generate_one_instrument(
            model, prompt, prompt_times,
            vocab, target_idx,
            type_names, head_sizes, starts,
            args, device,
        )

        all_generated[inst_name] = generated
        print(f"  Generated {len(generated)} tokens for {inst_name}")

        # Decode generated tokens to events for use as context in next stage
        # Parse the generated tokens to extract events
        inv_token = {}
        for g, spec in layout.items():
            s, n = spec["start"], spec["size"]
            for j in range(n):
                inv_token[s + j] = (g, j)

        inv_pitch = {}
        for short_name, mp in vocab.get("pitch_maps", {}).items():
            inv = {int(v): int(k) for k, v in mp.items()}
            inv_pitch[short_name] = inv
            inv_pitch["PITCH_" + short_name.upper()] = inv

        pitch_space_for_inst = vocab.get("pitch_space_for_inst", {})
        step_qn = float(vocab.get("time_shift_qn_step", 1.0 / 24.0))

        cur_time_qn = 0.0
        cur_vel = 64
        cur_dur_qn = 0.25

        for tok in generated:
            pair = inv_token.get(tok)
            if pair is None:
                continue
            group, local = pair

            if group == "TIME_SHIFT":
                cur_time_qn += (local + 1) * step_qn
            elif group == "VEL":
                vel_bins = vocab.get("velocity_bins", list(range(0, 128, 16)))
                cur_vel = int(vel_bins[local]) if local < len(vel_bins) else 64
            elif group == "DUR":
                dur_bins = vocab.get("duration_bins_qn", [0.25])
                cur_dur_qn = float(dur_bins[local]) if local < len(dur_bins) else 0.25
            elif group.startswith("PITCH"):
                space = pitch_space_for_inst.get(str(target_idx), "PITCH_GENERAL")
                inv_map = inv_pitch.get(space) or inv_pitch.get("general", {})
                midi = inv_map.get(int(local))
                if midi is not None:
                    start_sec = cur_time_qn * 60.0 / 120.0
                    all_events_raw.append(
                        (start_sec, target_idx, int(midi), cur_vel, cur_dur_qn)
                    )

        notes_this_stage = sum(
            1 for tok in generated
            if inv_token.get(tok, ("", 0))[0].startswith("PITCH")
        )
        print(f"  Notes extracted: {notes_this_stage}")

    dt = time.time() - t0
    print(f"\n── Generation complete ({dt:.1f}s) ──")
    print(f"Total events: {len(all_events_raw)}")

    # Write final multi-track MIDI
    out_dir = os.path.dirname(args.out_midi)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Assemble all events into a single token sequence for decode_to_midi
    all_events_raw.sort(key=lambda x: x[0])

    import numpy as np
    if all_events_raw:
        max_time = max(e[0] for e in all_events_raw)
        n_bars = int(max_time / 2.0) + 2
    else:
        n_bars = 4
    bar_starts = np.array([i * 2.0 for i in range(n_bars)])
    bars_meta = [(i * 2.0, (i + 1) * 2.0, 16) for i in range(n_bars)]

    full_tokens = tokenize_song(all_events_raw, 120.0, bar_starts, bars_meta, vocab)
    decode_to_midi(full_tokens, vocab, args.out_midi, tempo_bpm=120.0)
    enforce_monophony(args.out_midi)

    set_gm_programs(args.out_midi)
    print(f"Wrote MIDI → {args.out_midi}")
    render_midi_to_wav(args.out_midi)

    # Metadata
    meta = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "seed": seed,
        "ablation": args.ablation,
        "stages": list(order),
        "tokens_per_stage": {name: len(toks) for name, toks in all_generated.items()},
        "total_events": len(all_events_raw),
        "ckpt_path": args.ckpt,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "device": device,
        "generation_time_s": dt,
    }
    return meta


def main():
    args = get_args()
    meta = generate_cascade(args)
    if args.out_meta_json:
        with open(args.out_meta_json, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"Wrote {args.out_meta_json}")


if __name__ == "__main__":
    main()
