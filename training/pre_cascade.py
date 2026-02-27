#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cascade preprocessing: per-instrument splitting, chord extraction,
cascade example construction with [BOS] context [SEP] target [EOS] format.

Ablation A (6 stages): drums → bass → guitar → other → voxlead → voxharm
Ablation B (5 stages): drums → bass → CHORD_BED(guitar+other merged) → voxlead → voxharm

Usage:
  python training/pre_cascade.py --midi_folder data/blues_midi \\
      --data_folder runs/cascade_events --ablation A

Outputs:
  cascade_train.pkl   # {"sequences", "musical_times", "sep_positions", "stage_ids"}
  cascade_val.pkl
  cascade_vocab.json
"""

import os
import glob
import json
import random
import pickle
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np

from training.pre import (
    InstrumentConfig,
    make_instrument_config,
    INSTRUMENT_PRESETS,
    extract_multitrack_events,
    tokenize_song,
    build_pitch_maps,
    gather_bar_pairs,
    build_event_vocab,
    compact_vocab,
    augment_events_additive,
    token_time_qn_prefix,
    is_track_bluesy,
    qn_between,
    decode_to_midi,
    TIME_SHIFT_QN_STEP,
)

# ── Cascade ordering ──────────────────────────────────────────

CASCADE_ORDER_A = ["drums", "bass", "guitar", "other", "voxlead", "voxharm"]
CASCADE_ORDER_B = ["drums", "bass", "guitar", "voxlead", "voxharm"]
# In ablation B, "other" events are relabeled to "guitar" index

SEQ_LEN = 1024
SEQ_STRIDE = 512

# ── Chord label types ─────────────────────────────────────────

CHORD_QUALITIES = ["maj", "min", "dom7", "dim", "aug"]
CHORD_ROOT_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Interval templates (semitones above root) for quality detection
# Checked in order — first match wins
_QUALITY_TEMPLATES = [
    ("dom7", {4, 10}),      # major 3rd + minor 7th
    ("dim",  {3, 6}),       # minor 3rd + diminished 5th
    ("aug",  {4, 8}),       # major 3rd + augmented 5th
    ("min",  {3}),          # minor 3rd (without dim 5th)
    ("maj",  {4}),          # major 3rd (default)
]


# ── Vocab extension ───────────────────────────────────────────

def extend_vocab_for_cascade(vocab: dict) -> dict:
    """Add SEP, CHORD_ROOT, CHORD_QUAL token types to an existing vocab.

    Mutates and returns the vocab dict.
    """
    layout = vocab["layout"]
    # Find current max token index
    idx = max(spec["start"] + spec["size"] for spec in layout.values())

    layout["SEP"] = {"start": idx, "size": 1}
    idx += 1
    layout["CHORD_ROOT"] = {"start": idx, "size": 12}
    idx += 12
    layout["CHORD_QUAL"] = {"start": idx, "size": len(CHORD_QUALITIES)}
    idx += len(CHORD_QUALITIES)

    # Store quality names for decoding
    vocab["chord_quality_names"] = CHORD_QUALITIES
    vocab["chord_root_names"] = CHORD_ROOT_NAMES

    return vocab


# ── Instrument splitting ──────────────────────────────────────

def split_events_by_instrument(
    ev: List[Tuple[float, int, int, int, float]],
    config: InstrumentConfig,
) -> Dict[int, List[Tuple[float, int, int, int, float]]]:
    """Partition events by instrument index.

    Returns {inst_idx: [(start_sec, inst_idx, midi, vel, dur_qn), ...]}.
    """
    by_inst: Dict[int, List[Tuple[float, int, int, int, float]]] = {}
    for e in ev:
        inst = e[1]
        by_inst.setdefault(inst, []).append(e)
    return by_inst


def merge_streams_chronological(
    streams: List[List[Tuple[float, int, int, int, float]]],
) -> List[Tuple[float, int, int, int, float]]:
    """Merge multiple event streams back into one, sorted by onset time (stable)."""
    merged = []
    for s in streams:
        merged.extend(s)
    merged.sort(key=lambda x: x[0])
    return merged


# ── Chord extraction ──────────────────────────────────────────

def extract_chord_labels(
    ev: List[Tuple[float, int, int, int, float]],
    tempo_bpm: float,
    config: InstrumentConfig,
) -> List[Tuple[float, int, int]]:
    """Heuristic chord labeling at each beat boundary.

    Returns list of (time_qn, root_pc, quality_idx) sorted by time_qn.
    quality_idx indexes into CHORD_QUALITIES.
    """
    # Collect all melodic notes as (start_qn, end_qn, pitch, is_bass)
    bass_idx = config.bass_idx
    intervals = []
    for (start_s, inst, midi, vel, dur_qn) in ev:
        if config.drum_idx is not None and inst == config.drum_idx:
            continue
        start_qn = qn_between(0.0, start_s, tempo_bpm)
        end_qn = start_qn + dur_qn
        is_bass = (inst == bass_idx) if bass_idx is not None else False
        intervals.append((start_qn, end_qn, midi, is_bass))

    if not intervals:
        return []

    # Find the total duration in QN
    max_qn = max(end for (_, end, _, _) in intervals)

    chords: List[Tuple[float, int, int]] = []
    beat_qn = 0.0
    while beat_qn < max_qn:
        # Collect pitch classes sounding at this beat
        pcs = set()
        bass_pcs = []
        for (s, e, midi, is_bass) in intervals:
            if s <= beat_qn < e or (s == beat_qn):
                pc = midi % 12
                pcs.add(pc)
                if is_bass:
                    bass_pcs.append((midi, pc))

        if not pcs:
            beat_qn += 1.0
            continue

        # Root = pitch class of lowest bass note, or lowest pitch overall
        if bass_pcs:
            root_pc = min(bass_pcs, key=lambda x: x[0])[1]
        else:
            # Find lowest sounding pitch
            lowest_midi = 128
            lowest_pc = 0
            for (s, e, midi, _) in intervals:
                if (s <= beat_qn < e or s == beat_qn) and midi < lowest_midi:
                    lowest_midi = midi
                    lowest_pc = midi % 12
            root_pc = lowest_pc

        # Determine quality by interval template matching
        intervals_above_root = {(pc - root_pc) % 12 for pc in pcs} - {0}
        quality_idx = 0  # default: maj
        for qname, template in _QUALITY_TEMPLATES:
            if template.issubset(intervals_above_root):
                quality_idx = CHORD_QUALITIES.index(qname)
                break

        chords.append((beat_qn, root_pc, quality_idx))
        beat_qn += 1.0

    return chords


# ── Chord token injection ─────────────────────────────────────

def inject_chord_tokens(
    tokens: List[int],
    chord_labels: List[Tuple[float, int, int]],
    vocab: dict,
    tempo_bpm: float,
) -> Tuple[List[int], List[float]]:
    """Insert CHORD_ROOT + CHORD_QUAL tokens at beat boundaries.

    Returns (new_tokens, musical_times) where musical_times[i] is the
    cumulative QN time at token i.
    """
    layout = vocab["layout"]
    ts_start = layout["TIME_SHIFT"]["start"]
    ts_size = layout["TIME_SHIFT"]["size"]
    step_qn = float(vocab["time_shift_qn_step"])

    chord_root_start = layout["CHORD_ROOT"]["start"]
    chord_qual_start = layout["CHORD_QUAL"]["start"]

    # Build a dict: beat_qn -> (root_pc, quality_idx)
    chord_at_beat: Dict[int, Tuple[int, int]] = {}
    for (t_qn, root, qual) in chord_labels:
        beat_key = int(round(t_qn))
        chord_at_beat[beat_key] = (root, qual)

    new_tokens: List[int] = []
    musical_times: List[float] = []
    cum_qn = 0.0
    last_injected_beat = -1

    for tok in tokens:
        # Record current musical time
        new_tokens.append(tok)
        musical_times.append(cum_qn)

        # If this is a TIME_SHIFT token, advance cumulative time
        if ts_start <= tok < ts_start + ts_size:
            local = tok - ts_start
            cum_qn += (local + 1) * step_qn

        # Check if we've crossed a beat boundary — inject chord tokens
        current_beat = int(cum_qn)
        if current_beat > last_injected_beat and current_beat in chord_at_beat:
            root, qual = chord_at_beat[current_beat]
            # Inject CHORD_ROOT
            new_tokens.append(chord_root_start + root)
            musical_times.append(cum_qn)
            # Inject CHORD_QUAL
            new_tokens.append(chord_qual_start + qual)
            musical_times.append(cum_qn)
            last_injected_beat = current_beat

    return new_tokens, musical_times


# ── Musical time computation ──────────────────────────────────

def compute_musical_times(tokens: List[int], vocab: dict) -> List[float]:
    """Compute cumulative QN time for each token position."""
    layout = vocab["layout"]
    ts_start = layout["TIME_SHIFT"]["start"]
    ts_size = layout["TIME_SHIFT"]["size"]
    step_qn = float(vocab["time_shift_qn_step"])

    times: List[float] = []
    cum_qn = 0.0
    for tok in tokens:
        times.append(cum_qn)
        if ts_start <= tok < ts_start + ts_size:
            local = tok - ts_start
            cum_qn += (local + 1) * step_qn
    return times


# ── Cascade example construction ──────────────────────────────

def build_cascade_example(
    context_tokens: List[int],
    context_times: List[float],
    target_tokens: List[int],
    target_times: List[float],
    vocab: dict,
    stage_id: int,
) -> Tuple[List[int], List[float], int]:
    """Build one [BOS] context [SEP] target [EOS] sequence.

    Returns (tokens, musical_times, sep_position).
    """
    bos = vocab["layout"]["BOS"]["start"]
    eos = vocab["layout"]["EOS"]["start"]
    sep = vocab["layout"]["SEP"]["start"]

    # Strip BOS/EOS from context and target if present
    def strip_special(toks, times):
        out_t, out_m = [], []
        for t, m in zip(toks, times):
            if t == bos or t == eos:
                continue
            out_t.append(t)
            out_m.append(m)
        return out_t, out_m

    ctx_t, ctx_m = strip_special(context_tokens, context_times)
    tgt_t, tgt_m = strip_special(target_tokens, target_times)

    # Assemble: [BOS] ctx [SEP] tgt [EOS]
    tokens = [bos] + ctx_t + [sep] + tgt_t + [eos]
    times = [0.0] + ctx_m + [ctx_m[-1] if ctx_m else 0.0] + tgt_m + [tgt_m[-1] if tgt_m else 0.0]

    sep_pos = 1 + len(ctx_t)  # index of SEP token
    return tokens, times, sep_pos


def truncate_context_to_fit(
    tokens: List[int],
    musical_times: List[float],
    sep_pos: int,
    max_len: int,
) -> Tuple[List[int], List[float], int]:
    """Truncate conditioning context from the beginning to fit into max_len.

    Target stream (after SEP) is never truncated.
    Returns (tokens, musical_times, new_sep_pos).
    """
    if len(tokens) <= max_len:
        return tokens, musical_times, sep_pos

    excess = len(tokens) - max_len
    # Context is tokens[1:sep_pos] (between BOS and SEP)
    ctx_len = sep_pos - 1
    if excess >= ctx_len:
        # Remove entire context — keep BOS, SEP, target, EOS
        bos_tok = tokens[0]
        bos_time = musical_times[0]
        after_sep = tokens[sep_pos:]
        after_sep_times = musical_times[sep_pos:]
        tokens = [bos_tok] + after_sep
        musical_times = [bos_time] + after_sep_times
        new_sep_pos = 1
    else:
        # Truncate from start of context
        # Keep tokens[0] (BOS), remove tokens[1:1+excess], keep rest
        bos_tok = tokens[0]
        bos_time = musical_times[0]
        remaining_ctx = tokens[1 + excess:sep_pos]
        remaining_ctx_times = musical_times[1 + excess:sep_pos]
        after_sep = tokens[sep_pos:]
        after_sep_times = musical_times[sep_pos:]
        tokens = [bos_tok] + remaining_ctx + after_sep
        musical_times = [bos_time] + remaining_ctx_times + after_sep_times
        new_sep_pos = 1 + len(remaining_ctx)

    # Final safety: if still too long, truncate target from end
    if len(tokens) > max_len:
        tokens = tokens[:max_len]
        musical_times = musical_times[:max_len]

    return tokens, musical_times, new_sep_pos


def build_all_cascade_stages(
    ev: List[Tuple[float, int, int, int, float]],
    tempo_bpm: float,
    bar_starts: np.ndarray,
    bars_meta: list,
    vocab: dict,
    config: InstrumentConfig,
    chord_labels: List[Tuple[float, int, int]],
    ablation: str,
) -> List[Tuple[List[int], List[float], int, int]]:
    """Build all cascade stages for one song.

    Returns list of (tokens, musical_times, sep_pos, stage_id) tuples.
    """
    if ablation == "A":
        order = CASCADE_ORDER_A
    elif ablation == "B":
        order = CASCADE_ORDER_B
    else:
        raise ValueError(f"Unknown ablation: {ablation}")

    # Split events by instrument
    by_inst = split_events_by_instrument(ev, config)

    # For ablation B, merge "other" events into "guitar"
    if ablation == "B":
        guitar_idx = config.guitar_idx
        other_idx = config.other_idx
        if guitar_idx is not None and other_idx is not None:
            other_events = by_inst.pop(other_idx, [])
            # Relabel other events to guitar instrument index
            relabeled = [(s, guitar_idx, m, v, d) for (s, _, m, v, d) in other_events]
            by_inst.setdefault(guitar_idx, []).extend(relabeled)
            by_inst[guitar_idx].sort(key=lambda x: x[0])

    results = []
    conditioning_insts: List[int] = []

    for stage_id, inst_name in enumerate(order):
        # Resolve instrument index
        if inst_name not in config.names:
            continue
        target_idx = config.names.index(inst_name)

        target_events = by_inst.get(target_idx, [])
        if not target_events:
            # Skip stages with no events for this instrument
            conditioning_insts.append(target_idx)
            continue

        # Build conditioning context from all preceding instruments
        ctx_events = merge_streams_chronological(
            [by_inst.get(idx, []) for idx in conditioning_insts]
        )

        # Tokenize context and target
        if ctx_events:
            ctx_tokens = tokenize_song(ctx_events, tempo_bpm, bar_starts, bars_meta, vocab)
            ctx_tokens_with_chords, ctx_times = inject_chord_tokens(
                ctx_tokens, chord_labels, vocab, tempo_bpm
            )
        else:
            ctx_tokens_with_chords = []
            ctx_times = []

        tgt_tokens = tokenize_song(target_events, tempo_bpm, bar_starts, bars_meta, vocab)
        tgt_times = compute_musical_times(tgt_tokens, vocab)

        tokens, times, sep_pos = build_cascade_example(
            ctx_tokens_with_chords, ctx_times,
            tgt_tokens, tgt_times,
            vocab, stage_id,
        )

        # Truncate to fit SEQ_LEN
        tokens, times, sep_pos = truncate_context_to_fit(
            tokens, times, sep_pos, SEQ_LEN
        )

        results.append((tokens, times, sep_pos, stage_id))
        conditioning_insts.append(target_idx)

    return results


# ── Main pipeline ─────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser("pre_cascade: cascade preprocessing for multi-instrument generation.")
    ap.add_argument("--midi_folder", required=True, help="Folder containing per-song multi-track MIDI files.")
    ap.add_argument("--data_folder", required=True, help="Output folder for cascade pickles + vocab.")
    ap.add_argument("--ablation", default="A", choices=["A", "B"], help="Ablation variant (A=6 stages, B=5 stages with merged guitar+other).")
    ap.add_argument("--instrument_set", default="blues6", choices=list(INSTRUMENT_PRESETS.keys()),
                    help="Preset instrument configuration (default: blues6).")
    ap.add_argument("--blues_only", action="store_true", help="Filter out non-bluesy songs.")
    ap.add_argument("--no-aug", action="store_true", help="Disable train-time augmentation.")
    ap.add_argument("--seq_len", type=int, default=SEQ_LEN, help="Max sequence length (default: 1024).")
    args = ap.parse_args()

    seq_len = args.seq_len
    config = make_instrument_config(INSTRUMENT_PRESETS[args.instrument_set])
    print(f"Instrument config: {config.names} (drums={config.drum_idx})")
    print(f"Ablation: {args.ablation}")

    os.makedirs(args.data_folder, exist_ok=True)
    samples_dir = os.path.join(args.data_folder, "_samples")
    os.makedirs(samples_dir, exist_ok=True)

    # 1) Collect MIDI paths and split train/val
    paths = sorted(
        glob.glob(os.path.join(args.midi_folder, "*.mid")) +
        glob.glob(os.path.join(args.midi_folder, "*.midi"))
    )
    if not paths:
        raise RuntimeError(f"No MIDI files found in '{args.midi_folder}'.")
    if len(paths) < 2:
        raise RuntimeError(f"Need at least 2 MIDI files, got {len(paths)}.")

    random.seed(42)
    random.shuffle(paths)
    n_train = int(0.8 * len(paths))
    train_paths, val_paths = paths[:n_train], paths[n_train:]

    # 2) Extract events from all files
    song_meta: Dict[str, Tuple[list, float, np.ndarray, list]] = {}
    all_bars_meta = []
    skipped = 0
    for p in paths:
        try:
            ev, tempo, bar_starts, bars_meta = extract_multitrack_events(p, config)
            if not ev:
                raise RuntimeError("No events extracted.")
            if args.blues_only and not is_track_bluesy(ev, tempo, config=config):
                skipped += 1
                continue
        except Exception as e:
            print(f"Skipping {os.path.basename(p)}: {e}")
            continue
        song_meta[p] = (ev, tempo, bar_starts, bars_meta)
        all_bars_meta.append(bars_meta)

    if skipped > 0:
        print(f"Filtered out {skipped} non-bluesy songs.")
    if not song_meta:
        raise RuntimeError("No events extracted from any file.")

    # 3) Build base vocab (reuse existing pre.py infrastructure)
    aug_transposes = config.aug_transposes if not getattr(args, 'no_aug', False) else []
    aug_vel_deltas = config.aug_vel_deltas if config.aug_vel_deltas else [0]
    do_aug = not getattr(args, 'no_aug', False)

    events_for_vocab = []
    for p in paths:
        if p not in song_meta:
            continue
        ev = song_meta[p][0]
        events_for_vocab.extend(ev)
        if do_aug:
            for s in aug_transposes:
                for dv in aug_vel_deltas:
                    ev_aug = augment_events_additive(ev, semitone_shift=s, vel_delta=dv, config=config)
                    if ev_aug is not None:
                        events_for_vocab.extend(ev_aug)

    bar_pairs = gather_bar_pairs(all_bars_meta)
    pitch_maps = build_pitch_maps(events_for_vocab, config)
    vocab = build_event_vocab(pitch_maps, bar_pairs, config)

    # 4) Extend vocab with cascade tokens
    vocab = extend_vocab_for_cascade(vocab)
    print(f"Extended vocab with SEP, CHORD_ROOT, CHORD_QUAL tokens")

    # 5) Tokenize all songs into cascade examples
    def process_group(group_paths: List[str], do_aug: bool, split_name: str):
        all_seqs: List[List[int]] = []
        all_times: List[List[float]] = []
        all_seps: List[int] = []
        all_stages: List[int] = []

        for i, p in enumerate(group_paths):
            if i % 10 == 0:
                print(f"Cascade tokenize ({split_name}): {i}/{len(group_paths)} {os.path.basename(p)}")
            if p not in song_meta:
                continue
            ev, tempo, bar_starts, bars_meta = song_meta[p]

            def add_one(ev_local):
                chord_labels = extract_chord_labels(ev_local, tempo, config)
                stages = build_all_cascade_stages(
                    ev_local, tempo, bar_starts, bars_meta,
                    vocab, config, chord_labels, args.ablation,
                )
                for (tokens, times, sep_pos, stage_id) in stages:
                    all_seqs.append(tokens)
                    all_times.append(times)
                    all_seps.append(sep_pos)
                    all_stages.append(stage_id)

            add_one(ev)

            if do_aug:
                for s in aug_transposes:
                    for dv in aug_vel_deltas:
                        ev_aug = augment_events_additive(ev, s, dv, config=config)
                        if ev_aug is not None:
                            add_one(ev_aug)

        return all_seqs, all_times, all_seps, all_stages

    train_seqs, train_times, train_seps, train_stages = process_group(
        train_paths, do_aug=do_aug, split_name="train"
    )
    val_seqs, val_times, val_seps, val_stages = process_group(
        val_paths, do_aug=False, split_name="val"
    )

    # 6) Compact vocab (remap sequences in-place)
    vocab = compact_vocab(train_seqs, val_seqs, vocab)

    # 7) Save
    with open(os.path.join(args.data_folder, "cascade_train.pkl"), "wb") as f:
        pickle.dump({
            "sequences": train_seqs,
            "musical_times": train_times,
            "sep_positions": train_seps,
            "stage_ids": train_stages,
        }, f)

    with open(os.path.join(args.data_folder, "cascade_val.pkl"), "wb") as f:
        pickle.dump({
            "sequences": val_seqs,
            "musical_times": val_times,
            "sep_positions": val_seps,
            "stage_ids": val_stages,
        }, f)

    with open(os.path.join(args.data_folder, "cascade_vocab.json"), "w") as f:
        json.dump(vocab, f, indent=2)

    # 8) Round-trip a few samples
    for i in range(min(3, len(train_seqs))):
        outp = os.path.join(samples_dir, f"cascade_{i:03d}.mid")
        decode_to_midi(train_seqs[i], vocab, outp, tempo_bpm=120.0)

    # 9) Report
    layout = vocab["layout"]
    total_vocab = max(v["start"] + v["size"] for v in layout.values())
    print(f"\n── Cascade Preprocessing Summary ────────────")
    print(f"  Ablation:         {args.ablation}")
    print(f"  Total vocab:      {total_vocab}")
    print(f"  SEP token:        {layout['SEP']['start']}")
    print(f"  CHORD_ROOT size:  {layout['CHORD_ROOT']['size']}")
    print(f"  CHORD_QUAL size:  {layout['CHORD_QUAL']['size']}")
    print(f"  Train examples:   {len(train_seqs)}")
    print(f"  Val examples:     {len(val_seqs)}")
    print(f"  Max seq_len:      {seq_len}")
    if train_seqs:
        avg_len = sum(len(s) for s in train_seqs) / len(train_seqs)
        print(f"  Avg train len:    {avg_len:.0f}")
    stage_counts = {}
    for sid in train_stages:
        stage_counts[sid] = stage_counts.get(sid, 0) + 1
    order = CASCADE_ORDER_A if args.ablation == "A" else CASCADE_ORDER_B
    for sid, name in enumerate(order):
        cnt = stage_counts.get(sid, 0)
        print(f"  Stage {sid} ({name:>8}):  {cnt} examples")
    print(f"──────────────────────────────────────────────")


if __name__ == "__main__":
    main()
