#!/usr/bin/env python3
"""
Generate MIDI from a fine-tuned (or pretrained) MMT checkpoint.

Three generation modes:
  1. Unconditional — start from scratch (default)
  2. Instrument-conditioned — declare which instruments should appear
  3. Continuation — seed from the first N beats of a MIDI file

Usage:
    # Unconditional
    python finetune/mmt_generate.py \\
        --ckpt  finetune/runs/mmt_finetuned/checkpoints/best_model.pt \\
        --out_midi finetune/runs/mmt_out.mid

    # Continuation from your own MIDI
    python finetune/mmt_generate.py \\
        --ckpt  finetune/runs/mmt_finetuned/checkpoints/best_model.pt \\
        --prompt_midi "summer_midi/01 Hum.mid" \\
        --prompt_beats 4 \\
        --out_midi finetune/runs/mmt_continuation.mid

    # More creative / varied output
    python finetune/mmt_generate.py \\
        --ckpt ... --out_midi ... --temperature 1.1 --filter_thres 0.85
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

MMT_DIR = Path(__file__).resolve().parent.parent / "vendor" / "mmt" / "mmt"
_CONVERT  = Path(__file__).resolve().parent  # for mmt_convert import


def _add_paths():
    for p in (str(MMT_DIR), str(_CONVERT)):
        if p not in sys.path:
            sys.path.insert(0, p)


# --- Shared duration snap (mirrors mmt_convert.py, avoids circular import) ---
_KNOWN_DURATIONS = np.array([
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
    15, 16, 18, 20, 21, 24, 30, 36, 40, 42,
    48, 60, 72, 84, 96, 120, 144, 168, 192, 384,
])
_MMT_RES = 12
_MAX_BEAT = 1024


def _snap_dur(dur: int) -> int:
    if dur <= 0:
        return 1
    return int(_KNOWN_DURATIONS[np.argmin(np.abs(_KNOWN_DURATIONS - dur))])


def _midi_to_notes(midi_path: Path) -> np.ndarray | None:
    import muspy
    try:
        music = muspy.read_midi(str(midi_path))
    except Exception as e:
        print(f"  muspy error: {e}", file=sys.stderr)
        return None
    src_res = music.resolution
    notes = []
    for track in music.tracks:
        program = -1 if track.is_drum else track.program
        for note in track.notes:
            mmt_time = note.time * _MMT_RES / src_res
            beat     = int(mmt_time) // _MMT_RES
            position = int(mmt_time) % _MMT_RES
            duration = _snap_dur(max(1, round(note.duration * _MMT_RES / src_res)))
            if beat >= _MAX_BEAT or not 0 <= note.pitch <= 127:
                continue
            notes.append((beat, position, note.pitch, duration, program))
    if not notes:
        return None
    return np.array(sorted(notes), dtype=np.int32)


# --- Prompt builders ---

def _sos_row(encoding) -> list:
    row = [0] * len(encoding["dimensions"])
    row[encoding["dimensions"].index("type")] = encoding["type_code_map"]["start-of-song"]
    return row


def build_unconditional_prompt(encoding) -> np.ndarray:
    return np.array([_sos_row(encoding)], dtype=int)


def build_continuation_prompt(encoding, midi_path: str, n_beats: int) -> np.ndarray:
    import representation
    notes = _midi_to_notes(Path(midi_path))
    if notes is None or len(notes) == 0:
        print("Warning: empty prompt MIDI — falling back to unconditional.")
        return build_unconditional_prompt(encoding)
    notes = notes[notes[:, 0] < n_beats]
    if len(notes) == 0:
        print(f"Warning: no notes in first {n_beats} beats — falling back to unconditional.")
        return build_unconditional_prompt(encoding)
    codes = representation.encode_notes(notes, encoding)
    eos_type = encoding["type_code_map"]["end-of-song"]
    codes = codes[codes[:, 0] != eos_type]
    print(f"Prompt: {len(codes)} events from first {n_beats} beats of {Path(midi_path).name}")
    return codes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",          required=True,
                    help="Path to MMT checkpoint (best_model.pt)")
    ap.add_argument("--out_midi",      required=True)
    # Generation mode
    ap.add_argument("--prompt_midi",   default=None,
                    help="Seed from first N beats of this MIDI (continuation mode)")
    ap.add_argument("--prompt_beats",  type=int,   default=4)
    # Model architecture (must match checkpoint)
    ap.add_argument("--dim",       type=int,   default=512)
    ap.add_argument("--layers",    type=int,   default=6)
    ap.add_argument("--heads",     type=int,   default=8)
    ap.add_argument("--max_seq_len",   type=int, default=1024)
    ap.add_argument("--max_beat",      type=int, default=256)
    # Sampling
    ap.add_argument("--n_tokens",      type=int,   default=1024)
    ap.add_argument("--temperature",   type=float, default=1.0)
    ap.add_argument("--filter",        default="top_k",
                    choices=["top_k", "top_p"],
                    help="Sampling filter type")
    ap.add_argument("--filter_thres",  type=float, default=0.9,
                    help="top_k/top_p threshold (0–1 probability mass to keep)")
    ap.add_argument("--device",        default="auto")
    args = ap.parse_args()

    _add_paths()
    import music_x_transformers
    import representation

    if args.device == "auto":
        device = ("mps"  if torch.backends.mps.is_available() else
                  "cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = args.device
    print(f"Device: {device}")

    encoding = representation.get_encoding()

    print("Building model...")
    model = music_x_transformers.MusicXTransformer(
        dim=args.dim,
        encoding=encoding,
        depth=args.layers,
        heads=args.heads,
        max_seq_len=args.max_seq_len,
        max_beat=args.max_beat,
        use_abs_pos_emb=True,
        rotary_pos_emb=False,
        emb_dropout=0.0,
        attn_dropout=0.0,
        ff_dropout=0.0,
    ).to(device)

    print(f"Loading checkpoint: {args.ckpt}")
    state = torch.load(args.ckpt, map_location=device)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model.eval()

    # Build prompt
    if args.prompt_midi:
        prompt = build_continuation_prompt(encoding, args.prompt_midi, args.prompt_beats)
    else:
        prompt = build_unconditional_prompt(encoding)
        print("Mode: unconditional generation")

    prompt_tensor = torch.tensor(prompt, dtype=torch.long, device=device).unsqueeze(0)

    print(f"Generating {args.n_tokens} tokens "
          f"(temperature={args.temperature}, {args.filter}={args.filter_thres})...")
    with torch.no_grad():
        codes = model.generate(
            start_tokens=prompt_tensor,
            seq_len=args.n_tokens,
            temperature=args.temperature,
            filter_logits_fn=args.filter,
            filter_thres=args.filter_thres,
            monotonicity_dim=(0, 1),
        )

    codes_np = codes[0].cpu().numpy()
    print(f"Generated {len(codes_np)} events")

    music = representation.decode(codes_np, encoding)

    out_path = Path(args.out_midi)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    music.write(str(out_path))
    print(f"Saved MIDI → {out_path}")


if __name__ == "__main__":
    main()
