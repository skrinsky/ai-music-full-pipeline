#!/usr/bin/env python3
"""
Generate MIDI using a fine-tuned Notochord model.

Uses Notochord's built-in query() API for autoregressive generation.
Optionally primes the hidden state from one of your own tracks so the
model starts with context from your style.

Usage:
    python finetune/notochord_generate.py \\
        --checkpoint finetune/runs/noto_finetuned.pt \\
        --out_midi   finetune/runs/generated/noto_out.mid

    # Prime from one of your tracks:
    python finetune/notochord_generate.py \\
        --checkpoint finetune/runs/noto_finetuned.pt \\
        --prompt_midi summer_midi/my_song.mid \\
        --out_midi   finetune/runs/generated/noto_continuation.mid
"""

import argparse
from pathlib import Path

import pretty_midi
import torch
from notochord import Notochord


# GM programs for typical indie rock instruments — constrain generation to these
ROCK_PROGRAMS = {
    25,   # Acoustic Guitar (steel)
    26,   # Acoustic Guitar (nylon)
    27,   # Electric Guitar (clean)
    28,   # Electric Guitar (muted)
    29,   # Overdriven Guitar
    30,   # Distortion Guitar
    33,   # Electric Bass (finger)
    34,   # Electric Bass (pick)
    0,    # Acoustic Grand Piano
    4,    # Electric Piano 1
    80,   # Lead 1 (square) — synth lead
    81,   # Lead 2 (sawtooth)
    128,  # Drums (channel 10)
}


def events_to_midi(events: list[dict], out_path: Path):
    """Convert a list of {instrument, pitch, time, velocity} dicts to a MIDI file."""
    pm = pretty_midi.PrettyMIDI()

    # Collect note_on / note_off pairs per (instrument, pitch)
    tracks: dict[int, pretty_midi.Instrument] = {}
    pending: dict[tuple, float] = {}  # (inst, pitch) → note_on absolute time

    abs_time = 0.0
    for ev in events:
        abs_time += ev["time"]
        inst_id = ev["instrument"]
        pitch   = ev["pitch"]
        vel     = int(ev["velocity"])

        if inst_id not in tracks:
            is_drum = (inst_id == 128)
            program = 0 if is_drum else inst_id
            tracks[inst_id] = pretty_midi.Instrument(
                program=program, is_drum=is_drum,
                name=f"inst_{inst_id}")

        key = (inst_id, pitch)
        if vel > 0:
            pending[key] = abs_time
        else:
            if key in pending:
                on_time = pending.pop(key)
                dur = max(abs_time - on_time, 0.01)
                # Use the velocity from the note_on (we store it at on_time)
                note = pretty_midi.Note(
                    velocity=max(1, min(127, vel if vel > 0 else 64)),
                    pitch=pitch,
                    start=on_time,
                    end=on_time + dur,
                )
                tracks[inst_id].notes.append(note)

    # Close any notes still open at the end
    for (inst_id, pitch), on_time in pending.items():
        note = pretty_midi.Note(
            velocity=64, pitch=pitch,
            start=on_time, end=abs_time + 0.5)
        tracks[inst_id].notes.append(note)

    for inst in tracks.values():
        pm.instruments.append(inst)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pm.write(str(out_path))
    print(f"Saved MIDI → {out_path}  ({len(events)} events, {abs_time:.1f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",      required=True,
                    help="Fine-tuned checkpoint .pt file")
    ap.add_argument("--base_checkpoint", default=None,
                    help="Original pre-trained checkpoint (needed if fine-tuned ckpt "
                         "was saved without architecture kwargs, e.g. kw={})")
    ap.add_argument("--out_midi",     required=True)
    ap.add_argument("--prompt_midi",  default=None,
                    help="Optional: prime the model from this MIDI before generating")
    ap.add_argument("--n_events",     type=int,   default=1000,
                    help="Number of events to generate (~1000 ≈ 60–90s of music)")
    ap.add_argument("--temperature",  type=float, default=0.9,
                    help="Sampling temperature for instrument and pitch heads")
    ap.add_argument("--max_time",     type=float, default=2.0,
                    help="Max seconds between events (prevents long silences)")
    ap.add_argument("--device",       default="auto")
    args = ap.parse_args()

    device = (
        "mps"  if torch.backends.mps.is_available()  else
        "cuda" if torch.cuda.is_available()           else
        "cpu"
    ) if args.device == "auto" else args.device
    print(f"Device: {device}")

    print(f"Loading: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    # If the fine-tuned checkpoint has empty kw (saved by old buggy code),
    # load architecture from the base checkpoint instead.
    if not ckpt.get("kw") and args.base_checkpoint:
        print(f"  kw empty — loading architecture from {args.base_checkpoint}")
        model = Notochord.from_checkpoint(args.base_checkpoint)
        model.load_state_dict(ckpt["model_state"])
    else:
        model = Notochord.from_checkpoint(args.checkpoint)

    model = model.to(device)
    model.eval()

    # Prime from a MIDI file if provided
    model.reset()
    if args.prompt_midi:
        print(f"Priming from {args.prompt_midi} …")
        try:
            model.prompt(args.prompt_midi)
        except Exception as exc:
            print(f"  prompt() failed: {exc} — starting from scratch")

    # Generate events autoregressively
    print(f"Generating {args.n_events} events …")
    events = []
    with torch.no_grad():
        for i in range(args.n_events):
            try:
                ev = model.query(
                    max_time=args.max_time,
                    # Constrain to rock instruments (remove to allow any instrument)
                    include_inst=list(ROCK_PROGRAMS),
                )
                # query() returns a dict with instrument, pitch, time, vel
                events.append({
                    "instrument": int(ev.get("instrument", ev.get("inst", 0))),
                    "pitch":      int(ev.get("pitch", 60)),
                    "time":       float(ev.get("time", 0.1)),
                    "velocity":   float(ev.get("vel", ev.get("velocity", 64))),
                })
                # Feed the event back into the model
                model.feed(
                    inst=events[-1]["instrument"],
                    pitch=events[-1]["pitch"],
                    time=events[-1]["time"],
                    vel=events[-1]["velocity"],
                )
            except Exception as exc:
                print(f"  Event {i}: {exc}")
                break

    print(f"Generated {len(events)} events")
    events_to_midi(events, Path(args.out_midi))


if __name__ == "__main__":
    main()
