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



def events_to_midi(events: list[dict], out_path: Path, max_note_dur: float = 4.0):
    """Convert a list of {instrument, pitch, time, velocity} dicts to a MIDI file."""
    pm = pretty_midi.PrettyMIDI()

    # Collect note_on / note_off pairs per (instrument, pitch)
    tracks: dict[int, pretty_midi.Instrument] = {}
    # (inst, pitch) → (note_on absolute time, velocity)
    pending: dict[tuple, tuple[float, int]] = {}

    def close_note(inst_id, pitch, end_time):
        key = (inst_id, pitch)
        if key not in pending:
            return
        on_time, on_vel = pending.pop(key)
        dur = max(end_time - on_time, 0.01)
        tracks[inst_id].notes.append(pretty_midi.Note(
            velocity=on_vel, pitch=pitch,
            start=on_time, end=on_time + dur,
        ))

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
            # If this pitch is already open, close it first (implicit note-off)
            if key in pending:
                close_note(inst_id, pitch, abs_time)
            pending[key] = (abs_time, max(1, min(127, vel)))
        else:
            close_note(inst_id, pitch, abs_time)

    # Close any notes still open — cap at max_note_dur so one note doesn't span the piece
    for (inst_id, pitch), (on_time, on_vel) in list(pending.items()):
        end_time = min(on_time + max_note_dur, abs_time + 0.25)
        tracks[inst_id].notes.append(pretty_midi.Note(
            velocity=on_vel, pitch=pitch,
            start=on_time, end=end_time,
        ))

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
    ap.add_argument("--max_note_dur", type=float, default=4.0,
                    help="Cap note duration (seconds) — prevents stuck open notes")
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
    events_to_midi(events, Path(args.out_midi), max_note_dur=args.max_note_dur)


if __name__ == "__main__":
    main()
