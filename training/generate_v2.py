#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, math, json, time, argparse, importlib, hashlib, random
from collections import deque, Counter
from datetime import datetime
import pretty_midi


def prune_midi_tracks(midi_path: str, tracks_csv: str | None):
    """
    Keep only the requested track names in an existing MIDI file.
    Track names must match pretty_midi instrument.name (case-insensitive).
    """
    if not tracks_csv:
        return
    keep = [t.strip().lower() for t in tracks_csv.split(",") if t.strip()]
    if not keep:
        return

    pm = pretty_midi.PrettyMIDI(midi_path)
    new_insts = []
    for inst in pm.instruments:
        nm = (inst.name or "").strip().lower()
        if inst.is_drum and "drums" in keep:
            new_insts.append(inst)
        elif (not inst.is_drum) and nm in keep:
            new_insts.append(inst)
    pm.instruments = new_insts
    pm.write(midi_path)

def set_gm_programs(midi_path: str):
    """Set GM program numbers and rename tracks to GM instrument names.

    pretty_midi uses 0-based GM programs.  We also rename each track so
    that Logic Pro (and other DAWs) show the instrument name instead of
    the internal voice label.

    Note: Logic Pro defaults to GM Device tracks on MIDI-file open.
    To get Software Instrument tracks, drag the MIDI file into an
    existing project instead of using File > Open.
    """
    GM_BY_NAME: dict[str, tuple[int, str]] = {
        # voice  → (program, GM instrument name)
        "bass":    (33, "Electric Bass"),
        "guitar":  (27, "Electric Guitar"),
        "other":   (80, "Lead Synth"),
        "voxlead": (52, "Choir Aahs"),
        "voxharm": (54, "Synth Voice"),
        # Chorale voices
        "soprano": (73, "Flute"),
        "alto":    (69, "English Horn"),
        "tenor":   (71, "Clarinet"),
        "bassvox": (70, "Bassoon"),
    }
    pm = pretty_midi.PrettyMIDI(midi_path)
    changed = 0
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        nm = (inst.name or "").strip().lower()
        if nm in GM_BY_NAME:
            prog, gm_name = GM_BY_NAME[nm]
            inst.program = int(prog)
            inst.name = gm_name
            changed += 1
    if changed:
        pm.write(midi_path)


# Default SoundFont search order
_SF2_SEARCH = [
    os.path.expanduser("~/Library/Audio/Sounds/Banks/FluidR3_GM.sf2"),
    "/usr/share/sounds/sf2/FluidR3_GM.sf2",
    "/usr/local/share/soundfonts/default.sf2",
]

def find_soundfont() -> str | None:
    for p in _SF2_SEARCH:
        if os.path.isfile(p):
            return p
    return None

def render_midi_to_wav(midi_path: str, wav_path: str | None = None,
                       sample_rate: int = 44100) -> str | None:
    """Render a MIDI file to WAV using FluidSynth + GM SoundFont.

    Returns the WAV path on success, None if no SoundFont found.
    """
    import shutil, subprocess
    if shutil.which("fluidsynth") is None:
        print("WARNING: fluidsynth not found — skipping audio render.")
        print("  Install via: brew install fluid-synth")
        return None
    sf2 = find_soundfont()
    if sf2 is None:
        print("WARNING: No GM SoundFont found — skipping audio render.")
        print("  Install one at: ~/Library/Audio/Sounds/Banks/FluidR3_GM.sf2")
        return None
    if wav_path is None:
        wav_path = os.path.splitext(midi_path)[0] + ".wav"
    subprocess.run(
        ["fluidsynth", "-ni", "-F", wav_path, "-r", str(sample_rate),
         sf2, midi_path],
        check=True, capture_output=True,
    )
    print(f"Rendered audio → {wav_path}")
    return wav_path


def prune_midi_tracks(midi_path: str, tracks_csv: str | None):
    """Remove empty instruments, and if tracks_csv provided, keep only those instrument names.
    Expects instrument names in the MIDI to match track names (e.g. drums,bass,guitar,other,voxlead,voxharm).
    """
    if not midi_path:
        return
    pm = pretty_midi.PrettyMIDI(midi_path)
    keep = None
    if tracks_csv:
        keep = {t.strip() for t in str(tracks_csv).split(',') if t.strip()}
        keep = {k.lower() for k in keep}

    new_insts = []
    for inst in pm.instruments:
        # drop empty instruments
        if not inst.notes and not getattr(inst, 'control_changes', None) and not getattr(inst, 'pitch_bends', None):
            continue
        if keep is not None:
            nm = (inst.name or '').strip().lower()
            if nm not in keep:
                continue
        new_insts.append(inst)

    pm.instruments = new_insts
    pm.write(midi_path)


import torch
import torch.nn as nn

# =================== CLI ===================
def get_args():
    ap = argparse.ArgumentParser("Grammar-constrained factored ES generator (grid-snapped TIME_SHIFT).")
    ap.add_argument("--ckpt", required=True, help="Path to finetuned .pt")
    ap.add_argument("--vocab_json", required=True, help="Path to event_vocab.json used for training")
    ap.add_argument("--out_midi", required=True, help="Path to write .mid")
    ap.add_argument("--out_meta_json", default="", help="Optional path for generation metadata JSON")

    # sampling + grammar knobs
    ap.add_argument("--max_tokens", type=int, default=3000)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--ctx", type=int, default=512)

    ap.add_argument("--device", default="auto", help="auto|cuda|mps|cpu (auto picks cuda then mps then cpu)")
    ap.add_argument("--tracks", default="", help="Optional comma-separated subset of tracks to allow (overrides vocab_json). Examples: drums,bass | drums,bass,guitar | all. Aliases: voxbg/bgvox/backingvox/auxvox -> voxharm.")

    ap.add_argument("--max_inst_streak", type=int, default=8)
    ap.add_argument("--max_notes_per_step", type=int, default=12,
                    help="Max notes between TIME_SHIFT/BAR events before forcing a time advance.")
    ap.add_argument("--drum_bonus", type=float, default=0.4)

    # Make this a real toggle
    ap.add_argument("--anti_silence", action="store_true", default=False)
    ap.add_argument("--silence_penalty", type=float, default=0.35)
    ap.add_argument("--silence_horizon", type=int, default=64)

    ap.add_argument("--min_notes_before_stop", type=int, default=120)
    ap.add_argument("--rep_penalty", type=float, default=1.20)
    ap.add_argument("--pitch_hist_len", type=int, default=16)
    ap.add_argument("--entropy_floor", type=float, default=0.0)
    ap.add_argument("--entropy_ceiling", type=float, default=0.0,
                    help="If PITCH entropy exceeds this, skip the note and force a TIME_SHIFT instead. 0=disabled. Try 3.5.")

    # ===== Grid snapping controls =====
    ap.add_argument("--snap_time_shift", action="store_true", default=True,
                    help="Snap TIME_SHIFT deltas to a rhythmic grid (auto-detected pulse).")
    ap.add_argument("--grid_straight_steps", type=int, default=6,
                    help="Fallback grid step if pulse detection hasn't fired yet. 6=1/16 note.")
    ap.add_argument("--force_grid_step", type=int, default=0,
                    help="Force a fixed grid step, bypassing auto-detection. 6=1/16, 8=1/8-triplet, 12=1/8. 0=auto.")
    ap.add_argument("--grid_detect_window", type=int, default=24,
                    help="How many TIME_SHIFT deltas to collect before locking pulse, and window for re-detection.")
    ap.add_argument("--grid_lock_tokens", type=int, default=96,
                    help="Minimum tokens between pulse re-detection checks.")

    ap.add_argument("--tempo_bpm", type=float, default=0.0,
                    help="Playback tempo for output MIDI (BPM). 0 = read from vocab median_tempo_bpm, fallback 120.")
    ap.add_argument("--seed", type=int, default=None)

    # ===== MIDI seed (prompt) =====
    ap.add_argument("--seed_midi", default="", help="Path to a MIDI file to use as the opening prompt. The seed is fed as context but NOT written to the output MIDI.")
    ap.add_argument("--seed_bars", type=int, default=0, help="How many bars of the seed to use as context (0 = all).")
    ap.add_argument("--seed_start_bar", type=int, default=0, help="Bar offset within the seed MIDI to start from (default: 0 = beginning).")

    # ===== KNN-LM guidance =====
    ap.add_argument("--knn_index", default="", help="Path prefix for FAISS index built by build_knn_index.py (loads <prefix>.faiss + <prefix>.npz). Empty = disabled.")
    ap.add_argument("--knn_k", type=int, default=16, help="Number of nearest neighbors for KNN-LM. Try 8-32.")
    ap.add_argument("--knn_lambda", type=float, default=0.3, help="KNN interpolation weight (0=model only, 1=KNN only). Try 0.2-0.4.")

    return ap.parse_args()

# =================== Model ===================
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=8192):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        return x + self.pe[:x.size(1)].unsqueeze(0)

class FactorizedESModel(nn.Module):
    def __init__(self, pad_id: int, type_names, head_sizes, num_embeddings: int,
                 d_model: int, n_heads: int, n_layers: int, ff_mult: int, dropout: float):
        super().__init__()
        self.pad_id     = pad_id
        self.type_names = type_names
        self.head_sizes = head_sizes
        self.num_types  = len(type_names)

        self.tok_emb = nn.Embedding(num_embeddings, d_model, padding_idx=pad_id)
        self.pos_emb = PositionalEmbedding(d_model)

        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model*ff_mult,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.tr = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.drop = nn.Dropout(dropout)

        self.type_head   = nn.Linear(d_model, self.num_types, bias=True)
        self.value_heads = nn.ModuleList([nn.Linear(d_model, s, bias=True) for s in head_sizes])

    def forward(self, x, return_hidden=False):
        B, T = x.shape
        h = self.drop(self.pos_emb(self.tok_emb(x)))
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        pad  = (x == self.pad_id)
        h = self.tr(h, mask=mask, src_key_padding_mask=pad)
        type_logits  = self.type_head(h)
        value_logits = [head(h) for head in self.value_heads]
        if return_hidden:
            return type_logits, value_logits, h[:, -1, :].detach()
        return type_logits, value_logits

# =================== Utils ===================
def seed_everything(seed: int | None):
    if seed is None:
        seed = int.from_bytes(os.urandom(8), "little")
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np
        np.random.seed(seed % (2**32 - 1))
    except Exception:
        pass
    return seed

@torch.no_grad()
def sample_from_logits(logits_1d, temperature=1.0, top_p=1.0):
    if temperature <= 0:
        return int(torch.argmax(logits_1d).item())
    logits = logits_1d / max(1e-8, temperature)
    probs = torch.softmax(logits, dim=-1)
    if top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cdf = torch.cumsum(sorted_probs, dim=-1)
        keep = cdf <= top_p
        keep[..., 0] = True
        clipped = torch.where(keep, sorted_probs, torch.zeros_like(sorted_probs))
        clipped = clipped / (clipped.sum(dim=-1, keepdim=True) + 1e-12)
        choice_sorted = torch.multinomial(clipped, 1)
        return int(sorted_idx[choice_sorted].item())
    return int(torch.multinomial(probs, 1).item())

def maybe_raise_temperature_for_entropy(logits_1d, base_temp: float, entropy_floor: float) -> float:
    if entropy_floor <= 0:
        return base_temp
    p = torch.softmax(logits_1d / max(1e-6, base_temp), dim=-1)
    ent = -(p * torch.log(p + 1e-9)).sum()
    if ent.item() < entropy_floor:
        return min(1.2, base_temp * 1.1)
    return base_temp

def load_decoder(hint_paths=None):
    """
    Return decode_to_midi callable.

    Prefer the decoder in this repo (training/pre.py). Keep fallbacks for older layouts.
    """
    last_err = None

    # Ensure repo root + training dir are on sys.path
    try:
        here = os.path.dirname(os.path.abspath(__file__))      # .../training
        root = os.path.dirname(here)                           # repo root
        for p in [root, here]:
            if p and p not in sys.path:
                sys.path.insert(0, p)
    except Exception:
        pass

    # Add any hint paths passed in
    if hint_paths:
        for p in hint_paths:
            if p and os.path.isdir(p) and p not in sys.path:
                sys.path.insert(0, p)

    # Try these module locations in order
    candidates = [
        "training.pre",   # preferred
        "pre",            # if running from inside training/
        "preES4",         # legacy filename
        "preES3",         # older legacy
    ]

    for modname in candidates:
        try:
            mod = importlib.import_module(modname)
            if hasattr(mod, "decode_to_midi"):
                return mod.decode_to_midi
        except Exception as e:
            last_err = e

    raise ImportError(f"Could not import any decode_to_midi. Last error: {last_err}")

def hash_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]

def nearest_multiple(x: int, step: int) -> int:
    if step <= 1:
        return x
    # Use // and + to avoid Python banker's rounding snapping small values to 0
    rounded = int(step * round(float(x) / float(step)))
    return max(step, rounded)

def snap_delta(delta_steps: int, mode: str, straight_step: int, triplet_step: int, max_delta: int) -> int:
    delta_steps = max(1, min(int(delta_steps), int(max_delta)))
    if mode == "triplet":
        snapped = nearest_multiple(delta_steps, triplet_step)
    else:
        snapped = nearest_multiple(delta_steps, straight_step)

    snapped = max(1, min(int(snapped), int(max_delta)))
    return snapped

def fit_error(delta_steps: int, step: int) -> float:
    if step <= 1:
        return 0.0
    nearest = nearest_multiple(delta_steps, step)
    return abs(delta_steps - nearest) / float(step)

# =================== Seed MIDI ===================

def tokenize_seed_midi(midi_path: str, vocab: dict, max_bars: int = 0, start_bar: int = 0) -> list[int]:
    """Tokenize a MIDI file into event tokens using the training vocab.

    Returns token list *without* BOS/EOS so the caller can splice it
    into the generation sequence.
    start_bar: skip this many bars before collecting seed context.
    max_bars:  how many bars to use as context (0 = all remaining).
    """
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    for p in [root, here]:
        if p and p not in sys.path:
            sys.path.insert(0, p)
    from pre import extract_multitrack_events, tokenize_song

    ev, tempo, bar_starts, bars_meta = extract_multitrack_events(midi_path)
    if not ev:
        raise ValueError(f"Seed MIDI {midi_path} produced zero events")

    # Clamp start_bar to valid range
    start_bar = max(0, min(start_bar, len(bar_starts) - 1))
    start_sec = float(bar_starts[start_bar]) if start_bar < len(bar_starts) else 0.0

    # Determine end cutoff
    end_bar = start_bar + max_bars if max_bars > 0 else len(bar_starts)
    if end_bar < len(bar_starts):
        end_sec = float(bar_starts[end_bar])
        ev = [(s, i, m, v, d) for (s, i, m, v, d) in ev if start_sec <= s < end_sec]
    else:
        ev = [(s, i, m, v, d) for (s, i, m, v, d) in ev if s >= start_sec]

    if not ev:
        raise ValueError(f"Seed MIDI has no events in bars {start_bar}–{end_bar}")

    # Re-zero event times so bar start_bar becomes t=0
    if start_sec > 0:
        ev = [(s - start_sec, i, m, v, d) for (s, i, m, v, d) in ev]
        bar_starts = [b - start_sec for b in bar_starts[start_bar:]]
        bars_meta  = bars_meta[start_bar:] if bars_meta else bars_meta
    elif start_bar > 0:
        bar_starts = bar_starts[start_bar:]
        bars_meta  = bars_meta[start_bar:] if bars_meta else bars_meta

    tokens = tokenize_song(ev, tempo, bar_starts, bars_meta, vocab)
    bos = vocab["layout"]["BOS"]["start"]
    eos = vocab["layout"]["EOS"]["start"]
    tokens = [t for t in tokens if t != bos and t != eos]
    return tokens


def infer_phase_from_tokens(tokens: list[int], layout: dict, type_names: list[str]) -> tuple[str, int | None]:
    """Walk a token sequence and return the grammar phase + last_inst
    that the generation loop should resume from."""
    # Build reverse lookup: global_id → type_name
    id_to_type: dict[int, str] = {}
    for tname in type_names:
        spec = layout.get(tname)
        if spec is None:
            continue
        for local in range(spec["size"]):
            id_to_type[spec["start"] + local] = tname

    phase = "TIME"
    last_inst = None
    inst_start = layout.get("INST", {}).get("start", 0)

    for tok in tokens:
        tname = id_to_type.get(tok)
        if tname is None:
            continue
        if tname == "TIME_SHIFT":
            phase = "POST_TS"
        elif tname == "BAR":
            phase = "INST"
        elif tname == "INST":
            last_inst = tok - inst_start
            phase = "VEL"
        elif tname == "VEL":
            phase = "PITCH"
        elif tname.startswith("PITCH"):
            phase = "DUR"
        elif tname == "DUR":
            phase = "TIME"

    return phase, last_inst


# =================== Generation ===================
@torch.no_grad()
def generate(args):
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

    seed = seed_everything(args.seed)
    print(f"Seed: {seed} | Device: {device}")

    with open(args.vocab_json, "r") as f:
        vocab = json.load(f)

    layout = vocab["layout"]
    PAD_ID = layout["PAD"]["start"]
    BOS_ID = layout["BOS"]["start"]
    V      = max(spec["start"] + spec["size"] for spec in layout.values())

    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ckpt.get("model_config") or ckpt.get("config")
    if cfg is None:
        raise KeyError("Checkpoint missing model_config/config. Re-train with updated train.py or use a compatible checkpoint.")
    fact = ckpt["factored_meta"]

    type_names = fact["type_names"]
    head_sizes = fact["head_sizes"]
    starts     = {k: layout[k]["start"] for k in type_names}

    model = FactorizedESModel(
        pad_id=PAD_ID, type_names=type_names, head_sizes=head_sizes, num_embeddings=V,
        d_model=cfg["D_MODEL"], n_heads=cfg["N_HEADS"], n_layers=cfg["N_LAYERS"],
        ff_mult=cfg["FF_MULT"], dropout=cfg["DROPOUT"]
    ).to(device).eval()
    state = ckpt.get("model_state") or ckpt.get("model_state_dict")
    if state is None:
        raise KeyError("Checkpoint missing model_state/model_state_dict.")
    incompat = model.load_state_dict(state, strict=False)

    missing = getattr(incompat, "missing_keys", [])
    unexpected = getattr(incompat, "unexpected_keys", [])
    if unexpected:
        print("[warn] Ignored unexpected keys in checkpoint:", unexpected[:20], ("..." if len(unexpected) > 20 else ""))
    if missing:
        print("[warn] Missing keys when loading checkpoint:", missing[:20], ("..." if len(missing) > 20 else ""))

    # ===== KNN-LM index =====
    knn_index = None
    knn_next_tokens = None
    knn_pitch_type = None
    if args.knn_index:
        try:
            import faiss as _faiss
            import numpy as _np
            _fi = args.knn_index.rstrip("/")
            _faiss_path = _fi if _fi.endswith(".faiss") else _fi + ".faiss"
            _meta_path  = (_fi[:-6] if _fi.endswith(".faiss") else _fi) + ".npz"
            knn_index = _faiss.read_index(_faiss_path)
            _meta = _np.load(_meta_path)
            knn_next_tokens = _meta["next_tokens"]
            knn_pitch_type  = str(_meta["pitch_type"]) if "pitch_type" in _meta else "PITCH_GENERAL"
            print(f"KNN index: {knn_index.ntotal} vectors, pitch_type={knn_pitch_type}, k={args.knn_k}, lambda={args.knn_lambda}")
        except ImportError:
            print("WARNING: faiss not installed — KNN-LM disabled. pip install faiss-cpu")
            knn_index = None
        except Exception as _e:
            print(f"WARNING: failed to load KNN index: {_e} — KNN-LM disabled.")
            knn_index = None

    TYPE_IDX = {nm: i for i, nm in enumerate(type_names)}
    def has(name): return name in TYPE_IDX
    def maybe_idx(name): return TYPE_IDX[name] if has(name) else None
    PITCH_TYPES = [nm for nm in type_names if nm.startswith("PITCH")]

    # instrument→pitch type mapping
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

    # ---- track selection (restrict which INST values are allowed) ----
    # Default comes from vocab_json (written by preES4). You can override with --tracks.
    canonical_names = vocab.get("instrument_names", ["voxlead","voxharm","guitar","other","bass","drums"])

    # default allowed set from vocab_json
    if isinstance(vocab.get("tracks"), dict) and vocab["tracks"].get("allowed_inst_indices") is not None:
        allowed_inst_set = {int(i) for i in vocab["tracks"]["allowed_inst_indices"]}
    else:
        allowed_inst_set = set(range(layout.get("INST", {}).get("size", len(canonical_names))))

    # optional override from CLI
    if args.tracks:
        raw = args.tracks.strip().lower()
        alias = {"voxbg":"voxharm","bgvox":"voxharm","backingvox":"voxharm","auxvox":"voxharm"}
        if raw in ("all","*"):
            allowed_inst_set = set(range(layout.get("INST", {}).get("size", len(canonical_names))))
        else:
            wanted = []
            for part in [p.strip() for p in raw.split(",") if p.strip()]:
                part = alias.get(part, part)
                if part not in canonical_names:
                    raise ValueError(f"Track '{part}' not present in vocab instrument_names={canonical_names}")
                wanted.append(canonical_names.index(part))
            allowed_inst_set = set(wanted)

    allowed_inst_set = {i for i in allowed_inst_set if 0 <= i < layout.get("INST", {}).get("size", len(canonical_names))}
    print(f"Allowed tracks: {[canonical_names[i] for i in sorted(list(allowed_inst_set))]}")

    # grammar phases
    #   TIME     = after DUR (or start): allow TIME_SHIFT, BAR, INST
    #   POST_TS  = after TIME_SHIFT: allow BAR, INST (no consecutive TIME_SHIFT)
    #   INST     = after BAR: allow INST only
    def allowed_type_indices_for_phase(phase, last_inst, notes_this_step_=0):
        if phase == "TIME":
            if notes_this_step_ >= args.max_notes_per_step:
                # Too many notes at this timestep — force time advance
                opts = [maybe_idx("TIME_SHIFT"), maybe_idx("BAR")]
            else:
                opts = [maybe_idx("TIME_SHIFT"), maybe_idx("BAR"), maybe_idx("INST")]
            return [i for i in opts if i is not None] or list(range(len(type_names)))
        if phase == "POST_TS":
            # After TIME_SHIFT: BAR (73%) or INST (27%) — never consecutive TIME_SHIFT
            opts = [maybe_idx("BAR"), maybe_idx("INST")]
            return [i for i in opts if i is not None] or list(range(len(type_names)))
        if phase == "INST":
            return [maybe_idx("INST")] if has("INST") else list(range(len(type_names)))
        if phase == "VEL":
            return [maybe_idx("VEL")] if has("VEL") else list(range(len(type_names)))
        if phase == "PITCH":
            if not PITCH_TYPES:
                return list(range(len(type_names)))
            if last_inst is None:
                return [TYPE_IDX[p] for p in PITCH_TYPES]
            pt = inst_to_pitch_type.get(last_inst, None)
            if pt in TYPE_IDX:
                return [TYPE_IDX[pt]]
            return [TYPE_IDX[p] for p in PITCH_TYPES]
        if phase == "DUR":
            return [maybe_idx("DUR")] if has("DUR") else list(range(len(type_names)))
        return [maybe_idx("TIME_SHIFT")] if has("TIME_SHIFT") else list(range(len(type_names)))

    # state — optionally seed from a MIDI file
    if args.seed_midi:
        seed_tokens = tokenize_seed_midi(args.seed_midi, vocab, max_bars=args.seed_bars, start_bar=args.seed_start_bar)
        seq = [BOS_ID] + seed_tokens
        seed_offset = len(seq)   # generated tokens start here; seed is NOT written to output MIDI
        phase, last_inst = infer_phase_from_tokens(seed_tokens, layout, type_names)
        # count notes already in the seed
        notes_placed = sum(1 for t in seed_tokens if any(
            layout[n]["start"] <= t < layout[n]["start"] + layout[n]["size"]
            for n in type_names if n.startswith("PITCH")
        ))
        print(f"Seed MIDI: {args.seed_midi} → {len(seed_tokens)} tokens, {notes_placed} notes, resuming at phase={phase}")
        print("(seed tokens will be used as context only — NOT written to output MIDI)")
    else:
        seq = [BOS_ID]
        seed_offset = 1
        phase = "TIME"
        last_inst = None
        notes_placed = 0
    inst_streak = 0
    since_note_tokens = 0
    notes_this_step = 0          # notes placed since last TIME_SHIFT/BAR

    num_instruments = layout.get("INST", {}).get("size", 16)
    pitch_history = {i: deque(maxlen=args.pitch_hist_len) for i in range(num_instruments)}

    # ===== Adaptive grid state =====
    time_shift_size = layout.get("TIME_SHIFT", {}).get("size", None)
    max_delta_steps = int(time_shift_size) if time_shift_size is not None else 192
    # Candidate pulse steps in 1/24-QN units: 3=1/32, 4=1/16-triplet, 6=1/16, 8=1/8-triplet, 12=1/8, 16=1/4-triplet, 24=1/4
    GRID_CANDIDATES = [3, 4, 6, 8, 12, 16, 24]
    if args.force_grid_step > 0:
        grid_step   = args.force_grid_step
        grid_locked = True   # skip auto-detection entirely
        print(f"[grid] forced step={grid_step} (1/{int(round(96/grid_step))} note)")
    else:
        grid_step   = args.grid_straight_steps
        grid_locked = False
    grid_last_flip_at = 0
    recent_deltas = deque(maxlen=args.grid_detect_window)
    warmup_deltas = []                     # raw deltas collected before pulse is locked

    t0 = time.time()
    while len(seq) < args.max_tokens:
        ctx = torch.tensor(seq[-args.ctx:], dtype=torch.long, device=device).unsqueeze(0)
        if knn_index is not None:
            type_logits, value_logits_list, _h_last = model(ctx, return_hidden=True)
        else:
            type_logits, value_logits_list = model(ctx)
            _h_last = None
        tlog = type_logits[:, -1, :].squeeze(0)

        allowed = allowed_type_indices_for_phase(phase, last_inst, notes_this_step)
        masked = torch.full_like(tlog, -1e9)
        masked[allowed] = tlog[allowed]
        type_choice = sample_from_logits(masked, args.temperature, args.top_p)
        type_name = type_names[type_choice]

        v_logits = value_logits_list[type_choice][:, -1, :].squeeze(0).clone()

        # structure heuristics
        if type_name == "INST":
            # Restrict to allowed instruments
            if "allowed_inst_set" in locals() and allowed_inst_set is not None:
                mask_inst = torch.full_like(v_logits, -1e9)
                for ii in allowed_inst_set:
                    if 0 <= ii < v_logits.numel():
                        mask_inst[ii] = v_logits[ii]
                v_logits = mask_inst

            if last_inst is not None and inst_streak >= args.max_inst_streak and last_inst < v_logits.numel():
                v_logits[last_inst] -= 4.0
            if args.drum_bonus and DRUM_IDX is not None and DRUM_IDX < v_logits.numel():
                v_logits[DRUM_IDX] += float(args.drum_bonus)

        if args.anti_silence and type_name == "TIME_SHIFT":
            n = v_logits.numel()
            if n > 1:
                scale = args.silence_penalty * (1.5 if since_note_tokens >= args.silence_horizon else 1.0)
                penalty = torch.linspace(0.0, scale, steps=n, device=v_logits.device)
                v_logits -= penalty

        if type_name.startswith("PITCH") and last_inst is not None and last_inst != DRUM_IDX and args.rep_penalty > 1.0:
            for pid in set(pitch_history[last_inst]):
                if 0 <= pid < v_logits.numel():
                    v_logits[pid] /= float(args.rep_penalty)
            local_temp = maybe_raise_temperature_for_entropy(v_logits, args.temperature, args.entropy_floor)
        else:
            local_temp = args.temperature

        # ===== Entropy ceiling: skip note if model is guessing =====
        if args.entropy_ceiling > 0 and type_name.startswith("PITCH"):
            p = torch.softmax(v_logits / max(1e-6, local_temp), dim=-1)
            ent = -(p * torch.log(p + 1e-9)).sum().item()
            if ent > args.entropy_ceiling:
                # Model is too uncertain — abandon this note, force a time advance
                if maybe_idx("TIME_SHIFT") is not None:
                    global_id = starts["TIME_SHIFT"]
                    seq.append(global_id)
                    phase = "POST_TS"
                    since_note_tokens += 1
                    notes_this_step = 0
                continue

        # ===== KNN-LM: interpolate model distribution with nearest-neighbor distribution =====
        _knn_sampled = False
        if (knn_index is not None and knn_next_tokens is not None
                and type_name == knn_pitch_type and _h_last is not None):
            import numpy as _np
            _h_vec = _h_last.cpu().float().numpy()
            _, _I = knn_index.search(_h_vec, args.knn_k)
            _knn_dist = torch.zeros(v_logits.shape[0], device=device)
            for _nbr in _I[0]:
                if 0 <= _nbr < len(knn_next_tokens):
                    _tok = int(knn_next_tokens[_nbr])
                    if 0 <= _tok < _knn_dist.shape[0]:
                        _knn_dist[_tok] += 1.0
            if _knn_dist.sum() > 0:
                _knn_dist /= _knn_dist.sum()
                _model_probs = torch.softmax(v_logits / max(1e-6, local_temp), dim=-1)
                _mixed = (1.0 - args.knn_lambda) * _model_probs + args.knn_lambda * _knn_dist
                val_local = int(torch.multinomial(_mixed.clamp_min(1e-12), 1).item())
                _knn_sampled = True

        if not _knn_sampled:
            val_local = sample_from_logits(v_logits, local_temp, args.top_p)

        # ===== Snap TIME_SHIFT to adaptive grid =====
        if args.snap_time_shift and type_name == "TIME_SHIFT":
            raw_delta = int(val_local) + 1

            if not grid_locked:
                # Warmup: collect unsnapped deltas to detect the model's natural pulse
                warmup_deltas.append(raw_delta)
                if len(warmup_deltas) >= args.grid_detect_window:
                    # Find candidate step that minimizes mean rounding error
                    best_step, best_err = args.grid_straight_steps, float("inf")
                    for cand in GRID_CANDIDATES:
                        err = sum(fit_error(d, cand) for d in warmup_deltas) / len(warmup_deltas)
                        if err < best_err:
                            best_err, best_step = err, cand
                    grid_step = best_step
                    grid_locked = True
                    print(f"[grid] pulse detected: step={grid_step} (1/{int(round(96/grid_step))} note)  err={best_err:.3f}")
                # During warmup: pass through unsnapped
                val_local = int(raw_delta - 1)
            else:
                # Locked: snap to detected pulse, allow re-detection periodically
                recent_deltas.append(raw_delta)
                if ((len(seq) - grid_last_flip_at) >= args.grid_lock_tokens
                        and len(recent_deltas) >= max(8, args.grid_detect_window // 2)):
                    best_step, best_err = grid_step, float("inf")
                    for cand in GRID_CANDIDATES:
                        err = sum(fit_error(d, cand) for d in recent_deltas) / len(recent_deltas)
                        if err < best_err:
                            best_err, best_step = err, cand
                    if best_step != grid_step:
                        print(f"[grid] pulse shift: {grid_step}→{best_step} (1/{int(round(96/best_step))} note)  err={best_err:.3f}")
                        grid_step = best_step
                        grid_last_flip_at = len(seq)

                snapped = nearest_multiple(raw_delta, grid_step)
                snapped = max(grid_step, min(int(snapped), int(max_delta_steps)))
                val_local = int(snapped - 1)

        global_id = starts[type_name] + int(val_local)
        seq.append(global_id)

        # transitions
        if type_name == "TIME_SHIFT":
            phase = "POST_TS"; since_note_tokens += 1; notes_this_step = 0
        elif type_name == "BAR":
            phase = "INST"; since_note_tokens += 1; notes_this_step = 0
        elif type_name == "INST":
            last_inst = int(val_local)
            inst_streak = inst_streak + 1 if last_inst is not None else 0
            phase = "VEL"; since_note_tokens += 1
        elif type_name == "VEL":
            phase = "PITCH"; since_note_tokens += 1
        elif type_name.startswith("PITCH"):
            if last_inst is not None and last_inst != DRUM_IDX:
                pitch_history[last_inst].append(int(val_local))
            phase = "DUR"; notes_placed += 1; since_note_tokens = 0
            notes_this_step += 1
        elif type_name == "DUR":
            phase = "TIME"; since_note_tokens += 1
        else:
            phase = "TIME"; since_note_tokens += 1

        if len(seq) % 200 == 0:
            print(f"[{len(seq)}/{args.max_tokens}] notes={notes_placed} silent={since_note_tokens} grid={'warmup' if not grid_locked else grid_step}  {time.time()-t0:.1f}s")

        if notes_placed >= args.min_notes_before_stop and len(seq) >= int(args.max_tokens * 0.6):
            break

    out_dir = os.path.dirname(args.out_midi)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    events_json = args.out_midi.replace(".mid", ".events.json")
    with open(events_json, "w") as f:
        json.dump(seq, f)
    print(f"Generated {len(seq)} tokens. Saved raw events → {events_json}")

    hint_paths = [
        os.path.dirname(__file__),
        os.path.dirname(os.path.abspath(args.vocab_json)),
        os.path.dirname(os.path.abspath(args.ckpt)),
        os.path.dirname(os.path.dirname(os.path.abspath(args.ckpt))),
    ]
    decode_to_midi = load_decoder(hint_paths)

    # When a seed was used: decode only the generated tail, not the seed prompt.
    # BOS is re-prepended so the decoder gets a valid sequence header.
    decode_seq = [BOS_ID] + seq[seed_offset:] if args.seed_midi else seq
    if args.seed_midi:
        print(f"Seed stripped: decoding {len(decode_seq)} generated tokens (of {len(seq)} total)")

    # Resolve playback tempo: CLI > vocab median > 120 BPM fallback
    if args.tempo_bpm > 0:
        out_tempo = args.tempo_bpm
    else:
        out_tempo = float(vocab.get("median_tempo_bpm", 0) or 120.0)
    print(f"Output tempo: {out_tempo:.1f} BPM")

    try:
        decode_to_midi(decode_seq, args.vocab_json, args.out_midi, tempo_bpm=out_tempo)
    except TypeError:
        try:
            decode_to_midi(decode_seq, vocab, args.out_midi, tempo_bpm=out_tempo)
        except TypeError:
            decode_to_midi(decode_seq, vocab, args.out_midi)

    assert os.path.isfile(args.out_midi), f"Expected to write {args.out_midi} but file not found."

    # prune empty tracks + optionally drop non-requested tracks
    prune_midi_tracks(args.out_midi, args.tracks)

    set_gm_programs(args.out_midi)
    print(f"Wrote MIDI → {args.out_midi}")
    render_midi_to_wav(args.out_midi)

    return {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "seed": seed,
        "ckpt_path": args.ckpt,
        "ckpt_sha16": hash_file(args.ckpt),
        "vocab_json": args.vocab_json,
        "vocab_size": V,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "ctx": args.ctx,
        "grid_step_detected": grid_step,
        "grid_locked": grid_locked,
        "grid_detect_window": args.grid_detect_window,
        "grid_lock_tokens": args.grid_lock_tokens,
        "device": device,
        "out_midi": args.out_midi,
        "sequence_len": len(seq),
        "seed_midi": args.seed_midi or None,
        "seed_bars": args.seed_bars if args.seed_midi else None,
        "note": "TIME_SHIFT snapped to adaptive straight/triplet grid in 1/24-qn steps."
    }

def main():
    args = get_args()
    meta = generate(args)
    if args.out_meta_json:
        with open(args.out_meta_json, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"Wrote {args.out_meta_json}")

if __name__ == "__main__":
    main()