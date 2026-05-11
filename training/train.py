#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, json, math, pickle, random, time, argparse, atexit, signal
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
# ───────────────────────── AUTO-SCALING ─────────────────────────
# We keep SEQ_LEN fixed (default 512) and auto-scale model capacity (D_MODEL/N_HEADS/N_LAYERS)
# so that roughly: tokens / params ≈ target_tpp (tokens-per-parameter).
def _default_heads_for_dmodel(d_model: int) -> list[int]:
    # Choose head counts that give reasonable head_dim.
    table = {
        64:  [4],
        96:  [4, 6],
        128: [4, 8],
        160: [5, 8],
        192: [6, 8],
        256: [8, 16],
    }
    return table.get(int(d_model), [max(1, int(d_model) // 32)])

def estimate_transformer_params(vocab_size: int, d_model: int, n_layers: int, ff_mult: int) -> int:
    """Rough, monotonic estimate; used only to choose among configs."""
    d_ff = d_model * ff_mult
    emb = vocab_size * d_model

    # Per-layer params (approx):
    # - Self-attn: Q,K,V,Out projections ~ 4*d_model^2 (+ small biases)
    # - FFN: d_model->d_ff and d_ff->d_model ~ 2*d_model*d_ff (+ biases)
    # - Norms/bias overhead: small compared to matmuls
    per_layer = (4 * d_model * d_model) + (2 * d_model * d_ff) + (2 * d_ff) + (6 * d_model)
    core = n_layers * per_layer

    # Heads (type + value + aux) are relatively small; add a cushion.
    heads = int(0.20 * emb)

    return int(emb + core + heads)

def choose_auto_config(vocab_size: int, train_windows: int, seq_len: int, target_tpp: float, ff_mult: int,
                       min_params: int = 100_000, max_d_model: int = 256):
    tokens = int(train_windows) * int(seq_len)
    # Budget in params: params ≈ tokens / tpp
    budget = int(tokens / max(1e-6, float(target_tpp)))
    budget = max(int(min_params), budget)

    d_models = [64, 96, 128, 160, 192, 256]
    d_models = [d for d in d_models if d <= int(max_d_model)]
    layers_list = [2, 3, 4, 6]

    best = None  # (est_params, d_model, n_layers, n_heads)
    for d_model in d_models:
        for n_layers in layers_list:
            for n_heads in _default_heads_for_dmodel(d_model):
                if d_model % n_heads != 0:
                    continue
                est = estimate_transformer_params(vocab_size, d_model, n_layers, ff_mult)
                if est <= budget:
                    cand = (est, d_model, n_layers, n_heads)
                    if best is None or cand[0] > best[0]:
                        best = cand

    # If nothing fits, pick the smallest config.
    if best is None:
        d_model = d_models[0] if d_models else 64
        n_layers = 2
        n_heads = _default_heads_for_dmodel(d_model)[0]
        est = estimate_transformer_params(vocab_size, d_model, n_layers, ff_mult)
        best = (est, d_model, n_layers, n_heads)

    est, d_model, n_layers, n_heads = best
    info = {
        "tokens": tokens,
        "budget_params": budget,
        "est_params": est,
        "tpp_target": float(target_tpp),
        "tpp_est": float(tokens) / float(max(1, est)),
        "d_model": int(d_model),
        "n_layers": int(n_layers),
        "n_heads": int(n_heads),
        "ff_mult": int(ff_mult),
    }
    return info


# ─────────────────────────── CONFIG ───────────────────────────
DATA_DIR        = "data_events6"
TRAIN_PKL       = os.path.join(DATA_DIR, "events_train.pkl")
VAL_PKL         = os.path.join(DATA_DIR, "events_val.pkl")
VOCAB_JSON      = os.path.join(DATA_DIR, "event_vocab.json")

SAVE_PATH       = "esFullSummer_aux.pt"

# Model size (~1–1.5M params)
D_MODEL=192; N_HEADS=6; N_LAYERS=4; FF_MULT=3; DROPOUT=0.12

# Training
BATCH_SIZE      = 64
EPOCHS          = 200
SEQ_LEN         = 512
LR              = 2e-4
BETAS           = (0.9, 0.95)
WEIGHT_DECAY    = 1e-2
MAX_GRAD_NORM   = 1.0

# Label smoothing
LABEL_SMOOTH_TYPE   = 0.05   # for type classification
LABEL_SMOOTH_VALUE  = 0.04   # default for value heads
LABEL_SMOOTH_PER_TYPE = {
    "PITCH_GENERAL": 0.02,
    "PITCH_DRUMS":   0.02,
}
TOKEN_DROPOUT_P  = 0.07

# Loss mixing (token)
ALPHA_TYPE      = 0.2
ALPHA_VALUE     = 0.8

# Aux (polyphony instructor) head
AUX_ENABLED     = True
AUX_DIM_DEFAULT = 36
AUX_LOSS_WEIGHT = 0.05          # weight relative to token loss
AUX_HUBER_DELTA = 1.0           # robust to occasional big outliers

# Aux weighting inside aux vector (optional but recommended)
# aux layout: [max_poly(6), mean_poly(6), overlap(6), chord(4), pc_hist(12), swing(1), blues(1)] = 36
AUX_WEIGHTS = {
    "max_poly":   1.0,
    "mean_poly":  1.0,
    "overlap":    2.0,   # overlap ratio is in [0,1], often smaller gradients
    "chords":     1.0,
    "pc_hist":    1.0,
    "swing":      2.0,   # emphasize shuffle
    "blues":      2.0,   # emphasize blues scale adherence
}

SEED = 42

def pick_device(requested: str = "auto") -> torch.device:
    requested = (requested or "auto").lower()
    if requested in ("auto", "best"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda":
        return torch.device("cuda")
    if requested == "mps":
        return torch.device("mps")
    if requested == "cpu":
        return torch.device("cpu")
    raise ValueError("--device must be one of: auto,cuda,mps,cpu")


torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)

# ─────────────────────────── DATA ────────────────────────────
def load_vocab(vocab_path) -> Dict:
    with open(vocab_path, "r") as f:
        vocab = json.load(f)
    layout = vocab["layout"]

    type_names = [k for k in layout.keys() if k not in ("PAD","BOS","EOS")]
    head_sizes = [layout[k]["size"] for k in type_names]
    starts     = {k: layout[k]["start"] for k in layout}
    sizes      = {k: layout[k]["size"]  for k in layout}
    V = max(spec["start"] + spec["size"] for spec in layout.values())

    type_of_id  = np.full((V,), -1, dtype=np.int64)
    local_of_id = np.full((V,), -1, dtype=np.int64)
    name_to_type_idx = {nm: i for i, nm in enumerate(type_names)}
    for nm in type_names:
        s, n = layout[nm]["start"], layout[nm]["size"]
        t_idx = name_to_type_idx[nm]
        type_of_id[s:s+n]  = t_idx
        local_of_id[s:s+n] = np.arange(n, dtype=np.int64)

    aux_dim = AUX_DIM_DEFAULT
    if isinstance(vocab.get("aux"), dict) and "aux_dim" in vocab["aux"]:
        try:
            aux_dim = int(vocab["aux"]["aux_dim"])
        except Exception:
            aux_dim = AUX_DIM_DEFAULT

    info = {
        "V": V,
        "layout": layout,
        "type_names": type_names,
        "head_sizes": head_sizes,
        "starts": starts,
        "sizes": sizes,
        "type_of_id": torch.from_numpy(type_of_id),
        "local_of_id": torch.from_numpy(local_of_id),
        "aux_dim": aux_dim,
    }
    return vocab, info

class EventDataset(Dataset):
    def __init__(self, pkl_path: str, expect_aux: bool = True):
        with open(pkl_path, "rb") as f:
            obj = pickle.load(f)
        self.seqs: List[List[int]] = obj["sequences"]
        self.aux: Optional[List[np.ndarray]] = None
        if expect_aux and ("aux" in obj):
            self.aux = obj["aux"]
            if len(self.aux) != len(self.seqs):
                raise ValueError(f"aux length {len(self.aux)} != sequences length {len(self.seqs)} in {pkl_path}")

    def __len__(self): return len(self.seqs)

    def __getitem__(self, idx):
        x = torch.tensor(self.seqs[idx], dtype=torch.long)
        if self.aux is None:
            return x, None
        a = torch.tensor(np.asarray(self.aux[idx], dtype=np.float32), dtype=torch.float32)
        return x, a

def make_causal_mask(T: int, device):
    return torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)

def collate_random_crop(batch_seqs: List[torch.Tensor], pad_id: int, seq_len: int):
    out = []
    for seq in batch_seqs:
        L = seq.size(0)
        if L <= seq_len:
            padded = torch.full((seq_len,), pad_id, dtype=torch.long)
            padded[:L] = seq
            out.append(padded)
        else:
            start = random.randint(0, L - seq_len)
            out.append(seq[start:start+seq_len])
    return torch.stack(out, dim=0)  # (B,T)

# ────────────────────────── MODEL ────────────────────────────
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=8192):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):  # (B,T,D)
        return x + self.pe[:x.size(1)].unsqueeze(0)

def _aux_weight_vector(aux_dim: int, vocab: Optional[Dict] = None) -> torch.Tensor:
    """Build per-element weight vector for aux loss from vocab metadata.

    Reads the aux layout from vocab JSON to handle variable instrument counts.
    Falls back to uniform weights if layout isn't available.
    """
    w = torch.ones(aux_dim, dtype=torch.float32)

    # Try to parse aux layout from vocab
    aux_meta = None
    if vocab is not None and isinstance(vocab.get("aux"), dict):
        aux_meta = vocab["aux"]

    if aux_meta is None:
        return w

    # Determine instrument count from vocab
    inst_names = vocab.get("instrument_names", [])
    n_inst = len(inst_names) if inst_names else 6
    has_chords = aux_meta.get("has_chords", n_inst == 6)
    has_swing_blues = aux_meta.get("has_swing_blues", n_inst == 6)

    # Build weight vector based on dynamic layout
    idx = 0
    # max_poly[N]
    w[idx:idx+n_inst] = AUX_WEIGHTS.get("max_poly", 1.0)
    idx += n_inst
    # mean_poly[N]
    if idx + n_inst <= aux_dim:
        w[idx:idx+n_inst] = AUX_WEIGHTS.get("mean_poly", 1.0)
        idx += n_inst
    # overlap[N]
    if idx + n_inst <= aux_dim:
        w[idx:idx+n_inst] = AUX_WEIGHTS.get("overlap", 2.0)
        idx += n_inst
    # chord stats (4) — only if present
    if has_chords and idx + 4 <= aux_dim:
        w[idx:idx+4] = AUX_WEIGHTS.get("chords", 1.0)
        idx += 4
    # pc_hist (12)
    if idx + 12 <= aux_dim:
        w[idx:idx+12] = AUX_WEIGHTS.get("pc_hist", 1.0)
        idx += 12
    # swing + blues (2) — only if present
    if has_swing_blues and idx + 2 <= aux_dim:
        w[idx] = AUX_WEIGHTS.get("swing", 2.0)
        idx += 1
        w[idx] = AUX_WEIGHTS.get("blues", 2.0)
        idx += 1

    return w

class FactorizedESModel(nn.Module):
    """
    Single encoder → (type logits) + {value head per type} [+ optional aux head].
    """
    def __init__(self,
                 pad_id: int,
                 type_names: List[str],
                 head_sizes: List[int],
                 num_embeddings: int,
                 aux_dim: int = 0):
        super().__init__()
        self.pad_id     = pad_id
        self.type_names = type_names
        self.head_sizes = head_sizes
        self.num_types  = len(type_names)
        self.aux_dim    = int(aux_dim)

        self.tok_emb = nn.Embedding(num_embeddings, D_MODEL, padding_idx=pad_id)
        self.pos_emb = PositionalEmbedding(D_MODEL)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=N_HEADS, dim_feedforward=D_MODEL*FF_MULT,
            dropout=DROPOUT, activation="gelu", batch_first=True, norm_first=True
        )
        self.tr = nn.TransformerEncoder(enc_layer, num_layers=N_LAYERS)
        self.drop = nn.Dropout(DROPOUT)

        self.type_head   = nn.Linear(D_MODEL, self.num_types, bias=True)
        self.value_heads = nn.ModuleList([nn.Linear(D_MODEL, s, bias=True) for s in head_sizes])

        # Aux head predicts one vector per window; use final token hidden state by default.
        if self.aux_dim > 0:
            self.aux_head = nn.Sequential(
                nn.Linear(D_MODEL, D_MODEL),
                nn.GELU(),
                nn.Dropout(DROPOUT),
                nn.Linear(D_MODEL, self.aux_dim),
            )
        else:
            self.aux_head = None

        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)
        for m in list(self.value_heads) + [self.type_head]:
            nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        x: (B,T) global token ids.
        Returns:
          type_logits : (B,T,num_types)
          value_logits: list of (B,T,head_sizes[t])
          aux_pred    : (B,aux_dim) or None
        """
        B, T = x.shape
        h = self.tok_emb(x)
        h = self.pos_emb(h)
        h = self.drop(h)

        attn_mask = make_causal_mask(T, x.device)
        pad_mask  = (x == self.pad_id)
        h = self.tr(h, mask=attn_mask, src_key_padding_mask=pad_mask)

        type_logits  = self.type_head(h)
        value_logits = [head(h) for head in self.value_heads]

        aux_pred = None
        if self.aux_head is not None:
            # Use representation at last position (could also use mean pooling over non-pad)
            aux_pred = self.aux_head(h[:, -1, :])

        return type_logits, value_logits, aux_pred

# Backward-compat shim
def build_model(vocab_size: int, pad_id: int):
    with open(VOCAB_JSON, "r") as f:
        vocab = json.load(f)
    layout = vocab["layout"]
    type_names = [k for k in layout.keys() if k not in ("PAD","BOS","EOS")]
    head_sizes = [layout[k]["size"] for k in type_names]
    V = max(spec["start"] + spec["size"] for spec in layout.values())
    aux_dim = int(vocab.get("aux", {}).get("aux_dim", 0)) if isinstance(vocab.get("aux"), dict) else 0
    return FactorizedESModel(pad_id=pad_id, type_names=type_names, head_sizes=head_sizes, num_embeddings=V, aux_dim=aux_dim)

# ─────────────────────── TRAIN UTILS ─────────────────────────
@dataclass
class Batch:
    x: torch.Tensor          # (B,T)
    y: torch.Tensor          # (B,T)
    aux: Optional[torch.Tensor]  # (B,aux_dim) or None

def token_dropout_(x: torch.Tensor, p: float, protected_ids: List[int], replace_id: int):
    if p <= 0:
        return
    mask = torch.rand_like(x, dtype=torch.float32) < p
    for pid in protected_ids:
        mask &= (x != pid)
    x[mask] = replace_id


def collate_with_aux(batch, *, pad_id: int, bos_id: int, eos_id: int, layout: Dict, seq_len: int,
                     aux_dim_used: int, token_dropout_p: float, replace_id: int) -> "Batch":
    """Picklable DataLoader collate_fn (macOS spawn safe).

    batch: list of (seq_tensor, aux_tensor|None)
    Returns: Batch(x, y, aux)
    """
    seqs = [b[0] for b in batch]
    auxs = [b[1] for b in batch]

    x = collate_random_crop(seqs, pad_id, seq_len)   # (B,T)

    # next-token targets
    y = x.clone()
    y[:, :-1] = x[:, 1:]
    y[:, -1]  = pad_id

    aux_batch = None
    if aux_dim_used > 0 and auxs and (auxs[0] is not None):
        aux_batch = torch.stack(auxs, dim=0)  # (B,aux_dim)

    protected = [pad_id, bos_id, eos_id]
    for nm in ["BAR", "BEAT", "INST", "TEMPO"]:
        if nm in layout:
            protected.append(layout[nm]["start"])

    token_dropout_(x, token_dropout_p, protected_ids=protected, replace_id=replace_id)
    return Batch(x=x, y=y, aux=aux_batch)


def reconstruct_global_ids(pred_type: torch.Tensor,
                           pred_local: torch.Tensor,
                           starts: Dict[str,int],
                           type_names: List[str]) -> torch.Tensor:
    out = torch.zeros_like(pred_type)
    for t_idx, nm in enumerate(type_names):
        s = starts[nm]
        sel = (pred_type == t_idx)
        if sel.any():
            out[sel] = s + pred_local[sel]
    return out

class WarmupCosine:
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-6, base_lr=LR):
        self.opt = optimizer
        self.warmup = max(1, warmup_steps)
        self.total = max(self.warmup+1, total_steps)
        self.min_lr = min_lr
        self.base_lr = base_lr
        self.step_num = 0

    def step(self):
        self.step_num += 1
        if self.step_num <= self.warmup:
            lr = self.base_lr * self.step_num / self.warmup
        else:
            t = (self.step_num - self.warmup) / max(1, self.total - self.warmup)
            lr = self.min_lr + 0.5*(self.base_lr - self.min_lr)*(1 + math.cos(math.pi * t))
        for g in self.opt.param_groups:
            g['lr'] = lr

def weighted_huber(pred: torch.Tensor, target: torch.Tensor, weight_vec: torch.Tensor, delta: float = 1.0):
    """
    pred/target: (B,D), weight_vec: (D,)
    """
    # huber per element then weighted mean
    err = pred - target
    abs_err = err.abs()
    quad = torch.minimum(abs_err, torch.tensor(delta, device=pred.device, dtype=pred.dtype))
    lin  = abs_err - quad
    hub = 0.5 * quad * quad + delta * lin  # (B,D)
    w = weight_vec.to(pred.device).view(1, -1)
    return (hub * w).mean()

# ─────────────────────────── MAIN ────────────────────────────
def main():
    global DATA_DIR, TRAIN_PKL, VAL_PKL, VOCAB_JSON, SAVE_PATH, SEQ_LEN, D_MODEL, N_HEADS, N_LAYERS, FF_MULT
    ap = argparse.ArgumentParser("trainES4: train factored event-stream Transformer.")
    ap.add_argument("--data_dir", default=DATA_DIR)
    ap.add_argument("--train_pkl", default=TRAIN_PKL)
    ap.add_argument("--val_pkl", default=VAL_PKL)
    ap.add_argument("--vocab_json", default=VOCAB_JSON)
    ap.add_argument("--save_path", default=SAVE_PATH)
    ap.add_argument("--device", default="auto", help="auto|cuda|mps|cpu (auto picks cuda then mps then cpu)")
    ap.add_argument("--num_workers", type=int, default=None, help="DataLoader workers. Default: 2 on CUDA, 0 otherwise (macOS spawn safe).")
    ap.add_argument("--auto_scale", action="store_true", default=True, help="Auto-scale model size from train_windows (keeps --seq_len fixed).")
    ap.add_argument("--no_auto_scale", action="store_true", default=False, help="Disable auto-scaling (use defaults or manual overrides).")
    ap.add_argument("--target_tpp", type=float, default=8.0, help="Target tokens-per-parameter for auto_scale (7–10 is a good range).")
    ap.add_argument("--max_d_model", type=int, default=256, help="Max d_model considered by auto_scale.")
    ap.add_argument("--min_params", type=int, default=100000, help="Minimum parameter budget when auto_scale is enabled.")
    ap.add_argument("--patience", type=int, default=10, help="Early stop if val loss does not improve for this many epochs (0 disables).")
    ap.add_argument("--min_delta", type=float, default=1e-4, help="Minimum val-loss improvement to count as improvement (for patience reset).")
    ap.add_argument("--reset_best_val", action="store_true", help="Ignore any existing checkpoint's val loss when deciding whether to save (used with force_restart).")
    ap.add_argument("--d_model", type=int, default=None, help="Manual override d_model (disables auto_scale if set).")
    ap.add_argument("--n_layers", type=int, default=None, help="Manual override number of Transformer layers (disables auto_scale if set).")
    ap.add_argument("--n_heads", type=int, default=None, help="Manual override attention heads (disables auto_scale if set).")
    ap.add_argument("--ff_mult", type=int, default=None, help="Manual override FFN multiplier (dim_feedforward = d_model * ff_mult).")
    ap.add_argument("--seq_len", type=int, default=SEQ_LEN, help="Sequence length (keep at 512 unless you have a reason).")
    ap.add_argument("--resume", default=None, help="Path to checkpoint .pt to resume training from (restores model, optimizer, epoch, best_val).")
    args = ap.parse_args()

    DATA_DIR   = args.data_dir
    TRAIN_PKL  = args.train_pkl
    VAL_PKL    = args.val_pkl
    VOCAB_JSON = args.vocab_json
    SAVE_PATH  = args.save_path

    # ── single-instance lock ──────────────────────────────────
    lock_path = SAVE_PATH + ".lock"
    if os.path.exists(lock_path):
        try:
            other_pid = int(open(lock_path).read().strip())
            os.kill(other_pid, 0)  # check if alive
            print(f"ERROR: another train.py (PID {other_pid}) is already "
                  f"writing to {SAVE_PATH}. Kill it first or use a different --save_path.")
            sys.exit(1)
        except (ProcessLookupError, ValueError):
            pass  # stale lock — previous run died
        except PermissionError:
            # process exists but we can't signal it (different user)
            print(f"ERROR: another train.py is already writing to {SAVE_PATH} (lock PID in {lock_path}).")
            sys.exit(1)
    os.makedirs(os.path.dirname(lock_path) or ".", exist_ok=True)
    with open(lock_path, "w") as f:
        f.write(str(os.getpid()))

    def _remove_lock():
        try:
            if os.path.exists(lock_path) and open(lock_path).read().strip() == str(os.getpid()):
                os.remove(lock_path)
        except OSError:
            pass

    atexit.register(_remove_lock)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

    # nothing to clean up — single-file overwrite strategy

    # Allow seq_len override (default 512).
    SEQ_LEN = int(args.seq_len)

    device = pick_device(args.device)
    if device.type == "cuda":
        print(f"Using device: cuda ({torch.cuda.get_device_name(0)})")
    else:
        print(f"Using device: {device.type}")
    # ── vocab & layout ─────────────────────────────────────────
    vocab, vinfo = load_vocab(VOCAB_JSON)
    V           = vinfo["V"]
    layout      = vinfo["layout"]
    type_names  = vinfo["type_names"]
    head_sizes  = vinfo["head_sizes"]
    starts      = vinfo["starts"]
    type_of_id  = vinfo["type_of_id"].to(device)
    local_of_id = vinfo["local_of_id"].to(device)
    aux_dim     = int(vinfo.get("aux_dim", 0))

    PAD_ID = layout["PAD"]["start"]
    BOS_ID = layout["BOS"]["start"]
    EOS_ID = layout["EOS"]["start"]

    # benign replacement for token dropout = smallest TIME_SHIFT
    TIME_SHIFT_START = layout["TIME_SHIFT"]["start"]
    TIME_SHIFT_REPL  = TIME_SHIFT_START + 0

    # ── datasets & loaders ─────────────────────────────────────
    expect_aux = AUX_ENABLED
    train_ds = EventDataset(TRAIN_PKL, expect_aux=expect_aux)
    # ── model auto-scaling (capacity vs data) ────────────────────
    # NOTE: This only changes model hyperparams, not your tokenization.
    auto_scale = bool(args.auto_scale) and (not bool(args.no_auto_scale))

    # Manual overrides disable auto-scale.
    if args.ff_mult is not None:
        FF_MULT = int(args.ff_mult)
    if args.d_model is not None or args.n_layers is not None or args.n_heads is not None:
        auto_scale = False
        if args.d_model is not None:
            D_MODEL = int(args.d_model)
        if args.n_layers is not None:
            N_LAYERS = int(args.n_layers)
        if args.n_heads is not None:
            N_HEADS = int(args.n_heads)

    if auto_scale:
        info = choose_auto_config(
            vocab_size=int(V),
            train_windows=len(train_ds),
            seq_len=int(SEQ_LEN),
            target_tpp=float(args.target_tpp),
            ff_mult=int(FF_MULT),
            min_params=int(args.min_params),
            max_d_model=int(args.max_d_model),
        )
        D_MODEL  = int(info["d_model"])
        N_LAYERS = int(info["n_layers"])
        N_HEADS  = int(info["n_heads"])
        FF_MULT  = int(info["ff_mult"])
        print(
            f"Auto-scale: windows={len(train_ds)} seq_len={SEQ_LEN} → tokens≈{info['tokens']:,} | "
            f"budget≈{info['budget_params']:,} params | chosen d_model={D_MODEL} n_layers={N_LAYERS} n_heads={N_HEADS} ff_mult={FF_MULT} | "
            f"est_params≈{info['est_params']:,} (tpp≈{info['tpp_est']:.2f}, target={info['tpp_target']:.2f})"
        )
    else:
        print(f"Model config: d_model={D_MODEL} n_layers={N_LAYERS} n_heads={N_HEADS} ff_mult={FF_MULT} (auto_scale={'on' if args.auto_scale else 'off'}, no_auto_scale={args.no_auto_scale})")
    val_ds   = EventDataset(VAL_PKL,   expect_aux=expect_aux)

    aux_dim_used = aux_dim if (AUX_ENABLED and aux_dim > 0 and train_ds.aux is not None) else 0
    aux_wvec = _aux_weight_vector(aux_dim_used, vocab) if aux_dim_used > 0 else None


    # DataLoader settings (macOS uses spawn; keep collate_fn picklable)
    if args.num_workers is None:
        # partial() is picklable so spawn-safe on macOS; use workers for cuda and mps
        num_workers = 2 if device.type in ("cuda", "mps") else 0
    else:
        num_workers = int(args.num_workers)

    collate_fn = partial(
        collate_with_aux,
        pad_id=PAD_ID,
        bos_id=BOS_ID,
        eos_id=EOS_ID,
        layout=layout,
        seq_len=SEQ_LEN,
        aux_dim_used=aux_dim_used,
        token_dropout_p=TOKEN_DROPOUT_P,
        replace_id=TIME_SHIFT_REPL,
    )



    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=(device.type == "cuda"),
        num_workers=num_workers,
        collate_fn=collate_fn,
        persistent_workers=(num_workers > 0),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
        num_workers=num_workers,
        collate_fn=collate_fn,
        persistent_workers=(num_workers > 0),
    )

    # ── model ──────────────────────────────────────────────────
    model = FactorizedESModel(
        pad_id=PAD_ID,
        type_names=type_names,
        head_sizes=head_sizes,
        num_embeddings=V,
        aux_dim=aux_dim_used
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Using device: {device} | Trainable params: {n_params/1e6:.2f}M | Types={len(type_names)} | Vocab≈{V} | aux_dim={aux_dim_used}")

    # ── optimizer + sched ──────────────────────────────────────
    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=BETAS, weight_decay=WEIGHT_DECAY)
    steps_per_epoch = max(1, len(train_loader))
    total_steps = EPOCHS * steps_per_epoch
    warmup_steps = max(10, int(0.02 * total_steps))
    sched = WarmupCosine(opt, warmup_steps=warmup_steps, total_steps=total_steps, min_lr=1e-6, base_lr=LR)

    best_val = float('inf'); best_epoch = -1
    epochs_no_improve = 0
    start_epoch = 1

    # ── checkpoint guard: when starting fresh, don't clobber a better existing checkpoint ──
    if not args.resume and not args.reset_best_val and os.path.isfile(args.save_path):
        try:
            _existing = torch.load(args.save_path, map_location="cpu", weights_only=False)
            _floor = _existing.get("best_val", float('inf'))
            if _floor < float('inf'):
                best_val = _floor
                print(f"[guard] existing checkpoint has best_val={best_val:.4f} -- new training must beat this to save")
        except Exception:
            pass  # corrupted or unreadable — start with inf, let training proceed

    # ── resume from checkpoint ─────────────────────────────────
    if args.resume:
        if not os.path.isfile(args.resume):
            print(f"ERROR: --resume path not found: {args.resume}", file=sys.stderr)
            sys.exit(1)
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        opt.load_state_dict(ckpt["optimizer_state_dict"])
        resumed_epoch = ckpt.get("epoch", 0)
        start_epoch = resumed_epoch + 1
        best_val = ckpt.get("best_val", float('inf'))
        best_epoch = ckpt.get("best_epoch", resumed_epoch)
        # Fast-forward scheduler to the right step
        sched.step_num = resumed_epoch * steps_per_epoch
        print(f"Resumed from {args.resume} (epoch {resumed_epoch}, best_val={best_val:.4f}, best_epoch={best_epoch})")

    # ── epoch loop helpers ─────────────────────────────────────
    def run_epoch(loader, split: str, report_progress: bool = False,
                  compute_accuracy: bool = False):
        is_train = (split == "train")
        model.train(is_train)

        sum_loss_tok   = torch.tensor(0.0, device=device)
        sum_tloss_tok  = torch.tensor(0.0, device=device)
        sum_vloss_tok  = torch.tensor(0.0, device=device)
        sum_n_tokens   = torch.tensor(0,   device=device, dtype=torch.long)
        correct_exact  = 0
        correct_type   = 0
        correct_value  = 0

        # aux metrics
        aux_count = 0
        sum_aux_loss = 0.0
        sum_aux_mae  = 0.0

        total_batches = len(loader)
        report_every  = max(1, total_batches // 20)  # emit ~every 5%
        batch_idx     = 0

        for batch in loader:
            x = batch.x.to(device)   # (B,T)
            y = batch.y.to(device)   # (B,T)
            aux = batch.aux.to(device) if (batch.aux is not None) else None

            id_is_typed = (y != PAD_ID) & (y != BOS_ID) & (y != EOS_ID)
            mask_flat   = id_is_typed.reshape(-1)

            y_type   = type_of_id[y]
            y_local  = local_of_id[y]
            y_flat   = y.reshape(-1)
            y_type_f  = y_type.reshape(-1)[mask_flat]
            y_local_f = y_local.reshape(-1)[mask_flat]

            with torch.set_grad_enabled(is_train):
                type_logits, value_logits, aux_pred = model(x)

                # TYPE loss
                tlog = type_logits.reshape(-1, len(type_names))[mask_flat]
                type_loss = F.cross_entropy(tlog, y_type_f, label_smoothing=LABEL_SMOOTH_TYPE)

                # VALUE loss per true-type
                val_loss_acc = torch.tensor(0.0, device=device)
                val_count    = torch.tensor(0,   device=device, dtype=torch.long)
                y_type_flat  = y_type.reshape(-1)
                y_local_flat = y_local.reshape(-1)

                for t_idx, head in enumerate(value_logits):
                    h = head.reshape(-1, head.size(-1))
                    sel = (y_type_flat == t_idx) & mask_flat
                    n_sel = sel.sum()
                    if n_sel.item() > 0:
                        tname = type_names[t_idx]
                        ls = LABEL_SMOOTH_PER_TYPE.get(tname, LABEL_SMOOTH_VALUE)
                        ce = F.cross_entropy(h[sel], y_local_flat[sel], label_smoothing=ls)
                        val_loss_acc = val_loss_acc + ce * n_sel
                        val_count    = val_count + n_sel

                val_loss   = val_loss_acc / val_count.clamp(min=1)
                token_loss = ALPHA_TYPE * type_loss + ALPHA_VALUE * val_loss

                aux_loss = torch.tensor(0.0, device=device)
                if aux_dim_used > 0 and aux is not None and aux_pred is not None:
                    aux_loss = weighted_huber(aux_pred, aux, aux_wvec, delta=AUX_HUBER_DELTA)

                loss = token_loss + (AUX_LOSS_WEIGHT * aux_loss)

                if is_train:
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                    opt.step()
                    sched.step()

            # ── accumulate loss (stay on GPU to avoid sync) ───────
            n_tok = mask_flat.sum()
            sum_n_tokens  = sum_n_tokens  + n_tok
            sum_loss_tok  = sum_loss_tok  + loss.detach()       * n_tok
            sum_tloss_tok = sum_tloss_tok + type_loss.detach()  * n_tok
            sum_vloss_tok = sum_vloss_tok + val_loss.detach()   * n_tok

            # ── accuracy metrics (val only, or when explicitly requested) ─
            if compute_accuracy:
                with torch.no_grad():
                    pred_type = type_logits.argmax(dim=-1)
                    correct_type += (pred_type.reshape(-1)[mask_flat] == y_type_f).sum().item()

                    pred_local = torch.zeros_like(pred_type)
                    for t_idx, head in enumerate(value_logits):
                        sel = (pred_type == t_idx)
                        if sel.any():
                            pred_local[sel] = head.argmax(dim=-1)[sel]

                    pred_global = reconstruct_global_ids(pred_type, pred_local, starts, type_names)
                    correct_exact += (pred_global.reshape(-1)[mask_flat] == y_flat[mask_flat]).sum().item()

                    for t_idx, head in enumerate(value_logits):
                        sel_true = ((y_type == t_idx) & id_is_typed)
                        if sel_true.any():
                            correct_value += (head.argmax(dim=-1)[sel_true] == y_local[sel_true]).sum().item()

            batch_idx += 1
            if report_progress and batch_idx % report_every == 0:
                print(f"BATCH_PROGRESS {batch_idx}/{total_batches}", flush=True)

            # ── aux metrics ─────────────────────────────────────
            if aux_dim_used > 0 and aux is not None and aux_pred is not None:
                B_size = aux.size(0)
                aux_count    += B_size
                sum_aux_loss += aux_loss.item() * B_size
                sum_aux_mae  += (aux_pred.detach() - aux).abs().mean().item() * B_size

        tot_tok = sum_n_tokens.item()
        avg_loss  = (sum_loss_tok  / sum_n_tokens.clamp(min=1)).item()
        avg_tloss = (sum_tloss_tok / sum_n_tokens.clamp(min=1)).item()
        avg_vloss = (sum_vloss_tok / sum_n_tokens.clamp(min=1)).item()

        ppl = math.exp(min(20.0, avg_loss))
        acc_exact = correct_exact / max(1, tot_tok)
        acc_type  = correct_type  / max(1, tot_tok)
        acc_value = correct_value / max(1, tot_tok)

        avg_aux_loss = (sum_aux_loss / aux_count) if aux_count > 0 else 0.0
        avg_aux_mae  = (sum_aux_mae  / aux_count) if aux_count > 0 else 0.0

        return avg_loss, ppl, acc_exact, avg_tloss, avg_vloss, acc_type, acc_value, avg_aux_loss, avg_aux_mae

    # ── main training loop ─────────────────────────────────────
    for epoch in range(start_epoch, EPOCHS+1):
        t0 = time.time()
        tr = run_epoch(train_loader, "train", report_progress=True, compute_accuracy=False)
        va = run_epoch(val_loader, "val", compute_accuracy=True)
        dt = time.time() - t0

        (tr_loss, tr_ppl, tr_acc, tr_tloss, tr_vloss, tr_tacc, tr_vacc, tr_aux_l, tr_aux_mae) = tr
        (va_loss, va_ppl, va_acc, va_tloss, va_vloss, va_tacc, va_vacc, va_aux_l, va_aux_mae) = va

        msg = (f"Epoch {epoch:03d} | "
               f"train: loss={tr_loss:.3f} ppl={tr_ppl:.2f} acc={tr_acc:.3f} "
               f"(type_acc={tr_tacc:.3f}, val_acc={tr_vacc:.3f})")

        if aux_dim_used > 0:
            msg += f" aux_loss={tr_aux_l:.4f} aux_mae={tr_aux_mae:.4f}"

        msg += (f" | val: loss={va_loss:.3f} ppl={va_ppl:.2f} acc={va_acc:.3f} "
                f"(type_acc={va_tacc:.3f}, val_acc={va_vacc:.3f})")

        if aux_dim_used > 0:
            msg += f" aux_loss={va_aux_l:.4f} aux_mae={va_aux_mae:.4f}"

        msg += f" [{dt:.1f}s]"

        improved = (best_val - va_loss) > args.min_delta
        stop_now = False

        if improved:
            best_val = va_loss
            best_epoch = epoch
            epochs_no_improve = 0

            # ── overwrite single best checkpoint ──────────────────
            ckpt_payload = {
                "epoch": epoch,
                "best_val": best_val,
                "best_epoch": best_epoch,
                "model_state_dict": model.state_dict(),
                "model_state": model.state_dict(),  # alias
                "optimizer_state_dict": opt.state_dict(),
                "factored_meta": {
                    "type_names": type_names,
                    "head_sizes": head_sizes,
                    "starts": starts,
                    "ALPHA_TYPE": ALPHA_TYPE,
                    "ALPHA_VALUE": ALPHA_VALUE,
                    "AUX_ENABLED": aux_dim_used > 0,
                    "AUX_DIM": aux_dim_used,
                    "AUX_LOSS_WEIGHT": AUX_LOSS_WEIGHT,
                },
                "vocab_info": {
                    "PAD_ID": PAD_ID, "BOS_ID": BOS_ID, "EOS_ID": EOS_ID,
                    "VOCAB_JSON": VOCAB_JSON
                },
                "config": {
                    "D_MODEL": D_MODEL, "N_HEADS": N_HEADS, "N_LAYERS": N_LAYERS,
                    "FF_MULT": FF_MULT, "DROPOUT": DROPOUT, "SEQ_LEN": SEQ_LEN,
                    "DATA_DIR": DATA_DIR,
                },
                "model_config": {
                    "D_MODEL": D_MODEL, "N_HEADS": N_HEADS, "N_LAYERS": N_LAYERS,
                    "FF_MULT": FF_MULT, "DROPOUT": DROPOUT, "SEQ_LEN": SEQ_LEN,
                    "DATA_DIR": DATA_DIR,
                    "PAD_ID": PAD_ID, "BOS_ID": BOS_ID, "EOS_ID": EOS_ID,
                }
            }
            tmp_path = SAVE_PATH + ".tmp"
            torch.save(ckpt_payload, tmp_path)
            os.replace(tmp_path, SAVE_PATH)

            msg += f"  → Saved best (epoch {epoch}, val {va_loss:.4f}) at {time.strftime('%H:%M:%S')}"
        else:
            epochs_no_improve += 1

        if args.patience > 0 and epochs_no_improve >= args.patience:
            stop_now = True
            msg += (f"  → Early stop (patience={args.patience}, "
                    f"best_epoch={best_epoch}, best_val={best_val:.3f})")
        print(msg)
        if stop_now:
            break


    print(f"Done. Best epoch {best_epoch} | best val loss {best_val:.3f} | saved → {os.path.realpath(SAVE_PATH)}")

if __name__ == "__main__":
    main()