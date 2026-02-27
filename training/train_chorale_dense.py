#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Training loop for the dense chorale Transformer.

Single F.cross_entropy over 100 tokens (label_smoothing=0.1).
AdamW, warmup cosine schedule, early stopping.
Per-voice accuracy reporting (chord, soprano, bass, alto, tenor).

Reuses pick_device and WarmupCosine from training.train.
"""

import argparse
import atexit
import glob as globmod
import json
import math
import os
import pickle
import random
import signal
import sys
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from training.train import pick_device, WarmupCosine
from training.model_chorale_dense import ChoraleDenseModel
from training.pre_chorale_dense import PAD_ID, BOS_ID, EOS_ID, VOCAB_SIZE

# ────────────────────── DEFAULTS ──────────────────────
D_MODEL  = 128
N_HEADS  = 4
N_LAYERS = 4
FF_MULT  = 3
DROPOUT  = 0.15
BATCH_SIZE = 32
LR       = 3e-4
BETAS    = (0.9, 0.98)
WEIGHT_DECAY = 0.01
EPOCHS   = 300
PATIENCE = 30
LABEL_SMOOTHING = 0.1
TOKEN_DROPOUT_P = 0.05
MAX_GRAD_NORM   = 1.0
SEED     = 42

# ────────────────────── DATASET ──────────────────────

class ChoraleDataset(Dataset):
    """Dense chorale dataset.  Each item is a dict with 'tokens' and 'conts'."""

    def __init__(self, pkl_path: str):
        if not os.path.isfile(pkl_path):
            print(f"ERROR: dataset not found: {pkl_path}", file=sys.stderr)
            sys.exit(1)
        with open(pkl_path, "rb") as f:
            self.data: list[dict] = pickle.load(f)
        print(f"Loaded {len(self.data)} sequences from {pkl_path}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        d = self.data[idx]
        tokens = torch.tensor(d["tokens"], dtype=torch.long)
        conts = torch.tensor(d["conts"], dtype=torch.long)
        return tokens, conts


# ────────────────────── COLLATE ──────────────────────

@dataclass
class Batch:
    x: torch.Tensor       # (B, T) input tokens
    y: torch.Tensor       # (B, T) target tokens (shifted)
    cont: torch.Tensor    # (B, T) continuation counters for x

def collate_dense(batch: list[tuple[torch.Tensor, torch.Tensor]],
                  pad_id: int, token_dropout_p: float) -> Batch:
    """Pad sequences to max length in batch, create next-token targets."""
    tokens_list = [b[0] for b in batch]
    conts_list = [b[1] for b in batch]

    max_len = max(t.size(0) for t in tokens_list)

    x_batch = []
    cont_batch = []
    for tok, cont in zip(tokens_list, conts_list):
        L = tok.size(0)
        if L < max_len:
            tok = torch.cat([tok, torch.full((max_len - L,), pad_id, dtype=torch.long)])
            cont = torch.cat([cont, torch.zeros(max_len - L, dtype=torch.long)])
        x_batch.append(tok)
        cont_batch.append(cont)

    x = torch.stack(x_batch, dim=0)      # (B, T)
    cont = torch.stack(cont_batch, dim=0)  # (B, T)

    # Next-token targets
    y = x.clone()
    y[:, :-1] = x[:, 1:]
    y[:, -1] = pad_id

    # Token dropout on x (not on PAD/BOS/EOS)
    if token_dropout_p > 0:
        mask = torch.rand_like(x, dtype=torch.float32) < token_dropout_p
        for pid in [pad_id, BOS_ID, EOS_ID]:
            mask &= (x != pid)
        # Replace with a random pitch token (tokens 3-49)
        replacement = torch.randint(3, 50, x.shape, dtype=torch.long)
        x[mask] = replacement[mask]

    return Batch(x=x, y=y, cont=cont)


# ────────────────────── VOICE POSITION HELPERS ──────────────────────
# After BOS, each timestep is 5 tokens: chord, soprano, bass, alto, tenor
# Position within timestep: (pos - 1) % 5
#   0 = chord, 1 = soprano, 2 = bass, 3 = alto, 4 = tenor

VOICE_NAMES = ["chord", "soprano", "bass", "alto", "tenor"]

def position_to_voice(pos: int) -> str:
    """Map a token position (0-based) to voice name."""
    if pos == 0:
        return "BOS"
    adj = pos - 1
    return VOICE_NAMES[adj % 5]


# ────────────────────── MAIN ──────────────────────

def main():
    ap = argparse.ArgumentParser(description="Train dense chorale Transformer.")
    ap.add_argument("--data_dir", default="runs/chorale_dense_events")
    ap.add_argument("--train_pkl", default=None)
    ap.add_argument("--val_pkl", default=None)
    ap.add_argument("--vocab_json", default=None)
    ap.add_argument("--save_path", default="runs/checkpoints/chorale_dense_model.pt")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--d_model", type=int, default=None)
    ap.add_argument("--n_heads", type=int, default=None)
    ap.add_argument("--n_layers", type=int, default=None)
    ap.add_argument("--ff_mult", type=int, default=None)
    ap.add_argument("--dropout", type=float, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--label_smoothing", type=float, default=None)
    ap.add_argument("--token_dropout", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--keep_top_k", type=int, default=5)
    ap.add_argument("--min_delta", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=None)
    args = ap.parse_args()

    # Resolve paths
    DATA_DIR = args.data_dir
    TRAIN_PKL = args.train_pkl or os.path.join(DATA_DIR, "dense_train.pkl")
    VAL_PKL = args.val_pkl or os.path.join(DATA_DIR, "dense_val.pkl")
    VOCAB_JSON = args.vocab_json or os.path.join(DATA_DIR, "dense_vocab.json")
    SAVE_PATH = args.save_path

    # Hyperparams (CLI overrides)
    d_model = args.d_model or D_MODEL
    n_heads = args.n_heads or N_HEADS
    n_layers = args.n_layers or N_LAYERS
    ff_mult = args.ff_mult or FF_MULT
    dropout = args.dropout if args.dropout is not None else DROPOUT
    batch_size = args.batch_size or BATCH_SIZE
    lr = args.lr or LR
    epochs = args.epochs or EPOCHS
    patience = args.patience if args.patience is not None else PATIENCE
    label_smoothing = args.label_smoothing if args.label_smoothing is not None else LABEL_SMOOTHING
    token_dropout_p = args.token_dropout if args.token_dropout is not None else TOKEN_DROPOUT_P
    seed = args.seed if args.seed is not None else SEED

    # Seed
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = pick_device(args.device)
    print(f"Device: {device}")

    # ── Load vocab ──
    if not os.path.isfile(VOCAB_JSON):
        print(f"ERROR: vocab not found: {VOCAB_JSON}", file=sys.stderr)
        sys.exit(1)
    with open(VOCAB_JSON, "r") as f:
        vocab = json.load(f)
    vocab_size = vocab["vocab_size"]
    print(f"Vocab: {vocab_size} tokens")

    # ── Lock file ──
    os.makedirs(os.path.dirname(SAVE_PATH) or ".", exist_ok=True)
    lock_path = SAVE_PATH + ".lock"
    if os.path.exists(lock_path):
        try:
            other_pid = int(open(lock_path).read().strip())
            os.kill(other_pid, 0)
            print(f"ERROR: another training process (PID {other_pid}) is already "
                  f"writing to {SAVE_PATH}. Kill it first or use a different --save_path.")
            sys.exit(1)
        except (ProcessLookupError, ValueError):
            pass
        except PermissionError:
            print(f"ERROR: another training process is already writing to {SAVE_PATH}.")
            sys.exit(1)
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

    # ── Clean stale checkpoints ──
    if not args.resume:
        base, ext = os.path.splitext(SAVE_PATH)
        stale = globmod.glob(f"{base}_epoch*{ext}")
        if stale:
            print(f"Removing {len(stale)} stale checkpoint(s) from previous run")
            for f in stale:
                os.remove(f)
            if os.path.islink(SAVE_PATH):
                os.remove(SAVE_PATH)

    # ── Datasets ──
    train_ds = ChoraleDataset(TRAIN_PKL)
    val_ds = ChoraleDataset(VAL_PKL)

    if args.num_workers is None:
        num_workers = 2 if device.type == "cuda" else 0
    else:
        num_workers = int(args.num_workers)

    from functools import partial
    collate_fn = partial(collate_dense, pad_id=PAD_ID, token_dropout_p=token_dropout_p)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        pin_memory=(device.type == "cuda"),
        num_workers=num_workers, collate_fn=collate_fn,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        pin_memory=(device.type == "cuda"),
        num_workers=num_workers,
        collate_fn=partial(collate_dense, pad_id=PAD_ID, token_dropout_p=0.0),
        persistent_workers=(num_workers > 0),
    )

    # ── Model ──
    model = ChoraleDenseModel(
        vocab_size=vocab_size,
        pad_id=PAD_ID,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        ff_mult=ff_mult,
        dropout=dropout,
    ).to(device)

    n_params = model.count_parameters()
    print(f"Model: d_model={d_model} n_heads={n_heads} n_layers={n_layers} "
          f"ff_mult={ff_mult} dropout={dropout} | {n_params/1e3:.1f}K params")

    # ── Optimizer + scheduler ──
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=BETAS, weight_decay=WEIGHT_DECAY)
    steps_per_epoch = max(1, len(train_loader))
    total_steps = epochs * steps_per_epoch
    warmup_steps = max(10, int(0.02 * total_steps))
    sched = WarmupCosine(opt, warmup_steps=warmup_steps, total_steps=total_steps,
                         min_lr=1e-6, base_lr=lr)

    best_val = float("inf")
    best_epoch = -1
    epochs_no_improve = 0
    start_epoch = 1

    # ── Resume ──
    if args.resume:
        if not os.path.isfile(args.resume):
            print(f"ERROR: --resume path not found: {args.resume}", file=sys.stderr)
            sys.exit(1)
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        opt.load_state_dict(ckpt["optimizer_state_dict"])
        resumed_epoch = ckpt.get("epoch", 0)
        start_epoch = resumed_epoch + 1
        best_val = ckpt.get("best_val", float("inf"))
        best_epoch = ckpt.get("best_epoch", resumed_epoch)
        sched.step_num = resumed_epoch * steps_per_epoch
        print(f"Resumed from {args.resume} (epoch {resumed_epoch}, best_val={best_val:.4f})")

    # ── Per-voice position mapping (vectorized) ──
    # After BOS (pos 0), each timestep is 5 tokens: chord(0), soprano(1), bass(2), alto(3), tenor(4)
    # voice_idx_for_pos[t] = index into VOICE_NAMES, or -1 for BOS/EOS/PAD
    def _build_voice_index(T: int) -> torch.Tensor:
        """Map each position 0..T-1 to a voice index (0-4) or -1."""
        vi = torch.full((T,), -1, dtype=torch.long)
        for t in range(1, T):
            vi[t] = (t - 1) % 5  # 0=chord, 1=sop, 2=bass, 3=alto, 4=tenor
        return vi

    # ── Epoch helpers ──
    def run_epoch(loader: DataLoader, split: str):
        model.train(split == "train")
        tot_tok = 0
        sum_loss = 0.0
        correct_total = 0
        # Per-voice accuracy (vectorized)
        voice_correct = torch.zeros(5, dtype=torch.long)
        voice_count = torch.zeros(5, dtype=torch.long)

        for batch in loader:
            x = batch.x.to(device)
            y = batch.y.to(device)
            cont = batch.cont.to(device)

            with torch.set_grad_enabled(split == "train"):
                logits = model(x, cont)  # (B, T, vocab_size)

                # Mask: predict non-PAD targets
                mask = (y != PAD_ID)  # (B, T)
                mask_flat = mask.reshape(-1)

                logits_flat = logits.reshape(-1, vocab_size)
                y_flat = y.reshape(-1)

                loss = F.cross_entropy(
                    logits_flat[mask_flat], y_flat[mask_flat],
                    label_smoothing=label_smoothing if split == "train" else 0.0,
                )

                if split == "train":
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                    opt.step()
                    sched.step()

            n_tokens = mask_flat.sum().item()
            tot_tok += n_tokens
            sum_loss += loss.item() * n_tokens

            # Accuracy (vectorized)
            preds = logits.argmax(dim=-1)  # (B, T)
            correct = (preds == y) & mask  # (B, T)
            correct_total += correct.sum().item()

            # Per-voice accuracy (vectorized — no Python loops)
            B, T = x.shape
            vi = _build_voice_index(T)  # (T,) on CPU
            for v_idx in range(5):
                v_mask = (vi == v_idx).unsqueeze(0) & mask.cpu()  # (B, T)
                voice_count[v_idx] += v_mask.sum().item()
                voice_correct[v_idx] += (correct.cpu() & v_mask).sum().item()

        avg_loss = sum_loss / max(1, tot_tok)
        ppl = math.exp(min(20.0, avg_loss))
        acc = correct_total / max(1, tot_tok)

        voice_accs = {}
        for i, v in enumerate(VOICE_NAMES):
            voice_accs[v] = voice_correct[i].item() / max(1, voice_count[i].item())

        return avg_loss, ppl, acc, voice_accs

    # ── Training loop ──
    print(f"\nStarting training: {epochs} epochs, patience={patience}, "
          f"batch_size={batch_size}, lr={lr}", flush=True)
    print(f"Train: {len(train_ds)} sequences | Val: {len(val_ds)} sequences",
          flush=True)
    print(flush=True)

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()
        tr_loss, tr_ppl, tr_acc, tr_vaccs = run_epoch(train_loader, "train")
        va_loss, va_ppl, va_acc, va_vaccs = run_epoch(val_loader, "val")
        dt = time.time() - t0

        # Voice accuracy string
        def _vacc_str(vaccs):
            return " ".join(f"{v[:3]}={vaccs[v]:.3f}" for v in VOICE_NAMES)

        msg = (f"Epoch {epoch:03d} | "
               f"train: loss={tr_loss:.3f} ppl={tr_ppl:.2f} acc={tr_acc:.3f} ({_vacc_str(tr_vaccs)}) | "
               f"val: loss={va_loss:.3f} ppl={va_ppl:.2f} acc={va_acc:.3f} ({_vacc_str(va_vaccs)}) "
               f"[{dt:.1f}s]")

        improved = (best_val - va_loss) > args.min_delta
        if improved:
            best_val = va_loss
            best_epoch = epoch
            epochs_no_improve = 0

            # ── Versioned checkpoint save ──
            base, ext = os.path.splitext(SAVE_PATH)
            ckpt_path = f"{base}_epoch{epoch:03d}_val{va_loss:.4f}{ext}"
            ckpt_payload = {
                "epoch": epoch,
                "best_val": best_val,
                "best_epoch": best_epoch,
                "model_state_dict": model.state_dict(),
                "model_state": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "vocab_info": {
                    "PAD_ID": PAD_ID, "BOS_ID": BOS_ID, "EOS_ID": EOS_ID,
                    "VOCAB_JSON": VOCAB_JSON,
                },
                "config": {
                    "D_MODEL": d_model, "N_HEADS": n_heads, "N_LAYERS": n_layers,
                    "FF_MULT": ff_mult, "DROPOUT": dropout,
                    "VOCAB_SIZE": vocab_size, "DATA_DIR": DATA_DIR,
                },
                "model_config": {
                    "vocab_size": vocab_size, "pad_id": PAD_ID,
                    "d_model": d_model, "n_heads": n_heads, "n_layers": n_layers,
                    "ff_mult": ff_mult, "dropout": dropout,
                },
            }
            torch.save(ckpt_payload, ckpt_path)
            # Symlink latest best
            tmp_link = SAVE_PATH + ".tmp"
            os.symlink(os.path.basename(ckpt_path), tmp_link)
            os.replace(tmp_link, SAVE_PATH)

            # Prune old checkpoints
            if args.keep_top_k > 0:
                existing = sorted(globmod.glob(f"{base}_epoch*{ext}"))

                def _val_from_name(p: str) -> float:
                    try:
                        return float(p.rsplit("_val", 1)[1].replace(ext, ""))
                    except (IndexError, ValueError):
                        return float("inf")

                existing.sort(key=_val_from_name)
                for old in existing[args.keep_top_k:]:
                    os.remove(old)

            msg += f"  *SAVED* {os.path.basename(ckpt_path)}"
        else:
            epochs_no_improve += 1

        print(msg, flush=True)

        if patience > 0 and epochs_no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {patience} epochs)")
            print(f"Best val loss: {best_val:.4f} at epoch {best_epoch}")
            break

    print(f"\nTraining complete. Best val loss: {best_val:.4f} at epoch {best_epoch}")
    print(f"Checkpoint: {SAVE_PATH}")


if __name__ == "__main__":
    main()
