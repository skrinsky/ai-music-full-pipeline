#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Training loop for cascade model.

Usage:
  python training/train_cascade.py \\
      --data_dir runs/cascade_events \\
      --train_pkl runs/cascade_events/cascade_train.pkl \\
      --val_pkl runs/cascade_events/cascade_val.pkl \\
      --vocab_json runs/cascade_events/cascade_vocab.json \\
      --save_path runs/checkpoints/cascade_model.pt \\
      --device auto
"""

import os
import sys
import json
import math
import pickle
import random
import time
import argparse
import atexit
import signal
import glob as globmod
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from training.model_cascade import CascadedESModel
from training.train import (
    choose_auto_config,
    WarmupCosine,
    pick_device,
    token_dropout_,
)

# ─────────────────────── DEFAULTS ───────────────────────────

D_MODEL = 192
N_HEADS = 6
N_LAYERS = 4
FF_MULT = 3
DROPOUT = 0.12

BATCH_SIZE = 32
EPOCHS = 200
SEQ_LEN = 1024
LR = 2e-4
BETAS = (0.9, 0.95)
WEIGHT_DECAY = 1e-2
MAX_GRAD_NORM = 1.0

LABEL_SMOOTH_TYPE = 0.05
LABEL_SMOOTH_VALUE = 0.04
LABEL_SMOOTH_PER_TYPE = {
    "PITCH_GENERAL": 0.02,
    "PITCH_DRUMS": 0.02,
}
TOKEN_DROPOUT_P = 0.07

ALPHA_TYPE = 0.2
ALPHA_VALUE = 0.8

SEED = 42


# ─────────────────────── VOCAB LOADING ──────────────────────

def load_cascade_vocab(vocab_path: str) -> Tuple[dict, dict]:
    """Load cascade vocab and compute derived info."""
    with open(vocab_path, "r") as f:
        vocab = json.load(f)
    layout = vocab["layout"]

    type_names = [k for k in layout.keys() if k not in ("PAD", "BOS", "EOS")]
    head_sizes = [layout[k]["size"] for k in type_names]
    starts = {k: layout[k]["start"] for k in layout}
    V = max(spec["start"] + spec["size"] for spec in layout.values())

    type_of_id = np.full((V,), -1, dtype=np.int64)
    local_of_id = np.full((V,), -1, dtype=np.int64)
    name_to_type_idx = {nm: i for i, nm in enumerate(type_names)}
    for nm in type_names:
        s, n = layout[nm]["start"], layout[nm]["size"]
        t_idx = name_to_type_idx[nm]
        type_of_id[s : s + n] = t_idx
        local_of_id[s : s + n] = np.arange(n, dtype=np.int64)

    info = {
        "V": V,
        "layout": layout,
        "type_names": type_names,
        "head_sizes": head_sizes,
        "starts": starts,
        "type_of_id": torch.from_numpy(type_of_id),
        "local_of_id": torch.from_numpy(local_of_id),
    }
    return vocab, info


# ─────────────────────── DATASET ────────────────────────────

class CascadeDataset(Dataset):
    """Dataset for cascade training examples.

    Each item: (token_ids, musical_times, sep_position, stage_id)
    """

    def __init__(self, pkl_path: str):
        with open(pkl_path, "rb") as f:
            obj = pickle.load(f)
        self.seqs: List[List[int]] = obj["sequences"]
        self.times: List[List[float]] = obj["musical_times"]
        self.seps: List[int] = obj["sep_positions"]
        self.stages: List[int] = obj["stage_ids"]
        assert len(self.seqs) == len(self.times) == len(self.seps) == len(self.stages)

    def __len__(self) -> int:
        return len(self.seqs)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.seqs[idx], dtype=torch.long),
            torch.tensor(self.times[idx], dtype=torch.float32),
            self.seps[idx],
            self.stages[idx],
        )


@dataclass
class CascadeBatch:
    x: torch.Tensor           # (B, T) input tokens
    y: torch.Tensor           # (B, T) target tokens (shifted)
    times: torch.Tensor       # (B, T) musical times
    loss_mask: torch.Tensor   # (B, T) bool — True for target tokens (after SEP)


def collate_cascade(
    batch,
    *,
    pad_id: int,
    seq_len: int,
    token_dropout_p: float,
    replace_id: int,
    protected_ids: List[int],
) -> CascadeBatch:
    """Collate cascade batch with padding and loss masking."""
    seqs, times_list, seps, stages = zip(*batch)
    B = len(seqs)

    # Pad/crop to seq_len
    x_out = torch.full((B, seq_len), pad_id, dtype=torch.long)
    times_out = torch.zeros(B, seq_len, dtype=torch.float32)
    loss_mask = torch.zeros(B, seq_len, dtype=torch.bool)

    for i in range(B):
        seq = seqs[i]
        t = times_list[i]
        sep_pos = seps[i]
        L = min(len(seq), seq_len)

        x_out[i, :L] = seq[:L]
        times_out[i, :L] = t[:L]

        # Loss mask: tokens after SEP (indices sep_pos+1 through L-1)
        # We compute loss on next-token prediction, so mask[j] means
        # we care about predicting y[j] = x[j+1]
        # We want loss on positions where the TARGET is after SEP
        mask_start = sep_pos  # y[sep_pos] = x[sep_pos+1] which is first target token
        if mask_start < L - 1:
            loss_mask[i, mask_start : L - 1] = True

    # Next-token targets
    y_out = x_out.clone()
    y_out[:, :-1] = x_out[:, 1:]
    y_out[:, -1] = pad_id

    # Token dropout on input (not on protected tokens)
    token_dropout_(x_out, token_dropout_p, protected_ids, replace_id)

    return CascadeBatch(x=x_out, y=y_out, times=times_out, loss_mask=loss_mask)


# ─────────────────────── MAIN ───────────────────────────────

def main():
    global D_MODEL, N_HEADS, N_LAYERS, FF_MULT

    ap = argparse.ArgumentParser("train_cascade: train cascaded event-stream model.")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--train_pkl", required=True)
    ap.add_argument("--val_pkl", required=True)
    ap.add_argument("--vocab_json", required=True)
    ap.add_argument("--save_path", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--num_workers", type=int, default=None)
    ap.add_argument("--auto_scale", action="store_true", default=True)
    ap.add_argument("--no_auto_scale", action="store_true", default=False)
    ap.add_argument("--target_tpp", type=float, default=8.0)
    ap.add_argument("--max_d_model", type=int, default=256)
    ap.add_argument("--min_params", type=int, default=100000)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--min_delta", type=float, default=1e-4)
    ap.add_argument("--keep_top_k", type=int, default=3)
    ap.add_argument("--d_model", type=int, default=None)
    ap.add_argument("--n_layers", type=int, default=None)
    ap.add_argument("--n_heads", type=int, default=None)
    ap.add_argument("--ff_mult", type=int, default=None)
    ap.add_argument("--seq_len", type=int, default=SEQ_LEN)
    ap.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    seq_len = args.seq_len

    # ── single-instance lock ──
    lock_path = args.save_path + ".lock"
    if os.path.exists(lock_path):
        try:
            other_pid = int(open(lock_path).read().strip())
            os.kill(other_pid, 0)
            print(f"ERROR: another train_cascade.py (PID {other_pid}) is already "
                  f"writing to {args.save_path}.")
            sys.exit(1)
        except (ProcessLookupError, ValueError):
            pass
        except PermissionError:
            print(f"ERROR: lock held by another process for {args.save_path}.")
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

    # Clean stale checkpoints
    if not args.resume:
        base, ext = os.path.splitext(args.save_path)
        stale = globmod.glob(f"{base}_epoch*{ext}")
        if stale:
            print(f"Removing {len(stale)} stale checkpoint(s)")
            for f in stale:
                os.remove(f)
            if os.path.islink(args.save_path):
                os.remove(args.save_path)

    device = pick_device(args.device)
    print(f"Using device: {device}")

    # ── vocab ──
    vocab, vinfo = load_cascade_vocab(args.vocab_json)
    V = vinfo["V"]
    layout = vinfo["layout"]
    type_names = vinfo["type_names"]
    head_sizes = vinfo["head_sizes"]
    starts = vinfo["starts"]
    type_of_id = vinfo["type_of_id"].to(device)
    local_of_id = vinfo["local_of_id"].to(device)

    PAD_ID = layout["PAD"]["start"]
    BOS_ID = layout["BOS"]["start"]
    EOS_ID = layout["EOS"]["start"]
    SEP_ID = layout["SEP"]["start"]
    TIME_SHIFT_REPL = layout["TIME_SHIFT"]["start"]

    # ── datasets ──
    train_ds = CascadeDataset(args.train_pkl)
    val_ds = CascadeDataset(args.val_pkl)

    # ── model config ──
    if args.ff_mult is not None:
        FF_MULT = args.ff_mult
    auto_scale = args.auto_scale and not args.no_auto_scale
    if args.d_model is not None or args.n_layers is not None or args.n_heads is not None:
        auto_scale = False
        if args.d_model is not None:
            D_MODEL = args.d_model
        if args.n_layers is not None:
            N_LAYERS = args.n_layers
        if args.n_heads is not None:
            N_HEADS = args.n_heads

    if auto_scale:
        info = choose_auto_config(
            vocab_size=V,
            train_windows=len(train_ds),
            seq_len=seq_len,
            target_tpp=args.target_tpp,
            ff_mult=FF_MULT,
            min_params=args.min_params,
            max_d_model=args.max_d_model,
        )
        D_MODEL = info["d_model"]
        N_LAYERS = info["n_layers"]
        N_HEADS = info["n_heads"]
        FF_MULT = info["ff_mult"]
        print(
            f"Auto-scale: windows={len(train_ds)} seq_len={seq_len} → "
            f"d_model={D_MODEL} n_layers={N_LAYERS} n_heads={N_HEADS} ff_mult={FF_MULT} "
            f"est_params≈{info['est_params']:,}"
        )
    else:
        print(f"Model config: d_model={D_MODEL} n_layers={N_LAYERS} n_heads={N_HEADS} ff_mult={FF_MULT}")

    # ── dataloaders ──
    protected_ids = [PAD_ID, BOS_ID, EOS_ID, SEP_ID]
    for nm in ["BAR", "INST"]:
        if nm in layout:
            protected_ids.append(layout[nm]["start"])

    num_workers = args.num_workers if args.num_workers is not None else (2 if device.type == "cuda" else 0)

    collate_fn = partial(
        collate_cascade,
        pad_id=PAD_ID,
        seq_len=seq_len,
        token_dropout_p=TOKEN_DROPOUT_P,
        replace_id=TIME_SHIFT_REPL,
        protected_ids=protected_ids,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=(device.type == "cuda"),
        num_workers=num_workers,
        collate_fn=collate_fn,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
        num_workers=num_workers,
        collate_fn=collate_fn,
        persistent_workers=(num_workers > 0),
    )

    # ── model ──
    model = CascadedESModel(
        pad_id=PAD_ID,
        type_names=type_names,
        head_sizes=head_sizes,
        num_embeddings=V,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        ff_mult=FF_MULT,
        dropout=DROPOUT,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {n_params / 1e6:.2f}M | Types={len(type_names)} | Vocab≈{V}")

    # ── optimizer + scheduler ──
    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=BETAS, weight_decay=WEIGHT_DECAY)
    steps_per_epoch = max(1, len(train_loader))
    total_steps = EPOCHS * steps_per_epoch
    warmup_steps = max(10, int(0.02 * total_steps))
    sched = WarmupCosine(opt, warmup_steps=warmup_steps, total_steps=total_steps, min_lr=1e-6, base_lr=LR)

    best_val = float("inf")
    best_epoch = -1
    epochs_no_improve = 0
    start_epoch = 1

    # ── resume ──
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

    # ── epoch loop ──
    def run_epoch(loader, split: str):
        model.train(split == "train")

        tot_tok = 0
        sum_loss = 0.0
        sum_type_loss = 0.0
        sum_val_loss = 0.0
        correct_type = 0
        correct_exact = 0

        for batch in loader:
            x = batch.x.to(device)
            y = batch.y.to(device)
            times = batch.times.to(device)
            loss_mask = batch.loss_mask.to(device)

            # Flatten mask
            mask_flat = loss_mask.reshape(-1)
            if not mask_flat.any():
                continue

            y_type = type_of_id[y]
            y_local = local_of_id[y]
            y_type_f = y_type.reshape(-1)[mask_flat]
            y_local_f = y_local.reshape(-1)[mask_flat]

            with torch.set_grad_enabled(split == "train"):
                type_logits, value_logits = model(x, times)

                # TYPE loss
                tlog = type_logits.reshape(-1, len(type_names))[mask_flat]
                type_loss = F.cross_entropy(tlog, y_type_f, label_smoothing=LABEL_SMOOTH_TYPE)

                # VALUE loss per true-type
                val_loss_sum = 0.0
                val_count = 0
                y_type_flat = y_type.reshape(-1)
                y_local_flat = y_local.reshape(-1)

                for t_idx, head in enumerate(value_logits):
                    h = head.reshape(-1, head.size(-1))
                    sel = (y_type_flat == t_idx) & mask_flat
                    if sel.any():
                        tname = type_names[t_idx]
                        ls = LABEL_SMOOTH_PER_TYPE.get(tname, LABEL_SMOOTH_VALUE)
                        ce = F.cross_entropy(h[sel], y_local_flat[sel], label_smoothing=ls)
                        val_loss_sum += ce * sel.sum()
                        val_count += sel.sum()

                val_loss = (val_loss_sum / val_count) if val_count > 0 else torch.tensor(0.0, device=device)
                loss = ALPHA_TYPE * type_loss + ALPHA_VALUE * val_loss

                if split == "train":
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                    opt.step()
                    sched.step()

            n_tokens = mask_flat.sum().item()
            tot_tok += n_tokens
            sum_loss += loss.item() * n_tokens
            sum_type_loss += type_loss.item() * n_tokens
            sum_val_loss += val_loss.item() * n_tokens

            pred_type = type_logits.argmax(dim=-1)
            correct_type += (pred_type.reshape(-1)[mask_flat] == y_type_f).sum().item()

            # Exact accuracy (reconstruct global IDs)
            pred_local = torch.zeros_like(pred_type)
            for t_idx, head in enumerate(value_logits):
                pl = head.argmax(dim=-1)
                sel = pred_type == t_idx
                if sel.any():
                    pred_local[sel] = pl[sel]

            # Reconstruct global IDs
            pred_global = torch.zeros_like(pred_type)
            for t_idx, nm in enumerate(type_names):
                s = starts[nm]
                sel = pred_type == t_idx
                if sel.any():
                    pred_global[sel] = s + pred_local[sel]

            y_flat = y.reshape(-1)
            correct_exact += (pred_global.reshape(-1)[mask_flat] == y_flat[mask_flat]).sum().item()

        avg_loss = sum_loss / max(1, tot_tok)
        avg_tloss = sum_type_loss / max(1, tot_tok)
        avg_vloss = sum_val_loss / max(1, tot_tok)
        ppl = math.exp(min(20.0, avg_loss))
        acc_type = correct_type / max(1, tot_tok)
        acc_exact = correct_exact / max(1, tot_tok)

        return avg_loss, ppl, acc_exact, avg_tloss, avg_vloss, acc_type

    # ── main training loop ──
    for epoch in range(start_epoch, EPOCHS + 1):
        t0 = time.time()
        tr = run_epoch(train_loader, "train")
        va = run_epoch(val_loader, "val")
        dt = time.time() - t0

        tr_loss, tr_ppl, tr_acc, tr_tloss, tr_vloss, tr_tacc = tr
        va_loss, va_ppl, va_acc, va_tloss, va_vloss, va_tacc = va

        msg = (
            f"Epoch {epoch:03d} | "
            f"train: loss={tr_loss:.3f} ppl={tr_ppl:.2f} acc={tr_acc:.3f} (type={tr_tacc:.3f}) | "
            f"val: loss={va_loss:.3f} ppl={va_ppl:.2f} acc={va_acc:.3f} (type={va_tacc:.3f}) "
            f"[{dt:.1f}s]"
        )

        improved = (best_val - va_loss) > args.min_delta
        stop_now = False

        if improved:
            best_val = va_loss
            best_epoch = epoch
            epochs_no_improve = 0

            base, ext = os.path.splitext(args.save_path)
            ckpt_path = f"{base}_epoch{epoch:03d}_val{va_loss:.4f}{ext}"
            ckpt_payload = {
                "epoch": epoch,
                "best_val": best_val,
                "best_epoch": best_epoch,
                "model_state_dict": model.state_dict(),
                "model_state": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "factored_meta": {
                    "type_names": type_names,
                    "head_sizes": head_sizes,
                    "starts": starts,
                    "ALPHA_TYPE": ALPHA_TYPE,
                    "ALPHA_VALUE": ALPHA_VALUE,
                },
                "vocab_info": {
                    "PAD_ID": PAD_ID,
                    "BOS_ID": BOS_ID,
                    "EOS_ID": EOS_ID,
                    "SEP_ID": SEP_ID,
                    "VOCAB_JSON": args.vocab_json,
                },
                "config": {
                    "D_MODEL": D_MODEL,
                    "N_HEADS": N_HEADS,
                    "N_LAYERS": N_LAYERS,
                    "FF_MULT": FF_MULT,
                    "DROPOUT": DROPOUT,
                    "SEQ_LEN": seq_len,
                    "DATA_DIR": args.data_dir,
                },
                "model_config": {
                    "D_MODEL": D_MODEL,
                    "N_HEADS": N_HEADS,
                    "N_LAYERS": N_LAYERS,
                    "FF_MULT": FF_MULT,
                    "DROPOUT": DROPOUT,
                    "SEQ_LEN": seq_len,
                    "PAD_ID": PAD_ID,
                    "BOS_ID": BOS_ID,
                    "EOS_ID": EOS_ID,
                    "SEP_ID": SEP_ID,
                },
            }
            torch.save(ckpt_payload, ckpt_path)

            tmp_link = args.save_path + ".tmp"
            os.symlink(os.path.basename(ckpt_path), tmp_link)
            os.replace(tmp_link, args.save_path)

            if args.keep_top_k > 0:
                existing = sorted(globmod.glob(f"{base}_epoch*{ext}"))

                def _val_from_name(p: str) -> float:
                    try:
                        return float(p.rsplit("_val", 1)[1].replace(ext, ""))
                    except (IndexError, ValueError):
                        return float("inf")

                existing.sort(key=_val_from_name)
                for old in existing[args.keep_top_k :]:
                    os.remove(old)

            msg += f"  → Saved {os.path.basename(ckpt_path)}"
        else:
            epochs_no_improve += 1

        if args.patience > 0 and epochs_no_improve >= args.patience:
            stop_now = True
            msg += f"  → Early stop (patience={args.patience}, best_epoch={best_epoch})"

        print(msg)
        if stop_now:
            break

    print(f"Done. Best epoch {best_epoch} | best val loss {best_val:.3f} | saved → {os.path.realpath(args.save_path)}")


if __name__ == "__main__":
    main()
