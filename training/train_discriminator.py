#!/usr/bin/env python3
"""Train the note discriminator (scalar MLP or combined CNN+MLP).

Usage:
  Scalar MLP (default):
    python -m training.train_discriminator --data notes.h5 --out model.pt

  Combined CNN+MLP (requires spec_patches in HDF5):
    python -m training.train_discriminator --data notes.h5 --out model.pt --combined
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from training.note_discriminator import (
    CombinedNoteDataset, CombinedNoteDiscriminator,
    NoteDataset, NoteDiscriminator,
    N_FEATURES, N_MEL, N_FRAMES, FEATURE_NAMES,
)


def _metrics(logits, labels, threshold):
    preds = (torch.sigmoid(logits) >= threshold).float()
    tp = ((preds == 1) & (labels == 1)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()
    tn = ((preds == 0) & (labels == 0)).sum().item()
    prec = tp / (tp + fp + 1e-8)
    rec  = tp / (tp + fn + 1e-8)
    f1   = 2 * prec * rec / (prec + rec + 1e-8)
    return prec, rec, f1, tp, fp, fn, tn


def train_scalar(args, device):
    train_ds = NoteDataset(args.data, split="train")
    val_ds   = NoteDataset(args.data, split="val")
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=2, pin_memory=(device == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=2, pin_memory=(device == "cuda"))

    n_pos      = train_ds.labels.sum().item()
    n_neg      = len(train_ds) - n_pos
    pos_weight = torch.tensor([n_neg / (n_pos + 1e-8)], dtype=torch.float32).to(device)
    print(f"Label balance — TP={int(n_pos)}, FP={int(n_neg)}, pos_weight={pos_weight.item():.2f}")

    model     = NoteDiscriminator(n_features=N_FEATURES).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_f1  = -1.0
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for feats, labels in train_loader:
            feats, labels = feats.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(feats), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(labels)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        all_logits, all_labels = [], []
        with torch.no_grad():
            for feats, labels in val_loader:
                feats, labels = feats.to(device), labels.to(device)
                logits = model(feats)
                val_loss += criterion(logits, labels).item() * len(labels)
                all_logits.append(logits.cpu())
                all_labels.append(labels.cpu())
        val_loss /= len(val_ds)
        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels)

        prec, rec, f1, *_ = _metrics(all_logits, all_labels, args.threshold)
        scheduler.step(val_loss)
        print(f"Epoch {epoch:3d}/{args.epochs}  train={train_loss:.4f}  val={val_loss:.4f}  prec={prec:.3f}  rec={rec:.3f}  F1={f1:.3f}")

        if f1 > best_f1:
            best_f1 = f1
            torch.save({
                "model_type": "scalar",
                "state_dict": model.state_dict(),
                "n_features": N_FEATURES,
                "hidden":     (64, 32),
                "threshold":  args.threshold,
                "epoch":      epoch,
                "val_f1":     f1,
            }, out_path)
            print(f"  → saved (F1={f1:.3f})")

    return model, val_loader, best_f1


def train_combined(args, device):
    train_ds = CombinedNoteDataset(args.data, split="train")
    val_ds   = CombinedNoteDataset(args.data, split="val")
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")
    print(f"Mel patches: {N_MEL} × {N_FRAMES}")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=2, pin_memory=(device == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=2, pin_memory=(device == "cuda"))

    n_pos      = train_ds.labels.sum().item()
    n_neg      = len(train_ds) - n_pos
    pos_weight = torch.tensor([n_neg / (n_pos + 1e-8)], dtype=torch.float32).to(device)
    print(f"Label balance — TP={int(n_pos)}, FP={int(n_neg)}, pos_weight={pos_weight.item():.2f}")

    model     = CombinedNoteDiscriminator(n_scalar=N_FEATURES, n_mel=N_MEL, n_frames=N_FRAMES).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_f1  = -1.0
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for spec, feats, labels in train_loader:
            spec, feats, labels = spec.to(device), feats.to(device), labels.to(device)
            optimizer.zero_grad()
            combined_logits = model(spec, feats)
            scalar_logits   = model.forward_scalar_only(feats)
            # Combined loss + auxiliary scalar-only loss to train scalar_head
            loss = criterion(combined_logits, labels) + 0.5 * criterion(scalar_logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(labels)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        all_logits, all_labels = [], []
        with torch.no_grad():
            for spec, feats, labels in val_loader:
                spec, feats, labels = spec.to(device), feats.to(device), labels.to(device)
                logits = model(spec, feats)
                val_loss += criterion(logits, labels).item() * len(labels)
                all_logits.append(logits.cpu())
                all_labels.append(labels.cpu())
        val_loss /= len(val_ds)
        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels)

        prec, rec, f1, *_ = _metrics(all_logits, all_labels, args.threshold)
        scheduler.step(val_loss)
        print(f"Epoch {epoch:3d}/{args.epochs}  train={train_loss:.4f}  val={val_loss:.4f}  prec={prec:.3f}  rec={rec:.3f}  F1={f1:.3f}")

        if f1 > best_f1:
            best_f1 = f1
            torch.save({
                "model_type": "combined",
                "state_dict": model.state_dict(),
                "n_scalar":   N_FEATURES,
                "n_mel":      N_MEL,
                "n_frames":   N_FRAMES,
                "threshold":  args.threshold,
                "epoch":      epoch,
                "val_f1":     f1,
            }, out_path)
            print(f"  → saved (F1={f1:.3f})")

    return model, val_loader, best_f1


def final_report(model, val_loader, args, device, combined: bool):
    """Print confusion matrix + permutation feature importance."""
    ckpt = torch.load(args.out, map_location=device)
    if combined:
        model = CombinedNoteDiscriminator.load(args.out, device)
    else:
        model = NoteDiscriminator.load(args.out, device)
    model.eval()

    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            if combined:
                spec, feats, labels = batch
                spec, feats, labels = spec.to(device), feats.to(device), labels.to(device)
                logits = model(spec, feats)
            else:
                feats, labels = batch
                feats, labels = feats.to(device), labels.to(device)
                logits = model(feats)
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    prec, rec, f1, tp, fp, fn, tn = _metrics(all_logits, all_labels, args.threshold)

    print(f"\nBest val F1: {ckpt['val_f1']:.3f}")
    print(f"Confusion matrix (threshold={args.threshold}):")
    print(f"  TP={tp:<6} FP={fp}")
    print(f"  FN={fn:<6} TN={tn}")
    print(f"  Precision={prec:.3f}  Recall={rec:.3f}  F1={f1:.3f}")

    # Permutation importance on scalar features
    print("\nPermutation feature importance (scalar branch, F1 drop):")
    # Collect all scalar feats
    all_scalar = torch.cat([b[1] if combined else b[0] for b in val_loader])
    all_labs   = torch.cat([b[2] if combined else b[1] for b in val_loader])
    for fi, fname in enumerate(FEATURE_NAMES):
        shuffled = all_scalar.clone()
        shuffled[:, fi] = shuffled[torch.randperm(len(shuffled)), fi]
        with torch.no_grad():
            if combined:
                # Use scalar-only head for importance
                sh_logits = model.forward_scalar_only(shuffled.to(device)).cpu()
            else:
                sh_logits = model(shuffled.to(device)).cpu()
        _, _, sh_f1, *_ = _metrics(sh_logits, all_labs, args.threshold)
        print(f"  {fname:<22} {f1 - sh_f1:+.4f}")


def main():
    ap = argparse.ArgumentParser("train_discriminator")
    ap.add_argument("--data",      default="runs/discriminator_data/notes.h5")
    ap.add_argument("--out",       default="runs/discriminator/model.pt")
    ap.add_argument("--epochs",    type=int,   default=60)
    ap.add_argument("--lr",        type=float, default=1e-3)
    ap.add_argument("--batch",     type=int,   default=256)
    ap.add_argument("--threshold", type=float, default=0.35)
    ap.add_argument("--device",    default="auto")
    ap.add_argument("--combined",  action="store_true",
                    help="Train combined CNN+MLP model (requires spec_patches in HDF5).")
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    print(f"Device: {device}  Model: {'combined CNN+MLP' if args.combined else 'scalar MLP'}")

    if args.combined:
        model, val_loader, best_f1 = train_combined(args, device)
    else:
        model, val_loader, best_f1 = train_scalar(args, device)

    final_report(model, val_loader, args, device, combined=args.combined)


if __name__ == "__main__":
    main()
