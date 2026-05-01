#!/usr/bin/env python3
"""Train the NoteDiscriminator MLP on HDF5 data from build_discriminator_data.py."""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from training.note_discriminator import NoteDataset, NoteDiscriminator, N_FEATURES, FEATURE_NAMES


def _metrics(logits, labels, threshold):
    preds = (torch.sigmoid(logits) >= threshold).float()
    tp = ((preds == 1) & (labels == 1)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()
    tn = ((preds == 0) & (labels == 0)).sum().item()
    prec  = tp / (tp + fp + 1e-8)
    rec   = tp / (tp + fn + 1e-8)
    f1    = 2 * prec * rec / (prec + rec + 1e-8)
    return prec, rec, f1, tp, fp, fn, tn


def main():
    ap = argparse.ArgumentParser("train_discriminator: train note TP/FP MLP.")
    ap.add_argument("--data",      default="runs/discriminator_data/notes.h5")
    ap.add_argument("--out",       default="runs/discriminator/model.pt")
    ap.add_argument("--epochs",    type=int,   default=60)
    ap.add_argument("--lr",        type=float, default=1e-3)
    ap.add_argument("--batch",     type=int,   default=256)
    ap.add_argument("--threshold", type=float, default=0.35)
    ap.add_argument("--device",    default="auto")
    args = ap.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Device: {device}")

    train_ds = NoteDataset(args.data, split="train")
    val_ds   = NoteDataset(args.data, split="val")
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=2)

    # pos_weight to handle class imbalance
    all_labels = train_ds.labels
    n_pos = all_labels.sum().item()
    n_neg = len(all_labels) - n_pos
    pos_weight = torch.tensor([n_neg / (n_pos + 1e-8)], dtype=torch.float32).to(device)
    print(f"Label balance — TP={int(n_pos)}, FP={int(n_neg)}, pos_weight={pos_weight.item():.2f}")

    model = NoteDiscriminator(n_features=N_FEATURES).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_f1 = -1.0
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        # --- train ---
        model.train()
        train_loss = 0.0
        for feats, labels in train_loader:
            feats, labels = feats.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(feats)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(labels)
        train_loss /= len(train_ds)

        # --- val ---
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

        prec, rec, f1, tp, fp, fn, tn = _metrics(all_logits, all_labels, args.threshold)
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"prec={prec:.3f}  rec={rec:.3f}  F1={f1:.3f}"
        )

        if f1 > best_f1:
            best_f1 = f1
            torch.save({
                "state_dict": model.state_dict(),
                "n_features": N_FEATURES,
                "hidden":     (64, 32),
                "threshold":  args.threshold,
                "epoch":      epoch,
                "val_f1":     f1,
            }, out_path)
            print(f"  → Saved best model (F1={f1:.3f}) to {out_path}")

    # --- final evaluation ---
    print(f"\nBest val F1: {best_f1:.3f}")
    ckpt = torch.load(out_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    all_logits, all_labels = [], []
    with torch.no_grad():
        for feats, labels in val_loader:
            feats, labels = feats.to(device), labels.to(device)
            all_logits.append(model(feats).cpu())
            all_labels.append(labels.cpu())
    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)

    prec, rec, f1, tp, fp, fn, tn = _metrics(all_logits, all_labels, args.threshold)
    print(f"\nConfusion matrix (threshold={args.threshold}):")
    print(f"  TP={tp:<6} FP={fp}")
    print(f"  FN={fn:<6} TN={tn}")
    print(f"  Precision={prec:.3f}  Recall={rec:.3f}  F1={f1:.3f}")

    # --- permutation feature importance ---
    print("\nPermutation feature importance (F1 drop when feature is shuffled):")
    baseline_f1 = f1
    all_feats = torch.cat([feats for feats, _ in val_loader])
    all_labs  = torch.cat([labs  for _,     labs in val_loader])
    importances = []
    for fi, fname in enumerate(FEATURE_NAMES):
        shuffled = all_feats.clone()
        idx = torch.randperm(len(shuffled))
        shuffled[:, fi] = shuffled[idx, fi]
        with torch.no_grad():
            sh_logits = model(shuffled.to(device)).cpu()
        _, _, sh_f1, *_ = _metrics(sh_logits, all_labs, args.threshold)
        drop = baseline_f1 - sh_f1
        importances.append((drop, fname))
    importances.sort(reverse=True)
    for drop, fname in importances:
        print(f"  {fname:<20} F1 drop = {drop:+.4f}")


if __name__ == "__main__":
    main()
