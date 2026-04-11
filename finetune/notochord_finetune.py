#!/usr/bin/env python3
"""
Fine-tune a pre-trained Notochord model on personal MIDI data.

Loads the Lakh-pretrained checkpoint, then continues training on the
event sequences produced by notochord_convert.py.  Uses a lower LR
than pre-training so the model adapts style without forgetting structure.

Usage:
    python finetune/notochord_finetune.py \\
        --checkpoint finetune/notochord_lakh_50G_deep.pt \\
        --data_dir   finetune/runs/noto_data \\
        --out        finetune/runs/noto_finetuned.pt

    # More epochs if val loss is still dropping:
    python finetune/notochord_finetune.py \\
        --checkpoint finetune/notochord_lakh_50G_deep.pt \\
        --data_dir   finetune/runs/noto_data \\
        --out        finetune/runs/noto_finetuned.pt \\
        --epochs 20 --lr 5e-6
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from notochord import Notochord


class NotoDataset(Dataset):
    def __init__(self, data_dir: Path, split: str):
        self.insts   = torch.from_numpy(np.load(data_dir / f"{split}_insts.npy").astype(np.int64))
        self.pitches = torch.from_numpy(np.load(data_dir / f"{split}_pitches.npy").astype(np.int64))
        self.times   = torch.from_numpy(np.load(data_dir / f"{split}_times.npy").astype(np.float32))
        self.vels    = torch.from_numpy(np.load(data_dir / f"{split}_vels.npy").astype(np.float32))
        self.ends    = torch.from_numpy(np.load(data_dir / f"{split}_ends.npy").astype(np.int64))

    def __len__(self):
        return len(self.insts)

    def __getitem__(self, idx):
        return (self.insts[idx], self.pitches[idx], self.times[idx],
                self.vels[idx], self.ends[idx])


def compute_loss(outputs: dict) -> torch.Tensor:
    """Sum of negative log probs across all five prediction heads."""
    return -(
        outputs["instrument_log_probs"] +
        outputs["pitch_log_probs"] +
        outputs["time_log_probs"] +
        outputs["velocity_log_probs"] +
        outputs["end_log_probs"]
    ).mean()


def evaluate(model, loader, device):
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for insts, pitches, times, vels, ends in loader:
            out = model(insts.to(device), pitches.to(device),
                        times.to(device), vels.to(device), ends.to(device))
            total += compute_loss(out).item()
            n += 1
    return total / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True,
                    help="Pre-trained Notochord .pt file")
    ap.add_argument("--data_dir",   required=True,
                    help="Output of notochord_convert.py")
    ap.add_argument("--out",        required=True,
                    help="Path to save fine-tuned checkpoint")
    ap.add_argument("--epochs",     type=int,   default=10)
    ap.add_argument("--batch_size", type=int,   default=16)
    ap.add_argument("--lr",         type=float, default=1e-5,
                    help="Fine-tuning LR (pre-training used 1e-4, keep lower here)")
    ap.add_argument("--grad_clip",  type=float, default=1.0)
    ap.add_argument("--device",     default="auto")
    args = ap.parse_args()

    device = (
        "mps"  if torch.backends.mps.is_available()  else
        "cuda" if torch.cuda.is_available()           else
        "cpu"
    ) if args.device == "auto" else args.device
    print(f"Device: {device}")

    data_dir = Path(args.data_dir)
    meta     = json.loads((data_dir / "meta.json").read_text())
    print(f"Train: {meta['n_train']}  Val: {meta['n_val']}  seq_len: {meta['seq_len']}")

    # Load pre-trained model
    print(f"Loading checkpoint: {args.checkpoint}")
    model = Notochord.from_checkpoint(args.checkpoint)
    model = model.to(device)

    train_ds = NotoDataset(data_dir, "train")
    val_ds   = NotoDataset(data_dir, "val")
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  drop_last=True)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, drop_last=False)
    print(f"Train batches: {len(train_dl)}  Val batches: {len(val_dl)}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    best_val = float("inf")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss, t0 = 0.0, time.time()
        for insts, pitches, times, vels, ends in train_dl:
            optimizer.zero_grad()
            out = model(insts.to(device), pitches.to(device),
                        times.to(device), vels.to(device), ends.to(device))
            loss = compute_loss(out)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            train_loss += loss.item()

        avg_train = train_loss / len(train_dl)
        avg_val   = evaluate(model, val_dl, device)
        elapsed   = time.time() - t0
        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"train={avg_train:.4f}  val={avg_val:.4f}  ({elapsed:.0f}s)")

        if avg_val < best_val:
            best_val = avg_val
            # Save in the same format as the original Notochord checkpoint
            torch.save({
                "model_state":     model.state_dict(),
                "kw":              {},   # architecture already in the model object
                "optimizer_state": optimizer.state_dict(),
                "step":            epoch,
                "random_state":    torch.get_rng_state(),
            }, out_path)
            print(f"         → saved best (val={best_val:.4f}) → {out_path}")

    print(f"\nDone. Best val loss: {best_val:.4f}")
    print(f"\nNext:")
    print(f"  python finetune/notochord_generate.py \\")
    print(f"      --checkpoint {out_path} \\")
    print(f"      --out_midi   finetune/runs/generated/noto_out.mid")


if __name__ == "__main__":
    main()
