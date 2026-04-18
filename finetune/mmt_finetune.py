#!/usr/bin/env python3
"""
Fine-tune a pretrained MMT (Multitrack Music Transformer) checkpoint on your MIDI data.

Loads the LMD pretrained model (which knows pop/rock/jazz structure from ~170k songs),
then continues training on your data with a lower learning rate so it adapts to your
style without overwriting what it already knows.

Prerequisites:
    pip install -r finetune/mmt_requirements.txt
    # Download LMD checkpoint from UCSD SharePoint (see mmt_convert.py output)

Usage:
    # First convert your MIDIs:
    python finetune/mmt_convert.py --midi_dir summer_midi --out_dir data/summer/notes

    # Then fine-tune:
    python finetune/mmt_finetune.py \\
        --data_dir data/summer/notes \\
        --ckpt     <path/to/lmd_full/checkpoints/best_model.pt> \\
        --out_dir  finetune/runs/mmt_finetuned

    # Fewer steps / higher LR for faster style injection:
    python finetune/mmt_finetune.py ... --steps 5000 --lr 2e-4
"""
import argparse
import logging
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.utils.data

MMT_DIR = Path(__file__).resolve().parent.parent / "vendor" / "mmt" / "mmt"


def _add_mmt_to_path():
    p = str(MMT_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)


def get_lr_multiplier(step, warmup_steps, decay_end_steps, decay_end_multiplier):
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    if step > decay_end_steps:
        return decay_end_multiplier
    position = (step - warmup_steps) / (decay_end_steps - warmup_steps)
    return 1 - (1 - decay_end_multiplier) * position


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True,
                    help="Directory from mmt_convert.py (contains *.npy + train/valid-names.txt)")
    ap.add_argument("--ckpt",     required=True,
                    help="Path to pretrained MMT checkpoint (best_model.pt from LMD or SOD)")
    ap.add_argument("--out_dir",  required=True)
    # Model (must match the pretrained checkpoint's architecture)
    ap.add_argument("--dim",     type=int,   default=512)
    ap.add_argument("--layers",  type=int,   default=6)
    ap.add_argument("--heads",   type=int,   default=8)
    ap.add_argument("--max_seq_len", type=int, default=1024)
    ap.add_argument("--max_beat",    type=int, default=256)
    # Fine-tuning hyperparameters — lower LR and fewer steps than full training
    ap.add_argument("--steps",          type=int,   default=10000)
    ap.add_argument("--valid_steps",    type=int,   default=500)
    ap.add_argument("--batch_size",     type=int,   default=4)
    ap.add_argument("--lr",             type=float, default=5e-5,
                    help="Fine-tuning LR (10x lower than MMT default to preserve pretrained knowledge)")
    ap.add_argument("--lr_warmup",      type=int,   default=200)
    ap.add_argument("--grad_clip",      type=float, default=1.0)
    ap.add_argument("--aug",            action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--device",         default="auto")
    ap.add_argument("--jobs",           type=int,   default=0)
    args = ap.parse_args()

    _add_mmt_to_path()
    import dataset as mmt_dataset
    import music_x_transformers
    import representation

    # Device
    if args.device == "auto":
        device = ("mps"  if torch.backends.mps.is_available() else
                  "cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = args.device
    print(f"Device: {device}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(out_dir / "train.log", "w"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    # Encoding (fixed — determined by MMT's representation.py)
    encoding = representation.get_encoding()

    # Build model with same architecture as pretrained
    logging.info("Building model...")
    model = music_x_transformers.MusicXTransformer(
        dim=args.dim,
        encoding=encoding,
        depth=args.layers,
        heads=args.heads,
        max_seq_len=args.max_seq_len,
        max_beat=args.max_beat,
        use_abs_pos_emb=True,
        rotary_pos_emb=False,
        emb_dropout=0.2,
        attn_dropout=0.2,
        ff_dropout=0.2,
    ).to(device)

    # Load pretrained weights
    logging.info(f"Loading pretrained checkpoint: {args.ckpt}")
    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    # MMT checkpoints are saved as bare state_dicts
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    logging.info("Pretrained weights loaded.")

    n_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Parameters: {n_params:,}")

    # Datasets
    data_dir = Path(args.data_dir)
    train_ds = mmt_dataset.MusicDataset(
        filename=data_dir / "train-names.txt",
        data_dir=data_dir,
        encoding=encoding,
        max_seq_len=args.max_seq_len,
        max_beat=args.max_beat,
        use_augmentation=args.aug,
    )
    val_ds = mmt_dataset.MusicDataset(
        filename=data_dir / "valid-names.txt",
        data_dir=data_dir,
        encoding=encoding,
        max_seq_len=args.max_seq_len,
        max_beat=args.max_beat,
        use_augmentation=False,
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.jobs, collate_fn=mmt_dataset.MusicDataset.collate,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.jobs, collate_fn=mmt_dataset.MusicDataset.collate,
    )
    logging.info(f"Train songs: {len(train_ds)}  Val songs: {len(val_ds)}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: get_lr_multiplier(
            step, args.lr_warmup, args.steps, 0.1
        ),
    )

    loss_csv = open(out_dir / "loss.csv", "w")
    loss_csv.write("step,train_loss,valid_loss\n")

    step = 0
    best_val = float("inf")
    train_iterator = iter(train_loader)

    logging.info(f"Fine-tuning for {args.steps} steps...")
    while step < args.steps:
        # --- train for valid_steps batches ---
        model.train()
        recent = []
        for _ in range(min(args.valid_steps, args.steps - step)):
            try:
                batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_loader)
                batch = next(train_iterator)

            seq  = batch["seq"].to(device)
            mask = batch["mask"].to(device)

            optimizer.zero_grad()
            loss = model(seq, mask=mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()

            recent.append(float(loss))
            step += 1

        train_loss = float(np.mean(recent))

        # --- validate ---
        model.eval()
        total_val, count = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                seq  = batch["seq"].to(device)
                mask = batch["mask"].to(device)
                loss = model(seq, mask=mask)
                total_val += float(loss) * len(batch["seq"])
                count += len(batch["seq"])
        val_loss = total_val / max(count, 1)

        logging.info(f"Step {step:6d}/{args.steps}  train={train_loss:.4f}  val={val_loss:.4f}")
        loss_csv.write(f"{step},{train_loss},{val_loss}\n")
        loss_csv.flush()

        ckpt_path = out_dir / "checkpoints" / f"model_{step}.pt"
        torch.save(model.state_dict(), ckpt_path)
        if val_loss < best_val:
            best_val = val_loss
            shutil.copyfile(ckpt_path, out_dir / "checkpoints" / "best_model.pt")
            logging.info(f"  → new best (val={best_val:.4f})")

    loss_csv.close()
    logging.info(f"\nDone. Best val loss: {best_val:.4f}")
    logging.info(f"Checkpoint → {out_dir}/checkpoints/best_model.pt")
    logging.info("\nNext:")
    logging.info(f"  python finetune/mmt_generate.py \\")
    logging.info(f"      --ckpt {out_dir}/checkpoints/best_model.pt \\")
    logging.info(f"      --out_midi finetune/runs/mmt_out.mid")


if __name__ == "__main__":
    main()
