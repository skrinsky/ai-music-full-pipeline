#!/usr/bin/env python3
"""Download blues MIDI files from the GigaMIDI dataset (Hugging Face).

GigaMIDI contains 2.1M MIDI files with genre metadata. This script streams
the Parquet data, filters for blues across multiple style columns, and writes
matching MIDI files directly to disk — no need to download the full 5.5 GB zip.

Prerequisites:
  1. pip install datasets huggingface_hub
  2. Accept the dataset terms at https://huggingface.co/datasets/Metacreation/GigaMIDI
  3. huggingface-cli login   (or set HF_TOKEN env var)

Two modes:
  --info    Stream metadata, count blues tracks across all splits, print stats, exit.
  Default   Stream and save blues MIDI files into --out_dir.
"""

import argparse
import sys
from pathlib import Path

from huggingface_hub import get_token

DATASET_ID = "Metacreation/GigaMIDI"
DATASET_CONFIG = "v2.0.0"
SPLITS = ["train", "validation", "test"]

# ---------------------------------------------------------------------------
# Blues filter
# ---------------------------------------------------------------------------

BLUES_KEYWORDS = [
    "blues",
    "boogie woogie",
    "jump blues",
    "rhythm and blues",
    "r&b",
]


def is_blues(sample: dict) -> bool:
    """Check if any genre/style column contains a blues-related label."""
    for col in (
        "music_styles_curated",
        "music_style_audio_text_Discogs",
        "music_style_audio_text_Lastfm",
        "music_style_audio_text_Tagtraum",
    ):
        val = sample.get(col)
        if val and isinstance(val, list):
            for s in val:
                if s and any(kw in s.lower() for kw in BLUES_KEYWORDS):
                    return True

    # Scraped style is a single string, not a list
    scraped = sample.get("music_style_scraped")
    if scraped and isinstance(scraped, str):
        if any(kw in scraped.lower() for kw in BLUES_KEYWORDS):
            return True

    return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_auth() -> None:
    """Fail fast if no HuggingFace token is configured."""
    token = get_token()
    if not token:
        print(
            "ERROR: GigaMIDI is a gated dataset — HuggingFace authentication required.\n"
            "\n"
            "Setup steps:\n"
            "  1. Create account at https://huggingface.co/join\n"
            "  2. Accept terms at https://huggingface.co/datasets/Metacreation/GigaMIDI\n"
            "  3. Run:  huggingface-cli login\n"
            "     Or set:  export HF_TOKEN=hf_...\n",
            file=sys.stderr,
        )
        sys.exit(1)


def _load_split(split: str, streaming: bool = True):
    """Load a single split of GigaMIDI."""
    from datasets import load_dataset

    return load_dataset(
        DATASET_ID,
        DATASET_CONFIG,
        split=split,
        streaming=streaming,
    )


# ---------------------------------------------------------------------------
# Info mode
# ---------------------------------------------------------------------------


def run_info(splits: list[str]) -> None:
    """Stream all splits, count blues tracks, print stats."""
    from tqdm import tqdm

    print(f"\n=== GigaMIDI Blues Info (streaming {', '.join(splits)}) ===\n")

    total_scanned = 0
    total_blues = 0
    per_split: dict[str, tuple[int, int]] = {}
    style_counts: dict[str, int] = {}

    for split in splits:
        print(f"Scanning {split}…")
        ds = _load_split(split, streaming=True)
        scanned = 0
        blues_count = 0

        for sample in tqdm(ds, desc=f"  {split}", unit=" tracks"):
            scanned += 1
            if is_blues(sample):
                blues_count += 1
                # Collect style labels for breakdown
                for col in ("music_styles_curated",):
                    val = sample.get(col)
                    if val and isinstance(val, list):
                        for s in val:
                            if s and any(kw in s.lower() for kw in BLUES_KEYWORDS):
                                style_counts[s] = style_counts.get(s, 0) + 1

        per_split[split] = (scanned, blues_count)
        total_scanned += scanned
        total_blues += blues_count
        print(f"  {split}: {blues_count} blues / {scanned} total")

    print(f"\n{'─' * 50}")
    print(f"Total scanned: {total_scanned}")
    print(f"Total blues:   {total_blues}")
    print(f"{'─' * 50}")

    if style_counts:
        print(f"\nBlues sub-style breakdown (curated labels):")
        for style, count in sorted(style_counts.items(), key=lambda x: -x[1]):
            print(f"  {count:5d}  {style}")

    print()


# ---------------------------------------------------------------------------
# Fetch mode
# ---------------------------------------------------------------------------


def run_fetch(
    splits: list[str],
    out_dir: Path,
    max_tracks: int,
    instrument_filter: str,
) -> None:
    """Stream, filter for blues, write MIDI files to out_dir."""
    from tqdm import tqdm

    out_dir.mkdir(parents=True, exist_ok=True)

    # Instrument category filter: 0=drums-only, 1=all-instruments-with-drums, 2=no-drums
    inst_cat_col = "instrument_category__drums-only__0__all-instruments-with-drums__1_no-drums__2"
    inst_filter_map = {"all": None, "with-drums": 1, "no-drums": 2, "drums-only": 0}
    inst_cat_val = inst_filter_map.get(instrument_filter)

    saved = 0
    skipped_inst = 0
    scanned = 0

    for split in splits:
        print(f"Streaming {split}…")
        ds = _load_split(split, streaming=True)

        for sample in tqdm(ds, desc=f"  {split}", unit=" tracks"):
            scanned += 1
            if not is_blues(sample):
                continue

            # Optional instrument category filter
            if inst_cat_val is not None:
                cat = sample.get(inst_cat_col)
                if cat != inst_cat_val:
                    skipped_inst += 1
                    continue

            midi_bytes = sample.get("music")
            if not midi_bytes:
                continue

            md5 = sample.get("md5", f"track_{saved:06d}")
            dest = out_dir / f"{md5}.mid"
            dest.write_bytes(midi_bytes)
            saved += 1

            if max_tracks > 0 and saved >= max_tracks:
                break

        if max_tracks > 0 and saved >= max_tracks:
            break

    if saved == 0:
        print("ERROR: zero blues tracks extracted", file=sys.stderr)
        sys.exit(1)

    total_bytes = sum(f.stat().st_size for f in out_dir.glob("*.mid"))
    total_mb = total_bytes / (1 << 20)
    print(f"\n{'─' * 50}")
    print(f"Scanned:  {scanned} tracks")
    print(f"Saved:    {saved} blues MIDIs in {out_dir}/ ({total_mb:.1f} MiB)")
    if skipped_inst > 0:
        print(f"Skipped:  {skipped_inst} (instrument filter: {instrument_filter})")
    print(f"{'─' * 50}")
    print(f"\nNext step:")
    print(f"  python training/pre.py --midi_folder {out_dir} --data_folder runs/events")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download blues MIDI files from GigaMIDI (Hugging Face)."
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Count blues tracks and print stats (no files saved)",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("data/blues_midi"),
        help="Output directory for MIDI files (default: data/blues_midi)",
    )
    parser.add_argument(
        "--max_tracks",
        type=int,
        default=0,
        help="Limit track count, 0 = all (default: 0)",
    )
    parser.add_argument(
        "--splits",
        default="train,validation,test",
        help="Comma-separated splits to scan (default: train,validation,test)",
    )
    parser.add_argument(
        "--instruments",
        choices=["all", "with-drums", "no-drums", "drums-only"],
        default="all",
        help="Filter by instrument category (default: all)",
    )
    args = parser.parse_args()

    _check_auth()

    splits = [s.strip() for s in args.splits.split(",")]

    if args.info:
        run_info(splits)
    else:
        run_fetch(splits, args.out_dir, args.max_tracks, args.instruments)


if __name__ == "__main__":
    main()
