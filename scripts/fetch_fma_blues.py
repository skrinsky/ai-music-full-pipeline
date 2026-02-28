#!/usr/bin/env python3
"""Download blues tracks from the Free Music Archive (FMA).

Two modes:
  --info    Download metadata only, print blues track stats, exit.
  Default   Download audio zip, extract only blues MP3s into --out_dir.

Requires curl or wget on PATH. Dependencies: pandas, tqdm.
"""

import argparse
import csv
import hashlib
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# FMA URLs and checksums
# ---------------------------------------------------------------------------

FMA_META_URL = "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip"
FMA_META_SHA1 = "f0df49ffe5f2a6008d7dc83c6915b31835dfe733"

FMA_AUDIO_URLS: dict[str, str] = {
    "medium": "https://os.unil.cloud.switch.ch/fma/fma_medium.zip",
    "large": "https://os.unil.cloud.switch.ch/fma/fma_large.zip",
    "full": "https://os.unil.cloud.switch.ch/fma/fma_full.zip",
}

FMA_AUDIO_SHA1: dict[str, str] = {
    "medium": "c67b69ea232b82f8f3b2adca6a08d3bfd1e8f544",
    "large": "497b21b91bf224d1a43615de9af5fcea8a021c6c",
    "full": "0f0ace23fbe9ba30ecb7e95f763e435ea802b8ab",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_downloader() -> str:
    """Return 'curl' or 'wget', or exit."""
    for cmd in ("curl", "wget"):
        if shutil.which(cmd):
            return cmd
    print("ERROR: neither curl nor wget found on PATH", file=sys.stderr)
    sys.exit(1)


def _download(url: str, dest: Path, downloader: str) -> None:
    """Resumable download of *url* to *dest*."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if downloader == "curl":
        cmd = ["curl", "-L", "-C", "-", "-o", str(dest), url]
    else:
        cmd = ["wget", "-c", "-O", str(dest), url]
    print(f"Downloading {url}\n  → {dest}")
    subprocess.run(cmd, check=True)


def _sha1(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _verify_sha1(path: Path, expected: str) -> None:
    actual = _sha1(path)
    if actual != expected:
        print(
            f"SHA1 mismatch for {path}:\n"
            f"  expected {expected}\n"
            f"  got      {actual}",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"SHA1 OK: {path.name}")


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

SUBSET_ORDER = {"small": 0, "medium": 1, "large": 2, "full": 3}


def _load_tracks_csv(meta_dir: Path) -> pd.DataFrame:
    """Load tracks.csv with its multi-level header."""
    csv_path = meta_dir / "fma_metadata" / "tracks.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found", file=sys.stderr)
        sys.exit(1)
    tracks = pd.read_csv(csv_path, index_col=0, header=[0, 1])
    return tracks


def _filter_blues(tracks: pd.DataFrame, subset: str) -> pd.Index:
    """Return track IDs where genre_top=='Blues' and subset <= chosen."""
    genre_col = ("track", "genre_top")
    subset_col = ("set", "subset")

    if genre_col not in tracks.columns or subset_col not in tracks.columns:
        print("ERROR: expected columns not found in tracks.csv", file=sys.stderr)
        print(f"  Available columns: {list(tracks.columns)}", file=sys.stderr)
        sys.exit(1)

    max_rank = SUBSET_ORDER[subset]
    mask_genre = tracks[genre_col] == "Blues"
    mask_subset = tracks[subset_col].map(
        lambda s: SUBSET_ORDER.get(str(s), 999) <= max_rank
    )
    return tracks.index[mask_genre & mask_subset]


# ---------------------------------------------------------------------------
# Info mode
# ---------------------------------------------------------------------------


def _print_info(tracks: pd.DataFrame, subset: str) -> None:
    genre_col = ("track", "genre_top")
    subset_col = ("set", "subset")
    dur_col = ("track", "duration")

    blues = tracks[tracks[genre_col] == "Blues"]
    print(f"\n=== FMA Blues Track Info ===")
    print(f"Total blues tracks in FMA: {len(blues)}")

    if dur_col in tracks.columns:
        total_sec = blues[dur_col].sum()
        hours = total_sec / 3600
        print(f"Total duration: {hours:.1f} hours ({total_sec:.0f} s)")

    print(f"\nBreakdown by subset:")
    for s in ("small", "medium", "large"):
        count = (blues[subset_col] == s).sum()
        print(f"  {s:>8s}: {count:5d} tracks")

    # Cumulative counts for each --subset choice
    print(f"\nCumulative (what --subset gives you):")
    for s in ("medium", "large", "full"):
        ids = _filter_blues(tracks, s)
        print(f"  --subset {s:>6s}: {len(ids):5d} tracks")

    chosen = _filter_blues(tracks, subset)
    print(f"\nWith --subset {subset}: {len(chosen)} blues tracks")


# ---------------------------------------------------------------------------
# Extract
# ---------------------------------------------------------------------------


def _extract_blues_mp3s(
    zip_path: Path, track_ids: pd.Index, subset: str, out_dir: Path
) -> int:
    """Extract only blues MP3s from the FMA zip into out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    extracted = 0
    missing = 0

    with zipfile.ZipFile(zip_path, "r") as zf:
        nameset = set(zf.namelist())
        for tid in tqdm(track_ids, desc="Extracting blues MP3s"):
            tid_str = f"{tid:06d}"
            # FMA zip path: fma_medium/012/012345.mp3
            inner = f"fma_{subset}/{tid_str[:3]}/{tid_str}.mp3"
            if inner not in nameset:
                import warnings
                warnings.warn(f"Track {tid} not found in zip: {inner}")
                missing += 1
                continue
            data = zf.read(inner)
            dest = out_dir / f"{tid_str}.mp3"
            dest.write_bytes(data)
            extracted += 1

    if extracted == 0:
        print("ERROR: zero tracks extracted", file=sys.stderr)
        sys.exit(1)
    if missing > 0:
        print(f"WARNING: {missing} tracks missing from zip (known FMA issue)")
    return extracted


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download blues tracks from the Free Music Archive (FMA)."
    )
    parser.add_argument(
        "--info", action="store_true", help="Print blues track stats and exit"
    )
    parser.add_argument(
        "--subset",
        choices=["medium", "large", "full"],
        default="medium",
        help="FMA audio zip size (default: medium, 22 GiB)",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("data/blues"),
        help="Output directory for MP3s (default: data/blues)",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        default=Path("data/.fma_cache"),
        help="Cache directory for metadata + zip (default: data/.fma_cache)",
    )
    parser.add_argument(
        "--max_tracks",
        type=int,
        default=0,
        help="Limit track count, 0 = all (default: 0)",
    )
    args = parser.parse_args()

    # Preflight
    downloader = _find_downloader()

    # ----- Metadata -----
    meta_zip = args.cache_dir / "fma_metadata.zip"
    meta_dir = args.cache_dir / "metadata"

    if not meta_zip.exists():
        _download(FMA_META_URL, meta_zip, downloader)
        _verify_sha1(meta_zip, FMA_META_SHA1)
    elif not (meta_dir / "fma_metadata" / "tracks.csv").exists():
        _verify_sha1(meta_zip, FMA_META_SHA1)

    if not (meta_dir / "fma_metadata" / "tracks.csv").exists():
        print("Extracting metadata…")
        meta_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(meta_zip, "r") as zf:
            zf.extractall(meta_dir)

    tracks = _load_tracks_csv(meta_dir)

    # ----- Info mode -----
    if args.info:
        _print_info(tracks, args.subset)
        return

    # ----- Filter -----
    blues_ids = _filter_blues(tracks, args.subset)
    print(f"Found {len(blues_ids)} blues tracks for --subset {args.subset}")

    if args.max_tracks > 0:
        blues_ids = blues_ids[: args.max_tracks]
        print(f"Limiting to {len(blues_ids)} tracks (--max_tracks)")

    if len(blues_ids) == 0:
        print("ERROR: no blues tracks found", file=sys.stderr)
        sys.exit(1)

    # ----- Download audio zip -----
    audio_url = FMA_AUDIO_URLS[args.subset]
    audio_zip = args.cache_dir / f"fma_{args.subset}.zip"
    expected_sha1 = FMA_AUDIO_SHA1[args.subset]

    if not audio_zip.exists():
        _download(audio_url, audio_zip, downloader)
        _verify_sha1(audio_zip, expected_sha1)

    # ----- Extract -----
    n = _extract_blues_mp3s(audio_zip, blues_ids, args.subset, args.out_dir)

    # ----- Summary -----
    total_bytes = sum(f.stat().st_size for f in args.out_dir.glob("*.mp3"))
    total_mb = total_bytes / (1 << 20)
    print(f"\nDone: {n} blues MP3s in {args.out_dir}/ ({total_mb:.1f} MiB)")


if __name__ == "__main__":
    main()
