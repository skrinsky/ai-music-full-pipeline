#!/usr/bin/env python3
"""Download Slakh2100 from Zenodo with streaming tar extraction.

Two modes:
  Initial  (default): stream archive, extract first --n_tracks tracks (MIDI + stems + metadata).
  Continue (--continue_stems): stream archive again, fetch stems/ for tracks already downloaded.
                                Needed because the Zenodo tar lists per-track directories in
                                alphabetical order by path component, so stems/ come after MIDI/
                                in the archive and the initial extraction may miss them when
                                stopping early by track count.
"""

import argparse
import io
import tarfile
from pathlib import Path

import requests

DEFAULT_URL = "https://zenodo.org/records/4599666/files/slakh2100_flac_redux.tar.gz"


# --------------- streaming helper ----------------------------------------

class _StreamingIO(io.RawIOBase):
    """Wrap a requests iter_content stream as a file-like object for tarfile."""

    def __init__(self, iter_content):
        self._buf = b""
        self._iter = iter_content

    def readinto(self, b):
        while not self._buf:
            try:
                self._buf = next(self._iter)
            except StopIteration:
                return 0
        n = len(b)
        out = self._buf[:n]
        self._buf = self._buf[n:]
        b[: len(out)] = out
        return len(out)

    def readable(self):
        return True


def _open_archive(url: str):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    return tarfile.open(
        fileobj=io.BufferedReader(_StreamingIO(response.iter_content(chunk_size=1 << 20))),
        mode="r|gz",
    )


def _extract_member(tar, member, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if member.isfile():
        fobj = tar.extractfile(member)
        if fobj is not None:
            dest.write_bytes(fobj.read())
    elif member.isdir():
        dest.mkdir(parents=True, exist_ok=True)


# --------------- initial fetch -------------------------------------------

def fetch_initial(out_dir: Path, split: str, url: str, n_tracks: int):
    """Stream archive and extract the first n_tracks tracks (all files per track)."""
    prefix = f"slakh2100_flac_redux/{split}/Track"
    seen: set = set()

    print(f"Streaming {url}")
    print(f"Extracting first {n_tracks} tracks → {out_dir}/{split}/")

    with _open_archive(url) as tar:
        for member in tar:
            if not member.name.startswith(prefix):
                continue
            parts = member.name.split("/")
            if len(parts) < 3:
                continue
            track_name = parts[2]

            if n_tracks > 0 and track_name not in seen and len(seen) >= n_tracks:
                break

            if track_name not in seen:
                seen.add(track_name)
                print(f"  {track_name} ({len(seen)}/{n_tracks})", flush=True)

            rel  = "/".join(parts[1:])
            dest = out_dir / rel
            _extract_member(tar, member, dest)

    return seen


# --------------- continue: fetch stems for existing tracks ---------------

def fetch_stems(out_dir: Path, split: str, url: str):
    """Stream archive and fetch stems/ + metadata.yaml for already-downloaded tracks."""
    split_dir = out_dir / split
    wanted    = {t.name for t in split_dir.glob("Track*") if (t / "MIDI").exists()}
    has_stems = {
        t.name
        for t in split_dir.glob("Track*")
        if (t / "stems").exists() and list((t / "stems").glob("*.flac"))
    }
    missing = wanted - has_stems

    if not missing:
        print(f"All {len(wanted)} tracks already have stems. Nothing to do.")
        return has_stems

    print(f"Need stems for {len(missing)}/{len(wanted)} tracks.")
    print(f"Streaming {url} — this may require downloading most of the archive.")

    prefix    = f"slakh2100_flac_redux/{split}/Track"
    extracted: set = set()

    with _open_archive(url) as tar:
        for member in tar:
            if not member.name.startswith(prefix):
                continue
            parts = member.name.split("/")
            if len(parts) < 3:
                continue
            track_name = parts[2]

            if track_name not in wanted:
                continue  # not a track we care about

            # Skip MIDI — already have it
            inner = parts[3] if len(parts) > 3 else ""
            if inner == "MIDI":
                continue

            rel  = "/".join(parts[1:])
            dest = out_dir / rel
            _extract_member(tar, member, dest)

            # Track completion: at least one FLAC written for this track
            if inner == "stems":
                stems_dir = split_dir / track_name / "stems"
                if stems_dir.exists() and list(stems_dir.glob("*.flac")):
                    if track_name not in extracted:
                        extracted.add(track_name)
                        print(
                            f"  stems done: {track_name} "
                            f"({len(extracted)}/{len(missing)})",
                            flush=True,
                        )

            if extracted >= missing:
                print("All stems extracted — stopping early.")
                break

    return has_stems | extracted


# --------------- main ----------------------------------------------------

def main():
    ap = argparse.ArgumentParser("fetch_slakh: stream-download Slakh2100 from Zenodo.")
    ap.add_argument("--out_dir",        default="data/slakh")
    ap.add_argument("--n_tracks",       type=int, default=100,  help="Tracks for initial fetch.")
    ap.add_argument("--split",          default="train")
    ap.add_argument("--url",            default=DEFAULT_URL)
    ap.add_argument("--continue_stems", action="store_true",
                    help="Fetch stems/ for tracks already downloaded (MIDI present).")
    ap.add_argument("--force",          action="store_true",
                    help="Re-download even if sentinel exists.")
    args = ap.parse_args()

    out_dir  = Path(args.out_dir)
    sentinel = out_dir / ".fetched"

    if args.continue_stems:
        result = fetch_stems(out_dir, args.split, args.url)
        sentinel.write_text(f"fetched stems for {len(result)} tracks from {args.split}\n")
        print(f"\nSentinel updated: {sentinel}")
        return

    if sentinel.exists() and not args.force:
        print(
            "Already downloaded — delete data/slakh/.fetched to re-fetch, "
            "or use --continue_stems to add stems."
        )
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    seen = fetch_initial(out_dir, args.split, args.url, args.n_tracks)
    sentinel.write_text(f"extracted {len(seen)} tracks from {args.split}\n")
    print(f"\nDone. {len(seen)} tracks → {out_dir}")
    print(f"Sentinel written: {sentinel}")


if __name__ == "__main__":
    main()
