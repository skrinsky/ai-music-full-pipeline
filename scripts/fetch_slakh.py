#!/usr/bin/env python3
"""Download Slakh2100 from Zenodo with streaming tar extraction.

Stops after --n_tracks tracks without downloading the full 104 GB archive.
"""

import argparse
import io
import tarfile
from pathlib import Path

import requests

DEFAULT_URL = "https://zenodo.org/records/4599666/files/slakh2100_flac_redux.tar.gz"


def main():
    ap = argparse.ArgumentParser("fetch_slakh: stream-download Slakh2100 from Zenodo.")
    ap.add_argument("--out_dir",  default="data/slakh",   help="Output directory.")
    ap.add_argument("--n_tracks", type=int, default=100,  help="Tracks to extract (0 = all).")
    ap.add_argument("--split",    default="train",        help="Which split folder to extract.")
    ap.add_argument("--url",      default=DEFAULT_URL,    help="Zenodo download URL.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    sentinel = out_dir / ".fetched"

    if sentinel.exists():
        print("Already downloaded — delete data/slakh/.fetched to re-fetch.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"slakh2100_flac_redux/{args.split}/Track"

    print(f"Streaming {args.url}")
    print(f"Extracting split={args.split}, n_tracks={args.n_tracks or 'all'} → {out_dir}")

    response = requests.get(args.url, stream=True)
    response.raise_for_status()

    # Wrap the streaming response in a file-like object for tarfile.
    class _StreamingIO(io.RawIOBase):
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
            b[:len(out)] = out
            return len(out)

        def readable(self):
            return True

    stream_io = io.BufferedReader(_StreamingIO(response.iter_content(chunk_size=1 << 20)))

    seen_tracks = set()
    extracted = 0

    with tarfile.open(fileobj=stream_io, mode="r|gz") as tar:
        for member in tar:
            if not member.name.startswith(prefix):
                continue

            # Identify the Track directory (e.g. "slakh2100_flac_redux/train/Track00001")
            parts = member.name.split("/")
            # parts: ['slakh2100_flac_redux', 'train', 'Track00001', ...]
            if len(parts) < 3:
                continue
            track_name = parts[2]

            if args.n_tracks > 0 and track_name not in seen_tracks and len(seen_tracks) >= args.n_tracks:
                # We've already hit the limit and this is a new track — stop.
                break

            if track_name not in seen_tracks:
                seen_tracks.add(track_name)
                extracted += 1
                print(f"  extracting {track_name} ({extracted}/{args.n_tracks or '?'})", flush=True)

            # Strip the "slakh2100_flac_redux/" prefix so files land under out_dir/split/Track*/
            rel = "/".join(parts[1:])
            dest = out_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)

            if member.isfile():
                fobj = tar.extractfile(member)
                if fobj is not None:
                    dest.write_bytes(fobj.read())
            elif member.isdir():
                dest.mkdir(parents=True, exist_ok=True)

    sentinel.write_text(f"extracted {len(seen_tracks)} tracks from {args.split}\n")
    print(f"\nDone. {len(seen_tracks)} tracks → {out_dir}")
    print(f"Sentinel written: {sentinel}")


if __name__ == "__main__":
    main()
