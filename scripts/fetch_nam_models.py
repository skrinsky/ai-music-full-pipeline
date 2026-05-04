#!/usr/bin/env python3
"""Download free NAM amp captures for bass and guitar from GitHub.

Sources:
  - neural-amp-modeler test fixtures (guaranteed stable URLs)
  - pelennor2170/NAM_models community captures

Usage:
  python scripts/fetch_nam_models.py --out_dir data/nam_models
"""

import argparse
import json
import sys
import urllib.request
from pathlib import Path

# Stable .nam files from the official NAM repo test suite
NAM_TEST_BASE = "https://raw.githubusercontent.com/sdatkinson/neural-amp-modeler/main/tests/resources"

# Community captures from pelennor2170/NAM_models — bass + guitar selections
COMMUNITY_BASE = "https://raw.githubusercontent.com/pelennor2170/NAM_models/main"

MODELS = [
    # (filename, url, category)
    # Official test fixtures — small WaveNet models, always available
    ("test_bass.nam",   f"{NAM_TEST_BASE}/test_model_v1.nam",   "bass"),
    ("test_guitar.nam", f"{NAM_TEST_BASE}/test_model_v2.nam",   "guitar"),
]

# Community models — try these, skip silently if unavailable
COMMUNITY_MODELS = [
    ("ampeg_svt_bass.nam",     f"{COMMUNITY_BASE}/Bass/Ampeg_SVT.nam",          "bass"),
    ("darkglass_b7k_bass.nam", f"{COMMUNITY_BASE}/Bass/Darkglass_B7K.nam",      "bass"),
    ("jcm800_guitar.nam",      f"{COMMUNITY_BASE}/Guitar/Marshall_JCM800.nam",  "guitar"),
    ("5150_guitar.nam",        f"{COMMUNITY_BASE}/Guitar/EVH_5150.nam",         "guitar"),
]


def fetch(url: str, dest: Path) -> bool:
    try:
        urllib.request.urlretrieve(url, dest)
        # Verify it's valid JSON (.nam files are JSON)
        with open(dest) as f:
            json.load(f)
        return True
    except Exception as e:
        if dest.exists():
            dest.unlink()
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data/nam_models")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ok = 0
    for fname, url, category in MODELS + COMMUNITY_MODELS:
        dest = out / fname
        if dest.exists():
            print(f"  already exists: {fname}")
            ok += 1
            continue
        print(f"  fetching {fname} ...", end=" ", flush=True)
        if fetch(url, dest):
            print("ok")
            ok += 1
        else:
            print("not found (skip)")

    if ok == 0:
        print("\nNo models downloaded. Download .nam files manually from "
              "https://www.tone3000.com and place them in data/nam_models/")
        print("  bass captures  → data/nam_models/*bass*.nam or data/nam_models/bass/*.nam")
        print("  guitar captures → data/nam_models/*guitar*.nam or data/nam_models/guitar/*.nam")
        sys.exit(1)

    # Write a manifest so build_discriminator_data.py can find them by category
    manifest = {}
    for p in sorted(out.rglob("*.nam")):
        name = p.stem.lower()
        if "bass" in name:
            manifest.setdefault("bass", []).append(str(p))
        elif "guitar" in name:
            manifest.setdefault("guitar", []).append(str(p))
        else:
            manifest.setdefault("other", []).append(str(p))

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nManifest written to {out / 'manifest.json'}")
    for cat, paths in manifest.items():
        print(f"  {cat}: {len(paths)} model(s)")


if __name__ == "__main__":
    main()
