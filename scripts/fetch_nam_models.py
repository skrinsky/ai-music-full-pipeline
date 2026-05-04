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

NAM_RAW = "https://raw.githubusercontent.com"
NAM_TEST  = f"{NAM_RAW}/sdatkinson/neural-amp-modeler/main/tests/resources/models/identity"
COMMUNITY = f"{NAM_RAW}/pelennor2170/NAM_models/main"

def _enc(s):
    return s.replace(" ", "%20")

MODELS = [
    # (save_as, url, category)
    # Official NAM test WaveNet — always available, used as fallback
    ("wavenet_guitar.nam",
     f"{NAM_TEST}/wavenet_standard.nam", "guitar"),

    # Community bass captures (pedal/preamp style but real bass tones)
    ("tech21_dug_bass.nam",
     f"{COMMUNITY}/{_enc('Jason Z Tech21 dUg DP3X bass preamp pedal all dimed no shift.nam')}",
     "bass"),
    ("hm2_bass.nam",
     f"{COMMUNITY}/{_enc('Jason Z Boss HM2 v1 kinda bass heavy with medium distortion pure everything turned all the way up tone.nam')}",
     "bass"),

    # Community guitar captures
    ("ceriatone_guitar.nam",
     f"{COMMUNITY}/{_enc('George B Ceriatone King Kong Channel 1 60s mode.nam')}",
     "guitar"),
    ("5150_guitar.nam",
     f"{COMMUNITY}/{_enc('Helga B 5150 BlockLetter - NoBoost.nam')}",
     "guitar"),
]


def fetch(url: str, dest: Path) -> bool:
    import ssl
    try:
        urllib.request.urlretrieve(url, dest)
    except Exception:
        # Fall back to unverified SSL (safe for public GitHub raw URLs)
        try:
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            with urllib.request.urlopen(url, context=ctx) as r:
                dest.write_bytes(r.read())
        except Exception:
            if dest.exists():
                dest.unlink()
            return False
    try:
        with open(dest) as f:
            json.load(f)
        return True
    except Exception:
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
    for fname, url, category in MODELS:
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
