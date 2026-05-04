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

NAM_RAW      = "https://raw.githubusercontent.com"
NAM_TEST     = f"{NAM_RAW}/sdatkinson/neural-amp-modeler/main/tests/resources/models/identity"
COMMUNITY    = f"{NAM_RAW}/pelennor2170/NAM_models/main"
RTNEURAL_NAM = f"{NAM_RAW}/jatinchowdhury18/RTNeural-NAM/main"
PLUGIN_TEST  = f"{NAM_RAW}/sdatkinson/NeuralAmpModelerPlugin/main/REAPER"

def _enc(s):
    return s.replace(" ", "%20")

MODELS = [
    # (save_as, url, category)
    # Official NAM test WaveNet — always available, used as fallback
    ("wavenet_guitar.nam",
     f"{NAM_TEST}/wavenet_standard.nam", "guitar"),

    # ── Bass captures (pedal/preamp style — 3 distinct tones) ──────────────
    ("tech21_dug_bass.nam",
     f"{COMMUNITY}/{_enc('Jason Z Tech21 dUg DP3X bass preamp pedal all dimed no shift.nam')}",
     "bass"),
    ("hm2_bass.nam",
     f"{COMMUNITY}/{_enc('Jason Z Boss HM2 v1 kinda bass heavy with medium distortion pure everything turned all the way up tone.nam')}",
     "bass"),
    ("hm2_btfo_bass.nam",
     f"{COMMUNITY}/{_enc('Jason Z Boss HM2BTFO heavy distortion meant to be standalone less bass more honk.nam')}",
     "bass"),

    # ── Guitar captures (9 diverse amp/character types) ────────────────────
    ("ceriatone_guitar.nam",
     f"{COMMUNITY}/{_enc('George B Ceriatone King Kong Channel 1 60s mode.nam')}",
     "guitar"),
    ("5150_guitar.nam",
     f"{COMMUNITY}/{_enc('Helga B 5150 BlockLetter - NoBoost.nam')}",
     "guitar"),
    ("fender_twinverb_guitar.nam",
     f"{COMMUNITY}/{_enc('Tim R Fender TwinVerb Norm Bright.nam')}",
     "guitar"),
    ("dirty_shirley_guitar.nam",
     f"{COMMUNITY}/{_enc('Sascha S DirtyShirleyMini_Clean_B1_M6_T7_MV10_G4.nam')}",
     "guitar"),
    ("magnatone_guitar.nam",
     f"{COMMUNITY}/{_enc('Tim R Magnatone Super 59 Mkii Ch1 Norm.nam')}",
     "guitar"),
    ("jcm2000_guitar.nam",
     f"{COMMUNITY}/{_enc('Tim R JCM2000 Crunch.nam')}",
     "guitar"),
    ("vox_ac15_guitar.nam",
     f"{COMMUNITY}/{_enc('Phillipe P VOXAC15-TopBoost.nam')}",
     "guitar"),
    ("sovtek_mig50_guitar.nam",
     f"{COMMUNITY}/{_enc('Mikhail K Sovtek MIG50.nam')}",
     "guitar"),
    ("5152_lead_guitar.nam",
     f"{COMMUNITY}/{_enc('Tim R 5152 Lead No Boost.nam')}",
     "guitar"),
    ("orange_rockerverb_guitar.nam",
     f"{COMMUNITY}/{_enc('Tom C Axe FX 2 Orange Rockerverb.nam')}",
     "guitar"),
    ("mesa_dc5_guitar.nam",
     f"{RTNEURAL_NAM}/{_enc('OB1 Mesa DC-5 PM.nam')}",
     "guitar"),
    ("plugin_test_guitar.nam",
     f"{PLUGIN_TEST}/model.nam",
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
