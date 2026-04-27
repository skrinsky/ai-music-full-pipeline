"""
One-time DAW integration setup, run automatically when the server starts.

Handles:
  - pip-installing missing Python deps (python-reapy, python-osc, mido)
  - Configuring the reapy bridge files into REAPER's resource folder
  - Downloading and installing AbletonOSC into Live's Remote Scripts folder
"""

import subprocess
import sys
import threading
from pathlib import Path


# ── pip deps ──────────────────────────────────────────────────────────────────

_DEPS = ["python-reapy", "python-osc", "mido"]

def _install_deps():
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q"] + _DEPS,
            check=True,
            capture_output=True,
        )
        print("[daw_setup] Python deps OK")
    except subprocess.CalledProcessError as e:
        print(f"[daw_setup] pip install failed: {e.stderr.decode()[:200]}")


# ── Reaper ────────────────────────────────────────────────────────────────────

def _setup_reaper():
    try:
        import reapy  # type: ignore
    except ImportError:
        print("[daw_setup] python-reapy not yet importable — retry after pip step")
        return

    reaper_scripts = Path.home() / "Library/Application Support/REAPER/Scripts"
    bridge_file = reaper_scripts / "dist_api_enable.py"

    if bridge_file.exists():
        print("[daw_setup] reapy bridge already installed in REAPER")
        return

    try:
        reapy.config.configure_reaper()
        print("[daw_setup] reapy bridge installed.")
        print("[daw_setup] ACTION NEEDED: In REAPER → Actions → Show action list →")
        print("[daw_setup]   search 'dist_api_enable' → Run once.")
        print("[daw_setup]   After that, REAPER auto-connects every time.")
    except Exception as e:
        print(f"[daw_setup] reapy configure failed: {e}")


# ── Ableton Live (AbletonOSC) ─────────────────────────────────────────────────

_ABLETONOSC_URL = (
    "https://github.com/ideoforms/AbletonOSC/archive/refs/heads/main.zip"
)
_REMOTE_SCRIPTS = (
    Path.home() / "Music/Ableton/User Library/Remote Scripts"
)


def _setup_ableton():
    dest = _REMOTE_SCRIPTS / "AbletonOSC"
    if dest.exists():
        print("[daw_setup] AbletonOSC already in Ableton Remote Scripts")
        return

    _REMOTE_SCRIPTS.mkdir(parents=True, exist_ok=True)

    print("[daw_setup] Downloading AbletonOSC …")
    try:
        import urllib.request
        import zipfile
        import io
        import tempfile

        with urllib.request.urlopen(_ABLETONOSC_URL, timeout=15) as resp:
            data = resp.read()

        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            with tempfile.TemporaryDirectory() as tmp:
                zf.extractall(tmp)
                src = Path(tmp) / "AbletonOSC-main" / "AbletonOSC"
                if not src.exists():
                    print("[daw_setup] Unexpected zip layout — skipping AbletonOSC")
                    return
                import shutil
                shutil.copytree(src, dest)

        print("[daw_setup] AbletonOSC installed to Remote Scripts.")
        print("[daw_setup] ACTION NEEDED: Restart Ableton Live, then")
        print("[daw_setup]   Preferences → MIDI → Control Surface → select AbletonOSC.")
    except Exception as e:
        print(f"[daw_setup] AbletonOSC download failed: {e}")
        print("[daw_setup]   Manual install: https://github.com/ideoforms/AbletonOSC")


# ── public entry point ────────────────────────────────────────────────────────

def run_in_background():
    """Call at server startup — runs all setup steps in a daemon thread."""
    def _run():
        _install_deps()
        _setup_reaper()
        _setup_ableton()

    threading.Thread(target=_run, daemon=True, name="daw-setup").start()
