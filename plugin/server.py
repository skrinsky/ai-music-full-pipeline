#!/usr/bin/env python3
"""
Local FastAPI server that exposes the ai-music-full-pipeline to a DAW plugin.

Endpoints:
  GET  /health
  POST /process   { audio_folder, tracks?, normalize_key? }
  POST /train     { events_dir, ckpt_path?, device? }
  GET  /status
  POST /generate  { ckpt, vocab_json, seed_pkl?, temperature, top_p,
                    tempo_bpm, force_grid_step, n, top_k, min_score, max_tokens }
  GET  /midi/{job_id}

Start:
  python plugin/server.py --root /path/to/ai-music-full-pipeline --port 7437
"""

import argparse
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

import daw_insert
import daw_setup

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import uvicorn

# ── globals ───────────────────────────────────────────────────────────────────

ROOT: Path = Path(__file__).parent.parent.resolve()

# Prefer the repo's own venv so subprocesses (generate.py, train.py, pre.py)
# get the right packages regardless of which Python launched this server.
_venv_scripts = "Scripts" if sys.platform == "win32" else "bin"
_venv_exe     = "python.exe" if sys.platform == "win32" else "python"
_venv_python  = ROOT / ".venv" / _venv_scripts / _venv_exe
PYTHON = str(_venv_python) if _venv_python.exists() else sys.executable

app = FastAPI(title="AI Music Pipeline Server")

# One job runs at a time.
_job_lock = threading.Lock()

_status: dict = {
    "stage": "idle",       # idle | processing | training | generating | done | error
    "message": "",
    "epoch": None,
    "total_epochs": None,
    "train_loss": None,
    "val_loss": None,
    "error": None,
    "progress": None,      # 0.0–1.0 during preprocessing, None otherwise
    "batch_progress": None,  # 0.0–1.0 within current training epoch
    "events_dir": None,    # path to preprocessed events folder after processing
    "ckpt_path":  None,    # path to checkpoint written when training completes
    "daw_insert": None,    # 'reaper' | 'ableton' | '*_error' | 'unsupported' | None
    "midi_path":  None,    # absolute path to generated MIDI when generation completes
}

# generated MIDI files keyed by job_id
_midi_files: dict[str, Path] = {}

# currently running subprocess (so /cancel can kill it)
_current_proc: Optional[subprocess.Popen] = None
_cancelled = threading.Event()
_watchdog_restart = threading.Event()  # set by watchdog to request a restart

# Epoch timing — updated by _parse_train_line, read by watchdog
_epoch_start_ts:      float = 0.0   # wall time when current epoch's first batch fired
_last_epoch_duration: float = 0.0   # seconds the previous epoch took (0 = unknown)
_last_batch_ts:       float = 0.0   # wall time of last BATCH_PROGRESS line
_last_batch_val:      float = -1.0  # last batch_progress fraction seen


def _set_status(**kwargs):
    _status.update(kwargs)


# ── helpers ───────────────────────────────────────────────────────────────────

def _run_streaming(cmd: list[str], cwd: Path, parse_fn=None, on_start=None):
    """Run a subprocess, stream stdout, optionally parse each line.
    on_start(proc) is called immediately after the process launches (before blocking).
    Returns (returncode, last_lines) where last_lines is the tail of output."""
    global _current_proc
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(cwd),
    )
    _current_proc = proc
    if on_start:
        on_start(proc)
    tail: list[str] = []

    def _read():
        for line in proc.stdout:
            line = line.rstrip()
            try:
                print(line, flush=True)
            except BrokenPipeError:
                pass
            tail.append(line)
            if len(tail) > 20:
                tail.pop(0)
            if parse_fn:
                parse_fn(line)

    reader = threading.Thread(target=_read, daemon=True)
    reader.start()
    # Poll instead of blocking proc.wait() — if the process is stuck inside a
    # GPU driver (MPS/Metal deadlock) SIGKILL may never be delivered, and a
    # bare wait() would block forever.  Once a kill has been requested
    # (_watchdog_restart or _cancelled) give the process KILL_GRACE seconds to
    # exit on its own, then break out so the restart loop can continue.
    KILL_GRACE = 60.0
    kill_deadline: float = 0.0
    while True:
        try:
            proc.wait(timeout=1.0)
            break  # process exited normally
        except subprocess.TimeoutExpired:
            pass
        if _watchdog_restart.is_set() or _cancelled.is_set():
            if kill_deadline == 0.0:
                kill_deadline = time.time() + KILL_GRACE
            elif time.time() > kill_deadline:
                print("[server] process survived SIGKILL (Metal/GPU deadlock) -- forcing unblock for restart", flush=True)
                break
    reader.join(timeout=5.0)  # drain remaining output; don't hang if torch workers hold the pipe
    _current_proc = None
    return proc.poll() if proc.poll() is not None else -1, tail


_EPOCH_RE = re.compile(
    r"Epoch\s+(\d+).*?val:\s+loss=([\d.]+)", re.IGNORECASE
)
_BATCH_RE      = re.compile(r"^BATCH_PROGRESS\s+(\d+)/(\d+)$")
_PREPROCESS_RE = re.compile(r"^PREPROCESS\s+(\d+)/(\d+)$")


def _parse_train_line(line: str):
    global _epoch_start_ts, _last_epoch_duration, _last_batch_ts, _last_batch_val
    m = _EPOCH_RE.search(line)
    if m:
        now = time.time()
        if _epoch_start_ts > 0:
            _last_epoch_duration = now - _epoch_start_ts
        _epoch_start_ts = now          # reset for next epoch
        _last_batch_ts  = now          # epoch completion counts as progress
        _set_status(epoch=int(m.group(1)), val_loss=float(m.group(2)),
                    batch_progress=0.0)
        return
    m2 = _BATCH_RE.match(line)
    if m2:
        done, total = int(m2.group(1)), int(m2.group(2))
        frac = done / total if total > 0 else 0.0
        now  = time.time()
        if frac != _last_batch_val:    # only count actual movement
            _last_batch_ts  = now
            _last_batch_val = frac
            if _epoch_start_ts == 0:   # first batch ever — mark epoch start
                _epoch_start_ts = now
        _set_status(batch_progress=frac)


def _parse_preprocess_line(line: str):
    m = _PREPROCESS_RE.match(line)
    if m:
        done, total = int(m.group(1)), int(m.group(2))
        raw = done / total if total > 0 else 0.0
        _set_status(progress=0.40 + 0.60 * raw)  # step 3 fills 0.40 → 1.0


# ── request models ────────────────────────────────────────────────────────────

class ProcessRequest(BaseModel):
    audio_folder: str
    tracks: str = ""
    normalize_key: bool = True
    disc_intensity: float = 0.0   # 0 = off, 1.0 = max filtering
    project_name: str = ""        # if set, derives all output paths from runs/{project}/
    files_to_skip: list = []      # filenames (with ext) to use existing stems for


class TrainRequest(BaseModel):
    events_dir: str = "runs/events"
    ckpt_path: str = "runs/checkpoints/es_model.pt"
    device: str = "auto"
    epochs: int = 200
    seq_len: int = 512
    project_name: str = ""        # if set, overrides events_dir and ckpt_path
    pretrain_ckpt: str = ""       # if set, resume/fine-tune from this checkpoint
    force_restart: bool = False   # if True, ignore existing checkpoint and train from scratch


class GenerateRequest(BaseModel):
    ckpt: str
    vocab_json: str
    seed_pkl: str = ""
    use_seed: bool = False
    temperature: float = 0.75
    top_p: float = 0.95
    tempo_bpm: float = 75.0
    grid_straight_step: int = 6
    grid_triplet_step: int = 0
    max_tokens: int = 512
    project_name: str = ""


# ── endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"ok": True}


@app.get("/checkpoint_status")
def checkpoint_status(project_name: str = ""):
    """Return whether a checkpoint exists for a project and which epoch it's from."""
    import torch
    if project_name.strip():
        project_slug = re.sub(r"[^\w-]", "_", project_name.strip()) or "default"
        ckpt_path = ROOT / "runs" / project_slug / "checkpoints" / "model.pt"
    else:
        ckpt_path = ROOT / "runs" / "checkpoints" / "es_model.pt"
    if not ckpt_path.exists():
        return {"exists": False, "epoch": None}
    try:
        data = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        epoch = int(data.get("epoch", -1))
    except Exception:
        epoch = -1
    return {"exists": True, "epoch": epoch}


@app.get("/checkpoint_info")
def checkpoint_info(ckpt: str):
    import torch
    try:
        data = torch.load(ckpt, map_location="cpu", weights_only=False)
        seq_len = None
        for key in ("config", "model_config"):
            if key in data and "SEQ_LEN" in data[key]:
                seq_len = int(data[key]["SEQ_LEN"])
                break
        return {"seq_len": seq_len}
    except Exception as e:
        raise HTTPException(400, str(e))


@app.get("/status")
def status():
    return dict(_status)


@app.get("/latest_events")
def latest_events():
    """Return the most recently modified events folder that contains events_train.pkl.

    Scans both the new project layout (runs/{project}/events/) and the legacy
    layout (runs/events/{slug}/) so old sessions still resolve correctly.
    """
    runs_root = ROOT / "runs"
    best_path, best_mtime = None, 0.0
    if runs_root.exists():
        for pkl in runs_root.rglob("events_train.pkl"):
            mtime = pkl.stat().st_mtime
            if mtime > best_mtime:
                best_mtime = mtime
                best_path = str(pkl.parent)
    return {"events_dir": best_path}


@app.get("/disc_preview")
def disc_preview(events_dir: str = ""):
    """Return disc_preview.json from the given events dir (or the latest one)."""
    import json as _json
    if not events_dir:
        resp = latest_events()
        events_dir = resp.get("events_dir") or ""
    if not events_dir:
        raise HTTPException(404, "No events directory found")
    p = Path(events_dir) / "disc_preview.json"
    if not p.exists():
        raise HTTPException(404, "No preview data — re-run processing with discriminator enabled")
    with open(p) as f:
        return _json.load(f)


@app.post("/cancel")
def cancel():
    global _current_proc
    _cancelled.set()
    if _current_proc is not None and _current_proc.poll() is None:
        _current_proc.kill()  # SIGKILL — can't be ignored
    _set_status(stage="idle", message="cancelled", error=None,
                epoch=None, val_loss=None)
    # Shut down the server after the response is sent so it relaunches
    # fresh (with any updated code) on the next plugin action.
    def _shutdown():
        import time
        time.sleep(0.4)
        os._exit(0)
    threading.Thread(target=_shutdown, daemon=True).start()
    return {"cancelled": True}


@app.get("/check_existing")
def check_existing(audio_folder: str):
    """Return filenames that already have stems from a previous run."""
    stems_root = ROOT / "vendor" / "all-in-one-ai-midi-pipeline" / "data" / "stems" / "htdemucs_6s"
    audio_formats = ["*.wav", "*.mp3", "*.flac", "*.aiff", "*.aif", "*.m4a", "*.ogg"]
    folder = Path(audio_folder).resolve()
    if not folder.exists():
        return {"existing": []}
    existing = [
        f.name
        for fmt in audio_formats
        for f in folder.rglob(fmt)
        if (stems_root / f.stem).exists()
    ]
    return {"existing": existing}


@app.post("/process")
def process(req: ProcessRequest):
    if not _job_lock.acquire(blocking=False):
        raise HTTPException(409, "Another job is already running")

    def run():
        try:
            _set_status(stage="processing", message="starting…",
                        error=None, epoch=None, val_loss=None, events_dir=None,
                        progress=0.0)

            audio_folder = str(Path(req.audio_folder).resolve())
            if req.project_name.strip():
                project_slug = re.sub(r"[^\w-]", "_", req.project_name.strip()) or "default"
            else:
                project_slug = re.sub(r"[^\w-]", "_", Path(audio_folder).name) or "default"
            events_dir = str(ROOT / "runs" / project_slug / "events")
            midi_dir   = str(ROOT / "runs" / project_slug / "midis")

            pipe_dir  = ROOT / "vendor" / "all-in-one-ai-midi-pipeline"
            midi_data = pipe_dir / "data" / "midi"

            # Step 1: audio → MIDI (progress 0.0 → 0.35)
            audio_formats = ["*.wav", "*.mp3", "*.flac", "*.aiff", "*.aif", "*.m4a", "*.ogg"]
            audio_folder_path = Path(audio_folder)
            formats_found = [
                fmt for fmt in audio_formats
                if any(audio_folder_path.rglob(fmt))
            ]

            if not formats_found:
                _set_status(stage="error",
                            error=f"no audio files found in {audio_folder} "
                                  "(looked for wav, mp3, flac, aiff, m4a, ogg)")
                return

            extra = []
            if req.tracks:
                extra += ["--tracks", req.tracks]
            if req.normalize_key:
                extra += ["--normalize-key"]

            # Record time so we can detect tracks written/overwritten by THIS run
            import time as _time
            _run_start = _time.time()

            # total_audio set after files_to_run is built (below)
            stems_dir = pipe_dir / "data" / "stems" / "htdemucs_6s"

            _set_status(message="step 1/3: audio → MIDI", progress=0.02)

            # Background thread: watch for stem folders written by THIS run → progress 0.02→0.35
            _step1_done = threading.Event()
            def _watch_stems():
                while not _step1_done.is_set():
                    if stems_dir.exists():
                        new_stems = len([d for d in stems_dir.iterdir()
                                         if d.is_dir() and d.stat().st_mtime >= _run_start])
                        if total_audio > 0:
                            frac = min(new_stems / total_audio, 1.0)
                            _set_status(progress=0.02 + 0.33 * frac)
                    _step1_done.wait(timeout=2.0)

            watcher = threading.Thread(target=_watch_stems, daemon=True)
            watcher.start()

            # Determine which files to actually run through the pipeline
            skip_names = set(req.files_to_skip)
            files_to_run = [
                f for fmt in audio_formats
                for f in audio_folder_path.rglob(fmt)
                if f.name not in skip_names
            ]
            total_audio = len(files_to_run)

            for audio_file in files_to_run:
                rc, tail = _run_streaming(
                    ["python", "pipeline.py", "run-batch", str(audio_file)] + extra,
                    cwd=pipe_dir,
                )
                if rc != 0:
                    _step1_done.set()
                    if not _cancelled.is_set():
                        _set_status(stage="error",
                                    error=f"vendor pipeline failed ({audio_file.name}): " + " | ".join(tail[-3:]))
                    return

            _step1_done.set()
            watcher.join(timeout=3.0)

            # Step 2: collect MIDI for all audio files in the folder
            # When reprocessing: only tracks touched this run (mtime check)
            # When using existing: all tracks whose song IDs match files in the audio folder
            _set_status(message="step 2/3: export MIDI", progress=0.35)
            import shutil
            skip_sids = {Path(fn).stem for fn in req.files_to_skip}
            new_midi_tracks = [
                d for d in (midi_data.iterdir() if midi_data.exists() else [])
                if d.is_dir() and (d.stat().st_mtime >= _run_start or d.name in skip_sids)
            ]
            if not new_midi_tracks:
                _set_status(stage="error", error="no new MIDI tracks produced — check audio files")
                return
            out_path = Path(midi_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            for track_dir in new_midi_tracks:
                for mid in track_dir.glob("*.mid"):
                    shutil.copy2(mid, out_path / mid.name)
            print(f"[process] exported {len(new_midi_tracks)} tracks to {midi_dir}", flush=True)

            # Step 3: preprocess → events (progress 0.40 → 1.0)
            _set_status(message="step 3/3: preprocessing", progress=0.40)
            pre_args = [
                PYTHON, "-m", "training.pre",
                "--midi_folder", midi_dir,
                "--data_folder", events_dir,
                "--force",
            ]
            if req.tracks:
                pre_args += ["--tracks", req.tracks]

            # Use discriminator only when user requests it (disc_intensity > 0)
            if req.disc_intensity > 0.0:
                for _disc_name in ("combined_model.pt", "model.pt"):
                    _disc_path = ROOT / "runs" / "discriminator" / _disc_name
                    if _disc_path.exists():
                        # Map intensity [0,1] → threshold and bp_blend
                        _threshold = 0.05 + req.disc_intensity * 0.50
                        _bp_blend  = 0.95 - req.disc_intensity * 0.15
                        pre_args += ["--discriminator",   str(_disc_path),
                                     "--disc_threshold",  str(round(_threshold, 4)),
                                     "--disc_bp_blend",   str(round(_bp_blend,  4))]
                        print(f"[process] discriminator: {_disc_name}  "
                              f"intensity={req.disc_intensity:.2f}  "
                              f"threshold={_threshold:.3f}  bp_blend={_bp_blend:.3f}", flush=True)
                        _stems_dir = pipe_dir / "data" / "stems" / "htdemucs_6s"
                        if _stems_dir.exists():
                            pre_args += ["--stems_dir", str(_stems_dir)]
                        break

            rc, tail = _run_streaming(pre_args, cwd=ROOT, parse_fn=_parse_preprocess_line)
            if rc != 0:
                if not _cancelled.is_set():
                    _set_status(stage="error", error="preprocessing failed: " + " | ".join(tail[-3:]))
                return

            _set_status(stage="done", message="processing complete",
                        events_dir=events_dir, progress=1.0)
        finally:
            _cancelled.clear()
            _job_lock.release()

    threading.Thread(target=run, daemon=True).start()
    return {"started": True}


@app.post("/train")
def train(req: TrainRequest):
    if not _job_lock.acquire(blocking=False):
        raise HTTPException(409, "Another job is already running")

    def run():
        global _epoch_start_ts, _last_epoch_duration, _last_batch_ts, _last_batch_val
        try:
            if req.project_name.strip():
                project_slug = re.sub(r"[^\w-]", "_", req.project_name.strip()) or "default"
                events_dir   = str(ROOT / "runs" / project_slug / "events")
                ckpt_dir     = ROOT / "runs" / project_slug / "checkpoints"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                ckpt_path    = str(ckpt_dir / "model.pt")
            else:
                events_dir = str((ROOT / req.events_dir).resolve())
                ckpt_path  = str((ROOT / req.ckpt_path).resolve())

            while True:   # restart loop — re-enters on watchdog kill
                # Reset timing state for fresh epoch measurement each run
                _epoch_start_ts = _last_batch_ts = 0.0
                _last_epoch_duration = _last_batch_val = 0.0
                _watchdog_restart.clear()

                _set_status(stage="training", message="training started",
                            error=None, epoch=0, total_epochs=req.epochs,
                            train_loss=None, val_loss=None)

                cmd = [
                    PYTHON, str(ROOT / "training" / "train.py"),
                    "--data_dir",   events_dir,
                    "--train_pkl",  str(Path(events_dir) / "events_train.pkl"),
                    "--val_pkl",    str(Path(events_dir) / "events_val.pkl"),
                    "--vocab_json", str(Path(events_dir) / "event_vocab.json"),
                    "--save_path",  ckpt_path,
                    "--device",     req.device,
                    "--seq_len",    str(req.seq_len),
                ]

                # Auto-resume: continue from existing checkpoint unless force_restart requested.
                # Fine-tune base checkpoint takes priority; otherwise use the project checkpoint.
                if req.force_restart:
                    resume_from = ""
                    cmd += ["--reset_best_val"]
                    print("[train] force_restart=True -- training from scratch", flush=True)
                else:
                    resume_from = req.pretrain_ckpt or (ckpt_path if Path(ckpt_path).exists() else "")
                if resume_from:
                    cmd += ["--resume", resume_from]
                    label = "fine-tuning from" if req.pretrain_ckpt else "resuming from"
                    print(f"[train] {label}: {resume_from}", flush=True)

                # Watchdog: kills subprocess if batch_progress stalls for > 3x epoch duration
                def _watchdog(proc_ref):
                    global _last_batch_ts
                    POLL = 15          # check every 15 s
                    FALLBACK = 1800.0  # 30 min before we know epoch duration
                    last_poll_ts = time.time()
                    while not _cancelled.is_set():
                        time.sleep(POLL)
                        now = time.time()
                        # If wall time jumped >> POLL the system was asleep.
                        # Reset the stall timer so we don't treat sleep time as a training stall.
                        if now - last_poll_ts > POLL * 3:
                            gap = int(now - last_poll_ts)
                            print(f"[watchdog] system sleep detected ({gap}s gap) -- resetting stall timer", flush=True)
                            _last_batch_ts = now
                        last_poll_ts = now
                        if proc_ref.poll() is not None:
                            break
                        if _last_batch_ts == 0:
                            continue
                        timeout = (3.0 * _last_epoch_duration
                                   if _last_epoch_duration > 0 else FALLBACK)
                        elapsed = time.time() - _last_batch_ts
                        if elapsed > timeout:
                            mins     = int(elapsed / 60)
                            dur_mins = round(_last_epoch_duration / 60, 1) if _last_epoch_duration > 0 else "?"
                            print(f"[watchdog] stall {mins}m (epoch ~{dur_mins}m) -- killing", flush=True)
                            _set_status(message=f"Stall detected ({mins} min) -- restarting...")
                            _watchdog_restart.set()
                            proc_ref.kill()
                            break

                rc, tail = _run_streaming(cmd, cwd=ROOT, parse_fn=_parse_train_line,
                                          on_start=lambda p: threading.Thread(
                                              target=_watchdog, args=(p,), daemon=True).start())

                if _watchdog_restart.is_set() and not _cancelled.is_set():
                    time.sleep(2)   # brief pause before restarting
                    continue        # loop back — will auto-resume from checkpoint

                if rc != 0:
                    if not _cancelled.is_set():
                        _set_status(stage="error", error="training failed: " + " | ".join(tail[-3:]))
                    return

                _set_status(stage="done", message="training complete", ckpt_path=ckpt_path)
                return

        finally:
            _cancelled.clear()
            _job_lock.release()

    threading.Thread(target=run, daemon=True).start()
    return {"started": True}


@app.post("/generate")
def generate(req: GenerateRequest):
    if not _job_lock.acquire(blocking=False):
        raise HTTPException(409, "Another job is already running")

    job_id = str(uuid.uuid4())[:8]
    if req.project_name.strip():
        _proj_slug = re.sub(r"[^\w-]", "_", req.project_name.strip()) or "default"
        out_dir = ROOT / "runs" / _proj_slug / "generated" / job_id
    else:
        out_dir = ROOT / "runs" / "generated" / "plugin" / job_id
    out_dir.mkdir(parents=True, exist_ok=True)

    def run():
        try:
            _set_status(stage="generating", message=f"job {job_id}",
                        error=None, epoch=None, val_loss=None)

            # Resolve vocab_json: use supplied path if it exists, otherwise
            # search every known event directory under ROOT.
            # Project-specific dir goes first (most likely match); then a glob
            # sweep picks up any other project dirs; legacy fixed paths follow.
            _proj_events = (
                [f"runs/{_proj_slug}/events"] if req.project_name.strip() else []
            )
            _glob_events = sorted(
                str(p.relative_to(ROOT))
                for p in ROOT.glob("runs/*/events")
                if p.is_dir()
            )
            _EVENT_DIRS = list(dict.fromkeys(
                _proj_events + _glob_events + [
                    "runs/events", "runs/retrain_events", "runs/blues_events",
                    "runs/chorale_events", "runs/chorale_dense_events",
                    "runs/cascade_events_a", "runs/cascade_events_b",
                    "runs/chorale_cascade_events",
                ]
            ))
            vocab_path: Optional[Path] = None
            if req.vocab_json:
                p = Path(req.vocab_json).resolve()
                if p.exists():
                    vocab_path = p

            if vocab_path is None:
                # Find the vocab whose token count matches the checkpoint embedding size.
                import torch, json as _json
                try:
                    _ckpt = torch.load(str(Path(req.ckpt).resolve()),
                                       map_location="cpu")
                    _state = _ckpt.get("model_state") or _ckpt.get("model_state_dict") or {}
                    _emb = _state.get("tok_emb.weight")
                    required_V = int(_emb.shape[0]) if _emb is not None else None
                except Exception as _e:
                    required_V = None
                    print(f"[generate] could not read checkpoint size: {_e}")

                for rel in _EVENT_DIRS:
                    candidate = ROOT / rel / "event_vocab.json"
                    if not candidate.exists():
                        continue
                    if required_V is not None:
                        try:
                            layout = _json.load(open(candidate))["layout"]
                            V = max(s["start"] + s["size"] for s in layout.values())
                            if V != required_V:
                                continue
                        except Exception:
                            continue
                    vocab_path = candidate
                    print(f"[generate] matched vocab (V={required_V}): {vocab_path}")
                    break

            if vocab_path is None:
                _set_status(stage="error",
                            error=f"no event_vocab.json with V={required_V} found — "
                                  "download the matching vocab into any runs/*/event_vocab.json")
                return

            # Resolve seed: explicit path wins; use_seed auto-finds events_val.pkl
            seed_pkl = req.seed_pkl
            if req.use_seed and not seed_pkl and vocab_path is not None:
                candidate_seed = vocab_path.parent / "events_val.pkl"
                if candidate_seed.exists():
                    seed_pkl = str(candidate_seed)
                    print(f"[generate] seeding from {candidate_seed}")
                else:
                    print(f"[generate] seed requested but {candidate_seed} not found — generating randomly")

            out_mid = out_dir / "generated.mid"
            cmd = [
                PYTHON, str(ROOT / "training" / "generate_v2.py"),
                "--ckpt",            str(Path(req.ckpt).resolve()),
                "--vocab_json",      str(vocab_path),
                "--out_midi",        str(out_mid),
                "--temperature",     str(req.temperature),
                "--top_p",           str(req.top_p),
                "--tempo_bpm",       str(req.tempo_bpm),
                "--force_grid_step",   str(req.grid_straight_step),
                "--grid_triplet_step", str(req.grid_triplet_step),
                "--max_tokens",      str(req.max_tokens),
                "--device",          "cuda" if torch.cuda.is_available() else "cpu",
            ]
            if seed_pkl:
                cmd += ["--seed_pkl", str(Path(seed_pkl).resolve())]

            rc, tail = _run_streaming(cmd, cwd=ROOT)
            if rc != 0:
                if not _cancelled.is_set():
                    _set_status(stage="error", error=" | ".join(tail[-5:]) or "generation failed")
                return

            if not out_mid.exists():
                if not _cancelled.is_set():
                    _set_status(stage="error", error="no MIDI produced")
                return

            _midi_files[job_id] = out_mid

            try:
                daw_result = daw_insert.insert_midi(str(out_mid))
            except Exception as e:
                print(f"[generate] daw_insert error (ignored): {e}")
                daw_result = "insert_error"

            _set_status(stage="done", message=f"midi_id={job_id}",
                        daw_insert=daw_result, midi_path=str(out_mid))
        except Exception as e:
            if not _cancelled.is_set():
                _set_status(stage="error", error=str(e))
        finally:
            _cancelled.clear()
            _job_lock.release()

    threading.Thread(target=run, daemon=True).start()
    return {"job_id": job_id, "started": True}


@app.get("/midi/{job_id}")
def get_midi(job_id: str):
    path = _midi_files.get(job_id)
    if path is None or not path.exists():
        raise HTTPException(404, "MIDI not found — job may still be running")
    return FileResponse(str(path), media_type="audio/midi",
                        filename=path.name)


@app.get("/preview/{job_id}")
def get_preview(job_id: str, fs: int = 44100, bpm: float = 0.0):
    """Return a WAV preview of the generated MIDI.

    Uses FluidSynth + GM soundfont when available for proper instrument sounds;
    falls back to pretty_midi's numpy synthesizer otherwise.
    Pass bpm to render at the DAW session tempo instead of the MIDI's embedded tempo.
    Results are cached so repeated requests with the same fs+bpm are instant.
    """
    midi_path = _midi_files.get(job_id)
    if midi_path is None or not midi_path.exists():
        raise HTTPException(404, "MIDI not found — job may still be running")

    bpm_tag = f"_{int(round(bpm))}" if bpm > 0 else ""
    wav_path = midi_path.parent / f"preview_{fs}{bpm_tag}.wav"
    if not wav_path.exists():
        _render_preview_wav(midi_path, wav_path, fs, bpm)

    if not wav_path.exists():
        raise HTTPException(500, "Preview render failed — check server logs")

    return FileResponse(str(wav_path), media_type="audio/wav",
                        filename="preview.wav")


def _find_sf2() -> "str | None":
    """Return path to a GM SoundFont, or None."""
    if sys.platform == "win32":
        candidates = [
            os.path.expanduser("~/soundfonts/FluidR3_GM.sf2"),
            "C:/soundfonts/FluidR3_GM.sf2",
            "C:/soundfonts/default.sf2",
            "C:/Program Files/Music/soundfonts/FluidR3_GM.sf2",
            "C:/Program Files/Music/soundfonts/default.sf2",
        ]
    elif sys.platform == "darwin":
        candidates = [
            os.path.expanduser("~/Library/Audio/Sounds/Banks/FluidR3_GM.sf2"),
        ]
    else:
        candidates = [
            "/usr/share/sounds/sf2/FluidR3_GM.sf2",
            "/usr/local/share/soundfonts/FluidR3_GM.sf2",
            "/usr/local/share/soundfonts/default.sf2",
            "/usr/share/soundfonts/default.sf2",
        ]
    return next((p for p in candidates if os.path.isfile(p)), None)


def _render_preview_wav(midi_path: "Path", wav_path: "Path", fs: int,
                        target_bpm: float = 0.0) -> None:
    import shutil as _sh, tempfile
    import pretty_midi, numpy as np, scipy.io.wavfile

    # Determine if we need to rescale from the MIDI's embedded tempo to target_bpm
    need_rescale = False
    scale = 1.0
    if target_bpm > 0:
        try:
            _probe = pretty_midi.PrettyMIDI(str(midi_path))
            _, tempos = _probe.get_tempo_changes()
            embedded_bpm = float(tempos[0]) if len(tempos) > 0 else 120.0
            if abs(embedded_bpm - target_bpm) > 0.5:
                need_rescale = True
                scale = embedded_bpm / target_bpm
        except Exception:
            pass

    # 1. FluidSynth (best quality, proper GM sounds) — only when no BPM rescaling needed
    #    because FluidSynth reads the raw MIDI file and would ignore our in-memory edits.
    if not need_rescale:
        sf2 = _find_sf2()
        if sf2 and _sh.which("fluidsynth"):
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp_path = tmp.name
                # file must be closed before fluidsynth opens it (required on Windows)
                subprocess.run(
                    ["fluidsynth", "-ni", "-F", tmp_path,
                     "-r", str(fs), sf2, str(midi_path)],
                    check=True, capture_output=True, timeout=120,
                )
                _sh.move(tmp_path, wav_path)
                return
            except Exception as exc:
                print(f"[preview] FluidSynth failed: {exc}")

    # 2. pretty_midi synthesis (with optional BPM rescale)
    try:
        pm = pretty_midi.PrettyMIDI(str(midi_path))
        if need_rescale:
            for inst in pm.instruments:
                for note in inst.notes:
                    note.start *= scale
                    note.end   *= scale
                for cc in inst.control_changes:
                    cc.time *= scale
                for pb in inst.pitch_bends:
                    pb.time *= scale
                for pc in inst.program_changes:
                    pc.time *= scale
        audio = pm.synthesize(fs=fs)
        if len(audio) > 0:
            peak = np.abs(audio).max()
            if peak > 0:
                audio = audio / peak * 0.8
            scipy.io.wavfile.write(str(wav_path), fs,
                                   (audio * 32767).astype(np.int16))
    except Exception as exc:
        print(f"[preview] pretty_midi synthesis failed: {exc}")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT),
                    help="Path to ai-music-full-pipeline repo root")
    ap.add_argument("--port", type=int, default=7437)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    ROOT = Path(args.root).resolve()
    print(f"Pipeline root: {ROOT}")
    print(f"Server: http://{args.host}:{args.port}")

    daw_setup.run_in_background()

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
