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
_venv_python = ROOT / ".venv-ai-music" / "bin" / "python"
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
    "daw_insert": None,    # 'reaper' | 'ableton' | '*_error' | 'unsupported' | None
}

# generated MIDI files keyed by job_id
_midi_files: dict[str, Path] = {}

# currently running subprocess (so /cancel can kill it)
_current_proc: Optional[subprocess.Popen] = None
_cancelled = threading.Event()


def _set_status(**kwargs):
    _status.update(kwargs)


# ── helpers ───────────────────────────────────────────────────────────────────

def _run_streaming(cmd: list[str], cwd: Path, parse_fn=None):
    """Run a subprocess, stream stdout, optionally parse each line.
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
    proc.wait()                # returns as soon as the main process exits
    reader.join(timeout=5.0)  # drain remaining output; don't hang if torch workers hold the pipe
    _current_proc = None
    return proc.returncode, tail


_EPOCH_RE = re.compile(
    r"Epoch\s+(\d+).*?val:\s+loss=([\d.]+)", re.IGNORECASE
)


def _parse_train_line(line: str):
    m = _EPOCH_RE.search(line)
    if m:
        _set_status(epoch=int(m.group(1)), val_loss=float(m.group(2)))


# ── request models ────────────────────────────────────────────────────────────

class ProcessRequest(BaseModel):
    audio_folder: str
    tracks: str = ""
    normalize_key: bool = True


class TrainRequest(BaseModel):
    events_dir: str = "runs/events"
    ckpt_path: str = "runs/checkpoints/es_model.pt"
    device: str = "auto"
    epochs: int = 200
    seq_len: int = 512


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


# ── endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"ok": True}


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


@app.post("/cancel")
def cancel():
    global _current_proc
    _cancelled.set()
    if _current_proc is not None and _current_proc.poll() is None:
        _current_proc.kill()  # SIGKILL — can't be ignored
    _set_status(stage="idle", message="cancelled", error=None,
                epoch=None, val_loss=None)
    return {"cancelled": True}


@app.post("/process")
def process(req: ProcessRequest):
    if not _job_lock.acquire(blocking=False):
        raise HTTPException(409, "Another job is already running")

    def run():
        try:
            _set_status(stage="processing", message="audio → MIDI → preprocess",
                        error=None, epoch=None, val_loss=None)

            audio_folder = str(Path(req.audio_folder).resolve())
            midi_dir     = str(ROOT / "out_midis")
            events_dir   = str(ROOT / "runs" / "events")

            pipe_dir = ROOT / "vendor" / "all-in-one-ai-midi-pipeline"

            # Step 1: audio → MIDI via vendor pipeline
            pipe_args = ["python", "pipeline.py", "run-batch",
                         f"{audio_folder}/*.wav"]
            if req.tracks:
                pipe_args += ["--tracks", req.tracks]
            if req.normalize_key:
                pipe_args += ["--normalize-key"]

            rc, tail = _run_streaming(pipe_args, cwd=pipe_dir)
            if rc != 0:
                if not _cancelled.is_set():
                    _set_status(stage="error", error="vendor pipeline failed: " + " | ".join(tail[-3:]))
                return

            # Step 2: export MIDI
            rc, tail = _run_streaming(
                ["python", "pipeline.py", "export-midi", "--out", midi_dir],
                cwd=pipe_dir,
            )
            if rc != 0:
                if not _cancelled.is_set():
                    _set_status(stage="error", error="MIDI export failed: " + " | ".join(tail[-3:]))
                return

            # Step 3: preprocess → events
            pre_args = [
                PYTHON, str(ROOT / "training" / "pre.py"),
                "--midi_folder", midi_dir,
                "--data_folder", events_dir,
            ]
            if req.tracks:
                pre_args += ["--tracks", req.tracks]

            rc, tail = _run_streaming(pre_args, cwd=ROOT)
            if rc != 0:
                if not _cancelled.is_set():
                    _set_status(stage="error", error="preprocessing failed: " + " | ".join(tail[-3:]))
                return

            _set_status(stage="done", message="processing complete")
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
        try:
            events_dir = str((ROOT / req.events_dir).resolve())
            ckpt_path  = str((ROOT / req.ckpt_path).resolve())

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
                "--epochs",     str(req.epochs),
                "--seq_len",    str(req.seq_len),
            ]

            rc, tail = _run_streaming(cmd, cwd=ROOT, parse_fn=_parse_train_line)
            if rc != 0:
                if not _cancelled.is_set():
                    _set_status(stage="error", error="training failed: " + " | ".join(tail[-3:]))
                return

            _set_status(stage="done", message="training complete")
        finally:
            _cancelled.clear()
            _job_lock.release()

    threading.Thread(target=run, daemon=True).start()
    return {"started": True}


@app.post("/generate")
def generate(req: GenerateRequest):
    if not _job_lock.acquire(blocking=False):
        raise HTTPException(409, "Another job is already running")

    job_id  = str(uuid.uuid4())[:8]
    out_dir = ROOT / "runs" / "generated" / "plugin" / job_id
    out_dir.mkdir(parents=True, exist_ok=True)

    def run():
        try:
            _set_status(stage="generating", message=f"job {job_id}",
                        error=None, epoch=None, val_loss=None)

            # Resolve vocab_json: use supplied path if it exists, otherwise
            # search every known event directory under ROOT.
            _EVENT_DIRS = [
                "runs/events", "runs/retrain_events", "runs/blues_events",
                "runs/chorale_events", "runs/chorale_dense_events",
                "runs/cascade_events_a", "runs/cascade_events_b",
                "runs/chorale_cascade_events",
            ]
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
                "--device",          "auto",
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
                        daw_insert=daw_result)
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
