"""HTTP API and static interface. Run: uvicorn server.app:app --port 8000

All JSON. Errors: {"detail": "<what went wrong and what to do>"} with a 4xx or 5xx status.

  GET  /api/models
       -> [ModelInfo dict]                          name, label, native, open_weights, available, note
  GET  /api/datasets
       -> [{"name", "title", "modality", "labels", "n"}]
  GET  /api/senses?model=<name>
       -> [{"bridge", "kind", "modality", "dataset", "card": ReportCard dict}]
  DELETE /api/senses?model=<name>&modality=<modality>
       -> {"removed": bool}
  POST /api/fit                                     body: {"model", "dataset", "modality": "audio", "scope": "closed"|"open"}
       -> {"job_id": str}                           409 if a run is already in progress
  GET  /api/jobs/<job_id>
       -> JobState dict                             id, request, status, events[], cards[], installed, error
  GET  /api/samples?dataset=<name>
       -> [{"id", "label", "name"}]                 held-out example files the interface can offer
  GET  /api/samples/<id>
       -> the audio file
  POST /api/chat                                    multipart form: model, message, files (0..n uploads), sample_ids (0..n)
       -> {"reply": str, "used": [{"bridge", "kind", "source", "detail"}]}
          `used` lists how each file reached the model; a file whose modality has no installed sense is
          reported with kind "none" and is not shown to the model.
  GET  /                                            web/index.html; other files under web/ are served at /static/

The server holds no model itself: it calls sense.models, sense.registry, sense.datasets and agent.jobs.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import threading
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel as RequestBody

from agent import jobs
from sense import datasets, models, registry
from sense.config import REPO
from sense.contracts import FitRequest, SenseInput

WEB = REPO / "web"
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".au"}
SCOPES = ("closed", "open")
CHAT_TOKENS = 256  # the default 96 cut Qwen3.5-0.8B's replies off mid-sentence

app = FastAPI(title="Sense harness")
_sample_paths: dict[str, Path] = {}  # opaque id -> path, filled only from datasets.sample_files
_chat_lock = threading.Lock()  # one model call at a time: the model objects are not thread safe


class FitBody(RequestBody):
    model: str
    dataset: str
    modality: str = "audio"
    scope: str = "closed"


# ----------------------------------------------------------------------------- helpers
def _bad_request(detail: str) -> HTTPException:
    return HTTPException(status_code=400, detail=detail)


def _check_model(name: str) -> None:
    known = [m.name for m in models.available_models()]
    if name not in known:
        raise _bad_request(f"Unknown model '{name}'. Choose one of: {', '.join(known) or 'none available'}.")


def _check_dataset(name: str) -> None:
    known = [d["name"] for d in datasets.available_sets()]
    if name not in known:
        raise _bad_request(f"Unknown dataset '{name}'. Choose one of: {', '.join(known) or 'none available'}.")


def _sample_id(path: str) -> str:
    return hashlib.sha1(str(path).encode()).hexdigest()[:12]


def _modality_of(name: str) -> str | None:
    return "audio" if Path(name).suffix.lower() in AUDIO_EXTENSIONS else None


def _remember_samples(dataset: str) -> list[dict]:
    """List the held-out files of a dataset and remember which ids may be served."""
    out = []
    for item in datasets.sample_files(dataset):
        path = Path(item["path"])
        sid = _sample_id(str(path))
        _sample_paths[sid] = path
        out.append({"id": sid, "label": item["label"], "name": path.name})
    return out


def _sample_path(sample_id: str) -> Path | None:
    if sample_id not in _sample_paths:  # e.g. after a server restart: rebuild from the listed sets
        for d in datasets.available_sets():
            _remember_samples(d["name"])
    return _sample_paths.get(sample_id)


# ----------------------------------------------------------------------------- endpoints
@app.get("/api/models")
def api_models() -> list[dict]:
    return [m.to_dict() for m in models.available_models()]


@app.get("/api/datasets")
def api_datasets() -> list[dict]:
    return datasets.available_sets()


@app.get("/api/senses")
def api_senses(model: str) -> list[dict]:
    _check_model(model)
    return registry.installed(model)


@app.delete("/api/senses")
def api_remove_sense(model: str, modality: str) -> dict:
    _check_model(model)
    return {"removed": registry.uninstall(model, modality)}


@app.post("/api/fit")
def api_fit(body: FitBody) -> dict:
    _check_model(body.model)
    _check_dataset(body.dataset)
    if body.scope not in SCOPES:
        raise _bad_request(f"Unknown scope '{body.scope}'. Use 'closed' or 'open'.")
    request = FitRequest(model=body.model, dataset=body.dataset, modality=body.modality, scope=body.scope)
    try:
        return {"job_id": jobs.start(request)}
    except RuntimeError as exc:
        text = str(exc)
        raise HTTPException(status_code=409, detail=f"{text[:1].upper()}{text[1:]}. Wait for it to finish, then start again.")


@app.get("/api/jobs/{job_id}")
def api_job(job_id: str) -> dict:
    state = jobs.get(job_id)
    if state is None:
        raise _bad_request(f"Unknown job '{job_id}'. Start a run with POST /api/fit first.")
    return state.to_dict()


@app.get("/api/samples")
def api_samples(dataset: str) -> list[dict]:
    _check_dataset(dataset)
    return _remember_samples(dataset)


@app.get("/api/samples/{sample_id}")
def api_sample(sample_id: str) -> FileResponse:
    path = _sample_path(sample_id)
    if path is None or not path.is_file():
        raise HTTPException(status_code=404, detail=f"No sample '{sample_id}'. List samples with GET /api/samples.")
    return FileResponse(path)


@app.post("/api/chat")
def api_chat(
    model: str = Form(...),
    message: str = Form(...),
    files: list[UploadFile] = File(default=[]),
    sample_ids: list[str] = Form(default=[]),
) -> dict:
    _check_model(model)
    # (display name, path on disk) for every file the user attached
    attached: list[tuple[str, Path]] = []
    temp_paths: list[Path] = []
    try:
        for upload in files:
            if not upload.filename:
                continue
            fd, tmp = tempfile.mkstemp(suffix=Path(upload.filename).suffix.lower())
            with os.fdopen(fd, "wb") as handle:
                handle.write(upload.file.read())
            temp_paths.append(Path(tmp))
            attached.append((upload.filename, Path(tmp)))
        for sid in sample_ids:
            path = _sample_path(sid)
            if path is None or not path.is_file():
                raise _bad_request(f"Unknown sample '{sid}'. List samples with GET /api/samples.")
            attached.append((path.name, path))

        try:
            base = models.load_model(model)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Could not load model '{model}': {exc}")
        bridges = {b.modality: b for b in registry.get_bridges(model)}

        inputs: list[SenseInput] = []
        used: list[dict] = []
        for name, path in attached:
            modality = _modality_of(name)
            bridge = bridges.get(modality) if modality else None
            if bridge is None:
                if modality is None:
                    why = f"only audio files ({' '.join(sorted(AUDIO_EXTENSIONS))}) can be given to it."
                else:
                    why = f"{base.info.label} has no installed {modality} sense. Run the agent for this model first."
                used.append({"bridge": "", "kind": "none", "source": name,
                             "detail": f"'{name}' was not shown to the model: {why}"})
                continue
            try:
                item = bridge.prepare(path)
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"Could not read '{name}' with bridge '{bridge.name}': {exc}")
            item.source = name  # uploads sit under temporary names; show the name the user gave
            inputs.append(item)
            used.append({"bridge": bridge.name, "kind": bridge.kind, "source": item.source, "detail": item.detail})

        try:
            with _chat_lock:
                reply = base.chat(message, inputs, max_new_tokens=CHAT_TOKENS)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"The model failed to answer: {exc}")
        return {"reply": reply, "used": used}
    finally:
        for tmp in temp_paths:
            tmp.unlink(missing_ok=True)


# ----------------------------------------------------------------------------- interface
@app.get("/", response_model=None)
def index() -> HTMLResponse | FileResponse:
    page = WEB / "index.html"
    if not page.is_file():
        return HTMLResponse("<p>The interface is missing: web/index.html has not been written yet.</p>")
    return FileResponse(page)


app.mount("/static", StaticFiles(directory=WEB, check_dir=False), name="static")
