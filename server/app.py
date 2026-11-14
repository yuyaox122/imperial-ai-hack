"""HTTP API and static interface. Run: uvicorn server.app:app --port 8000

All JSON. Errors: {"detail": "<what went wrong and what to do>"} with a 4xx or 5xx status.

  GET  /api/models
       -> [ModelInfo dict]                          name, label, native, open_weights, available, note
  GET  /api/datasets
       -> [{"name", "title", "modality", "labels", "n", "origin", "meta"}]
          origin "builtin" (gtzan, esc50; meta {}) or "folder" (built with POST /api/datasets; meta says how:
          source, licence, folder, counts of files used, duplicates, conflicts, skipped files and near duplicates,
          embedding time; see sense.datasets.from_folder)
  POST /api/datasets                                body: {"path": str, "name": str|null, "question": str|null,
                                                          "title": str|null, "max_per_label": int|null}
       -> the new set's GET /api/datasets item      builds a labelled set from a folder on the server's disk with one
                                                    sub-folder of audio files per label, or reuses the saved one if
                                                    the files are unchanged; 400 with the reason if the folder
                                                    cannot make a usable set
  GET  /api/senses?model=<name>
       -> [{"bridge", "kind", "modality", "dataset", "card": ReportCard dict}]
  DELETE /api/senses?model=<name>&modality=<modality>
       -> {"removed": bool}
  POST /api/fit                                     body: {"model", "dataset", "modality": "audio", "scope": "closed"|"open"}
       -> {"job_id": str}                           409 if a run is already in progress; 400 if the run could not
                                                    get as far as verifying (sense.datasets.fit_problem): scope
                                                    "open" on a set with fewer than 3 labels or too few examples
                                                    to fit on, or a folder set embedded in the other mode
                                                    409 while a replay is in progress; a finished run is recorded
                                                    under <SENSE_HOME>/runs for replay
                                                    a fit or replay started just after a run ends first waits
                                                    until that run is recorded (409 if that takes over 10 s)
  GET  /api/jobs/<job_id>
       -> JobState dict                             id, request, status, events[], cards[], installed, error
                                                    a replayed job (id "replay-...") also has
                                                    "replay": {"of": run id, "recorded", "speed"}
  GET  /api/runs
       -> [{"id", "request", "status", "installed", "recorded", "origin", "seconds", "has_sense"}]
                                                    recorded runs, newest first; origin "server" or "cli";
                                                    has_sense: the record holds a copy of the sense it installed
  POST /api/replay                                  body: {"run": run id, "speed": 1.0}
       -> {"job_id": str}                           plays the recorded run back as a job (poll GET /api/jobs/<id>):
                                                    each event when event.t / speed seconds have passed, and the
                                                    recorded sense reinstalled at its install step. speed 0.1 to 1000.
                                                    400 for an unknown run or a bad speed; 409 while an agent run
                                                    or another replay is in progress
  GET  /api/samples?dataset=<name>
       -> [{"id", "label", "name"}]                 held-out example files the interface can offer
  GET  /api/samples/<id>
       -> the audio file
  POST /api/chat                                    multipart form: model, message, files (0..n uploads), sample_ids (0..n)
       -> {"reply": str, "used": [{"bridge", "kind", "source", "detail"}]}
          `used` lists how each file reached the model; a file whose modality has no installed sense, or that
          cannot be read, is reported with kind "none" and is not shown to the model.
          400 for an empty message; 409 while an agent run holds a different local model.
  GET  /                                            web/index.html; other files under web/ are served at /static/

The server holds no model itself: it calls sense.models, sense.registry, sense.datasets and agent.jobs.
Finished agent runs are recorded with sense.runs and played back by server.replay.
"""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel as RequestBody

from agent import jobs
from sense import datasets, models, registry
from sense.config import REPO
from sense.contracts import FitRequest, SenseInput
from sense import config, runs
from sense.specialists import AUDIO_EXTENSIONS  # what soundfile reads; not .m4a
from server import replay

WEB = REPO / "web"
SCOPES = ("closed", "open")
CHAT_TOKENS = 256  # the default 96 cut Qwen3.5-0.8B's replies off mid-sentence

app = FastAPI(title="Sense harness")
_sample_paths: dict[str, Path] = {}  # opaque id -> path, filled only from datasets.sample_files
_chat_lock = threading.Lock()  # one model call at a time: the model objects are not thread safe
_start_lock = threading.Lock()  # an agent run and a replay never start together
WATCH_SECONDS = 0.5  # how often a finished agent run is looked for, to record it
RECORD_WAIT_SECONDS = 10.0  # how long a fit or replay waits for the last finished agent run to be recorded
_recorder: tuple[str, threading.Thread] | None = None  # (job id, watcher) of the last agent run started here
log = logging.getLogger(__name__)


class FitBody(RequestBody):
    model: str
    dataset: str
    modality: str = "audio"
    scope: str = "closed"


class ReplayBody(RequestBody):
    run: str
    speed: Any = 1.0  # checked by server.replay, so that any bad speed is a 400


class FolderBody(RequestBody):
    path: str
    name: str | None = None
    question: str | None = None
    title: str | None = None
    max_per_label: int | None = None


# ----------------------------------------------------------------------------- helpers
def _bad_request(detail: str) -> HTTPException:
    return HTTPException(status_code=400, detail=detail)


def _check_model(name: str) -> None:
    known = [m.name for m in models.available_models()]
    if name not in known:
        raise _bad_request(f"Unknown model '{name}'. Choose one of: {', '.join(known) or 'none available'}.")


def _check_dataset(name: str) -> dict:
    known = {d["name"]: d for d in datasets.available_sets()}
    if name not in known:
        raise _bad_request(f"Unknown dataset '{name}'. Choose one of: {', '.join(known) or 'none available'}.")
    return known[name]


def _check_no_competing_run(model: str) -> None:
    """Refuse to load a second local model next to the one a running agent run still holds (24 GB laptop)."""
    job = jobs.latest()
    if job is None or job.status != "running" or job.request.model == model:
        return
    open_weights = {m.name for m in models.available_models() if m.open_weights}
    if model in open_weights and job.request.model in open_weights:
        raise HTTPException(
            status_code=409,
            detail=f"An agent run on {job.request.model} is in progress. Chat with that model or wait for the run to finish.",
        )


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


@app.post("/api/datasets")
def api_add_dataset(body: FolderBody) -> dict:
    options = {} if body.max_per_label is None else {"max_per_label": body.max_per_label}
    try:
        ds = datasets.from_folder(body.path, body.name, title=body.title, question=body.question, **options)
    except (ValueError, FileNotFoundError) as exc:
        raise _bad_request(str(exc))
    except RuntimeError as exc:  # e.g. the specialist could not be loaded
        raise HTTPException(status_code=500, detail=str(exc))
    entry = next((d for d in datasets.available_sets() if d["name"] == ds.name), None)
    if entry is None:
        raise HTTPException(
            status_code=500,
            detail=f"The set {ds.name} was built but cannot be listed; see the server log, then build it again.",
        )
    return entry


@app.get("/api/senses")
def api_senses(model: str) -> list[dict]:
    _check_model(model)
    return registry.installed(model)


@app.delete("/api/senses")
def api_remove_sense(model: str, modality: str) -> dict:
    _check_model(model)
    try:
        return {"removed": registry.uninstall(model, modality)}
    except ValueError as exc:
        raise _bad_request(f"{exc} Use a modality such as 'audio'.")


@app.post("/api/fit")
def api_fit(body: FitBody) -> dict:
    _check_model(body.model)
    dataset = _check_dataset(body.dataset)
    if body.scope not in SCOPES:
        raise _bad_request(f"Unknown scope '{body.scope}'. Use 'closed' or 'open'.")
    if body.scope == "open" or dataset.get("origin") == "folder":  # a built-in set is always big enough for closed
        try:
            problem = datasets.fit_problem(datasets.load_labelled_set(body.dataset), body.scope)
        except (ValueError, FileNotFoundError) as exc:
            raise _bad_request(str(exc))
        if problem:
            raise _bad_request(problem)
    if body.modality != dataset["modality"]:
        raise _bad_request(
            f"The set {body.dataset} holds {dataset['modality']} examples, but the request is for "
            f"{body.modality}. Pick a {body.modality} set, or ask for {dataset['modality']}."
        )
    request = FitRequest(model=body.model, dataset=body.dataset, modality=body.modality, scope=body.scope)
    with _start_lock:
        if replay.running():
            raise HTTPException(status_code=409, detail="A replay is in progress. Wait for it to finish, then start again.")
        _await_recording()
        try:
            job_id = jobs.start(request)
        except RuntimeError as exc:
            text = str(exc)
            raise HTTPException(status_code=409, detail=f"{text[:1].upper()}{text[1:]}. Wait for it to finish, then start again.")
        _record_when_finished(job_id)
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def api_job(job_id: str) -> dict:
    replayed = replay.get(job_id)
    if replayed is not None:
        state, info = replayed
        return state.to_dict() | {"replay": info}
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
    message: str = Form(""),
    files: list[UploadFile] = File(default=[]),
    sample_ids: list[str] = Form(default=[]),
) -> dict:
    _check_model(model)
    if not message.strip():
        raise _bad_request("The message is empty. Type a question for the model, then send it.")
    _check_no_competing_run(model)
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
        bridges = {}
        if any(_modality_of(name) for name, _ in attached):  # plain text chat must not depend on the senses
            try:
                bridges = {b.modality: b for b in registry.get_bridges(model)}
            except RuntimeError as exc:
                raise HTTPException(status_code=500, detail=str(exc))

        inputs: list[SenseInput] = []
        used: list[dict] = []
        for name, path in attached:
            modality = _modality_of(name)
            bridge = bridges.get(modality) if modality else None
            if bridge is None:
                if modality is None:
                    why = f"only audio files in these formats can be given to it: {' '.join(sorted(AUDIO_EXTENSIONS))}."
                else:
                    why = f"{base.info.label} has no installed {modality} sense. Run the agent for this model first."
                used.append({"bridge": "", "kind": "none", "source": name,
                             "detail": f"'{name}' was not shown to the model: {why}"})
                continue
            try:
                item = bridge.prepare(path)
            except (ValueError, FileNotFoundError) as exc:  # this file is unreadable; answer with the others
                why = str(exc).replace(str(path), name).replace(path.name, name)
                used.append({"bridge": "", "kind": "none", "source": name,
                             "detail": f"'{name}' was not shown to the model: {why}"})
                continue
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


# ----------------------------------------------------------------------------- replay
def _record_when_finished(job_id: str) -> None:
    """Record the agent run with sense.runs once it has finished, from a daemon thread. Never raises.
    Call with _start_lock held, so that a fit or replay started next finds this watcher in _recorder."""
    global _recorder
    home = config.home()

    def watch() -> None:
        try:
            while True:
                state = jobs.get(job_id)
                if state is None:  # the job is gone (jobs were reset): nothing to record
                    return
                if state.status != "running":
                    break
                time.sleep(WATCH_SECONDS)
            if config.home() != home:
                log.warning("Agent run %s was not recorded: SENSE_HOME changed while it ran.", job_id)
                return
            runs.save_run(state, origin="server")
        except Exception:
            log.exception("Agent run %s could not be recorded for replay.", job_id)

    watcher = threading.Thread(target=watch, name=f"record-run-{job_id}", daemon=True)
    watcher.start()
    _recorder = (job_id, watcher)


def _await_recording() -> None:
    """With _start_lock held: wait until the last agent run, if it has finished, has been recorded. A fit or a
    replay started in the gap could otherwise change the registry first, and the run would be recorded without
    the sense it installed. Raises a 409 if the recording takes longer than RECORD_WAIT_SECONDS."""
    if _recorder is None:
        return
    job_id, recorder = _recorder
    state = jobs.get(job_id)
    if not recorder.is_alive() or state is None or state.status == "running":
        return  # recorded already, nothing left to record, or still running (the caller then refuses to start)
    recorder.join(RECORD_WAIT_SECONDS)
    if recorder.is_alive():
        raise HTTPException(status_code=409, detail="The last agent run is still being recorded. Try again in a moment.")


@app.get("/api/runs")
def api_runs() -> list[dict]:
    return runs.list_runs()


@app.post("/api/replay")
def api_replay(body: ReplayBody) -> dict:
    with _start_lock:
        live = jobs.latest()
        if live is not None and live.status == "running":
            raise HTTPException(
                status_code=409,
                detail=f"An agent run on {live.request.model} is in progress. Wait for it to finish, then start the replay.",
            )
        _await_recording()
        try:
            return {"job_id": replay.start(body.run, body.speed)}
        except ValueError as exc:
            raise _bad_request(str(exc))
        except RuntimeError as exc:
            text = str(exc)
            raise HTTPException(status_code=409, detail=f"{text[:1].upper()}{text[1:]}. Wait for it to finish, then start again.")


# ----------------------------------------------------------------------------- interface
@app.get("/", response_model=None)
def index() -> HTMLResponse | FileResponse:
    page = WEB / "index.html"
    if not page.is_file():
        return HTMLResponse("<p>The interface is missing: web/index.html has not been written yet.</p>")
    return FileResponse(page)


app.mount("/static", StaticFiles(directory=WEB, check_dir=False), name="static")
