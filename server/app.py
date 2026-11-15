"""HTTP API and static interface. Run: uvicorn server.app:app --port 8000

All JSON. Errors: {"detail": "<what went wrong and what to do>"} with a 4xx or 5xx status.

  GET  /api/models
       -> [ModelInfo dict]                          name, label, native, open_weights, available, note, tools
                                                    (tools: the model can call tools, so it can take a tool bridge)
  GET  /api/datasets
       -> [{"name", "title", "modality", "labels", "n", "origin", "meta"}]
          origin "builtin" (gtzan, esc50; meta {}, except that real gtzan notes its duplicate recordings) or
          "folder" (built with POST /api/datasets; meta says how: source, licence, folder, counts of files used,
          duplicates, conflicts, skipped files and near duplicates, embedding time; see sense.datasets.from_folder)
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
  POST /api/auto                                    multipart form: model, goal (what the model should do with the
                                                    file), file (one upload), examples (optional: a folder on the
                                                    server with one sub-folder of files per label)
       -> {"job_id": str}                           an agent run from one file and a goal (agent.jobs.start_auto): it
                                                    detects the file's modality, researches a specialist and
                                                    labelled data (or uses examples), then fits, verifies and
                                                    installs or refuses. Poll GET /api/jobs/<job_id>; events come in
                                                    the order detect, research, data, inspect, plan, fit, verify,
                                                    claims, decide, install, done, and a native file or one that
                                                    needs the user ends earlier. job.request.dataset is "" until the
                                                    data step picks a set. The upload is kept under
                                                    <SENSE_HOME>/uploads, because the run reads it after this
                                                    returns. 400 for an unknown model, an empty goal, no file or a
                                                    missing examples folder; 409 while an agent run or a replay is
                                                    in progress. A finished run is recorded for replay as with fit
  GET  /api/jobs/<job_id>
       -> JobState dict                             id, request, status, events[], cards[], installed, error
                                                    a replayed job (id "replay-...") also has
                                                    "replay": {"of": run id, "recorded", "speed"}
  DELETE /api/jobs/<job_id>
       -> {"cancelled": bool}                       stops a replay or a live agent run: true when it was running
                                                    and is now asked to stop, false when it had already finished.
                                                    A cancelled replay stops at once (unless it is reinstalling a
                                                    sense: then once that ends; DELETE waits up to 5 s for it, so the
                                                    replay may still be running when DELETE answers true), and ends
                                                    "failed" with error "Cancelled." and a last [error] event (data
                                                    "cancelled": true); a sense it already reinstalled stays installed.
                                                    A live agent run stops at its next trace event (agent.jobs.cancel)
                                                    and is still recorded under <SENSE_HOME>/runs when it ends.
                                                    400 for an unknown job; 409 for a running agent run while
                                                    agent.jobs cannot cancel one yet
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
                                                    and, optionally, history
       -> {"reply": str, "used": [{"bridge", "kind", "source", "detail"}]}
          `used` lists how each file reached the model; a file whose modality has no installed sense, or that
          cannot be read, is reported with kind "none" and is not shown to the model. For kind "tool" (the model
          could call the listen tool, which asks CLAP about the file) the detail says whether the model called it
          and what the last call returned: "the model called listen 1 time: Similarity ... Best match: rock." or
          "the model could call listen but did not".
          history: a JSON string of the earlier turns, oldest first, [{"role": "user"|"assistant", "content": str}],
          text only (the files of earlier turns are not sent again). Only the last MAX_HISTORY_TURNS (8) are kept,
          each cut to HISTORY_CHARS (4000) characters. Missing, empty or [] starts a new conversation; anything
          else that is not such a list is a 400 that says what form it takes.
          400 for an empty message; 409 while an agent run holds a different local model.
  POST /api/chat/stream                             the same form as POST /api/chat
       -> text/event-stream: Server-Sent Events ("event: <name>\\ndata: <JSON>\\n\\n"), in this order
            used    the `used` list, as POST /api/chat gives it but sent before the reply, so a tool input's
                    detail still only says that the model can call the tool
            token   {"text": str}, one for each piece of the reply, in order; the pieces join to the reply
            done    {"reply": str, "used": [...]}: the whole reply, and `used` with each tool input's detail
                    saying what the model did with the tool; reply and used are what POST /api/chat would return
          or, if the model fails once the stream has begun, instead of done
            error   {"detail": "The model failed to answer: ..."}, and the stream ends
          The reply streams when the model has chat_stream (contracts.BaseModel); otherwise chat()'s reply comes
          as one token event. Everything found wrong before the stream begins is an ordinary JSON error, as for
          POST /api/chat: 400 for an empty message, a bad history or an unknown model or sample, 409 while an
          agent run holds a different local model, 500 if the model or the installed sense cannot be loaded.
          One model call runs at a time, here as in POST /api/chat: the stream holds the chat lock until the reply
          ends. Uploads are deleted when the stream ends, also when the client goes away first.
  GET  /                                            web/index.html; other files under web/ are served at /static/

The server holds no model itself: it calls sense.models, sense.registry, sense.datasets and agent.jobs.
Finished agent runs are recorded with sense.runs and played back by server.replay.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import threading
import time
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anyio
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.responses import StreamingResponse
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
MAX_HISTORY_TURNS = 8  # earlier chat turns passed to the model: the last 8 messages, user and assistant together
HISTORY_CHARS = 4000  # each earlier turn is cut to this many characters
HISTORY_FORM = (
    'a JSON list of the earlier turns, oldest first, such as [{"role": "user", "content": "What is this?"}, '
    '{"role": "assistant", "content": "A dog barking."}]'
)

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


UPLOADS = "uploads"  # under SENSE_HOME: files handed to POST /api/auto, kept for the run that reads them


def _keep_upload(upload: UploadFile) -> Path:
    """Save an upload under <SENSE_HOME>/uploads/<12 hex>/<its file name, made safe> and return the path. Unlike
    chat uploads it is kept: the agent run reads it from its own thread after the request has returned."""
    name = "".join(c if c.isalnum() or c in "._- " else "_" for c in Path(upload.filename or "").name).strip(" .")
    folder = config.home() / UPLOADS / os.urandom(6).hex()
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (name or "upload")
    with open(path, "wb") as handle:
        handle.write(upload.file.read())
    return path


@app.post("/api/auto")
def api_auto(
    model: str = Form(...),
    goal: str = Form(""),
    file: UploadFile | None = File(default=None),
    examples: str = Form(""),
) -> dict:
    _check_model(model)
    if not goal.strip():
        raise _bad_request("The goal is empty. Say what the model should do with the file, such as 'recognise music genres'.")
    if file is None or not file.filename:
        raise _bad_request("No file was attached. Attach the file the model should learn to read.")
    folder = None
    if examples.strip():
        folder = Path(examples.strip()).expanduser()
        if not folder.is_dir():
            raise _bad_request(
                f"The examples folder '{examples.strip()}' does not exist on the server. Give a folder with one "
                "sub-folder of files per label, or leave examples out."
            )
    path = _keep_upload(file)
    with _start_lock:
        try:
            if replay.running():
                raise HTTPException(status_code=409, detail="A replay is in progress. Wait for it to finish, then start again.")
            _await_recording()
            try:
                job_id = jobs.start_auto(model, path, goal.strip(), examples=folder)
            except RuntimeError as exc:
                text = str(exc)
                raise HTTPException(status_code=409, detail=f"{text[:1].upper()}{text[1:]}. Wait for it to finish, then start again.")
        except HTTPException:
            _delete([path])
            path.parent.rmdir()
            raise
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


@app.delete("/api/jobs/{job_id}")
def api_cancel_job(job_id: str) -> dict:
    if replay.get(job_id) is not None:
        return {"cancelled": replay.cancel(job_id)}
    state = jobs.get(job_id)
    if state is None:
        raise _bad_request(
            f"Unknown job '{job_id}': there is nothing to cancel. Use the job id that POST /api/fit or "
            "POST /api/replay returned."
        )
    if state.status != "running":
        return {"cancelled": False}
    cancel = getattr(jobs, "cancel", None)  # agent.jobs.cancel(job_id) -> bool, once agent.jobs has it
    if not callable(cancel):
        raise HTTPException(
            status_code=409,
            detail="Cancelling a running agent run is not supported yet: wait for it to finish.",
        )
    return {"cancelled": bool(cancel(job_id))}


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


@dataclass
class _ChatTurn:
    """A chat request, checked, with its files bridged: everything the model needs to answer it."""

    base: Any  # the loaded model
    message: str
    inputs: list[SenseInput]
    used: list[dict]  # how each file reached the model, as the response reports it
    tool_used: list[tuple[dict, SenseInput]]  # entries of `used` whose detail is only known after the reply
    history: list[dict]  # the earlier turns, checked and bounded

    def options(self) -> dict:
        """Keyword arguments for chat: history only when there is some, so a model without it still works."""
        return {"history": self.history} if self.history else {}

    def finish_tool_details(self) -> None:
        """After the reply: say in `used` what the model did with each tool input."""
        for entry, item in self.tool_used:
            entry["detail"] = _tool_detail(item)


def _parse_history(raw: str) -> list[dict]:
    """The form field `history` as earlier turns: checked, the last MAX_HISTORY_TURNS kept, each cut to
    HISTORY_CHARS characters. Empty means none. Raises a 400 that says what form the field takes."""
    if not raw.strip():
        return []
    try:
        turns = json.loads(raw)
    except (ValueError, RecursionError):  # RecursionError: nested deeper than json can read
        raise _bad_request(f"The history is not valid JSON. Send it as {HISTORY_FORM}.")
    if not isinstance(turns, list):
        raise _bad_request(f"The history must be a list, not {type(turns).__name__}. Send it as {HISTORY_FORM}.")
    try:
        turns = models.chat_history(turns)
    except ValueError as exc:
        raise _bad_request(f"{exc} Send the history as {HISTORY_FORM}.")
    return [{"role": t["role"], "content": t["content"][:HISTORY_CHARS]} for t in turns[-MAX_HISTORY_TURNS:]]


def _prepare_chat(
    model: str, message: str, history: str, files: list[UploadFile], sample_ids: list[str], temp_paths: list[Path],
) -> _ChatTurn:
    """What POST /api/chat and POST /api/chat/stream do before the model answers: check the request, save the
    uploads to temporary files (appended to `temp_paths`, which the caller deletes), load the model and bridge each
    file. Raises HTTPException."""
    _check_model(model)
    if not message.strip():
        raise _bad_request("The message is empty. Type a question for the model, then send it.")
    earlier = _parse_history(history)
    _check_no_competing_run(model)
    # (display name, path on disk) for every file the user attached
    attached: list[tuple[str, Path]] = []
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
    tool_used: list[tuple[dict, SenseInput]] = []  # entries whose detail is only known after the reply
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
        if item.kind == "tool" and item.tool is not None:
            tool_used.append((used[-1], item))
    return _ChatTurn(base=base, message=message, inputs=inputs, used=used, tool_used=tool_used, history=earlier)


def _delete(paths: list[Path]) -> None:
    for path in paths:
        path.unlink(missing_ok=True)


@app.post("/api/chat")
def api_chat(
    model: str = Form(...),
    message: str = Form(""),
    files: list[UploadFile] = File(default=[]),
    sample_ids: list[str] = Form(default=[]),
    history: str = Form(""),
) -> dict:
    temp_paths: list[Path] = []
    try:
        turn = _prepare_chat(model, message, history, files, sample_ids, temp_paths)
        try:
            with _chat_lock:
                reply = turn.base.chat(message, turn.inputs, max_new_tokens=CHAT_TOKENS, **turn.options())
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"The model failed to answer: {exc}")
        turn.finish_tool_details()
        return {"reply": reply, "used": turn.used}
    finally:
        _delete(temp_paths)


@app.post("/api/chat/stream")
def api_chat_stream(
    model: str = Form(...),
    message: str = Form(""),
    files: list[UploadFile] = File(default=[]),
    sample_ids: list[str] = Form(default=[]),
    history: str = Form(""),
) -> StreamingResponse:
    temp_paths: list[Path] = []
    try:
        turn = _prepare_chat(model, message, history, files, sample_ids, temp_paths)
    except BaseException:
        _delete(temp_paths)
        raise
    events = _chat_events(turn, temp_paths)
    first = next(events)  # starts the generator, so that its finally deletes the uploads however the response ends
    return StreamingResponse(
        _stream_events(first, events),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},  # no buffering by a proxy in between
    )


def _event(name: str, data: Any) -> str:
    """One Server-Sent Event; json.dumps keeps the data on one line."""
    return f"event: {name}\ndata: {json.dumps(data)}\n\n"


def _reply_pieces(turn: _ChatTurn) -> Iterator[str]:
    """The model's reply in pieces: its chat_stream if it has one, else chat()'s reply as one piece."""
    stream = getattr(turn.base, "chat_stream", None)
    if callable(stream):
        return iter(stream(turn.message, turn.inputs, max_new_tokens=CHAT_TOKENS, **turn.options()))
    return iter([turn.base.chat(turn.message, turn.inputs, max_new_tokens=CHAT_TOKENS, **turn.options())])


def _chat_events(turn: _ChatTurn, temp_paths: list[Path]) -> Iterator[str]:
    """The events of POST /api/chat/stream: used, a token per piece, then done (or error). The chat lock is held
    while the model answers; the uploads are deleted when this ends or is closed."""
    try:
        yield _event("used", turn.used)
        pieces: list[str] = []
        try:
            with _chat_lock:
                stream = _reply_pieces(turn)
                try:
                    for piece in stream:
                        if piece:
                            pieces.append(piece)
                            yield _event("token", {"text": piece})
                finally:
                    close = getattr(stream, "close", None)  # lets the model free its worker or HTTP response
                    if callable(close):
                        close()
        except Exception as exc:
            yield _event("error", {"detail": f"The model failed to answer: {exc}"})
            return
        turn.finish_tool_details()
        yield _event("done", {"reply": "".join(pieces), "used": turn.used})
    finally:
        _delete(temp_paths)


async def _stream_events(first: str, events: Iterator[str]) -> AsyncIterator[str]:
    """`first`, then the rest of `events`, each read in a worker thread (a model call blocks). `events` is closed
    however the response ends, also when the client goes away, so that it frees the chat lock and the uploads.

    The worker threads are counted by a limiter of this response's own, not by the shared pool's: every request
    waiting for the chat lock holds a thread of the shared pool, and the stream that holds the lock must never wait
    for one of those to go on or to close, or nothing would free the lock. One thread at a time, so that close never
    runs while a read is still going."""
    limiter = anyio.CapacityLimiter(1)
    try:
        yield first
        while (chunk := await anyio.to_thread.run_sync(next, events, None, limiter=limiter)) is not None:
            yield chunk
    finally:
        with anyio.CancelScope(shield=True):
            await anyio.to_thread.run_sync(events.close, limiter=limiter)


def _tool_detail(item: SenseInput) -> str:
    """What the model did with a tool input during its reply, for the interface."""
    tool = item.tool
    if not tool.calls:
        return f"the model could call {tool.name} but did not"
    n = len(tool.calls)
    return f"the model called {tool.name} {n} time{'' if n == 1 else 's'}: {tool.calls[-1]['result']}"


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
