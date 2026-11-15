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
       -> {"job_id": str}                           an agent run from one file of any type and a goal
                                                    (agent.jobs.start_auto, agent.auto.run_auto): it detects the
                                                    file's modality, then takes one of three flows, which the detect
                                                    event's data names as "flow". native: the model takes the file
                                                    as it is in chat (text; an image for an API model), events detect,
                                                    done. specialist (audio): researches a specialist and labelled
                                                    data (or uses examples), then fits, verifies and installs or
                                                    refuses, events in the order detect, research, data, inspect,
                                                    plan, fit, verify, claims, decide, install, done. adapter
                                                    (everything else: video, 3d, molecule, geo, csv, fasta, an
                                                    unknown format, an image for a local model): the agent designs
                                                    an adapter, writes and runs its code in a sandbox, finds labelled
                                                    probe files (the examples folder, else a public dataset, else
                                                    generated ones when the label is built into the content), verifies
                                                    and installs or refuses (agent.synth.run_synth), events in the
                                                    order detect, design, install_request and approval (only when the
                                                    design asks to install packages: see POST
                                                    /api/jobs/<job_id>/approval), build, data, verify, decide,
                                                    install, done. A run that needs the user (no labelled examples
                                                    found) ends earlier with done data needs_user true. Poll GET
                                                    /api/jobs/<job_id>. job.request.dataset is "" (and modality "")
                                                    until the data step picks a set. The upload is kept under
                                                    <SENSE_HOME>/uploads/<12 hex>/<its name, made safe and cut to 120
                                                    bytes>, because the run reads it after this returns, and removed
                                                    once the run has ended (a folder left by a server that stopped
                                                    mid-run is removed a day later). 400 for an unknown model, an
                                                    empty goal, no file or a missing examples folder; 413 for a file
                                                    over MAX_UPLOAD_BYTES (200 MB); 409 while an agent run or a
                                                    replay is in progress. A finished run is recorded for replay as
                                                    with fit
  GET  /api/jobs/latest
       -> {"job_id": str | null}                    the agent run in progress (agent.jobs.latest(), while it is
                                                    running), else null; replays are not listed. A tab that did not
                                                    start the run uses it to follow the run, answer its approval
                                                    request or stop it, rather than meet only the 409 of a new run
  GET  /api/jobs/<job_id>
       -> JobState dict                             id, request, status, events[], cards[], installed, error, and
                                                    pending_approval: the request the run is waiting on, {"id",
                                                    "kind" ("install"), "payload", "timeout", "asked"} (agent.jobs.
                                                    pending), or null. An install request's payload holds "plan"
                                                    (agent.installer.InstallPlan.to_dict()), "summary" (one sentence)
                                                    and "review" (an LLM's opinion, or null)
                                                    a replayed job (id "replay-...") also has
                                                    "replay": {"of": run id, "recorded", "speed"}; its
                                                    pending_approval is always null
  POST /api/jobs/<job_id>/approval                  body: {"id": str, "approve": bool}
       -> {"answered": true, "approve": bool}       answers the run's open request (agent.jobs.answer): the run
                                                    goes on at once, with the packages installed when approved and
                                                    without them when declined. A request not answered within its
                                                    timeout (600 s) counts as declined. 404 for an unknown job; 400
                                                    for a body that is not that JSON object (approve must be true or
                                                    false itself); 409 when that request is not open (answered
                                                    already, timed out, cancelled, a wrong id, or a replay)
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
                                                    400 for an unknown job
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
          `used` lists how each file reached the model, uploads first, then samples. A file's modality is decided
          from its bytes, as POST /api/auto decides it (agent.detect: magic bytes, then the content), and a listed
          audio extension (sense.specialists.AUDIO_EXTENSIONS) also means audio. An upload is saved under a
          temporary name that keeps its extension when that is a dot and 1 to 10 letters or digits (adapters see
          the right suffix). Kind "native" (bridge ""): the model takes the file as it is: a text file's text, at
          most TEXT_FILE_CHARS (8000) characters, is sent after the message; an image (PNG or JPEG, at most 20 MB)
          is shown to a model that takes images (an API model whose ModelInfo.native lists "image"; the local
          models take no images yet: sense.models.takes_as_is; a GIF or any other image format goes to a sense).
          Otherwise the installed sense for the file's detected modality carries it (video, 3d, molecule, geo,
          image, audio, other), and `kind` is its bridge kind. An agent-written adapter takes only files
          agent.detect describes as it described the files the adapter was verified on (its manifest's "detected":
          the same modality, a format of the user's file or of its probe files, and for an unknown format the same
          extension): an adapter built on OBJ files is not given a PLY file, and a file agent.detect calls "other"
          (CSV, FASTA, an unknown binary) goes only to an adapter built from such a file. An agent-written adapter
          (bridge "adapter:<name>", kind "text" or "image") runs its code on the file in its sandbox; the detail
          says what it produced (the start of its text, cut, or how many images); an image-kind adapter needs a
          model that takes images, else the file is not shown and the detail says why, as when the adapter fails
          on the file. A text file agent.detect reads as a mesh, a molecule or places (OBJ, SMILES,
          coordinates...) that no installed sense takes, or whose adapter fails on it, is given as text, the
          detail saying why its sense did not carry it. A file the model cannot take, whose
          modality has no installed sense, that cannot be read or that is empty, is reported with kind "none" and
          is not shown to the model; the detail says why. For kind "tool" (the model
          could call the listen tool, which asks CLAP about the file) the detail says whether the model called it
          and what the last call returned: "the model called listen 1 time: Similarity ... Best match: rock." or
          "the model could call listen but did not".
          history: a JSON string of the earlier turns, oldest first, [{"role": "user"|"assistant", "content": str}],
          text only (the files of earlier turns are not sent again). Only the last MAX_HISTORY_TURNS (8) are kept,
          each cut to HISTORY_CHARS (4000) characters. Missing, empty or [] starts a new conversation; anything
          else that is not such a list is a 400 that says what form it takes.
          400 for an empty message; 413 for a file over MAX_UPLOAD_BYTES (200 MB); 409 while an agent run holds a
          different local model; 500 if an installed sense the files need cannot be loaded (the installed senses
          are loaded only when a file may need one).
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
          POST /api/chat: 400 for an empty message, a bad history or an unknown model or sample, 413 for a file
          over 200 MB, 409 while an agent run holds a different local model, 500 if the model or the installed
          sense cannot be loaded.
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
import re
import shutil
import tempfile
import threading
import time
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anyio
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel as RequestBody

from agent import jobs
from agent.detect import MODALITIES as DETECT_MODALITIES, detect, is_text, words
from sense import datasets, models, registry
from sense.adapters import AdapterError
from sense.config import REPO
from sense.contracts import FitRequest, ModelInfo, SenseInput
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
MAX_UPLOAD_BYTES = 200 * 2**20  # the most one uploaded file may hold; a larger one is a 413
UPLOAD_CHUNK = 2**20  # uploads are copied to disk this many bytes at a time, never read into memory whole
NAME_BYTES = 120  # a kept upload's file name is cut to this many bytes, its extension kept (file systems allow 255)
UPLOAD_KEEP_SECONDS = 24 * 3600  # an upload folder this old was left by a server that stopped while a run read it
TEXT_FILE_CHARS = 8000  # a text file given to the model as it is: at most this many characters of it
IMAGE_BYTES = 20 * 2**20  # the most an image shown to the model as it is may hold (what the OpenAI API takes)
_SAFE_SUFFIX = re.compile(r"\.[a-z0-9]{1,10}")  # a chat upload's temporary file keeps an extension like this

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


def _kind_of(name: str, path: Path) -> tuple[str | None, str]:
    """The modality and agent.detect format of an attached file, decided from its bytes as POST /api/auto decides
    it (agent.detect), so that chat gives the model every file the agent taught it to read, whatever the file is
    called. Magic bytes (WAV, FLAC, Ogg, MP3, PNG, JPEG, GIF, MP4 and other video, GLB) decide first; then a listed
    audio extension means audio (format: the extension without its dot), since soundfile also reads AIFF and AU,
    and the bridge reports a file it cannot read; then agent.detect's modality from the content (text, 3d,
    molecule, geo or other). (None, "") for an empty file."""
    try:
        found = detect(path)
    except ValueError:  # an empty file
        found = None
    if found is not None and found.how == "magic bytes":
        return found.modality, found.format
    suffix = Path(name).suffix.lower()
    if suffix in AUDIO_EXTENSIONS:
        return "audio", suffix[1:]
    return (found.modality, found.format) if found is not None else (None, "")


def _takes_images(info: ModelInfo) -> bool:
    """Whether chat can show this model an image as it is (sense.models.takes_as_is). Only API model adapters take
    image inputs (sense.models_api); the local adapters in sense.models take text and vector inputs only, even when
    their ModelInfo.native lists "image"."""
    return models.takes_as_is(info, "image")


_SHOWN_IMAGES = ("png", "jpeg")  # the image formats chat shows a model as they are (what SenseInput.image holds)


def _not_native(info: ModelInfo, modality: str, fmt: str = "") -> str | None:
    """Why chat cannot give the model a file of `modality` (agent.detect format `fmt`) as it is, or None when it can
    (text, and PNG or JPEG images for a model that takes images)."""
    if modality == "text":
        return None if "text" in info.native else f"{info.label} does not read text files."
    if modality == "image":
        if _takes_images(info):
            if not fmt or fmt in _SHOWN_IMAGES:
                return None
            return (f"only PNG and JPEG images can be shown to it, and this is a {fmt.upper()} image. Run the agent "
                    f"on such a file to give it a sense for {fmt.upper()} images.")
        if "image" in info.native:
            return (f"{info.label} cannot be shown images here: image inputs are wired up for API models only. Run "
                    "the agent on an image to give it an image sense, which describes images as text.")
        return f"{info.label} does not read images. Run the agent on an image to give it an image sense."
    what = words(modality)
    return f"{info.label} does not read {what} files as they are. Run the agent on such a file to give it a {what} sense."


# agent.detect's modality names: a sense installed under any other name is an agent-written adapter for a file
# agent.detect calls "other" (csv, fasta, an unknown binary), installed under the name its design gave it (never
# one of these: agent.synth renames such a design)
_DETECT_MODALITIES = DETECT_MODALITIES


def _reads_as_text(modality: str | None, path: Path) -> bool:
    """Whether a file can fall back to being given as text when no sense takes it: a text file agent.detect reads
    as a molecule, places or a mesh (SMILES, coordinates, OBJ...), as before agent.detect knew those modalities.
    Never a file agent.detect calls "other" (CSV, FASTA, an unknown binary)."""
    return modality in ("3d", "molecule", "geo") and is_text(path)


def _could_have_sense(modality: str, rows: list[dict]) -> bool:
    """Whether the installed senses (registry.installed rows) may hold one for a file of `modality`."""
    names = {row.get("modality") for row in rows}
    if modality == "other":
        return any(name not in _DETECT_MODALITIES for name in names)
    return modality in names


def _installed(model: str) -> list[dict]:
    """registry.installed(model), or [] when it cannot be read: a chat that may need no sense still goes on (a
    sense that is really needed and broken is reported when it is loaded)."""
    try:
        return registry.installed(model)
    except Exception:
        log.warning("The installed senses of %s could not be listed.", model, exc_info=True)
        return []


def _adapter_formats(detected: dict) -> list[str]:
    """The agent.detect formats an agent-written adapter was built from or scored on (its manifest's "detected")."""
    formats = detected.get("formats")
    if isinstance(formats, list) and formats:
        return [str(f) for f in formats]
    return [str(detected.get("format") or "")]


def _sense_for(bridges: list, modality: str, fmt: str, path: Path) -> tuple[Any, str]:
    """The installed sense for a file of `modality` and agent.detect format `fmt`: (bridge, ""), or (None, why no
    installed sense takes it, or "" when none is installed for its modality). An agent-written adapter (a manifest
    with "detected": how agent.detect described the file it was built from, and the formats it was scored on) takes
    only files agent.detect describes alike: the same modality and one of those formats, and for format "unknown"
    the same file extension. Its card covers nothing else, so an OBJ adapter is never handed a PLY file. Any other
    sense (an audio bridge, or an adapter saved before manifests named their formats) takes every file of its
    modality."""
    suffix = path.suffix.lower()
    unverified = None  # an adapter for this modality, built and scored on other formats
    for bridge in bridges:
        detected = (getattr(bridge, "manifest", None) or {}).get("detected")
        if isinstance(detected, dict) and detected.get("modality"):
            if detected.get("modality") != modality:
                continue
            if fmt in _adapter_formats(detected) and (
                    fmt != "unknown" or (bool(suffix) and detected.get("suffix") == suffix)):
                return bridge, ""
            unverified = unverified or bridge
        elif modality != "other" and bridge.modality == modality:
            return bridge, ""
    if unverified is None or modality == "other":  # adapters for other kinds of file say nothing about this one
        return None, ""
    built = ", ".join(f.upper() for f in _adapter_formats(unverified.manifest["detected"]))
    return None, (f"the installed {words(modality)} sense {unverified.name} was built and checked on {built} files "
                  f"only, so it is not given a {fmt.upper()} file. Run the agent on this file to give it a sense for "
                  f"{fmt.upper()} files.")


def _read_text(path: Path) -> tuple[str, bool]:
    """The start of a text file, decoded as agent.detect reads it (UTF-16 after a byte-order mark, else UTF-8), at
    most TEXT_FILE_CHARS characters; and whether the file goes on past that."""
    with path.open("rb") as handle:
        raw = handle.read(TEXT_FILE_CHARS * 4 + 4)  # enough bytes for TEXT_FILE_CHARS characters in any encoding
        more = bool(handle.read(1))
    codec = "utf-16" if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else "utf-8-sig"
    text = raw.decode(codec, errors="replace")
    return text[:TEXT_FILE_CHARS], more or len(text) > TEXT_FILE_CHARS


def _cannot_save(name: str, exc: OSError) -> HTTPException:
    return HTTPException(
        status_code=500,
        detail=f"Could not save '{name}' on the server ({exc.strerror or type(exc).__name__}). "
        "Check the server's disk space, then send it again.",
    )


def _save_upload(upload: UploadFile, path: Path) -> None:
    """Copy an upload to `path`, a chunk at a time. A 413 when it holds more than MAX_UPLOAD_BYTES, a 500 when the
    server cannot write it; the caller removes what was written."""
    name = upload.filename or "the upload"
    total = 0
    try:
        with open(path, "wb") as handle:
            while chunk := upload.file.read(UPLOAD_CHUNK):
                total += len(chunk)
                if total > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"'{name}' is larger than {MAX_UPLOAD_BYTES // 2**20} MB, the most one file may be. "
                        "Send a shorter or smaller file.",
                    )
                handle.write(chunk)
    except OSError as exc:
        raise _cannot_save(name, exc) from exc


def _safe_name(filename: str | None) -> str:
    """A file name that is safe on the server's disk: the base name of `filename` with only letters, digits, spaces
    and ._- kept (anything else becomes _), cut to NAME_BYTES bytes with its extension kept; "upload" if nothing
    is left."""
    name = "".join(c if c.isalnum() or c in "._- " else "_" for c in Path(filename or "").name).strip(" .")
    if len(name.encode()) > NAME_BYTES:
        stem, suffix = os.path.splitext(name)
        if len(suffix.encode()) > 16:  # not an extension worth keeping
            stem, suffix = name, ""
        stem = stem.encode()[: NAME_BYTES - len(suffix.encode())].decode(errors="ignore")
        name = (stem.rstrip(" .") + suffix).strip(" .")
    return name or "upload"


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
            info = next(m for m in models.available_models() if m.name == body.model)  # _check_model passed
            problem = datasets.fit_problem(datasets.load_labelled_set(body.dataset), body.scope,
                                           open_weights=info.open_weights)
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
    chat uploads it outlives the request: the agent run reads it from its own thread after the request has
    returned, and the folder is removed once the run has ended (_record_when_finished). Raises a 413 or a 500,
    and then leaves nothing behind."""
    folder = config.home() / UPLOADS / os.urandom(6).hex()
    try:
        folder.mkdir(parents=True)
        path = folder / _safe_name(upload.filename)
        _save_upload(upload, path)
    except OSError as exc:
        shutil.rmtree(folder, ignore_errors=True)
        raise _cannot_save(upload.filename or "the upload", exc) from exc
    except BaseException:
        shutil.rmtree(folder, ignore_errors=True)
        raise
    return path


def _sweep_uploads() -> None:
    """Remove upload folders older than UPLOAD_KEEP_SECONDS: left by a server that stopped before the run that
    read them ended (no run takes that long). Never raises."""
    root = config.home() / UPLOADS
    cutoff = time.time() - UPLOAD_KEEP_SECONDS
    try:
        for folder in root.iterdir() if root.is_dir() else ():
            if folder.is_dir() and folder.stat().st_mtime < cutoff:
                shutil.rmtree(folder, ignore_errors=True)
    except OSError:
        log.warning("Old uploads under %s could not be removed.", root, exc_info=True)


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
    _sweep_uploads()
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
        except BaseException:  # no run will read the upload
            shutil.rmtree(path.parent, ignore_errors=True)
            raise
        _record_when_finished(job_id, upload=path.parent)
    return {"job_id": job_id}


@app.get("/api/jobs/latest")  # declared before /api/jobs/{job_id}, which would take "latest" for an id
def api_latest_job() -> dict:
    job = jobs.latest()
    return {"job_id": job.id if job is not None and job.status == "running" else None}


@app.get("/api/jobs/{job_id}")
def api_job(job_id: str) -> dict:
    replayed = replay.get(job_id)
    if replayed is not None:
        state, info = replayed
        return state.to_dict() | {"replay": info, "pending_approval": None}
    state = jobs.get(job_id)
    if state is None:
        raise _bad_request(f"Unknown job '{job_id}'. Start a run with POST /api/fit first.")
    return state.to_dict() | {"pending_approval": jobs.pending(job_id)}


@app.post("/api/jobs/{job_id}/approval")
async def api_answer_approval(job_id: str, request: Request) -> dict:
    # The body is read by hand, not by a pydantic model, so that every bad body is a 400 that says what to send
    # (FastAPI would answer 422), and a JSON 1 or "yes" is never taken for true.
    if replay.get(job_id) is None and jobs.get(job_id) is None:
        raise HTTPException(status_code=404, detail=f"Unknown job '{job_id}'. Use the job id POST /api/auto returned.")
    form = 'Send {"id": "<the request id from pending_approval>", "approve": true} or "approve": false.'
    try:
        body = json.loads(await request.body() or b"null")
    except (ValueError, RecursionError):
        raise _bad_request(f"The body is not valid JSON. {form}")
    if not isinstance(body, dict) or not isinstance(body.get("id"), str) or not body["id"].strip():
        raise _bad_request(f"The body needs the id of the open request as a string. {form}")
    if not isinstance(body.get("approve"), bool):
        raise _bad_request(f"\"approve\" must be true or false. {form}")
    if replay.get(job_id) is not None or not jobs.answer(job_id, body["id"], body["approve"]):
        open_request = jobs.pending(job_id) if replay.get(job_id) is None else None
        now = (f" The open request is {open_request['id']}." if open_request
               else " Nothing is waiting for an answer now.")
        raise HTTPException(
            status_code=409,
            detail=f"Request '{body['id']}' of job '{job_id}' is not open: it was answered, timed out or cancelled, "
            f"or the id is wrong.{now} See pending_approval in GET /api/jobs/{job_id}.",
        )
    return {"answered": True, "approve": body["approve"]}


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
    message: str  # what the model is sent: the user's message, then the text of each attached text file
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
    uploads to temporary files (appended to `temp_paths`, which the caller deletes; each keeps a short extension),
    load the model and give it each file: as it is (a text file, or an image for a model that takes images), through
    the installed sense for the file's modality (an agent-written adapter runs on the file in its sandbox), or as
    text when it is a mesh, molecule or places in text form that no sense takes. Raises HTTPException."""
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
        # the extension is kept (adapters and some readers go by it), but only a short one of letters and digits
        suffix = Path(upload.filename).suffix.lower()
        try:
            fd, tmp = tempfile.mkstemp(suffix=suffix if _SAFE_SUFFIX.fullmatch(suffix) else "")
            os.close(fd)
        except OSError as exc:
            raise _cannot_save(upload.filename, exc) from exc
        temp_paths.append(Path(tmp))
        _save_upload(upload, Path(tmp))
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
    kinds = [_kind_of(name, path) for name, path in attached]  # (modality, agent.detect format) of each file
    # Chat that needs no sense must not depend on the installed ones: they are loaded only for a file no other way
    # can carry (audio, video, an image the model cannot be shown, a binary mesh), or for a file that could go as
    # text (a mesh, molecule or places in text form; a file agent.detect calls "other") when the registry lists a
    # sense that may take it.
    sensed = [(m, path) for (_, path), (m, fmt) in zip(attached, kinds) if m and _not_native(base.info, m, fmt)]
    maybe = [m for m, path in sensed if m == "other" or _reads_as_text(m, path)]
    bridges: list = []
    needed = len(maybe) < len(sensed)
    if not needed and maybe:
        rows = _installed(model)
        needed = any(_could_have_sense(m, rows) for m in maybe)
    if needed:
        try:
            bridges = registry.get_bridges(model)
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    inputs: list[SenseInput] = []
    texts: list[str] = []  # the text files, added to the message after it
    used: list[dict] = []
    tool_used: list[tuple[dict, SenseInput]] = []  # entries whose detail is only known after the reply

    def not_shown(name: str, why: str) -> None:
        detail = f"'{name}' was not shown to the model: {why}"
        used.append({"bridge": "", "kind": "none", "source": name, "detail": detail})

    def give_text(name: str, path: Path, note: str = "") -> None:
        text, cut = _read_text(path)
        texts.append(f'The attached file "{name}" holds this text'
                     f'{" (only its start is given here: the file is longer)" if cut else ""}:\n{text}')
        detail = "its text was given to the model with the message"
        if cut:
            detail = f"the first {TEXT_FILE_CHARS} characters of its text were given to the model with the message"
        if note:  # why its sense did not carry it
            detail += f", since {note[:1].lower()}{note[1:]}"
        used.append({"bridge": "", "kind": "native", "source": name, "detail": detail})

    for (name, path), (modality, fmt) in zip(attached, kinds):
        if modality is None and path.stat().st_size == 0:
            not_shown(name, "it is empty.")
            continue
        why_not_native = _not_native(base.info, modality, fmt) if modality else "it is not a kind of file chat knows."
        bridge, why_no_sense = _sense_for(bridges, modality, fmt, path) if modality and why_not_native else (None, "")
        # a mesh, molecule or places in text form that no sense carries goes as text, as it did before senses
        as_text = bool(why_not_native) and "text" in base.info.native and _reads_as_text(modality, path)

        def without_sense(why: str, name: str = name, path: Path = path, as_text: bool = as_text) -> None:
            if as_text:
                give_text(name, path, note=why)
            else:
                not_shown(name, why)

        if bridge is None and as_text:
            give_text(name, path, note=why_no_sense)
            continue
        if bridge is None and modality in (None, "other"):
            images = ", PNG and JPEG images" if _takes_images(base.info) else ""
            if fmt != "unknown":
                what = f"a {fmt.upper()} file"
            elif path.suffix:
                what = f"a {path.suffix.lower()} file in a format agent.detect does not know"
            else:
                what = "a file in a format agent.detect does not know"
            not_shown(name, f"no installed sense takes this kind of file ({what}), and as it is the model can be given "
                            f"only text files{images} and audio files ({' '.join(sorted(AUDIO_EXTENSIONS))}). Run the "
                            "agent on this file to write it a sense.")
            continue
        if why_not_native is None and modality == "text":
            give_text(name, path)
            continue
        if why_not_native is None:  # a PNG or JPEG image, for a model that takes images
            if path.stat().st_size > IMAGE_BYTES:
                not_shown(name, f"it is larger than {IMAGE_BYTES // 2**20} MB, the most an image may be.")
                continue
            image = path.read_bytes()
            if not image.startswith((b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff")):  # what SenseInput.image may hold
                not_shown(name, "only PNG and JPEG images can be shown to it.")
                continue
            inputs.append(SenseInput(kind="image", modality="image", source=name, image=image,
                                     detail="shown to the model as an image"))
            used.append({"bridge": "", "kind": "native", "source": name, "detail": inputs[-1].detail})
            continue
        if bridge is None:
            if why_no_sense:
                not_shown(name, why_no_sense)
            elif modality == "audio":
                not_shown(name, f"{base.info.label} has no installed audio sense. Run the agent for this model first.")
            else:
                not_shown(name, why_not_native)
            continue
        if bridge.kind == "image" and not _takes_images(base.info):  # an adapter that renders the file as pictures
            without_sense(f"its sense {bridge.name} turns it into images, and {base.info.label} cannot be shown "
                          "images here (image inputs are wired up for API models only).")
            continue
        try:
            item = bridge.prepare(path)  # an adapter bridge runs the agent's code on the file in the sandbox
        except AdapterError as exc:  # the agent's code failed on this file; answer with the others (as text if it is)
            without_sense(f"the agent-written adapter {bridge.name} could not read it: "
                          f"{exc.reason.replace(str(path), name).replace(path.name, name)}")
            continue
        except (ValueError, FileNotFoundError) as exc:  # this file is unreadable; answer with the others
            not_shown(name, str(exc).replace(str(path), name).replace(path.name, name))
            continue
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Could not read '{name}' with bridge '{bridge.name}': {exc}")
        item.source = name  # uploads sit under temporary names; show the name the user gave
        inputs.append(item)
        used.append({"bridge": bridge.name, "kind": bridge.kind, "source": item.source, "detail": item.detail})
        if item.kind == "tool" and item.tool is not None:
            tool_used.append((used[-1], item))
    prompt = "\n\n".join([message, *texts])  # what the model is sent: the message, then each text file
    return _ChatTurn(base=base, message=prompt, inputs=inputs, used=used, tool_used=tool_used, history=earlier)


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
                reply = turn.base.chat(turn.message, turn.inputs, max_new_tokens=CHAT_TOKENS, **turn.options())
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
def _record_when_finished(job_id: str, upload: Path | None = None) -> None:
    """Record the agent run with sense.runs once it has finished, from a daemon thread, then remove `upload` (the
    folder of the file POST /api/auto kept for the run), whether or not the recording worked. Never raises.
    Call with _start_lock held, so that a fit or replay started next finds this watcher in _recorder."""
    global _recorder
    home = config.home()

    def watch() -> None:
        ended = False
        try:
            while True:
                state = jobs.get(job_id)
                if state is None:  # the job is gone (jobs were reset): nothing to record; _sweep_uploads removes it
                    return
                if state.status != "running":
                    break
                time.sleep(WATCH_SECONDS)
            ended = True
            if config.home() != home:
                log.warning("Agent run %s was not recorded: SENSE_HOME changed while it ran.", job_id)
                return
            runs.save_run(state, origin="server")
        except Exception:
            log.exception("Agent run %s could not be recorded for replay.", job_id)
        finally:
            if ended and upload is not None:  # the run has ended, so nothing reads its upload any more
                shutil.rmtree(upload, ignore_errors=True)

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
