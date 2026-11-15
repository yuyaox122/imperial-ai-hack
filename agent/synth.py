"""Agent-written adapters: the agent side. For a file of any kind, an LLM designs an adapter, writes the Python code
`adapt(path)` that turns the file into text (any base model reads it) or into images (for models that read
images), and repairs it on errors; the harness (sense.adapters) runs that code in a sandbox, verifies the resulting
bridge on labelled probe files whose labels come from an independent source, and the rule planner's text bars
decide whether it is installed.

Public surface (do not rename):
  DESIGN_SCHEMA, CODE_SCHEMA, GENERATOR_SCHEMA
      strict JSON schemas for agent.llm.ask_json: the design; {"code"} for an adapter; {"code"} for a generator
  Design
      modality, format, route ("text" | "images"), description, libraries (toolbox names), question, label_hint,
      labels (candidate answers; at least 2 when probe_plan is "generate"), probe_plan ("user" | "dataset" |
      "generate"), probe_reason; .kind is the bridge kind ("text" or "image"); to_dict()
  file_head(path, limit=2048) -> str
      the first bytes of a file for a prompt: text when they decode as text, else a hex dump of the first 768
  design(detection, goal, head, toolbox, models, *, name="", size=None) -> Design | None
      asks the LLM for a design from the detection (modality, format, how, detail), the goal, the file's head (at
      most 2 KB), its suffix and size, the toolbox and the models (dicts with "name" and "native"). Every field is
      checked (validate_design); a design with problems is sent back once with the problems listed. None when the
      LLM gives no usable answer (always in fake mode, where ask_json returns None)
  validate_design(answer, toolbox, models, detection=None) -> (Design | None, problems: list[str])
      the checks behind design(): route "text" or "images", "images" only when some model has "image" in native;
      libraries matched to the toolbox (case-insensitive, "pillow" reads "PIL", "opencv" reads "cv2", ...), with
      standard-library names dropped and network modules refused; non-empty modality, format, description,
      question and label_hint; probe_plan one of the three; labels cleaned so that they round-trip as folder names
      (no "_", "-" or "/"), distinct, at most 8, and at least 2 for "generate"
  write_adapter(design, sample, *, attempts=3, timeout=60, on_attempt=None) -> (code, output, attempts_used)
      asks the LLM for {"code"} defining adapt(path) (the prompt holds the design without its candidate labels and
      probe plan, the toolbox, the output contract of sense.adapters.run_adapter, the rules: describe the content
      as measurements rather than a verdict, never guess a label from the file name, no network; and the sample's
      head), checks it before running (it parses, defines adapt, imports only the standard library and the
      toolbox, no network modules, no ctypes or cffi), runs it with sense.adapters.run_adapter on the sample and
      checks the output (the design's route; text not empty). Every failure goes back to the LLM for a repair, up
      to `attempts` in all (the reason cut to 1000 characters, the traceback to 1500). on_attempt(k,
      error_or_None) is called after each attempt. Raises sense.adapters.AdapterError naming every attempt's error
      (and with .attempts, one dict per attempt) when none works, at once when the sandbox cannot be applied.
      Causes beyond sense.adapters': "llm" (no code came back) and "code" (the code failed a check before it ran)
  write_generator(design, labels, n_per_label, *, earlier_attempts=None, extension="") -> code
      asks the LLM for {"code"} defining generate(out_dir, labels, n): for each label, n files with `extension`
      written into os.path.join(out_dir, label), each built to have its label, which is never written into a
      file. Only for a design whose probe_plan is "generate" (ValueError otherwise); AdapterError when no usable
      code comes back (with .code, the rejected code, when it failed the checks before running)
  run_generator(code, out_dir, labels, n, *, timeout=300) -> {"per_label": {label: files}, "seconds", "notes"}
      runs generate(out_dir, labels, n) through sense.adapters.run_adapter (the generator is wrapped as an adapter
      whose input is a small JSON spec, with out_dir as its writable folder), so where the sandbox applies it has
      no network and may write only to out_dir and its own temporary folder; elsewhere only run_adapter's
      environment and proxy measures apply. AdapterError when it fails
  scratch_dir(prefix="sense-synth-") -> Path
      a new temporary folder (the caller removes it)
  generate_probes(design, labels=None, n_per_label=20, *, attempts=3, on_attempt=None, suffix="")
                  -> (ProbeSet, code, used)
      write_generator, run_generator in a scratch folder, check that every label got at least half of its files
      (with `suffix`, counting only files with that extension) and that the files do not name their own label
      (label_leaks), then sense.adapters.probe_folder(..., provenance="generated"), which copies them under
      SENSE_HOME with anonymous names; failures go back to the LLM with the failing code, up to `attempts`.
      AdapterError (with .attempts) when none works. The probe set's source says what such labels are worth
      (GENERATED_CAVEAT): the generator and the adapter come from the same LLM, so a pass is a consistency check
      of the adapter against what the generator built, not independent ground truth
  label_leaks(groups) -> list[str]
      for {label: [files]}: one sentence per label that is written in the content of most of its own files (and
      clearly less often in the other labels' files); [] when no label leaks. A heuristic: it matches the label's
      words in any case and inside longer words, also inside gzip streams and zip archives, and nothing else
  assess(request, card) -> (acceptable, sentence)
      agent.planner.RulePlanner().assess, with an "image" card judged by the text bars (accuracy >= chance + 0.15
      and >= swapped + 0.10); then at least MIN_VERIFY_ITEMS (10) scored probe files and the low end of the 95%
      Wilson interval above chance, so that a lucky adapter on a handful of files is not installed
  registry_loads(bridge) -> str | None
      None when sense.registry can load the adapter bridge back once installed (its load_bridge must know adapter
      bridges); else why not
  run_synth(model_name, file, goal, emit, *, examples=None, detection=None) -> (cards, installed)
      the whole run, as a trace of contracts.TraceEvent (the say() pattern of agent.auto; every event's data is
      JSON-serialisable; times count from the run's start):
        [design]  start; ok with the design's fields, detection, toolbox and head_preview (fail, then done, when
                  the LLM gives no usable design)
        [build]   start (sandboxed); fail, then done, when sense.adapters.sandbox_problem() refuses to run code;
                  info with warning "unsandboxed" when there is no sandbox to apply; info per failed attempt
                  (attempt, error); ok with attempts, libraries, code_sha256, code, preview (the first 300
                  characters of the output on the file, or the image sizes), output_kind, notes, seconds (fail,
                  then done, when no attempt works)
        [data]    probe files: a user examples folder when `examples` is given (only files with the user's file's
                  extension; MIN_PER_LABEL (5) per label; the set is named <modality>-examples-<hash>), else
                  generated files when the design's probe_plan is "generate" (info per failed generator attempt),
                  else no probes: info, then done with needs_user=True, saying which examples to provide (wanted:
                  labels, label_hint, format, min_per_label, question). An unusable examples folder or no working
                  generator: fail, then the same needs_user done. ok with name, provenance, source, licence,
                  labels, counts, n, split, meta, label_leaks (warnings: labels written in their own files'
                  content, extensions that differ between labels), and for generated files generator_sha256,
                  generator_code and caveat
        [verify]  sense.adapters.verify_files on every probe file (the adapter was written before any existed, so
                  all are held out; an API model scores at most config.api_verify_items() of them): ok with the
                  card dict ("card") and seconds (fail, then done, when the adapter fails on more than half)
        [decide]  assess() on the card (FitRequest: dataset = the probe set's name, modality = design.modality,
                  scope "closed"); the card's verdict becomes "installed" or "rejected", its reason the sentence
                  followed by the adapter's failures, the label-leak warnings and, for generated probes, the caveat
        [install] when assess passes and registry_loads(bridge) is None: ok after sense.registry.install(bridge,
                  card) (the generator's code is saved beside the bridge as generator.py); when assess passes but
                  the registry cannot load adapter bridges: info (blocked), nothing installed, verdict "candidate"
                  (installing would break the model's chat); info "Installed nothing" otherwise
        [done]    ok: installed (Bridge.name or None) and needs_user
        [error]   fail: step, error, after a fail event at the step that raised; the exception is re-raised
      The bridge is a sense.adapters.AdapterBridge; its code (adapter.py) and manifest (bridge.json: modality,
      format, route, description, libraries, written_by, attempts, question, goal, code_sha256, design, probes
      with label_leaks and, for generated probes, caveat) are saved with it. Returns ([card], installed name or
      None); ([], None) when the run stopped before a card.

Labels never reach an adapter through a file name or a folder: the adapter is written from the user's file alone,
before any probe exists, and never sees the candidate labels; every probe file is copied to <sha256[:16]><extension>
by sense.adapters.probe_folder, and sense.adapters.run_adapter hands the adapter a private copy named input<ext> in
a folder of its own, under a macOS sandbox that cannot read the probe store, the examples folder or anything else
in the user's folders (without that sandbox, only the copy and the anonymous names stand).
"""

from __future__ import annotations

import ast
import codecs
import dataclasses
import hashlib
import io
import json
import math
import os
import re
import shutil
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from agent import llm
from agent.detect import detect
from agent.planner import RulePlanner, pct
from sense import adapters, models, registry
from sense.adapters import AdapterBridge, AdapterError, ProbeSet
from sense.contracts import FitRequest, ReportCard, TraceEvent

HEAD_TEXT_BYTES = 2048  # at most this much of a file is shown to the LLM
HEAD_HEX_BYTES = 768  # a binary file's head is shown as a hex dump of this many bytes
ADAPTER_ATTEMPTS = 3
GENERATOR_ATTEMPTS = 3
GENERATED_PER_LABEL = 20  # probe files generated per label; every one of them is scored
GENERATOR_TIMEOUT = 300.0
ADAPTER_TIMEOUT = adapters.DEFAULT_TIMEOUT
MIN_PER_LABEL = 5  # user examples per label: with 2 labels, MIN_VERIFY_ITEMS files to score
MIN_VERIFY_ITEMS = 10  # an adapter card scored on fewer probe files is not installed
MAX_LABELS = 8
PREVIEW_CHARS = 300
DESIGN_TOKENS = 1500
CODE_TOKENS = 5000
LLM_TIMEOUT_S = 120.0  # writing code takes longer than agent.llm's default 30 s allows
_HEAD_CHARS = 6000  # cap on the head string sent with a design request
_ERROR_CHARS = 1500  # how much of a traceback goes back to the LLM per failed attempt
_REASON_CHARS = 1000  # and how much of an error's reason
_LEAK_READ_BYTES = 4 << 20  # label_leaks reads at most this much of each file (and of its unpacked members)

_NETWORK_MODULES = {
    "socket", "ssl", "http", "urllib", "ftplib", "smtplib", "poplib", "imaplib", "nntplib", "telnetlib", "xmlrpc",
    "socketserver", "webbrowser", "requests", "httpx", "aiohttp", "urllib3", "websocket", "websockets",
}
_FORBIDDEN_MODULES = {"ctypes", "_ctypes", "cffi"}  # an adapter needs no foreign calls (they reach past the sandbox)
# What the adapter's author sees of the design: not the candidate labels or where the probes come from, so the
# adapter reports what is in the file and the base model, not the adapter, makes the call.
_ADAPTER_DESIGN_FIELDS = ("modality", "format", "route", "description", "libraries", "question", "label_hint")
_STDLIB = set(sys.stdlib_module_names)
_ALIASES = {
    "pillow": "PIL", "pil": "PIL", "opencv": "cv2", "opencv-python": "cv2", "opencv_python": "cv2",
    "pyav": "av", "xml": "xml.etree", "xml.etree.elementtree": "xml.etree", "elementtree": "xml.etree",
    "scikit-image": "skimage", "ffmpeg-python": "ffmpeg",
}


# ----------------------------------------------------------------------------- schemas
def _strings(description: str) -> dict:
    return {"type": "array", "items": {"type": "string"}, "description": description}


DESIGN_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "modality", "format", "route", "description", "libraries", "question", "label_hint", "labels",
        "probe_plan", "probe_reason",
    ],
    "properties": {
        "modality": {"type": "string", "description": "short lowercase name of the kind of content, e.g. 3d-model"},
        "format": {"type": "string", "description": "the file format, e.g. obj, stl, mp4, smiles, geojson"},
        "route": {"type": "string", "enum": ["text", "images"]},
        "description": {"type": "string", "description": "what the adapter extracts or renders"},
        "libraries": _strings("names copied from the toolbox that the adapter needs"),
        "question": {"type": "string", "description": "the multiple-choice question verification asks per file"},
        "label_hint": {"type": "string", "description": "what kind of answer options the question has"},
        "labels": _strings("2 to 8 short candidate answers to the question"),
        "probe_plan": {"type": "string", "enum": ["user", "dataset", "generate"]},
        "probe_reason": {"type": "string", "description": "why that source of labelled probe files, in one sentence"},
    },
}

CODE_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "required": ["code"],
    "properties": {"code": {"type": "string", "description": "complete Python code defining adapt(path)"}},
}

GENERATOR_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "required": ["code"],
    "properties": {
        "code": {"type": "string", "description": "complete Python code defining generate(out_dir, labels, n)"},
    },
}


# ----------------------------------------------------------------------------- prompts
DESIGN_INSTRUCTION = (
    "You design an adapter that gives a frozen base model a new sense. The adapter is Python code, adapt(path), "
    "that turns one file of the given kind into something the model can read. Route \"text\": it writes a factual "
    "description of the file's content (measured quantities, structure, values, in plain words) that any model "
    "reads. Route \"images\": it renders the content into pictures (views, frames, plots) for a model that reads "
    "images; allowed only when images_allowed is true, and only when pictures carry the content better than text. "
    "You get the detection of the file from its bytes, its first bytes (head: text, or a hex dump), its name "
    "suffix and size, the user's goal, the toolbox (the only third-party libraries and programs the adapter may "
    "use; the Python standard library needs no listing) and the base models. Answer with: modality (a short "
    "lowercase name for the kind of content, such as 3d-model, video, molecule or location); format (the file "
    "format, such as obj, stl, mp4, smiles, sdf, geojson or csv); route; description (one or two sentences: what "
    "the adapter extracts or renders, so that the model can answer the question); libraries (names copied from "
    "the toolbox that the adapter needs; [] for the standard library alone); question (the multiple-choice "
    "question verification will ask about each file, matching the goal); label_hint (what kind of answer options "
    "the question has); labels (2 to 8 short, distinct candidate answers to the question; required when "
    "probe_plan is generate); probe_plan and probe_reason. probe_plan says where labelled probe files come from; "
    "their labels must not come from the adapter: \"generate\" only when a short script using the toolbox can "
    "build the labelled property into each file's content, so that the label is a fact of construction that the "
    "script itself computes (meshes built as known shapes, videos made with a known motion or colour, numbers in "
    "known ranges, points inside boxes the script defines); never when the label is knowledge about the content "
    "that the script would have to recall (a molecule's taste or smell, the country or city a coordinate lies "
    "in, a species, a style): such labels need \"dataset\" (a public labelled dataset fits the goal) or \"user\" "
    "(only the user can supply labelled examples). Prefer generate when such a script is possible. probe_reason "
    "says why, in one sentence. When problems_with_last_answer is present, your last answer had those problems: "
    "fix them."
)

ADAPTER_INSTRUCTION = (
    "You write the adapter of a design: Python code that defines adapt(path). It runs in a sandbox with no "
    "network, once per file, and what it returns is all the base model will know about the file. Follow "
    "contract and rules. Use only the Python standard library and the design's libraries, which come from "
    "toolbox (a name ending in \"(via subprocess)\" is a program to run with subprocess). sample shows the file "
    "it will first run on. When earlier_attempts is not empty, those attempts failed with the errors shown "
    "(last_code is the code of the last one): fix the causes. Answer with the complete code in code, without "
    "Markdown fences."
)

ADAPTER_CONTRACT = (
    'adapt(path) gets the absolute path of one file as a str and returns exactly one of {"text": str} or '
    '{"images": [...]}. Images may be PNG or JPEG bytes, PIL images, paths to image files or uint8 numpy arrays '
    f"(H x W or H x W x 3); at most {adapters.MAX_IMAGES} are kept, each scaled to at most "
    f"{adapters.MAX_IMAGE_SIDE} px on its long side. Text is cut to {adapters.MAX_TEXT_CHARS} characters. "
    "Returning anything else, raising, crashing or running out of time fails."
)

ADAPTER_RULES = [
    "Describe the file's content: what is in it, measured from its bytes (sizes, counts, shapes, values, colours, "
    "motion, positions), in plain words a reader can use to answer the design's question.",
    "Report measurements and observations, not a verdict: do not answer the design's question or name a category "
    "for the file. The base model reads your description and makes that call.",
    "Never guess or state a label from the file name or path: the adapter always gets a copy named input with the "
    "original extension, so the name says nothing. Read the content only.",
    'Return {"text": ...} for route text and {"images": [...]} for route images.',
    "No network: the sandbox blocks it. Write temporary files only under the working directory or tempfile.",
    "Be deterministic and quick (well under the time limit per file); sample large files instead of reading "
    "everything.",
    "When an optional detail cannot be computed, leave it out instead of failing; fail only when the file cannot "
    "be read at all.",
    "Keep the code short (under about 150 lines), with no __main__ block.",
]

GENERATOR_INSTRUCTION = (
    "You write a generator of labelled probe files, used to verify an adapter: Python code that defines "
    "generate(out_dir, labels, n). For each label in labels it must create the folder os.path.join(out_dir, "
    "label) and write n distinct files of the design's format into it, each one a file whose label is true by "
    "construction: its content is built to have that label (a mesh built as that shape, a video made with that "
    "motion, coordinates sampled inside that region). Follow rules. Use only the Python standard library and "
    "toolbox (a name ending in \"(via subprocess)\" is a program to run with subprocess). When earlier_attempts "
    "is not empty, those attempts failed with the errors shown (last_code is the code of the last one): fix the "
    "causes. Answer with the complete code in code, without Markdown fences."
)

GENERATOR_RULES = [
    "Never write the label, or any name, comment, title or metadata derived from it, into a file (not even in "
    "another case, plural or compressed): the label must be a property of the content alone. Name the files "
    "sample_000, sample_001 and so on, with the extension given in extension (the format's own when it is "
    "empty), and write no other files.",
    "Vary everything that is not the label at random (size, position, orientation, scale, noise, colours, "
    "duration, counts), so that only the labelled property separates the folders and no two files are identical.",
    "Make the labelled property plain to a careful observer of the content.",
    "Seed the random generator with a fixed number so the files are reproducible.",
    "No network: the sandbox blocks it. Write only inside out_dir (and tempfile for scratch files).",
    f"Finish within {GENERATOR_TIMEOUT:g} s for all files together; keep each file small (under 2 MB).",
    "Keep the code short (under about 150 lines), with no __main__ block.",
]

_GENERATE_WRAPPER = '''


def adapt(path):  # added by agent.synth: runs generate() from a JSON spec, in the adapter sandbox
    import json as _json
    import os as _os

    with open(path, encoding="utf-8") as _f:
        _spec = _json.load(_f)
    generate(_spec["out_dir"], list(_spec["labels"]), int(_spec["n"]))
    _counts = {}
    for _label in _spec["labels"]:
        _folder = _os.path.join(_spec["out_dir"], _label)
        _names = _os.listdir(_folder) if _os.path.isdir(_folder) else []
        _counts[_label] = sum(
            1 for _n in _names if not _n.startswith(".") and _os.path.isfile(_os.path.join(_folder, _n))
        )
    return {"text": _json.dumps(_counts)}
'''


# ----------------------------------------------------------------------------- the design
@dataclass
class Design:
    """What the LLM proposes for an adapter, checked field by field (validate_design)."""

    modality: str
    format: str
    route: str  # "text" | "images"
    description: str
    libraries: list[str]  # toolbox names
    question: str
    label_hint: str
    labels: list[str] = field(default_factory=list)
    probe_plan: str = "user"  # "user" | "dataset" | "generate"
    probe_reason: str = ""

    @property
    def kind(self) -> str:
        """The bridge kind: "text" for route text, "image" for route images."""
        return "text" if self.route == "text" else "image"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def file_head(path: Path, limit: int = HEAD_TEXT_BYTES) -> str:
    """The first bytes of `path` for a prompt: the text itself when they decode as UTF-8 text (at most `limit`
    bytes), else a hex dump of the first HEAD_HEX_BYTES bytes."""
    path = Path(path)
    size = path.stat().st_size
    with open(path, "rb") as f:
        raw = f.read(max(1, int(limit)))
    text = _as_text(raw, cut=size > len(raw))
    if text is not None:
        return text + (f"\n[... {size - len(raw)} more bytes]" if size > len(raw) else "")
    shown = raw[: min(HEAD_HEX_BYTES, len(raw))]
    lines = [f"[hex dump of the first {len(shown)} of {size} bytes]"]
    for offset in range(0, len(shown), 16):
        chunk = shown[offset : offset + 16]
        hexed = " ".join(f"{b:02x}" for b in chunk)
        ascii_ = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
        lines.append(f"{offset:08x}  {hexed:<47}  |{ascii_}|")
    return "\n".join(lines)


def _as_text(raw: bytes, *, cut: bool) -> str | None:
    if not raw:
        return ""
    try:
        text = codecs.getincrementaldecoder("utf-8")().decode(raw.removeprefix(codecs.BOM_UTF8), final=not cut)
    except UnicodeDecodeError:
        return None
    if "\x00" in text:
        return None
    control = sum(1 for ch in text if ord(ch) < 32 and ch not in "\n\r\t\f")
    return None if control > 0.02 * max(1, len(text)) else text


def _tool_key(entry: str) -> str:
    """"ffmpeg (via subprocess)" -> "ffmpeg"; other toolbox entries are their own key."""
    return entry.split(" (", 1)[0].strip()


def _match_library(name: str, toolbox: list[str]) -> str | None:
    """The toolbox entry `name` refers to, or None."""
    keys = {_tool_key(entry).lower(): entry for entry in toolbox}
    raw = " ".join(str(name).split()).strip()
    lowered = raw.lower()
    for candidate in (lowered, _tool_key(lowered), _ALIASES.get(lowered, "").lower(), lowered.split(".")[0]):
        if candidate and candidate in keys:
            return keys[candidate]
    return None


def _clean_text(value: Any, limit: int) -> str:
    if not isinstance(value, str):
        return ""
    text = " ".join(value.split())
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def _clean_label(value: Any) -> str:
    """A label that round-trips as a folder name through sense.adapters.probe_folder: no "_", "-" or path
    separators, single spaces, no leading dot, at most 40 characters."""
    if not isinstance(value, str):
        return ""
    text = re.sub(r"[_\-/\\:\x00]", " ", value)
    text = " ".join(text.split()).lstrip(".").strip()
    return text[:40].strip()


def _modality_slug(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", value.lower())).strip("-")[:40].strip("-")


def _images_allowed(models_: list[dict]) -> bool:
    return any("image" in (m.get("native") or []) for m in models_ if isinstance(m, dict))


def validate_design(
    answer: Any, toolbox: list[str], models: list[dict], detection: dict | None = None
) -> tuple[Design | None, list[str]]:
    """Check every field of an LLM's design answer; (Design, []) when usable, else (None, problems)."""
    if not isinstance(answer, dict):
        return None, ["the answer is not a JSON object with the design's fields"]
    problems: list[str] = []

    modality = _modality_slug(answer.get("modality"))
    if not modality:
        problems.append("modality is missing: give a short lowercase name for the kind of content")
    fmt = _modality_slug(answer.get("format"))
    if not fmt:
        problems.append("format is missing: name the file format")

    route = answer.get("route")
    if route not in ("text", "images"):
        problems.append(f'route is {route!r}; it must be "text" or "images"')
    elif route == "images" and not _images_allowed(models):
        names = ", ".join(str(m.get("name")) for m in models if isinstance(m, dict)) or "none"
        problems.append(f'route is "images", but no base model here reads images ({names}); use route "text"')

    libraries: list[str] = []
    raw_libraries = answer.get("libraries")
    if not isinstance(raw_libraries, list) or not all(isinstance(x, str) for x in raw_libraries):
        problems.append("libraries must be a list of names from the toolbox")
    else:
        for name in raw_libraries:
            entry = _match_library(name, toolbox)
            top = str(name).strip().split(".")[0].lower()
            if entry is not None:
                if entry not in libraries:
                    libraries.append(entry)
            elif top in _NETWORK_MODULES:
                problems.append(f"library {name!r} is for the network, which the sandbox blocks")
            elif top in {m.lower() for m in _STDLIB}:
                continue  # standard library: needs no listing
            else:
                problems.append(f"library {name!r} is not in the toolbox ({', '.join(toolbox) or 'empty'})")

    description = _clean_text(answer.get("description"), 600)
    if not description:
        problems.append("description is missing: say what the adapter extracts or renders")
    question = _clean_text(answer.get("question"), 300)
    if not question:
        problems.append("question is missing: give the multiple-choice question verification will ask")
    label_hint = _clean_text(answer.get("label_hint"), 300)
    if not label_hint:
        problems.append("label_hint is missing: say what kind of answer options the question has")

    plan = answer.get("probe_plan")
    if plan not in ("user", "dataset", "generate"):
        problems.append(f'probe_plan is {plan!r}; it must be "user", "dataset" or "generate"')
    reason = _clean_text(answer.get("probe_reason"), 400)

    labels: list[str] = []
    raw_labels = answer.get("labels", [])
    if not isinstance(raw_labels, list):
        problems.append("labels must be a list of short answers")
        raw_labels = []
    seen: set[str] = set()
    for value in raw_labels:
        label = _clean_label(value)
        if label and label.casefold() not in seen:
            seen.add(label.casefold())
            labels.append(label)
    labels = labels[:MAX_LABELS]
    if plan == "generate" and len(labels) < 2:
        problems.append(f"probe_plan is generate, but labels has {len(labels)} usable label(s); give 2 to {MAX_LABELS}")

    if problems:
        return None, problems
    return Design(
        modality=modality, format=fmt, route=str(route), description=description, libraries=libraries,
        question=question, label_hint=label_hint, labels=labels, probe_plan=str(plan), probe_reason=reason,
    ), []


def design(
    detection: dict,
    goal: str,
    head: str,
    toolbox: list[str],
    models: list[dict],
    *,
    name: str = "",
    size: int | None = None,
) -> Design | None:
    """Ask the LLM for an adapter design (see the module docstring). One repair round when the first answer has
    problems; None when there is still no usable design."""
    if hasattr(detection, "to_dict"):
        detection = detection.to_dict()
    detection = dict(detection or {})
    data = {
        "detection": {k: detection.get(k) for k in ("modality", "format", "how", "detail")},
        "goal": str(goal),
        "file": {"suffix": Path(name).suffix.lower() if name else "", "size_bytes": size},
        "head": str(head)[:_HEAD_CHARS],
        "toolbox": list(toolbox),
        "models": [{"name": m.get("name"), "native": list(m.get("native") or [])} for m in models if isinstance(m, dict)],
        "images_allowed": _images_allowed(models),
    }
    problems: list[str] = []
    for _ in range(2):
        payload = dict(data, problems_with_last_answer=problems) if problems else data
        answer = llm.ask_json(DESIGN_INSTRUCTION, payload, DESIGN_SCHEMA, max_tokens=DESIGN_TOKENS)
        if answer is None:
            problems = []
            continue
        found, problems = validate_design(answer, toolbox, models, detection)
        if found is not None:
            return found
    return None


# ----------------------------------------------------------------------------- code checks
def _code_from(answer: Any) -> str | None:
    if not isinstance(answer, dict) or not isinstance(answer.get("code"), str):
        return None
    code = answer["code"].strip()
    fenced = re.fullmatch(r"```[a-zA-Z0-9_+-]*\s*\n(.*?)\n?```", code, flags=re.S)
    if fenced:
        code = fenced.group(1)
    return code + "\n" if code.strip() else None


def _static_problem(code: str, toolbox: list[str], function: str) -> str | None:
    """Why `code` cannot be run, found without running it, or None."""
    try:
        tree = ast.parse(code)
    except SyntaxError as err:
        return f"The code does not parse: {err.msg} (line {err.lineno})."
    defined = {
        node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    if function not in defined:
        return f"The code defines no top-level function {function}(...); define it."
    keys = {_tool_key(entry) for entry in toolbox}
    for module in adapters.code_imports(code):
        if module in _NETWORK_MODULES:
            return f"The code uses {module}, a network module; the sandbox has no network, so do without it."
        if module in _FORBIDDEN_MODULES:
            return f"The code imports {module}; an adapter makes no foreign calls, so use the toolbox instead."
        if module in ("ffmpeg", "ffprobe"):
            if module not in keys:
                return f"The code runs {module}, which is not installed here; do without it."
            continue
        if module in _STDLIB or module in keys or module.split(".")[0] in {k.split(".")[0] for k in keys}:
            continue
        return (
            f"The code imports {module}, which is neither in the standard library nor in the toolbox "
            f"({', '.join(toolbox) or 'empty'}); use only those."
        )
    return None


def _route_problem(design_: Design, output: dict) -> str | None:
    if design_.route == "text":
        if "text" not in output:
            return 'The design\'s route is text, but adapt(path) returned images; return {"text": ...}.'
        if not output["text"].strip():
            return "adapt(path) returned empty text; describe the file's content."
        return None
    if "images" not in output:
        return 'The design\'s route is images, but adapt(path) returned text; return {"images": [...]}.'
    return None


def _failure(k: int, err: AdapterError, code: str | None) -> dict[str, Any]:
    """One failed attempt as it goes back to the LLM and into the trace: the reason and traceback cut short."""
    return {
        "attempt": k,
        "reason": _one_line(err.reason, _REASON_CHARS),
        "traceback": err.traceback[-_ERROR_CHARS:],
        "cause": err.cause,
        "code": code,
    }


def _gave_up(what: str, failures: list[dict]) -> AdapterError:
    """One AdapterError naming every attempt's error; .attempts holds them (without code)."""
    reasons = "; ".join(f"attempt {f['attempt']}: {_one_line(f['reason'], 300)}" for f in failures)
    details = "\n\n".join(
        f"Attempt {f['attempt']} ({f['cause']}): {f['reason']}" + (f"\n{f['traceback']}" if f["traceback"] else "")
        for f in failures
    )
    n = len(failures)
    err = AdapterError(
        f"No working {what} after {n} attempt{'' if n == 1 else 's'}: {reasons}.",
        details,
        cause=failures[-1]["cause"] if failures else "llm",
    )
    err.attempts = [{k: v for k, v in f.items() if k != "code"} for f in failures]
    return err


def _one_line(text: str, limit: int) -> str:
    line = " ".join(str(text).split())
    return line if len(line) <= limit else line[: limit - 3] + "..."


def _repair_context(failures: list[dict]) -> dict[str, Any]:
    if not failures:
        return {"earlier_attempts": []}
    return {
        "earlier_attempts": [{k: v for k, v in f.items() if k != "code"} for f in failures],
        "last_code": failures[-1].get("code") or "",
    }


_timeout_lock = threading.Lock()
_timeout_users = 0
_timeout_saved: float | None = None


@contextmanager
def _long_llm_calls():
    """Let agent.llm.ask_json wait up to LLM_TIMEOUT_S while code is being written (agent.llm reads TIMEOUT_S on
    every call). Restored when the last concurrent writer finishes."""
    global _timeout_users, _timeout_saved
    with _timeout_lock:
        if _timeout_users == 0:
            _timeout_saved = getattr(llm, "TIMEOUT_S", None)
            if isinstance(_timeout_saved, (int, float)) and _timeout_saved < LLM_TIMEOUT_S:
                llm.TIMEOUT_S = LLM_TIMEOUT_S
        _timeout_users += 1
    try:
        yield
    finally:
        with _timeout_lock:
            _timeout_users -= 1
            if _timeout_users == 0 and isinstance(_timeout_saved, (int, float)):
                llm.TIMEOUT_S = _timeout_saved


# ----------------------------------------------------------------------------- the adapter
def write_adapter(
    design: Design,
    sample: Path,
    *,
    attempts: int = ADAPTER_ATTEMPTS,
    timeout: float = ADAPTER_TIMEOUT,
    on_attempt: Callable[[int, AdapterError | None], None] | None = None,
) -> tuple[str, dict, int]:
    """Ask for adapt(path), run it on `sample`, repair on errors; (code, run_adapter output, attempts used)."""
    sample = Path(sample)
    if attempts < 1:
        raise ValueError(f"attempts is {attempts}; give at least 1.")
    if not sample.is_file():
        raise ValueError(f"{sample} is not a file; give a sample file to run the adapter on.")
    box = adapters.toolbox()
    problem = adapters.sandbox_problem()
    if problem is not None:
        raise AdapterError(problem, cause="sandbox")
    base = {
        "design": {key: value for key, value in design.to_dict().items() if key in _ADAPTER_DESIGN_FIELDS},
        "toolbox": box,
        "contract": ADAPTER_CONTRACT + f" Each file has {timeout:g} s.",
        "rules": ADAPTER_RULES,
        "sample": {"suffix": sample.suffix.lower(), "size_bytes": sample.stat().st_size, "head": file_head(sample)},
    }
    failures: list[dict] = []
    with _long_llm_calls():
        for k in range(1, attempts + 1):
            data = dict(base, attempt=k, **_repair_context(failures))
            code = _code_from(llm.ask_json(ADAPTER_INSTRUCTION, data, CODE_SCHEMA, max_tokens=CODE_TOKENS))
            if code is None:
                err = AdapterError("The LLM gave no adapter code.", cause="llm")
            elif (problem := _static_problem(code, box, "adapt")) is not None:
                err = AdapterError(problem, cause="code")
            else:
                try:
                    output = adapters.run_adapter(code, sample, timeout=timeout)
                except AdapterError as raised:
                    err = raised
                else:
                    problem = _route_problem(design, output)
                    if problem is None:
                        if on_attempt is not None:
                            on_attempt(k, None)
                        return code, output, k
                    err = AdapterError(problem, cause="output")
            failures.append(_failure(k, err, code))
            if on_attempt is not None:
                on_attempt(k, err)
            if err.cause == "sandbox":  # not the code's fault: no repair can help
                break
    raise _gave_up("adapter", failures)


# ----------------------------------------------------------------------------- generated probe files
def write_generator(
    design: Design,
    labels: list[str],
    n_per_label: int,
    *,
    earlier_attempts: list[dict] | None = None,
    extension: str = "",
) -> str:
    """Ask for generate(out_dir, labels, n) writing files whose label is true by construction; the code. When the
    code fails the checks before running, the AdapterError raised carries it as .code, for the repair round."""
    if design.probe_plan != "generate":
        raise ValueError(
            f'The design\'s probe_plan is "{design.probe_plan}", not "generate": its probe files come from '
            "elsewhere, so no generator is written."
        )
    box = adapters.toolbox()
    data = {
        "design": design.to_dict(),
        "labels": list(labels),
        "n": int(n_per_label),
        "extension": extension,
        "toolbox": box,
        "rules": GENERATOR_RULES,
        **_repair_context(list(earlier_attempts or [])),
    }
    with _long_llm_calls():
        code = _code_from(llm.ask_json(GENERATOR_INSTRUCTION, data, GENERATOR_SCHEMA, max_tokens=CODE_TOKENS))
    if code is None:
        raise AdapterError("The LLM gave no generator code.", cause="llm")
    problem = _static_problem(code, box, "generate")
    if problem is not None:
        err = AdapterError(problem, cause="code")
        err.code = code
        raise err
    return code


def scratch_dir(prefix: str = "sense-synth-") -> Path:
    """A new temporary folder (the caller removes it). run_generator lets the generator write only there."""
    return Path(tempfile.mkdtemp(prefix=prefix)).resolve()


def _count_files(folder: Path) -> int:
    if not folder.is_dir():
        return 0
    return sum(1 for p in folder.iterdir() if p.is_file() and not p.name.startswith("."))


def run_generator(
    code: str, out_dir: Path, labels: list[str], n: int, *, timeout: float = GENERATOR_TIMEOUT
) -> dict[str, Any]:
    """Run generate(out_dir, labels, n) from `code` in the adapter sandbox, which lets it write only to `out_dir`
    (and its own temporary folder); the files written per label."""
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    spec_dir = scratch_dir("sense-generator-")
    try:
        spec = spec_dir / "spec.json"
        spec.write_text(json.dumps({"out_dir": str(out_dir), "labels": list(labels), "n": int(n)}), encoding="utf-8")
        result = adapters.run_adapter(code + _GENERATE_WRAPPER, spec, timeout=timeout, writable=out_dir)
    finally:
        shutil.rmtree(spec_dir, ignore_errors=True)
    return {
        "per_label": {label: _count_files(out_dir / label) for label in labels},
        "seconds": result.get("seconds"),
        "notes": result.get("notes", []),
    }


def _label_pattern(label: str) -> re.Pattern[bytes]:
    """The label's words joined by a space, "_", "-" or nothing, matched anywhere (so "spheres", "Sphere001" and
    "SphereMesh" all match "sphere"); the text it is matched against is lowercased."""
    words = label.lower().split()
    variants = {" ".join(words), "_".join(words), "-".join(words), "".join(words)}
    alternatives = b"|".join(re.escape(v.encode("utf-8")) for v in sorted(variants, key=len, reverse=True))
    return re.compile(rb"(?:" + alternatives + rb")")


def _leak_text(path: Path) -> bytes:
    """A file's bytes for the leak check, lowercased: a gzip stream unpacked, and a zip archive's member names
    and unpacked members added (3MF, USDZ, KMZ, DOCX and the like are zip archives), all within _LEAK_READ_BYTES."""
    import zipfile
    import zlib

    def gunzip(data: bytes) -> bytes:
        try:
            return zlib.decompressobj(16 + zlib.MAX_WBITS).decompress(data, _LEAK_READ_BYTES)
        except zlib.error:
            return b""

    try:
        with open(path, "rb") as f:
            raw = f.read(_LEAK_READ_BYTES)
    except OSError:
        return b""
    parts = [raw]
    if raw[:2] == b"\x1f\x8b":
        parts.append(gunzip(raw))
    if raw[:4] == b"PK\x03\x04":
        try:
            with zipfile.ZipFile(path) as archive:
                budget = _LEAK_READ_BYTES
                for info in archive.infolist()[:200]:
                    parts.append(info.filename.encode("utf-8", "replace"))
                    if budget <= 0 or info.is_dir():
                        continue
                    with archive.open(info) as member:
                        data = member.read(budget)
                    budget -= len(data)
                    parts += [data, gunzip(data)] if data[:2] == b"\x1f\x8b" else [data]
        except (zipfile.BadZipFile, OSError, RuntimeError, NotImplementedError, EOFError, zlib.error):
            pass
    return b"\n".join(parts).lower()


def label_leaks(groups: dict[str, list[Path]]) -> list[str]:
    """Labels written in the content of most of their own files, and clearly less often in the others' files.

    A heuristic, not a proof: it looks for the label's words in any case and inside longer words (plurals,
    "Cube001", "SphereMesh"), also inside gzip streams and zip archives, in the first few MB of each file. A label
    encoded any other way (a number standing for it, a colour code, a different language) goes unseen."""
    out: list[str] = []
    contents: dict[str, list[bytes]] = {label: [_leak_text(p) for p in files] for label, files in groups.items()}
    for label, texts in contents.items():
        if len(label.replace(" ", "")) < 3 or not texts:
            continue
        pattern = _label_pattern(label)
        own = sum(bool(pattern.search(t)) for t in texts) / len(texts)
        others = [t for other, ts in contents.items() if other != label for t in ts]
        rest = sum(bool(pattern.search(t)) for t in others) / len(others) if others else 0.0
        if own > 0.5 and own - rest >= 0.25:
            out.append(
                f"{label!r} is written in {round(own * len(texts))} of its {len(texts)} files "
                f"(and in {pct(rest)} of the other labels' files)"
            )
    return out


def generate_probes(
    design: Design,
    labels: list[str] | None = None,
    n_per_label: int = GENERATED_PER_LABEL,
    *,
    attempts: int = GENERATOR_ATTEMPTS,
    on_attempt: Callable[[int, AdapterError | None], None] | None = None,
    suffix: str = "",
) -> tuple[ProbeSet, str, int]:
    """Write, run and repair a generator; the probe set it made (provenance "generated"), its code, attempts.
    `suffix` (for example ".obj", the user's file's extension) is the extension the files must have; others are
    not used as probes."""
    labels = [label for label in (_clean_label(x) for x in (labels if labels is not None else design.labels)) if label]
    labels = list(dict.fromkeys(labels))
    if len({label.casefold() for label in labels}) < 2:
        raise ValueError(f"Generated probes need at least 2 distinct labels, not {labels!r}.")
    if n_per_label < MIN_PER_LABEL:
        raise ValueError(f"n_per_label is {n_per_label}; generate at least {MIN_PER_LABEL} files per label.")
    if attempts < 1:
        raise ValueError(f"attempts is {attempts}; give at least 1.")
    suffix = suffix.lower()
    least = max(MIN_PER_LABEL, n_per_label // 2)
    failures: list[dict] = []
    for k in range(1, attempts + 1):
        scratch = scratch_dir("sense-probes-")
        code: str | None = None
        try:
            try:
                code = write_generator(design, labels, n_per_label, earlier_attempts=failures, extension=suffix)
            except AdapterError as err:
                code = getattr(err, "code", None)
                raise
            out = scratch / "out"
            report = run_generator(code, out, labels, n_per_label)
            groups = {
                label: sorted(p for p in (out / label).iterdir() if p.is_file() and not p.name.startswith(".")
                              and (not suffix or p.suffix.lower() == suffix))
                if (out / label).is_dir() else []
                for label in labels
            }
            short = {label: len(files) for label, files in groups.items() if len(files) < least}
            if short:
                listed = ", ".join(f"{label!r}: {count}" for label, count in short.items())
                kind = f" {suffix}" if suffix else ""
                raise AdapterError(
                    f"generate wrote too few{kind} files ({listed}); write {n_per_label}{kind} files into "
                    f"os.path.join(out_dir, label) for every label.",
                    cause="output",
                )
            leaks = label_leaks(groups)
            if leaks:
                raise AdapterError(
                    "The generated files name their own label in their content, so a check of them would prove "
                    f"nothing: {'; '.join(leaks)}. Make the label a property of the content and never write it.",
                    cause="output",
                )
            sha = hashlib.sha256(code.encode("utf-8")).hexdigest()
            try:
                probes = adapters.probe_folder(
                    out,
                    name=f"generated-{design.modality}-{sha[:10]}",
                    question=design.question,
                    modality=design.modality,
                    provenance="generated",
                    source=(
                        f"files written by the agent's own generator code (sha256 {sha[:12]}), labelled as the "
                        f"generator built them; {GENERATED_CAVEAT}"
                    ),
                    licence="generated here",
                    min_per_label=least,
                    max_per_label=n_per_label,
                    suffixes=[suffix] if suffix else None,
                )
            except ValueError as exc:
                raise AdapterError(str(exc), cause="output") from None
            probes.meta.update(
                folder="(a temporary folder, removed after copying)",
                generator_sha256=sha,
                generator_attempts=k,
                generated_per_label=report["per_label"],
                generator_seconds=report["seconds"],
            )
            if on_attempt is not None:
                on_attempt(k, None)
            return probes, code, k
        except AdapterError as err:
            failures.append(_failure(k, err, code))
            if on_attempt is not None:
                on_attempt(k, err)
            if err.cause == "sandbox":
                break
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
    raise _gave_up("probe generator", failures)


GENERATED_CAVEAT = (
    "the generator and the adapter are written by the same LLM, so a pass on these files shows that the adapter "
    "and the base model recover what the generator built, a consistency check rather than independent ground truth"
)


# ----------------------------------------------------------------------------- deciding
def assess(request: FitRequest, card: ReportCard) -> tuple[bool, str]:
    """RulePlanner().assess, with an image card held to the text bars; then, for a card that passes them, two
    checks of its evidence: at least MIN_VERIFY_ITEMS probe files were scored, and the low end of the 95% Wilson
    interval of its accuracy lies above chance (on a handful of files, a lucky adapter clears the point bars)."""
    from sense.verify import _wilson

    planner = RulePlanner()
    if card.kind == "image":
        ok, sentence = planner.assess(request, dataclasses.replace(card, kind="text"))
        ok, sentence = ok, sentence.replace("for a text bridge", "for an image bridge")
    else:
        ok, sentence = planner.assess(request, card)
    if not ok:
        return ok, sentence
    if card.n < MIN_VERIFY_ITEMS:
        return False, (
            f"{card.bridge} failed: it scored {pct(card.accuracy)}, but on only {card.n} probe files, fewer than the "
            f"{MIN_VERIFY_ITEMS} needed to tell it from luck; give more labelled examples."
        )
    low, high = card.interval if card.interval else _wilson(card.accuracy, card.n)
    if low <= card.chance:
        return False, (
            f"{card.bridge} failed: it scored {pct(card.accuracy)} on {card.n} probe files, but the 95% interval of "
            f"that score ({pct(low)} to {pct(high)}) reaches down to the {pct(card.chance)} chance level, so it "
            "cannot be told from luck; give more labelled examples."
        )
    return ok, sentence


# ----------------------------------------------------------------------------- the run
def _jsonable(value: Any) -> Any:
    """Trace data as plain JSON values."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, bool)) or value is None:
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (bytes, bytearray)):
        return f"<{len(value)} bytes>"
    return str(value)


def _preview(output: dict) -> str:
    if "text" in output:
        text = output["text"]
        return text if len(text) <= PREVIEW_CHARS else text[: PREVIEW_CHARS - 3] + "..."
    from PIL import Image

    sizes = []
    for data in output.get("images", []):
        try:
            sizes.append("{}x{}".format(*Image.open(io.BytesIO(data)).size))
        except Exception:
            sizes.append("?")
    n = len(sizes)
    return _one_line(f"{n} image{'' if n == 1 else 's'}: {', '.join(sizes)}", PREVIEW_CHARS)


def _used_libraries(code: str, design_: Design, toolbox: list[str]) -> list[str]:
    """Toolbox entries the code imports or runs, else the design's libraries."""
    used = []
    for module in adapters.code_imports(code):
        entry = _match_library(module, toolbox)
        if entry is not None and entry not in used:
            used.append(entry)
    return used or list(design_.libraries)


def _wanted(design_: Design) -> dict[str, Any]:
    return {
        "labels": list(design_.labels),
        "label_hint": design_.label_hint,
        "format": design_.format,
        "min_per_label": MIN_PER_LABEL,
        "question": design_.question,
    }


def _ask_for_examples(design_: Design) -> str:
    example = f" (for example {', '.join(design_.labels[:4])})" if design_.labels else ""
    return (
        f"To verify a {design_.modality} adapter I need labelled examples: a folder with one sub-folder per label, "
        f"named after it ({design_.label_hint}{example}), each holding at least {MIN_PER_LABEL} {design_.format} "
        f"files, for the question \"{design_.question}\". Run again with that folder as the examples."
    )


def registry_loads(bridge: AdapterBridge) -> str | None:
    """None when sense.registry can load `bridge` back once installed, else why not. Chat loads every installed
    sense of a model through registry.get_bridges, so installing one it cannot load would break that model's
    chat; this saves the bridge to a temporary folder and loads it the way the registry does."""
    folder = Path(tempfile.mkdtemp(prefix="sense-loadcheck-"))
    try:
        bridge.save(folder)
        loaded = registry.load_bridge(folder)
        if getattr(loaded, "name", None) != bridge.name or getattr(loaded, "code", None) != bridge.code:
            shown = getattr(loaded, "name", type(loaded).__name__)
            return f"the registry loads it back as {shown}, not as this adapter"
        return None
    except Exception as exc:
        return f"the registry cannot load an adapter bridge yet ({type(exc).__name__}: {_one_line(str(exc), 200)})"
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def _earlier(model_name: str, modality: str) -> str | None:
    try:
        return next((row["bridge"] for row in registry.installed(model_name) if row["modality"] == modality), None)
    except Exception:
        return None


def _card_summary(card: ReportCard) -> str:
    names = {"swapped": "swapped input", "no_input": "no input"}
    controls = ", ".join(f"{names.get(k, k)} {pct(v)}" for k, v in card.controls.items())
    return (
        f"{card.bridge} scored {pct(card.accuracy)} on {card.n} held-out probe files (chance {pct(card.chance)})"
        + (f"; controls: {controls}." if controls else ".")
    )


def run_synth(
    model_name: str,
    file: Path,
    goal: str,
    emit: Callable[[TraceEvent], None],
    *,
    examples: Path | None = None,
    detection: dict | None = None,
) -> tuple[list[ReportCard], str | None]:
    """Design, write, verify and install (or refuse) an agent-written adapter for `file` (module docstring)."""
    t0 = time.monotonic()
    step = "design"

    def say(status: str, message: str, **facts: Any) -> None:
        emit(TraceEvent(step, status, message, _jsonable(facts), round(time.monotonic() - t0, 3)))

    def finish(message: str, *, installed: str | None = None, needs_user: bool = False, **facts: Any) -> None:
        nonlocal step
        step = "done"
        say("ok", message, installed=installed, needs_user=needs_user, **facts)

    try:
        file = Path(file)
        info = next((m for m in models.available_models() if m.name == model_name), None)
        if info is None:
            known = ", ".join(m.name for m in models.available_models())
            raise ValueError(f"Unknown model {model_name!r}; choose one of: {known}.")
        say("start", f"Designing an adapter that lets {info.label} sense {file.name}: {goal}", file=file.name, goal=goal)
        if detection is None:
            detection = detect(file).to_dict()
        elif hasattr(detection, "to_dict"):
            detection = detection.to_dict()
        head = file_head(file)
        box = adapters.toolbox()
        found = design(detection, goal, head, box, [info.to_dict()], name=file.name, size=file.stat().st_size)
        if found is None:
            say("fail", "The LLM gave no usable adapter design (none came back, or it failed the checks twice).",
                detection=detection)
            finish(f"No sense was installed for {file.name}: no usable adapter design came back from the LLM.")
            return [], None
        d = found
        say("ok", f"A {d.route} adapter for {d.modality} ({d.format}) files: {d.description}", **d.to_dict(),
            detection=detection, toolbox=box, head_preview=_one_line(head, PREVIEW_CHARS))

        step = "build"
        libs = ", ".join(d.libraries) or "the standard library"
        sandboxed = adapters.sandboxed()
        say("start", f"Writing adapt(path) with {libs} and running it on {file.name} in the sandbox.",
            libraries=d.libraries, sandboxed=sandboxed, max_attempts=ADAPTER_ATTEMPTS)
        refusal = adapters.sandbox_problem()
        if refusal is not None:
            say("fail", refusal, sandboxed=False)
            finish(f"No {d.modality} sense was installed: agent-written code cannot run safely here. {refusal}")
            return [], None
        if not sandboxed:
            say("info", "Warning: there is no sandbox here, so the code the agent writes runs with network and file "
                "access; only its environment is cleaned and its HTTP proxies blocked.", sandboxed=False,
                warning="unsandboxed")

        def adapter_attempt(k: int, err: AdapterError | None) -> None:
            if err is not None:
                more = " Asking the LLM for a repair." if k < ADAPTER_ATTEMPTS else ""
                say("info", f"Attempt {k} failed: {_one_line(err.reason, 300)}{more}", attempt=k, error=err.to_dict())

        try:
            code, output, used = write_adapter(d, file, attempts=ADAPTER_ATTEMPTS, on_attempt=adapter_attempt)
        except AdapterError as err:
            say("fail", err.reason, attempts=ADAPTER_ATTEMPTS, error=err.to_dict())
            finish(f"No {d.modality} sense was installed: no adapter the agent wrote ran on {file.name}.")
            return [], None
        code_sha = hashlib.sha256(code.encode("utf-8")).hexdigest()
        libraries = _used_libraries(code, d, box)
        kind_out = "text" if "text" in output else "images"
        say("ok", f"The adapter ran on {file.name} after {used} attempt{'' if used == 1 else 's'} "
            f"({output.get('seconds', 0):.1f} s).", attempts=used, libraries=libraries, code_sha256=code_sha,
            code=code, preview=_preview(output), output_kind=kind_out, notes=output.get("notes", []),
            seconds=output.get("seconds"))

        step = "data"
        generator_code: str | None = None
        suffix = file.suffix.lower() if re.fullmatch(r"\.[A-Za-z0-9]{1,10}", file.suffix) else ""
        if examples is not None:
            examples = Path(examples)
            say("start", f"Reading labelled probe files from {examples}.", provenance="user", source=str(examples))
            # The probe set is named by a hash: its name is shown to no adapter, but nothing in it should point
            # back at the examples folder, whose sub-folder names are the labels.
            hashed = hashlib.sha256(str(examples.expanduser().resolve()).encode("utf-8")).hexdigest()[:10]
            try:
                probes = adapters.probe_folder(
                    examples, name=f"{d.modality}-examples-{hashed}", question=d.question, modality=d.modality,
                    provenance="user", source=str(examples), licence="user supplied", min_per_label=MIN_PER_LABEL,
                    suffixes=[suffix] if suffix else None,
                )
            except ValueError as exc:
                say("fail", str(exc), provenance="user", reason=str(exc))
                finish(f"I could not use the examples in {examples}: {exc} {_ask_for_examples(d)}",
                       needs_user=True, wanted=_wanted(d))
                return [], None
        elif d.probe_plan == "generate":
            say("start", f"Writing a generator of {GENERATED_PER_LABEL} {d.format} files per label, each built to "
                f"have its label: {', '.join(d.labels)}.", provenance="generated", labels=d.labels,
                n_per_label=GENERATED_PER_LABEL, caveat=GENERATED_CAVEAT)

            def generator_attempt(k: int, err: AdapterError | None) -> None:
                if err is not None:
                    more = " Asking the LLM for a repair." if k < GENERATOR_ATTEMPTS else ""
                    say("info", f"Generator attempt {k} failed: {_one_line(err.reason, 300)}{more}", attempt=k,
                        error=err.to_dict(), provenance="generated")

            try:
                probes, generator_code, _ = generate_probes(
                    d, d.labels, GENERATED_PER_LABEL, attempts=GENERATOR_ATTEMPTS, on_attempt=generator_attempt,
                    suffix=suffix,
                )
            except AdapterError as err:
                say("fail", err.reason, provenance="generated", error=err.to_dict())
                finish(f"No {d.modality} sense was installed: no generated probe files. {_ask_for_examples(d)}",
                       needs_user=True, wanted=_wanted(d))
                return [], None
        else:
            why = {
                "user": "only the user can supply labelled examples for this goal",
                "dataset": "a public labelled dataset fits this goal, but fetching one is not wired into this run",
            }[d.probe_plan]
            say("info", f"No probe files: {why}. {d.probe_reason}".strip(), probe_plan=d.probe_plan,
                probe_reason=d.probe_reason, wanted=_wanted(d))
            finish(_ask_for_examples(d), needs_user=True, wanted=_wanted(d), probe_plan=d.probe_plan)
            return [], None
        summary = probes.to_dict()
        leaks = label_leaks({lab: [p for p, c in probes.items if c == k] for k, lab in enumerate(probes.labels)})
        leaks += [str(w) for w in probes.meta.get("warnings", [])]
        facts: dict[str, Any] = {
            "name": probes.name, "provenance": probes.provenance, "source": probes.source, "licence": probes.licence,
            "labels": list(probes.labels), "counts": summary["per_label"], "n": summary["n"], "split": summary["split"],
            "meta": probes.meta, "label_leaks": leaks,
        }
        if generator_code is not None:
            facts.update(generator_sha256=hashlib.sha256(generator_code.encode("utf-8")).hexdigest(),
                         generator_code=generator_code, caveat=GENERATED_CAVEAT)
        warning = f" Warning: {'; '.join(leaks)}." if leaks else ""
        # The adapter was written from the user's file before any probe file existed, so it was fitted on none of
        # them: every probe file is held out, and every one is scored (the 60/20/20 split is not needed here).
        scored = np.arange(len(probes.items), dtype=np.int64)
        say("ok", f"Using {summary['n']} {probes.provenance} probe files in {len(probes.labels)} labels "
            f"({', '.join(probes.labels)}); all are held out, since the adapter was written before they existed."
            f"{warning}", **facts)

        step = "verify"
        notes = []
        if leaks:
            notes.append(f"Warning about the probe files: {'; '.join(leaks)}.")
        if generator_code is not None:
            notes.append(f"The probe files were generated: {GENERATED_CAVEAT}.")
        probe_facts = {k: summary[k] for k in ("name", "provenance", "source", "licence", "labels", "per_label")}
        probe_facts["label_leaks"] = leaks
        if generator_code is not None:
            probe_facts["caveat"] = GENERATED_CAVEAT
        manifest = {
            "modality": d.modality, "format": d.format, "route": d.route, "description": d.description,
            "libraries": libraries, "written_by": os.environ.get("SENSE_LLM_MODEL") or llm.DEFAULT_MODEL,
            "attempts": used, "question": d.question, "goal": goal, "code_sha256": code_sha, "design": d.to_dict(),
            "probes": probe_facts,
        }
        if generator_code is not None:
            manifest["generator_sha256"] = facts["generator_sha256"]
        bridge = AdapterBridge(kind=d.kind, modality=d.modality, model_name=model_name, code=code,
                               manifest=_jsonable(manifest), timeout=ADAPTER_TIMEOUT)
        request = FitRequest(model=model_name, dataset=probes.name, modality=d.modality, scope="closed")
        model = models.load_model(model_name)
        say("start", f"Scoring {bridge.name} on {probes.provenance} probe files against swapped-input and "
            "no-input controls.", bridge=bridge.name, n_test=len(scored))
        started = time.monotonic()
        try:
            card = adapters.verify_files(model, bridge, probes, request, idx=scored)
        except AdapterError as err:
            say("fail", err.reason, bridge=bridge.name, error=err.to_dict())
            finish(f"No {d.modality} sense was installed: {err.reason}")
            return [], None
        seconds = round(time.monotonic() - started, 2)
        say("ok", _card_summary(card), card=card.to_dict(), seconds=seconds)

        step = "decide"
        say("start", f"Checking {bridge.name} against the bars for a {d.kind} bridge: chance plus 15 points, "
            f"10 points above swapped input, at least {MIN_VERIFY_ITEMS} probe files, and a 95% interval clear of "
            "chance.", bridge=bridge.name)
        ok, sentence = assess(request, card)
        card.verdict = "installed" if ok else "rejected"
        card.reason = " ".join(part for part in (sentence, card.reason, *notes) if part)
        say("ok", sentence, install=bridge.name if ok else None, accepted=ok, card=card.to_dict())

        step = "install"
        installed: str | None = None
        blocked = registry_loads(bridge) if ok else None
        if ok and blocked is None:
            earlier = _earlier(model_name, d.modality)
            replacing = f", replacing {earlier}" if earlier else ""
            say("start", f"Installing {bridge.name} for {info.label}{replacing}.", bridge=bridge.name, replaced=earlier)
            registry.install(bridge, card)
            directory = registry.sense_dir(model_name, d.modality)
            if generator_code is not None:
                (directory / "generator.py").write_text(generator_code, encoding="utf-8")
            installed = bridge.name
            say("ok", f"Installed {bridge.name}; {info.label} can now take {d.modality} files in chat.",
                bridge=bridge.name, replaced=earlier, directory=str(directory))
        elif ok:
            card.verdict = "candidate"
            card.reason += (f" Not installed: {blocked}; installing it would stop {info.label} from loading its "
                            "senses in chat.")
            say("info", f"Installed nothing: {bridge.name} passed its checks, but {blocked}, and installing it "
                f"would stop {info.label} from loading its senses in chat.", bridge=bridge.name, blocked=blocked,
                card=card.to_dict())
        else:
            say("info", f"Installed nothing; {bridge.name} did not pass its checks.", bridge=bridge.name)

        if installed:
            message = (f"{info.label} can now take {d.modality} files through {installed}: it scored "
                       f"{pct(card.accuracy)} on {card.n} probe files, against {pct(card.chance)} chance.")
        elif ok:
            message = f"No {d.modality} sense was installed for {info.label}: {sentence} But {blocked}."
        else:
            message = f"No {d.modality} sense was installed for {info.label}: {sentence}"
        finish(message, installed=installed, bridge=bridge.name)
        return [card], installed

    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
        failed = step
        say("fail", detail)
        step = "error"
        say("fail", f"The agent run stopped during the {failed} step. {detail}", step=failed, error=type(exc).__name__)
        raise
