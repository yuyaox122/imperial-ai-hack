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
      "generate"), probe_reason, install ([{name, why}], at most MAX_INSTALL (3) PyPI packages not in the toolbox
      that would make the adapter much better; usually []); .kind is the bridge kind ("text" or "image"); to_dict()
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
      (no "_", "-" or "/"), distinct, at most 8, and at least 2 for "generate"; install a list of {name, why}
      (missing or null means []), without names the toolbox or the standard library already has, network modules
      refused, distinct, at most MAX_INSTALL
  write_adapter(design, sample, *, attempts=3, timeout=60, on_attempt=None, hide=()) -> (code, output, attempts_used)
      asks the LLM for {"code"} defining adapt(path) (the prompt holds writer_design(design, hide): the design
      without its candidate labels, probe plan and install requests, and with every label of design.labels and
      `hide` masked in its question, label_hint and description; the toolbox, the output contract of
      sense.adapters.run_adapter, the rules: describe the content as measurements rather than a verdict, never
      guess a label from the file name, no network; and the sample's head), checks it before running (it parses, defines adapt, imports only the standard library and the
      toolbox, no network modules, no ctypes or cffi), runs it with sense.adapters.run_adapter on the sample and
      checks the output (the design's route; text not empty). Every failure goes back to the LLM for a repair, up
      to `attempts` in all (the reason cut to 1000 characters, the traceback to 1500). on_attempt(k,
      error_or_None) is called after each attempt. Raises sense.adapters.AdapterError naming every attempt's error
      (and with .attempts, one dict per attempt) when none works, at once when the sandbox cannot be applied.
      Causes beyond sense.adapters': "llm" (no code came back) and "code" (the code failed a check before it ran)
  writer_design(design, hide=()) -> dict
      what the adapter's author sees of a design: the fields in _ADAPTER_DESIGN_FIELDS, with each label of
      design.labels and `hide` (whole words, any case, separators ignored, a plural "s" or "es" included; a label of
      one or two characters in its own case only) replaced by "[answer]" in question, label_hint and description
      (a list of them reads "[answers]")
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
      agent.planner.RulePlanner().assess, which judges a text or image card by the text bars (accuracy >= chance +
      0.15 and >= swapped + 0.10), plus accuracy >= native + 0.10 when the card has a native control and
      unseen_categories >= its control_chance + 0.15 when it has that one; then at least MIN_VERIFY_ITEMS (10)
      scored probe files and the low end of the 95% Wilson interval above chance, so that a lucky adapter on a
      handful of files is not installed
  registry_loads(bridge) -> str | None
      None when sense.registry can load the adapter bridge back once installed (sense.bridges.load_bridge loads
      adapter bridges; this still catches one it cannot load), else why not
  run_synth(model_name, file, goal, emit, *, examples=None, detection=None, native=None, unseen=None, web=False,
            on_request=None, t0=None) -> (cards, installed)
      the whole run, as a trace of contracts.TraceEvent (the say() pattern of agent.auto; every event's data is
      JSON-serialisable; times count from t0, a time.monotonic() value, by default the run's start). `native` asks
      verify_files for the native control (None: when the user's file, or at least half of the probe files, reads
      as text the way verify_files gives a file as it is); `unseen` is a second labelled folder (one sub-folder per
      label, files of the user's file's extension, MIN_UNSEEN_PER_LABEL (3) per label) or ProbeSet, for the
      unseen-categories control. `web` lets the data step look for labelled files on the public web
      (agent.hub_probes.find_probes) when there is no examples folder, and for unseen categories when `unseen` is
      None (agent.auto turns it on). on_request(FitRequest(model, dataset=the probe set's name,
      modality=design.modality, scope "closed")) is called once the probe set is known, before verification
      (agent.jobs keeps it as the job's request, which sense.runs records).
        [design]  start; ok with the design's fields, detection, toolbox and head_preview (fail, then done, when
                  the LLM gives no usable design). The designer is shown the model with only the inputs chat can
                  give it as they are (sense.models.takes_as_is), so route "images" is offered only for a model
                  that takes images in chat, never for a local model. When agent.detect named the file's modality
                  video, 3d, molecule, geo or image, the design takes that modality (designed_modality keeps the
                  LLM's own name), so the sense is installed under the name chat looks it up by
        [install_request]  only when the design asks to install packages (and code may run here): start with
                  requested; agent.installer.plan checks them; a plan refused by the checks (always in fake mode,
                  where the installer reaches nothing) is not put to the user: fail with the verdict. Otherwise
                  agent.jobs.request_approval("install", {"plan": plan.to_dict(), "summary": plan.summary(),
                  "review": agent.installer.ask_llm_review(plan)}) asks the user; inside a job, agent.jobs adds its
                  own "approval" start and ok/fail events here, which carry the id the interface answers. Approved:
                  agent.installer.install(plan, approved=True), then ok with installed (the toolbox, and so the
                  adapter's path and toolbox(), now has them, and their top-level modules join design.libraries,
                  which the adapter's author is told to use); declined or unanswered: info; a failed install (any
                  error): fail. The run goes on in every case, with the toolbox as it then is. Each event's data holds
                  requested, verdict, summary, approved (None when not asked), installed and note
        [build]   start (sandboxed); fail, then done, when sense.adapters.sandbox_problem() refuses to run code;
                  info with warning "unsandboxed" when there is no sandbox to apply; info per failed attempt
                  (attempt, error); ok with attempts, libraries, code_sha256, code, preview (the first 300
                  characters of the output on the file, or the image sizes), output_kind, notes, seconds (fail,
                  then done, when no attempt works)
        [data]    probe files, from the first of these that gives some: a user examples folder when `examples` is
                  given (only files with the user's file's extension; MIN_PER_LABEL (5) per label; the set is named
                  <modality>-examples-<hash>); with `web`, a public dataset (_web_probes: find_probes(goal,
                  design.modality, <SENSE_HOME>/fetched/probes/<modality>), provenance "dataset", the set named
                  <modality>-dataset-<hash>, any extension the dataset has; each event find_probes emits is said
                  here with its own status and facts, action "search" | "candidate" | "inspect" | "fetch", plus
                  purpose "probes" and provenance "dataset"; files that probe_folder cannot use: fail with action
                  "build"); generated files when the design's probe_plan is "generate" (info per failed generator
                  attempt); else no probes: info (with web_reason when the web was searched), then done with
                  needs_user=True, saying which examples to provide (wanted: labels, label_hint, format,
                  min_per_label, question). An unusable examples folder or no working generator: fail, then the
                  same needs_user done. ok with name, provenance, source, licence, labels, counts, n, split, meta,
                  label_leaks (warnings: labels written in their own files' content, extensions that differ between
                  labels), for generated files generator_sha256, generator_code and caveat, and for a public dataset
                  fetched (find_probes' hub_id, url, title, kind, source, licence, licence_source, label_values,
                  counts, mapping, rows_seen, skipped, notes, tried). Then the unseen categories: with `unseen`,
                  info with unseen (name, source, labels, per_label, n), or info with the reason when the folder
                  cannot be used; with `web` and no `unseen`, a second find_probes into
                  <SENSE_HOME>/fetched/probes/<modality>-unseen with exclude_sources the first dataset's source and
                  disjoint_from its labels (the examples' or generated labels when the probes came from there),
                  its events said with purpose "unseen", then the same info with unseen, or info "Unseen categories
                  not measured: <why>." with purpose "unseen" and reason. Whenever unseen categories are not
                  measured for such a reason, the card's reason says so too
        [verify]  sense.adapters.verify_files on every probe file (the adapter was written before any existed, so
                  all are held out; an API model scores at most config.api_verify_items() of them), with native and
                  the unseen set: start with native and unseen; ok with the card dict ("card"), its controls and
                  control_chance, and seconds (fail, then done, when the adapter fails on more than half)
        [decide]  assess() on the card (FitRequest: dataset = the probe set's name, modality = design.modality,
                  scope "closed"); the card's verdict becomes "installed" or "rejected", its reason the sentence
                  followed by the adapter's failures, the label-leak warnings and, for generated probes, the caveat
        [install] when assess passes and registry_loads(bridge) is None: ok after sense.registry.install(bridge,
                  card) (the generator's code is saved beside the bridge as generator.py); when assess passes but
                  the registry cannot load it back: info (blocked), nothing installed, verdict "candidate"
                  (installing would break the model's chat); info "Installed nothing" otherwise
        [done]    ok: installed (Bridge.name or None) and needs_user
        [error]   fail: step, error, after a fail event at the step that raised; the exception is re-raised
      The bridge is a sense.adapters.AdapterBridge; its code (adapter.py) and manifest (bridge.json: modality,
      format, route, description, libraries, written_by, attempts, question, goal, code_sha256, design, probes
      with label_leaks and, for generated probes, caveat, and for a public dataset fetched; detected, what
      agent.detect said about the user's file ({modality, format, suffix}: server/app.py finds the sense for a file
      agent.detect calls "other" by it); install, the install_request record, when the design asked for packages;
      unseen (name, source, labels, per_label) when that control ran) are saved with it.
      Returns ([card], installed name or None); ([], None) when the run stopped before a card.

Labels never reach an adapter through a file name or a folder: the adapter is written from the user's file alone,
before any probe exists, and never sees the candidate labels, the probe plan, the install requests or anything of
the probe or unseen sets (its request holds only writer_design: the design's modality, format, route,
description, libraries, question and label_hint, with the design's labels and the sub-folder names of the
examples and unseen folders masked, plus the toolbox, the contract, the rules and the sample's head); every probe
file is copied to <sha256[:16]><extension>
by sense.adapters.probe_folder, and sense.adapters.run_adapter hands the adapter a private copy named input<ext> in
a folder of its own, under a macOS sandbox that cannot read the probe store, the examples folder or anything else
in the user's folders (without that sandbox, only the copy and the anonymous names stand).
"""

from __future__ import annotations

import ast
import codecs
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
from sense import adapters, config, models, registry
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
MAX_INSTALL = 3  # packages a design may ask to install into the toolbox
MIN_UNSEEN_PER_LABEL = 3  # files per label in an unseen-categories folder (verify_files wants 4 files in all)
INSTALL_APPROVAL_TIMEOUT = 600.0  # seconds the user has to answer an install request
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
_MASKED_FIELDS = ("question", "label_hint", "description")  # free text where a design may name its labels
_MASK = "[answer]"
# agent.detect's names for the modalities adapters are written for: an adapter for such a file is installed under
# that name, which is the one chat looks a sense up by.
_DETECTED_MODALITIES = ("video", "3d", "molecule", "geo", "image")
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
        "probe_plan", "probe_reason", "install",
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
        "install": {
            "type": "array",
            "description": f"at most {MAX_INSTALL} PyPI packages not in the toolbox that would make the adapter much "
                           "better; [] when the toolbox is enough",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["name", "why"],
                "properties": {"name": {"type": "string"}, "why": {"type": "string"}},
            },
        },
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
    "probe_plan is generate). The adapter's author sees description, question and label_hint but must not see "
    "the candidate answers, so never name them there. Then probe_plan and probe_reason. probe_plan says where "
    "labelled probe files come from; their labels must not come from the adapter: \"generate\" only when a short "
    "script using the toolbox can build the labelled property into each file's content, so that the label is a "
    "fact of construction that the "
    "script itself computes (meshes built as known shapes, videos made with a known motion or colour, numbers in "
    "known ranges, points inside boxes the script defines); never when the label is knowledge about the content "
    "that the script would have to recall (a molecule's taste or smell, the country or city a coordinate lies "
    "in, a species, a style): such labels need \"dataset\" (a public labelled dataset fits the goal) or \"user\" "
    "(only the user can supply labelled examples). Prefer generate when such a script is possible. probe_reason "
    "says why, in one sentence. install lists at most 3 PyPI packages that are not in the toolbox and would make "
    "the adapter much better than the toolbox allows (name and why); [] when the toolbox is enough, which is the "
    "usual case. They are checked and the user is asked to approve them; the adapter is written with them only "
    "when approved, so never name them in libraries. When problems_with_last_answer is present, your last answer "
    "had those problems: fix them."
)

ADAPTER_INSTRUCTION = (
    "You write the adapter of a design: Python code that defines adapt(path). It runs in a sandbox with no "
    "network, once per file, and what it returns is all the base model will know about the file. Follow "
    "contract and rules. Use only the Python standard library and the design's libraries, which come from "
    "toolbox (a name ending in \"(via subprocess)\" is a program to run with subprocess); adapter_kit, the "
    "harness's tested helpers listed there, may always be imported and is often the shortest way. sample shows the file "
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
    install: list[dict] = field(default_factory=list)  # [{"name", "why"}]: packages to ask the user to approve

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

    install, install_problems = _clean_install(answer.get("install"), toolbox)
    problems += install_problems

    if problems:
        return None, problems
    return Design(
        modality=modality, format=fmt, route=str(route), description=description, libraries=libraries,
        question=question, label_hint=label_hint, labels=labels, probe_plan=str(plan), probe_reason=reason,
        install=install,
    ), []


def _clean_install(raw: Any, toolbox: list[str]) -> tuple[list[dict], list[str]]:
    """A design's install requests: [{name, why}], at most MAX_INSTALL, without what the toolbox or the standard
    library already has; and the problems found. A missing field means none (an older answer)."""
    if raw is None:
        return [], []
    if not isinstance(raw, list):
        return [], ["install must be a list of {name, why} (or [])"]
    out: list[dict] = []
    problems: list[str] = []
    for item in raw:
        name = " ".join(item.get("name").split()) if isinstance(item, dict) and isinstance(item.get("name"), str) else ""
        if not name:
            problems.append("each install entry needs a package name and why it helps")
            continue
        top = name.split("[")[0].split(".")[0].replace("-", "_").lower()
        if _match_library(name, toolbox) is not None or top in {m.lower() for m in _STDLIB}:
            continue  # already there: nothing to install
        if top in _NETWORK_MODULES:
            problems.append(f"install {name!r} is for the network, which the sandbox blocks")
            continue
        if name.casefold() not in {r["name"].casefold() for r in out}:
            out.append({"name": name, "why": _clean_text(item.get("why"), 300)})
    return out[:MAX_INSTALL], problems


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
def _label_words(label: str) -> re.Pattern | None:
    """A label as a whole word or phrase in free text: any case, any separators between its words, a plural "s" or
    "es"; a label of one or two characters ("A", "pH") in its own case only, so that it does not take every "a"."""
    words = re.findall(r"[^\W_]+", str(label))
    if not words:
        return None
    body = r"[\W_]*".join(re.escape(w) for w in words)
    short = len("".join(words)) <= 2
    return re.compile(rf"(?<![^\W_]){body}(?:e?s)?(?![^\W_])", 0 if short else re.IGNORECASE)


def writer_design(design: Design, hide: list[str] | tuple[str, ...] = ()) -> dict[str, Any]:
    """What the adapter's author sees of `design` (module docstring): no candidate labels, probe plan or install
    requests, and the labels of design.labels and `hide` masked in the free text that might name them."""
    shown = {key: value for key, value in design.to_dict().items() if key in _ADAPTER_DESIGN_FIELDS}
    labels = sorted({str(x) for x in [*design.labels, *hide] if str(x).strip()}, key=len, reverse=True)
    patterns = [p for p in map(_label_words, labels) if p is not None]
    for key in _MASKED_FIELDS:
        text = str(shown.get(key, ""))
        for pattern in patterns:
            text = pattern.sub(_MASK, text)
        # "[answer], [answer] or [answer]" reads better as one plural
        shown[key] = re.sub(r"\[answer\](?:\s*[,;/]?\s*(?:(?:or|and)\s+)?\[answer\])+", "[answers]", text)
    return shown


def write_adapter(
    design: Design,
    sample: Path,
    *,
    attempts: int = ADAPTER_ATTEMPTS,
    timeout: float = ADAPTER_TIMEOUT,
    on_attempt: Callable[[int, AdapterError | None], None] | None = None,
    hide: list[str] | tuple[str, ...] = (),
) -> tuple[str, dict, int]:
    """Ask for adapt(path), run it on `sample`, repair on errors; (code, run_adapter output, attempts used).
    `hide` lists more labels to mask in what the author sees of the design (writer_design)."""
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
        "design": writer_design(design, hide),
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
    """RulePlanner().assess, which holds a text or image card to the text bars and to its native and
    unseen-categories controls when it has them; then, for a card that passes, two checks of its evidence: at
    least MIN_VERIFY_ITEMS probe files were scored, and the low end of the 95% Wilson interval of its accuracy
    lies above chance (on a handful of files, a lucky adapter clears the point bars)."""
    from sense.verify import _wilson

    ok, sentence = RulePlanner().assess(request, card)
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
        return f"the registry cannot load the adapter bridge back ({type(exc).__name__}: {_one_line(str(exc), 200)})"
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def _earlier(model_name: str, modality: str) -> str | None:
    try:
        return next((row["bridge"] for row in registry.installed(model_name) if row["modality"] == modality), None)
    except Exception:
        return None


def _request_install(requested: list[dict], say: Callable[..., None]) -> dict[str, Any]:
    """Check the packages a design asks for, ask the user (agent.jobs.request_approval) unless the checks refuse
    them, and install them when approved. Every outcome is said at the install_request step; the record returned
    goes into the manifest: requested, verdict, summary, approved (None when not asked), installed, note."""
    from agent import installer, jobs

    names = ", ".join(r["name"] for r in requested)
    them = "it" if len(requested) == 1 else "them"
    record: dict[str, Any] = {
        "requested": requested, "verdict": None, "summary": None, "approved": None, "installed": [], "note": "",
    }
    say("start", f"The design asks for {names}, which the toolbox lacks: checking {them} before asking you.",
        requested=requested)
    try:
        plan = installer.plan([{"name": r["name"], "why": r["why"]} for r in requested])
    except Exception as exc:
        record["note"] = f"the checks could not run ({type(exc).__name__}: {_one_line(str(exc), 300)})"
        say("fail", f"Could not check {names}: {record['note']}. Writing the adapter without {them}.", **record)
        return record
    record.update(verdict=plan.verdict, summary=plan.summary())
    if plan.verdict == "refuse":
        record["note"] = "refused by the checks, so you were not asked"
        say("fail", f"{plan.summary()} Writing the adapter without {them}.", plan=plan.to_dict(), **record)
        return record
    review = installer.ask_llm_review(plan)
    approved = jobs.request_approval(
        "install", {"plan": plan.to_dict(), "summary": plan.summary(), "review": review},
        timeout=INSTALL_APPROVAL_TIMEOUT,
    )
    record["approved"] = approved
    if not approved:
        record["note"] = "declined or not answered"
        say("info", f"Not installing {names}: the request was declined or not answered. Writing the adapter with "
            "the toolbox as it is.", **record)
        return record
    try:
        result = installer.install(plan, approved=True)
    except Exception as exc:  # refused, failed, or anything else (a folder that cannot be made): the run goes on
        shown = _one_line(str(exc), 300) or type(exc).__name__
        if not isinstance(exc, (installer.InstallRefused, installer.InstallFailed)):
            shown = f"{type(exc).__name__}: {shown}"
        record["note"] = f"approved, but the install failed: {shown}"
        say("fail", f"{names} could not be installed ({shown}). Writing the adapter without {them}.",
            log_tail=getattr(exc, "log_tail", ""), **record)
        return record
    record["installed"] = list(result.get("installed") or [])
    shown = ", ".join(f"{p.get('name')} {p.get('version')}" for p in record["installed"]) or "nothing new"
    say("ok", f"Installed {shown} into the toolbox; the adapter may import {them}.", seconds=result.get("seconds"),
        **record)
    return record


def _reads_as_text(path: Path) -> bool:
    """Whether verify_files' native control could give the model this file as it is (sense.adapters._native_text:
    UTF-8 text, a few bytes that do not decode tolerated)."""
    try:
        return adapters._native_text(Path(path)) is not None
    except OSError:
        return False


def _mostly_text(paths: list[Path]) -> bool:
    """Whether at least half of the files read as text (_reads_as_text), as verify_files needs to measure native."""
    paths = list(paths)
    return bool(paths) and 2 * sum(_reads_as_text(p) for p in paths) >= len(paths)


def _folder_labels(folder: Path | ProbeSet | None) -> list[str]:
    """The labels a labelled folder (its sub-folder names, read as sense.adapters.probe_folder reads them) or a
    ProbeSet holds, to mask in what the adapter's author sees; [] when there are none or it cannot be read."""
    if folder is None:
        return []
    if isinstance(folder, ProbeSet):
        return list(folder.labels)
    try:
        return [
            " ".join(d.name.replace("_", " ").replace("-", " ").split())
            for d in sorted(Path(folder).expanduser().iterdir())
            if d.is_dir() and not d.name.startswith((".", "_"))
        ]
    except OSError:
        return []


def _unseen_probes(
    unseen: Path | ProbeSet | None, design_: Design, suffix: str, say: Callable[..., None]
) -> tuple[ProbeSet | None, str]:
    """The unseen-categories probe set (a ProbeSet as given, or one read from a folder with one sub-folder per
    label, files of the user's file's extension), said at the data step; and a note for the card when a folder
    cannot be used."""
    if unseen is None:
        return None, ""
    if isinstance(unseen, ProbeSet):
        found = unseen
    else:
        folder = Path(unseen)
        hashed = hashlib.sha256(str(folder.expanduser().resolve()).encode("utf-8")).hexdigest()[:10]
        try:
            found = adapters.probe_folder(
                folder, name=f"{design_.modality}-unseen-{hashed}", question=design_.question,
                modality=design_.modality, provenance="user", source=str(folder), licence="user supplied",
                min_per_label=MIN_UNSEEN_PER_LABEL, suffixes=[suffix] if suffix else None,
            )
        except ValueError as exc:
            say("info", f"The folder of unseen categories cannot be used: {exc}", unseen=str(folder), reason=str(exc))
            return None, f"Unseen categories not measured: {exc}"
    shown = found.to_dict()
    say("info", f"Also scoring {shown['n']} files of {len(found.labels)} other categories ({', '.join(found.labels)}), "
        "which the adapter was not built around.", unseen={k: shown[k] for k in ("name", "source", "labels",
                                                                                    "per_label", "n")})
    return found, ""


_FETCHED_KEYS = ("hub_id", "url", "title", "kind", "source", "licence", "licence_source", "label_values", "counts",
                 "mapping", "rows_seen", "skipped", "notes", "tried")


def _fetched_facts(info: dict) -> dict[str, Any]:
    """What the trace and the manifest keep of agent.hub_probes.find_probes' info: where the files came from."""
    return {k: info[k] for k in _FETCHED_KEYS if k in info}


def _web_message(event: dict, unseen: bool) -> str:
    """One sentence for an event agent.hub_probes.find_probes emits ({"action", "status", ...})."""
    what = "labelled files of other categories" if unseen else "labelled probe files"
    action, status = event.get("action"), event.get("status")
    source = event.get("source") or event.get("hub_id") or "a candidate"
    reason = _one_line(str(event.get("reason") or ""), 300)
    if action == "search":
        if status == "start":
            return f"Searching the Hugging Face Hub and the web for {what}."
        if status == "ok":
            n = event.get("found") or 0
            return f"Found {n} candidate dataset{'' if n == 1 else 's'} for {what}."
        return f"The search for {what} failed: {reason}"
    if action == "candidate":
        if status == "start":
            return f"Trying {source}."
        if status == "ok":
            labels = [str(label) for label in event.get("labels") or []]
            return f"Using {source}: {len(labels)} labels ({_one_line(', '.join(labels), 200)})."
        return f"Could not use {source}: {reason}"
    if action == "inspect":
        if status == "start":
            return f"Reading the columns of {source}."
        if status == "ok":
            mapping = event.get("mapping") or {}
            inputs = ", ".join(str(c) for c in mapping.get("input_columns") or []) or "?"
            return f"{source} fits: input {inputs}, label {mapping.get('label_column') or 'its 0/1 columns'}."
        return f"{source} does not fit: {reason}"
    if action == "fetch":
        if status == "start":
            return f"Fetching labelled files from {source}."
        if status == "ok":
            counts = event.get("counts") or {}
            total = sum(v for v in counts.values() if isinstance(v, int))
            return f"Fetched {total} files in {len(counts)} labels from {source}."
        return f"Fetching from {source} failed: {reason}"
    return f"{action or 'search'} {status or ''}: {reason}".strip(" :")


def _web_probes(
    goal: str, design_: Design, say: Callable[..., None], *, purpose: str, exclude_sources: list[str] | tuple = (),
    disjoint_from: list[str] | tuple = (),
) -> tuple[ProbeSet | None, dict | None, str]:
    """Labelled files of the design's modality from a public dataset: agent.hub_probes.find_probes(goal,
    design.modality, <SENSE_HOME>/fetched/probes/<modality>[-unseen]), read with sense.adapters.probe_folder as a
    ProbeSet of provenance "dataset" (named <modality>-dataset-<hash>, or <modality>-unseen-<hash> for purpose
    "unseen"). Every event find_probes emits is said at the data step with its own facts (action, ...) plus purpose
    ("probes" | "unseen") and provenance "dataset". Returns (probe set, find_probes' info, "") or (None, info or
    None, why there is none)."""
    from agent import hub_probes

    unseen = purpose == "unseen"
    root = config.home() / "fetched" / "probes" / (f"{design_.modality}-unseen" if unseen else design_.modality)

    def forward(event: dict) -> None:
        status = event.get("status")
        facts = {k: v for k, v in event.items() if k != "status"}
        say(status if status in ("start", "ok", "fail") else "info", _web_message(event, unseen), purpose=purpose,
            provenance="dataset", **facts)

    options = {"min_per_label": MIN_UNSEEN_PER_LABEL} if unseen else {}
    try:
        folder, info = hub_probes.find_probes(goal, design_.modality, root, emit=forward,
                                              exclude_sources=list(exclude_sources),
                                              disjoint_from=list(disjoint_from), **options)
    except hub_probes.ProbesUnavailable as exc:
        return None, None, str(exc) or type(exc).__name__
    source = str(info.get("source") or info.get("url") or "")
    hashed = hashlib.sha256(source.encode("utf-8")).hexdigest()[:10]
    try:
        found = adapters.probe_folder(
            folder, name=f"{design_.modality}-{'unseen' if unseen else 'dataset'}-{hashed}", question=design_.question,
            modality=design_.modality, provenance="dataset", source=str(info.get("url") or source),
            licence=str(info.get("licence") or "unknown"), min_per_label=MIN_UNSEEN_PER_LABEL if unseen else MIN_PER_LABEL,
        )
    except ValueError as exc:
        why = f"the files fetched from {source} cannot be used: {exc}"
        say("fail", why[:1].upper() + why[1:], action="build", purpose=purpose, provenance="dataset", reason=str(exc))
        return None, info, why
    return found, info, ""


def _card_summary(card: ReportCard) -> str:
    names = {"swapped": "swapped input", "no_input": "no input", "native": "the raw file",
             "unseen_categories": "unseen categories"}
    own = getattr(card, "control_chance", None) or {}
    controls = ", ".join(
        f"{names.get(k, k)} {pct(v)}" + (f" (chance {pct(own[k])})" if k in own else "") for k, v in card.controls.items()
    )
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
    native: bool | None = None,
    unseen: Path | ProbeSet | None = None,
    web: bool = False,
    on_request: Callable[[FitRequest], None] | None = None,
    t0: float | None = None,
) -> tuple[list[ReportCard], str | None]:
    """Design, write, verify and install (or refuse) an agent-written adapter for `file` (module docstring)."""
    t0 = time.monotonic() if t0 is None else t0
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
        # the designer may choose route "images" only for a model chat can show images to (models.takes_as_is): a
        # local model's ModelInfo.native lists "image", but chat gives it text and vector inputs only
        seen = info.to_dict() | {"native": [m for m in info.native if models.takes_as_is(info, m)]}
        found = design(detection, goal, head, box, [seen], name=file.name, size=file.stat().st_size)
        if found is None:
            say("fail", "The LLM gave no usable adapter design (none came back, or it failed the checks twice).",
                detection=detection)
            finish(f"No sense was installed for {file.name}: no usable adapter design came back from the LLM.")
            return [], None
        d = found
        renamed: dict[str, str] = {}
        detected = detection.get("modality") if isinstance(detection, dict) else None
        if detected in _DETECTED_MODALITIES and d.modality != detected:
            # chat finds an installed sense by agent.detect's modality name, not by the designer's ("3d-model")
            renamed["designed_modality"] = d.modality
            d.modality = detected
        say("ok", f"A {d.route} adapter for {d.modality} ({d.format}) files: {d.description}", **d.to_dict(),
            **renamed, detection=detection, toolbox=box, head_preview=_one_line(head, PREVIEW_CHARS))

        install_record: dict[str, Any] | None = None
        if d.install and adapters.sandbox_problem() is None:  # where no code may run, installing helps nothing
            step = "install_request"
            install_record = _request_install(d.install, say)
            before, box = box, adapters.toolbox()  # an approved install adds its modules
            if install_record.get("installed"):  # the author is told to use the design's libraries: these too
                d.libraries = d.libraries + [e for e in box if e not in before and e not in d.libraries]

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
            code, output, used = write_adapter(d, file, attempts=ADAPTER_ATTEMPTS, on_attempt=adapter_attempt,
                                               hide=_folder_labels(examples) + _folder_labels(unseen))
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
        fetched: dict | None = None  # agent.hub_probes.find_probes' info for probe files found on the web
        web_reason = ""  # why the web gave no probe files
        probes: ProbeSet | None = None
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
        elif web:  # labelled files from a public dataset come first: their labels are independent of the agent
            probes, fetched, web_reason = _web_probes(goal, d, say, purpose="probes")
        if probes is None and d.probe_plan == "generate":
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
        elif probes is None:
            if web:
                why = f"the search for labelled {d.format} files found none ({_one_line(web_reason, 300)})"
                if d.probe_plan == "user":
                    why += ", and only the user can supply labelled examples for this goal"
            else:
                why = {
                    "user": "only the user can supply labelled examples for this goal",
                    "dataset": "a public labelled dataset fits this goal, but fetching one is not wired into this run",
                }[d.probe_plan]
            web_facts = {"web_reason": web_reason} if web else {}
            say("info", f"No probe files: {why}. {d.probe_reason}".strip(), probe_plan=d.probe_plan,
                probe_reason=d.probe_reason, wanted=_wanted(d), **web_facts)
            searched = "I found no labelled examples on the web. " if web else ""
            finish(searched + _ask_for_examples(d), needs_user=True, wanted=_wanted(d), probe_plan=d.probe_plan,
                   **web_facts)
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
        if fetched is not None:
            facts["fetched"] = _fetched_facts(fetched)
        warning = f" Warning: {'; '.join(leaks)}." if leaks else ""
        # The adapter was written from the user's file before any probe file existed, so it was fitted on none of
        # them: every probe file is held out, and every one is scored (the 60/20/20 split is not needed here).
        scored = np.arange(len(probes.items), dtype=np.int64)
        say("ok", f"Using {summary['n']} {probes.provenance} probe files in {len(probes.labels)} labels "
            f"({', '.join(probes.labels)}); all are held out, since the adapter was written before they existed."
            f"{warning}", **facts)
        unseen_set, unseen_note = _unseen_probes(unseen, d, suffix, say)
        if unseen is None and web:  # categories the adapter was not built around, from another public source
            exclude = [fetched["source"]] if fetched and fetched.get("source") else []
            disjoint = list(dict.fromkeys([*((fetched or {}).get("label_values") or {}).values(), *probes.labels]))
            found_unseen, _, why = _web_probes(goal, d, say, purpose="unseen", exclude_sources=exclude,
                                               disjoint_from=disjoint)
            if found_unseen is not None:
                unseen_set, unseen_note = _unseen_probes(found_unseen, d, suffix, say)
            else:
                unseen_note = f"Unseen categories not measured: {_one_line(why, 300)}."
                say("info", unseen_note, purpose="unseen", reason=why)

        step = "verify"
        notes = []
        if leaks:
            notes.append(f"Warning about the probe files: {'; '.join(leaks)}.")
        if unseen_note:
            notes.append(unseen_note)
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
            # how chat recognises a file for this sense when agent.detect calls it "other" (csv, fasta, unknown)
            "detected": {"modality": detection.get("modality"), "format": detection.get("format"), "suffix": suffix},
        }
        if generator_code is not None:
            manifest["generator_sha256"] = facts["generator_sha256"]
        if fetched is not None:
            probe_facts["fetched"] = facts["fetched"]
        if install_record is not None:
            manifest["install"] = install_record
        if unseen_set is not None:
            manifest["unseen"] = {k: unseen_set.to_dict()[k] for k in ("name", "source", "labels", "per_label")}
        bridge = AdapterBridge(kind=d.kind, modality=d.modality, model_name=model_name, code=code,
                               manifest=_jsonable(manifest), timeout=ADAPTER_TIMEOUT)
        request = FitRequest(model=model_name, dataset=probes.name, modality=d.modality, scope="closed")
        if on_request is not None:  # agent.jobs keeps it as the job's request, which sense.runs records
            on_request(request)
        model = models.load_model(model_name)
        if native is None:  # the native control needs files the model can be given as they are
            native = _reads_as_text(file) or _mostly_text(probes.files)
        extra = (", the model reading each file as it is" if native else "") + (
            f", and {len(unseen_set.items)} files of categories it was not built around" if unseen_set else "")
        say("start", f"Scoring {bridge.name} on {probes.provenance} probe files against swapped-input and "
            f"no-input controls{extra}.", bridge=bridge.name, n_test=len(scored), native=native,
            unseen=unseen_set.name if unseen_set else None)
        started = time.monotonic()
        try:
            card = adapters.verify_files(model, bridge, probes, request, idx=scored, native=native, unseen=unseen_set)
        except AdapterError as err:
            say("fail", err.reason, bridge=bridge.name, error=err.to_dict())
            finish(f"No {d.modality} sense was installed: {err.reason}")
            return [], None
        seconds = round(time.monotonic() - started, 2)
        say("ok", _card_summary(card), card=card.to_dict(), controls=card.controls,
            control_chance=card.control_chance, seconds=seconds)

        step = "decide"
        bars = ""
        if "native" in card.controls:
            bars += ", 10 points above the model reading the raw file"
        if "unseen_categories" in card.controls:
            bars += ", 15 points above chance on the unseen categories"
        say("start", f"Checking {bridge.name} against the bars for {'an' if d.kind == 'image' else 'a'} {d.kind} "
            f"bridge: chance plus 15 points, 10 points above swapped input{bars}, at least {MIN_VERIFY_ITEMS} probe "
            "files, and a 95% interval clear of chance.", bridge=bridge.name)
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
