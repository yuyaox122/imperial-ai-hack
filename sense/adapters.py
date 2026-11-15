"""Agent-written adapters: the harness side. An LLM writes Python code `adapt(path)` that turns a file of any
modality into text (any base model reads it) or into images (for models that read images); this module runs that
code in a separate, locked-down process, wraps it as a bridge, and verifies the bridge on labelled probe files.

Public surface (do not rename):
  toolbox() -> list[str]
      what an adapter may use, checked at call time: the allow-list numpy, scipy, PIL, json, csv, math, struct,
      zipfile, gzip, io, re, xml.etree, statistics, plus trimesh, rdkit, soundfile, av, cv2, pandas and
      matplotlib when they are installed, plus "ffmpeg (via subprocess)" and "ffprobe (via subprocess)" when those
      binaries are on PATH (or in /opt/homebrew/bin or /usr/local/bin), plus "adapter_kit (...)" (sense.adapter_kit,
      always on the adapter's path; the parentheses list its functions in one line, KIT_SUMMARY), plus the
      importable top-level names of the packages agent.installer recorded in the toolbox's installed.json (only
      those whose files are in the toolbox). It is for the prompt that asks for an adapter; run_adapter does not
      police imports.
  KIT_SUMMARY
      the one-line description of adapter_kit that toolbox() shows
  matplotlib_cache(*, build=True) -> Path | None
      <config.home()>/cache/matplotlib holding matplotlib's font cache (fontlist-*.json), built once by a harness
      subprocess (MPLCONFIGDIR set there, `import matplotlib.font_manager`) when it is missing and `build` is
      true; None when matplotlib is not installed or the build failed (a failure is remembered for the process)
  code_imports(code) -> list[str]
      the top-level modules the code imports (and "ffmpeg"/"ffprobe" when it names them), for a manifest
  AdapterError(reason, traceback="", *, cause="error")
      .reason (one line: what went wrong, at most REASON_CHARS), .traceback (the tail of the traceback, or of the
      process's error output when there is none, at most TRACEBACK_CHARS), .cause ("exception", "output",
      "timeout", "exit", "limit", "sandbox" or "verify"); str() gives both texts
  sandboxed() -> bool
      whether adapters run under macOS sandbox-exec here (the binary exists and can apply a profile; a working
      sandbox is remembered, a failed check is tried again on the next call)
  sandbox_problem() -> str | None
      why run_adapter would refuse to run code here: on macOS it refuses whenever the sandbox cannot be applied
      (for example inside another sandbox), unless SENSE_ADAPTER_UNSANDBOXED=1; None when it would run
  redact(text) -> str
      text with the values of secret-looking environment variables and key-shaped strings replaced
  run_adapter(code, path, *, timeout=60, writable=None) -> dict
      runs adapt(<a private copy of path>) in a separate Python process and returns {"text": str} or
      {"images": [PNG bytes, ...]}, each with "notes" (what the caps cut) and "seconds"; `writable` is one more
      folder the code may read and write (a probe generator's output folder)
  AdapterBridge(*, kind, modality, model_name, code, manifest=None, slug=None, timeout=60)
      a contracts.Bridge for files: kind "text" or "image", name "adapter:<slug>"; prepare(path) runs the adapter
      (results cached per code and file sha256 within the process; so are the adapter's own exceptions and broken
      output, but not crashes, kills, timeouts, limits or a missing sandbox, which may not recur) and returns a
      SenseInput: kind "text" with the adapter's text, or kind "image" with one PNG contact sheet of its images;
      run(path) gives the cached raw result; from_embedding raises ValueError; save(directory) writes bridge.json
      ({"kind", "name", "modality", "model_name", "adapter": true, "manifest", "timeout"}) and adapter.py. The
      manifest holds modality, format, route, description, libraries, written_by, created and attempts (defaults
      filled in: route = kind, libraries = code_imports(code), created = now, attempts = 1)
  load_adapter_bridge(directory) -> AdapterBridge
  contact_sheet(images) -> bytes
      PNG images tiled in a grid of up to 3 columns, at most 1024 px wide (one image comes back unchanged)
  ProbeSet
      labelled probe files: name, title, modality, question, labels, items [(path, label index)], fit_idx, dev_idx,
      test_idx, source, licence, provenance ("user" | "dataset" | "generated"), meta; .y, .files and to_dict()
  probe_folder(path, *, name=None, question, modality, provenance="user", source="", licence="",
               min_per_label=4, max_per_label=40, seed=0, suffixes=None) -> ProbeSet
  verify_files(model, bridge, probes, request, *, max_items=None, seed=0, idx=None, native=False, unseen=None,
               unseen_max_items=None) -> contracts.ReportCard

The adapter contract. The code must define adapt(path), called once per file with the absolute path of a private
copy of the file as a str (<run folder>/input/input<extension>: the original's name and folder are never shown),
returning {"text": str} or {"images": [...]} (exactly one of those keys). Each image may be PNG or JPEG bytes (any
format PIL reads is accepted), a PIL image, a path to an image file (relative paths are read from the adapter's
working directory) or a uint8 numpy array; a single image not in a list is accepted too. The runner normalises
every image to PNG, at most MAX_IMAGE_SIDE (512) px on the long side, and keeps at most MAX_IMAGES (9); text is cut
to MAX_TEXT_CHARS (4000) characters including a note saying it was cut. The harness applies those caps again to
what it receives (the adapter shares the runner's interpreter, so it could forge the runner's payload): it reads at
most MAX_RESULT_BYTES, cuts the text, keeps MAX_IMAGES images and decodes, scales and re-encodes each one itself,
refusing images over MAX_IMAGE_PIXELS. Text and error messages are passed through redact() before they leave
run_adapter. Anything else (another return value, an exception, a non-zero exit, a crash, a timeout, too much disk
or memory) raises AdapterError with the reason and the tail of the traceback, so the agent can show it to the LLM
and ask for a repair.

The adapter's path. The runner puts <tmp>/lib at sys.path[0], holding a copy of sense/adapter_kit.py (and nothing
else), so an adapter can `import adapter_kit`; when the toolbox exists (agent.installer.toolbox_dir(), the packages
a user approved, under config.home()) it goes at the end of sys.path (sys.path.append, never site.addsitedir, so
no .pth file in it runs), so a module in it never shadows the standard library or the main site-packages.

matplotlib. Its first font scan takes over 10 s, and every run has a fresh HOME. So the harness builds the font
cache once (matplotlib_cache(), when the code imports matplotlib or adapter_kit and there is none yet; this is
not counted in the adapter's timeout) and copies its fontlist-*.json into the run's own MPLCONFIGDIR
(<tmp>/mplconfig), which the adapter's environment names.

The sandbox. run_adapter writes the code and a small runner script to a fresh temporary directory and runs
`sys.executable -I -B runner.py adapter.py <copy of the file>` there, in a new session (process group), with:
  - its own working directory (<tmp>/work), also its HOME and TMPDIR, so caches and stray files land there;
  - an environment without any variable whose name contains KEY, TOKEN, SECRET, PASSWORD, PASSWD, CREDENTIAL or
    AUTH, or starts with DATABRICKS_, OPENAI_, LINKUP_, ANTHROPIC_, AWS_, AZURE_, GOOGLE_, GH_ or GITHUB_, and with
    SENSE_ADAPTER_RUN set to a random token for this run;
  - HTTP_PROXY, HTTPS_PROXY and ALL_PROXY (and the lowercase forms) set to BLOCKED_PROXY, a closed local port
    (http://127.0.0.1:9), and NO_PROXY empty, so ordinary HTTP fails at once instead of hanging;
  - on macOS, a sandbox-exec profile that denies all network access; denies reading /Users, /Volumes,
    /private/tmp, /private/var/folders, the user's home and config.home() (the probe store), except the
    interpreter's prefixes and site-packages, this run's own folders and, when it exists, exactly the toolbox
    folder (not the rest of config.home(), not even its installed.json); denies reading .env* files and credential
    stores (~/.ssh, ~/.aws, ~/.codex, ~/.config, shell profiles, ...) even there; and denies writing anywhere but
    this run's temporary folder, `writable` and /dev. When sandbox-exec exists but cannot apply a profile (for
    example inside another sandbox), run_adapter refuses (AdapterError, cause "sandbox") unless
    SENSE_ADAPTER_UNSANDBOXED=1. On other systems there is no sandbox: only the environment and proxy measures and
    the limits below apply;
  - RLIMIT_FSIZE (FILE_SIZE_LIMIT, 1 GB) on every file the run writes, and checks every CHECK_EVERY seconds of the
    bytes in its folders (DISK_BUDGET, 2 GB) and of the resident memory of its processes (MEMORY_BUDGET, 2 GB);
    past either it is stopped (cause "limit");
  - a wall-clock timeout, enforced while the harness reads the result (a child left holding the result pipe
    cannot stall it). Afterwards, timeout or not, the harness kills the run's process group, the descendants it
    saw while the run lasted, and every process whose environment holds the run's token or whose command line
    names the run's folder. That catches forked children and the programs the adapter starts; a process that
    leaves the session, clears its environment and is reparented before it was seen can still escape, so this
    sweep is a best effort, not a guarantee.
Whatever the adapter prints, to stdout or stderr, and whatever its child processes print, goes to a log file; the
runner keeps the real stdout for its one JSON result (images as base64). The sandbox cannot stop an adapter from
reading other same-user processes' command lines and environments through sysctl (KERN_PROCARGS2): keep API keys
in .env files the server loads, not exported in the shell that starts it. redact() is the last line of defence for
text; an image can carry anything.

Probe files. A bridge is verified on files whose labels come from a source other than the adapter: a user's
examples folder, a public dataset, or files a generator script built to have their label (provenance "user",
"dataset" or "generated"; generated labels are only as independent as the generator, see agent.synth).
probe_folder reads a folder with one sub-folder per label: labels come from the folder names ("_" and "-" read as
spaces; folders starting with "." or "_" are ignored), files directly inside a label folder are used (with
`suffixes`, only files with those extensions; hidden, empty and other-type files and sub-folders are skipped and
counted), exact duplicates by sha256 are collapsed (the same bytes under two labels are dropped everywhere and
listed), at most max_per_label files per label are kept (a seeded pick), every label needs min_per_label files (at
least 3) and there must be 2 labels, and each label is split 60/20/20 (seeded; at least one dev and one test file
per label). Every file is copied, in a shuffled order across labels and with one fixed modification time, to
<config.home()>/probes/<name>/<sha256[:16]><extension> (extension lowercased) and the items point at the copies, so
that file names, inode order and times say nothing about the labels. The extension is kept, since many formats are
read by it: when extensions differ between labels, meta["warnings"] says so (pass `suffixes` to avoid it). Each
file must stand alone: a format that refers to sidecar files (OBJ with .mtl, glTF with .bin) loses them in the copy.

Verification. verify_files is sense.verify.verify for files: each scored item (`idx`, by default probes.test_idx;
at most max_items, None meaning 200 for an open-weight model and config.api_verify_items() otherwise, drawn with
`seed`) becomes a question with probes.question and up to five options (the true label and others, seeded,
sense.bridges.five_options), answered by model.choose with [bridge.prepare(file)]. Controls: "swapped" (the same
question with the bridged input of another scored file of another label that the adapter could read; when there is
none, that question counts as answered correctly) and "no_input". Chance is 1/min(5, n_labels); the interval is the
95% Wilson interval of sense.verify; card.dataset is probes.name, card.kind is bridge.kind and the verdict is
"candidate". An adapter that fails on a scored file counts that question as wrong (the model is not asked), and a
failed file never serves as a swapped input. Failures are listed in card.reason. When the adapter fails on more
than half of the scored files, verify_files raises AdapterError (cause "verify") instead of returning a card.
Adapter bridges are prepared in up to PREPARE_WORKERS processes at once.
Two more controls, on request:
  native=True   each scored question is asked again with the raw file as a text SenseInput (its UTF-8 text, at
                most MAX_TEXT_CHARS characters, a note saying when it was cut; a few bytes that are not UTF-8, at
                most 2 or 2% of the characters, read as U+FFFD): controls["native"], at the card's chance. Does the
                adapter add anything to what the model reads in the file as it is? It is measured file by file: a
                scored file that is not text (it does not decode, holds NUL, is blank or is more than 2% control
                characters) is asked with no input, since the model cannot read it as it is, and card.reason counts
                them. Only when more than half of the scored files are not text is native not measured, and
                card.reason says so.
  unseen=<ProbeSet>  a second, independent probe set whose categories the adapter was never built around. Its
                labels that equal a label of `probes` (case-insensitively, separators ignored) are dropped with their
                files; with fewer than 2 labels or 4 files left, card.reason says "Unseen categories not measured:
                <why>". Otherwise every remaining file (at most unseen_max_items, default as max_items, drawn with
                its own seeded stream) is a question with that set's question and up to five of its remaining
                labels: controls["unseen_categories"] is the accuracy and control_chance["unseen_categories"] is
                1/min(5, its labels); card.reason gives n, the labels and the set's source. A file the adapter fails
                on counts as wrong, as in the main set (the adapter was written before either set existed).

Fake mode needs nothing special: FakeModel picks the option a text input names, so a tiny adapter that reads, for
example, a test file's first line verifies end to end, offline.
"""

from __future__ import annotations

import ast
import base64
import binascii
import hashlib
import importlib.util
import io
import json
import os
import re
import secrets
import selectors
import shutil
import signal
import site
import subprocess
import sys
import tempfile
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import numpy as np

from sense import config
from sense.contracts import BaseModel, Bridge, ChoiceItem, FitRequest, ReportCard, SenseInput

MAX_TEXT_CHARS = 4000
MAX_IMAGES = 9
MAX_IMAGE_SIDE = 512
MAX_IMAGE_PIXELS = 4096 * 4096  # an image in a result larger than this is refused rather than decoded
SHEET_WIDTH = 1024  # the contact sheet of an image bridge is at most this wide
SHEET_GAP = 8
DEFAULT_TIMEOUT = 60.0
TAIL_CHARS = 2500  # how much of a traceback or error log an AdapterError keeps
REASON_CHARS = 1000  # an AdapterError's one-line reason is cut to this
TRACEBACK_CHARS = 8000  # and its traceback to this (a repair round's combined tracebacks fit)
MAX_RESULT_BYTES = 16 * 1024 * 1024  # what the harness reads from the runner's result pipe, at most
DISK_BUDGET = 2 * 1024**3  # bytes an adapter run may write (its temporary folder and `writable`), checked while it runs
MEMORY_BUDGET = 2 * 1024**3  # resident bytes of the adapter's processes, checked while it runs
FILE_SIZE_LIMIT = 1024**3  # RLIMIT_FSIZE in the runner: no single file it writes grows past this
CHECK_EVERY = 0.5  # seconds between the disk and memory checks of a running adapter
PREPARE_WORKERS = 4
BLOCKED_PROXY = "http://127.0.0.1:9"  # the discard port, closed: a request through it is refused at once
SANDBOX_EXEC = "/usr/bin/sandbox-exec"
UNSANDBOXED_ENV = "SENSE_ADAPTER_UNSANDBOXED"  # "1": run adapters even where the macOS sandbox cannot be applied
RUN_TOKEN_ENV = "SENSE_ADAPTER_RUN"  # a random token in every adapter process's environment, to find them all
_CACHE_SIZE = 1024
_CACHED_CAUSES = ("exception", "output")  # deterministic failures; a crash, a kill or a timeout may not recur

_CORE_LIBRARIES = [
    "numpy", "scipy", "PIL", "json", "csv", "math", "struct", "zipfile", "gzip", "io", "re", "xml.etree", "statistics",
]
_OPTIONAL_LIBRARIES = ["trimesh", "rdkit", "soundfile", "av", "cv2", "pandas", "matplotlib"]
_BINARIES = ["ffmpeg", "ffprobe"]
_BINARY_DIRS = ["/opt/homebrew/bin", "/usr/local/bin"]
KIT_FILE = Path(__file__).with_name("adapter_kit.py")  # copied into every run's lib/ folder
KIT_SUMMARY = (
    "this harness's helper module, on the adapter's path: import adapter_kit. video_frames(path, n=6) -> PIL "
    "frames and video_summary(path) -> dict, for video; contact_sheet(images) -> one PIL image; load_mesh(path), "
    "mesh_views(path, views=4) -> PIL renders and mesh_summary(path) -> dict with a plain description, for 3D "
    "meshes; read_molecule(path), molecule_summary(path) -> dict with a plain description and molecule_image(path), "
    "for molecules; read_coordinates(path) -> [(lat, lon)], coordinate_summary(lat, lon) -> dict and "
    "nearest_places(lat, lon), for coordinates; ValueError when a file cannot be read"
)
MPL_BUILD_TIMEOUT = 180.0  # seconds the one-off matplotlib font scan may take

_SECRET_WORDS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "PASSWD", "CREDENTIAL", "AUTH")
_SECRET_PREFIXES = ("DATABRICKS_", "OPENAI_", "LINKUP_", "ANTHROPIC_", "AWS_", "AZURE_", "GOOGLE_", "GH_", "GITHUB_")
_PROXY_VARS = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")
# Folders an adapter may not read under sandbox-exec (the interpreter, its libraries and the run's own temporary
# folder are allowed back), then credential stores denied even inside what was allowed back.
_READ_DENIED_ROOTS = ["/Users", "/Volumes", "/private/tmp", "/private/var/folders"]
_HOME_SECRETS = [  # relative to the user's home
    ".ssh", ".aws", ".netrc", ".config", ".git-credentials", ".docker", ".kube", ".databrickscfg", ".azure", ".gnupg",
    ".codex", ".claude", ".claude.json", ".zshrc", ".zprofile", ".zshenv", ".zlogin", ".bashrc", ".bash_profile",
    ".profile", ".npmrc", ".pypirc", "Library/Keychains", "Library/Cookies", "Library/Application Support",
]
# Key-shaped strings, redacted from whatever an adapter returns or raises before it reaches a trace or an LLM.
_KEY_SHAPES = re.compile(
    r"(?<![A-Za-z0-9])(?:sk|pk|rk|lk|ghp|gho|ghs|ghu|github_pat|xox[abprs]|dapi|AKIA|ASIA)[-_A-Za-z0-9]{16,}"
)


# ----------------------------------------------------------------------------- toolbox
def toolbox() -> list[str]:
    """What an adapter may use: the allow-listed modules that can be imported here (checked now, without importing
    them), then "ffmpeg (via subprocess)" and "ffprobe (via subprocess)" when those binaries exist, then
    "adapter_kit (<KIT_SUMMARY>)", then the top-level modules of the packages installed into the toolbox."""
    out = [name for name in _CORE_LIBRARIES + _OPTIONAL_LIBRARIES if _importable(name)]
    out += [f"{name} (via subprocess)" for name in _BINARIES if _binary(name)]
    out.append(f"adapter_kit ({KIT_SUMMARY})")
    out += [name for name in _toolbox_modules() if name not in out]
    return out


def _toolbox_site() -> Path | None:
    """The folder of packages a user approved (agent.installer.toolbox_dir()), resolved, when it exists."""
    from agent.installer import sys_path_entry  # imported here: sense does not need the agent package to load

    entry = sys_path_entry()
    return Path(entry).resolve() if entry else None


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _toolbox_modules() -> list[str]:
    """Top-level modules of the packages recorded in the toolbox's installed.json whose files are in the toolbox:
    each package's dist-info top_level.txt, else the first parts of the paths in its RECORD. Names starting with
    "_" (private extension modules) are left out."""
    site_dir = _toolbox_site()
    if site_dir is None:
        return []
    try:
        records = json.loads((site_dir.parent / "installed.json").read_text())
    except (OSError, ValueError):
        return []
    wanted = {_norm(r["name"]) for r in records if isinstance(r, dict) and isinstance(r.get("name"), str)} \
        if isinstance(records, list) else set()
    out: list[str] = []
    for info in sorted(site_dir.glob("*.dist-info")):
        if _norm(_dist_name(info)) not in wanted:
            continue
        for name in _top_levels(info):
            if name not in out and name.isidentifier() and not name.startswith("_") and _module_there(site_dir, name):
                out.append(name)
    return out


def _dist_name(info: Path) -> str:
    try:
        for line in (info / "METADATA").read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("name:"):
                return line.split(":", 1)[1].strip()
            if not line.strip():
                break
    except OSError:
        pass
    return info.name[: -len(".dist-info")].rsplit("-", 1)[0]


def _top_levels(info: Path) -> list[str]:
    try:
        listed = [line.strip() for line in (info / "top_level.txt").read_text().splitlines() if line.strip()]
        if listed:
            return listed
    except OSError:
        pass
    found: list[str] = []
    try:
        lines = (info / "RECORD").read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return found
    for line in lines:
        first = line.split(",", 1)[0].split("/", 1)
        if len(first) == 2:
            top = first[0]
        elif first[0].endswith(".py"):
            top = first[0][:-3]
        elif re.search(r"\.(so|pyd)$", first[0]):
            top = first[0].split(".", 1)[0]
        else:
            continue
        if top.endswith((".dist-info", ".data")) or top in ("__pycache__", "bin", "..") or top in found:
            continue
        found.append(top)
    return found


def _module_there(site_dir: Path, name: str) -> bool:
    return (site_dir / name).is_dir() or (site_dir / f"{name}.py").is_file() or any(site_dir.glob(f"{name}.*.so"))


def _importable(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _binary(name: str) -> str | None:
    found = shutil.which(name)
    if found:
        return found
    for directory in _BINARY_DIRS:
        candidate = Path(directory) / name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def code_imports(code: str) -> list[str]:
    """Top-level modules `code` imports, sorted, plus "ffmpeg"/"ffprobe" when the code names those commands.
    Code that does not parse gives []."""
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return []
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.add(node.module.split(".")[0])
    found.update(b for b in _BINARIES if re.search(rf"\b{b}\b", code))
    return sorted(found)


# ----------------------------------------------------------------------------- errors
class AdapterError(Exception):
    """An adapter failed: `reason` says what went wrong in one line, `traceback` holds the tail of the traceback
    (or of the process's error output when there is no traceback), `cause` is "exception", "output", "timeout",
    "exit", "limit", "sandbox" or "verify". The reason is cut to REASON_CHARS and the traceback to TRACEBACK_CHARS,
    so an error that quotes a whole file stays small in prompts and traces."""

    def __init__(self, reason: str, traceback: str = "", *, cause: str = "error"):
        reason = _clip(str(reason), REASON_CHARS)
        super().__init__(reason)
        self.reason = reason
        self.traceback = _tail(str(traceback or ""), TRACEBACK_CHARS)
        self.cause = cause

    def __str__(self) -> str:
        return self.reason if not self.traceback else f"{self.reason}\n{self.traceback.rstrip()}"

    def to_dict(self) -> dict[str, str]:
        return {"reason": self.reason, "traceback": self.traceback, "cause": self.cause}


def _clip(text: str, limit: int) -> str:
    """`text` cut to `limit` characters, saying how much was cut."""
    if len(text) <= limit:
        return text
    note = f" [... {len(text) - limit} more characters cut]"
    return text[: max(0, limit - len(note))] + note


def _tail(text: str, limit: int = TAIL_CHARS) -> str:
    text = text.rstrip()
    if len(text) <= limit:
        return text
    cut = text[-limit:]
    newline = cut.find("\n")
    return "..." + (cut[newline:] if 0 <= newline < 200 else cut)


def _secret_name(name: str) -> bool:
    upper = name.upper()
    return any(word in upper for word in _SECRET_WORDS) or upper.startswith(_SECRET_PREFIXES)


def redact(text: str) -> str:
    """`text` with the values of this process's secret-looking environment variables (names with KEY, TOKEN,
    SECRET, ...; at least 8 characters long) and key-shaped strings (sk-..., ghp_..., AKIA...) replaced by
    "[redacted]". Applied to everything an adapter returns as text or raises, before it reaches a trace or an LLM."""
    for name, value in os.environ.items():
        if len(value) >= 8 and _secret_name(name) and value in text:
            text = text.replace(value, "[redacted]")
    return _KEY_SHAPES.sub("[redacted]", text)


# ----------------------------------------------------------------------------- the runner (child process)
_RUNNER = r'''
import base64, io, json, os, sys, traceback

MAX_TEXT, MAX_IMAGES, MAX_SIDE, FSIZE = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
LIB, TOOLBOX = sys.argv[7], sys.argv[8]
sys.path.insert(0, LIB)  # adapter_kit
if TOOLBOX:  # the approved packages, last: they never shadow the standard library or the main site-packages;
    sys.path.append(TOOLBOX)  # never site.addsitedir, so no .pth file runs
try:  # no file this process or its children write grows past FSIZE (Python ignores SIGXFSZ: writes fail instead)
    import resource
    resource.setrlimit(resource.RLIMIT_FSIZE, (FSIZE, FSIZE))
except Exception:
    pass


class BadOutput(Exception):
    pass


def _png(item, k):
    from PIL import Image
    if isinstance(item, Image.Image):
        img = item
    elif isinstance(item, (bytes, bytearray, memoryview)):
        img = Image.open(io.BytesIO(bytes(item)))
    elif isinstance(item, (str, os.PathLike)):
        img = Image.open(os.fspath(item))
    elif type(item).__module__ == "numpy" and hasattr(item, "shape"):
        import numpy as np
        arr = np.asarray(item)
        if arr.dtype != np.uint8:
            lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
            arr = np.nan_to_num((arr - lo) / (hi - lo if hi > lo else 1.0) * 255).astype(np.uint8)
        img = Image.fromarray(arr)
    else:
        raise BadOutput(
            f"image {k + 1} is a {type(item).__name__}; give PNG or JPEG bytes, a PIL image or a path to an image file"
        )
    img.load()
    if img.width < 1 or img.height < 1:
        raise BadOutput(f"image {k + 1} is empty ({img.width}x{img.height})")
    if img.mode not in ("RGB", "RGBA", "L", "LA"):
        alpha = img.mode in ("PA", "RGBa", "La") or "transparency" in img.info
        img = img.convert("RGBA" if alpha else "RGB")
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _normalise(result):
    expected = 'adapt(path) must return {"text": str} or {"images": [...]}'
    if not isinstance(result, dict):
        raise BadOutput(f"{expected}, but it returned a {type(result).__name__}")
    keys = sorted(map(str, result))
    if keys == ["text"]:
        text = result["text"]
        if not isinstance(text, str):
            raise BadOutput(f'{expected}, but "text" is a {type(text).__name__}, not a str')
        notes = []
        if len(text) > MAX_TEXT:
            note = f"\n[cut by the harness: the adapter returned {len(text)} characters; the first "
            keep = max(0, MAX_TEXT - len(note) - len(str(MAX_TEXT)) - len(" are shown]"))
            note += f"{keep} are shown]"
            notes.append(f"text cut from {len(text)} to {keep} characters")
            text = text[:keep] + note
        return {"ok": True, "text": text, "notes": notes}
    if keys == ["images"]:
        images = result["images"]
        if not isinstance(images, (list, tuple)):
            images = [images]
        if not images:
            raise BadOutput("adapt(path) returned no images")
        notes = []
        if len(images) > MAX_IMAGES:
            notes.append(f"the adapter returned {len(images)} images; the first {MAX_IMAGES} are used")
            images = list(images)[:MAX_IMAGES]
        return {"ok": True, "images": [_png(item, k) for k, item in enumerate(images)], "notes": notes}
    raise BadOutput(f"{expected}, but it returned a dict with the keys {keys}")


def _traceback(exc, runner):
    te = traceback.TracebackException.from_exception(exc)
    te.stack = traceback.StackSummary.from_list([f for f in te.stack if f.filename != runner])
    return "".join(te.format())


def main():
    code_file, target = sys.argv[1], sys.argv[2]
    out = os.fdopen(os.dup(1), "w")  # the real stdout, for the one JSON result; not inherited by children
    os.dup2(2, 1)  # everything else printed, by the adapter or its child processes, goes to the error log
    status = 0
    try:
        with open(code_file, encoding="utf-8") as f:
            source = f.read()
        namespace = {"__name__": "adapter", "__file__": code_file, "__builtins__": __builtins__}
        sys.argv = [code_file, target]
        exec(compile(source, code_file, "exec"), namespace)
        adapt = namespace.get("adapt")
        if not callable(adapt):
            raise BadOutput("the code defines no function adapt(path)")
        payload = _normalise(adapt(target))
    except BadOutput as exc:
        payload, status = {"ok": False, "cause": "output", "error": str(exc), "traceback": ""}, 1
    except BaseException as exc:
        payload = {
            "ok": False,
            "cause": "exception",
            "error": "".join(traceback.format_exception_only(type(exc), exc)).strip(),
            "traceback": _traceback(exc, os.path.abspath(__file__)),
        }
        status = 1
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass
    out.write(json.dumps(payload))
    out.flush()
    os._exit(status)  # do not wait for threads the adapter may have left running


main()
'''


def _sbpl(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _interpreter_paths() -> list[str]:
    """What the child interpreter must read: its prefixes, its site-packages and the folders of ffmpeg/ffprobe."""
    paths = [sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix]
    paths.append(os.path.dirname(os.path.realpath(sys.executable)))
    try:
        paths += site.getsitepackages()
    except AttributeError:  # an old virtualenv's site module
        pass
    paths += [os.path.dirname(os.path.realpath(b)) for b in map(_binary, _BINARIES) if b]
    return [p for p in dict.fromkeys(os.path.realpath(p) for p in paths if p) if p != "/"]


def _profile(tmp: Path, writable: Path | None = None, toolbox_site: Path | None = None) -> str:
    """macOS sandbox profile: no network; no reads of the user's folders and temporary folders except the
    interpreter, its libraries, this run's own folders and the toolbox folder; no credential stores or .env files
    even there; writes only to this run's temporary folder (and `writable`)."""
    home = Path.home()
    own = [os.path.realpath(tmp)] + ([os.path.realpath(writable)] if writable is not None else [])
    denied = list(dict.fromkeys(_READ_DENIED_ROOTS + [os.path.realpath(home), os.path.realpath(config.home())]))
    readable = [os.path.realpath(toolbox_site)] if toolbox_site is not None else []
    allowed = list(dict.fromkeys(own + _interpreter_paths() + readable))
    ancestors = sorted({str(parent) for p in allowed for parent in Path(p).parents})
    secrets_ = " ".join(f"(subpath {_sbpl(str(home / p))})" for p in _HOME_SECRETS)
    return (
        "(version 1)\n"
        "(allow default)\n"
        "(deny network*)\n"
        f"(deny file-read* {' '.join(f'(subpath {_sbpl(p)})' for p in denied)})\n"
        f"(allow file-read* {' '.join(f'(subpath {_sbpl(p)})' for p in allowed)})\n"
        f"(allow file-read-metadata {' '.join(f'(literal {_sbpl(p)})' for p in ancestors)})\n"
        f'(deny file-read* (regex #"/\\.env[^/]*$") {secrets_})\n'
        "(deny file-write*)\n"
        f"(allow file-write* {' '.join(f'(subpath {_sbpl(p)})' for p in own)} (regex #\"^/dev/\"))\n"
    )


def _child_env(work: Path, token: str, mplconfig: Path | None = None) -> dict[str, str]:
    env = {name: value for name, value in os.environ.items() if not _secret_name(name)}
    for name in _PROXY_VARS:
        env[name] = BLOCKED_PROXY
    env["NO_PROXY"] = env["no_proxy"] = ""
    tmp = work / "tmp"
    tmp.mkdir(exist_ok=True)
    env.update(HOME=str(work), TMPDIR=str(tmp), MPLCONFIGDIR=str(mplconfig or work), XDG_CACHE_HOME=str(work / "cache"))
    env[RUN_TOKEN_ENV] = token
    path = env.get("PATH", os.defpath).split(os.pathsep)
    extra = [str(Path(b).parent) for b in map(_binary, _BINARIES) if b]
    env["PATH"] = os.pathsep.join(dict.fromkeys(path + extra))
    return env


_sandbox_works = False


def _probe_sandbox() -> bool:
    if sys.platform != "darwin" or not os.path.exists(SANDBOX_EXEC):
        return False
    try:
        done = subprocess.run(
            [SANDBOX_EXEC, "-p", "(version 1)(allow default)(deny network*)", "/usr/bin/true"],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return done.returncode == 0


def sandboxed() -> bool:
    """Whether adapters run under sandbox-exec here: macOS, the binary exists, and it can apply a profile (it
    cannot inside another sandbox). A working sandbox is remembered; a failed check is tried again next time."""
    global _sandbox_works
    if not _sandbox_works:
        _sandbox_works = _probe_sandbox()
    return _sandbox_works


def sandbox_problem() -> str | None:
    """Why run_adapter would refuse to run anything here, or None. On macOS (where /usr/bin/sandbox-exec exists)
    adapters run only inside the sandbox, unless SENSE_ADAPTER_UNSANDBOXED=1; elsewhere there is no sandbox to
    apply and only the environment and proxy measures stand."""
    if sys.platform != "darwin" or not os.path.exists(SANDBOX_EXEC) or sandboxed():
        return None
    if os.environ.get(UNSANDBOXED_ENV, "") == "1":
        return None
    return (
        "The macOS sandbox (sandbox-exec) cannot be applied here, perhaps because this process already runs inside "
        "another sandbox, so agent-written code is not run. Start the server from an ordinary terminal, or set "
        f"{UNSANDBOXED_ENV}=1 to run adapters without the sandbox (they then have network and file access)."
    )


def _kill_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        pass


_mpl_lock = threading.Lock()
_mpl_failed: set[str] = set()  # cache folders whose build failed in this process: not tried again


def _font_cache_in(folder: Path) -> list[Path]:
    try:
        return sorted(folder.glob("fontlist-*.json"))
    except OSError:
        return []


def matplotlib_cache(*, build: bool = True) -> Path | None:
    """<config.home()>/cache/matplotlib with matplotlib's font cache in it, built once (see the module docstring).
    None when matplotlib is not installed, when there is no cache and `build` is false, or when building failed."""
    if not _importable("matplotlib"):
        return None
    folder = config.home() / "cache" / "matplotlib"
    if _font_cache_in(folder):
        return folder
    if not build or str(folder) in _mpl_failed:
        return None
    with _mpl_lock:
        if _font_cache_in(folder):
            return folder
        staging = folder.with_name(f".matplotlib-{os.getpid()}-{secrets.token_hex(4)}")
        try:
            staging.mkdir(parents=True)
            env = {name: value for name, value in os.environ.items() if not _secret_name(name)}
            env.update(MPLCONFIGDIR=str(staging), MPLBACKEND="Agg")
            subprocess.run(
                [sys.executable, "-I", "-B", "-c", "import matplotlib.font_manager"], env=env,
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                timeout=MPL_BUILD_TIMEOUT, check=False,
            )
            if _font_cache_in(staging) and not folder.exists():
                os.rename(staging, folder)  # another process may have won the race: then its cache is used
        except (OSError, subprocess.SubprocessError):
            pass
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        if _font_cache_in(folder):
            return folder
        _mpl_failed.add(str(folder))
        return None


def _wants_matplotlib(code: str) -> bool:
    return bool({"matplotlib", "adapter_kit"} & set(code_imports(code)))


def _place_kit(tmp: Path) -> tuple[Path, Path]:
    """<tmp>/lib with a copy of adapter_kit.py, and <tmp>/mplconfig with matplotlib's font cache copied in when
    the harness has one."""
    lib, mplconfig = tmp / "lib", tmp / "mplconfig"
    lib.mkdir()
    mplconfig.mkdir()
    shutil.copyfile(KIT_FILE, lib / "adapter_kit.py")
    cache = matplotlib_cache(build=False)
    for font_list in _font_cache_in(cache) if cache is not None else []:
        shutil.copyfile(font_list, mplconfig / font_list.name)
    return lib, mplconfig


def _place_input(path: Path, folder: Path) -> Path:
    """A private copy of `path` as <folder>/input<extension>: the adapter sees no name, folder or neighbours of the
    original, and cannot change it. An APFS clone (copy-on-write, instant) when possible, else a full copy."""
    suffix = path.suffix.lower()
    if not re.fullmatch(r"\.[a-z0-9]{1,10}", suffix):
        suffix = ""
    folder.mkdir()
    dest = folder / f"input{suffix}"
    if sys.platform == "darwin":
        try:
            import ctypes

            libc = ctypes.CDLL(None, use_errno=True)
            if libc.clonefile(os.fsencode(path), os.fsencode(dest), 0) == 0:
                return dest
        except (OSError, AttributeError):
            pass
    shutil.copyfile(path, dest)
    return dest


def run_adapter(
    code: str, path: Path, *, timeout: float = DEFAULT_TIMEOUT, writable: Path | None = None
) -> dict:
    """Run adapt(path) from `code` on one file in a separate, sandboxed Python process (see the module docstring).

    The adapter gets a private copy of the file (<its folder>/input<extension>), not `path` itself. `writable` is a
    folder it may also read and write (for a probe generator's output); everything else it writes goes to its own
    temporary folder, removed afterwards.

    Returns {"text": str, "notes": [...], "seconds": float} or {"images": [PNG bytes, ...], "notes": [...],
    "seconds": float}. Raises AdapterError when the adapter raises, returns anything else, exits non-zero, crashes,
    runs longer than `timeout` seconds, writes too much or uses too much memory, or when the sandbox it needs cannot
    be applied (sandbox_problem()); ValueError when `path` is not a file or `timeout` is not positive."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"{path} is not a file; give the path of the file to adapt.")
    if not timeout or timeout <= 0:
        raise ValueError(f"timeout must be a positive number of seconds, not {timeout!r}.")
    if writable is not None:
        writable = Path(writable).expanduser().resolve()
        if not writable.is_dir():
            raise ValueError(f"{writable} is not a folder; give an existing folder the code may write to.")
    problem = sandbox_problem()
    if problem is not None:
        raise AdapterError(problem, cause="sandbox")
    if _wants_matplotlib(code):
        matplotlib_cache()  # the one-off font scan, before the adapter's clock starts
    toolbox_site = _toolbox_site()
    tmp = Path(tempfile.mkdtemp(prefix="sense-adapter-")).resolve()
    started = time.monotonic()
    try:
        work = tmp / "work"
        work.mkdir()
        target = _place_input(path, tmp / "input")
        lib, mplconfig = _place_kit(tmp)
        code_file, runner = tmp / "adapter.py", tmp / "runner.py"
        code_file.write_text(code, encoding="utf-8")
        runner.write_text(_RUNNER, encoding="utf-8")
        cmd = [
            sys.executable, "-I", "-B", str(runner), str(code_file), str(target),
            str(MAX_TEXT_CHARS), str(MAX_IMAGES), str(MAX_IMAGE_SIDE), str(FILE_SIZE_LIMIT),
            str(lib), str(toolbox_site or ""),
        ]
        if sandboxed():
            cmd = [SANDBOX_EXEC, "-p", _profile(tmp, writable, toolbox_site), *cmd]
        token = secrets.token_hex(12)
        log_path = tmp / "log.txt"
        with open(log_path, "wb") as log:
            proc = subprocess.Popen(
                cmd, cwd=work, env=_child_env(work, token, mplconfig), stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=log, start_new_session=True,
            )
            watch = _Watch(proc, token, [tmp] + ([writable] if writable is not None else []))
            try:
                stdout, stopped = _collect(proc, started + timeout, watch)
            finally:
                watch.stop()
                proc.stdout.close()
        if stopped is not None:
            raise _stopped(stopped, timeout, path, _log_tail(log_path, tmp))
        seconds = round(time.monotonic() - started, 3)
        return _result(stdout, proc.returncode, log_path, tmp, seconds)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _stopped(why: str, timeout: float, path: Path, tail: str) -> AdapterError:
    if why == "timeout":
        return AdapterError(
            f"The adapter ran for longer than {timeout:g} s on {path.name} and was stopped.", tail, cause="timeout"
        )
    if why == "output":
        return AdapterError(
            f"The adapter sent more than {_size(MAX_RESULT_BYTES)} of result data for {path.name} and was "
            "stopped; return less (the text and image caps apply anyway).", tail, cause="output",
        )
    if why == "disk":
        return AdapterError(
            f"The adapter wrote more than {_size(DISK_BUDGET)} of files on {path.name} and was stopped; "
            "write less (sample, or work in memory).", tail, cause="limit",
        )
    return AdapterError(
        f"The adapter used more than {_size(MEMORY_BUDGET)} of memory on {path.name} and was stopped; "
        "read the file in parts or sample it.", tail, cause="limit",
    )


def _size(n: int) -> str:
    return f"{n / 2**30:g} GB" if n >= 2**30 else f"{n / 2**20:g} MB"


def _collect(proc: subprocess.Popen, deadline: float, watch: "_Watch") -> tuple[bytes, str | None]:
    """The runner's result from its stdout, read with a deadline and a byte cap, never blocking past the deadline
    (a child the adapter left holding the pipe cannot stall the harness). (bytes, None) when the runner finished;
    (b"", "timeout" | "output" | "disk" | "memory") when it was stopped."""
    fd = proc.stdout.fileno()
    os.set_blocking(fd, False)
    chunks: list[bytes] = []
    size = 0
    exited = False
    with selectors.DefaultSelector() as selector:
        selector.register(fd, selectors.EVENT_READ)
        while True:
            now = time.monotonic()
            if now >= deadline:
                return b"", "timeout"
            if selector.select(min(CHECK_EVERY, deadline - now)):
                try:
                    data = os.read(fd, 1 << 16)
                except BlockingIOError:
                    data = None
                if data == b"":
                    return b"".join(chunks), None
                if data:
                    size += len(data)
                    if size > MAX_RESULT_BYTES:
                        return b"", "output"
                    chunks.append(data)
                    continue
            elif exited:  # the runner is gone and nothing more is waiting in the pipe
                return b"".join(chunks), None
            if proc.poll() is not None:
                exited = True  # read once more what is already in the pipe, then stop
                continue
            problem = watch.check()
            if problem is not None:
                return b"", problem


class _Watch:
    """Watches one adapter run: its disk use and the memory of its processes while it runs, then kills every
    process it can tie to the run (its process group, descendants seen while it ran, and any process whose
    environment carries the run's token or whose command line names its temporary folder)."""

    def __init__(self, proc: subprocess.Popen, token: str, folders: list[Path]):
        self.proc, self.token, self.folders = proc, token, folders
        self.tmp = str(folders[0])
        self.seen: dict[int, str] = {}  # descendant pid -> its command line when seen
        self.next_check = time.monotonic() + CHECK_EVERY

    def check(self) -> str | None:
        now = time.monotonic()
        if now < self.next_check:
            return None
        self.next_check = now + CHECK_EVERY
        if sum(_folder_size(f, DISK_BUDGET) for f in self.folders) > DISK_BUDGET:
            return "disk"
        if self._resident() > MEMORY_BUDGET:
            return "memory"
        return None

    def _resident(self) -> int:
        """Resident bytes of the runner's process group and descendants; remembers the descendants."""
        rows = _process_table()
        children: dict[int, list[int]] = {}
        for pid, ppid, _, _, _ in rows:
            children.setdefault(ppid, []).append(pid)
        ours, stack = set(), [self.proc.pid]
        while stack:
            pid = stack.pop()
            if pid not in ours:
                ours.add(pid)
                stack += children.get(pid, [])
        total = 0
        for pid, _, pgid, rss, command in rows:
            if pid in ours or pgid == self.proc.pid:
                total += rss
                if pid != self.proc.pid:
                    self.seen[pid] = command
        return total

    def stop(self) -> None:
        _kill_group(self.proc)
        victims = set()
        for pid, line in _environment_table():
            if f"{RUN_TOKEN_ENV}={self.token}" in line or self.tmp in line:
                victims.add(pid)
            elif pid in self.seen and line.startswith(self.seen[pid]):
                victims.add(pid)
        # macOS shows no environment for its own programs (/bin/sleep, say): their working folder gives them away
        victims |= {pid for pid, cwd in _working_dirs().items() if cwd == self.tmp or cwd.startswith(self.tmp + "/")}
        victims.discard(os.getpid())
        for pid in victims:
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                pass
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass


def _folder_size(folder: Path, stop_above: int) -> int:
    total = 0
    try:
        for root, _, files in os.walk(folder):
            for name in files:
                try:
                    total += os.lstat(os.path.join(root, name)).st_size
                except OSError:
                    pass
                if total > stop_above:
                    return total
    except OSError:
        pass
    return total


def _process_table() -> list[tuple[int, int, int, int, str]]:
    """(pid, ppid, pgid, resident bytes, command line) of every process; [] when ps fails."""
    try:
        out = subprocess.run(
            ["/bin/ps" if os.path.exists("/bin/ps") else "ps", "-A", "-o", "pid=,ppid=,pgid=,rss=,command="],
            capture_output=True, text=True, timeout=10,
            errors="replace",
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    rows = []
    for line in out.splitlines():
        parts = line.split(None, 4)
        if len(parts) >= 4 and all(p.isdigit() for p in parts[:4]):
            pid, ppid, pgid, rss = (int(p) for p in parts[:4])
            rows.append((pid, ppid, pgid, rss * 1024, parts[4] if len(parts) > 4 else ""))
    return rows


def _working_dirs() -> dict[int, str]:
    """pid -> working folder of every process of this user that can be inspected; {} when that is not possible."""
    out: dict[int, str] = {}
    if sys.platform != "darwin":
        proc_root = Path("/proc")
        for entry in proc_root.iterdir() if proc_root.is_dir() else []:
            if entry.name.isdigit():
                try:
                    out[int(entry.name)] = os.readlink(entry / "cwd")
                except OSError:
                    pass
        return out
    try:
        import ctypes

        libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        libproc.proc_listallpids.argtypes = [ctypes.c_void_p, ctypes.c_int]
        libproc.proc_pidinfo.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_int]
        count = libproc.proc_listallpids(None, 0)
        pids = (ctypes.c_int * (max(count, 0) + 256))()
        count = libproc.proc_listallpids(pids, ctypes.sizeof(pids))
        # struct proc_vnodepathinfo: the working folder's vnode_info (152 bytes), then its path (MAXPATHLEN)
        size, offset = 2352, 152
        buf = ctypes.create_string_buffer(size)
        for pid in pids[: max(count, 0)]:
            if pid > 0 and libproc.proc_pidinfo(pid, 9, 0, buf, size) == size:  # 9: PROC_PIDVNODEPATHINFO
                out[pid] = buf.raw[offset : offset + 1024].split(b"\0", 1)[0].decode("utf-8", "replace")
    except (OSError, AttributeError, ValueError):
        return {}
    return out


def _environment_table() -> list[tuple[int, str]]:
    """(pid, command line followed by the environment) of every process of this user that can be read."""
    rows: list[tuple[int, str]] = []
    if sys.platform == "darwin":
        try:
            out = subprocess.run(
                ["/bin/ps", "-E", "-A", "-ww", "-o", "pid=,command="], capture_output=True, text=True, timeout=10,
                errors="replace",
            ).stdout
        except (OSError, subprocess.SubprocessError):
            return rows
        for line in out.splitlines():
            pid, _, rest = line.strip().partition(" ")
            if pid.isdigit():
                rows.append((int(pid), rest.strip()))
        return rows
    proc_root = Path("/proc")
    if not proc_root.is_dir():
        return rows
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
            environ = (entry / "environ").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
        except OSError:
            continue
        rows.append((int(entry.name), f"{command.strip()} {environ}"))
    return rows


def _log_tail(log_path: Path, tmp: Path) -> str:
    try:
        with open(log_path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 4 * TAIL_CHARS))
            text = f.read().decode("utf-8", "replace")
    except OSError:
        return ""
    return _tail(redact(_scrub(text, tmp)))


def _scrub(text: str, tmp: Path) -> str:
    """Temporary paths out of a traceback: "<tmp>/adapter.py" reads "adapter.py"."""
    for prefix in {str(tmp), os.path.realpath(tmp)}:
        text = text.replace(prefix + os.sep + "work" + os.sep, "").replace(prefix + os.sep, "")
    return text


def _result(stdout: bytes, returncode: int, log_path: Path, tmp: Path, seconds: float) -> dict:
    """The runner's payload, checked again on this side: the adapter runs in the runner's interpreter and could
    have written its own payload, so the caps are applied anew and every image is decoded and re-encoded here."""
    try:
        payload = json.loads(stdout.decode("utf-8")) if stdout.strip() else None
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError):
        payload = None
    if not isinstance(payload, dict):
        if returncode and returncode < 0:
            try:
                name = signal.Signals(-returncode).name
            except ValueError:
                name = f"signal {-returncode}"
            reason = f"The adapter process was killed by {name} (a crash, or it ran out of memory)."
        elif returncode:
            reason = f"The adapter process exited with code {returncode} before returning a result."
        else:
            reason = "The adapter process ended without returning a result."
        raise AdapterError(reason, _log_tail(log_path, tmp), cause="exit")
    if not payload.get("ok"):
        cause = payload.get("cause") if payload.get("cause") in ("exception", "output") else "exception"
        error = _clip(str(payload.get("error") or "unknown error"), REASON_CHARS)
        lead = "adapt(path) raised " if cause == "exception" else ""
        tail = _scrub(str(payload.get("traceback") or "")[-4 * TAIL_CHARS :], tmp) or _log_tail(log_path, tmp)
        raise AdapterError(redact(f"{lead}{_scrub(error, tmp)}"), _tail(redact(tail)), cause=cause)
    raw_notes = payload.get("notes") if isinstance(payload.get("notes"), list) else []
    notes = [_clip(str(n), 300) for n in raw_notes[:10]]
    if isinstance(payload.get("text"), str):
        text = payload["text"]
        if len(text) > MAX_TEXT_CHARS:
            note = f"\n[cut by the harness: the adapter returned {len(text)} characters]"
            notes.append(f"text cut from {len(text)} characters")
            text = text[: MAX_TEXT_CHARS - len(note)] + note
        return {"text": redact(text), "notes": notes, "seconds": seconds}
    images = payload.get("images")
    if not isinstance(images, list) or not images:
        raise AdapterError("The adapter's result held neither text nor images.", cause="output")
    if len(images) > MAX_IMAGES:
        notes.append(f"the result held {len(images)} images; the first {MAX_IMAGES} are used")
        images = images[:MAX_IMAGES]
    return {"images": [_trusted_png(item, k) for k, item in enumerate(images)], "notes": notes, "seconds": seconds}


def _trusted_png(item: Any, k: int) -> bytes:
    """One result image, decoded here with a pixel cap, scaled to MAX_IMAGE_SIDE and re-encoded as PNG."""
    from PIL import Image

    try:
        data = base64.b64decode(item, validate=True) if isinstance(item, str) else None
    except (binascii.Error, ValueError):
        data = None
    if not data:
        raise AdapterError(f"Image {k + 1} of the adapter's result is not base64 image data.", cause="output")
    try:
        img = Image.open(io.BytesIO(data))
        if img.width * img.height > MAX_IMAGE_PIXELS:
            raise AdapterError(
                f"Image {k + 1} of the adapter's result is {img.width}x{img.height}, too large to decode.",
                cause="output",
            )
        img.load()
        if img.mode not in ("RGB", "RGBA", "L", "LA"):
            img = img.convert("RGBA" if img.mode in ("P", "PA", "RGBa", "La") else "RGB")
        img.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
    except AdapterError:
        raise
    except Exception as exc:
        raise AdapterError(f"Image {k + 1} of the adapter's result cannot be read: {exc}", cause="output") from None
    return buf.getvalue()


# ----------------------------------------------------------------------------- the bridge
_cache: OrderedDict[tuple[str, str, str], dict | AdapterError] = OrderedDict()
_cache_lock = threading.Lock()


def _sha256_file(path: Path) -> str:
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def _slug(text: str) -> str:
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9-]", "-", str(text).lower())).strip("-") or "file"


def contact_sheet(images: list[bytes]) -> bytes:
    """One PNG: the images tiled in a grid (up to 3 columns), at most SHEET_WIDTH px wide. One image comes back as
    it is."""
    from PIL import Image

    if not images:
        raise ValueError("A contact sheet needs at least one image.")
    if len(images) == 1:
        return images[0]
    cols = int(np.ceil(np.sqrt(len(images))))
    rows = int(np.ceil(len(images) / cols))
    cell = min(MAX_IMAGE_SIDE, (SHEET_WIDTH - SHEET_GAP * (cols + 1)) // cols)
    sheet = Image.new("RGB", (cols * cell + SHEET_GAP * (cols + 1), rows * cell + SHEET_GAP * (rows + 1)), "white")
    for k, data in enumerate(images):
        img = Image.open(io.BytesIO(data))
        img.load()
        if img.mode in ("RGBA", "LA", "P"):
            img = img.convert("RGBA")
            backed = Image.new("RGBA", img.size, "white")
            backed.alpha_composite(img)
            img = backed
        img = img.convert("RGB")
        img.thumbnail((cell, cell))
        r, c = divmod(k, cols)
        x = SHEET_GAP + c * (cell + SHEET_GAP) + (cell - img.width) // 2
        y = SHEET_GAP + r * (cell + SHEET_GAP) + (cell - img.height) // 2
        sheet.paste(img, (x, y))
    buf = io.BytesIO()
    sheet.save(buf, format="PNG")
    return buf.getvalue()


class AdapterBridge:
    """A bridge whose input comes from agent-written code run on the file itself: no specialist, no embedding."""

    def __init__(
        self,
        *,
        kind: Literal["text", "image"],
        modality: str,
        model_name: str,
        code: str,
        manifest: dict[str, Any] | None = None,
        slug: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        if kind not in ("text", "image"):
            raise ValueError(f'An adapter bridge is of kind "text" or "image", not {kind!r}.')
        self.kind = kind
        self.modality = modality
        self.model_name = model_name
        self.code = code
        self.timeout = float(timeout)
        defaults = {
            "modality": modality,
            "format": "",
            "route": kind,
            "description": "",
            "libraries": code_imports(code),
            "written_by": "",
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "attempts": 1,
        }
        self.manifest = {**defaults, **(manifest or {})}
        fmt = self.manifest.get("format") or ""
        self.name = f"adapter:{_slug(slug or (f'{modality}-{fmt}' if fmt else modality))}"
        self._code_sha = hashlib.sha256(code.encode("utf-8")).hexdigest()

    def from_embedding(self, emb: np.ndarray, source: str = "") -> SenseInput:
        raise ValueError(
            f"{self.name} is a file-based bridge: it reads the file with agent-written code and has no embedding. "
            "Call prepare(path) with the file instead."
        )

    def run(self, path: Path) -> dict:
        """The adapter's normalised output for one file, {"text": ...} or {"sheet": PNG bytes, "n": images}, cached
        per code and file sha256 within the process. Raises AdapterError; an exception the adapter raised or output
        that breaks the contract is cached too (the same code on the same bytes fails the same way), but a crash, a
        kill, a timeout, a limit or a missing sandbox is not, since those may not happen again."""
        path = Path(path)
        key = (self._code_sha, f"{self.timeout:g}", _sha256_file(path))
        with _cache_lock:
            hit = _cache.get(key)
            if hit is not None:
                _cache.move_to_end(key)
        if hit is None:
            try:
                out = run_adapter(self.code, path, timeout=self.timeout)
                if "text" in out:
                    hit = {"text": out["text"]}
                else:
                    hit = {"sheet": contact_sheet(out["images"]), "n": len(out["images"])}
            except AdapterError as err:
                if err.cause not in _CACHED_CAUSES:
                    raise
                hit = err
            with _cache_lock:
                _cache[key] = hit
                while len(_cache) > _CACHE_SIZE:
                    _cache.popitem(last=False)
        if isinstance(hit, AdapterError):
            raise hit
        return hit

    def prepare(self, path: Path) -> SenseInput:
        path = Path(path)
        out = self.run(path)
        if self.kind == "text":
            if "text" not in out:
                raise AdapterError(
                    f"{self.name} is a text bridge, but its adapter returned {out['n']} image(s) for {path.name}.",
                    cause="output",
                )
            text = out["text"]
            preview = " ".join(text.split())
            preview = preview if len(preview) <= 100 else preview[:97] + "..."
            return SenseInput(
                kind="text", modality=self.modality, source=path.name, text=text,
                detail=f'text from the agent-written adapter: "{preview}"',
            )
        if "sheet" not in out:
            raise AdapterError(
                f"{self.name} is an image bridge, but its adapter returned text for {path.name}.", cause="output"
            )
        n = out["n"]
        return SenseInput(
            kind="image", modality=self.modality, source=path.name, image=out["sheet"],
            detail=f"{n} image{'' if n == 1 else 's'} from the agent-written adapter",
        )

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        meta = {
            "kind": self.kind,
            "name": self.name,
            "modality": self.modality,
            "model_name": self.model_name,
            "adapter": True,
            "manifest": self.manifest,
            "timeout": self.timeout,
        }
        (directory / "adapter.py").write_text(self.code, encoding="utf-8")
        (directory / "bridge.json").write_text(json.dumps(meta, indent=2))


def load_adapter_bridge(directory: Path) -> AdapterBridge:
    directory = Path(directory)
    meta_path, code_path = directory / "bridge.json", directory / "adapter.py"
    if not meta_path.is_file() or not code_path.is_file():
        raise FileNotFoundError(f"{directory} does not hold a saved adapter bridge (bridge.json and adapter.py).")
    meta = json.loads(meta_path.read_text())
    if not meta.get("adapter"):
        raise ValueError(f"{meta_path} is not an adapter bridge; load it with sense.bridges.load_bridge.")
    return AdapterBridge(
        kind=meta["kind"],
        modality=meta["modality"],
        model_name=meta["model_name"],
        code=code_path.read_text(encoding="utf-8"),
        manifest=meta.get("manifest") or {},
        slug=str(meta["name"]).split(":", 1)[-1],
        timeout=float(meta.get("timeout", DEFAULT_TIMEOUT)),
    )


# ----------------------------------------------------------------------------- probe files
Provenance = Literal["user", "dataset", "generated"]


@dataclass
class ProbeSet:
    """Labelled probe files for verifying an adapter. Labels come from a source independent of the adapter."""

    name: str
    title: str
    modality: str
    question: str
    labels: list[str]
    items: list[tuple[Path, int]]  # (file, label index)
    fit_idx: np.ndarray
    dev_idx: np.ndarray
    test_idx: np.ndarray
    source: str = ""
    licence: str = ""
    provenance: Provenance = "user"
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def y(self) -> np.ndarray:
        return np.array([label for _, label in self.items], dtype=np.int64)

    @property
    def files(self) -> list[Path]:
        return [Path(p) for p, _ in self.items]

    def to_dict(self) -> dict[str, Any]:
        """A JSON-ready summary (counts, not files)."""
        y = self.y
        return {
            "name": self.name,
            "title": self.title,
            "modality": self.modality,
            "question": self.question,
            "labels": list(self.labels),
            "n": len(self.items),
            "per_label": {lab: int((y == c).sum()) for c, lab in enumerate(self.labels)},
            "split": {"fit": len(self.fit_idx), "dev": len(self.dev_idx), "test": len(self.test_idx)},
            "source": self.source,
            "licence": self.licence,
            "provenance": self.provenance,
            "meta": self.meta,
        }


def probe_folder(
    path: Path,
    *,
    name: str | None = None,
    question: str,
    modality: str,
    provenance: Provenance = "user",
    source: str = "",
    licence: str = "",
    min_per_label: int = 4,
    max_per_label: int = 40,
    seed: int = 0,
    suffixes: list[str] | None = None,
) -> ProbeSet:
    """A ProbeSet from a folder with one sub-folder of files per label (see the module docstring). `suffixes`
    (for example [".obj"]) keeps only files with those extensions, case-insensitively; others are skipped and
    counted, so that sidecar files (.mtl, .bin, textures) are not taken for examples."""
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"{root} is not a folder; give a folder with one sub-folder of example files per label.")
    if provenance not in ("user", "dataset", "generated"):
        raise ValueError(f'provenance is "user", "dataset" or "generated", not {provenance!r}.')
    if min_per_label < 3:
        raise ValueError(f"min_per_label is {min_per_label}, but each label needs at least 3 files (fit, dev, test).")
    if max_per_label < min_per_label:
        raise ValueError(f"max_per_label ({max_per_label}) is below min_per_label ({min_per_label}); raise it.")
    wanted = None
    if suffixes is not None:
        wanted = {("." + s.lower().lstrip(".")) if s else "" for s in suffixes}
    set_name = _slug(name or root.name)
    folders = _probe_label_folders(root)
    labels = [label for label, _ in folders]

    skipped = {"hidden": 0, "empty": 0, "subfolders": 0, "other_types": 0}
    copies: dict[str, list[tuple[int, Path]]] = {}
    for c, (_, directory) in enumerate(folders):
        for p in sorted(directory.iterdir()):
            if p.name.startswith("."):
                skipped["hidden"] += 1
            elif p.is_dir():
                skipped["subfolders"] += 1
            elif p.is_file():
                if wanted is not None and p.suffix.lower() not in wanted:
                    skipped["other_types"] += 1
                    continue
                if p.stat().st_size == 0:
                    skipped["empty"] += 1
                    continue
                copies.setdefault(_sha256_file(p), []).append((c, p))
    kept: list[list[tuple[Path, str]]] = [[] for _ in labels]
    duplicates, conflicts = [0] * len(labels), []
    for digest, found in copies.items():
        if len({c for c, _ in found}) > 1:
            conflicts += [str(p.relative_to(root)) for _, p in found]
            continue
        c, first = min(found, key=lambda cp: cp[1])
        kept[c].append((first, digest))
        duplicates[c] += len(found) - 1

    short = [(labels[c], len(files)) for c, files in enumerate(kept) if len(files) < min_per_label]
    if short:
        listed = "; ".join(f"{label!r} has {n}" for label, n in short)
        raise ValueError(
            f"Too few usable files in {root}: {listed}. Every label needs at least {min_per_label} distinct, "
            "non-empty files (min_per_label; exact duplicates count once, and files whose bytes appear under two "
            "labels are dropped). Add files to those folders, or lower min_per_label (3 at least)."
        )

    rng = np.random.default_rng(seed)
    capped = 0
    for c, files in enumerate(kept):
        files.sort()
        if len(files) > max_per_label:
            capped += len(files) - max_per_label
            files[:] = sorted(files[k] for k in rng.permutation(len(files))[:max_per_label])

    target = config.home() / "probes" / set_name
    target.mkdir(parents=True, exist_ok=True)
    # Copy in a shuffled order across labels and give every copy the same times, so that neither inode numbers,
    # creation order nor modification times line up with the labels.
    flat = [(p, digest) for files in kept for p, digest in files]
    copied: dict[str, Path] = {}
    for k in rng.permutation(len(flat)):
        p, digest = flat[k]
        copied[digest] = _copy_probe(p, digest, target)
    items: list[tuple[Path, int]] = []
    fit, dev, test = [], [], []
    for c, files in enumerate(kept):
        start = len(items)
        for p, digest in files:
            items.append((copied[digest], c))
        order = start + rng.permutation(len(files))
        n = len(files)
        n_dev = max(1, int(round(0.2 * n)))
        n_fit = min(int(round(0.6 * n)), n - n_dev - 1)
        fit += order[:n_fit].tolist()
        dev += order[n_fit : n_fit + n_dev].tolist()
        test += order[n_fit + n_dev :].tolist()

    counts = {labels[c]: len(files) for c, files in enumerate(kept)}
    warnings = []
    by_label = [{Path(p).suffix.lower() for p, _ in files} for files in kept]
    telling = sorted(set.union(*by_label) - set.intersection(*by_label))
    if telling:
        warnings.append(
            f"the file extensions differ between labels ({', '.join(e or 'none' for e in telling)} appear under some "
            "labels only), and the copies keep their extensions, so an extension can give a label away; use one "
            "file type for every label"
        )
    return ProbeSet(
        name=set_name,
        title=f"{root.name}: {len(labels)} labels, {len(items)} files",
        modality=modality,
        question=question,
        labels=labels,
        items=items,
        fit_idx=np.array(sorted(fit), dtype=np.int64),
        dev_idx=np.array(sorted(dev), dtype=np.int64),
        test_idx=np.array(sorted(test), dtype=np.int64),
        source=source,
        licence=licence,
        provenance=provenance,
        meta={
            "folder": str(root),
            "copied_to": str(target),
            "per_label": counts,
            "duplicates_collapsed": int(sum(duplicates)),
            "conflicts_dropped": sorted(conflicts),
            "capped": capped,
            "skipped": skipped,
            "suffixes": sorted(wanted) if wanted is not None else None,
            "warnings": warnings,
            "seed": seed,
            "min_per_label": min_per_label,
            "max_per_label": max_per_label,
        },
    )


def _probe_label_folders(root: Path) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    seen: dict[str, str] = {}
    for directory in sorted(root.iterdir()):
        if directory.name.startswith((".", "_")) or not directory.is_dir():
            continue
        label = " ".join(directory.name.replace("_", " ").replace("-", " ").split())
        if not label:
            raise ValueError(f"The folder {directory.name!r} in {root} gives an empty label. Rename it after its label.")
        key = label.casefold()
        if key in seen:
            raise ValueError(
                f"The folders {seen[key]!r} and {directory.name!r} in {root} both give the label {label!r}. "
                "Merge them into one folder, or rename one of them."
            )
        seen[key] = directory.name
        out.append((label, directory))
    if len(out) < 2:
        found = f" ({out[0][1].name!r})" if out else ""
        raise ValueError(
            f"{root} has {len(out)} label folder{'' if len(out) == 1 else 's'}{found}, but probes need at least 2 "
            "labels. Put the example files of each label in their own sub-folder, named after the label."
        )
    return out


def _copy_probe(src: Path, digest: str, target: Path) -> Path:
    """Copy `src` to <target>/<sha256[:16]><extension>, a name that says nothing about its label."""
    suffix = src.suffix.lower()
    if not re.fullmatch(r"\.[a-z0-9]{1,10}", suffix):
        suffix = ""
    dest = target / f"{digest[:16]}{suffix}"
    if not (dest.is_file() and dest.stat().st_size == src.stat().st_size):
        partial = target / f".{dest.name}.{os.getpid()}.{threading.get_ident()}.part"
        shutil.copyfile(src, partial)
        os.replace(partial, dest)
    os.utime(dest, (_PROBE_TIME, _PROBE_TIME))
    return dest


_PROBE_TIME = 946684800  # 2000-01-01: every probe copy gets this access and modification time


# ----------------------------------------------------------------------------- verification
def verify_files(
    model: BaseModel,
    bridge: Bridge,
    probes: ProbeSet,
    request: FitRequest,
    *,
    max_items: int | None = None,
    seed: int = 0,
    idx: np.ndarray | list[int] | None = None,
    native: bool = False,
    unseen: ProbeSet | None = None,
    unseen_max_items: int | None = None,
) -> ReportCard:
    """Score a file-based bridge on held-out probe files against swapped-input and no-input controls, and with
    `native` and `unseen` the native and unseen-categories controls (module docstring). `idx` names the probe
    items to score (None: probes.test_idx); an adapter written before any probe existed was fitted on none of
    them, so the caller may pass every item."""
    from sense.bridges import five_options
    from sense.verify import _subsample, _wilson

    if bridge.model_name != model.info.name:
        raise ValueError(f"{bridge.name} was built for {bridge.model_name}, not {model.info.name}; build it for this model.")
    if bridge.kind == "image" and "image" not in model.info.native:
        raise ValueError(
            f"{model.info.label} does not read images, so {bridge.name} cannot reach it; use a text adapter."
        )
    n_labels = len(probes.labels)
    if n_labels < 2:
        raise ValueError(f"Probe set {probes.name} has {n_labels} label; verification needs at least 2.")
    if max_items is None:
        max_items = 200 if model.info.open_weights else config.api_verify_items()
    rng = np.random.default_rng(seed)
    pool = probes.test_idx if idx is None else np.asarray(idx, dtype=np.int64)
    test = _subsample(pool, max_items, rng)
    if len(test) == 0:
        raise ValueError(f"Probe set {probes.name} has no held-out files to verify on.")
    y = probes.y
    if len(set(y[test].tolist())) < 2:
        raise ValueError(f"The held-out files of {probes.name} hold one label; verification needs at least two.")
    options, answers = five_options(y[test], n_labels, rng)

    prepared = _prepare_all(bridge, [probes.items[i][0] for i in sorted(set(test.tolist()))])
    inputs = {int(i): prepared[probes.items[int(i)][0]] for i in test}
    failed = [int(i) for i in test if isinstance(inputs[int(i)], AdapterError)]
    if len(failed) * 2 > len(test):
        first = inputs[failed[0]]
        raise AdapterError(
            f"The adapter failed on {len(failed)} of {len(test)} held-out files, more than half, so {bridge.name} "
            f"cannot be verified. First failure ({Path(probes.items[failed[0]][0]).name}): {first.reason}",
            first.traceback,
            cause="verify",
        )

    # Swap partners come only from files the adapter read: a failed partner would turn the control into a
    # no-input question and flatter an adapter that fails on one label. With no readable file of another label,
    # the swapped question counts as answered correctly (the conservative mirror of "a failure counts as wrong").
    failed_set = set(failed)
    readable = np.array([int(i) for i in test if int(i) not in failed_set], dtype=np.int64)
    swap: list[int | None] = []
    for i in test:
        partners = readable[y[readable] != y[int(i)]] if len(readable) else readable
        swap.append(int(partners[rng.integers(len(partners))]) if len(partners) else None)

    def ask(per_item: list[SenseInput | AdapterError | None | bool]) -> float:
        return _ask(model, probes.question, probes.labels, options, answers, per_item)

    accuracy = ask([inputs[int(i)] for i in test])
    controls = {
        "swapped": ask([True if j is None else inputs[j] for j in swap]),
        "no_input": ask([None for _ in test]),
    }
    control_chance: dict[str, float] = {}
    reasons: list[str] = []
    if failed:
        shown = "; ".join(
            f"{Path(probes.items[i][0]).name}: {_clip(' '.join(inputs[i].reason.split()), 200)}" for i in failed[:5]
        )
        more = f"; and {len(failed) - 5} more" if len(failed) > 5 else ""
        reasons.append(
            f"The adapter failed on {len(failed)} of {len(test)} held-out files; those questions count as wrong, and "
            f"no failed file serves as a swapped input. {shown}{more}."
        )
    if native:
        # File by file: one file that is not text must not switch the control off for a whole set of text files.
        files = [Path(probes.items[int(i)][0]) for i in test]
        texts = [_native_text(p) for p in files]
        not_text = [p for p, text in zip(files, texts) if text is None]
        if len(not_text) * 2 > len(files):
            reasons.append(
                f"Native not measured: {len(not_text)} of the {len(files)} scored files are not text (first: "
                f"{not_text[0].name}), so the model cannot be given most of them as they are."
            )
        else:
            controls["native"] = ask([
                None if text is None else SenseInput(
                    kind="text", modality=bridge.modality, source=p.name, text=text,
                    detail="the file as it is, given as text",
                )
                for p, text in zip(files, texts)
            ])
            if not_text:
                reasons.append(
                    f"Native: {len(not_text)} of the {len(files)} scored files are not text (first: "
                    f"{not_text[0].name}); the model, which cannot read those as they are, was asked them with no "
                    "input."
                )
    if unseen is not None:
        score, chance_unseen, note = _unseen_categories(
            model, bridge, unseen, probes.labels, max_items if unseen_max_items is None else unseen_max_items, seed,
        )
        if score is not None:
            controls["unseen_categories"] = score
            control_chance["unseen_categories"] = chance_unseen
        reasons.append(note)
    return ReportCard(
        model=model.info.name,
        modality=bridge.modality,
        dataset=probes.name,
        scope=request.scope,
        bridge=bridge.name,
        kind=bridge.kind,
        n=len(test),
        chance=round(1 / min(5, n_labels), 4),
        accuracy=accuracy,
        controls=controls,
        verdict="candidate",
        reason=" ".join(reasons),
        created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        interval=_wilson(accuracy, len(test)),
        control_chance=control_chance,
    )


def _ask(
    model: BaseModel,
    question: str,
    labels: list[str],
    options: list[list[int]],
    answers: np.ndarray,
    per_item: list[SenseInput | AdapterError | None | bool],
) -> float:
    """Accuracy over the questions: an AdapterError counts as wrong and True as right, without asking; None asks
    with no input."""
    asked = [k for k, inp in enumerate(per_item) if not isinstance(inp, (AdapterError, bool))]
    items = [
        ChoiceItem(
            question=question,
            options=[labels[c] for c in options[k]],
            inputs=[] if per_item[k] is None else [per_item[k]],
        )
        for k in asked
    ]
    picks = np.full(len(per_item), -1, dtype=np.int64)
    if items:
        picks[asked] = model.choose(items)
    right = picks == answers
    right[[k for k, inp in enumerate(per_item) if inp is True]] = True
    return round(float(right.mean()), 4)


def _native_text(path: Path) -> str | None:
    """The file as it is, for the native control: its UTF-8 text (a byte-order mark dropped; a few bytes that do
    not decode read as U+FFFD, so one Latin-1 letter does not make a note binary: at most 2 of them or 2% of the
    characters, whichever is more, and never more than 10%), at most MAX_TEXT_CHARS characters with a note when
    cut; None when it is not text (more does not decode, it holds NUL or is more than 2% control characters, or
    is blank)."""
    import codecs

    with open(path, "rb") as f:
        raw = f.read(MAX_TEXT_CHARS * 4 + 4)
        more = bool(f.read(1))
    body = raw.removeprefix(codecs.BOM_UTF8)
    try:
        text = codecs.getincrementaldecoder("utf-8")().decode(body, final=not more)
    except UnicodeDecodeError:
        text = codecs.getincrementaldecoder("utf-8")(errors="replace").decode(body, final=not more)
        bad = text.count("\ufffd")
        if bad > max(2, 0.02 * len(text)) or bad > 0.1 * len(text):
            return None
    control = sum(1 for ch in text if ord(ch) < 32 and ch not in "\n\r\t\f")
    if not text.strip() or "\x00" in text or control > 0.02 * len(text):
        return None
    if more or len(text) > MAX_TEXT_CHARS:
        note = "\n[cut: the file goes on]"
        text = text[: MAX_TEXT_CHARS - len(note)] + note
    return text


def _label_key(label: str) -> str:
    """A label compared case-insensitively with its separators ignored: "Sci-Fi", "sci fi" and "sci_fi" match."""
    return re.sub(r"[\W_]+", "", label.casefold())


def _label_keys(label: str) -> set[str]:
    """_label_key(label) and the other names of the same country (country_aliases), so a probe set labelled by
    country code and one labelled by country name ("BR", "brazil") share a label."""
    key = _label_key(label)
    return {key} | country_aliases().get(key, set())


def country_aliases(path: str | None = None) -> dict[str, set[str]]:
    """Loose label key -> loose keys of the same country: its ISO 3166 alpha-2 code and Natural Earth's country
    name (with and without "of America" and "The"), read from the gazetteer sense.adapter_kit uses (`path`, else
    SENSE_GAZETTEER, else ~/.cache/sense-harness/gazetteer/ne_10m_populated_places_simple.geojson); {} when it is
    missing or unreadable. Cached per path. Used only to keep two probe sets' categories apart, where one source
    labels countries by code and the other by name; a mistaken alias can only treat more labels as shared."""
    path = path or os.environ.get("SENSE_GAZETTEER") or os.path.expanduser(
        "~/.cache/sense-harness/gazetteer/ne_10m_populated_places_simple.geojson")
    return _country_aliases_at(path)


@lru_cache(maxsize=8)
def _country_aliases_at(path: str) -> dict[str, set[str]]:
    try:
        with open(path, encoding="utf-8") as handle:
            features = json.load(handle).get("features") or []
    except (OSError, ValueError, AttributeError):
        features = []
    groups: dict[str, set[str]] = {}
    for feature in features:
        props = (feature or {}).get("properties") or {}
        code, name = str(props.get("iso_a2") or ""), str(props.get("adm0name") or "")
        if len(code) != 2 or not code.isalpha() or not name:
            continue
        short = re.sub(r"\b(of america|the)\b", "", name, flags=re.IGNORECASE)
        keys = {_label_key(text) for text in (code, name, short)} - {""}
        for key in keys:
            groups.setdefault(key, set()).update(keys)
    return groups


def _unseen_categories(
    model: BaseModel, bridge: Bridge, unseen: ProbeSet, main_labels: list[str], max_items: int, seed: int,
) -> tuple[float | None, float, str]:
    """The unseen-categories control: (accuracy or None, its chance, a sentence for card.reason). Labels that
    match a main label are dropped with their files; every other file is scored (at most max_items, seeded) with
    the unseen set's question and up to five of its remaining labels as options. A file the adapter fails on
    counts as wrong."""
    from sense.bridges import five_options
    from sense.verify import _subsample

    taken = set().union(*(_label_keys(label) for label in main_labels))
    keep = [c for c, label in enumerate(unseen.labels) if not (_label_keys(label) & taken)]
    shared = [label for c, label in enumerate(unseen.labels) if c not in keep]
    items = [(Path(p), keep.index(c)) for p, c in unseen.items if c in keep]
    dropped = f" (dropped as already probed: {', '.join(shared)})" if shared else ""
    if len(keep) < 2:
        return None, 0.0, (
            f"Unseen categories not measured: {unseen.name} has {len(keep)} label{'' if len(keep) == 1 else 's'} "
            f"the probe set does not already have{dropped}; at least 2 are needed."
        )
    if len(items) < 4:
        return None, 0.0, (
            f"Unseen categories not measured: {unseen.name} has {len(items)} files in labels the probe set does not "
            f"already have{dropped}; at least 4 are needed."
        )
    labels = [unseen.labels[c] for c in keep]
    rng = np.random.default_rng([seed, 1])  # its own stream: the main questions stay as they were
    chosen = _subsample(np.arange(len(items)), max_items, rng)
    y = np.array([items[k][1] for k in chosen], dtype=np.int64)
    options, answers = five_options(y, len(labels), rng)
    prepared = _prepare_all(bridge, sorted({items[k][0] for k in chosen}))
    per_item = [prepared[items[k][0]] for k in chosen]
    score = _ask(model, unseen.question, labels, options, answers, per_item)
    chance = round(1 / min(5, len(labels)), 4)
    failed = sum(isinstance(inp, AdapterError) for inp in per_item)
    failures = f"; the adapter failed on {failed} of them, counted as wrong" if failed else ""
    source = unseen.source or unseen.name
    return score, chance, (
        f"Unseen categories: {score * 100:.1f}% on {len(chosen)} files of {len(labels)} categories the adapter "
        f"was not built around ({', '.join(labels)}; source: {source}){dropped}, against "
        f"{chance * 100:.1f}% chance{failures}."
    )


def _prepare_all(bridge: Bridge, paths: list[Path]) -> dict[Path, SenseInput | AdapterError]:
    """bridge.prepare on every file, an AdapterError kept in place of the input it could not make. Adapter bridges
    run in parallel processes; other bridges one file at a time."""

    def one(p: Path) -> SenseInput | AdapterError:
        try:
            return bridge.prepare(p)
        except AdapterError as err:
            return err

    if isinstance(bridge, AdapterBridge) and len(paths) > 1:
        with ThreadPoolExecutor(max_workers=min(PREPARE_WORKERS, len(paths))) as pool:
            return dict(zip(paths, pool.map(one, paths)))
    return {p: one(p) for p in paths}
