"""Agent-written adapters: the harness side. An LLM writes Python code `adapt(path)` that turns a file of any
modality into text (any base model reads it) or into images (for models that read images); this module runs that
code in a separate, locked-down process, wraps it as a bridge, and verifies the bridge on labelled probe files.

Public surface (do not rename):
  toolbox() -> list[str]
      what an adapter may use, checked at call time: the allow-list numpy, scipy, PIL, json, csv, math, struct,
      zipfile, gzip, io, re, xml.etree, statistics, plus trimesh, rdkit, soundfile, av, cv2, pandas and
      matplotlib when they are installed, plus "ffmpeg (via subprocess)" and "ffprobe (via subprocess)" when those
      binaries are on PATH (or in /opt/homebrew/bin or /usr/local/bin). It is for the prompt that asks for an
      adapter; run_adapter does not police imports.
  code_imports(code) -> list[str]
      the top-level modules the code imports (and "ffmpeg"/"ffprobe" when it names them), for a manifest
  AdapterError(reason, traceback="", *, cause="error")
      .reason (one line: what went wrong), .traceback (the tail of the traceback, or of the process's error output
      when there is none), .cause ("exception", "output", "timeout", "exit" or "verify"); str() gives both texts
  sandboxed() -> bool
      whether adapters run under macOS sandbox-exec here (checked once: the binary exists and can apply a profile)
  run_adapter(code, path, *, timeout=60) -> dict
      runs adapt(path) on one file in a separate Python process and returns {"text": str} or
      {"images": [PNG bytes, ...]}, each with "notes" (what the caps cut) and "seconds"
  AdapterBridge(*, kind, modality, model_name, code, manifest=None, slug=None, timeout=60)
      a contracts.Bridge for files: kind "text" or "image", name "adapter:<slug>"; prepare(path) runs the adapter
      (results cached per code and file sha256 within the process; errors too, but not timeouts) and returns a
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
               min_per_label=4, max_per_label=40, seed=0) -> ProbeSet
  verify_files(model, bridge, probes, request, *, max_items=None, seed=0) -> contracts.ReportCard

The adapter contract. The code must define adapt(path), called once per file with the file's absolute path as a
str, returning {"text": str} or {"images": [...]} (exactly one of those keys). Each image may be PNG or JPEG bytes
(any format PIL reads is accepted), a PIL image, a path to an image file (relative paths are read from the
adapter's working directory) or a uint8 numpy array; a single image not in a list is accepted too. The runner
normalises every image to PNG, at most MAX_IMAGE_SIDE (512) px on the long side, and keeps at most MAX_IMAGES (9);
text is cut to MAX_TEXT_CHARS (4000) characters including a note saying it was cut. Anything else (another return
value, an exception, a non-zero exit, a crash, a timeout) raises AdapterError with the reason and the tail of the
traceback, so the agent can show it to the LLM and ask for a repair.

The sandbox. run_adapter writes the code and a small runner script to a fresh temporary directory and runs
`sys.executable -I -B runner.py adapter.py <path>` there, in a new session (process group), with:
  - its own working directory (<tmp>/work), also its HOME and TMPDIR, so caches and stray files land there;
  - an environment without any variable whose name contains KEY, TOKEN, SECRET, PASSWORD, PASSWD, CREDENTIAL or
    AUTH, or starts with DATABRICKS_, OPENAI_, LINKUP_, ANTHROPIC_, AWS_, AZURE_, GOOGLE_, GH_ or GITHUB_;
  - HTTP_PROXY, HTTPS_PROXY and ALL_PROXY (and the lowercase forms) set to BLOCKED_PROXY, a closed local port
    (http://127.0.0.1:9), and NO_PROXY empty, so ordinary HTTP fails at once instead of hanging;
  - on macOS, when /usr/bin/sandbox-exec exists and works (sandboxed()), a sandbox profile that denies all
    network access, denies reading any file named .env* and the usual credential stores in the user's home
    (~/.ssh, ~/.aws, ~/.netrc, ~/.config/gh, ...), and denies writing anywhere but the temporary directory,
    /private/tmp, the per-user temporary folders (/private/var/folders) and /dev;
  - a wall-clock timeout, after which the whole process group is killed (it is killed after a normal exit too, so
    nothing the adapter started lives on).
Without a working sandbox-exec (another OS, or already inside a sandbox) only the environment and proxy measures
apply. Whatever the adapter prints, to
stdout or stderr, and whatever its child processes print, goes to a log file; the runner keeps the real stdout for
its one JSON result (images as base64).

Probe files. A bridge is verified on files whose labels come from a source independent of the adapter: a user's
examples folder, a public dataset, or files generated so that the label is true by construction (provenance
"user", "dataset" or "generated"). probe_folder reads a folder with one sub-folder per label: labels come from the
folder names ("_" and "-" read as spaces; folders starting with "." or "_" are ignored), files of any type directly
inside a label folder are used (hidden and empty files and sub-folders are skipped and counted), exact duplicates
by sha256 are collapsed (the same bytes under two labels are dropped everywhere and listed), at most max_per_label
files per label are kept (a seeded pick), every label needs min_per_label files (at least 3) and there must be 2
labels, and each label is split 60/20/20 (seeded; at least one dev and one test file per label). Every file is
copied to <config.home()>/probes/<name>/<sha256[:16]><extension> (extension lowercased) and the items point at the
copies, so no adapter ever sees a file name that gives the label away.

Verification. verify_files is sense.verify.verify for files: each held-out item (probes.test_idx, at most max_items;
None means 200 for an open-weight model and config.api_verify_items() otherwise, drawn with `seed`) becomes a
question with probes.question and up to five options (the true label and others, seeded, sense.bridges.five_options),
answered by model.choose with [bridge.prepare(file)]. Controls: "swapped" (the same question with the bridged input
of a held-out file of another label) and "no_input". Chance is 1/min(5, n_labels); the interval is the 95% Wilson
interval of sense.verify; card.dataset is probes.name, card.kind is bridge.kind and the verdict is "candidate". An
adapter that fails on a held-out file counts that question as wrong (the model is not asked); in the swapped control
a failed partner file is asked with no input. Failures are listed in card.reason. When the adapter fails on more than
half of the held-out files, verify_files raises AdapterError (cause "verify") instead of returning a card.
Adapter bridges are prepared in up to PREPARE_WORKERS processes at once.

Fake mode needs nothing special: FakeModel picks the option a text input names, so a tiny adapter that reads, for
example, a test file's first line verifies end to end, offline.
"""

from __future__ import annotations

import ast
import base64
import hashlib
import importlib.util
import io
import json
import os
import re
import shutil
import signal
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
SHEET_WIDTH = 1024  # the contact sheet of an image bridge is at most this wide
SHEET_GAP = 8
DEFAULT_TIMEOUT = 60.0
TAIL_CHARS = 2500  # how much of a traceback or error log an AdapterError keeps
PREPARE_WORKERS = 4
BLOCKED_PROXY = "http://127.0.0.1:9"  # the discard port, closed: a request through it is refused at once
SANDBOX_EXEC = "/usr/bin/sandbox-exec"
_CACHE_SIZE = 1024

_CORE_LIBRARIES = [
    "numpy", "scipy", "PIL", "json", "csv", "math", "struct", "zipfile", "gzip", "io", "re", "xml.etree", "statistics",
]
_OPTIONAL_LIBRARIES = ["trimesh", "rdkit", "soundfile", "av", "cv2", "pandas", "matplotlib"]
_BINARIES = ["ffmpeg", "ffprobe"]
_BINARY_DIRS = ["/opt/homebrew/bin", "/usr/local/bin"]

_SECRET_WORDS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "PASSWD", "CREDENTIAL", "AUTH")
_SECRET_PREFIXES = ("DATABRICKS_", "OPENAI_", "LINKUP_", "ANTHROPIC_", "AWS_", "AZURE_", "GOOGLE_", "GH_", "GITHUB_")
_PROXY_VARS = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")
_HOME_SECRETS = [  # relative to the user's home; never readable by an adapter under sandbox-exec
    ".ssh", ".aws", ".netrc", ".config/gh", ".git-credentials", ".docker", ".kube", ".config/gcloud",
    ".databrickscfg", ".azure", ".gnupg", "Library/Keychains",
]


# ----------------------------------------------------------------------------- toolbox
def toolbox() -> list[str]:
    """What an adapter may use: the allow-listed modules that can be imported here (checked now, without importing
    them), then "ffmpeg (via subprocess)" and "ffprobe (via subprocess)" when those binaries exist."""
    out = [name for name in _CORE_LIBRARIES + _OPTIONAL_LIBRARIES if _importable(name)]
    out += [f"{name} (via subprocess)" for name in _BINARIES if _binary(name)]
    return out


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
    "exit" or "verify"."""

    def __init__(self, reason: str, traceback: str = "", *, cause: str = "error"):
        super().__init__(reason)
        self.reason = reason
        self.traceback = traceback
        self.cause = cause

    def __str__(self) -> str:
        return self.reason if not self.traceback else f"{self.reason}\n{self.traceback.rstrip()}"

    def to_dict(self) -> dict[str, str]:
        return {"reason": self.reason, "traceback": self.traceback, "cause": self.cause}


def _tail(text: str, limit: int = TAIL_CHARS) -> str:
    text = text.rstrip()
    if len(text) <= limit:
        return text
    cut = text[-limit:]
    newline = cut.find("\n")
    return "..." + (cut[newline:] if 0 <= newline < 200 else cut)


# ----------------------------------------------------------------------------- the runner (child process)
_RUNNER = r'''
import base64, io, json, os, sys, traceback

MAX_TEXT, MAX_IMAGES, MAX_SIDE = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])


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


def _profile(tmp: Path) -> str:
    """macOS sandbox profile: no network, no secrets, writes only to temporary places."""
    home = Path.home()
    secrets = " ".join(f"(subpath {_sbpl(str(home / p))})" for p in _HOME_SECRETS)
    writable = " ".join(
        f"(subpath {_sbpl(p)})" for p in dict.fromkeys([os.path.realpath(tmp), "/private/tmp", "/private/var/folders"])
    )
    return (
        "(version 1)\n"
        "(allow default)\n"
        "(deny network*)\n"
        f'(deny file-read* (regex #"/\\.env[^/]*$") {secrets})\n'
        "(deny file-write*)\n"
        f'(allow file-write* {writable} (regex #"^/dev/"))\n'
    )


def _child_env(work: Path) -> dict[str, str]:
    env = {}
    for name, value in os.environ.items():
        upper = name.upper()
        if any(word in upper for word in _SECRET_WORDS) or upper.startswith(_SECRET_PREFIXES):
            continue
        env[name] = value
    for name in _PROXY_VARS:
        env[name] = BLOCKED_PROXY
    env["NO_PROXY"] = env["no_proxy"] = ""
    tmp = work / "tmp"
    tmp.mkdir(exist_ok=True)
    env.update(HOME=str(work), TMPDIR=str(tmp), MPLCONFIGDIR=str(work), XDG_CACHE_HOME=str(work / "cache"))
    path = env.get("PATH", os.defpath).split(os.pathsep)
    extra = [str(Path(b).parent) for b in map(_binary, _BINARIES) if b]
    env["PATH"] = os.pathsep.join(dict.fromkeys(path + extra))
    return env


@lru_cache(maxsize=1)
def sandboxed() -> bool:
    """Whether adapters run under sandbox-exec here: macOS, the binary exists, and it can apply a profile (it
    cannot inside another sandbox). Checked once per process."""
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


def _kill_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        pass


def run_adapter(code: str, path: Path, *, timeout: float = DEFAULT_TIMEOUT) -> dict:
    """Run adapt(path) from `code` on one file in a separate, sandboxed Python process (see the module docstring).

    Returns {"text": str, "notes": [...], "seconds": float} or {"images": [PNG bytes, ...], "notes": [...],
    "seconds": float}. Raises AdapterError when the adapter raises, returns anything else, exits non-zero, crashes
    or runs longer than `timeout` seconds; ValueError when `path` is not a file or `timeout` is not positive."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"{path} is not a file; give the path of the file to adapt.")
    if not timeout or timeout <= 0:
        raise ValueError(f"timeout must be a positive number of seconds, not {timeout!r}.")
    tmp = Path(tempfile.mkdtemp(prefix="sense-adapter-")).resolve()
    started = time.monotonic()
    try:
        work = tmp / "work"
        work.mkdir()
        code_file, runner = tmp / "adapter.py", tmp / "runner.py"
        code_file.write_text(code, encoding="utf-8")
        runner.write_text(_RUNNER, encoding="utf-8")
        cmd = [
            sys.executable, "-I", "-B", str(runner), str(code_file), str(path),
            str(MAX_TEXT_CHARS), str(MAX_IMAGES), str(MAX_IMAGE_SIDE),
        ]
        if sandboxed():
            cmd = [SANDBOX_EXEC, "-p", _profile(tmp), *cmd]
        log_path = tmp / "log.txt"
        with open(log_path, "wb") as log:
            proc = subprocess.Popen(
                cmd, cwd=work, env=_child_env(work), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=log,
                start_new_session=True,
            )
            try:
                stdout, _ = proc.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                _kill_group(proc)
                proc.communicate()
                raise AdapterError(
                    f"The adapter ran for longer than {timeout:g} s on {path.name} and was stopped.",
                    _log_tail(log_path, tmp),
                    cause="timeout",
                ) from None
            finally:
                _kill_group(proc)
        seconds = round(time.monotonic() - started, 3)
        return _result(stdout, proc.returncode, log_path, tmp, seconds)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _log_tail(log_path: Path, tmp: Path) -> str:
    try:
        with open(log_path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 4 * TAIL_CHARS))
            text = f.read().decode("utf-8", "replace")
    except OSError:
        return ""
    return _tail(_scrub(text, tmp))


def _scrub(text: str, tmp: Path) -> str:
    """Temporary paths out of a traceback: "<tmp>/adapter.py" reads "adapter.py"."""
    for prefix in {str(tmp), os.path.realpath(tmp)}:
        text = text.replace(prefix + os.sep + "work" + os.sep, "").replace(prefix + os.sep, "")
    return text


def _result(stdout: bytes, returncode: int, log_path: Path, tmp: Path, seconds: float) -> dict:
    try:
        payload = json.loads(stdout.decode("utf-8")) if stdout.strip() else None
    except (UnicodeDecodeError, json.JSONDecodeError):
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
        error = str(payload.get("error") or "unknown error")
        lead = "adapt(path) raised " if cause == "exception" else ""
        tail = _scrub(str(payload.get("traceback") or ""), tmp) or _log_tail(log_path, tmp)
        raise AdapterError(f"{lead}{_scrub(error, tmp)}", _tail(tail), cause=cause)
    notes = [str(n) for n in payload.get("notes", [])]
    if "text" in payload:
        return {"text": str(payload["text"]), "notes": notes, "seconds": seconds}
    return {"images": [base64.b64decode(b) for b in payload["images"]], "notes": notes, "seconds": seconds}


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
        per code and file sha256 within the process. Raises AdapterError (also cached, except timeouts)."""
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
                if err.cause == "timeout":
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
) -> ProbeSet:
    """A ProbeSet from a folder with one sub-folder of files per label (see the module docstring)."""
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"{root} is not a folder; give a folder with one sub-folder of example files per label.")
    if provenance not in ("user", "dataset", "generated"):
        raise ValueError(f'provenance is "user", "dataset" or "generated", not {provenance!r}.')
    if min_per_label < 3:
        raise ValueError(f"min_per_label is {min_per_label}, but each label needs at least 3 files (fit, dev, test).")
    if max_per_label < min_per_label:
        raise ValueError(f"max_per_label ({max_per_label}) is below min_per_label ({min_per_label}); raise it.")
    set_name = _slug(name or root.name)
    folders = _probe_label_folders(root)
    labels = [label for label, _ in folders]

    skipped = {"hidden": 0, "empty": 0, "subfolders": 0}
    copies: dict[str, list[tuple[int, Path]]] = {}
    for c, (_, directory) in enumerate(folders):
        for p in sorted(directory.iterdir()):
            if p.name.startswith("."):
                skipped["hidden"] += 1
            elif p.is_dir():
                skipped["subfolders"] += 1
            elif p.is_file():
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
    items: list[tuple[Path, int]] = []
    fit, dev, test = [], [], []
    for c, files in enumerate(kept):
        start = len(items)
        for p, digest in files:
            items.append((_copy_probe(p, digest, target), c))
        order = start + rng.permutation(len(files))
        n = len(files)
        n_dev = max(1, int(round(0.2 * n)))
        n_fit = min(int(round(0.6 * n)), n - n_dev - 1)
        fit += order[:n_fit].tolist()
        dev += order[n_fit : n_fit + n_dev].tolist()
        test += order[n_fit + n_dev :].tolist()

    counts = {labels[c]: len(files) for c, files in enumerate(kept)}
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
    if dest.is_file() and dest.stat().st_size == src.stat().st_size:
        return dest
    partial = target / f".{dest.name}.{os.getpid()}.{threading.get_ident()}.part"
    shutil.copyfile(src, partial)
    os.replace(partial, dest)
    return dest


# ----------------------------------------------------------------------------- verification
def verify_files(
    model: BaseModel,
    bridge: Bridge,
    probes: ProbeSet,
    request: FitRequest,
    *,
    max_items: int | None = None,
    seed: int = 0,
) -> ReportCard:
    """Score a file-based bridge on the held-out probe files against swapped-input and no-input controls."""
    from sense.bridges import five_options
    from sense.verify import _subsample, _swap_partners, _wilson

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
    test = _subsample(probes.test_idx, max_items, rng)
    if len(test) == 0:
        raise ValueError(f"Probe set {probes.name} has no held-out files to verify on.")
    y = probes.y
    options, answers = five_options(y[test], n_labels, rng)
    swap = _swap_partners(test, y, rng)

    prepared = _prepare_all(bridge, [probes.items[i][0] for i in sorted(set(test.tolist()) | set(swap.tolist()))])
    inputs = {i: prepared[probes.items[i][0]] for i in set(test.tolist()) | set(swap.tolist())}
    failed = [int(i) for i in test if isinstance(inputs[int(i)], AdapterError)]
    if len(failed) * 2 > len(test):
        first = inputs[failed[0]]
        raise AdapterError(
            f"The adapter failed on {len(failed)} of {len(test)} held-out files, more than half, so {bridge.name} "
            f"cannot be verified. First failure ({Path(probes.items[failed[0]][0]).name}): {first.reason}",
            first.traceback,
            cause="verify",
        )

    def ask(per_item: list[SenseInput | AdapterError | None]) -> float:
        """Accuracy over the questions; an AdapterError counts as wrong without asking, None asks with no input."""
        idx = [k for k, inp in enumerate(per_item) if not isinstance(inp, AdapterError)]
        items = [
            ChoiceItem(
                question=probes.question,
                options=[probes.labels[c] for c in options[k]],
                inputs=[] if per_item[k] is None else [per_item[k]],
            )
            for k in idx
        ]
        picks = np.full(len(per_item), -1, dtype=np.int64)
        if items:
            picks[idx] = model.choose(items)
        return round(float((picks == answers).mean()), 4)

    accuracy = ask([inputs[int(i)] for i in test])
    swapped = [inputs[int(j)] for j in swap]
    controls = {
        "swapped": ask([None if isinstance(inp, AdapterError) else inp for inp in swapped]),
        "no_input": ask([None for _ in test]),
    }
    reason = ""
    if failed:
        shown = "; ".join(f"{Path(probes.items[i][0]).name}: {inputs[i].reason}" for i in failed[:5])
        more = f"; and {len(failed) - 5} more" if len(failed) > 5 else ""
        reason = (
            f"The adapter failed on {len(failed)} of {len(test)} held-out files; those questions count as wrong. "
            f"{shown}{more}."
        )
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
        reason=reason,
        created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        interval=_wilson(accuracy, len(test)),
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
