"""Vet, then install, extra Python packages an adapter needs, into a toolbox kept apart from the main environment.

The agent asks for packages; plan() checks them and has no side effects; the user reads the plan and approves
it; install() installs exactly that pinned plan and nothing else. plan() never installs anything, and install()
refuses anything the user has not approved.

Public surface (do not rename):
  toolbox_dir() -> Path                 <sense.config.home()>/toolbox/site, the --target directory
  sys_path_entry() -> str | None        toolbox_dir() as a string when it exists, else None
  PackageCheck(name, version, verdict, reasons, facts)
      verdict "ok" | "caution" | "refuse"; reasons are plain sentences; facts (always all of these keys):
        summary, licence, home_page, project_urls, first_release, latest_release, latest_version, n_releases,
        wheel_for_this_platform, wheel_filename, wheel_url, wheel_sha256, yanked,
        vulnerabilities [{id, summary, severity, aliases}], similar_to, already_installed, requested, via
  InstallPlan(requested, packages, verdict, requirements, created, problems, target)
      requested [{name, version, why}]; packages [PackageCheck]; verdict is the worst of the packages' verdicts,
      and "refuse" when problems (plan-level reasons, such as a failed resolution) is not empty; requirements is
      the pinned, hashed requirements text install() will use; created is an ISO 8601 UTC time; target is
      {python, version, platform, macos}. to_dict() -> JSON-ready dict (with "summary"); from_dict(d) rebuilds
      one (for the approval round trip); summary() -> one sentence for the user.
  plan(requests, *, python=sys.executable) -> InstallPlan
      requests are [{"name": "open3d", "version": optional, "why": "..."}]
  install(plan, *, approved, python=sys.executable) -> {"installed": [{name, version, sha256}], "target",
      "seconds", "log_tail"}
  ask_llm_review(plan) -> str | None    a short opinion from agent.llm.ask_json; None in fake mode or on any
                                        failure. Advisory only: it is shown beside the verdict, never used as a gate.
  InstallRefused(PermissionError)       install() was not approved, or the plan is refused or inconsistent
  InstallFailed(RuntimeError)           uv is missing, timed out or failed; .log_tail holds its last output lines

How plan() works
  1. Each request name must match the PEP 508 name pattern, optionally with extras made of names ("pkg[extra]").
     URLs, paths, VCS links, options such as "--index-url x", markers and anything else are refused, and an
     invalid request never reaches uv. "version" is optional: a bare version ("1.2.3", meaning ==) or comparison
     clauses such as ">=1.2,<2".
  2. uv pip compile resolves the full dependency set from PyPI only, wheels only, with hashes:
       uv pip compile requirements.in --constraint constraints.txt --generate-hashes --only-binary :all:
         --python <python> --python-version <x.y.z> --python-platform <triple> --index-url https://pypi.org/simple
         --no-config --no-python-downloads --no-header --annotation-style line --color never
     with MACOSX_DEPLOYMENT_TARGET set to the running macOS version (uv's darwin triples otherwise assume an old
     macOS and skip newer wheels). The constraints pin every package already installed in the interpreter's
     environment and in the toolbox to its installed version, so the toolbox never holds a second copy of a
     package the main environment has, and approving one plan never changes what an earlier plan installed.
  3. Each resolved package already installed at that version (importlib.metadata, in the environment or the
     toolbox) is marked already_installed and is neither vetted nor installed. Every other package is vetted:
     PyPI's JSON API (https://pypi.org/pypi/<name>/<version>/json and https://pypi.org/pypi/<name>/json) for
     the summary, licence, project URLs, release history and this platform's best wheel with its sha256; OSV
     (POST https://api.osv.dev/v1/query) for known vulnerabilities of that exact version; and a typosquat check
     against a built-in list of popular names (Levenshtein distance 1 or 2; only 1 for names of 4 characters or
     fewer, where 2 edits make a different word, not a typo).
  4. If uv cannot resolve the request (for example because a package has no wheel for this platform), the plan
     is refused with uv's message, and each valid requested package is still vetted at its pinned or latest
     version so the user sees why.
  Rules. refuse: an invalid name; no wheel for this platform (installing an sdist would run its build code);
  PyPI cannot be reached or has no such release; the wheel's sha256 is not among the hashes uv resolved; a known
  vulnerability of that exact version rated critical or high, or one with no severity at all (OSV aliases such
  as PYSEC and GHSA records of one CVE are grouped and share their best known severity; reasons name the IDs).
  caution: first release under 90 days ago; fewer than 3 releases; no licence; no project URLs; a typosquat
  match; a medium or low vulnerability; a yanked release; release history or OSV unavailable. Otherwise ok.
  Severity comes from the records' own labels (GHSA's database_specific.severity), else a CVSS v3 base score.

How install() works
  It refuses unless approved is True (the boolean, not merely truthy), the plan is not refused (recomputed from
  its packages, not only read from plan.verdict), the plan was made for this interpreter, and plan.requirements
  is exactly the text implied by the vetted packages: "name==version" with the one sha256 of the vetted wheel.
  Then it runs, with a timeout and an environment holding no secrets:
       uv pip install --target <toolbox_dir()> --require-hashes --only-binary :all: --no-deps -r <file>
         --python <python> --index-url https://pypi.org/simple --no-config --no-python-downloads --color never
  checks every planned package is in the toolbox at its planned version, and appends one record per package to
  <toolbox_dir()>/../installed.json: {name, version, sha256, wheel_url, approved_at, requested_by "agent", why}.
  The main virtual environment is never touched: --target writes only into the toolbox.

Using the toolbox
  sense.adapters' runner puts sys_path_entry() at the end of the adapter's path with sys.path.append (so a module
  in the toolbox never shadows the standard library or the main site-packages), never site.addsitedir, so .pth
  files that wheels may carry are never executed; its sandbox may read that folder and nothing else under
  config.home(). sense.adapters.toolbox() lists the top-level modules of the packages installed.json records, so
  the adapter's author is told it may import them.

Network and processes
  Every HTTP request goes through _http (urllib, https only, to pypi.org and api.osv.dev only, User-Agent
  "sense-harness", no credentials, a timeout and a size cap) and every subprocess through _run; tests patch
  both. In fake mode (sense.config.fake(), checked on every call) the real _http and _run refuse to run, so a
  plan made in fake mode is refused and nothing is fetched or installed.
"""

from __future__ import annotations

import concurrent.futures
import inspect
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from sense import config

PYPI = "https://pypi.org"
PYPI_SIMPLE = "https://pypi.org/simple"
OSV_QUERY = "https://api.osv.dev/v1/query"
ALLOWED_HOSTS = frozenset({"pypi.org", "api.osv.dev"})
USER_AGENT = "sense-harness"

HTTP_TIMEOUT_S = 20.0
HTTP_MAX_BYTES = 64_000_000  # PyPI's project JSON for big projects runs to megabytes
COMPILE_TIMEOUT_S = 180.0
INSTALL_TIMEOUT_S = 900.0
PROBE_TIMEOUT_S = 30.0
VET_WORKERS = 8
MAX_REQUESTS = 10
MAX_OSV_PAGES = 5

NEW_PROJECT_DAYS = 90
MIN_RELEASES = 3

VERDICTS = ("ok", "caution", "refuse")
_RANK = {verdict: rank for rank, verdict in enumerate(VERDICTS)}

# Popular names for the typosquat check (PEP 503 normalised). Legitimate packages within two edits of a listed
# name (dash and dask, numba and numpy, ipython and biopython, scapy and scipy, ...) are listed too, so neither
# is flagged as a copy of the other.
POPULAR = frozenset("""
numpy scipy pandas torch torchvision torchaudio tensorflow keras jax jaxlib requests urllib3 httpx aiohttp
pillow matplotlib seaborn plotly dash dask trimesh rdkit open3d opencv-python opencv-python-headless
scikit-learn scikit-image transformers tokenizers datasets huggingface-hub safetensors accelerate diffusers
sentencepiece librosa soundfile pydub numba networkx sympy h5py pyyaml lxml beautifulsoup4 flask django
fastapi pydantic boto3 botocore setuptools certifi cryptography biopython astropy nibabel pydicom shapely
pyproj rasterio geopandas xarray netcdf4 imageio tqdm click rich pytest openai anthropic polars pyarrow
ipython ipykernel jupyter notebook httpie boto scapy pyaml simpy numpyro astroid dataset pyro-ppl
""".split())

_NAME = r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?"
_REQUEST_NAME = re.compile(rf"^({_NAME})(\[{_NAME}(?:,{_NAME})*\])?$")
_VERSION = r"\d+(?:\.\d+)*(?:\.\*)?(?:(?:a|b|rc)\d+)?(?:\.post\d+)?(?:\.dev\d+)?"
_BARE_VERSION = re.compile(rf"^{_VERSION}$")
_CLAUSE = re.compile(rf"^(==|!=|~=|>=|<=|>|<)\s*({_VERSION})$")
_PEP440 = re.compile(  # a public PEP 440 version, as the constraints file needs
    r"^v?(?:\d+!)?\d+(?:\.\d+)*(?:[-_.]?(?:a|b|c|rc|alpha|beta|pre|preview)[-_.]?\d*)?"
    r"(?:-\d+|[-_.]?(?:post|rev|r)[-_.]?\d*)?(?:[-_.]?dev[-_.]?\d*)?$", re.IGNORECASE)

_UV_REQ = re.compile(rf"^({_NAME})==([A-Za-z0-9.!+_-]+)(?:\s*;[^\\]*)?\s*\\?\s*$")
_UV_HASH = re.compile(r"^\s+--hash=sha256:([0-9a-f]{64})\s*\\?\s*$")
_UV_VIA = re.compile(r"^\s+#\s*via\s+(.*?)\s*$")
_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_REQ_TEXT_LINE = re.compile(rf"^({_NAME})==([A-Za-z0-9.!_-]+) \\$|^    --hash=sha256:[0-9a-f]{{64}}$")

# The only environment variables uv sees: none can carry an API key or redirect the index.
_ENV_KEEP = (
    "PATH", "HOME", "USER", "LOGNAME", "TMPDIR", "TEMP", "TMP", "LANG", "LC_ALL", "LC_CTYPE", "SYSTEMROOT",
    "APPDATA", "LOCALAPPDATA", "USERPROFILE", "XDG_CACHE_HOME", "UV_CACHE_DIR", "SSL_CERT_FILE", "SSL_CERT_DIR",
    "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "no_proxy", "all_proxy",
)


class InstallRefused(PermissionError):
    """install() will not run: not approved, a refused or inconsistent plan, or another interpreter's plan."""


class InstallFailed(RuntimeError):
    """uv is missing, timed out or failed; log_tail holds the last lines it printed."""

    def __init__(self, message: str, log_tail: str = ""):
        super().__init__(message)
        self.log_tail = log_tail


# ---------------------------------------------------------------- data


def _empty_facts() -> dict:
    return {
        "summary": None, "licence": None, "home_page": None, "project_urls": {}, "first_release": None,
        "latest_release": None, "latest_version": None, "n_releases": None, "wheel_for_this_platform": False,
        "wheel_filename": None, "wheel_url": None, "wheel_sha256": None, "yanked": False, "vulnerabilities": [],
        "similar_to": [], "already_installed": False, "requested": False, "via": [],
    }


@dataclass
class PackageCheck:
    name: str
    version: str | None
    verdict: str
    reasons: list[str] = field(default_factory=list)
    facts: dict = field(default_factory=_empty_facts)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "PackageCheck":
        facts = _empty_facts()
        facts.update(data.get("facts") or {})
        return cls(name=str(data["name"]), version=data.get("version"), verdict=str(data["verdict"]),
                   reasons=[str(r) for r in data.get("reasons") or []], facts=facts)


@dataclass
class InstallPlan:
    requested: list[dict]
    packages: list[PackageCheck]
    verdict: str
    requirements: str
    created: str
    problems: list[str] = field(default_factory=list)
    target: dict = field(default_factory=dict)

    def to_install(self) -> list[PackageCheck]:
        return [p for p in self.packages if not p.facts.get("already_installed") and p.facts.get("wheel_sha256")]

    def summary(self) -> str:
        """One sentence for the user: the verdict, what would be installed, and the first reasons to look at."""
        names = ", ".join(r["name"] for r in self.requested) or "nothing"
        new = len(self.to_install())
        have = sum(1 for p in self.packages if p.facts.get("already_installed"))
        sizes = f"{new} new package{'s' if new != 1 else ''} for the toolbox, {have} already installed"
        flagged = [f"{p.name}: {p.reasons[0]}" for p in self.packages if p.verdict == self.verdict and p.reasons]
        notes = (self.problems if self.verdict == "refuse" else []) + flagged
        detail = "; ".join(notes[:3]) + (f"; and {len(notes) - 3} more" if len(notes) > 3 else "")
        if self.verdict == "ok":
            return f"OK to install {names} ({sizes}): every check passed."
        if self.verdict == "caution":
            return f"Caution, {names} needs a look before approval ({sizes}): {detail}."
        return f"Refused, {names} cannot be installed: {detail}."

    def to_dict(self) -> dict:
        return {
            "requested": [dict(r) for r in self.requested],
            "packages": [p.to_dict() for p in self.packages],
            "verdict": self.verdict,
            "requirements": self.requirements,
            "created": self.created,
            "problems": list(self.problems),
            "target": dict(self.target),
            "summary": self.summary(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "InstallPlan":
        """Rebuild a plan from to_dict(). install() trusts the vetted facts, so keep plans server-side."""
        return cls(
            requested=[dict(r) for r in data.get("requested") or []],
            packages=[PackageCheck.from_dict(p) for p in data.get("packages") or []],
            verdict=str(data["verdict"]),
            requirements=str(data.get("requirements") or ""),
            created=str(data.get("created") or ""),
            problems=[str(p) for p in data.get("problems") or []],
            target=dict(data.get("target") or {}),
        )


# ---------------------------------------------------------------- paths


def toolbox_dir() -> Path:
    return config.home() / "toolbox" / "site"


def sys_path_entry() -> str | None:
    """The toolbox directory for the end of an adapter's sys.path (sys.path.append, never site.addsitedir: .pth
    files must not run), when it exists."""
    path = toolbox_dir()
    return str(path) if path.is_dir() else None


def _ledger() -> Path:
    return toolbox_dir().parent / "installed.json"


# ---------------------------------------------------------------- helpers patched by tests


class _SameHostRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _check_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _check_url(url: str) -> None:
    parts = urllib.parse.urlsplit(url)
    if parts.scheme != "https" or parts.hostname not in ALLOWED_HOSTS or parts.username or parts.password:
        raise ValueError(f"the installer only talks to https://pypi.org and https://api.osv.dev, not {url!r}")


def _http(method: str, url: str, *, body: dict | None = None, timeout: float = HTTP_TIMEOUT_S) -> tuple[int, dict | None]:
    """One JSON request: (status, parsed body or None). Raises on network failure; HTTP errors return their status."""
    if config.fake():
        raise ConnectionError("fake mode has no network")
    _check_url(url)
    data = None if body is None else json.dumps(body).encode()
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.build_opener(_SameHostRedirects).open(request, timeout=timeout) as response:
            raw = response.read(HTTP_MAX_BYTES + 1)
            status = response.status
    except urllib.error.HTTPError as exc:
        exc.close()
        return exc.code, None
    if len(raw) > HTTP_MAX_BYTES:
        raise ValueError(f"{url} answered with more than {HTTP_MAX_BYTES} bytes")
    parsed = json.loads(raw) if raw.strip() else None
    return status, parsed if isinstance(parsed, dict) else None


def _run(cmd: list[str], *, timeout: float, env: dict, cwd: str | None = None) -> subprocess.CompletedProcess:
    """Run one command, capturing text output. Raises subprocess.TimeoutExpired on timeout."""
    if config.fake():
        raise RuntimeError("fake mode runs no installer subprocess")
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env, cwd=cwd, check=False,
                          stdin=subprocess.DEVNULL)


def _find_uv() -> str | None:
    found = shutil.which("uv")
    if found:
        return found
    for candidate in (Path.home() / ".local" / "bin" / "uv", Path.home() / ".cargo" / "bin" / "uv"):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------- the interpreter


def _describe_interpreter() -> dict:
    """Self-contained (it is also sent to a foreign interpreter with python -I -c)."""
    import platform
    import sys
    import sysconfig
    from importlib import metadata

    installed = {}
    try:
        for dist in metadata.distributions():
            name = dist.metadata["Name"]
            if name and dist.version:
                installed.setdefault(name, dist.version)
    except Exception:
        pass
    try:
        from packaging import tags

        tag_list = [str(tag) for tag in tags.sys_tags()]
    except Exception:
        tag_list = None
    return {
        "version": platform.python_version(),
        "implementation": sys.implementation.name,
        "abi": f"cp{sys.version_info[0]}{sys.version_info[1]}",
        "machine": platform.machine(),
        "system": sys.platform,
        "mac": platform.mac_ver()[0],
        "libc": list(platform.libc_ver()) if sys.platform.startswith("linux") else ["", ""],
        "sysconfig_platform": sysconfig.get_platform(),
        "tags": tag_list,
        "installed": installed,
    }


def _fallback_tags(info: dict) -> list[str]:
    plat = re.sub(r"[.-]", "_", info["sysconfig_platform"])
    abi = info["abi"]
    return [f"{abi}-{abi}-{plat}", f"{abi}-abi3-{plat}", f"{abi}-none-{plat}", f"py3-none-{plat}",
            f"{abi}-none-any", "py3-none-any"]


def _uv_platform(info: dict) -> str | None:
    arch = {"arm64": "aarch64", "aarch64": "aarch64", "x86_64": "x86_64", "amd64": "x86_64"}.get(
        str(info.get("machine", "")).lower())
    system = str(info.get("system", ""))
    if not arch:
        return None
    if system == "darwin":
        return f"{arch}-apple-darwin"
    if system.startswith("linux"):
        return f"{arch}-unknown-linux-{'gnu' if (info.get('libc') or [''])[0] == 'glibc' else 'musl'}"
    if system == "win32":
        return f"{arch}-pc-windows-msvc"
    return None


def _macos_target(info: dict) -> str | None:
    if info.get("system") != "darwin" or not info.get("mac"):
        return None
    parts = str(info["mac"]).split(".")
    if not parts[0].isdigit():
        return None
    return f"{parts[0]}.0" if int(parts[0]) >= 11 else ".".join(parts[:2])


def _target(python: str) -> dict:
    """What uv and the wheel check need about the interpreter: version, platform, wheel tags, installed packages."""
    if os.path.abspath(python) == os.path.abspath(sys.executable):
        info = _describe_interpreter()
    else:
        snippet = textwrap.dedent(inspect.getsource(_describe_interpreter))
        snippet += "\nimport json\nprint(json.dumps(_describe_interpreter()))\n"
        result = _run([python, "-I", "-c", snippet], timeout=PROBE_TIMEOUT_S, env=_subprocess_env())
        if result.returncode != 0:
            raise RuntimeError(f"could not query the interpreter {python}: {_tail(result.stderr, 3)}")
        info = json.loads(result.stdout)
    return {
        "python": python,
        "version": info["version"],
        "platform": _uv_platform(info),
        "macos": _macos_target(info),
        "tags": info.get("tags") or _fallback_tags(info),
        "installed": {_normalise(name): version for name, version in (info.get("installed") or {}).items()},
    }


def _toolbox_installed() -> dict[str, str]:
    from importlib import metadata

    path = toolbox_dir()
    if not path.is_dir():
        return {}
    found = {}
    for dist in metadata.distributions(path=[str(path)]):
        name = dist.metadata["Name"]
        if name and dist.version:
            found.setdefault(_normalise(name), dist.version)
    return found


def _subprocess_env(macos: str | None = None) -> dict:
    env = {key: os.environ[key] for key in _ENV_KEEP if key in os.environ}
    env["NO_COLOR"] = "1"
    if macos:
        env["MACOSX_DEPLOYMENT_TARGET"] = macos
    return env


# ---------------------------------------------------------------- small pure helpers


def _normalise(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _public(version: str) -> str:
    return str(version).split("+", 1)[0].strip().lower()


def _uv_message(stderr: str, limit: int = 600) -> str:
    """uv's error as one line, without its box-drawing decoration."""
    text = re.sub(r"[\u2500-\u257f\u25b6\u00d7]+", " ", _ANSI.sub("", stderr or ""))
    return " ".join(text.split())[:limit]


def _tail(text: str, lines: int = 20, chars: int = 4000) -> str:
    cleaned = _ANSI.sub("", text or "").strip().splitlines()
    return "\n".join(cleaned[-lines:])[-chars:]


def _clean_text(value, limit: int) -> str | None:
    if not isinstance(value, str):
        return None
    value = re.sub(r"[\x00-\x1f\x7f]+", " ", value).strip()
    return value[:limit] if value else None


def _levenshtein(a: str, b: str) -> int:
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def _similar_to(name: str) -> list[str]:
    name = _normalise(name)
    if name in POPULAR:
        return []
    return sorted(p for p in POPULAR if 1 <= _levenshtein(name, p) <= (1 if len(p) <= 4 else 2))


def _worst(verdicts) -> str:
    return max(verdicts, key=lambda v: _RANK[v], default="ok")


def _wheel_tags(filename: str) -> set[str]:
    parts = filename[:-4].split("-")
    if len(parts) < 5:
        return set()
    pys, abis, plats = parts[-3].split("."), parts[-2].split("."), parts[-1].split(".")
    return {f"{p}-{a}-{pl}" for p in pys for a in abis for pl in plats}


def _best_wheel(files: list, tags: list[str]) -> dict | None:
    rank = {tag: i for i, tag in enumerate(tags)}
    best, best_rank = None, None
    for f in files:
        filename = str(f.get("filename", ""))
        if f.get("packagetype") != "bdist_wheel" or not filename.endswith(".whl"):
            continue
        ranks = [rank[t] for t in _wheel_tags(filename) if t in rank]
        sha = (f.get("digests") or {}).get("sha256")
        if ranks and isinstance(sha, str) and re.fullmatch(r"[0-9a-f]{64}", sha):
            if best_rank is None or min(ranks) < best_rank:
                best, best_rank = f, min(ranks)
    return best


def _parse_time(value) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _licence(info: dict) -> str | None:
    for key in ("license_expression", "license"):
        text = info.get(key)
        if isinstance(text, str) and text.strip() and text.strip().upper() not in ("UNKNOWN", "NONE"):
            return _clean_text(text.strip().splitlines()[0], 100)
    for classifier in info.get("classifiers") or []:
        if isinstance(classifier, str) and classifier.startswith("License ::"):
            label = classifier.split("::")[-1].strip()
            if label and label not in ("OSI Approved", "Other/Proprietary License"):
                return label[:100]
    return None


# ---------------------------------------------------------------- vulnerabilities

_SEV_RANK = {"low": 1, "medium": 2, "high": 3, "critical": 4}
_SEV_LABEL = {"LOW": "low", "MODERATE": "medium", "MEDIUM": "medium", "HIGH": "high", "CRITICAL": "critical"}


def _roundup(value: float) -> float:
    as_int = round(value * 100000)
    return as_int / 100000.0 if as_int % 10000 == 0 else (as_int // 10000 + 1) / 10.0


def _cvss3_score(vector: str) -> float | None:
    """CVSS v3.x base score from a vector string, or None when it is not a complete v3 vector."""
    if not vector.startswith("CVSS:3."):
        return None
    metrics = dict(part.split(":", 1) for part in vector.split("/")[1:] if ":" in part)
    try:
        scope_changed = metrics["S"] == "C"
        av = {"N": 0.85, "A": 0.62, "L": 0.55, "P": 0.2}[metrics["AV"]]
        ac = {"L": 0.77, "H": 0.44}[metrics["AC"]]
        pr = {"N": 0.85, "L": 0.68 if scope_changed else 0.62, "H": 0.5 if scope_changed else 0.27}[metrics["PR"]]
        ui = {"N": 0.85, "R": 0.62}[metrics["UI"]]
        cia = [{"H": 0.56, "L": 0.22, "N": 0.0}[metrics[k]] for k in ("C", "I", "A")]
    except KeyError:
        return None
    iss = 1 - (1 - cia[0]) * (1 - cia[1]) * (1 - cia[2])
    impact = 7.52 * (iss - 0.029) - 3.25 * (iss - 0.02) ** 15 if scope_changed else 6.42 * iss
    exploitability = 8.22 * av * ac * pr * ui
    if impact <= 0:
        return 0.0
    total = 1.08 * (impact + exploitability) if scope_changed else impact + exploitability
    return _roundup(min(total, 10.0))


def _score_label(score: float) -> str | None:
    if score >= 9.0:
        return "critical"
    if score >= 7.0:
        return "high"
    if score >= 4.0:
        return "medium"
    return "low" if score > 0 else None


def _vuln_severity(vuln: dict) -> str | None:
    labels = []
    for holder in [vuln.get("database_specific")] + [
            part for a in vuln.get("affected") or [] if isinstance(a, dict)
            for part in (a.get("ecosystem_specific"), a.get("database_specific"))]:
        if isinstance(holder, dict) and isinstance(holder.get("severity"), str):
            label = _SEV_LABEL.get(holder["severity"].strip().upper())
            if label:
                labels.append(label)
    for entry in vuln.get("severity") or []:
        if isinstance(entry, dict) and isinstance(entry.get("score"), str):
            score = _cvss3_score(entry["score"])
            if score is not None and _score_label(score):
                labels.append(_score_label(score))
    return max(labels, key=_SEV_RANK.get) if labels else None


def _group_vulns(vulns: list[dict]) -> list[dict]:
    """Group records of one issue (shared IDs or aliases); each group takes its members' best known severity."""
    vulns = [v for v in vulns if isinstance(v, dict) and isinstance(v.get("id"), str) and not v.get("withdrawn")]
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for v in vulns:
        for alias in v.get("aliases") or []:
            if isinstance(alias, str):
                parent[find(alias)] = find(v["id"])
    groups: dict[str, list[dict]] = {}
    for v in vulns:
        groups.setdefault(find(v["id"]), []).append(v)
    out = []
    for members in groups.values():
        ids = sorted({m["id"] for m in members})
        primary = next((i for prefix in ("GHSA-", "PYSEC-") for i in ids if i.startswith(prefix)), ids[0])
        aliases = sorted({a for m in members for a in (m.get("aliases") or []) if isinstance(a, str)} | set(ids))
        aliases.remove(primary)
        summary = next((s for m in members for s in [_clean_text(m.get("summary"), 200)] if s), None)
        if summary is None:
            summary = next((s for m in members for s in [_clean_text(m.get("details"), 200)] if s), "")
        labels = [label for m in members for label in [_vuln_severity(m)] if label]
        severity = max(labels, key=_SEV_RANK.get) if labels else None
        out.append({"id": primary, "summary": summary, "severity": severity, "aliases": aliases})
    return sorted(out, key=lambda g: (-_SEV_RANK.get(g["severity"], 5), g["id"]))


def _osv(name: str, version: str) -> list[dict]:
    body: dict = {"package": {"name": name, "ecosystem": "PyPI"}, "version": version}
    vulns: list[dict] = []
    for _ in range(MAX_OSV_PAGES):
        status, data = _http("POST", OSV_QUERY, body=body)
        if status != 200 or data is None:
            raise ConnectionError(f"OSV answered HTTP {status}")
        vulns += [v for v in data.get("vulns") or [] if isinstance(v, dict)]
        token = data.get("next_page_token")
        if not token:
            return vulns
        body = {**body, "page_token": token}
    return vulns


# ---------------------------------------------------------------- vetting


def _installed_check(name: str, version: str, where: str, *, requested: bool, via: list[str]) -> PackageCheck:
    facts = _empty_facts()
    facts.update(already_installed=True, requested=requested, via=via)
    return PackageCheck(name, version, "ok", [f"already installed in the {where} at {version}; not vetted or reinstalled"],
                        facts)


def _vet(name: str, version: str | None, *, target: dict, uv_hashes: set[str] | None, requested: bool,
         via: list[str]) -> PackageCheck:
    facts = _empty_facts()
    facts.update(requested=requested, via=via, similar_to=_similar_to(name))
    refuse: list[str] = []
    caution: list[str] = []
    quoted = urllib.parse.quote(name, safe="")

    project, project_status = None, None
    try:
        project_status, project = _http("GET", f"{PYPI}/pypi/{quoted}/json")
        if project_status != 200:
            project = None
    except Exception:
        project = None
    if facts["similar_to"]:
        caution.append(f"name is within two edits of the popular {', '.join(facts['similar_to'])} (possible typosquat)")
    if project is not None:
        info = project.get("info") or {}
        facts["latest_version"] = _clean_text(info.get("version"), 60)
        if version is None:
            version = facts["latest_version"]
        dates, n_releases = [], 0
        for files in (project.get("releases") or {}).values():
            stamps = [t for f in files or [] if isinstance(f, dict)
                      for t in [_parse_time(f.get("upload_time_iso_8601") or f.get("upload_time"))] if t]
            if stamps:
                n_releases += 1
                dates += stamps
        if dates:
            first, latest = min(dates), max(dates)
            facts.update(first_release=first.isoformat(), latest_release=latest.isoformat(), n_releases=n_releases)
            age = (_now() - first).days
            if age < NEW_PROJECT_DAYS:
                caution.append(f"first release on PyPI was {age} days ago (under {NEW_PROJECT_DAYS})")
            if n_releases < MIN_RELEASES:
                caution.append(f"only {n_releases} release{'s' if n_releases != 1 else ''} on PyPI "
                               f"(fewer than {MIN_RELEASES})")
        else:
            caution.append("PyPI shows no release history for it")
    elif project_status != 404:
        caution.append("its release history on PyPI could not be read")
    if version is None:
        refuse.append(f"PyPI has no project named {name}" if project_status == 404 else
                      "PyPI could not be reached to find its latest version")
        return PackageCheck(name, None, "refuse", refuse + caution, facts)

    try:
        status, release = _http("GET", f"{PYPI}/pypi/{quoted}/{urllib.parse.quote(version, safe='')}/json")
    except Exception as exc:
        status, release = None, None
        refuse.append(f"PyPI could not be reached to vet it ({type(exc).__name__})")
    if status == 404:
        refuse.append(f"PyPI has no release {name} {version}")
    elif status is not None and (status != 200 or release is None):
        refuse.append(f"PyPI answered HTTP {status} for {name} {version}")
    if release is not None and status == 200:
        info = release.get("info") or {}
        urls = info.get("project_urls") if isinstance(info.get("project_urls"), dict) else {}
        facts["project_urls"] = {str(k)[:60]: str(v)[:300] for k, v in list(urls.items())[:12] if isinstance(v, str)}
        facts["home_page"] = _clean_text(info.get("home_page"), 300) or next(iter(facts["project_urls"].values()), None)
        facts["summary"] = _clean_text(info.get("summary"), 300)
        facts["licence"] = _licence(info)
        if not facts["licence"]:
            caution.append("no licence is declared")
        if not facts["project_urls"] and not facts["home_page"]:
            caution.append("no project URLs (no homepage or source repository)")
        if info.get("yanked"):
            facts["yanked"] = True
            reason = _clean_text(info.get("yanked_reason"), 200)
            caution.append(f"this release was yanked from PyPI{': ' + reason if reason else ''}")
        files = [f for f in release.get("urls") or [] if isinstance(f, dict)]
        wheel = _best_wheel(files, target["tags"])
        if wheel is None:
            kinds = sorted({f.get("packagetype", "?") for f in files})
            have = "only a source distribution" if kinds == ["sdist"] else (
                "wheels for other platforms only" if "bdist_wheel" in kinds else "no files")
            refuse.append(f"no wheel for this platform ({target.get('platform') or 'this interpreter'}); PyPI has "
                          f"{have}, and building from source would run the package's own build code")
        else:
            sha = wheel["digests"]["sha256"]
            facts.update(wheel_for_this_platform=True, wheel_filename=_clean_text(wheel["filename"], 200),
                         wheel_url=_clean_text(wheel.get("url"), 400), wheel_sha256=sha)
            if uv_hashes is not None and sha not in uv_hashes:
                refuse.append(f"the wheel PyPI lists for this platform ({wheel['filename']}) is not among the files "
                              "the resolver saw")

        try:
            groups = _group_vulns(_osv(name, version))
        except Exception:
            groups = _group_vulns([v for v in release.get("vulnerabilities") or [] if isinstance(v, dict)])
            caution.append("OSV could not be reached; only PyPI's own vulnerability list was checked")
        facts["vulnerabilities"] = groups
        for g in groups:
            ids = g["id"] + (f" ({', '.join(g['aliases'])})" if g["aliases"] else "")
            if g["severity"] in ("critical", "high"):
                refuse.append(f"known {g['severity']} vulnerability in {name} {version}: {ids}")
            elif g["severity"] is None:
                refuse.append(f"known vulnerability with no severity score in {name} {version}: {ids}")
            else:
                caution.append(f"known {g['severity']} vulnerability in {name} {version}: {ids}")

    verdict = "refuse" if refuse else "caution" if caution else "ok"
    return PackageCheck(name, version, verdict, refuse + caution, facts)


# ---------------------------------------------------------------- requests and resolution


def _parse_request(item) -> tuple[dict, str | None, str | None]:
    """(sanitised request record, requirement line for uv or None, reason it is invalid or None).

    The record's "name" is the project name without extras when the request is valid."""
    if not isinstance(item, dict):
        return {"name": _clean_text(repr(item), 100) or "?", "version": None, "why": None}, None, \
            "a request must be an object with a name"
    raw_name = item.get("name")
    raw_version = item.get("version")
    record = {
        "name": (_clean_text(raw_name, 100) if isinstance(raw_name, str) else _clean_text(repr(raw_name), 100)) or "?",
        "version": _clean_text(raw_version, 60) if isinstance(raw_version, str) else None,
        "why": _clean_text(item.get("why"), 300),
    }
    match = _REQUEST_NAME.fullmatch(raw_name) if isinstance(raw_name, str) and len(raw_name) <= 100 else None
    if not match:
        return record, None, ("not a valid package name: only a PyPI project name (letters, digits, '.', '_', '-', "
                              "optionally with [extras]) is accepted, never a URL, path, VCS link or option")
    record["name"] = match[1]
    if raw_version is None or (isinstance(raw_version, str) and not raw_version.strip()):
        return record, raw_name, None
    if not isinstance(raw_version, str) or len(raw_version) > 60:
        return record, None, "not a valid version"
    version = raw_version.strip()
    if _BARE_VERSION.fullmatch(version) and "*" not in version:
        return record, f"{raw_name}=={version}", None
    clauses = [c.strip() for c in version.split(",")]
    if all(_CLAUSE.fullmatch(c) for c in clauses):
        return record, raw_name + ",".join(re.sub(r"\s+", "", c) for c in clauses), None
    return record, None, "not a valid version: use a version such as 1.2.3 or clauses such as >=1.2,<2"


def _parse_compiled(text: str) -> tuple[list[dict], list[str]]:
    """uv's requirements output -> [{name, version, hashes, via}], plus any lines it could not read."""
    entries: list[dict] = []
    odd: list[str] = []
    for line in text.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        m = _UV_REQ.match(line)
        if m:
            entries.append({"name": _normalise(m[1]), "version": _public(m[2]), "hashes": set(), "via": []})
            continue
        h = _UV_HASH.match(line)
        if h and entries:
            entries[-1]["hashes"].add(h[1])
            continue
        v = _UV_VIA.match(line)
        if v and entries:
            via = [s.strip() for s in v[1].split(",") if s.strip()]
            entries[-1]["via"] = ["requested" if s.startswith("-r ") else _normalise(s) for s in via]
            continue
        odd.append(line.strip()[:120])
    return entries, odd


def _requirements_text(packages: list[PackageCheck]) -> str:
    lines = []
    for p in sorted(packages, key=lambda p: p.name):
        if p.facts.get("already_installed") or not p.facts.get("wheel_sha256") or not p.version:
            continue
        lines += [f"{p.name}=={p.version} \\", f"    --hash=sha256:{p.facts['wheel_sha256']}"]
    return "\n".join(lines) + ("\n" if lines else "")


def _constraints(installed: dict[str, str]) -> str:
    lines = []
    for name, version in sorted(installed.items()):
        public = _public(version)
        if re.fullmatch(_NAME, name) and _PEP440.fullmatch(public):
            lines.append(f"{name}=={public}")
    return "\n".join(lines) + "\n"


def _compile(lines: list[str], target: dict, installed: dict[str, str]) -> tuple[list[dict] | None, str | None]:
    uv = _find_uv()
    if uv is None:
        return None, "uv was not found (looked on PATH, in ~/.local/bin and ~/.cargo/bin)"
    with tempfile.TemporaryDirectory(prefix="sense-installer-") as tmp:
        (Path(tmp) / "requirements.in").write_text("\n".join(lines) + "\n")
        (Path(tmp) / "constraints.txt").write_text(_constraints(installed))
        cmd = [uv, "pip", "compile", "requirements.in", "--constraint", "constraints.txt", "--generate-hashes",
               "--only-binary", ":all:", "--python", target["python"], "--python-version", target["version"]]
        if target.get("platform"):
            cmd += ["--python-platform", target["platform"]]
        cmd += ["--index-url", PYPI_SIMPLE, "--no-config", "--no-python-downloads", "--no-header",
                "--annotation-style", "line", "--color", "never"]
        try:
            result = _run(cmd, timeout=COMPILE_TIMEOUT_S, env=_subprocess_env(target.get("macos")), cwd=tmp)
        except subprocess.TimeoutExpired:
            return None, f"uv took longer than {COMPILE_TIMEOUT_S:.0f} s to resolve the request"
        except Exception as exc:
            return None, f"uv could not be run: {exc}"
    if result.returncode != 0:
        message = _uv_message(result.stderr) or f"exit {result.returncode}"
        held = sorted(f"{n}=={_public(v)}" for n, v in installed.items()
                      if re.search(rf"\b{re.escape(n)}==", message) and f"{n}=={_public(v)}" in message)
        if held:
            message += (f" ({', '.join(held)} {'is' if len(held) == 1 else 'are'} already installed and held at "
                        "that version: the toolbox never replaces an installed package)")
        return None, "uv could not resolve the request: " + message
    entries, odd = _parse_compiled(result.stdout)
    if odd:
        return None, f"uv's resolution had lines that are not pinned, hashed requirements: {odd[:3]}"
    if not entries:
        return None, "uv resolved nothing"
    return entries, None


def plan(requests: list[dict], *, python: str = sys.executable) -> InstallPlan:
    """Check requested packages and their dependencies. No side effects: nothing is installed or written."""
    created = _now().isoformat()
    problems: list[str] = []
    checks: list[PackageCheck] = []
    records: list[dict] = []
    lines: list[str] = []
    valid: dict[str, dict] = {}  # normalised name -> request record
    if not isinstance(requests, list) or not requests:
        problems.append("no packages were requested")
        requests = []
    if len(requests) > MAX_REQUESTS:
        problems.append(f"at most {MAX_REQUESTS} packages can be requested at once")
        requests = requests[:MAX_REQUESTS]
    for item in requests:
        record, line, invalid = _parse_request(item)
        records.append(record)
        if invalid:
            facts = _empty_facts()
            facts["requested"] = True
            checks.append(PackageCheck(record["name"], record["version"], "refuse", [invalid], facts))
        else:
            if _normalise(record["name"]) not in valid:
                valid[_normalise(record["name"])] = record
                lines.append(line)

    try:
        target = _target(python)
    except Exception as exc:
        problems.append(f"could not describe the interpreter {python}: {exc}")
        target = None
    if target is None or not valid:
        return _finish(records, checks, problems, created, target, python)

    in_env = dict(target["installed"])
    in_toolbox = _toolbox_installed()
    entries, failure = _compile(lines, target, {**in_toolbox, **in_env})
    if failure:
        problems.append(failure)
        entries = []
        for name, record in valid.items():
            pin = record["version"] if record["version"] and _BARE_VERSION.fullmatch(record["version"]) else None
            entries.append({"name": name, "version": pin, "hashes": None, "via": ["requested"]})

    def check(entry: dict) -> PackageCheck:
        name, version, requested = entry["name"], entry["version"], entry["name"] in valid
        for where, have in (("main environment", in_env), ("toolbox", in_toolbox)):
            if version and name in have and _public(have[name]) == version:
                return _installed_check(name, version, where, requested=requested, via=entry["via"])
        if version is None:
            for where, have in (("main environment", in_env), ("toolbox", in_toolbox)):
                if name in have:
                    return _installed_check(name, _public(have[name]), where, requested=requested, via=entry["via"])
        return _vet(name, version, target=target, uv_hashes=entry["hashes"], requested=requested, via=entry["via"])

    with concurrent.futures.ThreadPoolExecutor(max_workers=VET_WORKERS) as pool:
        vetted = list(pool.map(check, entries))
    vetted.sort(key=lambda p: (not p.facts["requested"], p.name))
    return _finish(records, checks + vetted, problems, created, target, python)


def _finish(records, checks, problems, created, target, python) -> InstallPlan:
    verdict = _worst([c.verdict for c in checks] + (["refuse"] if problems else []))
    return InstallPlan(
        requested=records,
        packages=checks,
        verdict=verdict,
        requirements=_requirements_text(checks),
        created=created,
        problems=problems,
        target={"python": python, "version": target["version"], "platform": target["platform"],
                "macos": target["macos"]} if target else {"python": python},
    )


# ---------------------------------------------------------------- install


def install(plan: InstallPlan, *, approved: bool, python: str = sys.executable) -> dict:
    """Install exactly the approved, pinned plan into toolbox_dir(). Raises InstallRefused or InstallFailed."""
    if approved is not True:
        raise InstallRefused("the user has not approved this plan")
    if any(v not in VERDICTS for v in [plan.verdict] + [p.verdict for p in plan.packages]):
        raise InstallRefused("the plan has an unknown verdict")
    verdict = _worst([p.verdict for p in plan.packages] + (["refuse"] if plan.problems else []) + [plan.verdict])
    if verdict == "refuse":
        raise InstallRefused("the plan is refused: " + plan.summary())
    if os.path.abspath(str(plan.target.get("python", ""))) != os.path.abspath(python):
        raise InstallRefused(f"the plan was made for {plan.target.get('python')!r}, not {python!r}")
    if plan.requirements != _requirements_text(plan.packages):
        raise InstallRefused("the requirements do not match the vetted packages")
    for line in plan.requirements.splitlines():
        if not _REQ_TEXT_LINE.fullmatch(line):
            raise InstallRefused(f"unexpected requirements line {line[:80]!r}")
    target = toolbox_dir()
    packages = plan.to_install()
    if not packages:
        return {"installed": [], "target": str(target), "seconds": 0.0, "log_tail": "nothing to install"}
    uv = _find_uv()
    if uv is None:
        raise InstallFailed("uv was not found (looked on PATH, in ~/.local/bin and ~/.cargo/bin)")
    target.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="sense-installer-") as tmp:
        req = Path(tmp) / "requirements.txt"
        req.write_text(plan.requirements)
        cmd = [uv, "pip", "install", "--target", str(target), "--require-hashes", "--only-binary", ":all:",
               "--no-deps", "-r", str(req), "--python", python, "--index-url", PYPI_SIMPLE, "--no-config",
               "--no-python-downloads", "--color", "never"]
        try:
            result = _run(cmd, timeout=INSTALL_TIMEOUT_S, env=_subprocess_env(plan.target.get("macos")), cwd=tmp)
        except subprocess.TimeoutExpired as exc:
            out = (exc.stdout or b"").decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            raise InstallFailed(f"uv took longer than {INSTALL_TIMEOUT_S:.0f} s to install", _tail(out)) from None
    seconds = round(time.monotonic() - started, 2)
    log_tail = _tail((result.stdout or "") + "\n" + (result.stderr or ""))
    if result.returncode != 0:
        raise InstallFailed(f"uv pip install failed with exit code {result.returncode}", log_tail)
    present = _toolbox_installed()
    missing = [f"{p.name}=={p.version}" for p in packages if _public(present.get(p.name, "")) != p.version]
    if missing:
        raise InstallFailed(f"uv finished but the toolbox lacks {', '.join(missing)}", log_tail)
    approved_at = _now().isoformat()
    why = {_normalise(r["name"]): r.get("why") for r in plan.requested if isinstance(r.get("name"), str)}
    records = [{
        "name": p.name, "version": p.version, "sha256": p.facts["wheel_sha256"], "wheel_url": p.facts.get("wheel_url"),
        "approved_at": approved_at, "requested_by": "agent",
        "why": why.get(p.name) if p.facts.get("requested") else
        f"dependency of {', '.join(p.facts.get('via') or []) or 'a requested package'}",
    } for p in packages]
    _append_ledger(records)
    return {
        "installed": [{"name": p.name, "version": p.version, "sha256": p.facts["wheel_sha256"]} for p in packages],
        "target": str(target),
        "seconds": seconds,
        "log_tail": log_tail,
    }


def _append_ledger(records: list[dict]) -> None:
    path = _ledger()
    try:
        existing = json.loads(path.read_text())
        if not isinstance(existing, list):
            existing = []
    except (OSError, ValueError):
        existing = []
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(existing + records, indent=2) + "\n")
    os.replace(tmp, path)


# ---------------------------------------------------------------- advisory review

_REVIEW_SCHEMA = {
    "type": "object",
    "properties": {"opinion": {"type": "string"}},
    "required": ["opinion"],
    "additionalProperties": False,
}


def ask_llm_review(plan: InstallPlan) -> str | None:
    """A short plain-English opinion on the plan, shown beside the deterministic verdict; never a gate."""
    if config.fake():
        return None
    from agent import llm

    data = {
        "requested": plan.requested,
        "verdict": plan.verdict,
        "problems": plan.problems,
        "packages": [{
            "name": p.name, "version": p.version, "verdict": p.verdict, "reasons": p.reasons,
            **{k: p.facts.get(k) for k in ("summary", "licence", "home_page", "first_release", "n_releases",
                                           "vulnerabilities", "similar_to", "via")},
        } for p in plan.packages if not p.facts.get("already_installed")][:40],
    }
    answer = llm.ask_json(
        "You advise a user who must approve or reject installing these Python packages for an AI agent's adapter "
        "code. In at most three plain sentences, say what the packages are for, whether anything looks risky "
        "(new or obscure projects, names close to popular ones, vulnerabilities, missing licences), and what you "
        "would check before approving. A separate rule-based verdict is already shown, so do not restate it.",
        data, _REVIEW_SCHEMA, max_tokens=250)
    opinion = answer.get("opinion") if isinstance(answer, dict) else None
    return _clean_text(opinion, 800) if isinstance(opinion, str) else None
