"""Find labelled probe files for a new kind of input on the Hugging Face Hub, with no API key.

An adapter turns an input file (a 3D model, a video, a molecule, a pair of coordinates, ...) into text or images,
and the harness verifies it on labelled probe files. The labels must come from an independent source, so this
module finds them on the public Hub: it searches the dataset index, reads a dataset's columns through the public
dataset viewer, picks an input column and a label column, and writes a capped sample as one sub-folder per label
(the folder format agent.data.build_set and sense.adapters.probe_folder read).

Public surface (do not rename):
  ProbesUnavailable                 raised when no usable probe set can be had; the message says why
  HubCandidate                      hub_id, title, downloads, likes, licence, tags, url, gated; to_dict()
  Mapping                           which columns are input and label; to_dict()
  FetchReport                       folder, labels, counts, rows_seen, skipped, licence, source_url, notes, ...
  SEARCH_SCHEMA, MAPPING_SCHEMA     structured-output schemas for agent.llm.ask_json
  search_hub(goal, modality, *, limit=20) -> list[HubCandidate]
  choose_mapping(candidate, goal, modality) -> Mapping
  fetch_probes(candidate, mapping, dest, *, wanted_labels=None, n_per_label=20, max_labels=8, min_per_label=6,
               budget_s=120, max_bytes=150_000_000) -> FetchReport
  find_probes(goal, modality, dest_root, *, wanted_labels=None, emit=None, max_candidates=3, max_inspect=12,
              **fetch_kwargs) -> (folder, info)
  safe_label(text, index) -> str

modality is "audio", "image", "video", "3d", "chemical", "coordinates" or "text", or an alias such as smell, taste,
molecule, smiles, location, place, gps or mesh; any other word still works with the generic rules.

search_hub asks agent.llm.ask_json for up to 5 short queries (SEARCH_SCHEMA); when it answers None (fake mode, no
key, any failure) or nonsense, a rule builds up to 6: the goal's first two content words together, then the
modality's own search words (odor/smell/olfaction for molecules, cities/geonames/country for coordinates,
modelnet/objects/shapenet for 3D, action/ucf101/kinetics/hmdb for video) interleaved with the goal's words. A word
that only names the input (molecule, place, video, 3d, ...) is searched only inside the first query. Each query is
one GET of https://huggingface.co/api/datasets?search=..&limit=..&full=true&sort=downloads. Results are merged and
deduplicated; gated, private and disabled datasets and those whose card turns the viewer off are dropped. Order:
relevance (2 for each topic word and 1 for each input word found in the dataset's id or title, plus 1 for the Hub's
modality tag of this modality and 1 for a *-classification task tag), then how many queries found it, then
downloads. The licence is cardData.license, else a "license:" tag, else None.

choose_mapping reads the viewer's /splits (config "default" if there is one, split train, else test, validation)
and /first-rows (else the first /rows page) for the real features and sample values. It asks ask_json
(MAPPING_SCHEMA) which columns are input and label, then validates the answer against those features: the columns
exist, the label column is not an input, the label is a ClassLabel, a low-cardinality string, a list of strings or
a ";" or "|" delimited string list with at least 2 values, a media input is one viewer media column (Audio, Image,
Video, Mesh) of the modality's type that links to https Hugging Face URLs, and a text template names exactly the
input columns. An answer that fails is ignored and the rule decides: a media column whose type fits the modality,
else string or number columns named for it with fitting values (smiles... with SMILES-like strings; lat/latitude
and lon/lng/longitude within range; text...). The label is the best column named for the goal's topic (the goal's
words and the modality's label words such as genre, action, odor, descriptor, country, less the input words);
a generic one (label/class/category/target/tags, any ClassLabel) or, last, a block of at least 3 columns holding
only 0/1 ("multihot": one column per odour, say) counts only when the dataset's own id or title names the topic,
so an unrelated benchmark's labels are never taken for the goal's. When the first rows show no usable label (a
split sorted by label), one /rows page from the middle of the split is read too. No usable mapping raises
ProbesUnavailable with the reason.

fetch_probes owns dest: it replaces it, deletes it again on any failure, and writes only
dest/<safe_label>/<row index><extension>, so no label or URL can place a file elsewhere. It reads up to 12 /rows
pages of 100 rows spread evenly over the split (only /first-rows when the viewer refuses /rows, which the notes
say), then picks labels: the wanted ones (matched case-insensitively, with space, "_" and "-" alike, against the
dataset's own values; all must be present) or else the most frequent values with at least min_per_label rows, at
most max_labels, never a placeholder such as unknown, none or other. For list and multihot labels only rows
carrying exactly one of the chosen labels are kept (other, unchosen labels a row carries are ignored), and the
notes say so. Text inputs are written from the template after a leak check: a row whose input text contains its
label as a word is dropped, and more than half of the chosen labels' rows leaking refuses the dataset. Media cells
are downloaded DOWNLOAD_WORKERS at a time, the label with the fewest files first, only from https URLs on
huggingface.co or hf.co hosts (redirects too), each at most FILE_MAX_BYTES and all written files together at most
max_bytes, within budget_s; failures are skipped and counted by reason. Labels with fewer than min_per_label files
are dropped; fewer than 2 left, or a wanted label missing, raises ProbesUnavailable. Labels are always the
dataset's own values, never generated.

find_probes searches, then inspects candidates in rank order with choose_mapping (at most max_inspect: a look at
the columns costs two cached viewer calls and most search hits do not fit) and runs fetch_probes into
dest_root/<owner>__<name> for at most max_candidates of those whose columns fit. emit (if given) is called with
small JSON-serialisable dicts {"action": "search" | "candidate" | "inspect" | "fetch", "status": "start" | "ok" |
"fail", ...}. It returns the folder and an info dict (hub_id, url, title, licence, labels, label_values, counts,
mapping, rows_seen, skipped, queries, query_origin, tried: [{hub_id, reason}], notes), or raises ProbesUnavailable
naming every candidate's reason.

Every request goes through _http_get (User-Agent "sense-harness", a timeout, a byte cap, https Hugging Face hosts
only), so tests replace that one function; a JSON request answered HTTP 429 or 503 is retried after RETRY_WAITS.
No API key is used or needed.
"""

from __future__ import annotations

import http.client
import json
import os
import re
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections import Counter, deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Callable

from agent import llm

HUB_SEARCH = "https://huggingface.co/api/datasets"
VIEWER = "https://datasets-server.huggingface.co"
USER_AGENT = "sense-harness"
TIMEOUT_S = 20.0
ROWS_TIMEOUT_S = 40.0  # the viewer's /rows reads parquet on demand and can be slow on large splits
JSON_MAX_BYTES = 50_000_000
FILE_MAX_BYTES = 25_000_000
PAGE_ROWS = 100  # the most rows the dataset viewer returns in one page
MAX_PAGES = 12
MAX_QUERIES = 6
MAX_TEXT_CHARS = 20_000
DOWNLOAD_WORKERS = 4

_now = time.monotonic  # tests replace it to run the budget
_sleep = time.sleep  # tests replace it to skip the waits
RETRY_WAITS = (5.0, 15.0)  # waits before retrying a rate-limited or busy viewer or Hub request

SEARCH_SCHEMA = {
    "type": "object",
    "properties": {"queries": {"type": "array", "items": {"type": "string"}}},
    "required": ["queries"],
    "additionalProperties": False,
}

MAPPING_SCHEMA = {
    "type": "object",
    "properties": {
        "input_columns": {"type": "array", "items": {"type": "string"}},
        "template": {"type": "string"},
        "label_column": {"type": "string"},
        "reason": {"type": "string"},
    },
    "required": ["input_columns", "template", "label_column", "reason"],
    "additionalProperties": False,
}

_SEARCH_INSTRUCTION = (
    "Suggest up to 5 short search queries (one to three words each) for the Hugging Face Hub dataset search that "
    "would find a public dataset of labelled examples for this goal, where each example is an input of the given "
    "modality and its label is the category the goal asks about. The search matches words in dataset names, so "
    "prefer benchmark names and plain domain words. Return JSON with queries."
)

_MAPPING_INSTRUCTION = (
    "Choose the columns of this Hugging Face dataset that give labelled examples for the goal. input_columns: the "
    "column or columns holding the input itself (one media column, or text or number columns such as a SMILES string "
    "or latitude and longitude). template: for text inputs a format string over the input columns such as "
    "'{latitude}, {longitude}', else an empty string. label_column: the column holding the category the goal asks "
    "about; it must not be an input column and must not be derived from the input. Give a one-sentence reason. "
    "Return JSON with input_columns, template, label_column and reason."
)

_MEDIA_TYPES = ("Audio", "Image", "Video", "Mesh")
_MEDIA_EXT = {
    "Audio": (".wav", (".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".aac")),
    "Image": (".jpg", (".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff")),
    "Video": (".mp4", (".mp4", ".webm", ".avi", ".mkv", ".mov", ".m4v")),
    "Mesh": (".glb", (".glb", ".gltf", ".obj", ".ply", ".stl", ".off")),
}
_STOP = frozenset("""
a an the this that these those it its is are was were be been being of in on at to for from by with as and or not
no do does did done what which who whom whose where when why how can could would should will shall may might must
make makes made tell me my you your i we our us let give gives say says show find recognise recognize identify
classify detect predict guess know learn sense new happening happen happens like kind type sort model models file
files data dataset input inputs please any some each into about there here thing things one two get gets
""".split())
_LABEL_WORDS = ("label", "labels", "class", "classes", "category", "categories", "target", "tag", "tags")
_NOT_LABEL = frozenset(("id", "idx", "index", "name", "names", "title", "url", "path", "file", "filename", "source",
                        "split", "description", "caption", "text", "prompt", "question", "date", "time", "timestamp",
                        "license", "author", "authors", "uid", "hash", "sha", "md5", "fold"))


@dataclass(frozen=True)
class _Slot:
    words: tuple[str, ...]  # column-name words, best first
    check: str = "any"  # "any" | "smiles" | "lat" | "lon"


@dataclass(frozen=True)
class _Profile:
    words: tuple[str, ...]  # Hub search words
    media: tuple[str, ...] = ()  # viewer feature types accepted as the input
    slots: tuple[_Slot, ...] = ()  # text inputs: one slot per column
    ext: str = ".txt"
    tag: str | None = None  # the Hub's modality:<tag>
    label_words: tuple[str, ...] = ()  # column names that label this modality by themselves
    input_words: tuple[str, ...] = ()  # words that only name the input, never the label's topic


_PROFILES = {
    "audio": _Profile(("audio", "sound"), media=("Audio",), tag="audio", label_words=("genre",),
                      input_words=("audio", "sound", "recording", "clip", "wav")),
    "image": _Profile(("image",), media=("Image",), tag="image", input_words=("image", "picture", "photo")),
    "video": _Profile(("action", "ucf101", "kinetics", "video", "hmdb"), media=("Video",), tag="video",
                      label_words=("action", "activity"), input_words=("video", "clip", "movie", "footage")),
    "3d": _Profile(("modelnet", "objects", "3d", "mesh", "shapenet"), media=("Mesh",), tag="3d",
                   label_words=("object", "shape"), input_words=("3d", "mesh", "model", "point", "cloud", "glb")),
    "chemical": _Profile(("odor", "smell", "olfaction", "odorant", "scent"),
                         slots=(_Slot(("smiles", "smile", "mol", "molecule"), "smiles"),), ext=".smi",
                         label_words=("odor", "odour", "descriptor", "smell", "scent", "taste", "flavor"),
                         input_words=("molecule", "molecular", "smiles", "chemical", "chemistry", "compound", "mol",
                                      "chem", "structure")),
    "coordinates": _Profile(("cities", "city", "geonames", "country", "countries"),
                            slots=(_Slot(("latitude", "lat"), "lat"),
                                   _Slot(("longitude", "lng", "lon", "long"), "lon")),
                            tag="geospatial", label_words=("country", "continent", "region", "state"),
                            input_words=("place", "location", "coordinate", "latitude", "longitude", "lat", "lon",
                                         "gps", "geo", "point", "position")),
    "text": _Profile((), slots=(_Slot(("text", "sentence", "content", "document")),), tag="text",
                     input_words=("text", "sentence", "document")),
}
_ALIASES = {
    "sound": "audio", "music": "audio", "speech": "audio", "picture": "image", "photo": "image", "images": "image",
    "videos": "video", "clip": "video", "mesh": "3d", "3d model": "3d", "3d-model": "3d", "model3d": "3d",
    "pointcloud": "3d", "point cloud": "3d", "smell": "chemical", "taste": "chemical", "molecule": "chemical",
    "molecules": "chemical", "smiles": "chemical", "chemistry": "chemical", "olfaction": "chemical",
    "odor": "chemical", "odour": "chemical", "coordinate": "coordinates", "location": "coordinates",
    "geo": "coordinates", "gps": "coordinates", "latlon": "coordinates", "lat/lon": "coordinates",
    "geolocation": "coordinates", "place": "coordinates",
}
_PLACEHOLDERS = frozenset(("unknown", "none", "null", "nan", "n/a", "na", "other", "others", "misc", "-", "?"))
_SMILES = re.compile(r"^[A-Za-z0-9@+\-\[\]()=#$:/\\.%*~]+$")


class ProbesUnavailable(RuntimeError):
    """No usable probe set: the reason is in the message."""


class _HttpError(RuntimeError):
    """One request failed; `reason` is a short key such as host_refused, too_large, timeout, network or http_404."""

    def __init__(self, reason: str, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.reason = reason
        self.retry_after = retry_after  # seconds the server asked to wait (HTTP 429 or 503), if it said


@dataclass
class HubCandidate:
    hub_id: str
    title: str
    downloads: int
    likes: int
    licence: str | None
    tags: list[str]
    url: str
    gated: bool = False
    score: int = 0  # search relevance; ordering only

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Mapping:
    config: str
    split: str
    input_columns: list[str]
    input_kind: str  # "media" | "text"
    template: str  # text: e.g. "{latitude}, {longitude}"; media: ""
    extension: str  # ".wav", ".jpg", ".mp4", ".glb", ".smi", ".txt", ...
    label_column: str  # "" for multihot
    label_kind: str  # "classlabel" | "string" | "list" | "multihot"
    label_names: list[str] | None = None  # ClassLabel names, or the 0/1 columns of a multihot label
    label_separator: str | None = None  # a "list" label stored as one delimited string
    media_type: str | None = None  # "Audio" | "Image" | "Video" | "Mesh" for media inputs
    origin: str = "rule"  # "llm" | "rule"
    reason: str = ""

    def to_dict(self) -> dict:
        out = asdict(self)
        if self.label_names is not None and len(self.label_names) > 50:
            out["label_names"] = self.label_names[:50] + [f"... {len(self.label_names) - 50} more"]
        return out


@dataclass
class FetchReport:
    folder: Path
    labels: list[str]  # folder names kept, most files first
    counts: dict[str, int]  # folder name -> files written
    rows_seen: int
    skipped: dict[str, int]  # reason -> rows or files skipped
    licence: str | None
    source_url: str
    notes: list[str]
    hub_id: str = ""
    label_values: dict[str, str] = field(default_factory=dict)  # folder name -> the dataset's own label value
    bytes: int = 0  # the kept files' total size

    def to_dict(self) -> dict:
        return asdict(self) | {"folder": str(self.folder)}


# ----------------------------------------------------------------------------- HTTP
class _Redirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not _allowed_url(newurl):
            raise _HttpError("host_refused", f"refused a redirect to {_host(newurl)}")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _http_get(url: str, *, timeout: float = TIMEOUT_S, max_bytes: int = JSON_MAX_BYTES) -> bytes:
    """GET an https Hugging Face URL: at most max_bytes, within timeout seconds, or _HttpError."""
    if not _allowed_url(url):
        raise _HttpError("host_refused", f"only https Hugging Face URLs are fetched, not {_host(url)}")
    if timeout <= 0:
        raise _HttpError("timeout", "no time left")
    deadline = time.monotonic() + timeout
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "*/*"})
    try:
        with urllib.request.build_opener(_Redirects).open(request, timeout=timeout) as response:
            length = response.headers.get("Content-Length")
            if length and length.isdigit() and int(length) > max_bytes:
                raise _HttpError("too_large", f"{_host(url)} offers {int(length):,} bytes, over the {max_bytes:,} cap")
            chunks, size = [], 0
            while chunk := response.read(min(1 << 16, max_bytes + 1 - size)):
                size += len(chunk)
                chunks.append(chunk)
                if size > max_bytes:
                    raise _HttpError("too_large", f"{_host(url)} sent over the {max_bytes:,} byte cap")
                if time.monotonic() > deadline:
                    raise _HttpError("timeout", f"{_host(url)} took over {timeout:.0f} s")
            return b"".join(chunks)
    except _HttpError:
        raise
    except urllib.error.HTTPError as exc:
        detail = _error_detail(exc)
        wait = _number(exc.headers.get("Retry-After") if exc.headers else None)
        exc.close()
        raise _HttpError(f"http_{exc.code}", f"HTTP {exc.code} from {_host(url)}{': ' + detail if detail else ''}",
                         wait) from None
    except TimeoutError:
        raise _HttpError("timeout", f"{_host(url)} took over {timeout:.0f} s") from None
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, TimeoutError):
            raise _HttpError("timeout", f"{_host(url)} took over {timeout:.0f} s") from None
        raise _HttpError("network", f"URLError reaching {_host(url)}") from None
    except (OSError, http.client.HTTPException, ValueError) as exc:
        raise _HttpError("network", f"{type(exc).__name__} reaching {_host(url)}") from None


def _error_detail(exc: urllib.error.HTTPError) -> str:
    try:
        body = exc.read(2000)
        answer = json.loads(body)
        text = answer.get("error") if isinstance(answer, dict) else None
    except Exception:  # an error body is optional detail only
        return ""
    return re.sub(r"\s+", " ", text)[:200] if isinstance(text, str) else ""


def _allowed_url(url: object) -> bool:
    try:
        url.encode("ascii")
        parts = urllib.parse.urlsplit(url)
        host, port = (parts.hostname or "").lower(), parts.port
    except (UnicodeError, ValueError, AttributeError):
        return False
    return (parts.scheme == "https" and not parts.username and not parts.password and port in (None, 443)
            and (host in ("huggingface.co", "hf.co") or host.endswith((".huggingface.co", ".hf.co"))))


def _host(url: object) -> str:
    try:
        return urllib.parse.urlsplit(str(url)).hostname or "an invalid URL"
    except ValueError:
        return "an invalid URL"


def _get_json(base: str, params: dict, *, timeout: float = TIMEOUT_S, what: str) -> object:
    """One JSON answer; a rate limit (HTTP 429) or a busy server (503) is retried after RETRY_WAITS."""
    url = base + "?" + urllib.parse.urlencode(params)
    for wait in RETRY_WAITS + (None,):
        try:
            return json.loads(_http_get(url, timeout=timeout, max_bytes=JSON_MAX_BYTES))
        except _HttpError as exc:
            if wait is None or exc.reason not in ("http_429", "http_503"):
                raise ProbesUnavailable(f"{what} failed: {exc}") from None
            _sleep(min(max(exc.retry_after or 0.0, wait), 30.0))
        except (ValueError, UnicodeError):
            raise ProbesUnavailable(f"{what} did not answer JSON") from None
    raise AssertionError("unreachable")


# ----------------------------------------------------------------------------- words
def _profile(modality: str) -> _Profile | None:
    key = re.sub(r"\s+", " ", str(modality).strip().lower())
    return _PROFILES.get(_ALIASES.get(key, key))


def _singular(word: str) -> str:
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith("sses"):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def _content_words(goal: str) -> list[str]:
    out = []
    for word in re.findall(r"[a-z0-9]+", str(goal).lower()):
        if word in _STOP or (len(word) < 3 and not re.search(r"\d", word)) or word.isdigit():
            continue
        word = _singular(word)
        if word not in out and word not in _STOP:
            out.append(word)
    return out


def _tokens(name: str) -> list[str]:
    """Column-name words: camelCase and separators split, lower case."""
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(name))
    return [t for t in re.split(r"[^a-z0-9]+", spaced.lower()) if t]


def _name_hits(name: str, words) -> bool:
    for token in _tokens(name):
        for word in words:
            if token == word or (len(word) >= 5 and token.endswith(word)) or _singular(token) == word:
                return True
    return False


# ----------------------------------------------------------------------------- search
def search_hub(goal: str, modality: str, *, limit: int = 20) -> list[HubCandidate]:
    return _search(goal, modality, limit)[0]


def _search(goal: str, modality: str, limit: int) -> tuple[list[HubCandidate], list[str], str]:
    queries = _llm_queries(llm.ask_json(_SEARCH_INSTRUCTION, {"goal": goal, "modality": modality}, SEARCH_SCHEMA))
    origin = "llm" if queries else "rule"
    queries = queries or _rule_queries(goal, modality)
    if not queries:
        raise ProbesUnavailable(f"no search words could be taken from the goal {goal!r}")
    found: dict[str, HubCandidate] = {}
    hits: Counter = Counter()
    errors = []
    for query in queries:
        try:
            answer = _get_json(HUB_SEARCH, {"search": query, "limit": limit, "full": "true", "sort": "downloads"},
                               what=f"the Hub search for {query!r}")
        except ProbesUnavailable as exc:
            errors.append(str(exc))
            continue
        if not isinstance(answer, list):
            errors.append(f"the Hub search for {query!r} did not answer a list")
            continue
        for item in answer:
            candidate = _candidate(item)
            if candidate is not None:
                found.setdefault(candidate.hub_id, candidate)
                hits[candidate.hub_id] += 1
    if errors and len(errors) == len(queries):
        raise ProbesUnavailable(f"the Hub search failed for every query: {errors[0]}")
    profile = _profile(modality)
    words = {_singular(w) for q in queries for w in re.findall(r"[a-z0-9]+", q.lower()) if w not in _STOP}
    inputs = set(profile.input_words) if profile else set(_content_words(modality))
    for candidate in found.values():
        candidate.score = (2 * _relevance(candidate, words - inputs) + _relevance(candidate, words & inputs)
                           + _tag_bonus(candidate.tags, profile))
    ranked = sorted(found.values(), key=lambda c: (-c.score, -hits[c.hub_id], -c.downloads, c.hub_id))
    return ranked[:limit], queries, origin


def _relevance(candidate: HubCandidate, words: set[str]) -> int:
    """How many of the search words appear in the dataset's id or title (plurals and word stems alike)."""
    tokens = set(_tokens(candidate.hub_id) + _tokens(candidate.title))
    return sum(1 for word in words if any(_same_word(token, word) for token in tokens))


def _same_word(token: str, word: str) -> bool:
    if token == word or _singular(token) == word:
        return True
    if len(word) >= 4 and token.startswith(word):
        return True
    common = len(os.path.commonprefix([token, word]))
    return common >= 6 and common >= min(len(token), len(word)) - 2


def _llm_queries(answer: object) -> list[str]:
    raw = answer.get("queries") if isinstance(answer, dict) else None
    if not isinstance(raw, list):
        return []
    out = []
    for query in raw:
        if isinstance(query, str) and 0 < len(query.strip()) <= 60 and query.strip().isprintable():
            query = re.sub(r"\s+", " ", query.strip())
            if query.lower() not in (q.lower() for q in out):
                out.append(query)
    return out[:5]


def _rule_queries(goal: str, modality: str) -> list[str]:
    """The goal's first two content words together, then the modality's search words interleaved with the goal's
    words; a word that only names the input (molecule, video, 3d, ...) is searched only inside the first query."""
    profile = _profile(modality)
    inputs = set(profile.input_words) if profile else set(_content_words(modality))
    goal_words = _content_words(goal)
    queries = [" ".join(goal_words[:2])] if len(goal_words) >= 2 else []
    words = [w for w in goal_words if w not in inputs]
    extra = [w for w in (profile.words if profile else _content_words(modality)) if w not in inputs]
    if not words and not extra:
        words = goal_words or _content_words(modality)
    for i in range(max(len(words), len(extra))):
        queries += [w for w in (extra[i] if i < len(extra) else None, words[i] if i < len(words) else None) if w]
    out = []
    for query in queries:
        if query not in out:
            out.append(query)
    return out[:MAX_QUERIES]


def _tag_bonus(tags: list[str], profile: _Profile | None) -> int:
    bonus = 1 if profile and profile.tag and f"modality:{profile.tag}" in tags else 0
    return bonus + (1 if any(t.startswith("task_categories:") and t.endswith("-classification") for t in tags) else 0)


_HUB_ID = re.compile(r"^[\w.-]+/[\w.-]+$")


def _candidate(item: object) -> HubCandidate | None:
    if not isinstance(item, dict):
        return None
    hub_id = item.get("id")
    if not isinstance(hub_id, str) or not _HUB_ID.match(hub_id) or {".", ".."} & set(hub_id.split("/")):
        return None
    if item.get("gated") or item.get("private") or item.get("disabled"):
        return None
    card = item.get("cardData") if isinstance(item.get("cardData"), dict) else {}
    if card.get("viewer") is False:
        return None
    tags = [t for t in item.get("tags") or [] if isinstance(t, str)] if isinstance(item.get("tags"), list) else []
    title = card.get("pretty_name") if isinstance(card.get("pretty_name"), str) else hub_id
    return HubCandidate(hub_id, title.strip()[:120] or hub_id, _int(item.get("downloads")), _int(item.get("likes")),
                        _licence(card, tags), tags[:40], f"https://huggingface.co/datasets/{hub_id}", False)


def _int(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _licence(card: dict, tags: list[str]) -> str | None:
    value = card.get("license")
    if isinstance(value, list):
        value = ", ".join(str(v) for v in value if isinstance(v, str) and v) or None
    if isinstance(value, str) and value.strip():
        return value.strip()
    for tag in tags:
        if tag.startswith("license:") and tag.partition(":")[2]:
            return tag.partition(":")[2]
    return None


# ----------------------------------------------------------------------------- choosing columns
@dataclass
class _Column:
    name: str
    kind: str  # "media" | "classlabel" | "string" | "number" | "list" | "other"
    media_type: str | None = None
    names: list[str] | None = None  # ClassLabel names (also for a list of ClassLabel)


def choose_mapping(candidate: HubCandidate, goal: str, modality: str) -> Mapping:
    hub_id = candidate.hub_id
    splits = _get_json(f"{VIEWER}/splits", {"dataset": hub_id}, what=f"the viewer's splits of {hub_id}")
    config, split = _pick_split(splits, hub_id)
    params = {"dataset": hub_id, "config": config, "split": split}
    what = f"the viewer's rows of {hub_id}"
    try:  # the viewer precomputes first-rows, so it is quick and rarely fails
        first = _get_json(f"{VIEWER}/first-rows", params, what=what + " (first rows)")
        if not isinstance(first, dict) or not isinstance(first.get("rows"), list):
            raise ProbesUnavailable(f"{what} (first rows) did not answer a page of rows")
    except ProbesUnavailable:
        first = _rows_page(params, 0, what=what, timeout=ROWS_TIMEOUT_S)
    columns = _columns(first.get("features"), hub_id)
    rows = _rows(first)
    if not rows:
        raise ProbesUnavailable(f"the viewer shows no rows for {hub_id} ({config}/{split})")
    profile = _profile(modality)
    try:
        mapping = _map(candidate, goal, modality, columns, rows, profile)
    except _NoLabel:
        # A split sorted by label can show one label in its first rows: look once more, in the middle.
        try:
            total = _rows_page(params, 0, what=what, timeout=ROWS_TIMEOUT_S, length=1).get("num_rows_total")
            if not isinstance(total, int) or total <= 2 * PAGE_ROWS:
                raise ProbesUnavailable("no more rows")
            rows += _rows(_rows_page(params, total // 2, what=what, timeout=ROWS_TIMEOUT_S))
        except ProbesUnavailable:
            raise _NoLabel(_map_error(candidate, goal, modality, columns, rows, profile)) from None
        mapping = _map(candidate, goal, modality, columns, rows, profile)
    mapping.config, mapping.split = config, split
    return mapping


class _NoLabel(ProbesUnavailable):
    """The columns fit the input but no label column was found in the rows looked at."""


def _map_error(candidate, goal, modality, columns, rows, profile) -> str:
    try:
        _map(candidate, goal, modality, columns, rows, profile)
    except ProbesUnavailable as exc:
        return str(exc)
    return f"{candidate.hub_id} has no usable label column"


def _map(candidate, goal, modality, columns, rows, profile) -> Mapping:
    mapping = _llm_mapping(candidate, goal, modality, columns, rows, profile)
    return mapping if mapping is not None else _rule_mapping(candidate, goal, columns, rows, profile, modality)


def _pick_split(answer: object, hub_id: str) -> tuple[str, str]:
    raw = answer.get("splits") if isinstance(answer, dict) else None
    splits = [s for s in raw or [] if isinstance(s, dict) and isinstance(s.get("config"), str)
              and isinstance(s.get("split"), str)] if isinstance(raw, list) else []
    if not splits:
        error = answer.get("error") if isinstance(answer, dict) else None
        raise ProbesUnavailable(f"the viewer lists no splits for {hub_id}" + (f": {str(error)[:200]}" if error else ""))
    configs = [s["config"] for s in splits]
    config = "default" if "default" in configs else configs[0]
    in_config = [s["split"] for s in splits if s["config"] == config]
    return config, next((s for s in ("train", "test", "validation") if s in in_config), in_config[0])


def _first_page(params: dict, *, what: str, timeout: float = TIMEOUT_S) -> tuple[dict, str | None]:
    """The first /rows page, else /first-rows with the reason /rows failed: the viewer refuses /rows for some
    splits (large row groups) while it still serves their first 100 rows."""
    try:
        return _rows_page(params, 0, what=what, timeout=timeout), None
    except ProbesUnavailable as exc:
        try:
            answer = _get_json(f"{VIEWER}/first-rows", params, what=what + " (first rows)", timeout=timeout)
        except ProbesUnavailable:
            raise exc from None
        if not isinstance(answer, dict) or not isinstance(answer.get("rows"), list):
            raise exc from None
        return answer, str(exc)


def _rows_page(params: dict, offset: int, *, what: str, timeout: float = TIMEOUT_S, length: int = PAGE_ROWS) -> dict:
    answer = _get_json(f"{VIEWER}/rows", params | {"offset": offset, "length": length}, what=what, timeout=timeout)
    if not isinstance(answer, dict) or not isinstance(answer.get("rows"), list):
        raise ProbesUnavailable(f"{what} did not answer a page of rows")
    return answer


def _rows(page: dict) -> list[tuple[int, dict, set]]:
    """(row index, row, truncated column names) for each well-formed row of a viewer page."""
    out = []
    for entry in page.get("rows") or []:
        if not isinstance(entry, dict):
            continue
        idx, row = entry.get("row_idx"), entry.get("row")
        if isinstance(idx, int) and not isinstance(idx, bool) and idx >= 0 and isinstance(row, dict):
            cut = entry.get("truncated_cells")
            out.append((idx, row, set(cut) if isinstance(cut, list) else set()))
    return out


def _columns(features: object, hub_id: str) -> dict[str, _Column]:
    out = {}
    for feature in features if isinstance(features, list) else []:
        if not isinstance(feature, dict) or not isinstance(feature.get("name"), str):
            continue
        name, kind = feature["name"], feature.get("type")
        kind = kind if isinstance(kind, dict) else {}
        out[name] = _column(name, kind)
    if not out:
        raise ProbesUnavailable(f"the viewer shows no columns for {hub_id}")
    return out


def _column(name: str, kind: dict) -> _Column:
    typ, dtype = kind.get("_type"), kind.get("dtype")
    if typ in _MEDIA_TYPES:
        return _Column(name, "media", media_type=typ)
    if typ == "ClassLabel" and isinstance(kind.get("names"), list):
        return _Column(name, "classlabel", names=[str(n) for n in kind["names"]])
    if typ == "Value" and dtype in ("string", "large_string"):
        return _Column(name, "string")
    if typ == "Value" and isinstance(dtype, str) and re.fullmatch(r"u?int\d+|float\d+|bool", dtype):
        return _Column(name, "number")
    if typ in ("List", "Sequence", "LargeList") and isinstance(kind.get("feature"), dict):
        inner = kind["feature"]
        if inner.get("_type") == "Value" and inner.get("dtype") in ("string", "large_string"):
            return _Column(name, "list")
        if inner.get("_type") == "ClassLabel" and isinstance(inner.get("names"), list):
            return _Column(name, "list", names=[str(n) for n in inner["names"]])
    return _Column(name, "other")


def _llm_mapping(candidate, goal, modality, columns, rows, profile) -> Mapping | None:
    data = {"goal": goal, "modality": modality, "dataset": candidate.hub_id, "title": candidate.title,
            "columns": [_describe(col, rows) for col in list(columns.values())[:80]]}
    answer = llm.ask_json(_MAPPING_INSTRUCTION, data, MAPPING_SCHEMA, max_tokens=300)
    if not isinstance(answer, dict):
        return None
    inputs, template, label = answer.get("input_columns"), answer.get("template"), answer.get("label_column")
    if not (isinstance(inputs, list) and inputs and len(inputs) <= 4 and all(isinstance(c, str) for c in inputs)
            and len(set(inputs)) == len(inputs) and all(c in columns for c in inputs)
            and isinstance(label, str) and label in columns and label not in inputs and isinstance(template, str)):
        return None
    labelled = _label_fit(columns[label], rows)
    if labelled is None:
        return None
    kind, separator = labelled
    reason = answer.get("reason").strip()[:300] if isinstance(answer.get("reason"), str) else ""
    picked = [columns[c] for c in inputs]
    if picked[0].kind == "media":
        if len(picked) != 1 or (profile and profile.media and picked[0].media_type not in profile.media):
            return None
        if (profile and not profile.media) or _media_ext(picked[0], rows) is None:
            return None
        return Mapping("", "", inputs, "media", "", _media_ext(picked[0], rows), label, kind, columns[label].names,
                       separator, picked[0].media_type, "llm", reason)
    if any(col.kind not in ("string", "number") for col in picked) or (profile and profile.media):
        return None
    template = template.strip() or ", ".join("{" + c + "}" for c in inputs)
    if set(re.findall(r"\{([^{}]*)\}", template)) != set(inputs):
        return None
    ext = profile.ext if profile else ".txt"
    return Mapping("", "", inputs, "text", template, ext, label, kind, columns[label].names, separator, None, "llm",
                   reason)


def _describe(col: _Column, rows) -> dict:
    out = {"name": col.name, "kind": col.kind}
    if col.media_type:
        out["media_type"] = col.media_type
    if col.names:
        out["label_names"] = col.names[:20]
    elif col.kind != "media":
        out["examples"] = [str(row.get(col.name))[:80] for _, row, _ in rows[:3]]
    return out


def _rule_mapping(candidate, goal, columns, rows, profile, modality) -> Mapping:
    hub_id, names = candidate.hub_id, ", ".join(list(columns)[:30])
    inputs, media_type, ext, template = _rule_inputs(columns, rows, profile, modality)
    if not inputs:
        if profile and profile.media:
            raise ProbesUnavailable(f"{hub_id} has no {'/'.join(profile.media)} column for {modality} "
                                    f"(columns: {names})")
        wanted = " and ".join("/".join(slot.words) for slot in profile.slots) if profile else "text"
        raise ProbesUnavailable(f"{hub_id} has no input column for {modality}: wanted a column named like {wanted} "
                                f"with fitting values (columns: {names})")
    if media_type and ext is None:
        raise ProbesUnavailable(f"{hub_id}'s {inputs[0]} cells do not link to https Hugging Face files "
                                f"(e.g. {_example_src(columns[inputs[0]], rows)})")
    topic = _topic_words(goal, modality, profile)
    on_topic = any(_same_word(token, word) for token in _tokens(hub_id) + _tokens(candidate.title) for word in topic)
    label = _rule_label(_topic_words(goal, modality, profile, search_words=False), columns, rows, set(inputs),
                        on_topic)
    if label is None:
        raise _NoLabel(
            f"{hub_id} has no label column named for the goal (a string or list column named like "
            f"{'/'.join(sorted(topic)[:8]) or 'the goal'} with at least 2 values)"
            + ("" if on_topic else ", and its name does not mention the goal's topic, so a generic label/class "
                                   "column, ClassLabel or block of 0/1 columns is not trusted")
            + f" (columns: {names})")
    column, kind, separator, label_names = label
    kind_of = "media" if media_type else "text"
    reason = f"rule: input {', '.join(inputs)}; label {column or 'the 0/1 columns'} ({kind})"
    return Mapping("", "", inputs, kind_of, template, ext, column, kind, label_names, separator, media_type, "rule",
                   reason)


def _rule_inputs(columns, rows, profile, modality):
    """(input columns, media type or None, extension or None, template)."""
    media_types = profile.media if profile else tuple(t for t in _MEDIA_TYPES if t.lower() == str(modality).lower())
    if media_types:
        media = [c for c in columns.values() if c.kind == "media" and c.media_type in media_types]
        media.sort(key=lambda c: (_media_ext(c, rows) is None, not _name_hits(c.name, (c.media_type.lower(),))))
        if media:
            return [media[0].name], media[0].media_type, _media_ext(media[0], rows), ""
        return [], None, None, ""
    if not profile or not profile.slots:
        return [], None, None, ""
    picked = []
    for slot in profile.slots:
        best = None
        for rank, word in enumerate(slot.words):
            for col in columns.values():
                if col.kind in ("string", "number") and col.name not in picked and _name_hits(col.name, (word,)) \
                        and _values_fit(col.name, rows, slot.check):
                    best = col.name
                    break
            if best:
                break
        if best is None:
            return [], None, None, ""
        picked.append(best)
    return picked, None, profile.ext, ", ".join("{" + c + "}" for c in picked)


def _values_fit(name: str, rows, check: str) -> bool:
    values = [row.get(name) for _, row, _ in rows]
    present = [v for v in values if v is not None and str(v).strip()]
    if len(present) < max(1, len(values) // 2):
        return False
    if check == "smiles":
        return sum(1 for v in present if isinstance(v, str) and _SMILES.match(v.strip())) >= 0.8 * len(present)
    if check in ("lat", "lon"):
        limit = 90 if check == "lat" else 180
        numbers = [_number(v) for v in present]
        return all(n is not None and -limit <= n <= limit for n in numbers)
    return True


def _number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _media_ext(col: _Column, rows) -> str | None:
    """The file extension of the column's first downloadable cell; None when no cell links to an allowed URL."""
    default, known = _MEDIA_EXT.get(col.media_type, (".bin", ()))
    for _, row, _ in rows:
        src = _src(row.get(col.name))
        if src and _allowed_url(src):
            suffix = PurePosixPath(urllib.parse.urlsplit(src).path).suffix.lower()
            return suffix if suffix in known else default
    return None


def _example_src(col: _Column, rows) -> str:
    for _, row, _ in rows:
        src = _src(row.get(col.name))
        if src:
            return src.split("::")[0][:60]
    return "no file links at all"


def _src(cell: object) -> str | None:
    """The file URL of a viewer media cell: a dict with "src", or a one-item list of one."""
    if isinstance(cell, list) and len(cell) == 1:
        cell = cell[0]
    src = cell.get("src") if isinstance(cell, dict) else None
    return src if isinstance(src, str) else None


def _topic_words(goal: str, modality: str, profile: _Profile | None, *, search_words: bool = True) -> set[str]:
    """Words that can name what the label is about: the goal's content words and the modality's label words (and,
    for naming a whole dataset, its search words), less the words that only name the input."""
    inputs = set(profile.input_words) if profile else set(_content_words(modality))
    words = set(_content_words(goal)) | set(profile.label_words if profile else ())
    if search_words and profile:
        words |= set(profile.words)
    return words - inputs


def _rule_label(topic: set[str], columns, rows, inputs: set, on_topic: bool):
    """(column, kind, separator, names) of the best label, or None. A column named for the goal's topic always
    counts; a generic one (label, class, category, any ClassLabel, a block of 0/1 columns) only when the dataset
    itself is named for the topic (on_topic), so an unrelated benchmark's labels are never taken for the goal's."""
    topic = tuple(topic)
    best, best_score = None, 0
    for col in columns.values():
        if col.name in inputs or col.kind not in ("classlabel", "string", "list"):
            continue
        named = _name_hits(col.name, topic)
        if not named and not on_topic:
            continue
        score = 4 if named else 0
        score += 3 if col.name.lower() in _LABEL_WORDS else 2 if _name_hits(col.name, _LABEL_WORDS) else 0
        score -= 3 if set(_tokens(col.name)) & _NOT_LABEL else 0
        if col.kind == "classlabel":
            score = max(score, 0) + 1
        if score <= 0:
            continue
        fit = _label_fit(col, rows)
        if fit is not None and score > best_score:
            best, best_score = (col.name, fit[0], fit[1], col.names), score
    if best is not None:
        return best
    if not on_topic:
        return None
    binary = [c.name for c in columns.values() if c.kind == "number" and c.name not in inputs
              and not set(_tokens(c.name)) & _NOT_LABEL and _binary(c.name, rows)]
    if len(binary) >= 3:
        return "", "multihot", None, binary
    return None


def _binary(name: str, rows) -> bool:
    values = [row.get(name) for _, row, _ in rows if row.get(name) is not None]
    return bool(values) and all(v in (0, 1) for v in values) and any(v == 1 for v in values)


def _label_fit(col: _Column, rows) -> tuple[str, str | None] | None:
    """(label kind, separator) when the column can label probes, else None."""
    if col.kind == "classlabel":
        return ("classlabel", None) if col.names and len(col.names) >= 2 else None
    cells = [row.get(col.name) for _, row, _ in rows]
    present = [c for c in cells if c is not None and c != "" and c != []]
    if len(present) < max(2, len(cells) // 2):
        return None
    if col.kind == "list":
        if col.names is not None:
            return ("list", None) if len(col.names) >= 2 else None
        values = [v.strip() for c in present if isinstance(c, list) for v in c if isinstance(v, str) and v.strip()]
        return ("list", None) if len(set(values)) >= 2 else None
    if col.kind != "string":
        return None
    texts = [c.strip() for c in present if isinstance(c, str) and c.strip()]
    for separator in (";", "|"):
        if sum(separator in t for t in texts) >= 0.3 * len(texts):
            values = [v.strip() for t in texts for v in t.split(separator) if v.strip()]
            if 2 <= len(set(values)) <= 0.8 * len(values):
                return "list", separator
    distinct = len(set(texts))
    return ("string", None) if 2 <= distinct <= 0.6 * len(texts) and max(map(len, texts)) <= 100 else None


# ----------------------------------------------------------------------------- fetching probes
def fetch_probes(candidate: HubCandidate, mapping: Mapping, dest: Path, *, wanted_labels: list[str] | None = None,
                 n_per_label: int = 20, max_labels: int = 8, min_per_label: int = 6, budget_s: float = 120,
                 max_bytes: int = 150_000_000) -> FetchReport:
    dest = Path(dest)
    shutil.rmtree(dest, ignore_errors=True)  # dest belongs to this fetch: an older sample is replaced
    try:
        dest.mkdir(parents=True, exist_ok=True)
        return _fetch(candidate, mapping, dest, wanted_labels or [], n_per_label, max_labels, min_per_label,
                      _now() + budget_s, max_bytes)
    except ProbesUnavailable:
        shutil.rmtree(dest, ignore_errors=True)
        raise
    except OSError as exc:
        shutil.rmtree(dest, ignore_errors=True)
        raise ProbesUnavailable(f"could not write probes under {dest}: {exc}") from exc
    except BaseException:
        shutil.rmtree(dest, ignore_errors=True)
        raise


def _fetch(candidate, mapping, dest, wanted, n_per_label, max_labels, min_per_label, deadline, max_bytes):
    hub_id = candidate.hub_id
    params = {"dataset": hub_id, "config": mapping.config, "split": mapping.split}
    what = f"the viewer's rows of {hub_id}"
    skipped: Counter = Counter()
    notes: list[str] = []
    first, refused = _first_page(params, what=what, timeout=_left(deadline))
    total = first.get("num_rows_total") if isinstance(first.get("num_rows_total"), int) else PAGE_ROWS
    if refused:
        total = PAGE_ROWS
        notes.append(f"only the split's first rows could be read ({refused})")
    seen: dict[int, tuple[str, list[str]]] = {}  # row index -> (input text or URL, label values)
    read: set[int] = set()
    rows_seen = _take(first, mapping, seen, read, skipped)
    for offset in _offsets(total)[1:]:
        if deadline - _now() < 5:
            notes.append(f"stopped reading rows at the time budget after {rows_seen} rows")
            break
        try:
            page = _rows_page(params, offset, what=what, timeout=_left(deadline))
        except ProbesUnavailable as exc:
            skipped["page_failed"] += 1
            notes.append(f"a page at offset {offset} failed: {exc}")
            continue
        rows_seen += _take(page, mapping, seen, read, skipped)
    if not seen:
        raise ProbesUnavailable(f"none of the {rows_seen} rows read from {hub_id} had both an input and a label")
    chosen = _choose_labels(seen, mapping, wanted, max_labels, min_per_label, notes, hub_id)
    by_label = _assign(seen, chosen, mapping, skipped, notes)
    if mapping.input_kind == "text":
        _leak_check(by_label, chosen, skipped, notes, hub_id)
    folders = {value: safe_label(value, i) for i, value in enumerate(chosen)}
    queues = {value: _spread(sorted(by_label[value]), n_per_label) for value in chosen}
    if mapping.input_kind == "text":
        counts, sizes = _write_texts(queues, folders, mapping, dest, n_per_label)
    else:
        counts, sizes = _download_all(queues, folders, mapping, dest, n_per_label, deadline, max_bytes, skipped, notes)
    kept = {}
    for value in chosen:
        if counts[value] >= min_per_label:
            kept[value] = counts[value]
        else:
            shutil.rmtree(dest / folders[value], ignore_errors=True)
            notes.append(f"dropped label {value!r}: {counts[value]} files, fewer than {min_per_label}")
    missing = [w for w in wanted if _key(w) not in {_key(v) for v in kept}]
    if missing:
        raise ProbesUnavailable(f"wanted labels without {min_per_label} probe files in {hub_id}: {missing}")
    if len(kept) < 2:
        raise ProbesUnavailable(f"only {len(kept)} label(s) of {hub_id} reached {min_per_label} probe files "
                                f"({sum(counts.values())} files written; skipped: {dict(skipped)})")
    order = sorted(kept, key=lambda v: -kept[v])
    return FetchReport(dest, [folders[v] for v in order], {folders[v]: kept[v] for v in order}, rows_seen,
                       dict(skipped), candidate.licence, candidate.url, notes, hub_id,
                       {folders[v]: v for v in order}, sum(sizes[v] for v in order))


def _left(deadline: float, cap: float = ROWS_TIMEOUT_S) -> float:
    left = deadline - _now()
    if left <= 0:
        raise ProbesUnavailable("the probe time budget was spent")
    return min(cap, left)


def _offsets(total: int) -> list[int]:
    """Page offsets: every page when the split fits in MAX_PAGES pages, else MAX_PAGES spread evenly over it."""
    if total <= PAGE_ROWS * MAX_PAGES:
        return list(range(0, max(total, 1), PAGE_ROWS))
    last = total - PAGE_ROWS
    return sorted({int(k * last / (MAX_PAGES - 1)) for k in range(MAX_PAGES)})


def _take(page: dict, mapping: Mapping, seen: dict, read: set, skipped: Counter) -> int:
    """Read one page's rows into seen; returns how many new rows it held."""
    n = 0
    for idx, row, truncated in _rows(page):
        if idx in read:
            continue
        read.add(idx)
        n += 1
        if truncated & set(mapping.input_columns):
            skipped["truncated"] += 1
            continue
        value = _input_of(row, mapping)
        if value is None:
            skipped["no_input"] += 1
            continue
        labels = _labels_of(row, mapping)
        if not labels:
            skipped["no_label"] += 1
            continue
        seen[idx] = (value, labels)
    return n


def _input_of(row: dict, mapping: Mapping) -> str | None:
    if mapping.input_kind == "media":
        return _src(row.get(mapping.input_columns[0]))
    values = {}
    for name in mapping.input_columns:
        value = row.get(name)
        if value is None or isinstance(value, (dict, list)) or not str(value).strip():
            return None
        values[name] = str(value).strip()
    text = re.sub(r"\{([^{}]*)\}", lambda m: values.get(m.group(1), m.group(0)), mapping.template)
    return text if len(text) <= MAX_TEXT_CHARS else None


def _labels_of(row: dict, mapping: Mapping) -> list[str]:
    if mapping.label_kind == "multihot":
        return [name for name in mapping.label_names or [] if row.get(name) in (1, True)]
    cell = row.get(mapping.label_column)
    names = mapping.label_names
    if mapping.label_kind == "classlabel":
        return [names[cell]] if isinstance(cell, int) and not isinstance(cell, bool) and 0 <= cell < len(names) else []
    if mapping.label_kind == "string":
        return [cell.strip()] if isinstance(cell, str) and cell.strip() else []
    if isinstance(cell, str) and mapping.label_separator:
        cell = cell.split(mapping.label_separator)
    if not isinstance(cell, list):
        return []
    out = []
    for value in cell:
        if names is not None and isinstance(value, int) and not isinstance(value, bool) and 0 <= value < len(names):
            value = names[value]
        if isinstance(value, str) and value.strip() and value.strip() not in out:
            out.append(value.strip())
    return out


def _key(label: str) -> str:
    return re.sub(r"[\s_-]+", " ", str(label).strip().lower())


def _choose_labels(seen, mapping, wanted, max_labels, min_per_label, notes, hub_id) -> list[str]:
    multi = mapping.label_kind in ("list", "multihot")
    rows_labels = [labels for _, labels in seen.values()]
    freq = Counter(v for labels in rows_labels for v in labels)
    if wanted:
        known = {}
        for value in list(mapping.label_names or []) + list(freq):
            known.setdefault(_key(value), value)
        missing = [w for w in wanted if _key(w) not in known]
        if missing:
            sample = ", ".join(sorted(freq, key=lambda v: -freq[v])[:15])
            raise ProbesUnavailable(f"wanted labels not among {hub_id}'s label values: {missing} (seen: {sample})")
        chosen = list(dict.fromkeys(known[_key(w)] for w in wanted))
        folders = [_folder(v) for v in chosen]
        if None in folders or len(set(folders)) < len(folders):
            raise ProbesUnavailable(f"wanted labels {chosen} have no distinct ASCII folder names")
        notes.append(f"labels are the wanted ones, matched to the dataset's values: {chosen}")
        return chosen
    chosen, taken = [], set()
    for value, count in sorted(freq.items(), key=lambda kv: -kv[1]):
        if len(chosen) >= max_labels or count < min_per_label:
            break
        folder = _folder(value)
        if folder is None or folder in taken:
            notes.append(f"skipped label {value!r}: its folder name would be empty or shared")
            continue
        if _key(value) in _PLACEHOLDERS:
            notes.append(f"skipped label {value!r}: it names no category")
            continue
        trial = chosen + [value]
        if multi and min(_exclusive(rows_labels, trial).values()) < min_per_label:
            continue
        chosen, taken = trial, taken | {folder}
    if len(chosen) < 2:
        raise ProbesUnavailable(f"{hub_id} has fewer than 2 label values with {min_per_label} usable rows among "
                                f"{len(seen)} rows read" + (" (rows carrying exactly one label)" if multi else ""))
    return chosen


def _folder(value: str) -> str | None:
    folder = safe_label(value, 0)
    return None if folder == "label_0" and _key(value) != "label 0" else folder


def _exclusive(rows_labels: list[list[str]], chosen: list[str]) -> Counter:
    wanted = set(chosen)
    counts = Counter({v: 0 for v in chosen})
    for labels in rows_labels:
        hit = wanted.intersection(labels)
        if len(hit) == 1:
            counts[hit.pop()] += 1
    return counts


def _assign(seen, chosen, mapping, skipped, notes) -> dict[str, list[tuple[int, str]]]:
    """Each chosen label's rows; list and multihot rows must carry exactly one chosen label."""
    wanted = set(chosen)
    by_label: dict[str, list[tuple[int, str]]] = {v: [] for v in chosen}
    multi = 0
    for idx, (value, labels) in seen.items():
        hit = wanted.intersection(labels)
        if len(hit) == 1:
            by_label[hit.pop()].append((idx, value))
        elif len(hit) > 1:
            multi += 1
    if mapping.label_kind in ("list", "multihot"):
        skipped["multi_label"] += multi
        notes.append(f"the dataset gives several labels per row: only rows carrying exactly one of the chosen labels "
                     f"are kept ({multi} rows with two or more were dropped); other labels a row carries are ignored")
    return by_label


def _leak_check(by_label, chosen, skipped, notes, hub_id) -> None:
    total = leaked = 0
    for value in chosen:
        pattern = _word_pattern(value)
        clean = [(idx, text) for idx, text in by_label[value] if not pattern.search(text)]
        total += len(by_label[value])
        leaked += len(by_label[value]) - len(clean)
        by_label[value] = clean
    if leaked:
        skipped["leak"] += leaked
        notes.append(f"dropped {leaked} of {total} rows whose input text names their own label")
    if total and leaked * 2 > total:
        raise ProbesUnavailable(f"{hub_id}'s input text names the label in {leaked} of {total} rows, so it cannot "
                                "test a sense")


def _word_pattern(label: str) -> re.Pattern:
    forms = {label.strip().lower(), re.sub(r"[_-]+", " ", label.strip().lower())}
    body = "|".join(re.escape(f) for f in sorted(forms, key=len, reverse=True) if f)
    return re.compile(rf"(?<![a-z0-9])(?:{body})(?![a-z0-9])", re.IGNORECASE)


def _spread(items: list, k: int) -> list:
    """k items evenly spaced through the list first, then the rest as spares for failed downloads."""
    if len(items) <= k:
        return list(items)
    picks = sorted({int(i * len(items) / k) for i in range(k)})
    chosen = set(picks)
    return [items[j] for j in picks] + [x for j, x in enumerate(items) if j not in chosen]


def _write_texts(queues, folders, mapping, dest, n_per_label) -> tuple[Counter, Counter]:
    counts, sizes = Counter({value: 0 for value in queues}), Counter()
    for value, queue in queues.items():
        for idx, text in queue[:n_per_label]:
            data = (text + "\n").encode("utf-8")
            _write(dest / folders[value] / f"{idx}{mapping.extension}", data)
            counts[value] += 1
            sizes[value] += len(data)
    return counts, sizes


def _download_all(queues, folders, mapping, dest, n_per_label, deadline, max_bytes, skipped, notes):
    """Download with DOWNLOAD_WORKERS at once, always for the label with the fewest files so far, until each label is
    full or out of rows, or the byte or time budget is spent. A file that would take the written total over
    max_bytes is discarded, so the folder never holds more than max_bytes."""
    pending = {value: deque(queue) for value, queue in queues.items()}
    counts, busy, sizes = Counter({value: 0 for value in queues}), Counter(), Counter()
    default, known = _MEDIA_EXT.get(mapping.media_type, (mapping.extension, ()))
    spent, stop, running = 0, None, {}

    def next_row():
        open_labels = [v for v in pending if pending[v] and counts[v] + busy[v] < n_per_label]
        if not open_labels:
            return None
        value = min(open_labels, key=lambda v: counts[v] + busy[v])
        return value, *pending[value].popleft()

    with ThreadPoolExecutor(DOWNLOAD_WORKERS) as pool:
        while True:
            while stop is None and len(running) < DOWNLOAD_WORKERS:
                if spent >= max_bytes:
                    stop = f"stopped downloading at the {max_bytes:,} byte cap"
                elif deadline - _now() < 1:
                    stop = "stopped downloading at the time budget"
                elif (job := next_row()) is None:
                    break
                elif not _allowed_url(job[2]):
                    skipped["host_refused"] += 1
                else:
                    busy[job[0]] += 1
                    timeout = min(TIMEOUT_S, deadline - _now())
                    running[pool.submit(_http_get, job[2], timeout=timeout,
                                        max_bytes=min(FILE_MAX_BYTES, max_bytes - spent))] = job
            if not running:
                break
            done, _ = wait(running, return_when=FIRST_COMPLETED)
            for future in done:
                value, idx, src = running.pop(future)
                busy[value] -= 1
                try:
                    data = future.result()
                except _HttpError as exc:
                    skipped[exc.reason if exc.reason in ("too_large", "host_refused") else "download_failed"] += 1
                    continue
                if not data:
                    skipped["download_failed"] += 1
                    continue
                if spent + len(data) > max_bytes:
                    skipped["over_byte_cap"] += 1
                    stop = f"stopped downloading at the {max_bytes:,} byte cap"
                    continue
                suffix = PurePosixPath(urllib.parse.urlsplit(src).path).suffix.lower()
                _write(dest / folders[value] / f"{idx}{suffix if suffix in known else mapping.extension or default}",
                       data)
                counts[value] += 1
                sizes[value] += len(data)
                spent += len(data)
    if stop:
        notes.append(stop)
    return counts, sizes


def _write(target: Path, data: bytes) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.{uuid.uuid4().hex[:8]}.part")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)


def safe_label(text: str, index: int) -> str:
    """Only a-z, 0-9 and inner "_", at most 40 characters, never empty: so never ".", ".." or a path."""
    out = re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")[:40].strip("_")
    return out or f"label_{index}"


# ----------------------------------------------------------------------------- the whole step
def find_probes(goal: str, modality: str, dest_root: Path, *, wanted_labels: list[str] | None = None,
                emit: Callable[[dict], None] | None = None, max_candidates: int = 3, max_inspect: int = 12,
                **fetch_kwargs) -> tuple[Path, dict]:
    say = emit or (lambda event: None)
    dest_root = Path(dest_root)
    say({"action": "search", "status": "start", "goal": str(goal)[:300], "modality": str(modality)})
    try:
        candidates, queries, origin = _search(goal, modality, 20)
    except ProbesUnavailable as exc:
        say({"action": "search", "status": "fail", "reason": str(exc)})
        raise
    say({"action": "search", "status": "ok", "queries": queries, "origin": origin, "found": len(candidates),
         "top": [c.hub_id for c in candidates[:max_inspect]]})
    if not candidates:
        raise ProbesUnavailable(f"the Hub search found no open dataset for queries {queries}")
    tried, fetched = [], 0
    for candidate in candidates[:max_inspect]:
        if fetched >= max_candidates:
            break
        hub_id = candidate.hub_id
        say({"action": "candidate", "status": "start", "hub_id": hub_id, "downloads": candidate.downloads,
             "licence": candidate.licence, "url": candidate.url})
        say({"action": "inspect", "status": "start", "hub_id": hub_id})
        try:
            mapping = choose_mapping(candidate, goal, modality)
        except ProbesUnavailable as exc:
            tried.append({"hub_id": hub_id, "reason": f"inspect: {exc}"})
            say({"action": "inspect", "status": "fail", "hub_id": hub_id, "reason": str(exc)})
            say({"action": "candidate", "status": "fail", "hub_id": hub_id, "reason": str(exc)})
            continue
        say({"action": "inspect", "status": "ok", "hub_id": hub_id, "mapping": mapping.to_dict()})
        fetched += 1
        dest = dest_root / hub_id.replace("/", "__")
        say({"action": "fetch", "status": "start", "hub_id": hub_id, "folder": str(dest)})
        try:
            report = fetch_probes(candidate, mapping, dest, wanted_labels=wanted_labels, **fetch_kwargs)
        except ProbesUnavailable as exc:
            tried.append({"hub_id": hub_id, "reason": f"fetch: {exc}"})
            say({"action": "fetch", "status": "fail", "hub_id": hub_id, "reason": str(exc)})
            say({"action": "candidate", "status": "fail", "hub_id": hub_id, "reason": str(exc)})
            continue
        say({"action": "fetch", "status": "ok", "hub_id": hub_id, "counts": dict(report.counts),
             "rows_seen": report.rows_seen, "skipped": dict(report.skipped), "bytes": report.bytes})
        say({"action": "candidate", "status": "ok", "hub_id": hub_id, "labels": list(report.labels)})
        info = {"hub_id": hub_id, "url": candidate.url, "title": candidate.title, "licence": candidate.licence,
                "labels": list(report.labels), "label_values": dict(report.label_values),
                "counts": dict(report.counts), "mapping": mapping.to_dict(), "rows_seen": report.rows_seen,
                "skipped": dict(report.skipped), "queries": queries, "query_origin": origin, "tried": tried,
                "notes": [mapping.reason] + list(report.notes)}
        return report.folder, info
    raise ProbesUnavailable("no candidate dataset gave usable probes: "
                            + "; ".join(f"{t['hub_id']} ({t['reason']})" for t in tried))
