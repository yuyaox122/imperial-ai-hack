"""The one module through which the agent reaches the network: Linkup web search, the Hugging Face Hub and
the Hugging Face dataset viewer, with an on-disk cache and canned answers for fake mode.

Public surface (do not rename):
  ResearchUnavailable                       raised when an answer cannot be had; the message names the cause
  Research(data, sources, origin)           sources are [{"name", "url", "snippet"}]; origin "live" | "cache" | "fake"
  search_structured(query, schema, *, include_domains=None, max_results=8, timeout=25.0) -> Research
  hub_model(hub_id) -> dict | None          {"id", "pipeline_tag", "licence", "downloads"}; None if absent/unreachable
  hub_dataset(hub_id) -> dict | None        {"id", "licence", "downloads"}; None if absent or unreachable
  viewer(endpoint, **params) -> dict        GET https://datasets-server.huggingface.co/<endpoint>?<params>
  download(url, dest, *, timeout=20.0, max_bytes=20_000_000) -> int   bytes written to dest

Every request goes through _http, the only function that opens a connection (tests patch it).
Linkup: POST https://api.linkup.so/v1/search with "Authorization: Bearer $LINKUP_API_KEY" (read at call time,
after sense.config has loaded .env), structured output and sources; HTTP 429 means rate limit or no credit.
"licence" is the Hub's cardData.license, else its "license:<id>" tag, else None.

Cache: each successful live JSON answer is saved as <SENSE_HOME>/research/<sha256(method, url, body)[:16]>.json
(never the key or any header). On a network or HTTP error, or with no Linkup key, the cached answer is served
with origin "cache"; with none, ResearchUnavailable names the cause. A Hub 401 or 404 means "no such repo"
(the Hub answers 401 for a missing repo when no token is sent) and gives None without reading the cache.
Downloads are not cached: download writes beside dest under a temporary name and renames on success, so a
failed or oversized download leaves nothing at dest.

Fake mode (SENSE_FAKE, checked on every call): search_structured returns canned answers chosen by the schema's
first top-level property ("specialists" or "datasets"); hub_model and hub_dataset answer from small tables;
viewer and download raise ResearchUnavailable("fake mode has no network").
"""

from __future__ import annotations

import copy
import hashlib
import http.client
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path

from sense import config

LINKUP_URL = "https://api.linkup.so/v1/search"
HUB_API = "https://huggingface.co/api"
VIEWER_URL = "https://datasets-server.huggingface.co"
USER_AGENT = "sense-agent/0.1"
SNIPPET_CHARS = 300  # a source's content is cut to this many characters
MAX_RESPONSE_BYTES = 200_000_000  # _http never holds more than this in memory
_ERROR_BODY_CHARS = 200  # how much of an error response is quoted in ResearchUnavailable
_HUB_ID = re.compile(r"^[\w.-]+(/[\w.-]+)?$")
_ENDPOINT = re.compile(r"^[a-z][a-z-]*$")
_NO_REPO = (401, 404)
_FAKE_URL = "https://example.org/fake/"


class ResearchUnavailable(RuntimeError):
    """No answer: no network, no key, an HTTP error, or fake mode, and nothing cached."""


@dataclass
class Research:
    data: dict
    sources: list[dict]  # [{"name", "url", "snippet"}]
    origin: str  # "live" | "cache" | "fake"


# ----------------------------------------------------------------------------- fake answers
_FAKE_SEARCH = {
    "specialists": {
        "data": {
            "specialists": [
                {
                    "hub_id": "laion/clap-htsat-unfused",
                    "statement": "Fake claim for tests: 0.90 on a fake benchmark",
                    "source_url": _FAKE_URL + "clap-htsat-unfused",
                    "source_name": "Fake source for tests",
                    "metric": "a fake benchmark",
                    "value": 0.90,
                },
                {
                    "hub_id": "example/unrunnable-audio-model",
                    "statement": "Fake claim for tests: a model this harness cannot load, with no number",
                    "source_url": _FAKE_URL + "unrunnable-audio-model",
                    "source_name": "Fake source for tests",
                    "metric": None,
                    "value": None,
                },
            ]
        },
        "sources": [
            {"name": "Fake source for tests", "url": _FAKE_URL + "clap-htsat-unfused", "snippet": "Fake snippet."},
            {"name": "Fake source for tests", "url": _FAKE_URL + "unrunnable-audio-model", "snippet": "Fake snippet."},
        ],
    },
    "datasets": {
        "data": {
            "datasets": [
                {
                    "hub_id": "example/fake-sounds",
                    "title": "Fake sounds",
                    "licence": "cc-by-4.0",
                    "n_examples": 200,
                    "labels": ["bell", "dog", "rain", "siren"],
                    "statement": "Fake claim for tests: 200 examples in 4 labels under cc-by-4.0",
                    "source_url": _FAKE_URL + "fake-sounds",
                }
            ]
        },
        "sources": [{"name": "Fake source for tests", "url": _FAKE_URL + "fake-sounds", "snippet": "Fake snippet."}],
    },
}

_FAKE_MODELS = {
    "laion/clap-htsat-unfused": {
        "id": "laion/clap-htsat-unfused",
        "pipeline_tag": "zero-shot-audio-classification",
        "licence": "apache-2.0",
        "downloads": 1000,
    },
    "example/unrunnable-audio-model": {
        "id": "example/unrunnable-audio-model",
        "pipeline_tag": "audio-classification",
        "licence": "mit",
        "downloads": 10,
    },
}

_FAKE_DATASETS = {"example/fake-sounds": {"id": "example/fake-sounds", "licence": "cc-by-4.0", "downloads": 100}}


# ----------------------------------------------------------------------------- public functions
def search_structured(
    query: str,
    schema: dict,
    *,
    include_domains: list[str] | None = None,
    max_results: int = 8,
    timeout: float = 25.0,
) -> Research:
    if config.fake():
        first = next(iter(schema.get("properties", {})), None)
        canned = _FAKE_SEARCH.get(first, {"data": {first: []} if first else {}, "sources": []})
        canned = copy.deepcopy(canned)
        return Research(canned["data"], canned["sources"], "fake")
    body = {
        "q": query,
        "depth": "standard",
        "outputType": "structured",
        "structuredOutputSchema": json.dumps(schema),
        "includeSources": True,
        "maxResults": max_results,
    }
    if include_domains:
        body["includeDomains"] = list(include_domains)
    key = os.environ.get("LINKUP_API_KEY")
    if not key:
        answer, origin = _cached_or_raise("POST", LINKUP_URL, body, "LINKUP_API_KEY is not set")
    else:
        headers = {"Authorization": f"Bearer {key}"}
        answer, origin = _fetch_json("POST", LINKUP_URL, body=body, headers=headers, timeout=timeout)
    data = answer.get("data") if isinstance(answer, dict) else None
    if not isinstance(data, dict):
        raise ResearchUnavailable("Linkup answered without a structured data object")
    sources = [
        {
            "name": str(s.get("name") or ""),
            "url": str(s.get("url") or ""),
            "snippet": str(s.get("content") or s.get("snippet") or "")[:SNIPPET_CHARS],
        }
        for s in answer.get("sources") or []
        if isinstance(s, dict)
    ]
    return Research(data, sources, origin)


def hub_model(hub_id: str) -> dict | None:
    if config.fake():
        return copy.deepcopy(_FAKE_MODELS.get(hub_id))
    info = _hub("models", hub_id)
    if info is None:
        return None
    return {
        "id": info.get("id") or hub_id,
        "pipeline_tag": info.get("pipeline_tag"),
        "licence": _licence(info),
        "downloads": info.get("downloads"),
    }


def hub_dataset(hub_id: str) -> dict | None:
    if config.fake():
        return copy.deepcopy(_FAKE_DATASETS.get(hub_id))
    info = _hub("datasets", hub_id)
    if info is None:
        return None
    return {"id": info.get("id") or hub_id, "licence": _licence(info), "downloads": info.get("downloads")}


def viewer(endpoint: str, **params) -> dict:
    if config.fake():
        raise ResearchUnavailable("fake mode has no network")
    if not _ENDPOINT.match(endpoint):
        raise ResearchUnavailable(f"{endpoint!r} is not a dataset viewer endpoint name such as 'splits' or 'rows'")
    url = f"{VIEWER_URL}/{endpoint}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    answer, _ = _fetch_json("GET", url, timeout=25.0)
    if not isinstance(answer, dict):
        raise ResearchUnavailable(f"the dataset viewer's {endpoint} answer is not a JSON object")
    return answer


def download(url: str, dest: Path, *, timeout: float = 20.0, max_bytes: int = 20_000_000) -> int:
    if config.fake():
        raise ResearchUnavailable("fake mode has no network")
    if urllib.parse.urlsplit(url).scheme not in ("http", "https"):
        raise ResearchUnavailable(f"only http and https downloads are allowed, not {url[:80]!r}")
    try:
        data = _http("GET", url, timeout=timeout)
    except _NETWORK_ERRORS as exc:
        raise ResearchUnavailable(f"download from {_host(url)} failed: {_describe(exc, url)}") from exc
    if len(data) > max_bytes:
        raise ResearchUnavailable(f"download from {_host(url)} is {len(data):,} bytes, over the {max_bytes:,} limit")
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".{dest.name}.{uuid.uuid4().hex[:8]}.part")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, dest)
    finally:
        tmp.unlink(missing_ok=True)
    return len(data)


# ----------------------------------------------------------------------------- network and cache
def _http(method: str, url: str, *, body: dict | None = None, headers: dict | None = None, timeout: float) -> bytes:
    """The only function that opens a connection. Raises urllib/OS errors as they come."""
    data = json.dumps(body).encode() if body is not None else None
    all_headers = {"User-Agent": USER_AGENT, "Accept": "application/json, */*"}
    if data is not None:
        all_headers["Content-Type"] = "application/json"
    all_headers.update(headers or {})
    request = urllib.request.Request(url, data=data, headers=all_headers, method=method)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = response.read(MAX_RESPONSE_BYTES + 1)
    if len(payload) > MAX_RESPONSE_BYTES:
        raise ResearchUnavailable(f"the answer from {_host(url)} is over {MAX_RESPONSE_BYTES:,} bytes")
    return payload


_NETWORK_ERRORS = (OSError, http.client.HTTPException, ResearchUnavailable)  # URLError and timeouts are OSError


def _fetch_json(
    method: str,
    url: str,
    *,
    body: dict | None = None,
    headers: dict | None = None,
    timeout: float,
    absent: tuple[int, ...] = (),
) -> tuple[object, str]:
    """(parsed answer, "live") and the answer cached; on failure (cached answer, "cache") or ResearchUnavailable.
    An HTTP status in `absent` is a definite "not there": (None, "live"), and the cache is not read."""
    try:
        answer = json.loads(_http(method, url, body=body, headers=headers, timeout=timeout))
    except urllib.error.HTTPError as exc:
        if exc.code in absent:
            return None, "live"
        return _cached_or_raise(method, url, body, f"{_host(url)} answered {_describe(exc, url)}", exc)
    except _NETWORK_ERRORS + (ValueError,) as exc:
        cause = _describe(exc, url)
        return _cached_or_raise(method, url, body, f"{_host(url)} could not be reached: {cause}", exc)
    _write_cache(method, url, body, answer)
    return answer, "live"


def _cached_or_raise(
    method: str, url: str, body: dict | None, cause: str, exc: Exception | None = None
) -> tuple[object, str]:
    path = _cache_path(method, url, body)
    try:
        return json.loads(path.read_text())["answer"], "cache"
    except (OSError, ValueError, KeyError, TypeError):
        raise ResearchUnavailable(f"{cause}, and nothing is cached for this request") from exc


def _cache_path(method: str, url: str, body: dict | None) -> Path:
    blob = json.dumps([method, url, body], sort_keys=True).encode()
    return config.home() / "research" / (hashlib.sha256(blob).hexdigest()[:16] + ".json")


def _write_cache(method: str, url: str, body: dict | None, answer: object) -> None:
    """Saves the answer only (never headers); a failure to write is not an error for the caller."""
    path = _cache_path(method, url, body)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}.part")
        tmp.write_text(json.dumps({"method": method, "url": url, "answer": answer}))
        os.replace(tmp, path)
    except OSError:
        pass


def _hub(kind: str, hub_id: str) -> dict | None:
    """The Hub's JSON for a model or dataset repo; None when the id is malformed, absent or unreachable."""
    if not isinstance(hub_id, str) or not _HUB_ID.match(hub_id) or {".", ".."} & set(hub_id.split("/")):
        return None
    url = f"{HUB_API}/{kind}/{urllib.parse.quote(hub_id, safe='/')}"
    try:
        answer, _ = _fetch_json("GET", url, timeout=15.0, absent=_NO_REPO)
    except ResearchUnavailable:
        return None
    return answer if isinstance(answer, dict) else None


def _licence(info: dict) -> str | None:
    """cardData.license (a string or a list of them), else the first "license:<id>" tag, else None."""
    card = info.get("cardData") or {}
    value = card.get("license") if isinstance(card, dict) else None
    if isinstance(value, list):
        value = ", ".join(str(v) for v in value if v) or None
    if isinstance(value, str) and value:
        return value
    for tag in info.get("tags") or []:
        if isinstance(tag, str) and tag.startswith("license:"):
            return tag.partition(":")[2] or None
    return None


def _host(url: str) -> str:
    return urllib.parse.urlsplit(url).hostname or "the server"


def _describe(exc: Exception, url: str) -> str:
    """A short cause for an error message. Never includes headers, and the Linkup key is masked."""
    if isinstance(exc, urllib.error.HTTPError):
        text = f"HTTP {exc.code}"
        if exc.code == 429:
            text += " (rate limit or credit used up)" if _host(url) == "api.linkup.so" else " (rate limit)"
        detail = _error_body(exc)
        if detail:
            text += f": {detail}"
    elif isinstance(exc, urllib.error.URLError):
        text = str(exc.reason)
    else:
        text = str(exc) or type(exc).__name__
    key = os.environ.get("LINKUP_API_KEY")
    return text.replace(key, "***") if key else text


def _error_body(exc: urllib.error.HTTPError) -> str:
    """The "error" or "message" field of a JSON error body, else its first characters."""
    try:
        raw = exc.read()[:4096].decode("utf-8", "replace")
    except Exception:
        return ""
    try:
        parsed = json.loads(raw)
    except ValueError:
        parsed = None
    if isinstance(parsed, dict):
        field = parsed.get("error") or parsed.get("message") or ""
        if isinstance(field, dict):
            field = field.get("message") or ""
        raw = str(field)
    return " ".join(raw.split())[:_ERROR_BODY_CHARS]
