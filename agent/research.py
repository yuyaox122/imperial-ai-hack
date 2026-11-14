"""The one module through which the agent reaches the network: Linkup web search, the Hugging Face Hub and
the Hugging Face dataset viewer, with an on-disk cache and canned answers for fake mode.

Public surface (do not rename):
  ResearchUnavailable                       raised when an answer cannot be had; the message names the cause
  Research(data, sources, origin)           sources are [{"name", "url", "snippet"}]; origin "live" | "cache" | "fake"
  search_structured(query, schema, *, include_domains=None, max_results=8, timeout=25.0) -> Research
  hub_model(hub_id) -> dict | None          {"id", "pipeline_tag", "licence", "downloads"}; None only for HTTP 404; raises if unavailable
  hub_dataset(hub_id) -> dict | None        {"id", "licence", "downloads"}; None only for HTTP 404; raises if unavailable
  viewer(endpoint, **params) -> dict        GET https://datasets-server.huggingface.co/<endpoint>?<params>
  download(url, dest, *, timeout=20.0, max_bytes=20_000_000) -> int   bytes written to dest

Every request goes through _http, which supervises an isolated HTTP worker (tests patch _http).
Linkup: POST https://api.linkup.so/v1/search with "Authorization: Bearer $LINKUP_API_KEY" (read at call time,
after sense.config has loaded .env), structured output and sources; HTTP 429 means rate limit or no credit.
A key with whitespace or control characters is refused ("LINKUP_API_KEY is malformed", never the value). A
200 answer without a "data" object is not cached and the error names the top-level keys that did arrive.
Redirects are followed only to http and https URLs, and Authorization is dropped when one changes host or
port or downgrades https to http.
"licence" is the Hub's cardData.license, else its "license:<id>" tag, else None.

Cache: each successful live JSON answer is saved as <SENSE_HOME>/research/<sha256(method, url, body)[:16]>.json
(never the key or any header). On a network or HTTP error, or with no Linkup key, the cached answer is served
with origin "cache"; with none, ResearchUnavailable names the cause. A Hub 404 gives None without reading the cache. Access denial (including 401), outages and malformed
records remain unavailable, since they cannot establish that a repository is missing.
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
import io
import math
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from sense import config

from ._http_transport import ResearchUnavailable, _RedirectGuard

LINKUP_URL = "https://api.linkup.so/v1/search"
HUB_API = "https://huggingface.co/api"
VIEWER_URL = "https://datasets-server.huggingface.co"
USER_AGENT = "sense-agent/0.1"
SNIPPET_CHARS = 300  # a source's content is cut to this many characters
MAX_RESPONSE_BYTES = 200_000_000  # JSON body cap; downloads pass their smaller allowance
_ERROR_BODY_CHARS = 200  # how much of an error response is quoted in ResearchUnavailable
_HUB_ID = re.compile(r"^[\w.-]+(/[\w.-]+)?$")
_ENDPOINT = re.compile(r"^[a-z][a-z-]*$")
_NO_REPO = (404,)
_FAKE_URL = "https://example.org/fake/"


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
        "pipeline_tag": "feature-extraction",  # as the live Hub reports it
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
        if any(c.isspace() or not c.isprintable() for c in key):
            raise ResearchUnavailable("LINKUP_API_KEY is malformed")
        headers = {"Authorization": f"Bearer {key}"}
        answer, origin = _fetch_json(
            "POST", LINKUP_URL, body=body, headers=headers, timeout=timeout, check=_linkup_problem
        )
    problem = _linkup_problem(answer)
    if problem:
        raise ResearchUnavailable(problem)
    data = answer["data"]
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


def _linkup_problem(answer: object) -> str | None:
    """Why a Linkup answer is unusable (no "data" object), naming at most 10 top-level keys; None if usable."""
    if isinstance(answer, dict) and isinstance(answer.get("data"), dict):
        if answer.get("sources") is not None and not isinstance(answer["sources"], list):
            return "Linkup answered with invalid sources (expected a list)"
        return None
    if isinstance(answer, dict):
        keys = ", ".join(sorted(str(k) for k in answer)[:10]) or "none"
        return f"Linkup answered without a data object; top-level keys were: {keys}"
    return f"Linkup answered with a JSON {type(answer).__name__}, not an object with a data field"


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


def viewer(endpoint: str, *, timeout: float = 25.0, **params) -> dict:
    if config.fake():
        raise ResearchUnavailable("fake mode has no network")
    if not _ENDPOINT.match(endpoint):
        raise ResearchUnavailable(f"{endpoint!r} is not a dataset viewer endpoint name such as 'splits' or 'rows'")
    url = f"{VIEWER_URL}/{endpoint}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    answer, _ = _fetch_json("GET", url, timeout=timeout)
    if not isinstance(answer, dict):
        raise ResearchUnavailable(f"the dataset viewer's {endpoint} answer is not a JSON object")
    return answer


def download(url: str, dest: Path, *, timeout: float = 20.0, max_bytes: int = 20_000_000) -> int:
    if config.fake():
        raise ResearchUnavailable("fake mode has no network")
    _check_download_url(url)
    try:
        data = _http("GET", url, timeout=timeout, max_bytes=max_bytes)
    except _NETWORK_ERRORS + (ValueError,) as exc:  # ValueError includes UnicodeError
        cause = _describe(exc, url, read_body=False)
        if isinstance(exc, urllib.error.HTTPError):
            exc.close()
        failure = ResearchUnavailable(f"download from {_host(url)} failed: {cause}")
        failure.bytes_received = getattr(exc, "bytes_received", 0)
        failure.transfer_incomplete = getattr(exc, "transfer_incomplete", False)
        raise failure from exc
    if len(data) > max_bytes:
        failure = ResearchUnavailable(f"download from {_host(url)} is {len(data):,} bytes, over the {max_bytes:,} limit")
        failure.bytes_received = len(data)
        raise failure
    dest = Path(dest)
    tmp = dest.with_name(f".{dest.name}.{uuid.uuid4().hex[:8]}.part")
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            tmp.write_bytes(data)
            os.replace(tmp, dest)
        finally:
            tmp.unlink(missing_ok=True)
    except OSError as exc:
        failure = ResearchUnavailable(f"could not save download under {dest.parent}: {exc}")
        failure.bytes_received = len(data)
        failure.transfer_incomplete = True
        raise failure from exc
    return len(data)


def _check_download_url(url: str) -> None:
    """ResearchUnavailable unless `url` is an ASCII http or https URL with a host."""
    try:
        url.encode("ascii")
        parts = urllib.parse.urlsplit(url)
        ok = parts.scheme in ("http", "https") and bool(parts.hostname)
    except (UnicodeError, ValueError, AttributeError):
        ok = False
    if not ok:
        raise ResearchUnavailable(f"only ASCII http and https URLs with a host can be downloaded, not {url[:80]!r}")


# ----------------------------------------------------------------------------- network and cache
def _http(method: str, url: str, *, body: dict | None = None, headers: dict | None = None,
          timeout: float, max_bytes: int = MAX_RESPONSE_BYTES) -> bytes:
    """Bound all network phases with a killable, reaped stdlib worker.

    The deadline includes worker startup. OS process creation/scheduling and
    reaping add unavoidable overhead; server-controlled I/O cannot extend it.
    Body bytes are capped at max_bytes plus one overflow sentinel. No worker
    or socket continues running after this call returns, including on failure.
    """
    if not math.isfinite(timeout) or timeout <= 0:
        raise ResearchUnavailable("response deadline reached")
    if not isinstance(max_bytes, int) or max_bytes < 0:
        raise ResearchUnavailable("invalid response byte limit")
    deadline = time.monotonic() + timeout
    request = json.dumps(dict(method=method, url=url, body=body, headers=headers,
                              timeout=timeout, max_bytes=max_bytes)).encode()
    worker = Path(__file__).with_name("_http_transport.py")
    # Isolated stdlib startup avoids app imports and executes no site hooks.
    # Secrets are sent through the pipe, never command arguments or disk.
    with subprocess.Popen([sys.executable, "-I", "-S", str(worker)], stdin=subprocess.PIPE,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE) as process:
        try:
            output, error = process.communicate(request, timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired as exc:
            process.kill()
            output, _ = process.communicate()
            failure = ResearchUnavailable("response deadline reached")
            failure.bytes_received = len(output)
            failure.transfer_incomplete = True
            raise failure from exc
        except BaseException:
            process.kill()
            process.communicate()
            raise
    if process.returncode:
        try:
            record = json.loads(error)
        except (ValueError, UnicodeError):
            record = {"message": "HTTP worker exited without a result", "transfer_incomplete": True}
        if "status" in record:
            failure = urllib.error.HTTPError(url, record["status"], record.get("message", "HTTP error"),
                                             {}, io.BytesIO())
        else:
            failure = ResearchUnavailable(record.get("message", "HTTP worker failed"))
        failure.bytes_received = max(len(output), record.get("bytes_received", 0))
        failure.transfer_incomplete = record.get("transfer_incomplete", True)
        raise failure
    return output


_NETWORK_ERRORS = (OSError, http.client.HTTPException, ResearchUnavailable)  # URLError and timeouts are OSError


def _fetch_json(
    method: str,
    url: str,
    *,
    body: dict | None = None,
    headers: dict | None = None,
    timeout: float,
    absent: tuple[int, ...] = (),
    check: Callable[[object], str | None] | None = None,
) -> tuple[object, str]:
    """(parsed answer, "live") and the answer cached; on failure (cached answer, "cache") or ResearchUnavailable.
    An HTTP status in `absent` is a definite "not there": (None, "live"), and the cache is not read.
    `check` returns why a live answer is unusable; such an answer raises ResearchUnavailable and is not cached."""
    try:
        answer = json.loads(_http(method, url, body=body, headers=headers, timeout=timeout))
    except urllib.error.HTTPError as exc:
        if exc.code in absent:
            return None, "live"
        return _cached_or_raise(method, url, body, f"{_host(url)} answered {_describe(exc, url)}", exc)
    except _NETWORK_ERRORS + (ValueError,) as exc:
        cause = _describe(exc, url)
        return _cached_or_raise(method, url, body, f"{_host(url)} could not be reached: {cause}", exc)
    problem = check(answer) if check else None
    if problem:
        raise ResearchUnavailable(problem)
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
    """The Hub record; None only for HTTP 404, unavailable/invalid responses raise."""
    if not isinstance(hub_id, str) or not _HUB_ID.match(hub_id) or {".", ".."} & set(hub_id.split("/")):
        raise ResearchUnavailable("invalid Hugging Face repository id")
    url = f"{HUB_API}/{kind}/{urllib.parse.quote(hub_id, safe='/')}"
    answer, origin = _fetch_json("GET", url, timeout=15.0, absent=_NO_REPO, check=_hub_problem)
    # _fetch_json produces live None only for the explicit absent status; cached
    # JSON null is malformed data, never evidence that a repository is missing.
    if answer is None and origin == "live":
        return None
    problem = _hub_problem(answer)
    if problem:
        raise ResearchUnavailable(problem)
    return answer


def _hub_problem(answer: object) -> str | None:
    if not isinstance(answer, dict) or not isinstance(answer.get("id"), str) or not answer["id"]:
        return "the Hugging Face Hub returned an invalid repository record (expected an object with an id)"
    return None


def _licence(info: dict) -> str | None:
    """cardData.license (a string or a list of them), else the first "license:<id>" tag, else None."""
    card = info.get("cardData") or {}
    value = card.get("license") if isinstance(card, dict) else None
    if isinstance(value, list):
        value = ", ".join(str(v) for v in value if v) or None
    if isinstance(value, str) and value:
        return value
    tags = info.get("tags")
    for tag in tags if isinstance(tags, list) else []:
        if isinstance(tag, str) and tag.startswith("license:"):
            return tag.partition(":")[2] or None
    return None


def _host(url: str) -> str:
    try:
        return urllib.parse.urlsplit(url).hostname or "the server"
    except ValueError:
        return "the server"


def _describe(exc: Exception, url: str, *, read_body: bool = True) -> str:
    """A short cause for an error message. Never includes headers, and the Linkup key is masked."""
    if isinstance(exc, urllib.error.HTTPError):
        text = f"HTTP {exc.code}"
        if exc.code == 429:
            text += " (rate limit or credit used up)" if _host(url) == "api.linkup.so" else " (rate limit)"
        detail = _error_body(exc) if read_body else ""
        if detail:
            text += f": {detail}"
    elif isinstance(exc, urllib.error.URLError):
        text = str(exc.reason)
    else:
        text = str(exc) or type(exc).__name__
    key = os.environ.get("LINKUP_API_KEY")
    return text.replace(key, "***") if key else text


def _error_body(exc: urllib.error.HTTPError) -> str:
    """The "error" or "message" field of a JSON error body (and Linkup's "details", when given), else the
    body's first characters. Reads at most 4,096 bytes."""
    try:
        raw = (exc.read(4096) or b"").decode("utf-8", "replace")
    except Exception:
        return ""
    try:
        parsed = json.loads(raw)
    except ValueError:
        parsed = None
    if isinstance(parsed, dict):
        field = parsed.get("error") or parsed.get("message") or ""
        details = parsed.get("details")
        if isinstance(field, dict):
            details = field.get("details", details)
            field = field.get("message") or field.get("code") or ""
        raw = str(field)
        if details:
            raw += f" (details: {_squash(json.dumps(details, ensure_ascii=False))[:_ERROR_BODY_CHARS]})"
        return _squash(raw)[: 2 * _ERROR_BODY_CHARS]
    return _squash(raw)[:_ERROR_BODY_CHARS]


def _squash(text: str) -> str:
    return " ".join(text.split())
