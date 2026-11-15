"""The research seam: Linkup search, Hugging Face Hub and dataset viewer, the on-disk cache, fake answers.

No external network is used. Transport integration tests use loopback HTTP servers;
other live-mode tests patch research._http. The Hub and viewer fixtures in
tests/fixtures/research/ were recorded on 2026-11-14; linkup_structured_from_docs.json was built by hand from
the documented response shape (see its "_note").
Run: SENSE_FAKE=1 .venv/bin/python -m pytest -q tests/test_agent_research.py
"""

from __future__ import annotations

import io
import json
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from agent import research

FIXTURES = Path(__file__).parent / "fixtures" / "research"
SPECIALIST_SCHEMA_LIKE = {"type": "object", "properties": {"specialists": {"type": "array"}}}
DATASET_SCHEMA_LIKE = {"type": "object", "properties": {"datasets": {"type": "array"}}}


def fixture(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def boom(*args, **kwargs):
    raise AssertionError("_http must not be called")


def http_error(code: int, body: bytes = b"") -> urllib.error.HTTPError:
    return urllib.error.HTTPError("https://example.org/x", code, "error", {}, io.BytesIO(body))


def raising(exc: Exception):
    def http(method, url, *, body=None, headers=None, timeout, **limits):
        raise exc

    return http


def returning(data: bytes, calls: list | None = None):
    def http(method, url, *, body=None, headers=None, timeout, **limits):
        if calls is not None:
            calls.append((method, url, body, headers))
        return data

    return http


@pytest.fixture
def live_mode(monkeypatch, tmp_path):
    monkeypatch.setenv("SENSE_FAKE", "0")
    monkeypatch.setenv("SENSE_HOME", str(tmp_path))
    monkeypatch.setenv("LINKUP_API_KEY", "test-key")
    return tmp_path


# ----------------------------------------------------------------------------- Linkup search
def test_linkup_request_is_exactly_as_documented(monkeypatch, tmp_path, live_mode):
    calls = []

    def http(method, url, *, body=None, headers=None, timeout, **limits):
        calls.append((method, url, body, headers))
        return json.dumps({"data": {"specialists": []}, "sources": []}).encode()

    monkeypatch.setattr(research, "_http", http)
    research.search_structured(
        "q?", {"type": "object", "properties": {"specialists": {"type": "array"}}}, include_domains=["huggingface.co"]
    )
    method, url, body, headers = calls[0]
    assert (method, url) == ("POST", "https://api.linkup.so/v1/search")
    assert headers["Authorization"] == "Bearer test-key"
    assert body["outputType"] == "structured" and body["includeSources"] is True and body["depth"] == "standard"
    assert json.loads(body["structuredOutputSchema"])["type"] == "object"
    assert body["includeDomains"] == ["huggingface.co"]
    assert body["q"] == "q?" and body["maxResults"] == 8


def test_include_domains_is_left_out_when_not_given(monkeypatch, live_mode):
    calls = []
    monkeypatch.setattr(research, "_http", returning(b'{"data": {}, "sources": []}', calls))
    research.search_structured("q", SPECIALIST_SCHEMA_LIKE)
    assert "includeDomains" not in calls[0][2]


def test_live_answer_is_cached_and_served_when_the_network_fails(monkeypatch, live_mode):
    monkeypatch.setattr(research, "_http", returning(fixture("linkup_structured_from_docs.json")))
    live = research.search_structured("which encoders?", SPECIALIST_SCHEMA_LIKE)
    assert live.origin == "live"
    assert live.data["specialists"][0]["hub_id"] == "laion/clap-htsat-unfused"
    assert [set(s) for s in live.sources] == [{"name", "url", "snippet"}] * 2
    assert len(live.sources[0]["snippet"]) == research.SNIPPET_CHARS
    assert live.sources[1]["snippet"] == "Short illustrative content."

    monkeypatch.setattr(research, "_http", raising(urllib.error.URLError("network down")))
    cached = research.search_structured("which encoders?", SPECIALIST_SCHEMA_LIKE)
    assert cached.origin == "cache"
    assert cached.data == live.data and cached.sources == live.sources


def test_the_cache_never_holds_the_key(monkeypatch, live_mode):
    monkeypatch.setattr(research, "_http", returning(fixture("linkup_structured_from_docs.json")))
    research.search_structured("q", SPECIALIST_SCHEMA_LIKE)
    files = list((live_mode / "research").glob("*.json"))
    assert len(files) == 1 and len(files[0].stem) == 16
    assert "test-key" not in files[0].read_text() and "Authorization" not in files[0].read_text()


def test_no_key_and_no_cache_is_unavailable(monkeypatch, live_mode):
    monkeypatch.delenv("LINKUP_API_KEY")
    monkeypatch.setattr(research, "_http", boom)
    with pytest.raises(research.ResearchUnavailable, match="LINKUP_API_KEY"):
        research.search_structured("q", SPECIALIST_SCHEMA_LIKE)


def test_http_429_names_rate_limit_or_credit(monkeypatch, live_mode):
    monkeypatch.setattr(research, "_http", raising(http_error(429)))
    with pytest.raises(research.ResearchUnavailable, match="rate limit or credit") as info:
        research.search_structured("q", SPECIALIST_SCHEMA_LIKE)
    assert "test-key" not in str(info.value)


def test_an_error_body_never_leaks_the_key(monkeypatch, live_mode):
    monkeypatch.setattr(research, "_http", raising(http_error(401, b'{"error": "bad key test-key"}')))
    with pytest.raises(research.ResearchUnavailable) as info:
        research.search_structured("q", SPECIALIST_SCHEMA_LIKE)
    assert "test-key" not in str(info.value) and "401" in str(info.value)


# ----------------------------------------------------------------------------- fake mode
def test_fake_mode_never_touches_the_network(monkeypatch, tmp_path):
    monkeypatch.setenv("SENSE_FAKE", "1")
    monkeypatch.setattr(research, "_http", boom)
    r = research.search_structured("q", SPECIALIST_SCHEMA_LIKE)
    assert r.origin == "fake" and r.data["specialists"][0]["hub_id"] == "laion/clap-htsat-unfused"
    assert research.hub_model("laion/clap-htsat-unfused")["id"] == "laion/clap-htsat-unfused"
    assert research.hub_model("example/does-not-exist") is None
    assert research.hub_dataset("example/fake-sounds")["id"] == "example/fake-sounds"
    assert research.hub_dataset("example/does-not-exist") is None
    with pytest.raises(research.ResearchUnavailable, match="fake mode has no network"):
        research.viewer("splits", dataset="example/fake-sounds")
    with pytest.raises(research.ResearchUnavailable, match="fake mode has no network"):
        research.download("https://example.org/x.wav", tmp_path / "x.wav")
    assert not (tmp_path / "x.wav").exists()


def test_fake_answers_read_as_fake(monkeypatch):
    monkeypatch.setenv("SENSE_FAKE", "1")
    specialists = research.search_structured("q", SPECIALIST_SCHEMA_LIKE)
    datasets = research.search_structured("q", DATASET_SCHEMA_LIKE)
    items = specialists.data["specialists"]
    assert [i["hub_id"] for i in items] == ["laion/clap-htsat-unfused", "example/unrunnable-audio-model"]
    assert items[0]["value"] == 0.90 and items[1]["value"] is None
    assert [i["hub_id"] for i in datasets.data["datasets"]] == ["example/fake-sounds"]
    for item in items + datasets.data["datasets"]:
        assert item["statement"].startswith("Fake claim for tests")
        assert item["source_url"].startswith("https://example.org/fake/")
    for source in specialists.sources + datasets.sources:
        assert source["url"].startswith("https://example.org/fake/")


def test_fake_mode_is_decided_at_call_time(monkeypatch, live_mode):
    monkeypatch.setattr(research, "_http", returning(fixture("linkup_structured_from_docs.json")))
    assert research.search_structured("q", SPECIALIST_SCHEMA_LIKE).origin == "live"
    monkeypatch.setenv("SENSE_FAKE", "1")
    assert research.search_structured("q", SPECIALIST_SCHEMA_LIKE).origin == "fake"


def test_fake_answers_are_copies(monkeypatch):
    monkeypatch.setenv("SENSE_FAKE", "1")
    research.search_structured("q", SPECIALIST_SCHEMA_LIKE).data["specialists"].clear()
    research.hub_model("laion/clap-htsat-unfused")["licence"] = "changed"
    assert research.search_structured("q", SPECIALIST_SCHEMA_LIKE).data["specialists"]
    assert research.hub_model("laion/clap-htsat-unfused")["licence"] != "changed"


# ----------------------------------------------------------------------------- Hugging Face Hub
def test_hub_parsers_read_the_recorded_fixtures(monkeypatch, live_mode):
    calls = []
    monkeypatch.setattr(research, "_http", returning(fixture("hub_model_laion_clap-htsat-unfused.json"), calls))
    model = research.hub_model("laion/clap-htsat-unfused")
    assert calls[0][:2] == ("GET", "https://huggingface.co/api/models/laion/clap-htsat-unfused")
    assert model == {
        "id": "laion/clap-htsat-unfused",
        "pipeline_tag": "feature-extraction",
        "licence": "apache-2.0",
        "downloads": 931344,
    }

    monkeypatch.setattr(research, "_http", returning(fixture("hub_dataset_renumics_esc50.json"), calls))
    dataset = research.hub_dataset("renumics/esc50")
    assert calls[1][1] == "https://huggingface.co/api/datasets/renumics/esc50"
    assert dataset == {"id": "renumics/esc50", "licence": "cc-by-nc-2.0", "downloads": 96}
    assert isinstance(model["pipeline_tag"], str) and isinstance(dataset["licence"], str)


def test_a_dataset_with_no_licence_on_the_hub_has_licence_none(monkeypatch, live_mode):
    monkeypatch.setattr(research, "_http", returning(fixture("hub_dataset_ashraq_esc50.json")))
    assert research.hub_dataset("ashraq/esc50") == {"id": "ashraq/esc50", "licence": None, "downloads": 7307}


def test_licence_falls_back_to_the_licence_tag(monkeypatch, live_mode):
    body = {"id": "a/b", "tags": ["audio", "license:mit"], "downloads": 3}
    monkeypatch.setattr(research, "_http", returning(json.dumps(body).encode()))
    assert research.hub_dataset("a/b")["licence"] == "mit"


@pytest.mark.parametrize("code", [404])
def test_hub_404_is_none(monkeypatch, live_mode, code):
    monkeypatch.setattr(research, "_http", raising(http_error(code, fixture("hub_model_missing_http401.json"))))
    assert research.hub_model("example/does-not-exist") is None
    assert research.hub_dataset("example/does-not-exist") is None


def test_unreachable_hub_raises_without_cache_and_serves_cached_answer(monkeypatch, live_mode):
    monkeypatch.setattr(research, "_http", raising(urllib.error.URLError("network down")))
    with pytest.raises(research.ResearchUnavailable, match="network down"):
        research.hub_model("laion/clap-htsat-unfused")
    monkeypatch.setattr(research, "_http", returning(fixture("hub_model_laion_clap-htsat-unfused.json")))
    live = research.hub_model("laion/clap-htsat-unfused")
    monkeypatch.setattr(research, "_http", raising(TimeoutError("timed out")))
    assert research.hub_model("laion/clap-htsat-unfused") == live


def test_a_repo_that_has_gone_is_none_even_with_a_cached_answer(monkeypatch, live_mode):
    monkeypatch.setattr(research, "_http", returning(fixture("hub_model_laion_clap-htsat-unfused.json")))
    assert research.hub_model("laion/clap-htsat-unfused") is not None
    monkeypatch.setattr(research, "_http", raising(http_error(404)))
    assert research.hub_model("laion/clap-htsat-unfused") is None


def test_a_malformed_hub_id_is_unavailable_without_a_request(monkeypatch, live_mode):
    monkeypatch.setattr(research, "_http", boom)
    with pytest.raises(research.ResearchUnavailable, match="invalid"):
        research.hub_model("../../etc/passwd")
    with pytest.raises(research.ResearchUnavailable, match="invalid"):
        research.hub_dataset("not a hub id")


# ----------------------------------------------------------------------------- dataset viewer
def test_viewer_builds_the_url_and_returns_the_recorded_rows(monkeypatch, live_mode):
    calls = []
    monkeypatch.setattr(research, "_http", returning(fixture("viewer_rows_renumics_esc50.json"), calls))
    rows = research.viewer("rows", dataset="renumics/esc50", config="default", split="train", offset=0, length=3)
    method, url, _, _ = calls[0]
    assert method == "GET"
    assert url == (
        "https://datasets-server.huggingface.co/rows?dataset=renumics%2Fesc50&config=default&split=train"
        "&offset=0&length=3"
    )
    label = next(f for f in rows["features"] if f["name"] == "label")
    assert label["type"]["_type"] == "ClassLabel" and label["type"]["names"][0] == "dog"
    assert rows["rows"][0]["row"]["audio"][0]["src"].endswith("/audio.wav")
    assert rows["num_rows_total"] == 2000


def test_viewer_error_names_the_cause(monkeypatch, live_mode):
    body = fixture("viewer_rows_ashraq_esc50_http500.json")
    monkeypatch.setattr(research, "_http", raising(http_error(500, body)))
    with pytest.raises(research.ResearchUnavailable, match="HTTP 500.*Scan size limit"):
        research.viewer("rows", dataset="ashraq/esc50", config="default", split="train", offset=0, length=3)


def test_viewer_refuses_an_endpoint_that_is_not_a_plain_name(monkeypatch, live_mode):
    monkeypatch.setattr(research, "_http", boom)
    with pytest.raises(research.ResearchUnavailable):
        research.viewer("../admin", dataset="x/y")


# ----------------------------------------------------------------------------- download
def test_download_writes_the_bytes_and_leaves_no_temporary_file(monkeypatch, live_mode, tmp_path):
    monkeypatch.setattr(research, "_http", returning(b"RIFF0123WAVE"))
    dest = tmp_path / "out" / "dog" / "0.wav"
    assert research.download("https://example.org/a.wav", dest) == 12
    assert dest.read_bytes() == b"RIFF0123WAVE"
    assert [p.name for p in dest.parent.iterdir()] == ["0.wav"]


def test_download_stops_at_max_bytes(monkeypatch, live_mode, tmp_path):
    monkeypatch.setattr(research, "_http", returning(b"x" * 30))
    dest = tmp_path / "out" / "0.wav"
    with pytest.raises(research.ResearchUnavailable, match="10"):
        research.download("https://example.org/a.wav", dest, max_bytes=10)
    assert not dest.exists()
    assert not (tmp_path / "out").exists() or list((tmp_path / "out").iterdir()) == []


def test_a_failed_download_leaves_nothing_at_dest(monkeypatch, live_mode, tmp_path):
    monkeypatch.setattr(research, "_http", raising(urllib.error.URLError("reset")))
    dest = tmp_path / "0.wav"
    with pytest.raises(research.ResearchUnavailable, match="reset"):
        research.download("https://example.org/a.wav", dest)
    assert not dest.exists()


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://example.org/a.wav", "not a url"])
def test_download_only_fetches_http_urls(monkeypatch, live_mode, tmp_path, url):
    monkeypatch.setattr(research, "_http", boom)
    with pytest.raises(research.ResearchUnavailable):
        research.download(url, tmp_path / "x.wav")


# ----------------------------------------------------------------------------- fix round 1
def redirect(from_url: str, to_url: str, headers: dict | None = None):
    req = urllib.request.Request(from_url, headers=headers or {"Authorization": "Bearer test-key"}, method="GET")
    return research._RedirectGuard().redirect_request(req, io.BytesIO(), 302, "Found", {}, to_url)


def test_a_redirect_to_ftp_is_refused():
    with pytest.raises(research.ResearchUnavailable, match="ftp"):
        redirect("https://huggingface.co/a", "ftp://example.org/a")


def test_a_redirect_to_another_host_drops_authorization_and_the_same_host_keeps_it():
    assert redirect("https://api.linkup.so/v1/a", "https://api.linkup.so/v1/b").get_header("Authorization") == (
        "Bearer test-key"
    )
    assert not redirect("https://api.linkup.so/v1/a", "https://evil.example/b").has_header("Authorization")
    assert not redirect("https://api.linkup.so/v1/a", "https://api.linkup.so:8443/b").has_header("Authorization")
    assert not redirect("https://api.linkup.so/v1/a", "http://api.linkup.so/v1/b").has_header("Authorization")
    assert redirect("https://api.linkup.so/v1/a", "https://evil.example/b").get_header("User-agent") is None


def test_http_uses_the_redirect_guard(monkeypatch):
    seen = []

    class Opener:
        def open(self, request, timeout):
            raise urllib.error.URLError("stop here")

    def build_opener(*handlers):
        seen.extend(handlers)
        return Opener()

    monkeypatch.setattr(research.urllib.request, "build_opener", build_opener)
    with pytest.raises(urllib.error.URLError):
        direct_transport("GET", "https://huggingface.co/api/models/a/b", timeout=1)
    assert seen == [research._RedirectGuard]


@pytest.mark.parametrize("url", ["https://example.org/café.wav", "http://[::1/a.wav", "https:///a.wav"])
def test_download_of_an_invalid_url_is_unavailable(monkeypatch, live_mode, tmp_path, url):
    monkeypatch.setattr(research, "_http", boom)
    dest = tmp_path / "x.wav"
    with pytest.raises(research.ResearchUnavailable):
        research.download(url, dest)
    assert not dest.exists()


def test_download_turns_a_value_error_from_http_into_unavailable(monkeypatch, live_mode, tmp_path):
    monkeypatch.setattr(research, "_http", raising(UnicodeEncodeError("ascii", "é", 0, 1, "no")))
    with pytest.raises(research.ResearchUnavailable):
        research.download("https://example.org/a.wav", tmp_path / "x.wav")
    assert not (tmp_path / "x.wav").exists()


@pytest.mark.parametrize("key", ["test key", "test-key\n", "test\x07key", "\ttest-key"])
def test_a_malformed_key_is_refused_without_showing_it(monkeypatch, live_mode, key):
    monkeypatch.setenv("LINKUP_API_KEY", key)
    monkeypatch.setattr(research, "_http", boom)
    with pytest.raises(research.ResearchUnavailable, match="LINKUP_API_KEY is malformed") as info:
        research.search_structured("q", SPECIALIST_SCHEMA_LIKE)
    assert "test" not in str(info.value)


def test_a_200_without_data_is_not_cached_and_names_the_keys(monkeypatch, live_mode):
    answer = {"answer": "x", "sources": [], **{f"k{i:02d}": i for i in range(12)}}
    monkeypatch.setattr(research, "_http", returning(json.dumps(answer).encode()))
    with pytest.raises(research.ResearchUnavailable, match="answer, k00, k01") as info:
        research.search_structured("q", SPECIALIST_SCHEMA_LIKE)
    assert "k09" not in str(info.value)  # at most 10 keys, sorted: answer, k00..k08
    assert not (live_mode / "research").exists() or not list((live_mode / "research").glob("*.json"))


def test_linkup_error_details_are_named_and_the_key_is_masked(monkeypatch, live_mode):
    body = {
        "statusCode": 400,
        "error": {
            "code": "VALIDATION_ERROR",
            "message": "Validation failed",
            "details": [{"field": "depth", "message": "depth must be standard or deep, not test-key"}],
        },
    }
    monkeypatch.setattr(research, "_http", raising(http_error(400, json.dumps(body).encode())))
    with pytest.raises(research.ResearchUnavailable, match="Validation failed.*depth must be") as info:
        research.search_structured("q", SPECIALIST_SCHEMA_LIKE)
    assert "test-key" not in str(info.value)


def test_an_error_body_is_read_at_most_4096_bytes(monkeypatch, live_mode):
    sizes = []

    class Body(io.BytesIO):
        def read(self, size=-1):
            sizes.append(size)
            return super().read(size)

    exc = urllib.error.HTTPError("https://x", 500, "error", {}, Body(b"y" * 100_000))
    monkeypatch.setattr(research, "_http", raising(exc))
    with pytest.raises(research.ResearchUnavailable):
        research.viewer("rows", dataset="a/b")
    assert sizes and all(0 <= s <= 4096 for s in sizes)


def test_fake_clap_has_the_live_pipeline_tag(monkeypatch):
    monkeypatch.setenv("SENSE_FAKE", "1")
    assert research.hub_model("laion/clap-htsat-unfused")["pipeline_tag"] == "feature-extraction"


@pytest.mark.parametrize("sources", [1, {}, "bad"])
def test_malformed_search_sources_are_unavailable(monkeypatch, live_mode, sources):
    monkeypatch.setattr(research, "_http", returning(json.dumps({"data": {}, "sources": sources}).encode()))
    with pytest.raises(research.ResearchUnavailable, match="sources"):
        research.search_structured("q", SPECIALIST_SCHEMA_LIKE)


@pytest.mark.parametrize("error", [TimeoutError("timed out"), http_error(503)])
@pytest.mark.parametrize("lookup", [research.hub_model, research.hub_dataset])
def test_hub_outage_is_unknown_not_missing(monkeypatch, live_mode, error, lookup):
    monkeypatch.setattr(research, "_http", raising(error))
    with pytest.raises(research.ResearchUnavailable, match="timed out|503"):
        lookup("example/unavailable")


class ControlledResponse:
    """Transport fixture exposing bounded reads, elapsed time and socket timeouts."""
    def __init__(self, payload, clock=None, delay=0, fail_after=None):
        self.payload, self.clock, self.delay, self.fail_after = payload, clock, delay, fail_after
        self.received, self.requests, self.timeouts, self.closed = 0, [], [], False
        from types import SimpleNamespace
        self.fp = SimpleNamespace(raw=SimpleNamespace(_sock=self))

    def settimeout(self, timeout):
        self.timeouts.append(timeout)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def read(self, size):
        return self.read1(size)

    def read1(self, size):
        self.requests.append(size)
        if self.clock is not None:
            remaining = self.timeouts[-1] if self.timeouts else float("inf")
            self.clock[0] += min(self.delay, remaining)
            if self.delay >= remaining:
                raise TimeoutError("timed out during stream")
        if self.fail_after is not None and self.received >= self.fail_after:
            raise ConnectionResetError("stream reset")
        chunk = self.payload[self.received:self.received + min(size, 4)]
        self.received += len(chunk)
        return chunk


def direct_transport(method, url, *, timeout, max_bytes=research.MAX_RESPONSE_BYTES, **kwargs):
    """Exercise worker internals with controlled responses; no process supervisor."""
    from agent import _http_transport
    output = io.BytesIO()
    _http_transport.fetch(method, url, timeout=timeout, max_bytes=max_bytes, sink=output.write, **kwargs)
    return output.getvalue()


def transport(monkeypatch, response):
    monkeypatch.setattr(research, "_http", direct_transport)
    from types import SimpleNamespace
    monkeypatch.setattr(research.urllib.request, "build_opener",
                        lambda *a: SimpleNamespace(open=lambda *a, **k: response))


def test_download_enforces_byte_limit_in_transport(monkeypatch, live_mode, tmp_path):
    response = ControlledResponse(b"x" * 1000)
    transport(monkeypatch, response)
    with pytest.raises(research.ResearchUnavailable, match="limit|over") as caught:
        research.download("https://example.org/audio.wav", tmp_path / "audio.wav", max_bytes=10)
    assert response.received <= 11 and max(response.requests) <= 11
    assert caught.value.bytes_received == response.received and response.closed
    assert not (tmp_path / "audio.wav").exists()


def test_download_deadline_bounds_slow_stream(monkeypatch, live_mode, tmp_path):
    clock = [0.0]
    import time
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    response = ControlledResponse(b"x" * 1000, clock, delay=0.4)
    transport(monkeypatch, response)
    with pytest.raises(research.ResearchUnavailable, match="timed out|deadline") as caught:
        research.download("https://example.org/audio.wav", tmp_path / "audio.wav", timeout=1)
    assert clock[0] <= 1.0 and response.received <= 8 and response.closed
    assert caught.value.bytes_received == response.received
    assert not (tmp_path / "audio.wav").exists()


@pytest.mark.parametrize("code", [401, 403])
def test_hub_access_denial_is_unknown(monkeypatch, live_mode, code):
    monkeypatch.setattr(research, "_http", raising(http_error(code)))
    with pytest.raises(research.ResearchUnavailable, match=str(code)):
        research.hub_model("example/private")


@pytest.mark.parametrize("payload", [b"null", b"[]", b"{}", b'{"error": "maintenance"}', b"not json"])
def test_malformed_hub_record_is_unknown(monkeypatch, live_mode, payload):
    monkeypatch.setattr(research, "_http", returning(payload))
    with pytest.raises(research.ResearchUnavailable):
        research.hub_dataset("example/dataset")


def test_sample_aborts_partial_transfer_and_cleans_up(monkeypatch, live_mode, tmp_path):
    from agent import data
    response = ControlledResponse(b"x" * 1000, fail_after=4)
    opened = install_sample_transport(monkeypatch, [response])
    with pytest.raises(data.DataUnavailable, match="discarded bytes.*stream reset"):
        data.fetch_sample(data.DatasetCandidate("o/d", "D", "https://example.org"), tmp_path / "sample", "closed")
    assert len(opened) == 1 and response.received == 4 and response.closed
    assert not (tmp_path / "sample").exists()


def test_sample_passes_remaining_byte_budget_to_transport(monkeypatch, live_mode, tmp_path):
    from agent import data
    monkeypatch.setattr(data, "MAX_BYTES", 10)
    success, oversized = ControlledResponse(b"1234"), ControlledResponse(b"x" * 1000)
    opened = install_sample_transport(monkeypatch, [success, oversized])
    with pytest.raises(data.DataUnavailable, match="byte limit"):
        data.fetch_sample(data.DatasetCandidate("o/d", "D", "https://example.org"), tmp_path / "sample", "closed")
    assert len(opened) == 2 and success.received == 4 and oversized.received <= 7
    assert max(oversized.requests) <= 7 and not (tmp_path / "sample").exists()


def test_sample_passes_remaining_deadline_to_stream(monkeypatch, live_mode, tmp_path):
    from agent import data
    clock = [0.0]
    monkeypatch.setattr(data, "_now", lambda: clock[0])
    monkeypatch.setattr(research.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(data, "DEADLINE_S", 1.0)
    response = ControlledResponse(b"x" * 1000, clock, delay=0.3)
    opened = install_sample_transport(monkeypatch, [response], clock)
    with pytest.raises(data.DataUnavailable):
        data.fetch_sample(data.DatasetCandidate("o/d", "D", "https://example.org"), tmp_path / "sample", "closed")
    assert len(opened) == 1 and clock[0] <= 1.0 and response.closed
    assert response.timeouts[0] == pytest.approx(0.6)
    assert not (tmp_path / "sample").exists()


def install_sample_transport(monkeypatch, responses, clock=None):
    monkeypatch.setattr(research, "_http", direct_transport)
    from types import SimpleNamespace
    opened = []
    def viewer(endpoint, **params):
        if clock is not None:
            clock[0] += 0.2
        if endpoint == "splits":
            return {"splits": [{"config": "default", "split": "train"}]}
        return {"features": [{"name": "audio", "type": {"_type": "Audio"}},
                              {"name": "label", "type": {"_type": "ClassLabel", "names": ["a", "b"]}}],
                "num_rows_total": 40,
                "rows": [{"row_idx": i, "row": {"label": i % 2, "audio": {"src": f"https://example.org/{i}.wav"}}}
                         for i in range(40)]}
    def open_response(request, timeout):
        opened.append((request, timeout))
        assert len(opened) <= len(responses), "a failed transfer must not start another download"
        return responses[len(opened) - 1]
    monkeypatch.setattr(research, "viewer", viewer)
    monkeypatch.setattr(research.urllib.request, "build_opener", lambda *a: SimpleNamespace(open=open_response))
    return opened


def test_download_error_body_is_not_an_unbudgeted_transfer(monkeypatch, live_mode, tmp_path):
    class UnreadableBody(io.BytesIO):
        def read(self, *args):
            pytest.fail("an error body must not be read outside the download budget")
    body = UnreadableBody(b"x" * 1000)
    error = urllib.error.HTTPError("https://example.org/x", 503, "unavailable", {}, body)
    monkeypatch.setattr(research, "_http", raising(error))
    with pytest.raises(research.ResearchUnavailable, match="503") as caught:
        research.download("https://example.org/x", tmp_path / "audio.wav", max_bytes=10)
    assert caught.value.bytes_received == 0 and body.closed
    assert not (tmp_path / "audio.wav").exists()


@pytest.fixture
def local_http():
    """Real urllib ownership and process supervision, with no external requests."""
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    servers = []
    def start(respond):
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                try:
                    respond(self)
                except (BrokenPipeError, ConnectionResetError):
                    pass  # The bounded client deliberately closes slow streams.
            do_POST = do_GET
            def log_message(self, *args):
                pass
        try:
            server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        except PermissionError:
            pytest.skip("sandbox denies loopback listeners; run transport checks with local-network permission")
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
        thread.start()
        servers.append((server, thread))
        return f"http://127.0.0.1:{server.server_port}"
    yield start
    for server, thread in servers:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)
        assert not thread.is_alive()


def send_response(handler, body=b"{}", status=200, **headers):
    handler.send_response(status)
    handler.send_header("Content-Length", str(len(body)))
    for name, value in headers.items():
        handler.send_header(name, value)
    handler.end_headers()
    handler.wfile.write(body)


@pytest.mark.parametrize("body", [b"{}", b"RIFF0123WAVE", b"", b"x" * 150000])
def test_real_content_length_response_succeeds(local_http, live_mode, tmp_path, body):
    url = local_http(lambda h: send_response(h, body))
    dest = tmp_path / "audio.wav"
    assert research.download(url, dest, max_bytes=len(body)) == len(body)
    assert dest.read_bytes() == body


def test_owned_content_length_socket_is_not_reused_after_final_chunk(monkeypatch):
    import http.client
    import socket
    client, server = socket.socketpair()
    try:
        server.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n{}")
        response = http.client.HTTPResponse(client)
        response.begin()
        client.close()  # urllib relinquishes ownership to the response's fp.
        transport(monkeypatch, response)
        assert direct_transport("GET", "http://local/", timeout=1) == b"{}"
    finally:
        client.close()
        server.close()


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
def test_redirect_handler_never_drains_body(status):
    from types import SimpleNamespace
    class NeverRead(io.BytesIO):
        def read(self, *args):
            pytest.fail("redirect body must not be drained")
    body = NeverRead(b"x" * 100000)
    handler = research._RedirectGuard()
    handler.parent = SimpleNamespace(open=lambda *args, **kwargs: "followed")
    request = urllib.request.Request("http://local/a")
    request.timeout = 1
    assert getattr(handler, f"http_error_{status}")(request, body, status, "redirect", {"location": "/b"}) == "followed"
    assert body.closed


def test_redirect_body_not_downloaded_and_authorization_removed(local_http, live_mode):
    import time
    seen = []
    target = local_http(lambda h: (seen.append(h.headers.get("Authorization")), send_response(h)))
    def redirect(h):
        h.send_response(302)
        h.send_header("Location", target)
        h.send_header("Content-Length", "99999999")
        h.end_headers()
        time.sleep(0.7)  # Following must not wait for or consume this body.
    source = local_http(redirect)
    assert research._http("GET", source, headers={"Authorization": "Bearer secret"}, timeout=0.5, max_bytes=2) == b"{}"
    assert seen == [None]


@pytest.mark.parametrize("phase", ["status", "headers", "redirect_headers", "chunk_framing", "body"])
def test_network_deadline_kills_and_reaps_worker(local_http, live_mode, tmp_path, monkeypatch, phase):
    import time
    processes = []
    real_popen = research.subprocess.Popen
    def record_process(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        processes.append(process)
        return process
    monkeypatch.setattr(research.subprocess, "Popen", record_process)
    def drip(h):
        if phase == "redirect_headers" and h.path == "/":
            send_response(h, b"", status=302, Location="/slow")
            return
        if phase == "status":
            wire = b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n{}"
        elif phase in ("headers", "redirect_headers"):
            h.wfile.write(b"HTTP/1.1 200 OK\r\n")
            wire = b"X-Slow: " + b"a" * 40 + b"\r\nContent-Length: 2\r\n\r\n{}"
        elif phase == "chunk_framing":
            h.wfile.write(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n")
            wire = b"0" * 40 + b"1\r\nx\r\n0\r\n\r\n"
        else:
            h.wfile.write(b"HTTP/1.1 200 OK\r\nContent-Length: 40\r\n\r\n")
            wire = b"x" * 40
        for byte in wire:
            h.wfile.write(bytes([byte]))
            h.wfile.flush()
            time.sleep(0.02)
    url = local_http(drip)
    started = time.monotonic()
    with pytest.raises(research.ResearchUnavailable, match="deadline") as caught:
        research.download(url, tmp_path / "audio.wav", timeout=0.2)
    assert time.monotonic() - started < 0.6
    assert caught.value.transfer_incomplete
    assert len(processes) == 1 and processes[0].poll() is not None
    assert not (tmp_path / "audio.wav").exists()


def test_real_oversize_and_truncated_responses_are_rejected(local_http, live_mode, tmp_path):
    url = local_http(lambda h: send_response(h, b"x" * 10000))
    with pytest.raises(research.ResearchUnavailable, match="byte limit") as caught:
        research.download(url, tmp_path / "audio.wav", max_bytes=10)
    assert caught.value.bytes_received == 11
    def truncated(h):
        h.wfile.write(b"HTTP/1.1 200 OK\r\nContent-Length: 10\r\n\r\nabc")
    url = local_http(truncated)
    with pytest.raises(research.ResearchUnavailable, match="IncompleteRead") as caught:
        research.download(url, tmp_path / "audio.wav")
    assert caught.value.bytes_received == 3
    assert not (tmp_path / "audio.wav").exists()


@pytest.mark.parametrize("operation", ["mkdir", "write_bytes", "replace"])
def test_save_failure_preserves_transfer_accounting(monkeypatch, live_mode, tmp_path, operation):
    monkeypatch.setattr(research, "_http", returning(b"audio"))
    def fail(*args, **kwargs):
        raise OSError("disk full")
    if operation == "replace":
        monkeypatch.setattr(research.os, operation, fail)
    else:
        monkeypatch.setattr(Path, operation, fail)
    with pytest.raises(research.ResearchUnavailable, match="disk full") as caught:
        research.download("http://local/a", tmp_path / "audio.wav")
    assert caught.value.bytes_received == 5
    assert caught.value.transfer_incomplete
    assert not list(tmp_path.glob("*.part"))
    assert not (tmp_path / "audio.wav").exists()


def test_stalled_dns_is_killed_without_leaving_a_worker(monkeypatch):
    """Hang getaddrinfo inside the real child; no DNS or external connection occurs."""
    import time
    real_popen = research.subprocess.Popen
    processes = []
    def stall_dns(args, **kwargs):
        bootstrap = ("import runpy,socket,sys,time; "
                     "socket.getaddrinfo=lambda *a,**k: time.sleep(5); "
                     "runpy.run_path(sys.argv[1],run_name='__main__')")
        process = real_popen(args[:-1] + ["-c", bootstrap, args[-1]], **kwargs)
        processes.append(process)
        return process
    monkeypatch.setattr(research.subprocess, "Popen", stall_dns)
    started = time.monotonic()
    with pytest.raises(research.ResearchUnavailable, match="deadline") as caught:
        research._http("GET", "http://127.0.0.1/", timeout=0.2)
    assert time.monotonic() - started < 0.6
    assert caught.value.transfer_incomplete
    assert len(processes) == 1 and processes[0].poll() is not None


def test_real_http_status_and_cache_fallback(local_http, live_mode):
    state = {"status": 200}
    url = local_http(lambda h: send_response(h, b'{"id":"o/d"}', status=state["status"]))
    assert research._fetch_json("GET", url, timeout=1) == ({"id": "o/d"}, "live")
    state["status"] = 503
    assert research._fetch_json("GET", url, timeout=1) == ({"id": "o/d"}, "cache")
    state["status"] = 404
    assert research._fetch_json("GET", url, timeout=1, absent=(404,)) == (None, "live")
    with pytest.raises(urllib.error.HTTPError) as caught:
        research._http("GET", url, timeout=1)
    assert caught.value.code == 404
    assert caught.value.bytes_received == 0
