"""The research seam: Linkup search, Hugging Face Hub and dataset viewer, the on-disk cache, fake answers.

No test touches the network: every live-mode test patches research._http. The Hub and viewer fixtures in
tests/fixtures/research/ were recorded on 2026-11-14; linkup_structured_from_docs.json was built by hand from
the documented response shape (see its "_note").
Run: SENSE_FAKE=1 .venv/bin/python -m pytest -q tests/test_agent_research.py
"""

from __future__ import annotations

import io
import json
import urllib.error
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
    def http(method, url, *, body=None, headers=None, timeout):
        raise exc

    return http


def returning(data: bytes, calls: list | None = None):
    def http(method, url, *, body=None, headers=None, timeout):
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

    def http(method, url, *, body=None, headers=None, timeout):
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


@pytest.mark.parametrize("code", [404, 401])  # the Hub answers 401, not 404, for a missing repo without a token
def test_hub_404_is_none(monkeypatch, live_mode, code):
    monkeypatch.setattr(research, "_http", raising(http_error(code, fixture("hub_model_missing_http401.json"))))
    assert research.hub_model("example/does-not-exist") is None
    assert research.hub_dataset("example/does-not-exist") is None


def test_unreachable_hub_with_no_cache_is_none_and_a_cached_answer_is_served(monkeypatch, live_mode):
    monkeypatch.setattr(research, "_http", raising(urllib.error.URLError("network down")))
    assert research.hub_model("laion/clap-htsat-unfused") is None
    monkeypatch.setattr(research, "_http", returning(fixture("hub_model_laion_clap-htsat-unfused.json")))
    live = research.hub_model("laion/clap-htsat-unfused")
    monkeypatch.setattr(research, "_http", raising(TimeoutError("timed out")))
    assert research.hub_model("laion/clap-htsat-unfused") == live


def test_a_repo_that_has_gone_is_none_even_with_a_cached_answer(monkeypatch, live_mode):
    monkeypatch.setattr(research, "_http", returning(fixture("hub_model_laion_clap-htsat-unfused.json")))
    assert research.hub_model("laion/clap-htsat-unfused") is not None
    monkeypatch.setattr(research, "_http", raising(http_error(404)))
    assert research.hub_model("laion/clap-htsat-unfused") is None


def test_a_malformed_hub_id_is_none_without_a_request(monkeypatch, live_mode):
    monkeypatch.setattr(research, "_http", boom)
    assert research.hub_model("../../etc/passwd") is None
    assert research.hub_dataset("not a hub id") is None


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
