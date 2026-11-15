"""Finding labelled probe files on the Hugging Face Hub (agent.hub_probes), offline.

No test touches the network: hub_probes._http_get is replaced by FakeHub, which answers the Hub search, the dataset
viewer's /splits and /rows, and media file URLs from small in-memory tables shaped like the real answers (the
trimmed real responses in tests/fixtures/hub/ pin those shapes). ask_json answers None in fake mode, so the rule
fallbacks run unless a test replaces it.
Run: SENSE_FAKE=1 .venv/bin/python -m pytest -q tests/test_hub_probes.py
"""

from __future__ import annotations

import hashlib
import json
import shutil
import struct
import subprocess
import time
import urllib.parse
import zlib
from pathlib import Path

import pytest

from agent import hub_probes
from agent.hub_probes import (
    FetchReport,
    HubCandidate,
    Mapping,
    ProbesUnavailable,
    choose_mapping,
    fetch_probes,
    find_probes,
    safe_label,
    search_hub,
)

FIXTURES = Path(__file__).parent / "fixtures" / "hub"
ASSETS = "https://datasets-server.huggingface.co/assets"


class Dataset:
    def __init__(self, features, row, total, *, splits=(("default", "train"),), truncated=None, rows_error=None,
                 first_rows=True, flaky=None, sized=False):
        self.features, self.row, self.total = features, row, total
        self.splits, self.truncated, self.rows_error = splits, truncated or {}, rows_error
        self.first_rows = first_rows  # whether /first-rows answers (it does even when /rows fails, as on the Hub)
        self.flaky = flaky or {}  # /rows offset -> HTTP codes its next requests fail with, one per request
        self.sized = sized  # whether /size answers

    def page(self, offset, length):
        return [{"row_idx": i, "row": self.row(i), "truncated_cells": self.truncated.get(i, [])}
                for i in range(offset, min(offset + length, self.total))]


class FakeHub:
    """Stands in for _http_get: the Hub search, the viewer's /splits and /rows, and file URLs."""

    def __init__(self):
        self.search: dict[str, list] = {}
        self.datasets: dict[str, Dataset] = {}
        self.files: dict[str, bytes] = {}
        self.calls: list[str] = []

    def __call__(self, url, *, timeout=20.0, max_bytes=50_000_000):
        assert timeout > 0
        self.calls.append(url)
        parts = urllib.parse.urlsplit(url)
        query = {k: v[0] for k, v in urllib.parse.parse_qs(parts.query).items()}
        if parts.netloc == "huggingface.co" and parts.path == "/api/datasets":
            assert query["full"] == "true"
            return json.dumps(self.search.get(query["search"], [])).encode()
        if parts.netloc == "datasets-server.huggingface.co" and parts.path in ("/splits", "/rows", "/first-rows",
                                                                               "/size"):
            ds = self.datasets.get(query["dataset"])
            if ds is None:
                raise hub_probes._HttpError("http_404", "HTTP 404 from datasets-server.huggingface.co: not found")
            if parts.path == "/splits":
                splits = [{"dataset": query["dataset"], "config": c, "split": s} for c, s in ds.splits]
                return json.dumps({"splits": splits, "pending": [], "failed": []}).encode()
            if parts.path == "/first-rows":
                if not ds.first_rows:
                    raise hub_probes._HttpError("http_500", "HTTP 500 from the viewer: Unexpected error.")
                return json.dumps({"dataset": query["dataset"], "features": ds.features, "rows": ds.page(0, 100),
                                   "truncated": ds.total > 100}).encode()
            if parts.path == "/size":
                if not ds.sized:
                    raise hub_probes._HttpError("http_404", "HTTP 404 from datasets-server.huggingface.co")
                splits = [{"dataset": query["dataset"], "config": c, "split": s, "num_rows": ds.total}
                          for c, s in ds.splits]
                return json.dumps({"size": {"splits": splits}, "pending": [], "failed": []}).encode()
            if ds.rows_error:
                raise hub_probes._HttpError("http_500", f"HTTP 500 from the viewer: {ds.rows_error}")
            offset, length = int(query["offset"]), int(query["length"])
            if ds.flaky.get(offset):
                code = ds.flaky[offset].pop(0)
                raise hub_probes._HttpError(f"http_{code}", f"HTTP {code} from datasets-server.huggingface.co")
            return json.dumps({"features": ds.features, "rows": ds.page(offset, length), "num_rows_total": ds.total,
                               "num_rows_per_page": 100, "partial": False}).encode()
        if url in self.files:
            data = self.files[url]
            if len(data) > max_bytes:
                raise hub_probes._HttpError("too_large", f"over the {max_bytes:,} byte cap")
            return data
        raise hub_probes._HttpError("http_404", f"HTTP 404 from {parts.netloc}")

    def fetched(self, prefix):
        return [u for u in self.calls if u.startswith(prefix)]


@pytest.fixture
def hub(monkeypatch):
    fake = FakeHub()
    monkeypatch.setattr(hub_probes, "_http_get", fake)
    monkeypatch.setattr(hub_probes, "_sleep", lambda s: None)
    monkeypatch.setattr(hub_probes, "_ffmpeg", lambda: None)  # the fake files are not real media
    return fake


def value(name, dtype="string"):
    return {"name": name, "type": {"dtype": dtype, "_type": "Value"}}


def class_label(name, names):
    return {"name": name, "type": {"names": list(names), "_type": "ClassLabel"}}


def media(name, kind):
    return {"name": name, "type": {"_type": kind}}


def string_list(name):
    return {"name": name, "type": {"feature": {"dtype": "string", "_type": "Value"}, "_type": "List"}}


def item(hub_id, downloads=0, **extra):
    return {"id": hub_id, "downloads": downloads, "likes": 0, "gated": False, "private": False, "disabled": False,
            "tags": extra.pop("tags", []), "cardData": extra.pop("cardData", {})} | extra


def cand(hub_id, licence="mit"):
    return HubCandidate(hub_id, hub_id, 1, 0, licence, [], f"https://huggingface.co/datasets/{hub_id}")


def searched(hub) -> list[str]:
    return [urllib.parse.parse_qs(urllib.parse.urlsplit(u).query)["search"][0] for u in hub.calls]


def files_in(folder: Path) -> dict[str, list[str]]:
    return {d.name: sorted(f.name for f in d.iterdir()) for d in sorted(folder.iterdir()) if d.is_dir()}


# ----------------------------------------------------------------------------- search
def test_fixtures_are_real_shapes():
    search = json.loads((FIXTURES / "search_music_genre.json").read_text())
    found = [hub_probes._candidate(x) for x in search]
    assert [c.hub_id for c in found if c] == [x["id"] for x in search]
    assert found[0].licence == "mit"  # from cardData.license
    page = json.loads((FIXTURES / "rows_music_genres.json").read_text())
    columns = hub_probes._columns(page["features"], "lewtun/music_genres")
    assert columns["audio"].media_type == "Audio" and columns["genre"].kind == "string"
    rows = hub_probes._rows(page)
    assert hub_probes._src(rows[0][1]["audio"]).startswith(ASSETS + "/lewtun/music_genres/")  # a list of one cell
    assert hub_probes._label_fit(columns["genre"], rows) == ("string", None)


def test_search_rule_terms_order_dedupe_and_drop_gated(hub):
    hub.search = {
        "cities": [item("a/world-cities", 10, cardData={"license": "cc-by-4.0"}),
                   item("g/cities-gated", 99999, gated="manual"), item("p/cities-private", 99999, private=True),
                   item("v/cities-noviewer", 99999, cardData={"viewer": False}),
                   item("y/cities-less", 5), item("z/cities-more", 50, tags=["license:odbl"])],
        "country": [item("a/world-cities", 10), item("b/misc-data", 5000)],
        "geonames": [item("d/disabled-geonames", 9999, disabled=True), item("c/geonames-countries", 1,
                                                                            cardData={"license": ["mit", "cc0-1.0"]})],
    }
    found = search_hub("Which country is this place in?", "coordinates")
    # "place" only names the input, so it is searched only inside the first query
    assert searched(hub) == ["country place", "cities", "country", "city", "geonames", "countries"]
    ids = [c.hub_id for c in found]
    # two topic words in the name beat one; then more queries finding it; then downloads
    assert ids == ["c/geonames-countries", "a/world-cities", "z/cities-more", "y/cities-less", "b/misc-data"]
    assert len(ids) == len(set(ids))
    by_id = {c.hub_id: c for c in found}
    assert by_id["a/world-cities"].licence == "cc-by-4.0"
    assert by_id["z/cities-more"].licence == "odbl"
    assert by_id["c/geonames-countries"].licence == "mit, cc0-1.0"
    assert by_id["b/misc-data"].licence is None and by_id["a/world-cities"].url.endswith("/datasets/a/world-cities")
    assert all(not c.gated for c in found)


def test_search_uses_llm_queries_and_ignores_nonsense(hub, monkeypatch):
    answer = {"queries": ["gtzan", " music  genre ", 7, "x" * 99]}
    monkeypatch.setattr(hub_probes.llm, "ask_json", lambda *a, **k: answer)
    search_hub("Make it recognise music genres", "audio")
    assert searched(hub) == ["gtzan", "music genre"]
    hub.calls.clear()
    monkeypatch.setattr(hub_probes.llm, "ask_json", lambda *a, **k: {"queries": "gtzan"})
    search_hub("Make it recognise music genres", "audio")
    assert searched(hub) == ["music genre", "music", "genre"]


def test_search_fails_only_when_every_query_fails(hub, monkeypatch):
    def down(url, **kwargs):
        raise hub_probes._HttpError("network", "URLError reaching huggingface.co")

    monkeypatch.setattr(hub_probes, "_http_get", down)
    with pytest.raises(ProbesUnavailable, match="failed for every query"):
        search_hub("Which country is this place in?", "coordinates")


def test_rate_limited_requests_are_retried(hub, monkeypatch):
    answers = [hub_probes._HttpError("http_429", "HTTP 429", 1.0), hub_probes._HttpError("http_429", "HTTP 429")]
    waits = []

    def flaky(url, **kwargs):
        if answers:
            raise answers.pop(0)
        return b"[]"

    monkeypatch.setattr(hub_probes, "_http_get", flaky)
    monkeypatch.setattr(hub_probes, "_sleep", waits.append)
    assert hub_probes._get_json(hub_probes.HUB_SEARCH, {"search": "x"}, what="search") == []
    assert waits == list(hub_probes.RETRY_WAITS)
    answers[:] = [hub_probes._HttpError("http_500", "HTTP 500")]
    with pytest.raises(ProbesUnavailable, match="HTTP 500"):
        hub_probes._get_json(hub_probes.HUB_SEARCH, {"search": "x"}, what="search")


# ----------------------------------------------------------------------------- hosts
@pytest.mark.parametrize("url, ok", [
    ("https://datasets-server.huggingface.co/assets/o/d/x.wav?sig=1", True),
    ("https://huggingface.co/datasets/o/d/resolve/main/a.mp4", True),
    ("https://cas-bridge.xethub.hf.co/x", True),
    ("http://huggingface.co/x", False),
    ("https://example.org/huggingface.co/x.wav", False),
    ("https://huggingface.co.example.org/x", False),
    ("https://user@huggingface.co/x", False),
    ("https://huggingface.co:8443/x", False),
    ("zip://a.avi::hf://datasets/o/d@1/a.zip", False),
])
def test_only_https_hugging_face_hosts(url, ok):
    assert hub_probes._allowed_url(url) is ok


def test_http_helper_refuses_other_hosts_and_redirects_before_any_io():
    with pytest.raises(hub_probes._HttpError) as caught:
        hub_probes._http_get("https://example.org/x.wav")
    assert caught.value.reason == "host_refused"
    request = hub_probes.urllib.request.Request("https://huggingface.co/x")
    with pytest.raises(hub_probes._HttpError, match="example.org"):
        hub_probes._Redirects().redirect_request(request, None, 302, "Found", {}, "https://example.org/x")


# ----------------------------------------------------------------------------- a media dataset
GENRES = ["rock", "jazz", "pop"]


def audio_src(hub_id, i):
    return f"{ASSETS}/{hub_id}/--/default/train/{i}/audio/audio.wav?Expires=1&Signature=s"


def unique_bytes(i, size):
    """size bytes that differ for every i (identical files are deduplicated) and spell no label word."""
    return f"{i:06d}".encode().ljust(size, b"R")


def audio_dataset(hub, hub_id="o/music-genres", total=90, big=(), foreign=()):
    def src(i):
        if i in foreign:
            return "https://example.org/stolen.wav"
        return audio_src(hub_id, i)

    def row(i):
        return {"audio": [{"src": src(i), "type": "audio/wav"}], "genre": i % 3}

    hub.datasets[hub_id] = Dataset([media("audio", "Audio"), class_label("genre", GENRES)], row, total)
    for i in range(total):
        hub.files[src(i)] = unique_bytes(i, 5_000 if i in big else 100)
    return hub_id


def test_media_classlabel_dataset_into_label_folders_with_caps_and_host_refusal(hub, tmp_path, monkeypatch):
    monkeypatch.setattr(hub_probes, "FILE_MAX_BYTES", 1_000)
    hub_id = audio_dataset(hub, big={1}, foreign={0})
    mapping = choose_mapping(cand(hub_id), "Make it recognise music genres", "audio")
    assert (mapping.input_columns, mapping.input_kind, mapping.media_type) == (["audio"], "media", "Audio")
    assert (mapping.label_column, mapping.label_kind, mapping.label_names) == ("genre", "classlabel", GENRES)
    assert (mapping.config, mapping.split, mapping.extension, mapping.origin) == ("default", "train", ".wav", "rule")
    report = fetch_probes(cand(hub_id), mapping, tmp_path / "p", n_per_label=5, min_per_label=3)
    assert isinstance(report, FetchReport)
    got = files_in(tmp_path / "p")
    assert sorted(got) == ["jazz", "pop", "rock"] and all(len(v) == 5 for v in got.values())
    assert all(name.endswith(".wav") and name[:-4].isdigit() for v in got.values() for name in v)
    assert all(int(name[:-4]) % 3 == GENRES.index(label) for label, v in got.items() for name in v)
    assert report.skipped["host_refused"] == 1 and report.skipped["too_large"] == 1
    assert "0.wav" not in got["rock"] and "1.wav" not in got["jazz"]
    assert not hub.fetched("https://example.org")  # refused before any request
    assert report.counts == {"rock": 5, "jazz": 5, "pop": 5} and report.bytes == 1_500
    assert report.licence == "mit" and report.source_url == "https://huggingface.co/datasets/o/music-genres"
    assert report.rows_seen == 90


def test_total_byte_cap_holds_with_parallel_downloads(hub, tmp_path, monkeypatch):
    hub_id = audio_dataset(hub)
    mapping = choose_mapping(cand(hub_id), "Make it recognise music genres", "audio")

    def slow(url, **kwargs):  # downloads overlap, so some finish after the cap is reached
        if url.startswith(ASSETS):
            time.sleep(0.05)
        return hub(url, **kwargs)

    monkeypatch.setattr(hub_probes, "_http_get", slow)
    report = fetch_probes(cand(hub_id), mapping, tmp_path / "p", n_per_label=10, min_per_label=1, max_bytes=700)
    assert sum(report.counts.values()) == 7 and report.bytes == 700
    assert sum(f.stat().st_size for f in (tmp_path / "p").rglob("*.wav")) == 700
    assert report.skipped["over_byte_cap"] >= 1 and any("byte cap" in note for note in report.notes)


def test_media_cells_that_are_not_hub_urls_are_refused_at_inspection(hub):
    hub.datasets["o/hmdb-actions"] = Dataset(
        [media("video", "Video"), class_label("label", ["run", "walk"])],
        lambda i: {"video": {"src": f"zip://run/{i}.avi::hf://datasets/o/hmdb-actions@1/a.zip"}, "label": i % 2}, 10)
    with pytest.raises(ProbesUnavailable, match="do not link to https Hugging Face files"):
        choose_mapping(cand("o/hmdb-actions"), "What action is happening in this video?", "video")


def test_media_column_must_have_the_modality_type(hub):
    hub.datasets["o/music-genres"] = Dataset([media("image", "Image"), class_label("genre", GENRES)],
                                             lambda i: {"image": {"src": f"{ASSETS}/x/{i}.jpg"}, "genre": 0}, 10)
    with pytest.raises(ProbesUnavailable, match="no Audio column"):
        choose_mapping(cand("o/music-genres"), "Make it recognise music genres", "audio")


# ----------------------------------------------------------------------------- text datasets
COUNTRIES = ["France", "Japan", "Brazil", "Kenya"]


def cities_dataset(hub, hub_id="o/world-cities", total=80):
    def row(i):
        return {"city": f"City {i}", "country": COUNTRIES[i % 4], "latitude": (10.5 + i) % 80,
                "longitude": -20.25 - i % 150, "population": 1000 + i}

    hub.datasets[hub_id] = Dataset([value("city"), value("country"), value("latitude", "float64"),
                                    value("longitude", "float64"), value("population", "int64")], row, total)
    return hub_id


def test_coordinates_dataset_written_from_the_template(hub, tmp_path):
    hub_id = cities_dataset(hub)
    mapping = choose_mapping(cand(hub_id, "cc-by-4.0"), "Which country is this place in?", "coordinates")
    assert (mapping.input_columns, mapping.input_kind, mapping.template) == (["latitude", "longitude"], "text",
                                                                             "{latitude}, {longitude}")
    assert (mapping.label_column, mapping.label_kind, mapping.extension) == ("country", "string", ".txt")
    report = fetch_probes(cand(hub_id, "cc-by-4.0"), mapping, tmp_path / "p", n_per_label=6, min_per_label=4)
    got = files_in(tmp_path / "p")
    assert sorted(got) == ["brazil", "france", "japan", "kenya"] and all(len(v) == 6 for v in got.values())
    assert (tmp_path / "p" / "japan" / "1.txt").read_text() == "11.5, -21.25\n"
    assert report.label_values["japan"] == "Japan" and report.licence == "cc-by-4.0"
    assert report.skipped == {}


def test_inspection_reads_first_rows_and_looks_mid_split_when_sorted_by_label(hub):
    hub_id = "o/world-cities"
    hub.datasets[hub_id] = Dataset(
        [value("latitude", "float64"), value("longitude", "float64"), value("country")],
        lambda i: {"latitude": 1.0, "longitude": 2.0, "country": COUNTRIES[i // 100]}, 400)
    mapping = choose_mapping(cand(hub_id), "Which country is this place in?", "coordinates")
    assert mapping.label_column == "country"
    viewer = [urllib.parse.urlsplit(u) for u in hub.fetched("https://datasets-server.huggingface.co")]
    assert [u.path for u in viewer] == ["/splits", "/first-rows", "/rows", "/rows"]
    assert urllib.parse.parse_qs(viewer[-1].query)["offset"] == ["200"]


def test_fetch_falls_back_to_first_rows_when_rows_are_refused(hub, tmp_path):
    hub_id = cities_dataset(hub, total=300)
    hub.datasets[hub_id].rows_error = "Parquet error: Scan size limit exceeded"
    mapping = choose_mapping(cand(hub_id), "Which country is this place in?", "coordinates")
    report = fetch_probes(cand(hub_id), mapping, tmp_path / "p", n_per_label=30, min_per_label=5)
    assert report.rows_seen == 100 and report.counts == {"france": 25, "japan": 25, "brazil": 25, "kenya": 25}
    assert any("only the split's first rows" in note and "Scan size limit" in note for note in report.notes)


def smiles_dataset(hub, hub_id, rows, *, column="descriptors", feature=None):
    hub.datasets[hub_id] = Dataset([value("smiles"), feature or string_list(column)],
                                   lambda i: {"smiles": "C" * (i + 1) + "O", column: rows[i % len(rows)]}, 60)
    return hub_id


def test_smiles_with_a_list_label_keeps_only_single_label_rows(hub, tmp_path):
    rows = [["fruity"], ["floral"], ["fruity", "floral"], ["woody"], ["fruity", "sweet"], ["floral", "green"]]
    hub_id = smiles_dataset(hub, "o/odor-molecules", rows)
    mapping = choose_mapping(cand(hub_id), "What does this molecule smell like?", "chemical")
    assert (mapping.input_columns, mapping.template, mapping.extension) == (["smiles"], "{smiles}", ".smi")
    assert (mapping.label_column, mapping.label_kind) == ("descriptors", "list")
    report = fetch_probes(cand(hub_id), mapping, tmp_path / "p", n_per_label=20, min_per_label=5, max_labels=3)
    got = files_in(tmp_path / "p")
    assert sorted(got) == ["floral", "fruity", "woody"]
    single = {"fruity": 0, "floral": 1, "woody": 3}
    assert all(int(name[:-4]) % 6 in (single[label], 4, 5) for label, v in got.items() for name in v)
    assert not any(int(name[:-4]) % 6 == 2 for v in got.values() for name in v)  # fruity+floral rows are out
    assert report.skipped["multi_label"] == 10
    assert any("exactly one of the chosen labels" in note for note in report.notes)
    assert (tmp_path / "p" / "woody" / "3.smi").read_text() == "CCCCO\n"


def test_a_delimited_descriptor_string_is_a_list_label(hub, tmp_path):
    hub_id = smiles_dataset(hub, "o/odor-gs", ["fruity;sweet", "fruity", "woody", "woody;green", "fruity"],
                            feature=value("descriptors"))
    mapping = choose_mapping(cand(hub_id), "What does this molecule smell like?", "chemical")
    assert (mapping.label_kind, mapping.label_separator) == ("list", ";")
    report = fetch_probes(cand(hub_id), mapping, tmp_path / "p", n_per_label=40, min_per_label=5)
    # sweet and green never reach 5 rows carrying exactly one chosen label, so they are not chosen, and a
    # "fruity;sweet" row then carries exactly one chosen label: fruity
    assert report.counts == {"fruity": 36, "woody": 24} and report.skipped["multi_label"] == 0


def test_multihot_columns_label_only_a_dataset_named_for_the_topic(hub, tmp_path):
    names = ["fruity", "green", "woody", "musky"]

    def row(i):
        return {"smiles": "C" * (i + 1)} | {n: int(j == i % 4 or (i % 7 == 0 and j == 0)) for j, n in enumerate(names)}

    features = [value("smiles")] + [value(n, "int64") for n in names]
    hub.datasets["o/molecular-odor-dataset"] = Dataset(features, row, 80)
    hub.datasets["o/moleculenet-tox"] = Dataset(features, row, 80)
    mapping = choose_mapping(cand("o/molecular-odor-dataset"), "What does this molecule smell like?", "chemical")
    assert (mapping.label_kind, mapping.label_column, mapping.label_names) == ("multihot", "", names)
    report = fetch_probes(cand("o/molecular-odor-dataset"), mapping, tmp_path / "p", min_per_label=5, max_labels=4)
    assert sorted(report.labels) == sorted(names) and report.skipped["multi_label"] > 0
    with pytest.raises(ProbesUnavailable, match="does not mention the goal's topic"):
        choose_mapping(cand("o/moleculenet-tox"), "What does this molecule smell like?", "chemical")


def test_a_generic_label_counts_only_in_a_dataset_named_for_the_topic(hub):
    features = [value("smiles"), class_label("label", ["penetrant", "blocked"])]
    for hub_id in ("o/odor-classes", "o/moleculenet-bbbp"):
        hub.datasets[hub_id] = Dataset(features, lambda i: {"smiles": "CCO", "label": i % 2}, 40)
    mapping = choose_mapping(cand("o/odor-classes"), "What does this molecule smell like?", "chemical")
    assert (mapping.label_column, mapping.label_kind) == ("label", "classlabel")
    with pytest.raises(ProbesUnavailable, match="does not mention the goal's topic"):
        choose_mapping(cand("o/moleculenet-bbbp"), "What does this molecule smell like?", "chemical")


# ----------------------------------------------------------------------------- the leak check
def sentence_dataset(hub, hub_id, leak_every):
    def row(i):
        country = COUNTRIES[i % 4]
        named = country.upper() if i % 2 else country
        text = f"A note about {named} food, number {i}" if i % leak_every == 0 else f"A note number {i}"
        return {"text": text, "country": country}

    hub.datasets[hub_id] = Dataset([value("text"), value("country")], row, 80)
    return hub_id


def test_leak_check_drops_rows_whose_text_names_the_label(hub, tmp_path):
    hub_id = sentence_dataset(hub, "o/country-notes", leak_every=5)
    mapping = choose_mapping(cand(hub_id), "Which country is this sentence about?", "text")
    assert (mapping.input_columns, mapping.label_column) == (["text"], "country")
    report = fetch_probes(cand(hub_id), mapping, tmp_path / "p", n_per_label=30, min_per_label=5)
    assert report.skipped["leak"] == 16
    texts = [f.read_text() for f in (tmp_path / "p").rglob("*.txt")]
    assert len(texts) == 64 and not any("food" in t for t in texts)


def test_leak_check_refuses_when_most_rows_leak(hub, tmp_path):
    hub_id = sentence_dataset(hub, "o/country-notes", leak_every=1)
    mapping = choose_mapping(cand(hub_id), "Which country is this sentence about?", "text")
    with pytest.raises(ProbesUnavailable, match="names the label in 80 of 80 rows"):
        fetch_probes(cand(hub_id), mapping, tmp_path / "p", min_per_label=5)
    assert not (tmp_path / "p").exists()


# ----------------------------------------------------------------------------- wanted labels
def test_wanted_labels_match_the_dataset_values_case_insensitively(hub, tmp_path):
    hub_id = "o/music-genres"
    names = ["Jazz", "hip hop", "rock"]
    hub.datasets[hub_id] = Dataset([media("audio", "Audio"), value("genre")],
                                   lambda i: {"audio": {"src": f"{ASSETS}/{hub_id}/{i}.mp3"},
                                              "genre": names[i % 3]}, 60)
    for i in range(60):
        hub.files[f"{ASSETS}/{hub_id}/{i}.mp3"] = unique_bytes(i, 10)
    mapping = choose_mapping(cand(hub_id), "Make it recognise music genres", "audio")
    report = fetch_probes(cand(hub_id), mapping, tmp_path / "p", wanted_labels=["JAZZ", "hip-hop"], min_per_label=5)
    assert sorted(report.labels) == ["hip_hop", "jazz"]
    assert report.label_values == {"jazz": "Jazz", "hip_hop": "hip hop"}
    assert sorted(files_in(tmp_path / "p")) == ["hip_hop", "jazz"]
    assert all(name.endswith(".mp3") for name in files_in(tmp_path / "p")["jazz"])
    assert not hub.fetched(f"{ASSETS}/{hub_id}/2.mp3")  # rock is never downloaded
    with pytest.raises(ProbesUnavailable, match="wanted labels not among"):
        fetch_probes(cand(hub_id), mapping, tmp_path / "q", wanted_labels=["jazz", "polka"], min_per_label=5)
    assert not (tmp_path / "q").exists()


# ----------------------------------------------------------------------------- the language model's mapping
def test_a_language_model_mapping_is_validated_against_the_features(hub, monkeypatch):
    hub_id = cities_dataset(hub)
    answers = {
        "label is an input": {"input_columns": ["latitude", "country"], "template": "{latitude} {country}",
                              "label_column": "country", "reason": "x"},
        "no such column": {"input_columns": ["lat"], "template": "{lat}", "label_column": "country", "reason": "x"},
        "label too varied": {"input_columns": ["latitude", "longitude"], "template": "{latitude}, {longitude}",
                             "label_column": "city", "reason": "x"},
        "template off": {"input_columns": ["latitude", "longitude"], "template": "{latitude}, {city}",
                         "label_column": "country", "reason": "x"},
    }
    for answer in answers.values():
        monkeypatch.setattr(hub_probes.llm, "ask_json", lambda *a, _answer=answer, **k: _answer)
        assert choose_mapping(cand(hub_id), "Which country is this place in?", "coordinates").origin == "rule"
    good = {"input_columns": ["longitude", "latitude"], "template": "lon {longitude} lat {latitude}",
            "label_column": "country", "reason": "coordinates give the country"}
    monkeypatch.setattr(hub_probes.llm, "ask_json", lambda *a, **k: good)
    mapping = choose_mapping(cand(hub_id), "Which country is this place in?", "coordinates")
    assert (mapping.origin, mapping.template, mapping.reason) == ("llm", "lon {longitude} lat {latitude}",
                                                                   "coordinates give the country")


# ----------------------------------------------------------------------------- the whole step
def test_no_candidate_works_names_every_reason(hub, tmp_path):
    hub.search = {"cities": [item("a/cities-down", 30), item("b/cities-nolabel", 20), item("c/cities-noinput", 10),
                             item("d/cities-missing", 5)]}
    hub.datasets["a/cities-down"] = Dataset([value("x")], lambda i: {}, 10, rows_error="Unexpected error.",
                                            first_rows=False)
    hub.datasets["b/cities-nolabel"] = Dataset([value("latitude", "float64"), value("longitude", "float64")],
                                               lambda i: {"latitude": 1.0 * i, "longitude": 2.0}, 10)
    hub.datasets["c/cities-noinput"] = Dataset([value("name"), value("country")],
                                               lambda i: {"name": f"n{i}", "country": "FR" if i % 2 else "JP"}, 10)
    events = []
    with pytest.raises(ProbesUnavailable) as caught:
        find_probes("Which country is this place in?", "coordinates", tmp_path, emit=events.append)
    message = str(caught.value)
    assert "a/cities-down" in message and "Unexpected error" in message
    assert "b/cities-nolabel" in message and "no label column" in message
    assert "c/cities-noinput" in message and "no input column" in message
    assert "d/cities-missing" in message and "HTTP 404" in message
    assert [e["hub_id"] for e in events if e["action"] == "candidate" and e["status"] == "fail"] == [
        "a/cities-down", "b/cities-nolabel", "c/cities-noinput", "d/cities-missing"]
    assert list(tmp_path.iterdir()) == []


def test_find_probes_returns_the_folder_and_info_and_emits_json_events(hub, tmp_path):
    hub.search = {"cities": [item("a/cities-down", 30), item("o/world-cities", 20, cardData={"license": "cc-by-4.0"})]}
    hub.datasets["a/cities-down"] = Dataset([value("x")], lambda i: {}, 10, rows_error="Unexpected error.",
                                            first_rows=False)
    cities_dataset(hub)
    events = []
    folder, info = find_probes("Which country is this place in?", "coordinates", tmp_path, emit=events.append,
                               n_per_label=5, min_per_label=4)
    assert folder == tmp_path / "o__world-cities" and sorted(p.name for p in folder.iterdir()) == [
        "brazil", "france", "japan", "kenya"]
    assert (info["hub_id"], info["licence"], info["url"]) == ("o/world-cities", "cc-by-4.0",
                                                              "https://huggingface.co/datasets/o/world-cities")
    assert info["counts"] == {"france": 5, "japan": 5, "brazil": 5, "kenya": 5}
    assert info["mapping"]["template"] == "{latitude}, {longitude}" and info["query_origin"] == "rule"
    assert info["tried"] == [{"hub_id": "a/cities-down", "reason": info["tried"][0]["reason"]}]
    assert "Unexpected error" in info["tried"][0]["reason"]
    json.dumps(info)
    for event in events:
        assert json.loads(json.dumps(event)) == event
        assert event["action"] in ("search", "candidate", "inspect", "fetch")
        assert event["status"] in ("start", "ok", "fail")
    steps = [(e["action"], e["status"]) for e in events]
    assert steps[:2] == [("search", "start"), ("search", "ok")]
    assert ("inspect", "fail") in steps and steps[-2:] == [("fetch", "ok"), ("candidate", "ok")]


def test_max_candidates_limits_fetch_attempts_not_inspections(hub, tmp_path):
    hub.search = {"cities": [item(f"o/cities-{n}", 100 - n) for n in range(5)]}
    for n in range(5):
        hub.datasets[f"o/cities-{n}"] = Dataset(
            [value("latitude", "float64"), value("longitude", "float64"), value("country")],
            lambda i: {"latitude": 1.0, "longitude": 2.0, "country": "FR"}, 10)  # one label only: fetch fails
    with pytest.raises(ProbesUnavailable):
        find_probes("Which country is this place in?", "coordinates", tmp_path, max_candidates=2, min_per_label=2)
    assert len(hub.fetched("https://datasets-server.huggingface.co/splits")) <= 5
    with pytest.raises(ProbesUnavailable) as caught:
        find_probes("Which country is this place in?", "coordinates", tmp_path, max_candidates=2, max_inspect=1)
    assert "o/cities-0" in str(caught.value) and "o/cities-1" not in str(caught.value)


# ----------------------------------------------------------------------------- dest is owned and cleaned
def test_placeholder_values_are_never_chosen_as_labels(hub, tmp_path):
    hub.datasets["o/world-cities"] = Dataset(
        [value("latitude", "float64"), value("longitude", "float64"), value("country")],
        lambda i: {"latitude": 1.0 + i / 10, "longitude": 2.0,
                   "country": ["Unknown", "Unknown", "France", "Japan"][i % 4]}, 80)
    mapping = choose_mapping(cand("o/world-cities"), "Which country is this place in?", "coordinates")
    report = fetch_probes(cand("o/world-cities"), mapping, tmp_path / "p", min_per_label=5)
    assert sorted(report.labels) == ["france", "japan"]
    assert any("'Unknown': it names no category" in note for note in report.notes)


def test_dest_is_replaced_and_removed_on_failure(hub, tmp_path):
    hub_id = cities_dataset(hub)
    dest = tmp_path / "p"
    (dest / "stale").mkdir(parents=True)
    (dest / "stale" / "old.txt").write_text("old")
    mapping = choose_mapping(cand(hub_id), "Which country is this place in?", "coordinates")
    fetch_probes(cand(hub_id), mapping, dest, n_per_label=3, min_per_label=3)
    assert not (dest / "stale").exists()
    with pytest.raises(ProbesUnavailable, match="fewer than 2 label values"):
        fetch_probes(cand(hub_id), mapping, dest, n_per_label=3, min_per_label=50)
    assert not dest.exists()


def test_truncated_and_empty_cells_are_skipped(hub, tmp_path):
    hub_id = "o/odor-molecules"
    rows = {i: {"smiles": None if i == 0 else "C" * (i + 1) + "O",
                "descriptors": [] if i == 1 else [["fruity", "woody"][i % 2]]} for i in range(40)}
    hub.datasets[hub_id] = Dataset([value("smiles"), string_list("descriptors")], rows.__getitem__, 40,
                                   truncated={2: ["smiles"]})
    mapping = choose_mapping(cand(hub_id), "What does this molecule smell like?", "chemical")
    report = fetch_probes(cand(hub_id), mapping, tmp_path / "p", min_per_label=5)
    assert report.skipped == {"no_input": 1, "no_label": 1, "truncated": 1, "multi_label": 0}
    assert sum(report.counts.values()) == 37
    assert isinstance(mapping, Mapping) and mapping.to_dict()["label_kind"] == "list"


# ----------------------------------------------------------------------------- media metadata that names the label
def filler(i, n=256):
    """Deterministic bytes that look like compressed data and differ for every i."""
    return b"".join(hashlib.sha256(f"{i}-{k}".encode()).digest() for k in range(n // 32))


def video_dataset(hub, hub_id, labels, leaky, total=60):
    def src(i):
        return f"{ASSETS}/{hub_id}/--/default/train/{i}/video/video.mp4?Expires=1&Signature=s"

    def row(i):
        return {"video": {"src": src(i)}, "label": labels[i % len(labels)]}

    hub.datasets[hub_id] = Dataset([media("video", "Video"), value("label")], row, total)
    for i in range(total):
        label = labels[i % len(labels)]
        tag = f"JnameH:\\HumanActionDB\\MotionClips\\{label}\\clip{i}.avi" if leaky(i) else f"clip number {i}"
        hub.files[src(i)] = b"\x00\x00\x00\x18ftypisom\x00\x00" + tag.encode() + b"\x00" + filler(i)
    return hub_id


def test_media_files_whose_metadata_names_their_label_are_dropped(hub, tmp_path):
    hub_id = video_dataset(hub, "o/hmdb-actions", ["drink", "walk"], leaky=lambda i: i % 6 == 0)
    mapping = choose_mapping(cand(hub_id), "What action is happening in this video?", "video")
    report = fetch_probes(cand(hub_id), mapping, tmp_path / "p", n_per_label=30, min_per_label=5)
    assert report.skipped["leak"] == 10 and report.counts == {"walk": 30, "drink": 20}
    assert not any(b"HumanActionDB" in f.read_bytes() for f in (tmp_path / "p").rglob("*.mp4"))
    assert any("metadata names their own label" in note for note in report.notes)
    assert any("ffmpeg is not installed" in note for note in report.notes)  # the fixture hides ffmpeg


def test_a_dataset_whose_media_mostly_name_their_label_is_refused_early(hub, tmp_path):
    hub_id = video_dataset(hub, "o/hmdb-actions", ["drink", "walk"], leaky=lambda i: True)
    mapping = choose_mapping(cand(hub_id), "What action is happening in this video?", "video")
    with pytest.raises(ProbesUnavailable, match="name their own label in their metadata"):
        fetch_probes(cand(hub_id), mapping, tmp_path / "p", n_per_label=30, min_per_label=5)
    assert not (tmp_path / "p").exists()
    assert len(hub.fetched(ASSETS)) <= hub_probes.LEAK_SAMPLE + hub_probes.DOWNLOAD_WORKERS


@pytest.mark.parametrize("data, label, leaks", [
    (b"\x00JnameH:\\HumanActionDB\\drink\\clip.avi\x00", "drink", True),
    (b"\x00T\x00C\x00O\x00N" + "Rock music".encode("utf-16-le") + b"\x00\x00", "rock", True),
    (b"\x00\x00" + "Rock music".encode("utf-16-be") + b"\x00", "rock", True),
    (b"\x01\\sword_exercise\\x.avi\x01", "sword_exercise", True),
    (b"\x01genre: Hip-Hop\x01", "hip hop", True),
    (b"\x01genre: pop music\x01", "pop", True),
    (b"\x01pop\x01", "pop", False),  # a three-letter word only counts inside a run of 6 or more
    (b"\x01VideoHandler\x01", "video", False),  # part of a longer word
    (b"\x01scene 1.5\x01", "1", False),  # digits only are never looked for
    (b"\x00" * (hub_probes.SCAN_BYTES + 9) + b"\x01\\drink\\\x01" + b"\x00" * (hub_probes.SCAN_BYTES + 9), "drink",
     False),  # the middle of a big file is media data, not metadata
])
def test_the_media_leak_scan(data, label, leaks):
    assert hub_probes._names_label(data, label) is leaks


def glb(doc: dict, binary: bytes) -> bytes:
    body = json.dumps(doc).encode()
    body += b" " * (-len(body) % 4)
    binary += b"\x00" * (-len(binary) % 4)
    chunks = struct.pack("<II", len(body), 0x4E4F534A) + body + struct.pack("<II", len(binary), 0x004E4942) + binary
    return b"glTF" + struct.pack("<II", 2, 12 + len(chunks)) + chunks


def glb_parts(data: bytes) -> tuple[dict, bytes]:
    assert data[:4] == b"glTF" and struct.unpack_from("<II", data, 4) == (2, len(data))
    size, kind = struct.unpack_from("<II", data, 12)
    assert kind == 0x4E4F534A and size % 4 == 0
    return json.loads(data[20:20 + size]), data[20 + size:]


def mesh_doc(name: str) -> dict:
    return {"asset": {"version": "2.0", "generator": "test", "extras": {"source": name}},
            "scenes": [{"name": name, "nodes": [0]}], "nodes": [{"name": name, "mesh": 0}],
            "meshes": [{"name": name, "primitives": [{"attributes": {"POSITION": 0}, "extras": {"part": name}}]}],
            "materials": [{"name": "scan", "extensions": {"KHR_test": {"name": "kept"}}}],
            "buffers": [{"byteLength": 36}]}


def test_glb_names_and_extras_are_stripped_and_the_geometry_kept():
    binary = struct.pack("<9f", *range(9))
    data = glb(mesh_doc("ACE_Coffee_Mug_Kristen_16_oz_cup"), binary)
    out, how = hub_probes._strip_metadata(data, ".glb", "Mesh", None, 30)
    doc, rest = glb_parts(out)
    assert how == "stripped" and b"Coffee" not in out
    assert "name" not in json.dumps({k: v for k, v in doc.items() if k != "materials"})
    assert "extras" not in json.dumps(doc)
    assert doc["materials"] == [{"extensions": {"KHR_test": {"name": "kept"}}}]  # extension data is left alone
    assert doc["meshes"][0]["primitives"][0]["attributes"] == {"POSITION": 0}
    assert rest[8:8 + len(binary)] == binary


def test_glb_probes_never_carry_the_object_names(hub, tmp_path):
    hub_id = "o/scanned-objects"
    products = {"kitchen": "ACE_Coffee_Mug", "electronics": "Asus_Wireless_Router"}

    def row(i):
        return {"mesh": {"src": f"{ASSETS}/{hub_id}/{i}.glb"}, "category": list(products)[i % 2]}

    hub.datasets[hub_id] = Dataset([media("mesh", "Mesh"), value("category")], row, 40)
    for i in range(40):
        label = list(products)[i % 2]
        name = f"{label}_bowl_{i}" if i % 4 == 0 else f"{products[label]}_{i}"  # some even name their label
        hub.files[f"{ASSETS}/{hub_id}/{i}.glb"] = glb(mesh_doc(name), filler(i, 64))
    mapping = choose_mapping(cand(hub_id), "Tell me what object this 3D model is", "3d")
    report = fetch_probes(cand(hub_id), mapping, tmp_path / "p", n_per_label=20, min_per_label=5)
    assert report.counts == {"kitchen": 20, "electronics": 20} and report.skipped.get("leak", 0) == 0
    for f in (tmp_path / "p").rglob("*.glb"):
        doc, _ = glb_parts(f.read_bytes())
        assert b"Coffee" not in f.read_bytes() and b"Asus" not in f.read_bytes() and b"bowl" not in f.read_bytes()
        assert "name" not in json.dumps(doc["meshes"]) + json.dumps(doc["scenes"]) + json.dumps(doc["nodes"])
    assert any("stripped the metadata" in note for note in report.notes)


def png_chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))


def test_png_text_chunks_are_stripped():
    ihdr = png_chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 0, 0, 0, 0))
    idat, iend = png_chunk(b"IDAT", zlib.compress(b"\x00\x00")), png_chunk(b"IEND", b"")
    data = (hub_probes._PNG_SIGNATURE + ihdr + png_chunk(b"tEXt", b"Title\x00golden retriever") + idat
            + png_chunk(b"iTXt", b"Description\x00\x00\x00\x00\x00a retriever") + iend)
    out, how = hub_probes._strip_metadata(data, ".png", "Image", None, 30)
    assert how == "stripped" and out == hub_probes._PNG_SIGNATURE + ihdr + idat + iend


def jpeg_segment(marker: int, payload: bytes) -> bytes:
    return bytes((0xFF, marker)) + struct.pack(">H", len(payload) + 2) + payload


def test_jpeg_metadata_segments_are_stripped_and_the_image_kept():
    app0 = jpeg_segment(0xE0, b"JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00")
    icc, dqt = jpeg_segment(0xE2, b"ICC_PROFILE\x00\x01\x01sRGB"), jpeg_segment(0xDB, b"\x00" + bytes(64))
    scan = b"\xff\xda" + struct.pack(">H", 8) + b"\x01\x01\x00\x00\x3f\x00" + b"pixels\xff\x00more" + b"\xff\xd9"
    data = (b"\xff\xd8" + app0 + jpeg_segment(0xE1, b"Exif\x00\x00golden retriever") + icc
            + jpeg_segment(0xED, b"Photoshop 3.0\x00retriever") + jpeg_segment(0xFE, b"a golden retriever") + dqt
            + scan)
    out, how = hub_probes._strip_metadata(data, ".jpg", "Image", None, 30)
    assert how == "stripped" and out == b"\xff\xd8" + app0 + icc + dqt + scan
    assert hub_probes._strip_metadata(b"\xff\xd8\xff\xe0\x00", ".jpg", "Image", None, 30)[1] == "unstripped"


FFMPEG = shutil.which("ffmpeg")


@pytest.mark.skipif(not FFMPEG, reason="ffmpeg is not installed")
def test_ffmpeg_remux_drops_video_and_audio_tags_and_keeps_the_streams(tmp_path):
    video, sound = tmp_path / "v.mp4", tmp_path / "a.wav"
    subprocess.run([FFMPEG, "-nostdin", "-v", "error", "-f", "lavfi", "-i", "color=c=red:s=16x16:d=0.2", "-c:v",
                    "mpeg4", "-metadata", "title=ACE Coffee Mug", "-metadata:s:v:0",
                    "handler_name=H:\\clips\\drink\\x.avi", str(video)], check=True)
    subprocess.run([FFMPEG, "-nostdin", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=0.2",
                    "-metadata", "genre=Rock", "-c:a", "pcm_s16le", str(sound)], check=True)
    assert b"Coffee" in video.read_bytes() and b"drink" in video.read_bytes() and b"Rock" in sound.read_bytes()
    out, how = hub_probes._strip_metadata(video.read_bytes(), ".mp4", "Video", FFMPEG, 30)
    assert how == "stripped" and b"Coffee" not in out and b"drink" not in out and b"Lavf" not in out
    probe = tmp_path / "out.mp4"
    probe.write_bytes(out)
    assert subprocess.run([FFMPEG, "-nostdin", "-v", "error", "-i", str(probe), "-f", "null", "-"]).returncode == 0
    out, how = hub_probes._strip_metadata(sound.read_bytes(), ".wav", "Audio", FFMPEG, 30)
    original = sound.read_bytes()
    assert how == "stripped" and b"Rock" not in out
    assert out[out.index(b"data") + 8:] == original[original.index(b"data") + 8:]  # the samples are untouched
    assert hub_probes._strip_metadata(original, ".wav", "Audio", None, 30) == (original, "unstripped")


# ----------------------------------------------------------------------------- the same input under two labels
def test_the_same_input_under_two_labels_is_dropped_and_repeats_are_merged(hub, tmp_path):
    def row(i):  # long format: two rows per molecule, one per odor
        m = i // 2
        if m < 10:  # fruity and green
            odor = ["fruity", "green"][i % 2]
        elif m < 30:  # one odor, twice
            odor = ["fruity", "green"][m % 2]
        else:  # fruity and sweet: sweet never stands alone, so it cannot be a label
            odor = ["fruity", "sweet"][i % 2]
        return {"smiles": "C" * (m + 1) + "O", "odor": odor}

    hub.datasets["o/odor-pairs"] = Dataset([value("smiles"), value("odor")], row, 72)
    mapping = choose_mapping(cand("o/odor-pairs"), "What does this molecule smell like?", "chemical")
    assert mapping.label_kind == "string"
    report = fetch_probes(cand("o/odor-pairs"), mapping, tmp_path / "p", n_per_label=30, min_per_label=5)
    fruity = {f.read_text() for f in (tmp_path / "p" / "fruity").iterdir()}
    green = {f.read_text() for f in (tmp_path / "p" / "green").iterdir()}
    assert not fruity & green and report.counts == {"fruity": 16, "green": 10}
    assert report.skipped["duplicate_input"] == 36 and report.skipped["conflicting_labels"] == 10
    assert any("different labels" in note for note in report.notes)


def test_identical_media_files_are_kept_once_and_never_under_two_labels(hub, tmp_path):
    hub_id = audio_dataset(hub)
    hub.files[audio_src(hub_id, 3)] = hub.files[audio_src(hub_id, 0)]  # rock twice
    hub.files[audio_src(hub_id, 2)] = hub.files[audio_src(hub_id, 1)]  # jazz and pop
    mapping = choose_mapping(cand(hub_id), "Make it recognise music genres", "audio")
    report = fetch_probes(cand(hub_id), mapping, tmp_path / "p", n_per_label=40, min_per_label=5)
    assert report.skipped["duplicate_input"] == 1 and report.skipped["conflicting_labels"] == 2
    files = [f.read_bytes() for f in (tmp_path / "p").rglob("*.wav")]
    assert len(files) == len(set(files)) == 87
    assert not (tmp_path / "p" / "jazz" / "1.wav").exists() and not (tmp_path / "p" / "pop" / "2.wav").exists()
    assert report.counts == {"rock": 29, "jazz": 29, "pop": 29}


# ----------------------------------------------------------------------------- multi-label strings
def test_a_comma_joined_descriptor_string_is_a_list_label(hub, tmp_path):
    hub_id = smiles_dataset(hub, "o/odor-molecules", ["fruity", "green", "fruity, green"], column="odor",
                            feature=value("odor"))
    mapping = choose_mapping(cand(hub_id), "What does this molecule smell like?", "chemical")
    assert (mapping.label_kind, mapping.label_separator) == ("list", ",")
    report = fetch_probes(cand(hub_id), mapping, tmp_path / "p", n_per_label=40, min_per_label=5)
    assert report.counts == {"fruity": 20, "green": 20} and report.skipped["multi_label"] == 20
    assert any("exactly one of the chosen labels" in note for note in report.notes)


def test_a_list_literal_label_is_a_list(hub, tmp_path):
    hub_id = smiles_dataset(hub, "o/odor-molecules", ["['fruity']", '["green"]', "['fruity', 'green']"],
                            column="odor", feature=value("odor"))
    mapping = choose_mapping(cand(hub_id), "What does this molecule smell like?", "chemical")
    assert (mapping.label_kind, mapping.label_separator) == ("list", hub_probes.LIST_LITERAL)
    report = fetch_probes(cand(hub_id), mapping, tmp_path / "p", n_per_label=40, min_per_label=5)
    assert report.counts == {"fruity": 20, "green": 20} and report.skipped["multi_label"] == 20


def test_a_comma_inside_one_name_keeps_it_one_label(hub):
    names = ["France", "Japan", "Korea, Republic of"]
    hub.datasets["o/world-cities"] = Dataset(
        [value("latitude", "float64"), value("longitude", "float64"), value("country")],
        lambda i: {"latitude": i / 10, "longitude": 2.0, "country": names[i % 3]}, 60)
    mapping = choose_mapping(cand("o/world-cities"), "Which country is this place in?", "coordinates")
    assert (mapping.label_column, mapping.label_kind, mapping.label_separator) == ("country", "string", None)


def test_a_value_combining_other_values_is_never_a_label(hub, tmp_path):
    hub_id = smiles_dataset(hub, "o/odor-molecules", ["fruity", "fruity", "green", "green", "fruity, green"],
                            column="odor", feature=value("odor"))
    mapping = choose_mapping(cand(hub_id), "What does this molecule smell like?", "chemical")
    assert mapping.label_kind == "string"  # too few joined cells to read the column as a list
    report = fetch_probes(cand(hub_id), mapping, tmp_path / "p", n_per_label=40, min_per_label=5)
    assert sorted(report.labels) == ["fruity", "green"]
    assert any("'fruity, green': it combines other label values" in note for note in report.notes)


# ----------------------------------------------------------------------------- an input that gives the label away
CODES = {"FR": "France", "JP": "Japan", "BR": "Brazil", "KE": "Kenya"}


def coded_cities(hub, hub_id="o/world-cities"):
    def row(i):
        code = list(CODES)[i % 4]
        return {"text": f"Postcard number {i}", "city": f"{CODES[code]} town {i}", "country_name": CODES[code],
                "country_code": code, "latitude": (10.5 + i) % 80, "longitude": -20.25 - i % 150}

    hub.datasets[hub_id] = Dataset([value("text"), value("city"), value("country_name"), value("country_code"),
                                    value("latitude", "float64"), value("longitude", "float64")], row, 80)
    return hub_id


def test_a_language_model_cannot_add_an_input_column_that_gives_the_label_away(hub, tmp_path, monkeypatch):
    hub_id = coded_cities(hub)
    answer = {"input_columns": ["latitude", "longitude", "country_name"], "label_column": "country_code",
              "template": "{latitude}, {longitude} ({country_name})", "reason": "x"}
    monkeypatch.setattr(hub_probes.llm, "ask_json", lambda *a, **k: answer)
    mapping = choose_mapping(cand(hub_id), "Which country is this place in?", "coordinates")
    assert (mapping.origin, mapping.input_columns, mapping.label_column) == ("rule", ["latitude", "longitude"],
                                                                             "country_code")
    report = fetch_probes(cand(hub_id), mapping, tmp_path / "p", n_per_label=5, min_per_label=4)
    assert not any(name in f.read_text() for f in (tmp_path / "p").rglob("*.txt") for name in CODES.values())
    assert report.counts == {"fr": 5, "jp": 5, "br": 5, "ke": 5}
    # a city (every value distinct, so no column-determines-label check can see it) is not a coordinate slot
    answer = {"input_columns": ["latitude", "longitude", "city"], "label_column": "country_code",
              "template": "{latitude}, {longitude} ({city})", "reason": "x"}
    mapping = choose_mapping(cand(hub_id), "Which country is this place in?", "coordinates")
    assert (mapping.origin, mapping.input_columns) == ("rule", ["latitude", "longitude"])
    # with no typed slots (plain text), the column-determines-label check still refuses it
    answer = {"input_columns": ["text", "country_name"], "label_column": "country_code",
              "template": "{text} ({country_name})", "reason": "x"}
    mapping = choose_mapping(cand(hub_id), "Which country is this postcard from?", "text")
    assert (mapping.origin, mapping.input_columns, mapping.label_column) == ("rule", ["text"], "country_code")


def test_a_language_model_mapping_must_use_input_columns_of_the_right_kind(hub, monkeypatch):
    hub.datasets["o/music-genres"] = Dataset(
        [media("cover", "Image"), media("audio", "Audio"), value("genre")],
        lambda i: {"cover": {"src": f"{ASSETS}/c/{i}.jpg"}, "audio": {"src": f"{ASSETS}/a/{i}.wav"},
                   "genre": GENRES[i % 3]}, 30)
    answer = {"input_columns": ["cover"], "template": "", "label_column": "genre", "reason": "x"}
    monkeypatch.setattr(hub_probes.llm, "ask_json", lambda *a, **k: answer)
    mapping = choose_mapping(cand("o/music-genres"), "Make it recognise music genres", "audio")
    assert (mapping.origin, mapping.input_columns, mapping.media_type) == ("rule", ["audio"], "Audio")
    hub.datasets["o/country-notes"] = Dataset(
        [value("text"), string_list("keywords"), value("country")],
        lambda i: {"text": f"A note number {i}", "keywords": [f"k{i % 7}"], "country": COUNTRIES[i % 4]}, 40)
    answer = {"input_columns": ["keywords"], "template": "{keywords}", "label_column": "country", "reason": "x"}
    mapping = choose_mapping(cand("o/country-notes"), "Which country is this sentence about?", "text")
    assert (mapping.origin, mapping.input_columns) == ("rule", ["text"])


# ----------------------------------------------------------------------------- which column is the label
EMOTIONS = ["angry", "happy", "sad"]
TRUTH = ("emotion", "emotion_label", "gold_emotion", "emotion_category")  # the other emotion columns are predictions


def emotion_dataset(hub, hub_id, columns, classlabels=()):
    def cell(column, i):
        if column == "speaker":
            return f"speaker {i % 4}"
        index = i % 3 if column in TRUTH else (i + 1) % 3
        return index if column in classlabels else EMOTIONS[index]

    features = [class_label(c, EMOTIONS) if c in classlabels else value(c) for c in columns]
    hub.datasets[hub_id] = Dataset([media("audio", "Audio")] + features,
                                   lambda i: {"audio": {"src": f"{ASSETS}/{hub_id}/{i}.wav"}}
                                   | {c: cell(c, i) for c in columns}, 30)
    return hub_id


SPEAKS = "What emotion is this speaker expressing?"


@pytest.mark.parametrize("columns, classlabels, goal, label", [
    (["qwen_emotion", "emotion", "kimi_emotion"], (), SPEAKS, "emotion"),  # nEMO's model columns, truth not first
    (["qwen_emotion", "emotion"], ("qwen_emotion",), SPEAKS, "emotion"),  # a variant of a bare column loses points
    (["predicted_emotion", "gold_emotion"], (), SPEAKS, "gold_emotion"),  # prediction words lose points
    (["emotion_category_v2", "emotion_category"], (), SPEAKS, "emotion_category"),  # a tie goes to fewer words
    (["speaker", "emotion"], (), SPEAKS, "emotion"),  # the goal's focus beats its later words
    (["emotion", "speaker"], (), "Which speaker is expressing this emotion?", "speaker"),
    (["speaker", "emotion"], (), "Listen to this speaker and say which emotion they express", "emotion"),
])
def test_the_rule_label_prefers_the_asked_ground_truth_column(hub, columns, classlabels, goal, label):
    hub_id = emotion_dataset(hub, "o/nemo-speakers", columns, classlabels)
    assert choose_mapping(cand(hub_id), goal, "audio").label_column == label


def test_the_asked_word_outweighs_a_name_column_penalty(hub):
    regions = ["Europe", "Asia", "Americas", "Africa"]
    hub.datasets["o/world-cities"] = Dataset(
        [value("latitude", "float64"), value("longitude", "float64"), value("region"), value("country_name")],
        lambda i: {"latitude": i / 10, "longitude": 2.0, "region": regions[i % 4], "country_name": COUNTRIES[i % 4]},
        40)
    assert choose_mapping(cand("o/world-cities"), "Which country is this place in?",
                          "coordinates").label_column == "country_name"


CIFAR = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def test_a_benchmark_class_label_is_trusted_when_its_names_fit_the_goal_or_wanted_labels(hub):
    hub.datasets["o/cifar10"] = Dataset([media("img", "Image"), class_label("label", CIFAR)],
                                        lambda i: {"img": {"src": f"{ASSETS}/o/cifar10/{i}.png"}, "label": i % 10}, 50)
    mapping = choose_mapping(cand("o/cifar10"), "Is this a cat or a dog?", "image")
    assert (mapping.label_column, mapping.label_kind) == ("label", "classlabel")
    mapping = choose_mapping(cand("o/cifar10"), "Which animal is in this photo?", "image", wanted_labels=["Cat", "dog"])
    assert mapping.label_column == "label"
    with pytest.raises(ProbesUnavailable, match="present but not trusted: 'label' .a ClassLabel: airplane"):
        choose_mapping(cand("o/cifar10"), "Which animal is in this photo?", "image")
    hub.datasets["o/handwritten-digits"] = Dataset(
        [media("image", "Image"), value("label", "int32")],
        lambda i: {"image": {"src": f"{ASSETS}/d/{i}.png"}, "label": i % 10}, 50)
    with pytest.raises(ProbesUnavailable, match="'label' hold numbers, not label names"):
        choose_mapping(cand("o/handwritten-digits"), "What handwritten digit is this?", "image")


# ----------------------------------------------------------------------------- word matching
@pytest.mark.parametrize("label, text, found", [
    ("1", "1.5, 20.25", False),
    ("2", "-2.75, 31.0", False),
    ("20", "20.25, 1.0", False),
    ("5", "rated 5 stars", True),
    ("5", "rated 5.", True),
    ("France", "near FRANCE today", True),
    ("hip hop", "a Hip-Hop beat", True),
    ("hip hop", "hiphop", True),
    ("rock", "rocket", False),
    ("sword_exercise", "\\sword_exercise\\", True),
])
def test_the_label_word_pattern(label, text, found):
    assert bool(hub_probes._word_pattern(label).search(text)) is found


def test_numeric_labels_do_not_leak_into_decimal_coordinates(hub, tmp_path):
    hub.datasets["o/zone-cities"] = Dataset(
        [value("latitude", "float64"), value("longitude", "float64"), value("region")],
        lambda i: {"latitude": 1.5 + i / 100, "longitude": -2.75 - i / 100, "region": str(1 + i % 3)}, 60)
    mapping = choose_mapping(cand("o/zone-cities"), "Which region is this place in?", "coordinates")
    report = fetch_probes(cand("o/zone-cities"), mapping, tmp_path / "p", n_per_label=20, min_per_label=5)
    assert report.counts == {"1": 20, "2": 20, "3": 20} and "leak" not in report.skipped


# ----------------------------------------------------------------------------- paths stay inside dest
@pytest.mark.parametrize("label, folder", [("../../x", "x"), ("/etc/passwd", "etc_passwd"), ("..", "label_7"),
                                           ("a/b", "a_b"), ("C:\\Windows\\System32", "c_windows_system32")])
def test_safe_label_is_one_plain_folder_name(label, folder):
    assert safe_label(label, 7) == folder


def test_label_values_shaped_like_paths_stay_inside_dest(hub, tmp_path):
    values = ["../../evil", "/etc/passwd", "...", "ok"]  # a ClassLabel, so the values reach safe_label whole
    hub.datasets["o/world-cities"] = Dataset(
        [value("latitude", "float64"), value("longitude", "float64"), class_label("country", values)],
        lambda i: {"latitude": i / 10, "longitude": 2.0, "country": i % 4}, 80)
    mapping = choose_mapping(cand("o/world-cities"), "Which country is this place in?", "coordinates")
    report = fetch_probes(cand("o/world-cities"), mapping, tmp_path / "p", n_per_label=5, min_per_label=5)
    assert sorted(report.labels) == ["etc_passwd", "evil", "ok"]
    assert report.label_values["evil"] == "../../evil"
    assert any("'...': its folder name would be empty" in note for note in report.notes)
    assert [p.name for p in tmp_path.iterdir()] == ["p"]
    assert all(len(f.relative_to(tmp_path / "p").parts) == 2 for f in (tmp_path / "p").rglob("*") if f.is_file())


def test_search_drops_ids_that_are_not_plain_owner_and_name(hub):
    hub.search = {"cities": [item(i, 10) for i in ("..", "a/..", "../x", "a/b/c", "./x", "a/.", "o/ok-cities")]}
    assert [c.hub_id for c in search_hub("Which country is this place in?", "coordinates")] == ["o/ok-cities"]


def test_licence_other_carries_the_cards_name_and_link():
    card = {"license": "other", "license_name": "hmdb-research-only", "license_link": "https://example.org/terms"}
    found = hub_probes._candidate({"id": "o/d", "cardData": card, "tags": ["license:other"]})
    assert found.licence == "other (hmdb-research-only, https://example.org/terms)"
    assert hub_probes._candidate({"id": "o/d", "cardData": {"license": "other"}}).licence == "other"


# ----------------------------------------------------------------------------- the real HTTP helper
class FakeResponse:
    def __init__(self, body, length=None, clock=None):
        self.body, self.pos, self.clock = body, 0, clock
        self.headers = {"Content-Length": str(len(body)) if length is None else length}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read1(self, n):
        if self.clock is not None:
            self.clock[0] += 1.0  # each read takes a second
        chunk = self.body[self.pos:self.pos + min(n, 1000)]
        self.pos += len(chunk)
        return chunk


def fake_opener(monkeypatch, response):
    seen = {}

    class Opener:
        def open(self, request, timeout):
            seen["request"], seen["timeout"] = request, timeout
            return response

    def build_opener(*handlers):
        seen["handlers"] = handlers
        return Opener()

    monkeypatch.setattr(hub_probes.urllib.request, "build_opener", build_opener)
    return seen


def test_the_real_http_helper_caps_bytes_and_time_and_guards_redirects(monkeypatch):
    url = "https://huggingface.co/api/datasets?search=x"
    seen = fake_opener(monkeypatch, FakeResponse(b"x" * 2500))
    assert hub_probes._http_get(url, max_bytes=3000) == b"x" * 2500
    assert hub_probes._Redirects in seen["handlers"] and seen["request"].get_header("User-agent") == "sense-harness"
    for response in (FakeResponse(b"x" * 10, length="5000"), FakeResponse(b"x" * 5000, length="")):
        fake_opener(monkeypatch, response)  # an announced size over the cap, then a body streamed past it
        with pytest.raises(hub_probes._HttpError) as caught:
            hub_probes._http_get(url, max_bytes=3000)
        assert caught.value.reason == "too_large"
    clock = [0.0]
    monkeypatch.setattr(hub_probes, "_now", lambda: clock[0])
    fake_opener(monkeypatch, FakeResponse(b"x" * 9000, clock=clock))
    with pytest.raises(hub_probes._HttpError) as caught:
        hub_probes._http_get(url, timeout=2.5)
    assert caught.value.reason == "timeout" and clock[0] <= 3.0


# ----------------------------------------------------------------------------- cleanup on any failure
def test_dest_is_removed_when_writing_fails(hub, tmp_path, monkeypatch):
    hub_id = cities_dataset(hub)
    mapping = choose_mapping(cand(hub_id), "Which country is this place in?", "coordinates")
    real, written = hub_probes._write, []

    def full_disk(target, data):
        written.append(target)
        if len(written) > 3:
            raise OSError(28, "No space left on device")
        real(target, data)

    monkeypatch.setattr(hub_probes, "_write", full_disk)
    with pytest.raises(ProbesUnavailable, match="could not write probes"):
        fetch_probes(cand(hub_id), mapping, tmp_path / "p", n_per_label=5, min_per_label=4)
    assert len(written) == 4 and not (tmp_path / "p").exists()


def test_dest_is_removed_when_a_download_is_interrupted(hub, tmp_path, monkeypatch):
    hub_id = audio_dataset(hub)
    mapping = choose_mapping(cand(hub_id), "Make it recognise music genres", "audio")
    downloads = []

    def interrupted(url, **kwargs):
        if url.startswith(ASSETS):
            downloads.append(url)
            if len(downloads) > 6:
                raise KeyboardInterrupt
        return hub(url, **kwargs)

    monkeypatch.setattr(hub_probes, "_http_get", interrupted)
    with pytest.raises(KeyboardInterrupt):
        fetch_probes(cand(hub_id), mapping, tmp_path / "p", n_per_label=20, min_per_label=5)
    time.sleep(0.05)  # let any worker still running finish (workers never write into dest)
    assert not (tmp_path / "p").exists()


# ----------------------------------------------------------------------------- time budget
def test_retry_waits_are_capped_and_never_run_past_the_deadline(monkeypatch):
    clock, sleeps, timeouts = [0.0], [], []
    monkeypatch.setattr(hub_probes, "_now", lambda: clock[0])

    def sleep(seconds):
        sleeps.append(seconds)
        clock[0] += seconds

    def busy(url, *, timeout, max_bytes):
        timeouts.append(timeout)
        clock[0] += 0.5
        raise hub_probes._HttpError("http_503", "HTTP 503 from datasets-server.huggingface.co", 999.0)

    monkeypatch.setattr(hub_probes, "_sleep", sleep)
    monkeypatch.setattr(hub_probes, "_http_get", busy)
    with pytest.raises(ProbesUnavailable, match="HTTP 503"):
        hub_probes._get_json(hub_probes.VIEWER + "/rows", {}, what="rows")
    assert sleeps == [hub_probes.MAX_RETRY_WAIT] * 2  # Retry-After 999 waits 30 s at most
    clock[0], sleeps[:], timeouts[:] = 0.0, [], []
    monkeypatch.setattr(hub_probes, "RETRY_WAITS", (5.0, 5.0))
    monkeypatch.setattr(hub_probes, "MAX_RETRY_WAIT", 5.0)
    with pytest.raises(ProbesUnavailable, match="HTTP 503"):
        hub_probes._get_json(hub_probes.VIEWER + "/rows", {}, what="rows", deadline=12.0)
    assert sleeps == [5.0] and timeouts == [12.0, 6.5] and clock[0] <= 12.0


def test_a_fetch_stays_within_its_budget_when_the_viewer_is_busy(monkeypatch, tmp_path):
    clock, sleeps, requests = [0.0], [], []
    monkeypatch.setattr(hub_probes, "_now", lambda: clock[0])
    monkeypatch.setattr(hub_probes, "_sleep", lambda s: (sleeps.append(s), clock.__setitem__(0, clock[0] + s)))

    def busy(url, *, timeout, max_bytes):
        requests.append(timeout)
        clock[0] += min(timeout, 1.0)
        raise hub_probes._HttpError("http_503", "HTTP 503 from datasets-server.huggingface.co")

    monkeypatch.setattr(hub_probes, "_http_get", busy)
    mapping = Mapping("default", "train", ["latitude", "longitude"], "text", "{latitude}, {longitude}", ".txt",
                      "country", "string")
    with pytest.raises(ProbesUnavailable):
        fetch_probes(cand("o/world-cities"), mapping, tmp_path / "p", budget_s=3)
    assert sleeps == [] and clock[0] <= 3.0 and all(t <= 3.0 for t in requests)


def test_find_probes_keeps_one_deadline_over_search_inspection_and_fetch(hub, tmp_path, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(hub_probes, "_now", lambda: clock[0])

    def slow(url, *, timeout=20.0, max_bytes=50_000_000):  # every request takes 10 s
        if timeout < 10:
            clock[0] += timeout
            raise hub_probes._HttpError("timeout", f"took over {timeout:.0f} s")
        clock[0] += 10.0
        return hub(url, timeout=timeout, max_bytes=max_bytes)

    monkeypatch.setattr(hub_probes, "_http_get", slow)
    hub.search = {"cities": [item(f"o/cities-{n}", 100 - n) for n in range(8)]}
    for n in range(8):  # coordinates but no label: every inspection fails
        hub.datasets[f"o/cities-{n}"] = Dataset([value("latitude", "float64"), value("longitude", "float64")],
                                                lambda i: {"latitude": i / 10, "longitude": 2.0}, 10)
    with pytest.raises(ProbesUnavailable, match="the 100 s time budget ran out") as caught:
        find_probes("Which country is this place in?", "coordinates", tmp_path, total_budget_s=100)
    assert clock[0] <= 100 and "o/cities-0" in str(caught.value) and "o/cities-7" not in str(caught.value)
    clock[0] = 0.0  # six search queries take 60 s: a 35 s budget cuts the search short too
    with pytest.raises(ProbesUnavailable, match="none was tried; the 35 s time budget ran out"):
        find_probes("Which country is this place in?", "coordinates", tmp_path, total_budget_s=35)
    assert clock[0] <= 35


def test_media_page_reading_leaves_half_the_budget_for_downloads(hub, tmp_path, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(hub_probes, "_now", lambda: clock[0])
    hub_id = audio_dataset(hub, total=2400)

    def slow(url, *, timeout=20.0, max_bytes=50_000_000):
        clock[0] += 10.0 if "/rows" in url else 1.0 if url.startswith(ASSETS) else 0.0  # 10 s a page, 1 s a file
        return hub(url, timeout=timeout, max_bytes=max_bytes)

    monkeypatch.setattr(hub_probes, "_http_get", slow)
    mapping = choose_mapping(cand(hub_id), "Make it recognise music genres", "audio")
    report = fetch_probes(cand(hub_id), mapping, tmp_path / "p", n_per_label=12, min_per_label=5, budget_s=120)
    assert len(hub.fetched(hub_probes.VIEWER + "/rows")) <= 7 and report.rows_seen <= 700
    assert any("stopped reading rows at the time budget" in note for note in report.notes)
    assert report.counts == {"rock": 12, "jazz": 12, "pop": 12}


def test_a_passing_failure_of_the_first_page_still_reads_the_spread_pages(hub, tmp_path):
    def row(i):
        return {"latitude": (i * 0.037) % 80, "longitude": -(i * 0.011) % 150, "country": COUNTRIES[i % 4]}

    hub.datasets["o/world-cities"] = Dataset(
        [value("latitude", "float64"), value("longitude", "float64"), value("country")], row, 2400,
        flaky={0: [502, 504, 502], 2300: [502]}, sized=True)
    mapping = choose_mapping(cand("o/world-cities"), "Which country is this place in?", "coordinates")
    report = fetch_probes(cand("o/world-cities"), mapping, tmp_path / "p", n_per_label=10, min_per_label=5)
    assert report.rows_seen == 1200 and "page_failed" not in report.skipped  # offset 2300's 502 was retried
    assert any("/size" in note for note in report.notes)
    offsets = [urllib.parse.parse_qs(urllib.parse.urlsplit(u).query)["offset"][0]
               for u in hub.fetched(hub_probes.VIEWER + "/rows")]
    assert offsets.count("2300") == 2 and offsets.count("0") == 3


def test_after_a_page_fails_when_retried_later_pages_get_one_try(hub, tmp_path):
    def row(i):
        return {"latitude": (i * 0.037) % 80, "longitude": -(i * 0.011) % 150, "country": COUNTRIES[i % 4]}

    hub.datasets["o/world-cities"] = Dataset(  # pages at 0, 209, 418, 627, ... 2300
        [value("latitude", "float64"), value("longitude", "float64"), value("country")], row, 2400,
        flaky={209: [503, 503, 503], 418: [503], 627: [503]})
    mapping = choose_mapping(cand("o/world-cities"), "Which country is this place in?", "coordinates")
    report = fetch_probes(cand("o/world-cities"), mapping, tmp_path / "p", n_per_label=10, min_per_label=5)
    offsets = [urllib.parse.parse_qs(urllib.parse.urlsplit(u).query)["offset"][0]
               for u in hub.fetched(hub_probes.VIEWER + "/rows")]
    assert (offsets.count("209"), offsets.count("418"), offsets.count("627")) == (3, 1, 1)
    assert report.skipped["page_failed"] == 3 and report.rows_seen == 900
    assert any("later pages get one try each" in note for note in report.notes)
