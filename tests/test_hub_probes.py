"""Finding labelled probe files on the Hugging Face Hub (agent.hub_probes), offline.

No test touches the network: hub_probes._http_get is replaced by FakeHub, which answers the Hub search, the dataset
viewer's /splits and /rows, and media file URLs from small in-memory tables shaped like the real answers (the
trimmed real responses in tests/fixtures/hub/ pin those shapes). ask_json answers None in fake mode, so the rule
fallbacks run unless a test replaces it.
Run: SENSE_FAKE=1 .venv/bin/python -m pytest -q tests/test_hub_probes.py
"""

from __future__ import annotations

import json
import time
import urllib.parse
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
    search_hub,
)

FIXTURES = Path(__file__).parent / "fixtures" / "hub"
ASSETS = "https://datasets-server.huggingface.co/assets"


class Dataset:
    def __init__(self, features, row, total, *, splits=(("default", "train"),), truncated=None, rows_error=None,
                 first_rows=True):
        self.features, self.row, self.total = features, row, total
        self.splits, self.truncated, self.rows_error = splits, truncated or {}, rows_error
        self.first_rows = first_rows  # whether /first-rows answers (it does even when /rows fails, as on the Hub)

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
        if parts.netloc == "datasets-server.huggingface.co" and parts.path in ("/splits", "/rows", "/first-rows"):
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
            if ds.rows_error:
                raise hub_probes._HttpError("http_500", f"HTTP 500 from the viewer: {ds.rows_error}")
            offset, length = int(query["offset"]), int(query["length"])
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


def audio_dataset(hub, hub_id="o/music-genres", total=90, big=(), foreign=()):
    def src(i):
        if i in foreign:
            return "https://example.org/stolen.wav"
        return f"{ASSETS}/{hub_id}/--/default/train/{i}/audio/audio.wav?Expires=1&Signature=s"

    def row(i):
        return {"audio": [{"src": src(i), "type": "audio/wav"}], "genre": i % 3}

    hub.datasets[hub_id] = Dataset([media("audio", "Audio"), class_label("genre", GENRES)], row, total)
    for i in range(total):
        hub.files[src(i)] = b"R" * (5_000 if i in big else 100)
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
        text = f"A note about {country.lower()} food" if i % leak_every == 0 else f"A note number {i}"
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
        hub.files[f"{ASSETS}/{hub_id}/{i}.mp3"] = b"x" * 10
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
        lambda i: {"latitude": 1.0, "longitude": 2.0, "country": ["Unknown", "Unknown", "France", "Japan"][i % 4]}, 80)
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
    rows = {i: {"smiles": None if i == 0 else "CCO", "descriptors": [] if i == 1 else [["fruity", "woody"][i % 2]]}
            for i in range(40)}
    hub.datasets[hub_id] = Dataset([value("smiles"), string_list("descriptors")], rows.__getitem__, 40,
                                   truncated={2: ["smiles"]})
    mapping = choose_mapping(cand(hub_id), "What does this molecule smell like?", "chemical")
    report = fetch_probes(cand(hub_id), mapping, tmp_path / "p", min_per_label=5)
    assert report.skipped == {"no_input": 1, "no_label": 1, "truncated": 1, "multi_label": 0}
    assert sum(report.counts.values()) == 37
    assert isinstance(mapping, Mapping) and mapping.to_dict()["label_kind"] == "list"
