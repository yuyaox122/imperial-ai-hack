"""Facts the agent trace needs, in fake mode: the specialist behind each bridge and its timing, the fit time of
each candidate block, and which bridge kinds are available for a model and why.

Run: SENSE_FAKE=1 .venv/bin/python -m pytest -q tests/test_trace_facts.py
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pytest

from sense import bridges, config, datasets, models, specialists
from sense.contracts import ModelInfo
from sense.specialists import FakeSpecialist

KINDS = ["native", "text", "tool", "image", "vector"]


@pytest.fixture(autouse=True)
def fake_home(monkeypatch, tmp_path):
    monkeypatch.setenv("SENSE_FAKE", "1")
    monkeypatch.setenv("SENSE_HOME", str(tmp_path / "home"))
    assert config.fake()
    return tmp_path / "home"


@pytest.fixture
def fresh_specialists():
    """Drop the cached specialists before and after the test, so timing counters start at zero."""
    specialists._specialist.cache_clear()
    yield
    specialists._specialist.cache_clear()


@pytest.fixture
def model():
    return models.load_model("fake")


@pytest.fixture
def gtzan():
    return datasets.load_labelled_set("gtzan")


@pytest.fixture
def vector_bridge(model, gtzan):
    return bridges.fit_vector_bridge(model, gtzan, gtzan.fit_idx)


def _samples(tmp_path, n=3):
    paths = []
    for k in range(n):
        path = tmp_path / "clips" / f"clip{k}.wav"
        FakeSpecialist.write_sample(k, path, seed=k)
        paths.append(path)
    return paths


def _round_trips(value) -> bool:
    return json.loads(json.dumps(value)) == value


# ----------------------------------------------------------------------------- specialists
def test_specialist_describe_reports_the_fake_specialist(fresh_specialists):
    facts = specialists.describe("audio")
    assert facts == {
        "name": "fake-clap",
        "label": "Fake CLAP (tones, fake mode)",
        "modality": "audio",
        "dim": specialists.FAKE_DIM,
        "embed_seconds_per_file": None,
        "loaded": True,
    }
    assert _round_trips(facts)


def test_specialist_describe_rejects_an_unknown_modality_like_get_specialist():
    with pytest.raises(ValueError) as from_get:
        specialists.get_specialist("video")
    with pytest.raises(ValueError) as from_describe:
        specialists.describe("video")
    assert str(from_describe.value) == str(from_get.value)


def test_specialist_describe_never_loads_clap(monkeypatch, fresh_specialists):
    """Outside fake mode describe reports CLAP from its settings alone: torch cannot even be imported here."""
    monkeypatch.setenv("SENSE_FAKE", "0")
    monkeypatch.setenv("SENSE_CLAP", "/nowhere/clap-for-the-test")
    monkeypatch.setitem(sys.modules, "torch", None)  # any `import torch` now raises ImportError
    monkeypatch.setitem(sys.modules, "transformers", None)
    facts = specialists.describe("audio")
    assert facts["name"] == "/nowhere/clap-for-the-test"
    assert facts["label"] == "CLAP"
    assert facts["dim"] == 512
    assert facts["loaded"] is False
    assert facts["embed_seconds_per_file"] is None
    assert specialists.get_specialist("audio")._model is None
    assert _round_trips(facts)


def test_embed_seconds_per_file_is_none_until_files_are_embedded(tmp_path):
    spec = FakeSpecialist()
    assert spec.embed_seconds_per_file is None
    spec.embed_texts(["rock music"])  # text embeddings are not timed
    spec.embed_files([])  # nothing embedded: still no figure
    assert spec.embed_seconds_per_file is None
    spec.embed_files(_samples(tmp_path, 2))
    first = spec.embed_seconds_per_file
    assert isinstance(first, float) and first > 0
    spec.embed_files(_samples(tmp_path, 3))
    assert spec._embed_count == 5
    assert spec.embed_seconds_per_file == pytest.approx(spec._embed_seconds / 5)
    assert FakeSpecialist().embed_seconds_per_file is None  # counters belong to the instance


def test_describe_shows_the_timing_of_the_cached_specialist(tmp_path, fresh_specialists):
    assert specialists.describe("audio")["embed_seconds_per_file"] is None
    specialists.get_specialist("audio").embed_files(_samples(tmp_path, 2))
    per_file = specialists.describe("audio")["embed_seconds_per_file"]
    assert isinstance(per_file, float) and per_file > 0


# ----------------------------------------------------------------------------- bridges
def test_vector_bridge_records_fit_time_per_block(vector_bridge):
    assert vector_bridge.dev_accuracy
    assert set(vector_bridge.fit_seconds_by_layer) == set(vector_bridge.dev_accuracy)
    assert all(isinstance(k, int) for k in vector_bridge.fit_seconds_by_layer)
    assert all(v >= 0 for v in vector_bridge.fit_seconds_by_layer.values())
    assert vector_bridge.fit_seconds >= 0


def test_vector_bridge_at_a_given_layer_times_that_layer(model, gtzan):
    bridge = bridges.fit_vector_bridge(model, gtzan, gtzan.fit_idx, layer=2)
    assert bridge.fit_seconds_by_layer == {2: bridge.fit_seconds}
    assert bridge.dev_accuracy == {}


def test_fit_seconds_by_layer_survives_save_and_load(vector_bridge, tmp_path):
    vector_bridge.save(tmp_path / "b")
    meta = json.loads((tmp_path / "b" / "bridge.json").read_text())
    assert set(meta["fit_seconds_by_layer"]) == {str(k) for k in vector_bridge.fit_seconds_by_layer}
    again = bridges.load_bridge(tmp_path / "b")
    assert again.fit_seconds_by_layer == vector_bridge.fit_seconds_by_layer
    assert again.fit_seconds == vector_bridge.fit_seconds
    assert bridges.describe(again) == bridges.describe(vector_bridge)


def test_bridge_saved_before_the_change_still_loads(vector_bridge, tmp_path):
    vector_bridge.save(tmp_path / "old")
    path = tmp_path / "old" / "bridge.json"
    meta = json.loads(path.read_text())
    del meta["fit_seconds_by_layer"]
    path.write_text(json.dumps(meta, indent=2))
    old = bridges.load_bridge(tmp_path / "old")
    assert old.fit_seconds_by_layer == {}
    assert bridges.describe(old)["fit_seconds_by_layer"] == {}
    emb = datasets.load_labelled_set("gtzan").emb[:1]
    np.testing.assert_allclose(old.vectors(emb), vector_bridge.vectors(emb))


def test_describe_a_vector_bridge(vector_bridge, gtzan):
    facts = bridges.describe(vector_bridge)
    assert _round_trips(facts)
    assert facts["name"] == "vector:gtzan" and facts["kind"] == "vector"
    assert facts["modality"] == "audio" and facts["model"] == "fake"
    assert facts["specialist"] == specialists.describe("audio")
    assert facts["layer"] == vector_bridge.layer
    assert facts["dev_accuracy"] == {str(k): v for k, v in vector_bridge.dev_accuracy.items()}
    assert set(facts["fit_seconds_by_layer"]) == set(facts["dev_accuracy"])
    assert facts["fit_seconds"] == vector_bridge.fit_seconds
    assert facts["n_fit"] == len(gtzan.fit_idx)
    assert facts["source"] == "gtzan"
    assert facts["n_phrases"] is None


@pytest.mark.parametrize("closed", [True, False], ids=["labels", "captions"])
def test_describe_a_text_bridge(model, gtzan, closed):
    bridge = bridges.make_text_bridge(model, gtzan if closed else None)
    facts = bridges.describe(bridge)
    assert _round_trips(facts)
    assert facts["kind"] == "text" and facts["model"] == "fake"
    assert facts["source"] == ("gtzan" if closed else "captions")
    assert facts["n_phrases"] == len(bridge.phrases) > 0
    assert facts["specialist"]["name"] == "fake-clap"
    for absent in ("fit_seconds", "fit_seconds_by_layer", "layer", "dev_accuracy", "n_fit"):
        assert facts[absent] is None


# ----------------------------------------------------------------------------- availability
def _by_kind(entries):
    assert [e["kind"] for e in entries] == KINDS
    assert all(set(e) == {"kind", "available", "why"} and isinstance(e["available"], bool) for e in entries)
    assert all(isinstance(e["why"], str) and e["why"] for e in entries)
    assert _round_trips(entries)
    return {e["kind"]: e for e in entries}


def test_availability_for_the_fake_model(model):
    entries = _by_kind(bridges.availability(model.info, "audio"))
    assert {k: e["available"] for k, e in entries.items()} == {
        "native": False, "text": True, "tool": False, "image": False, "vector": True,
    }
    assert entries["native"]["why"] == f"{model.info.label} does not read audio natively"
    assert "Fake CLAP" in entries["text"]["why"]
    assert entries["tool"]["why"] == "the tool bridge is not built yet"
    assert "spectrogram" in entries["image"]["why"]
    assert "open-weight" in entries["vector"]["why"]


def test_availability_for_a_closed_api_model():
    info = ModelInfo(name="api", label="API model", native=["text", "image"], open_weights=False)
    entries = _by_kind(bridges.availability(info, "audio"))
    assert {k: e["available"] for k, e in entries.items()} == {
        "native": False, "text": True, "tool": False, "image": False, "vector": False,
    }
    assert "not open-weight: an API exposes no hidden states to write into" in entries["vector"]["why"]


def test_availability_when_the_model_reads_the_modality_natively():
    info = ModelInfo(name="api-audio", label="Audio API model", native=["text", "audio"], open_weights=False)
    entries = _by_kind(bridges.availability(info, "audio"))
    assert entries["native"]["available"] is True
    assert entries["native"]["why"] == "Audio API model reads audio natively"


def test_availability_without_a_specialist(model):
    entries = _by_kind(bridges.availability(model.info, "video"))
    assert not any(e["available"] for e in entries.values())
    assert "no video specialist" in entries["text"]["why"]
    assert "no video specialist" in entries["vector"]["why"]
