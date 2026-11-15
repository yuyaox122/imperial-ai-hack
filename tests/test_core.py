"""Core sense harness in fake mode: bridges, verification, persistence and the registry.

Run: SENSE_FAKE=1 .venv/bin/python -m pytest -q tests/test_core.py
"""

from __future__ import annotations

import os
import shutil
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from sense import bridges, config, datasets, models, registry
from sense.contracts import Bridge, FitRequest, LabelledSet, ModelInfo, OpenModel
from sense.verify import verify


@pytest.fixture(autouse=True)
def fake_home(monkeypatch, tmp_path):
    monkeypatch.setenv("SENSE_FAKE", "1")
    monkeypatch.setenv("SENSE_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("SENSE_SETS_DIR", raising=False)  # folder sets live under SENSE_HOME, so the test sees none
    assert config.fake()
    return tmp_path / "home"


@pytest.fixture
def model():
    return models.load_model("fake")


@pytest.fixture
def gtzan():
    return datasets.load_labelled_set("gtzan")


@pytest.fixture
def vector_bridge(model, gtzan):
    return bridges.fit_vector_bridge(model, gtzan, gtzan.fit_idx)


def test_fake_mode_lists_only_the_fake_models(model):
    listed = models.available_models()
    assert [m.name for m in listed] == ["fake", "fake-api"]
    fake, fake_api = listed
    assert fake.open_weights and not fake.tools
    assert not fake_api.open_weights and fake_api.tools and fake_api.available
    assert models.load_model("fake") is model
    assert isinstance(model, OpenModel)
    api = models.load_model("fake-api")
    assert api.info.tools is True and api.info.name == "fake-api" and api is models.load_model("fake-api")
    with pytest.raises(ValueError, match="fake mode"):
        models.load_model("qwen3.5-0.8b")


def test_datasets_have_disjoint_splits():
    sets = {s["name"]: s for s in datasets.available_sets()}
    assert set(sets) == {"gtzan", "esc50"}
    for name in sets:
        ds = datasets.load_labelled_set(name)
        assert len(ds.y) == len(ds.emb) == sets[name]["n"]
        assert ds.label_text_emb.shape == (len(ds.labels), ds.emb.shape[1])
        splits = [set(ds.fit_idx), set(ds.dev_idx), set(ds.test_idx)]
        assert all(splits) and not (splits[0] & splits[1] or splits[0] & splits[2] or splits[1] & splits[2])
    assert "hip hop" in sets["gtzan"]["labels"]


def test_vector_bridge_beats_its_controls(model, gtzan, vector_bridge):
    assert isinstance(vector_bridge, Bridge)
    assert vector_bridge.name == "vector:gtzan" and vector_bridge.kind == "vector"
    assert vector_bridge.layer in range(5)
    card = verify(model, vector_bridge, gtzan, FitRequest(model="fake", dataset="gtzan"))
    assert card.verdict == "candidate" and card.chance == pytest.approx(0.2)
    assert card.n == len(gtzan.test_idx)
    assert card.accuracy >= 0.9
    assert card.accuracy >= card.controls["swapped"] + 0.5
    assert card.accuracy >= card.controls["no_input"] + 0.5
    assert card.controls["no_input"] < 0.4 and card.controls["swapped"] < 0.4
    assert "lookup" in card.controls and "unseen_classes" not in card.controls


def test_text_bridges(model, gtzan):
    labels = bridges.make_text_bridge(model, gtzan)
    assert labels.name == "text:gtzan" and labels.kind == "text"
    card = verify(model, labels, gtzan, FitRequest(model="fake", dataset="gtzan"))
    assert card.accuracy >= card.controls["swapped"] + 0.5

    esc = datasets.load_labelled_set("esc50")
    captions = bridges.make_text_bridge(model)
    assert captions.name == "text:captions"
    inp = captions.from_embedding(esc.emb[esc.test_idx[0]], source="clip.wav")
    assert inp.kind == "text" and inp.source == "clip.wav" and "nearest caption" in inp.detail
    card = verify(model, captions, esc, FitRequest(model="fake", dataset="esc50"))
    assert card.accuracy >= card.controls["no_input"] + 0.5


def test_text_bridge_is_a_control_for_the_vector_bridge(model, gtzan, vector_bridge):
    text = bridges.make_text_bridge(model, gtzan)
    card = verify(model, vector_bridge, gtzan, FitRequest(model="fake", dataset="gtzan"), text_bridge=text)
    assert card.controls["text_bridge"] >= 0.9


def test_open_scope_adds_unseen_classes_and_they_fail(model, gtzan, vector_bridge):
    card = verify(model, vector_bridge, gtzan, FitRequest(model="fake", dataset="gtzan", scope="open"))
    assert card.scope == "open"
    assert card.controls["unseen_classes"] < card.chance + 0.25  # the vector bridge transfers labels, not sounds
    text = bridges.make_text_bridge(model, gtzan)
    assert "unseen_classes" not in verify(model, text, gtzan, FitRequest(model="fake", dataset="gtzan", scope="open")).controls


def test_lookup_answers_the_same_five_option_questions(model):
    """Each test clip lies nearest to fitted examples of the next label (a decoy), then of its own. Searching all
    labels the lookup is always wrong; on the bridge's five-option questions it is right whenever the decoy is
    not among the options."""
    labels = ["blues", "classical", "country", "disco", "jazz", "metal", "pop", "reggae", "rock", "soul"]
    n_labels, per_label = len(labels), 6
    eye = np.eye(n_labels)
    noise = np.random.default_rng(1).normal(scale=0.01, size=(n_labels * per_label, n_labels))
    y = np.repeat(np.arange(n_labels), per_label)
    fit_rows = eye[y] + noise  # each fitted example sits on its own label's axis
    test_rows = 0.6 * eye[y] + 0.8 * eye[(y + 1) % n_labels]  # closer to the decoy's axis than to its own
    ds = LabelledSet(
        name="toy", title="Toy", modality="audio", question="Which genre?", labels=labels,
        y=np.concatenate([y, y]), emb=np.vstack([fit_rows, test_rows]), label_text_emb=eye,
        fit_idx=np.arange(len(y)), dev_idx=np.arange(0), test_idx=np.arange(len(y), 2 * len(y)),
    )
    card = verify(model, bridges.make_text_bridge(model, ds), ds, FitRequest(model="fake", dataset="toy"))

    options, _ = bridges.five_options(ds.y[ds.test_idx], n_labels, np.random.default_rng(0))  # verify's draw
    decoy_absent = [(t + 1) % n_labels not in opts for t, opts in zip(ds.y[ds.test_idx], options)]
    assert card.controls["lookup"] == pytest.approx(np.mean(decoy_absent), abs=1e-4)
    assert 0.3 < card.controls["lookup"] < 0.8  # the old all-label search scored 0.0 here


def test_verify_is_deterministic(model, gtzan, vector_bridge):
    request = FitRequest(model="fake", dataset="gtzan")
    a, b = verify(model, vector_bridge, gtzan, request), verify(model, vector_bridge, gtzan, request)
    assert (a.accuracy, a.controls) == (b.accuracy, b.controls)


def test_save_load_round_trip(model, gtzan, vector_bridge, tmp_path):
    x = gtzan.emb[gtzan.test_idx[:3]]
    vector_bridge.save(tmp_path / "vector")
    again = bridges.load_bridge(tmp_path / "vector")
    assert (again.name, again.kind, again.model_name, again.layer) == ("vector:gtzan", "vector", "fake", vector_bridge.layer)
    for row in x:
        np.testing.assert_allclose(again.from_embedding(row).vector, vector_bridge.from_embedding(row).vector, rtol=1e-6)

    for text in (bridges.make_text_bridge(model, gtzan), bridges.make_text_bridge(model)):
        text.save(tmp_path / text.name.replace(":", "_"))
        again = bridges.load_bridge(tmp_path / text.name.replace(":", "_"))
        assert again.name == text.name
        assert [again.from_embedding(r).text for r in x] == [text.from_embedding(r).text for r in x]


def test_registry_install_replace_uninstall(model, gtzan, vector_bridge, fake_home):
    assert registry.installed("fake") == [] and registry.get_bridges("fake") == []
    card = verify(model, vector_bridge, gtzan, FitRequest(model="fake", dataset="gtzan"))
    registry.install(vector_bridge, card)
    assert (fake_home / "fake" / "audio" / "card.json").is_file()

    [entry] = registry.installed("fake")
    assert entry["bridge"] == "vector:gtzan" and entry["kind"] == "vector" and entry["modality"] == "audio"
    assert entry["dataset"] == "gtzan" and entry["card"]["verdict"] == "installed"
    assert entry["card"]["accuracy"] == card.accuracy
    [loaded] = registry.get_bridges("fake")
    assert loaded.name == "vector:gtzan"

    text = bridges.make_text_bridge(model, gtzan)
    registry.install(text, verify(model, text, gtzan, FitRequest(model="fake", dataset="gtzan")))
    assert [e["bridge"] for e in registry.installed("fake")] == ["text:gtzan"]  # same model and modality: replaced

    assert registry.uninstall("fake", "audio") is True
    assert registry.uninstall("fake", "audio") is False
    assert registry.installed("fake") == []


def _read_while(action, model_name: str = "fake") -> list:
    """Run action() while another thread keeps calling registry.get_bridges, as chats would, with every directory
    deletion slowed down; return what each read found: bridge names, or the error text."""
    real_rmtree = shutil.rmtree
    deleting = threading.Event()

    def slow_rmtree(path, *args, **kwargs):
        deleting.set()
        time.sleep(0.2)
        return real_rmtree(path, *args, **kwargs)

    seen, stop = [], threading.Event()

    def chat():
        while not stop.is_set():
            try:
                seen.append([b.name for b in registry.get_bridges(model_name)])
            except RuntimeError as err:
                seen.append(str(err))

    reader = threading.Thread(target=chat)
    reader.start()
    try:
        shutil.rmtree = slow_rmtree
        action()
    finally:
        shutil.rmtree = real_rmtree
        stop.set()
        reader.join()
    assert deleting.is_set() and seen
    return seen


def test_registry_replace_and_uninstall_never_show_a_missing_or_half_deleted_sense(model, gtzan, fake_home):
    text = bridges.make_text_bridge(model, gtzan)
    card = verify(model, text, gtzan, FitRequest(model="fake", dataset="gtzan"), max_items=20)
    registry.install(text, card)
    seen = _read_while(lambda: registry.install(text, card))  # a second run installs the same sense again
    assert all(found == ["text:gtzan"] for found in seen), {str(found) for found in seen}
    seen = _read_while(lambda: registry.uninstall("fake", "audio"))
    assert all(found in (["text:gtzan"], []) for found in seen), {str(found) for found in seen}
    assert registry.installed("fake") == [] and list((fake_home / "fake").iterdir()) == []


def test_registry_install_keeps_the_old_sense_when_the_new_one_cannot_be_moved_in(
    model, gtzan, vector_bridge, fake_home, monkeypatch
):
    registry.install(vector_bridge, verify(model, vector_bridge, gtzan, FitRequest(model="fake", dataset="gtzan")))
    text = bridges.make_text_bridge(model, gtzan)
    card = verify(model, text, gtzan, FitRequest(model="fake", dataset="gtzan"), max_items=20)
    real_rename = Path.rename
    failing = [".installing-"]

    def failing_rename(self, target):
        if any(part in self.name for part in failing):
            raise OSError("disk full")
        return real_rename(self, target)

    monkeypatch.setattr(Path, "rename", failing_rename)
    with pytest.raises(OSError, match="disk full"):
        registry.install(text, card)
    assert [b.name for b in registry.get_bridges("fake")] == ["vector:gtzan"]
    assert [p.name for p in (fake_home / "fake").iterdir()] == ["audio"]  # no staging or removed copy left behind

    failing.append(".removing-")  # the old sense cannot be put back either: it is kept aside, never deleted
    with pytest.raises(OSError, match="disk full"):
        registry.install(text, card)
    monkeypatch.setattr(Path, "rename", real_rename)
    [kept] = (fake_home / "fake").iterdir()
    assert kept.name.startswith(".audio.removing-") and bridges.load_bridge(kept).name == "vector:gtzan"


def test_install_rejects_a_card_for_another_bridge(model, gtzan, vector_bridge):
    text = bridges.make_text_bridge(model, gtzan)
    card = verify(model, text, gtzan, FitRequest(model="fake", dataset="gtzan"))
    with pytest.raises(ValueError, match="report card"):
        registry.install(vector_bridge, card)


def test_chat_with_a_sample_file_through_an_installed_bridge(model, gtzan, vector_bridge):
    registry.install(vector_bridge, verify(model, vector_bridge, gtzan, FitRequest(model="fake", dataset="gtzan")))
    samples = datasets.sample_files("gtzan")
    assert len(samples) == len(gtzan.labels)
    sample = next(s for s in samples if s["label"] == "rock")
    [bridge] = registry.get_bridges("fake")
    inp = bridge.prepare(Path(sample["path"]))
    assert inp.kind == "vector" and inp.layer == vector_bridge.layer and '"rock"' in inp.detail
    assert "rock" in model.chat("What genre is this?", [inp])
    assert "No audio" in model.chat("What genre is this?", [])


def test_vector_bridge_needs_an_open_weight_model(gtzan):
    class ApiModel:
        info = ModelInfo(name="api", label="API model", native=["text"], open_weights=False)

        def chat(self, message, inputs, max_new_tokens=96):
            return ""

        def choose(self, items):
            return [0] * len(items)

    with pytest.raises(ValueError, match="not an open-weight model"):
        bridges.fit_vector_bridge(ApiModel(), gtzan, gtzan.fit_idx)


# ----------------------------------------------------------------------------- settings
@pytest.mark.parametrize("value, on", [
    ("1", True), ("true", True), ("yes", True), ("on", True),
    ("", False), ("0", False), ("false", False), ("False", False), ("FALSE", False), ("no", False), ("No", False),
    ("off", False), ("OFF", False), (" 0 ", False),
])
def test_sense_fake_is_off_for_every_spelling_of_off(monkeypatch, value, on):
    monkeypatch.setenv("SENSE_FAKE", value)
    assert config.fake() is on


def test_env_local_wins_over_env_and_the_environment_wins_over_both(tmp_path, monkeypatch):
    keys = ["SENSE_TEST_DATA", "SENSE_TEST_ONLY_ENV", "SENSE_TEST_NEAR", "SENSE_TEST_SHELL"]
    for key in keys:  # set, then delete: the test leaves the environment as it found it
        monkeypatch.setenv(key, "x")
        monkeypatch.delenv(key)
    (tmp_path / ".env").write_text(
        "SENSE_TEST_DATA=/old/teammate/path\nSENSE_TEST_ONLY_ENV=from-env\nSENSE_TEST_SHELL=from-file\n"
    )
    (tmp_path / ".env.local").write_text("SENSE_TEST_DATA=/Users/me/corrected/path\nSENSE_TEST_NEAR=far\n")
    near = tmp_path / "sub"
    near.mkdir()
    (near / ".env").write_text("SENSE_TEST_NEAR=near\n")
    monkeypatch.setenv("SENSE_TEST_SHELL", "from-shell")
    monkeypatch.chdir(near)
    monkeypatch.setattr(config, "REPO", tmp_path)
    config._load_env_files()
    assert os.environ["SENSE_TEST_DATA"] == "/Users/me/corrected/path"  # .env.local beats .env in one directory
    assert os.environ["SENSE_TEST_ONLY_ENV"] == "from-env"
    assert os.environ["SENSE_TEST_NEAR"] == "near"  # a nearer directory beats its parents
    assert os.environ["SENSE_TEST_SHELL"] == "from-shell"  # the environment beats every file
