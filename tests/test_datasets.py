"""Folder sets: sense.datasets.from_folder, loading and listing them, and the HTTP routes and the CLI that use them.

All in fake mode, except one test that reads saved sets in real mode; that touches only set.json and set.npz and
loads no weights.

Run: SENSE_FAKE=1 .venv/bin/python -m pytest -q tests/test_datasets.py
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import shutil
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from agent import jobs
from agent import planner as planner_mod
from sense import config, datasets
from sense.specialists import AUDIO_EXTENSIONS, FAKE_CONCEPTS, FAKE_LABELS, FakeSpecialist
from server import app as server_app


@pytest.fixture(autouse=True)
def fake_home(monkeypatch, tmp_path):
    monkeypatch.setenv("SENSE_FAKE", "1")
    monkeypatch.setenv("SENSE_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("SENSE_SETS_DIR", raising=False)
    datasets._read_folder_set.cache_clear()
    assert config.fake()
    yield tmp_path / "home"
    datasets._read_folder_set.cache_clear()


def label_of(folder: str) -> str:
    return " ".join(folder.replace("_", " ").replace("-", " ").split())


def write_label(root: Path, folder: str, n: int, start: int = 0) -> list[Path]:
    """n distinct fake clips of the concept named by `folder` (e.g. "sea_waves" for "sea waves")."""
    concept = FAKE_CONCEPTS.index(label_of(folder))
    paths = []
    for i in range(start, start + n):
        path = root / folder / f"{folder}_{i:03d}.wav"
        FakeSpecialist.write_sample(concept, path, seed=1000 * concept + i)
        paths.append(path)
    return paths


def make_folder(root: Path, counts: dict[str, int]) -> Path:
    for folder, n in counts.items():
        write_label(root, folder, n)
    return root


def same_samples(src: Path, dst: Path) -> None:
    """Write the decoded samples of src to dst in dst's format: other bytes, the same audio."""
    wav, sr = sf.read(src, dtype="float32")
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(dst, wav, sr, subtype="PCM_16")


def split_of(ds) -> dict[str, str]:
    """Relative path ("dog/dog_000.wav") -> "fit", "dev" or "test"."""
    root = Path(ds.meta["folder"])
    out = {}
    for split, idx in (("fit", ds.fit_idx), ("dev", ds.dev_idx), ("test", ds.test_idx)):
        for i in idx:
            out[Path(ds.files[i]).relative_to(root).as_posix()] = split
    return out


@pytest.fixture
def folder(tmp_path):
    return make_folder(tmp_path / "Farm Sounds", {"dog": 10, "rooster": 10, "sea_waves": 10})


@pytest.fixture
def embed_calls(monkeypatch):
    """Every list of paths the fake specialist is asked to embed."""
    calls: list[list[Path]] = []
    original = FakeSpecialist.embed_files

    def counting(self, paths):
        calls.append(list(paths))
        return original(self, paths)

    monkeypatch.setattr(FakeSpecialist, "embed_files", counting)
    return calls


# ----------------------------------------------------------------------------- building
def test_a_folder_becomes_a_set_with_disjoint_splits(folder, fake_home):
    (folder / "__MACOSX" / "dog").mkdir(parents=True)  # ignored folders and files
    (folder / "__MACOSX" / "dog" / "._dog_000.wav").write_bytes(b"resource fork")
    (folder / ".cache").mkdir()
    (folder / "dog" / ".DS_Store").write_bytes(b"finder")
    (folder / "README.txt").write_text("not a label")

    ds = datasets.from_folder(folder)
    assert ds.name == "farm-sounds"
    assert ds.labels == ["dog", "rooster", "sea waves"]
    assert ds.title == "Farm Sounds (folder, 3 labels)"
    assert ds.question == "Which sound is in the audio clip?" and ds.modality == "audio"
    assert ds.emb.shape == (30, FakeSpecialist.dim) and ds.emb.dtype == np.float32
    assert ds.label_text_emb.shape == (3, FakeSpecialist.dim)
    assert list(np.bincount(ds.y)) == [10, 10, 10]

    fit, dev, test = set(ds.fit_idx), set(ds.dev_idx), set(ds.test_idx)
    assert not (fit & dev or fit & test or dev & test) and fit | dev | test == set(range(30))
    for c in range(3):  # 60/20/20 of 10 per label
        assert [int((ds.y[idx] == c).sum()) for idx in (ds.fit_idx, ds.dev_idx, ds.test_idx)] == [6, 2, 2]
    assert all(Path(f).is_absolute() and Path(f).is_file() for f in ds.files)
    assert all(Path(f).parent.name == ["dog", "rooster", "sea_waves"][c] for f, c in zip(ds.files, ds.y))

    m = ds.meta
    assert m["origin"] == "folder" and m["folder"] == str(folder.resolve()) and m["specialist"] == "fake"
    assert (m["n_found"], m["n_used"], m["non_audio"], m["duplicates"], m["n_skipped"]) == (30, 30, 0, 0, 0)
    assert m["per_label"] == {"dog": 10, "rooster": 10, "sea waves": 10}
    assert m["splits"] == {"fit": 18, "dev": 6, "test": 6}
    assert m["conflicts"] == [] and m["skipped"] == [] and m["near_duplicate_groups"] == 0
    assert m["embed_seconds"] >= 0 and m["embed_seconds_per_file"] >= 0 and m["created"].endswith("+00:00")
    assert (m["max_per_label"], m["min_per_label"], m["seed"]) == (50, 5, 0)
    json.dumps(m)  # the interface receives it as JSON

    saved = fake_home / "datasets" / "farm-sounds"
    assert (saved / "set.json").is_file() and (saved / "set.npz").is_file()
    assert [p.name for p in (fake_home / "datasets").iterdir()] == ["farm-sounds"]  # no staging left behind


def test_the_label_text_follows_the_question(tmp_path):
    folder = make_folder(tmp_path / "genres", {"jazz": 5, "rock": 5})
    spec = FakeSpecialist()
    music = datasets.from_folder(folder, "genres", question="Which genre of music is this?")
    np.testing.assert_allclose(music.label_text_emb, spec.embed_texts(["jazz music", "rock music"]))
    sounds = datasets.from_folder(folder, "genre-sounds")
    np.testing.assert_allclose(sounds.label_text_emb, spec.embed_texts(["the sound of jazz", "the sound of rock"]))


def test_the_same_seed_gives_the_same_split_and_another_seed_another(folder):
    a = datasets.from_folder(folder, "a", seed=0)
    b = datasets.from_folder(folder, "b", seed=0)
    c = datasets.from_folder(folder, "c", seed=1)
    assert split_of(a) == split_of(b)
    assert split_of(a) != split_of(c)
    np.testing.assert_array_equal(a.emb, c.emb)  # the same files, split differently


def test_the_cap_is_a_seeded_pick_and_bounds_the_embedding_pass(folder, embed_calls):
    a = datasets.from_folder(folder, "capped", max_per_label=6)
    assert sum(len(c) for c in embed_calls) == 18
    assert a.meta["per_label"] == {"dog": 6, "rooster": 6, "sea waves": 6} and a.meta["n_capped"] == 12
    assert a.meta["n_found"] == 30 and a.meta["n_used"] == 18
    again = datasets.from_folder(folder, "capped-again", max_per_label=6)
    other = datasets.from_folder(folder, "capped-other", max_per_label=6, seed=3)
    assert again.files == a.files
    assert other.files != a.files


def test_a_saved_set_is_reused_and_a_changed_folder_is_built_again(folder, embed_calls, fake_home):
    first = datasets.from_folder(folder, "farm")
    assert sum(len(c) for c in embed_calls) == 30
    embed_calls.clear()

    again = datasets.from_folder(folder, "farm")
    assert embed_calls == []
    np.testing.assert_array_equal(again.emb, first.emb)
    assert split_of(again) == split_of(first) and again.meta["created"] == first.meta["created"]

    datasets._read_folder_set.cache_clear()  # a restart: only the disk is left
    renamed = datasets.from_folder(folder, "farm", title="Farm", source="recorded by us", licence="CC0")
    assert embed_calls == []  # title, source and licence do not need the files embedded again
    assert renamed.title == "Farm" and renamed.meta["licence"] == "CC0"
    assert datasets.load_labelled_set("farm").title == "Farm"

    write_label(folder, "dog", 1, start=10)  # one more file
    rebuilt = datasets.from_folder(folder, "farm", title="Farm", source="recorded by us", licence="CC0")
    assert sum(len(c) for c in embed_calls) == 31
    assert rebuilt.meta["per_label"]["dog"] == 11
    assert len(datasets.load_labelled_set("farm").y) == 31  # the rebuilt set is read again
    assert [p.name for p in (fake_home / "datasets").iterdir()] == ["farm"]
    embed_calls.clear()

    datasets.from_folder(folder, "farm", min_per_label=4, title="Farm", source="recorded by us", licence="CC0")
    assert embed_calls  # another setting is another fingerprint


@pytest.mark.parametrize(
    "change",
    [
        {"question": "Which genre of music is this?"},
        {"seed": 1},
        {"max_per_label": 9},
        "specialist",
    ],
    ids=["question", "seed", "max_per_label", "specialist"],
)
def test_every_fingerprint_setting_rebuilds(folder, embed_calls, monkeypatch, change):
    datasets.from_folder(folder, "farm")
    embed_calls.clear()
    if change == "specialist":  # what a real-mode CLAP build would record, without loading CLAP
        monkeypatch.setattr(datasets, "_specialist_id", lambda: "laion/clap-htsat-unfused")
        change = {}
    ds = datasets.from_folder(folder, "farm", **change)
    assert embed_calls  # not the saved set
    if "question" in change:
        assert ds.question == change["question"]
        np.testing.assert_allclose(ds.label_text_emb[0], FakeSpecialist().embed_texts(["dog music"])[0])


def test_a_file_rewritten_in_place_rebuilds(folder, embed_calls):
    first = datasets.from_folder(folder, "farm")
    clip = (folder / "dog" / "dog_000.wav").resolve()
    row = first.files.index(str(clip))
    FakeSpecialist.write_sample(FAKE_CONCEPTS.index("dog"), clip, seed=999)  # other audio under the same name
    later = clip.stat().st_mtime_ns + 5_000_000_000
    os.utime(clip, ns=(later, later))
    embed_calls.clear()
    again = datasets.from_folder(folder, "farm")
    assert embed_calls
    assert not np.allclose(again.emb[again.files.index(str(clip))], first.emb[row])


def test_a_default_call_keeps_the_recorded_title_source_and_licence(folder, embed_calls):
    datasets.from_folder(folder, "farm", title="Farm sounds", source="https://example.org/farm", licence="CC-BY 4.0")
    embed_calls.clear()
    kept = datasets.from_folder(folder, "farm")  # as POST /api/datasets and demo_cli.py --folder call it
    assert embed_calls == []
    assert (kept.title, kept.meta["source"], kept.meta["licence"]) == ("Farm sounds", "https://example.org/farm",
                                                                       "CC-BY 4.0")
    on_disk = datasets.load_labelled_set("farm")
    assert (on_disk.title, on_disk.meta["source"], on_disk.meta["licence"]) == (kept.title, kept.meta["source"],
                                                                                kept.meta["licence"])

    rebuilt = datasets.from_folder(folder, "farm", max_per_label=8)  # another setting: built again
    assert embed_calls
    assert (rebuilt.meta["source"], rebuilt.meta["licence"]) == ("https://example.org/farm", "CC-BY 4.0")
    assert rebuilt.title == "Farm Sounds (folder, 3 labels)"  # a rebuild gets the default title unless given one

    renamed = datasets.from_folder(folder, "farm", max_per_label=8, title="Farm", licence="CC0")
    assert (renamed.title, renamed.meta["source"], renamed.meta["licence"]) == ("Farm", "https://example.org/farm",
                                                                                "CC0")


def test_a_damaged_saved_set_is_built_again(folder, embed_calls, fake_home):
    datasets.from_folder(folder, "farm")
    (fake_home / "datasets" / "farm" / "set.npz").write_bytes(b"truncated")
    datasets._read_folder_set.cache_clear()
    embed_calls.clear()
    ds = datasets.from_folder(folder, "farm")
    assert embed_calls and len(ds.y) == 30
    assert len(datasets.load_labelled_set("farm").y) == 30


# ----------------------------------------------------------------------------- duplicates and bad files
def test_an_identical_copy_within_a_label_collapses(folder):
    shutil.copy(folder / "dog" / "dog_000.wav", folder / "dog" / "zz_copy.wav")
    ds = datasets.from_folder(folder)
    assert ds.meta["duplicates"] == 1 and ds.meta["conflicts"] == []
    assert ds.meta["n_found"] == 31 and ds.meta["per_label"]["dog"] == 10
    assert not any(f.endswith("zz_copy.wav") for f in ds.files)  # the first copy by path is kept


def test_a_copy_under_two_labels_is_dropped_from_both(folder):
    shutil.copy(folder / "dog" / "dog_000.wav", folder / "rooster" / "mislabelled.wav")
    ds = datasets.from_folder(folder)
    assert ds.meta["conflicts"] == ["dog/dog_000.wav", "rooster/mislabelled.wav"]
    assert ds.meta["per_label"] == {"dog": 9, "rooster": 10, "sea waves": 10}
    assert not any(f.endswith(("dog_000.wav", "mislabelled.wav")) for f in ds.files)


def test_near_duplicates_never_span_splits(tmp_path):
    folder = make_folder(tmp_path / "near", {"rooster": 10, "sea_waves": 10})
    for path in write_label(folder, "dog", 8):  # each dog clip twice: as WAV and as FLAC, the same audio
        same_samples(path, path.with_suffix(".flac"))
    same_samples(folder / "rooster" / "rooster_000.wav", folder / "sea_waves" / "odd.flac")  # across labels
    for seed in range(6):
        ds = datasets.from_folder(folder, f"near-{seed}", seed=seed)
        assert ds.meta["near_duplicate_groups"] == 8 and ds.meta["duplicates"] == 0
        assert ds.meta["near_duplicates_across_labels"] == 1  # counted, not dropped
        assert ds.meta["per_label"] == {"dog": 16, "rooster": 10, "sea waves": 11}
        where = split_of(ds)
        for i in range(8):
            assert where[f"dog/dog_{i:03d}.wav"] == where[f"dog/dog_{i:03d}.flac"]
        dog = [s for f, s in where.items() if f.startswith("dog/")]
        assert {"fit", "dev", "test"} == set(dog)


def test_the_near_duplicate_threshold_is_a_cosine_of_0_999():
    def at(angle: float) -> np.ndarray:
        return np.array([[1.0, 0.0], [np.cos(angle), np.sin(angle)]])

    assert datasets._near_duplicate_groups(at(0.03)) == [[0, 1]]  # cosine 0.99955
    assert datasets._near_duplicate_groups(at(0.05)) == [[0], [1]]  # cosine 0.99875


def test_a_large_group_does_not_empty_dev_or_test(tmp_path):
    folder = make_folder(tmp_path / "groups", {"rooster": 10})
    first, *_ = write_label(folder, "dog", 3)
    for suffix in (".flac", ".aiff", ".au"):  # one group of 4 and two single clips
        same_samples(first, first.with_suffix(suffix))
    for seed in range(6):
        ds = datasets.from_folder(folder, f"groups-{seed}", seed=seed)
        assert ds.meta["near_duplicate_groups"] == 1
        dog = {s for f, s in split_of(ds).items() if f.startswith("dog/")}
        assert dog == {"fit", "dev", "test"}


def test_a_split_with_too_few_examples_to_fit_on_is_refused(tmp_path):
    two = make_folder(tmp_path / "two", {"dog": 4, "rooster": 4})
    with pytest.raises(ValueError, match="leaves 4 examples to fit on \\(dog 2, rooster 2\\).*at least 5"):
        datasets.from_folder(two, min_per_label=4)

    # at the defaults, near duplicates can push clips out of fit: 5 files but 3 distinct clips per label
    folder = tmp_path / "samey"
    for label in ("dog", "rooster"):
        first, *_ = write_label(folder, label, 3)
        same_samples(first, first.with_suffix(".flac"))
        same_samples(first, first.with_suffix(".aiff"))
    refused = 0
    for seed in range(6):
        try:
            ds = datasets.from_folder(folder, f"samey-{seed}", seed=seed)
        except ValueError as err:
            assert "examples to fit on" in str(err)
            refused += 1
        else:
            assert len(ds.fit_idx) >= 5
    assert refused


def test_fit_problem_says_when_a_run_cannot_get_going(tmp_path):
    three = datasets.from_folder(make_folder(tmp_path / "three", {"dog": 4, "rooster": 4, "sea_waves": 4}),
                                 min_per_label=4)
    assert len(three.fit_idx) == 6
    assert datasets.fit_problem(three, "closed") is None
    assert "1 label left out, which can leave 4 examples" in datasets.fit_problem(three, "open")
    two = datasets.from_folder(make_folder(tmp_path / "pair", {"dog": 6, "rooster": 6}))
    assert datasets.fit_problem(two, "closed") is None
    assert "at least 3 labels" in datasets.fit_problem(two, "open")
    for name in ("gtzan", "esc50"):
        assert datasets.fit_problem(datasets.load_labelled_set(name), "open") is None


def test_fit_problem_skips_the_vector_checks_for_a_model_that_is_not_open_weight(tmp_path):
    """An API model gets only text and tool bridges: nothing is fitted and there is no unseen-classes control,
    so an open run on two labels gets as far as verifying, and installs."""
    from dataclasses import replace

    from agent import loop
    from sense.contracts import FitRequest

    two = datasets.from_folder(make_folder(tmp_path / "pair", {"dog": 6, "rooster": 6}))
    assert "at least 3 labels" in datasets.fit_problem(two, "open", open_weights=True)
    assert datasets.fit_problem(two, "open", open_weights=False) is None
    assert datasets.fit_problem(two, "closed", open_weights=False) is None
    three = datasets.from_folder(make_folder(tmp_path / "three", {"dog": 4, "rooster": 4, "sea_waves": 4}),
                                 min_per_label=4)
    assert datasets.fit_problem(three, "open", open_weights=False) is None
    cards, installed = loop.run_fit(FitRequest(model="fake-api", dataset=two.name, scope="open"), lambda event: None)
    assert {card.kind for card in cards} == {"text", "tool"}
    assert all("unseen_classes" not in card.controls for card in cards)
    assert installed
    elsewhere = two.meta | {"specialist": "laion/clap-htsat-unfused"}  # the mode check still applies
    assert "was embedded by the laion/clap-htsat-unfused specialist" in datasets.fit_problem(
        replace(two, meta=elsewhere), "open", open_weights=False
    )


def test_non_audio_and_unreadable_files_are_skipped_and_counted(folder):
    (folder / "dog" / "notes.txt").write_text("recorded at the farm")
    (folder / "dog" / "broken.wav").write_bytes(b"not audio at all")
    ds = datasets.from_folder(folder)
    m = ds.meta
    assert m["non_audio"] == 1 and m["n_found"] == 31 and m["n_skipped"] == 1
    [skipped] = m["skipped"]
    assert skipped["file"] == "dog/broken.wav" and "Could not read broken.wav as audio" in skipped["why"]
    assert str(folder) not in skipped["why"]
    assert m["per_label"]["dog"] == 10  # the rest of the batch was embedded file by file


def test_a_short_label_is_reported_with_why_its_files_were_not_used(tmp_path, embed_calls):
    folder = make_folder(tmp_path / "phone", {"rooster": 8})
    for i in range(8):  # voice memos: audio, but of a type the specialist does not read
        (folder / "dog" / f"memo_{i}.m4a").parent.mkdir(exist_ok=True)
        (folder / "dog" / f"memo_{i}.m4a").write_bytes(f"m4a {i}".encode())
    [clip] = write_label(folder, "sea_waves", 1)
    for i in range(5):
        shutil.copy(clip, folder / "sea_waves" / f"copy_{i}.wav")
    with pytest.raises(ValueError) as err:
        datasets.from_folder(folder)
    message = str(err.value)
    assert "dog has 0 of 8 files: 8 of a type that is not read (.m4a)" in message
    assert "sea waves has 1 of 6 files: 5 exact duplicates" in message
    assert ".flac" in message and "rooster" not in message
    assert embed_calls == []  # found before any embedding

    folder = make_folder(tmp_path / "broken", {"rooster": 8, "dog": 8})
    for i in range(8):
        (folder / "cat" / f"cat_{i}.wav").parent.mkdir(exist_ok=True)
        (folder / "cat" / f"cat_{i}.wav").write_bytes(f"not audio {i}".encode())
    with pytest.raises(ValueError, match=r"cat has 0 of 8 files: 8 could not be read \(e.g. cat/cat_0.wav: Could not"):
        datasets.from_folder(folder)


def test_the_skipped_list_is_capped_but_counted(folder):
    for i in range(25):
        (folder / "rooster" / f"broken_{i:02d}.wav").write_bytes(f"not audio {i}".encode())  # distinct bytes
    ds = datasets.from_folder(folder)
    assert ds.meta["n_skipped"] == 25 and len(ds.meta["skipped"]) == 20 and ds.meta["duplicates"] == 0
    assert ds.meta["per_label"]["rooster"] == 10


# ----------------------------------------------------------------------------- errors
def test_errors_say_what_is_wrong(tmp_path, folder):
    with pytest.raises(ValueError, match="not an existing folder"):
        datasets.from_folder(tmp_path / "missing")
    with pytest.raises(ValueError, match="not an existing folder"):
        datasets.from_folder(folder / "dog" / "dog_000.wav")
    with pytest.raises(ValueError, match="No folder was given"):
        datasets.from_folder("")

    one = make_folder(tmp_path / "one", {"dog": 10})
    with pytest.raises(ValueError, match="1 label folder .*at least 2"):
        datasets.from_folder(one)

    short = make_folder(tmp_path / "short", {"dog": 10, "rooster": 3, "sea_waves": 2})
    with pytest.raises(ValueError, match="rooster has 3; sea waves has 2. .*at least 5.*lower min_per_label"):
        datasets.from_folder(short)

    with pytest.raises(ValueError, match="min_per_label is 3"):
        datasets.from_folder(folder, min_per_label=3)
    with pytest.raises(ValueError, match="max_per_label"):
        datasets.from_folder(folder, max_per_label=4, min_per_label=5)

    for name in ("gtzan", "ESC50"):
        with pytest.raises(ValueError, match="taken by a built-in set. Pass another name"):
            datasets.from_folder(folder, name)
    builtin_named = make_folder(tmp_path / "gtzan", {"dog": 5, "rooster": 5})
    with pytest.raises(ValueError, match="taken by a built-in set"):
        datasets.from_folder(builtin_named)
    with pytest.raises(ValueError, match="empty set name"):
        datasets.from_folder(folder, "!!!")
    with pytest.raises(ValueError, match="No specialist is known"):
        datasets.from_folder(folder, modality="video")

    clash = make_folder(tmp_path / "clash", {"dog": 5, "sea_waves": 5})
    write_label(clash, "sea-waves", 5)
    with pytest.raises(ValueError, match="'sea-waves' and 'sea_waves'|'sea_waves' and 'sea-waves'"):
        datasets.from_folder(clash)


def test_a_source_folder_inside_the_sets_dir_is_refused_and_left_alone(tmp_path, monkeypatch):
    monkeypatch.setenv("SENSE_SETS_DIR", str(tmp_path / "my_sets"))
    birds = make_folder(tmp_path / "my_sets" / "birds", {"dog": 6, "rooster": 6})
    before = sorted(birds.rglob("*"))
    for path in (birds, tmp_path / "my_sets"):
        with pytest.raises(ValueError, match="lies inside SENSE_SETS_DIR.*Move the folder out of it"):
            datasets.from_folder(path)
    assert sorted(birds.rglob("*")) == before


def test_a_name_taken_by_another_directory_is_refused_and_left_alone(folder, fake_home, embed_calls):
    other = fake_home / "datasets" / "farm"
    other.mkdir(parents=True)
    (other / "notes.txt").write_text("not a saved set")
    with pytest.raises(ValueError, match="is not a set saved by from_folder, so it is left alone. Pass another name"):
        datasets.from_folder(folder, "farm")
    assert (other / "notes.txt").read_text() == "not a saved set" and embed_calls == []

    (other / "notes.txt").unlink()  # an empty directory is replaced
    assert len(datasets.from_folder(folder, "farm").y) == 30
    assert sorted(p.name for p in other.iterdir()) == ["set.json", "set.npz"]


def test_a_label_of_near_identical_clips_cannot_be_split(tmp_path):
    folder = make_folder(tmp_path / "samey", {"rooster": 10})
    for path in write_label(folder, "dog", 2):  # six files but only two distinct clips
        same_samples(path, path.with_suffix(".flac"))
        same_samples(path, path.with_suffix(".aiff"))
    with pytest.raises(ValueError, match="Too few distinct clips.*dog \\(6 usable files, 2 distinct\\)"):
        datasets.from_folder(folder)


def test_the_internal_leak_check_raises():
    datasets._check_no_leak(np.array([0, 0, 2]), np.array([0, 0, 1]), ["a", "b", "c"])  # one group, one split
    with pytest.raises(RuntimeError, match="near-duplicate group spans two splits"):
        datasets._check_no_leak(np.array([0, 1]), np.array([7, 7]), ["a", "b"])
    with pytest.raises(RuntimeError, match="exact-duplicate group spans two splits"):
        datasets._check_no_leak(np.array([0, 1]), np.array([1, 2]), ["same", "same"])


def test_names_are_cleaned():
    assert datasets._set_name("My Sounds!  (v2)") == "my-sounds-v2"
    assert datasets._set_name("--Dog--Bark--") == "dog-bark"
    assert ".aiff" in AUDIO_EXTENSIONS and server_app.AUDIO_EXTENSIONS is AUDIO_EXTENSIONS


# ----------------------------------------------------------------------------- loading, listing, samples
def test_a_folder_set_survives_a_restart(folder):
    built = datasets.from_folder(folder, "farm", source="https://example.org/farm", licence="CC-BY 4.0")
    datasets._read_folder_set.cache_clear()

    ds = datasets.load_labelled_set("farm")
    assert ds.labels == built.labels and ds.files == built.files and ds.meta == built.meta
    np.testing.assert_array_equal(ds.emb, built.emb)
    np.testing.assert_array_equal(ds.label_text_emb, built.label_text_emb)
    assert split_of(ds) == split_of(built)

    listing = {d["name"]: d for d in datasets.available_sets()}
    assert list(listing) == ["gtzan", "esc50", "farm"]
    assert listing["gtzan"]["origin"] == "builtin" and listing["gtzan"]["meta"] == {}
    farm = listing["farm"]
    assert farm["origin"] == "folder" and farm["n"] == 30 and farm["labels"] == ["dog", "rooster", "sea waves"]
    assert farm["meta"]["source"] == "https://example.org/farm" and farm["meta"]["licence"] == "CC-BY 4.0"
    assert set(farm) == {"name", "title", "modality", "labels", "n", "origin", "meta"}


def test_unknown_and_damaged_sets(folder, fake_home, caplog):
    datasets.from_folder(folder, "farm")
    with pytest.raises(ValueError, match="Unknown dataset 'nope'. Choose one of: gtzan, esc50, farm.*from_folder"):
        datasets.load_labelled_set("nope")
    with pytest.raises(ValueError, match="Unknown dataset"):
        datasets.load_labelled_set("../home")

    broken = fake_home / "datasets" / "broken"
    broken.mkdir()
    (broken / "set.json").write_text("{not json")
    with caplog.at_level(logging.WARNING, logger="sense.datasets"):
        names = [d["name"] for d in datasets.available_sets()]
    assert names == ["gtzan", "esc50", "farm"]
    assert "broken" in caplog.text
    with pytest.raises(ValueError, match="cannot be read.*Build it again"):
        datasets.load_labelled_set("broken")
    with pytest.raises(ValueError, match="Unknown dataset 'gone'"):  # removed between finding and reading it
        datasets._load_folder_set("gone")


def test_names_cannot_reach_outside_the_sets_dir(folder, fake_home):
    datasets.from_folder(folder, "farm")
    shutil.copytree(fake_home / "datasets" / "farm", fake_home / "evil")
    with pytest.raises(ValueError, match="Unknown dataset"):
        datasets.load_labelled_set("../evil")


def test_folder_sets_load_in_real_mode(folder, monkeypatch, tmp_path, caplog):
    built = datasets.from_folder(folder, "farm")
    datasets._read_folder_set.cache_clear()
    monkeypatch.delenv("SENSE_FAKE")
    monkeypatch.setenv("SENSE_DATA_DIR", str(tmp_path / "no-builtins"))
    assert not config.fake()
    with caplog.at_level(logging.WARNING, logger="sense.datasets"):
        assert [d["name"] for d in datasets.available_sets()] == ["farm"]
    assert "fake specialist" in caplog.text
    assert datasets.load_labelled_set("farm").files == built.files
    samples = datasets.sample_files("farm")
    assert len(samples) == 3 and all(Path(s["path"]).is_file() for s in samples)
    problem = datasets.fit_problem(datasets.load_labelled_set("farm"))
    assert "embedded by the fake specialist, but the harness is in real mode" in problem


def test_readers_never_see_a_set_missing_while_it_is_replaced(folder):
    datasets.from_folder(folder, "farm")
    errors: list[str] = []
    done = threading.Event()

    def read() -> None:
        while not done.is_set():
            try:
                datasets.load_labelled_set("farm")
                assert "farm" in [d["name"] for d in datasets.available_sets()]
            except Exception as err:  # noqa: BLE001 - every failure counts
                errors.append(f"{type(err).__name__}: {err}")

    reader = threading.Thread(target=read)
    reader.start()
    try:
        for i in range(40):  # the save path only: a new title, no embedding
            datasets.from_folder(folder, "farm", title=f"Farm {i}")
    finally:
        done.set()
        reader.join()
    assert errors == []
    assert datasets.load_labelled_set("farm").title == "Farm 39"


def test_sets_dir_follows_sense_sets_dir(folder, monkeypatch, tmp_path):
    monkeypatch.setenv("SENSE_SETS_DIR", str(tmp_path / "elsewhere"))
    assert config.sets_dir() == tmp_path / "elsewhere"
    datasets.from_folder(folder, "farm")
    assert (tmp_path / "elsewhere" / "farm" / "set.json").is_file()
    assert "farm" in [d["name"] for d in datasets.available_sets()]


def test_sample_files_are_the_real_held_out_files(folder):
    ds = datasets.from_folder(folder, "farm")
    samples = datasets.sample_files("farm")
    assert [s["label"] for s in samples] == ["dog", "rooster", "sea waves"]
    test_files = {ds.files[i] for i in ds.test_idx}
    for s in samples:
        assert s["path"] in test_files and Path(s["path"]).is_file()
        assert Path(s["path"]).is_relative_to(folder.resolve())
    assert len(datasets.sample_files("farm", per_label=2)) == 6
    fake = datasets.sample_files("gtzan")  # the synthetic built-in sets still write their own files
    assert len(fake) == len(FAKE_LABELS["gtzan"]) and all(Path(s["path"]).is_file() for s in fake)


# ----------------------------------------------------------------------------- HTTP
@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(jobs, "_jobs", {})
    monkeypatch.setattr(jobs, "_latest_id", None)
    monkeypatch.setattr(planner_mod, "_preferred", None)  # the rule planner, even if an LLM planner is registered
    server_app._sample_paths.clear()
    return TestClient(server_app.app)


def test_post_datasets_builds_a_set(client, folder, tmp_path):
    r = client.post("/api/datasets", json={"path": str(folder), "name": "Farm", "title": "Farm animals and the sea"})
    assert r.status_code == 200, r.text
    entry = r.json()
    assert entry["name"] == "farm" and entry["origin"] == "folder" and entry["n"] == 30
    assert entry["title"] == "Farm animals and the sea" and entry["meta"]["n_used"] == 30
    assert entry in client.get("/api/datasets").json()

    r = client.post("/api/datasets", json={"path": str(folder), "max_per_label": 5, "question": "What is this?"})
    assert r.status_code == 200, r.text
    assert r.json()["name"] == "farm-sounds" and r.json()["meta"]["per_label"]["dog"] == 5
    assert datasets.load_labelled_set("farm-sounds").question == "What is this?"

    r = client.post("/api/datasets", json={"path": str(tmp_path / "missing")})
    assert r.status_code == 400 and "not an existing folder" in r.json()["detail"]
    r = client.post("/api/datasets", json={"path": str(folder), "name": "esc50"})
    assert r.status_code == 400 and "taken by a built-in set" in r.json()["detail"]


def test_open_scope_needs_three_labels(client, tmp_path):
    two = make_folder(tmp_path / "two", {"dog": 6, "rooster": 6})
    assert client.post("/api/datasets", json={"path": str(two)}).status_code == 200
    r = client.post("/api/fit", json={"model": "fake", "dataset": "two", "scope": "open"})
    assert r.status_code == 400 and "at least 3 labels" in r.json()["detail"]

    datasets.from_folder(make_folder(tmp_path / "small", {"dog": 4, "rooster": 4, "sea_waves": 4}), min_per_label=4)
    r = client.post("/api/fit", json={"model": "fake", "dataset": "small", "scope": "open"})
    assert r.status_code == 400 and "can leave 4 examples" in r.json()["detail"]


def test_an_api_model_may_fit_open_scope_on_two_labels(client, tmp_path):
    """POST /api/fit passes the model's open_weights to fit_problem: an API model builds no vector bridge and
    runs no unseen-classes control, so a 2-label set is enough for scope open (a local model is still refused)."""
    two = make_folder(tmp_path / "two", {"dog": 6, "rooster": 6})
    assert client.post("/api/datasets", json={"path": str(two)}).status_code == 200
    assert client.post("/api/fit", json={"model": "fake", "dataset": "two", "scope": "open"}).status_code == 400
    r = client.post("/api/fit", json={"model": "fake-api", "dataset": "two", "scope": "open"})
    assert r.status_code == 200, r.text
    for _ in range(400):
        state = client.get(f"/api/jobs/{r.json()['job_id']}").json()
        if state["status"] != "running":
            break
        time.sleep(0.02)
    assert state["status"] == "done", state["error"]


def test_posting_a_folder_again_keeps_what_the_agent_recorded(client, folder):
    datasets.from_folder(folder, "farm", title="Farm sounds", source="https://example.org/farm", licence="CC-BY 4.0")
    r = client.post("/api/datasets", json={"path": str(folder), "name": "farm"})
    assert r.status_code == 200, r.text
    entry = r.json()
    assert entry["title"] == "Farm sounds"
    assert (entry["meta"]["source"], entry["meta"]["licence"]) == ("https://example.org/farm", "CC-BY 4.0")


def test_a_closed_fit_on_a_folder_set_is_checked_with_fit_problem(client, folder, monkeypatch):
    datasets.from_folder(folder, "farm")
    monkeypatch.setattr(datasets, "fit_problem", lambda ds, scope="closed", **kw: f"{ds.name} was embedded elsewhere")
    r = client.post("/api/fit", json={"model": "fake", "dataset": "farm", "scope": "closed"})
    assert r.status_code == 400 and r.json()["detail"] == "farm was embedded elsewhere"


def test_a_folder_set_runs_end_to_end(client, tmp_path):
    folder = make_folder(tmp_path / "field", {"dog": 10, "rooster": 10, "sea_waves": 10, "chainsaw": 10})
    assert client.post("/api/datasets", json={"path": str(folder), "name": "field"}).status_code == 200

    r = client.post("/api/fit", json={"model": "fake", "dataset": "field", "scope": "closed"})
    assert r.status_code == 200, r.text
    job_id = r.json()["job_id"]
    for _ in range(400):
        state = client.get(f"/api/jobs/{job_id}").json()
        if state["status"] != "running":
            break
        time.sleep(0.02)
    assert state["status"] == "done", state["error"]
    assert state["installed"] == "vector:field"
    json.dumps(state)
    assert [s["bridge"] for s in client.get("/api/senses", params={"model": "fake"}).json()] == ["vector:field"]

    samples = client.get("/api/samples", params={"dataset": "field"}).json()
    assert sorted(s["label"] for s in samples) == ["chainsaw", "dog", "rooster", "sea waves"]
    rooster = next(s for s in samples if s["label"] == "rooster")
    assert client.get(f"/api/samples/{rooster['id']}").status_code == 200
    r = client.post("/api/chat", data={"model": "fake", "message": "What is this?", "sample_ids": [rooster["id"]]})
    assert r.status_code == 200, r.text
    assert "rooster" in r.json()["reply"] and r.json()["used"][0]["bridge"] == "vector:field"


# ----------------------------------------------------------------------------- scripts/demo_cli.py
@pytest.fixture
def cli(monkeypatch, client):  # client: a fresh job table and the rule planner
    spec = importlib.util.spec_from_file_location("demo_cli", config.REPO / "scripts" / "demo_cli.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def run(*argv: str) -> int:
        monkeypatch.setattr(sys, "argv", ["demo_cli.py", *argv])
        return module.main()

    return run, module


def test_the_cli_builds_a_set_from_a_folder_and_runs_on_it(cli, tmp_path, capsys):
    run, module = cli
    assert module.CONTROL_NAMES["specialist_alone"] == "specialist alone"
    folder = make_folder(tmp_path / "field", {"dog": 10, "rooster": 10, "sea_waves": 10})
    code = run("--model", "fake", "--folder", str(folder), "--name", "My Set", "--question", "What made this sound?",
               "--scope", "closed")
    out = capsys.readouterr().out
    assert code == 0, out
    assert "Folder set my-set: 3 labels, 30 of 30 audio files used (18 fit, 6 dev, 6 test)" in out
    assert "Installed: vector:my-set" in out
    assert datasets.load_labelled_set("my-set").question == "What made this sound?"

    with pytest.raises(SystemExit):
        run("--model", "fake", "--dataset", "my-set", "--name", "other")
    assert "--name and --question go with --folder" in capsys.readouterr().err


def test_the_cli_checks_open_scope_for_folder_and_dataset_alike(cli, tmp_path, capsys):
    run, _ = cli
    two = make_folder(tmp_path / "two", {"dog": 6, "rooster": 6})
    assert run("--model", "fake", "--folder", str(two), "--scope", "open") == 1
    assert "at least 3 labels" in capsys.readouterr().err
    assert run("--model", "fake", "--dataset", "two", "--scope", "open") == 1
    captured = capsys.readouterr()
    assert "at least 3 labels" in captured.err and "two has 2" in captured.err and "Agent run" not in captured.out


# ----------------------------------------------------------------------------- built-in gtzan duplicates (#41)
def test_split_without_duplicates_matches_the_plain_split_when_there_are_none():
    ds = datasets._fake_set("gtzan")  # no two fake clips are near-identical
    fit, dev, test, info = datasets._split_without_duplicates(ds.emb, ds.y, len(ds.labels), seed=0)
    plain = datasets._split_per_label(ds.y, len(ds.labels), seed=0)
    for got, want in zip((fit, dev, test), plain):
        assert got.tolist() == want.tolist()
    assert info == {"duplicate_groups_kept_in_one_split": 0, "left_out_idx": []}


def test_split_without_duplicates_keeps_copies_together_and_drops_cross_label_copies():
    rng = np.random.default_rng(3)
    n_labels, per_label = 3, 20
    y = np.repeat(np.arange(n_labels), per_label)
    emb = rng.normal(size=(len(y), 16)).astype(np.float32)
    copies = [(0, 1), (0, 2), (25, 33), (41, 59)]  # within labels 0, 1 and 2; 0-1-2 is one group of three
    for a, b in copies:
        emb[b] = emb[a]
    emb[45] = emb[5]  # clip 5 (label 0) and clip 45 (label 2) are one recording under two labels
    for seed in range(10):
        fit, dev, test, info = datasets._split_without_duplicates(emb, y, n_labels, seed=seed)
        where = {int(i): k for k, split in enumerate((fit, dev, test)) for i in split}
        for a, b in copies:
            assert where[a] == where[b], (seed, a, b)
        assert 5 not in where and 45 not in where
        assert info["left_out_idx"] == [5, 45]
        assert info["duplicate_groups_kept_in_one_split"] == 3
        assert len(where) == len(y) - 2 and len(fit) + len(dev) + len(test) == len(where)


def test_real_gtzan_split_has_no_duplicate_across_splits():
    path = config.data_dir() / "gtzan_clap.npz"
    if not path.is_file():
        pytest.skip("the real gtzan_clap.npz is not on this machine")
    ds = datasets._real_set("gtzan", str(config.data_dir()), ())
    unit = ds.emb / np.linalg.norm(ds.emb, axis=1, keepdims=True)
    split = np.full(len(ds.y), -1)
    for k, idx in enumerate((ds.fit_idx, ds.dev_idx, ds.test_idx)):
        split[idx] = k
    near = np.triu(unit @ unit.T >= 0.999, k=1)
    for i, j in zip(*np.nonzero(near)):
        assert split[i] == split[j] or -1 in (split[i], split[j])
    assert ds.meta["left_out"] == ["metal.00058.wav", "rock.00016.wav"]
    assert ds.meta["duplicate_groups_kept_in_one_split"] == 13
    assert len(ds.fit_idx) + len(ds.dev_idx) + len(ds.test_idx) == len(ds.y) - 2
