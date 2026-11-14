"""Free-text scoring (sense.verify.free_text, scripts/score_free_text.py) and run records (scripts/demo_cli.py --save),
in fake mode.

Run: SENSE_FAKE=1 .venv/bin/python -m pytest -q tests/test_free_text.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from sense import bridges, config, datasets, models, registry
from sense.contracts import FitRequest, JobState
from sense.verify import free_text, labels_named, verify

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


@pytest.fixture(autouse=True)
def fake_env(monkeypatch, tmp_path):
    monkeypatch.setenv("SENSE_FAKE", "1")
    monkeypatch.setenv("SENSE_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("SENSE_DATA_DIR", str(tmp_path / "data"))
    assert config.fake()


@pytest.fixture
def model():
    return models.load_model("fake")


@pytest.fixture
def gtzan():
    return datasets.load_labelled_set("gtzan")


@pytest.fixture
def vector_bridge(model, gtzan):
    return bridges.fit_vector_bridge(model, gtzan, gtzan.fit_idx)


def load_script(name: str):
    """Import scripts/<name>.py as a module, so its main() can be called in this process."""
    spec = importlib.util.spec_from_file_location(f"script_{name}", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ----------------------------------------------------------------------------- labels_named
@pytest.mark.parametrize("text", ["Hip-hop", "hiphop", "hip hop", "It is HIP_HOP.", "some hip  hop, I think"])
def test_every_spelling_of_hip_hop_names_it(text):
    labels = ["rock", "hip hop", "pop"]
    assert labels_named(text, labels) == [1]
    assert labels_named(text, ["hiphop"]) == [0]  # the raw GTZAN spelling too


@pytest.mark.parametrize(
    ("text", "label"),
    [
        ("dogs barking", "dog"),
        ("A dog.", "dog"),
        ("sea waves crashing", "sea waves"),
        ("waves of the sea", "sea waves"),
        ("a sea wave", "sea waves"),
        ("church bells ringing", "church_bells"),
        ("boxes falling", "box"),
    ],
)
def test_a_label_is_named_by_its_words_or_a_simple_plural(text, label):
    assert labels_named(text, [label]) == [0]


@pytest.mark.parametrize(
    ("text", "label"),
    [
        ("rockabilly", "rock"),
        ("a rocky road", "rock"),
        ("the sea", "sea waves"),
        ("a ticking clock", "clock tick"),
        ("", "dog"),
    ],
)
def test_partial_words_and_missing_words_do_not_name_a_label(text, label):
    assert labels_named(text, [label]) == []


def test_a_text_naming_two_labels_returns_both_in_label_order():
    labels = ["blues", "classical", "jazz", "rock"]
    assert labels_named("Rock, or maybe jazz.", labels) == [2, 3]
    assert labels_named("Nothing I recognise.", labels) == []
    assert labels_named("anything", ["", "--"]) == []  # a label with no words is never named


# ----------------------------------------------------------------------------- free_text
def test_free_text_through_a_vector_bridge_beats_its_controls(model, gtzan, vector_bridge):
    result = free_text(model, vector_bridge, gtzan)
    assert json.loads(json.dumps(result)) == result  # plain JSON all the way down
    assert {"model", "dataset", "bridge", "kind", "question", "n", "seed", "max_new_tokens", "result", "controls",
            "examples", "seconds", "created"} <= set(result)
    assert (result["model"], result["dataset"], result["bridge"], result["kind"]) == ("fake", "gtzan", "vector:gtzan", "vector")
    assert result["question"] == gtzan.question + " Answer in a few words."
    assert result["n"] == len(gtzan.test_idx) and result["max_new_tokens"] == 32
    assert set(result["controls"]) == {"swapped", "no_input"}
    for stats in [result["result"], *result["controls"].values()]:
        assert set(stats) == {"n", "names_true", "names_true_only", "names_other", "names_none", "interval"}
        assert stats["n"] == result["n"]
        low, high = stats["interval"]
        assert 0.0 <= low <= stats["names_true_only"] <= high <= 1.0
        assert stats["names_true_only"] <= stats["names_true"]

    assert result["result"]["names_true_only"] >= 0.8
    assert result["controls"]["swapped"]["names_true"] <= 0.4
    assert result["controls"]["no_input"]["names_none"] >= 0.8

    examples = result["examples"]
    assert 1 <= len(examples) <= 10
    assert len({ex["label"] for ex in examples}) > 1  # spread over the held-out clips, not ten of one label
    for ex in examples:
        assert set(ex) == {"source", "label", "reply", "named"}
        assert ex["label"] in gtzan.labels and ex["reply"]
        assert ex["named"] == [gtzan.labels[k] for k in labels_named(ex["reply"], gtzan.labels)]


def test_free_text_is_seeded_and_respects_max_items(model, gtzan, vector_bridge):
    a = free_text(model, vector_bridge, gtzan, max_items=12, seed=3, max_new_tokens=8)
    b = free_text(model, vector_bridge, gtzan, max_items=12, seed=3, max_new_tokens=8)
    assert a["n"] == 12 and a["result"]["n"] == 12 and a["max_new_tokens"] == 8 and a["seed"] == 3
    assert (a["result"], a["controls"], a["examples"]) == (b["result"], b["controls"], b["examples"])
    assert all(len(ex["reply"].split()) <= 8 for ex in a["examples"])


def test_free_text_through_a_text_bridge(model):
    esc = datasets.load_labelled_set("esc50")
    result = free_text(model, bridges.make_text_bridge(model, esc), esc, max_items=40)
    assert result["kind"] == "text" and result["n"] == 40
    assert result["result"]["names_true_only"] >= 0.8  # multi-word labels such as "sea waves" are matched
    assert result["controls"]["no_input"]["names_none"] >= 0.8


def test_free_text_refuses_a_bridge_built_for_another_model(model, gtzan, vector_bridge):
    vector_bridge.model_name = "another-model"
    with pytest.raises(ValueError, match="built for another-model"):
        free_text(model, vector_bridge, gtzan)


# ----------------------------------------------------------------------------- scripts/score_free_text.py
def test_score_free_text_writes_results_with_build_and_machine(tmp_path, capsys):
    out = tmp_path / "results" / "free_text.json"
    script = load_script("score_free_text")
    assert script.main(["--model", "fake", "--dataset", "gtzan", "--max-items", "20", "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "names true only" in printed and "swapped input" in printed and str(out) in printed

    record = json.loads(out.read_text())
    assert record["bridge"] == "vector:gtzan" and record["n"] == 20
    assert record["result"]["names_true_only"] >= 0.8
    build = record["build"]
    assert {"seconds", "fit_seconds", "layer", "dev_accuracy"} <= set(build) and build["installed"] is False
    assert build["seconds"] >= 0 and build["layer"] in range(5) and build["dev_accuracy"]
    assert set(record["machine"]) == {"platform", "python", "machine"}

    text_out = tmp_path / "text.json"
    assert script.main(["--model", "fake", "--bridge", "text", "--max-items", "10", "--out", str(text_out)]) == 0
    text = json.loads(text_out.read_text())
    assert text["kind"] == "text" and text["build"]["layer"] is None and text["build"]["dev_accuracy"] is None


def test_score_free_text_default_out_names_model_dataset_and_bridge():
    script = load_script("score_free_text")
    path = script.default_out("qwen3.5-0.8b", "gtzan", "vector")
    assert path == config.REPO / "socket_pilot" / "artifacts" / "free_text_qwen3.5-0.8b_gtzan_vector.json"


def test_score_free_text_uses_the_installed_sense(model, gtzan, vector_bridge, tmp_path, capsys):
    script = load_script("score_free_text")
    out = tmp_path / "installed.json"
    assert script.main(["--model", "fake", "--installed", "--out", str(out)]) == 1
    assert "No audio sense is installed for fake" in capsys.readouterr().err
    assert not out.exists()

    registry.install(vector_bridge, verify(model, vector_bridge, gtzan, FitRequest(model="fake", dataset="gtzan")))
    assert script.main(["--model", "fake", "--installed", "--dataset", "esc50", "--out", str(out)]) == 1
    assert "was verified on gtzan, not esc50" in capsys.readouterr().err
    assert script.main(["--model", "fake", "--installed", "--bridge", "text", "--out", str(out)]) == 1
    assert "not a text bridge" in capsys.readouterr().err
    assert not out.exists()

    assert script.main(["--model", "fake", "--installed", "--max-items", "10", "--out", str(out)]) == 0
    record = json.loads(out.read_text())
    assert record["bridge"] == "vector:gtzan" and record["build"]["installed"] is True
    assert record["build"]["layer"] == vector_bridge.layer


def test_score_free_text_fails_clearly_for_an_unknown_model(tmp_path, capsys):
    out = tmp_path / "x.json"
    assert load_script("score_free_text").main(["--model", "qwen3.5-0.8b", "--out", str(out)]) == 1
    assert "only model is 'fake'" in capsys.readouterr().err and not out.exists()


# ----------------------------------------------------------------------------- scripts/demo_cli.py --save
def run_demo_cli(monkeypatch, *args: str) -> int:
    monkeypatch.setattr(sys, "argv", ["demo_cli.py", *args])
    return load_script("demo_cli").main()


def test_demo_cli_save_writes_a_run_record_shaped_like_a_job(monkeypatch, tmp_path, capsys):
    path = tmp_path / "runs" / "closed.json"
    assert run_demo_cli(monkeypatch, "--model", "fake", "--dataset", "gtzan", "--scope", "closed", "--save", str(path)) == 0
    assert "Saved the run record (done" in capsys.readouterr().out

    record = json.loads(path.read_text())
    job_keys = set(JobState(id="x", request=FitRequest(model="fake", dataset="gtzan")).to_dict())
    assert job_keys == {"id", "request", "status", "events", "cards", "installed", "error"}
    assert job_keys | {"origin", "recorded", "seconds", "machine"} == set(record)
    assert len(record["id"]) == 12 and int(record["id"], 16) >= 0
    assert record["origin"] == "cli" and record["recorded"].endswith("+00:00")
    assert record["request"] == FitRequest(model="fake", dataset="gtzan", scope="closed").to_dict()
    assert (record["status"], record["installed"], record["error"]) == ("done", "vector:gtzan", None)
    assert {c["bridge"] for c in record["cards"]} == {"vector:gtzan", "text:gtzan"}
    assert record["events"] and all("t" in e and e["t"] >= 0 for e in record["events"])
    ts = [e["t"] for e in record["events"]]
    assert ts == sorted(ts) and record["seconds"] >= 0
    assert [e["step"] for e in record["events"]][-1] == "done"
    assert set(record["machine"]) == {"platform", "python", "machine"}


def test_demo_cli_save_records_a_failed_run(monkeypatch, tmp_path, capsys):
    path = tmp_path / "failed.json"
    assert run_demo_cli(monkeypatch, "--model", "fake", "--dataset", "nope", "--save", str(path)) == 1
    record = json.loads(path.read_text())
    assert record["status"] == "failed" and record["installed"] is None and record["cards"] == []
    assert "Unknown dataset 'nope'" in record["error"]
    assert record["events"][-1]["step"] == "error" and all("t" in e for e in record["events"])


def test_demo_cli_save_to_an_unwritable_path_fails_clearly(monkeypatch, tmp_path, capsys):
    blocker = tmp_path / "a-file"
    blocker.write_text("")
    assert run_demo_cli(monkeypatch, "--model", "fake", "--dataset", "gtzan", "--save", str(blocker / "run.json")) == 1
    assert "Could not save the run record" in capsys.readouterr().err
