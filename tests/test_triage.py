"""eval.triage in fake mode, and its Databricks rows without a workspace.

Run: SENSE_FAKE=1 .venv/bin/python -m pytest -q tests/test_triage.py
"""

from __future__ import annotations

import json

import pytest

from agent import loop
from agent import planner as planner_mod
from eval import triage
from integrations.databricks import sink
from sense.contracts import FitRequest
from sense.specialists import FAKE_CONCEPTS, FakeSpecialist


@pytest.fixture
def installed(monkeypatch, tmp_path):
    monkeypatch.setenv("SENSE_FAKE", "1")
    monkeypatch.setenv("SENSE_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(planner_mod, "_preferred", None)
    _, name = loop.run_fit(FitRequest(model="fake", dataset="esc50", scope="open"), lambda e: None)
    assert name
    return name


def columns(name):
    return {c.split()[0] for c in " ".join(sink.DDL[name].split()).split(", ")}


def test_triage_flags_clips_and_scores_against_hidden_labels(installed):
    record = triage.run()
    assert record["sense"]["bridge"] == installed
    assert record["batch"] == {"source": "esc50 fold 5", "n": len(record["clips"]), "has_truth": True}
    for c in record["clips"]:
        assert c["sense"] in (*triage.TARGETS, triage.NONE)
        assert c["flagged"] == (c["sense"] != triage.NONE)
    for method in triage.METHODS:
        s = record["summary"][method]
        assert s["flagged"] == sum(c[method] != triage.NONE for c in record["clips"])
        assert s["threats"] == sum(c["truth"] != triage.NONE for c in record["clips"]) > 0
        assert 0 <= s["caught"] <= s["threats"]
    # The lookup control answers from the labelled fit set; on the separable fake world it is exact.
    assert record["summary"]["lookup"]["recall"] == 1.0
    assert record["summary"]["lookup"]["false_alarm_rate"] == 0.0
    assert "recall" in triage.table(record) and "Flagged for a ranger" in triage.table(record)


def test_same_seed_same_record(installed):
    a, b = triage.run(), triage.run()
    assert [c["sense"] for c in a["clips"]] == [c["sense"] for c in b["clips"]]
    assert a["summary"] == b["summary"]


def test_limit_keeps_every_target_clip(installed):
    full = triage.run()
    small = triage.run(limit=10)
    assert small["batch"]["n"] == 10
    assert small["summary"]["sense"]["threats"] == full["summary"]["sense"]["threats"]


def test_folder_batch_has_flags_but_no_scores(installed, tmp_path):
    folder = tmp_path / "field"
    folder.mkdir()
    for k, concept in enumerate(["chainsaw", "dog", "rain"]):
        FakeSpecialist.write_sample(FAKE_CONCEPTS.index(concept), folder / f"clip{k}.wav", seed=k)
    record = triage.run(folder=folder)
    assert record["batch"]["has_truth"] is False and record["batch"]["n"] == 3
    assert "recall" not in record["summary"]["sense"]
    assert record["clips"][0]["lookup"] == "chainsaw" and record["clips"][1]["lookup"] == triage.NONE


def test_no_sense_installed_says_how_to_get_one(monkeypatch, tmp_path):
    monkeypatch.setenv("SENSE_FAKE", "1")
    monkeypatch.setenv("SENSE_HOME", str(tmp_path / "empty"))
    with pytest.raises(RuntimeError, match="no audio sense installed"):
        triage.run()


class FakeSQL:
    def __init__(self):
        self.calls = []

    def run(self, statement, params=None):
        self.calls.append((statement, params or {}))
        return []


def test_triage_rows_match_their_tables_and_replace_the_batch(installed):
    record = triage.run()
    batches, clips = sink.triage_rows(record)
    assert [b["method"] for b in batches] == list(triage.METHODS)
    assert all(set(b) == columns("triage_batches") for b in batches)
    assert all(set(c) == columns("triage_clips") for c in clips) and len(clips) == record["batch"]["n"]
    sql = FakeSQL()
    counts = sink.write_triage(record, sql, mlflow_runs=False)
    assert counts == {"triage_batches": 3, "triage_clips": len(clips)}
    statements = [s for s, _ in sql.calls]
    assert statements[0].startswith("DELETE FROM workspace.sensorium.triage_batches WHERE batch_id = :batch_id")
    assert sum(s.startswith("INSERT INTO workspace.sensorium.triage_clips") for s in statements) == -(-len(clips) // 50)


@pytest.mark.parametrize("question", triage.QUESTIONS)
def test_every_question_gives_targets_or_none_on_the_dev_fold(installed, question):
    record = triage.run(question=question, fold="dev")
    assert record["question"] == question and record["batch"]["source"] == "esc50 fold 4 (dev)"
    assert all(c[m] in (*triage.TARGETS, triage.NONE) for c in record["clips"] for m in triage.METHODS)
    assert ("reply" in record["clips"][0]) == (question == "chat")
    batches, clips = sink.triage_rows(record)
    assert {b["question"] for b in batches} == {question}
    assert all(set(c) == columns("triage_clips") for c in clips)


def test_ranger_question_reproduces_the_first_record_layout(installed):
    """The original four-way question keeps "none of these" last, as in the 0.8B record of PR #56."""
    assert triage._options(0, 3)[-1] == triage.NONE
    assert sorted(triage._options(0, 3, shuffle_none=True)) == sorted([*triage.TARGETS, triage.NONE])


def test_named_target_reads_variants_and_takes_the_first():
    assert triage._named_target("A chain saw cutting wood") == "chainsaw"
    assert triage._named_target("Engines idling, then a handsaw") == "engine"
    assert triage._named_target("Birds chirping in a forest") == triage.NONE
    assert triage._named_target("sawing") == triage.NONE


def test_compare_lists_each_question(installed):
    records = [triage.run(question=q, fold="dev", limit=20) for q in ("neutral", "yesno")]
    out = triage.compare(records)
    assert "neutral" in out and "yesno" in out and "precision" in out


def test_compare_names_model_and_batch_when_records_differ(installed):
    dev = triage.run(question="neutral", fold="dev", limit=20)
    test = triage.run(question="neutral", fold="test", limit=30)
    other = dict(dev, model="other-model")
    out = triage.compare([dev, test, other]).splitlines()
    assert "more than one model or batch" in out[0]
    assert out[1].split()[:4] == ["model", "source", "clips", "lookup"]
    assert "fake" in out[2] and "esc50 fold 4 (dev)" in out[2] and " 20 " in out[2]
    assert "esc50 fold 5" in out[3] and " 30 " in out[3]
    assert out[4].startswith("other-model")
    same = triage.compare([dev, triage.run(question="yesno", fold="dev", limit=20)])
    assert same.startswith("fake: 20 clips from esc50 fold 4 (dev)") and "model" not in same.splitlines()[1]


def test_check_record_takes_a_saved_record_and_refuses_others(installed):
    triage.check_record(triage.run(limit=10))
    with pytest.raises(ValueError, match="lacks .*sense.*batch.source"):
        triage.check_record({"id": "x", "recorded": "", "task": "", "user": "", "model": "", "note": "", "batch": {},
                             "summary": {}, "clips": [], "seconds": 0})
    with pytest.raises(ValueError, match="not a JSON object"):
        triage.check_record([])


def test_main_refuses_save_with_every_question(installed, tmp_path, capsys):
    with pytest.raises(SystemExit) as stop:
        triage.main(["--question", "all", "--fold", "dev", "--limit", "20", "--save", str(tmp_path / "x.json")])
    assert stop.value.code == 2 and "--save writes one record" in capsys.readouterr().err
    assert not (tmp_path / "home" / "triage").exists()  # nothing ran


def test_main_without_a_sense_says_so_without_a_traceback(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("SENSE_FAKE", "1")
    monkeypatch.setenv("SENSE_HOME", str(tmp_path / "empty"))
    assert triage.main(["--limit", "10"]) == 1
    assert "Triage failed: fake has no audio sense installed" in capsys.readouterr().err
    assert triage.main(["--folder", str(tmp_path / "nowhere")]) == 1


def test_main_saves_the_record_where_asked(installed, tmp_path, capsys):
    path = tmp_path / "out" / "one.json"
    assert triage.main(["--fold", "dev", "--limit", "20", "--save", str(path)]) == 0
    triage.check_record(json.loads(path.read_text()))


def test_databricks_triage_runs_saves_and_writes_a_new_batch(installed, monkeypatch, capsys):
    from integrations.databricks import __main__ as cli
    from integrations.databricks import conn

    sql = FakeSQL()
    monkeypatch.setattr(conn, "warehouse", lambda: sql)
    assert cli.main(["triage", "--no-mlflow", "--fold", "dev", "--limit", "20"]) == 0
    out = capsys.readouterr().out
    saved = out.split("Saved ")[1].splitlines()[0]
    record = json.loads(open(saved).read())
    assert f"Wrote triage {record['id']}" in out
    assert any(p.get("batch_id_0") == record["id"] for s, p in sql.calls if s.startswith("INSERT INTO"))


def test_databricks_triage_says_how_to_retry_when_the_write_fails(installed, monkeypatch, capsys):
    from integrations.databricks import __main__ as cli
    from integrations.databricks import conn

    class Down(FakeSQL):
        def run(self, statement, params=None):
            if statement.startswith("INSERT"):
                raise RuntimeError("Databricks SQL failed (warehouse stopped)")
            return super().run(statement, params)

    monkeypatch.setattr(conn, "warehouse", Down)
    assert cli.main(["triage", "--no-mlflow", "--fold", "dev", "--limit", "20"]) == 1
    err = capsys.readouterr().err
    assert "Could not write triage" in err and "warehouse stopped" in err and "--from " in err


def test_a_record_from_before_the_variants_reads_as_ranger(installed):
    record = triage.run(question="ranger", limit=10)
    record["question"] = triage.RANGER  # what the first 0.8B record (PR #56) holds
    assert {b["question"] for b in sink.triage_rows(record)[0]} == {"ranger"}
    assert "question 'ranger'" in triage.table(record)
