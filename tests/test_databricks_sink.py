"""integrations.databricks.sink without a workspace: rows from a run record and the SQL it sends."""

import pytest

from integrations.databricks import conn, sink

RECORD = {
    "id": "run1",
    "origin": "cli",
    "recorded": "2026-11-14T18:55:33+00:00",
    "request": {"model": "fake", "dataset": "gtzan", "modality": "audio", "scope": "closed"},
    "status": "done",
    "installed": "vector:gtzan",
    "error": None,
    "seconds": 0.3,
    "events": [
        {"step": "plan", "status": "ok", "message": "Bridges to try", "data": {"kinds": ["vector", "text"]}, "t": 0.026},
        {"step": "done", "status": "ok", "message": "done", "data": {}, "t": 0.3},
    ],
    "cards": [
        {"model": "fake", "modality": "audio", "dataset": "gtzan", "scope": "closed", "bridge": "vector:gtzan",
         "kind": "vector", "n": 60, "chance": 0.2, "accuracy": 1.0, "interval": [0.94, 1.0],
         "controls": {"swapped": 0.1, "no_input": 0.2333}, "verdict": "installed", "reason": "r", "created": "x"},
        {"model": "fake", "modality": "audio", "dataset": "gtzan", "scope": "closed", "bridge": "text:gtzan",
         "kind": "text", "n": 60, "chance": 0.2, "accuracy": 0.5, "interval": None,
         "controls": {}, "verdict": "rejected", "reason": "r2", "created": "x"},
    ],
}


class FakeSQL:
    def __init__(self):
        self.calls = []

    def run(self, statement, params=None):
        self.calls.append((statement, params or {}))
        return []


def test_card_rows_flatten_controls_and_mark_the_installed_bridge():
    rows = sink.card_rows(RECORD)
    assert [r["bridge"] for r in rows] == ["vector:gtzan", "text:gtzan"]
    assert rows[0]["installed"] is True and rows[1]["installed"] is False
    assert rows[0]["swapped"] == 0.1 and rows[0]["lookup"] is None
    assert rows[0]["interval_low"] == 0.94 and rows[1]["interval_low"] is None
    assert set(rows[0]) == {c.split()[0] for c in " ".join(sink.DDL["report_cards"].split()).split(", ")}


def test_run_and_event_rows_match_their_tables():
    for name, rows in (("runs", [sink.run_row(RECORD)]), ("trace_events", sink.event_rows(RECORD))):
        cols = {c.split()[0] for c in " ".join(sink.DDL[name].split()).split(", ")}
        assert all(set(r) == cols for r in rows)
    assert sink.event_rows(RECORD)[0]["data_json"] == '{"kinds": ["vector", "text"]}'


def test_write_run_replaces_the_run_rows_with_bound_parameters():
    sql = FakeSQL()
    counts = sink.write_run(RECORD, sql, upload=False, mlflow_runs=False)
    assert counts == {"runs": 1, "report_cards": 2, "trace_events": 2}
    statements = [s for s, _ in sql.calls]
    assert statements[0].startswith(f"DELETE FROM {conn.table('runs')} WHERE run_id = :run_id")
    inserts = [(s, p) for s, p in sql.calls if s.startswith("INSERT")]
    assert len(inserts) == 3
    s, p = inserts[1]
    assert "CAST(:recorded_0 AS TIMESTAMP)" in s and p["bridge_1"] == "text:gtzan"
    assert "CAST(:lookup_1 AS DOUBLE)" in s and p["lookup_1"] is None  # a NULL control keeps the column's type
    assert "vector:gtzan" not in s  # values travel as parameters, never inside the SQL text


def test_names_refuse_anything_but_identifiers(monkeypatch):
    monkeypatch.setenv("SENSE_DBX_SCHEMA", "x; DROP TABLE y")
    with pytest.raises(ValueError):
        conn.table("runs")


def test_record_from_state_matches_the_run_record_shape():
    from sense.contracts import FitRequest, JobState, ReportCard, TraceEvent

    state = JobState(
        id="j1", request=FitRequest(model="fake", dataset="gtzan"), status="done",
        events=[TraceEvent(step="done", status="ok", message="m", t=1.5)],
        cards=[ReportCard(model="fake", modality="audio", dataset="gtzan", scope="closed", bridge="vector:gtzan",
                          kind="vector", n=10, chance=0.2, accuracy=0.9, controls={"swapped": 0.1})],
        installed="vector:gtzan",
    )
    record = sink.record_from_state(state)
    assert record["seconds"] == 1.5 and record["request"]["dataset"] == "gtzan"
    assert sink.card_rows(record)[0]["installed"] is True


def test_listener_writes_a_finished_fake_job(monkeypatch):
    import threading

    from agent import jobs
    from sense.contracts import FitRequest

    monkeypatch.setenv("SENSE_FAKE", "1")
    sql = FakeSQL()
    done = threading.Event()
    listener = sink.Listener(sql, upload=False, mlflow_runs=False)
    original = listener._write
    monkeypatch.setattr(listener, "_write", lambda job_id: (original(job_id), done.set()))
    jobs.add_listener(listener)
    try:
        job_id = jobs.start(FitRequest(model="fake", dataset="gtzan"))
        assert done.wait(60)
    finally:
        jobs.remove_listener(listener)
    inserts = [p for s, p in sql.calls if s.startswith(f"INSERT INTO {conn.table('report_cards')}")]
    assert inserts and inserts[0]["run_id_0"] == job_id
    assert any(p.get("installed_0") or p.get("installed_1") for p in inserts)
