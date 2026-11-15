"""Replay mode in fake mode: recorded runs (sense.runs), the replay engine (server.replay) and their HTTP endpoints.

Run: SENSE_FAKE=1 .venv/bin/python -m pytest -q tests/test_replay.py
"""

from __future__ import annotations

import json
import logging
import threading
import time

import pytest
from fastapi.testclient import TestClient

from agent import jobs, loop
from agent import planner as planner_mod
from sense import config, runs
from sense.contracts import FitRequest, JobState, TraceEvent
from server import app as server_app
from server import replay


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("SENSE_FAKE", "1")
    monkeypatch.setenv("SENSE_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(jobs, "_jobs", {})
    monkeypatch.setattr(jobs, "_latest_id", None)
    monkeypatch.setattr(planner_mod, "_preferred", None)  # the rule planner, even if an LLM planner is registered
    monkeypatch.setattr(replay, "_replays", {})
    monkeypatch.setattr(replay, "_current_id", None)
    monkeypatch.setattr(server_app, "_recorder", None)  # no watcher left over from another test
    server_app._sample_paths.clear()
    assert config.fake()
    return TestClient(server_app.app)


def wait_for(predicate, seconds: float = 10.0):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.02)
    raise AssertionError(f"gave up after {seconds} s")


def finished(client: TestClient, job_id: str) -> dict:
    def poll():
        state = client.get(f"/api/jobs/{job_id}").json()
        return state if state["status"] != "running" else None

    state = wait_for(poll)
    json.dumps(state)  # everything the interface receives must be plain JSON
    return state


def run_job(client: TestClient, scope: str = "closed") -> dict:
    r = client.post("/api/fit", json={"model": "fake", "dataset": "gtzan", "modality": "audio", "scope": scope})
    assert r.status_code == 200, r.text
    return finished(client, r.json()["job_id"])


def recorded_listing(client: TestClient, run_id: str) -> dict:
    """The GET /api/runs entry of a run, once the server's watcher has recorded it."""
    return wait_for(lambda: next((r for r in client.get("/api/runs").json() if r["id"] == run_id), None))


def synthetic(run_id: str = "r1", last_t: float = 0.2, status: str = "done", error: str | None = None) -> JobState:
    return JobState(
        id=run_id,
        request=FitRequest(model="fake", dataset="gtzan"),
        status=status,
        events=[
            TraceEvent(step="inspect", status="start", message="Loading.", t=0.0),
            TraceEvent(step="done", status="ok", message="Nothing installed.", data={"installed": None}, t=last_t),
        ],
        error=error,
    )


def lines(events: list[dict]) -> list[tuple]:
    return [(e["step"], e["status"], e["message"]) for e in events]


# ----------------------------------------------------------------------------- end to end
def test_a_recorded_run_replays_after_a_restart_and_brings_its_sense_back(client, monkeypatch):
    state = run_job(client, "closed")
    assert state["status"] == "done" and state["installed"] == "vector:gtzan"
    listed = recorded_listing(client, state["id"])
    assert listed["origin"] == "server" and listed["has_sense"] is True
    assert listed["status"] == "done" and listed["installed"] == "vector:gtzan"
    assert listed["request"] == state["request"]
    assert listed["seconds"] == state["events"][-1]["t"]
    assert isinstance(listed["recorded"], str) and listed["recorded"].endswith("+00:00")

    # a server restart: the jobs are gone; then the sense is removed too
    monkeypatch.setattr(jobs, "_jobs", {})
    monkeypatch.setattr(jobs, "_latest_id", None)
    server_app._sample_paths.clear()
    assert client.get(f"/api/jobs/{state['id']}").status_code == 400
    assert client.delete("/api/senses", params={"model": "fake", "modality": "audio"}).json() == {"removed": True}
    assert client.get("/api/senses", params={"model": "fake"}).json() == []

    r = client.post("/api/replay", json={"run": state["id"], "speed": 1000})
    assert r.status_code == 200, r.text
    job_id = r.json()["job_id"]
    assert job_id.startswith("replay-") and len(job_id) == len("replay-") + 12
    replayed = finished(client, job_id)
    assert replayed["status"] == "done" and replayed["error"] is None
    assert replayed["replay"] == {"of": state["id"], "recorded": listed["recorded"], "speed": 1000.0}
    assert replayed["id"] == job_id and replayed["request"] == state["request"]
    assert lines(replayed["events"]) == lines(state["events"])
    assert [e["t"] for e in replayed["events"]] == [e["t"] for e in state["events"]]
    assert replayed["cards"] == state["cards"]
    assert replayed["installed"] == state["installed"]
    assert not replay.running()

    senses = client.get("/api/senses", params={"model": "fake"}).json()
    assert [s["bridge"] for s in senses] == ["vector:gtzan"]
    assert senses[0]["card"]["verdict"] == "installed"
    samples = client.get("/api/samples", params={"dataset": "gtzan"}).json()
    rock = next(s for s in samples if s["label"] == "rock")
    r = client.post("/api/chat", data={"model": "fake", "message": "Which genre?", "sample_ids": [rock["id"]]})
    assert r.status_code == 200, r.text
    assert "rock" in r.json()["reply"]
    assert r.json()["used"][0]["bridge"] == "vector:gtzan" and r.json()["used"][0]["kind"] == "vector"

    assert [run["id"] for run in client.get("/api/runs").json()] == [state["id"]]  # a replay is not recorded again


def test_replay_refuses_an_unknown_run_and_a_bad_speed(client):
    runs.save_run(synthetic("r1"))
    r = client.post("/api/replay", json={"run": "nope"})
    assert r.status_code == 400 and "No recorded run 'nope'" in r.json()["detail"]
    r = client.post("/api/replay", json={"run": "../home"})
    assert r.status_code == 400 and "not a valid run id" in r.json()["detail"]
    for speed in (0, 0.05, 5000, "fast", None):
        r = client.post("/api/replay", json={"run": "r1", "speed": speed})
        assert r.status_code == 400, speed
        assert "speed" in r.json()["detail"] and "0.1 to 1000" in r.json()["detail"]
    assert not replay.running()
    r = client.post("/api/replay", json={"run": "r1", "speed": "1000"})
    assert r.status_code == 200, r.text
    assert finished(client, r.json()["job_id"])["status"] == "done"


def test_replay_is_refused_while_an_agent_run_is_in_progress(client, monkeypatch):
    runs.save_run(synthetic("r1"))
    live = JobState(id="live", request=FitRequest(model="fake", dataset="gtzan"))
    monkeypatch.setattr(jobs, "latest", lambda: live)
    r = client.post("/api/replay", json={"run": "r1", "speed": 1000})
    assert r.status_code == 409
    assert r.json()["detail"] == "An agent run on fake is in progress. Wait for it to finish, then start the replay."
    live.status = "done"
    r = client.post("/api/replay", json={"run": "r1", "speed": 1000})
    assert r.status_code == 200, r.text
    finished(client, r.json()["job_id"])


def test_a_fit_and_a_second_replay_are_refused_while_a_replay_plays(client):
    runs.save_run(synthetic("r1", last_t=1.0))
    r = client.post("/api/replay", json={"run": "r1", "speed": 1})
    assert r.status_code == 200, r.text
    job_id = r.json()["job_id"]
    fit = client.post("/api/fit", json={"model": "fake", "dataset": "gtzan"})
    assert fit.status_code == 409
    assert fit.json()["detail"] == "A replay is in progress. Wait for it to finish, then start again."
    second = client.post("/api/replay", json={"run": "r1", "speed": 1000})
    assert second.status_code == 409 and "A replay is already in progress" in second.json()["detail"]
    playing = client.get(f"/api/jobs/{job_id}").json()
    assert playing["status"] == "running" and len(playing["events"]) < 2 and playing["replay"]["speed"] == 1.0

    assert finished(client, job_id)["status"] == "done"  # played in real time: about one second
    state = run_job(client)  # a live run may start again
    assert state["status"] == "done"
    recorded_listing(client, state["id"])  # let the watcher finish before the test's home goes away


def test_a_finished_run_is_recorded_before_the_next_fit_or_replay_starts(client):
    """The watcher records a run up to WATCH_SECONDS after it ends. A fit or a replay started in that gap must wait
    for the record, or it could change the registry first and the run would be recorded without its sense."""
    first = run_job(client)
    r = client.post("/api/fit", json={"model": "fake", "dataset": "gtzan"})  # as soon as `first` reports done
    assert r.status_code == 200, r.text
    assert [(x["id"], x["has_sense"]) for x in client.get("/api/runs").json()] == [(first["id"], True)]
    second = finished(client, r.json()["job_id"])
    assert second["installed"] == "vector:gtzan"
    r = client.post("/api/replay", json={"run": first["id"], "speed": 1000})  # as soon as `second` reports done
    assert r.status_code == 200, r.text
    assert {x["id"]: x["has_sense"] for x in client.get("/api/runs").json()} == {first["id"]: True, second["id"]: True}
    assert finished(client, r.json()["job_id"])["status"] == "done"


def test_a_fit_or_replay_is_refused_while_the_last_run_is_still_being_recorded(client, monkeypatch):
    runs.save_run(synthetic("r1"))
    release = threading.Event()
    save_run = runs.save_run

    def slow_save(state, **kwargs):
        release.wait(10)
        return save_run(state, **kwargs)

    monkeypatch.setattr(runs, "save_run", slow_save)
    monkeypatch.setattr(server_app, "RECORD_WAIT_SECONDS", 0.2)
    state = run_job(client)
    busy = "The last agent run is still being recorded. Try again in a moment."
    r = client.post("/api/replay", json={"run": "r1", "speed": 1000})
    assert r.status_code == 409 and r.json()["detail"] == busy
    r = client.post("/api/fit", json={"model": "fake", "dataset": "gtzan"})
    assert r.status_code == 409 and r.json()["detail"] == busy
    assert not replay.running()

    release.set()
    r = client.post("/api/replay", json={"run": "r1", "speed": 1000})
    assert r.status_code == 200, r.text
    assert next(x for x in client.get("/api/runs").json() if x["id"] == state["id"])["has_sense"] is True
    assert finished(client, r.json()["job_id"])["status"] == "done"


def test_a_replay_that_fails_ends_failed_and_does_not_block_the_server(client):
    state = run_job(client)
    assert recorded_listing(client, state["id"])["has_sense"] is True
    (runs.runs_dir() / state["id"] / "sense" / "bridge.json").write_text(json.dumps({"kind": "bogus"}))
    r = client.post("/api/replay", json={"run": state["id"], "speed": 1000})
    assert r.status_code == 200, r.text
    replayed = finished(client, r.json()["job_id"])
    assert replayed["status"] == "failed" and replayed["error"].startswith("Replay stopped:")
    assert (replayed["events"][-1]["step"], replayed["events"][-1]["status"]) == ("error", "fail")
    assert not replay.running()
    state = run_job(client)  # a live run may start again
    assert state["status"] == "done"
    recorded_listing(client, state["id"])  # let the watcher finish before the test's home goes away


def test_a_faster_replay_takes_less_time(client):
    runs.save_run(synthetic("slow", last_t=3.0))
    began = time.monotonic()
    r = client.post("/api/replay", json={"run": "slow", "speed": 1000})
    assert finished(client, r.json()["job_id"])["status"] == "done"
    assert time.monotonic() - began < 1.0  # 3 s / 1000, with room for a slow machine
    began = time.monotonic()
    r = client.post("/api/replay", json={"run": "slow", "speed": 10})
    assert finished(client, r.json()["job_id"])["status"] == "done"
    assert time.monotonic() - began >= 0.3  # 3 s / 10


def test_the_sense_is_back_when_the_install_event_shows(client):
    state = run_job(client)
    recorded_listing(client, state["id"])
    path = runs.runs_dir() / state["id"] / "run.json"
    record = json.loads(path.read_text())
    at = next(i for i, e in enumerate(record["events"]) if (e["step"], e["status"]) == ("install", "ok"))
    assert at < len(record["events"]) - 1
    for event in record["events"][at + 1:]:  # [done] a second after [install]
        event["t"] = record["events"][at]["t"] + 1.0
    path.write_text(json.dumps(record))
    assert client.delete("/api/senses", params={"model": "fake", "modality": "audio"}).json() == {"removed": True}

    r = client.post("/api/replay", json={"run": state["id"], "speed": 1})
    assert r.status_code == 200, r.text
    job_id = r.json()["job_id"]

    def installed():
        playing = client.get(f"/api/jobs/{job_id}").json()
        return playing if ("install", "ok") in [(e["step"], e["status"]) for e in playing["events"]] else None

    playing = wait_for(installed)
    assert playing["status"] == "running" and playing["events"][-1]["step"] == "install"
    assert [s["bridge"] for s in client.get("/api/senses", params={"model": "fake"}).json()] == ["vector:gtzan"]
    assert finished(client, job_id)["status"] == "done"


def test_a_failed_run_replays_as_failed_with_its_error(client):
    runs.save_run(synthetic("broken", status="failed", error="ValueError: boom"))
    r = client.post("/api/replay", json={"run": "broken", "speed": 1000})
    replayed = finished(client, r.json()["job_id"])
    assert replayed["status"] == "failed" and replayed["error"] == "ValueError: boom"
    assert lines(replayed["events"]) == lines(synthetic().to_dict()["events"])


# ----------------------------------------------------------------------------- records
def test_list_runs_skips_a_corrupt_record(client, caplog):
    runs.save_run(synthetic("good"))
    root = runs.runs_dir()
    (root / "bad").mkdir()
    (root / "bad" / "run.json").write_text("{not json")
    (root / "empty").mkdir()
    (root / "other").mkdir()
    (root / "other" / "run.json").write_text(json.dumps({"id": "someone-else"}))
    (root / "list").mkdir()
    (root / "list" / "run.json").write_text("[]")
    with caplog.at_level(logging.WARNING, logger="sense.runs"):
        listed = runs.list_runs()
    assert [r["id"] for r in listed] == ["good"]
    assert listed[0] | {"recorded": ""} == {
        "id": "good", "request": FitRequest(model="fake", dataset="gtzan").to_dict(), "status": "done",
        "installed": None, "recorded": "", "origin": "server", "seconds": 0.2, "has_sense": False,
    }
    skipped = " ".join(record.getMessage() for record in caplog.records)
    for name in ("bad", "empty", "other", "list"):
        assert f"/runs/{name}" in skipped
    assert client.get("/api/runs").json() == listed
    with pytest.raises(ValueError, match="cannot be read"):
        runs.load_run("bad")
    assert runs.restore_sense("good") is False


def cli_record(run_id: str, **extra) -> dict:
    """What scripts/demo_cli.py --save writes."""
    return {
        "id": run_id,
        "origin": "cli",
        "recorded": "2026-11-14T09:00:00+00:00",
        "request": FitRequest(model="fake", dataset="gtzan").to_dict(),
        "status": "done",
        "events": [e.to_dict() for e in synthetic().events],
        "cards": [],
        "installed": None,
        "error": None,
        "seconds": 12.5,
        "machine": "second-laptop",
        **extra,
    }


def test_import_record_keeps_a_cli_run_with_its_origin_and_time(client, tmp_path):
    path = tmp_path / "cli-run.json"
    path.write_text(json.dumps(cli_record("cli-run-1")))
    runs.save_run(synthetic("server-run"))  # recorded now, so listed first
    assert runs.import_record(path) == "cli-run-1"
    listed = client.get("/api/runs").json()
    assert [r["id"] for r in listed] == ["server-run", "cli-run-1"]
    cli = listed[1]
    assert cli["origin"] == "cli" and cli["recorded"] == "2026-11-14T09:00:00+00:00"
    assert cli["seconds"] == 12.5 and cli["has_sense"] is False and cli["status"] == "done"
    record = runs.load_run("cli-run-1")
    assert record["machine"] == "second-laptop" and record["format"] == 1

    path.write_text(json.dumps(cli_record("cli-run-2", status="running")))
    with pytest.raises(ValueError, match="only a finished run"):
        runs.import_record(path)
    path.write_text(json.dumps({"id": "cli-run-3"}))
    with pytest.raises(ValueError, match="lacks request"):
        runs.import_record(path)
    with pytest.raises(ValueError, match="does not exist"):
        runs.import_record(tmp_path / "missing.json")


def test_import_copies_the_matching_sense_and_a_replay_without_one_says_so(client, tmp_path):
    request = FitRequest(model="fake", dataset="gtzan")
    events: list[TraceEvent] = []
    cards, installed = loop.run_fit(request, events.append)  # what the CLI does
    assert installed == "vector:gtzan"
    record = cli_record(
        "cli-with-sense",
        events=[e.to_dict() for e in events],
        cards=[c.to_dict() for c in cards],
        installed=installed,
        seconds=events[-1].t,
    )
    path = tmp_path / "cli-run.json"
    path.write_text(json.dumps(record, default=float))
    runs.import_record(path)
    runs.import_record(path)  # importing the same run again replaces it
    path.write_text(json.dumps(record | {"id": "cli-no-sense"}, default=float))
    runs.import_record(path, with_sense=False)
    has_sense = {r["id"]: r["has_sense"] for r in runs.list_runs()}
    assert has_sense == {"cli-with-sense": True, "cli-no-sense": False}
    assert not [p.name for p in runs.runs_dir().iterdir() if p.name.startswith(".")]  # no staging left behind

    client.delete("/api/senses", params={"model": "fake", "modality": "audio"})
    r = client.post("/api/replay", json={"run": "cli-no-sense", "speed": 1000})
    replayed = finished(client, r.json()["job_id"])
    notes = [e for e in replayed["events"] if e["data"].get("replay")]
    assert len(notes) == 1 and notes[0]["step"] == "install" and notes[0]["status"] == "info"
    assert "not reinstalled" in notes[0]["message"]
    assert client.get("/api/senses", params={"model": "fake"}).json() == []

    r = client.post("/api/replay", json={"run": "cli-with-sense", "speed": 1000})
    replayed = finished(client, r.json()["job_id"])
    assert lines(replayed["events"]) == lines(record["events"])
    assert [s["bridge"] for s in client.get("/api/senses", params={"model": "fake"}).json()] == ["vector:gtzan"]


def test_save_run_checks_its_input_and_replaces_an_earlier_record(client):
    with pytest.raises(ValueError, match="not a valid run id"):
        runs.save_run(synthetic("../escape"))
    with pytest.raises(ValueError, match="only a finished run"):
        runs.save_run(synthetic("r1", status="running"))
    with pytest.raises(ValueError, match="Unknown run origin"):
        runs.save_run(synthetic("r1"), origin="web")
    assert runs.save_run(synthetic("r1").to_dict(), seconds=3.0) == "r1"  # a JobState-shaped dict works too
    assert runs.load_run("r1")["seconds"] == 3.0
    runs.save_run(synthetic("r1", status="failed", error="RuntimeError: late"))
    record = runs.load_run("r1")
    assert record["status"] == "failed" and record["error"] == "RuntimeError: late" and record["seconds"] == 0.2
    assert [p.name for p in runs.runs_dir().iterdir()] == ["r1"]
    with pytest.raises(ValueError, match="No recorded run"):
        runs.load_run("r2")
    with pytest.raises(ValueError, match="No recorded run"):
        runs.restore_sense("r2")


def test_a_sense_installed_by_another_run_is_not_recorded_as_this_runs(client, tmp_path):
    state = run_job(client, "closed")
    assert recorded_listing(client, state["id"])["has_sense"] is True
    card_path = tmp_path / "home" / "fake" / "audio" / "card.json"
    card = json.loads(card_path.read_text())
    card_path.write_text(json.dumps(card | {"created": "2000-01-01T00:00:00+00:00"}))  # as if a later run installed it
    runs.save_run(state)  # the JobState dict served over HTTP
    assert [r["has_sense"] for r in runs.list_runs()] == [False]


# ----------------------------------------------------------------------------- ReportCard.control_chance
OLD_CARD = {  # a card as recorded before ReportCard.control_chance existed
    "model": "fake", "modality": "audio", "dataset": "gtzan", "scope": "closed", "bridge": "text:gtzan",
    "kind": "text", "n": 20, "chance": 0.2, "accuracy": 0.6, "controls": {"swapped": 0.2, "no_input": 0.2},
    "verdict": "rejected", "reason": "r", "created": "2026-11-14T10:00:00+00:00", "interval": [0.39, 0.78],
}


def test_cards_recorded_without_control_chance_load_and_replay(client):
    raw = synthetic("old-cards").to_dict()
    raw["cards"] = [
        OLD_CARD,
        dict(OLD_CARD, bridge="adapter:a", control_chance=None),
        dict(OLD_CARD, bridge="adapter:b", controls={"swapped": 0.2, "unseen_categories": 0.5},
             control_chance={"unseen_categories": 0.3333}),
    ]
    runs.save_run(raw)
    _, _, cards = runs.parse_record(runs.load_run("old-cards"))
    assert [c.control_chance for c in cards] == [{}, {}, {"unseen_categories": 0.3333}]
    assert cards[1].control_chance.get("unseen_categories") is None  # every reader may call .get on it

    r = client.post("/api/replay", json={"run": "old-cards", "speed": 1000})
    replayed = finished(client, r.json()["job_id"])
    assert replayed["status"] == "done"
    assert [c["control_chance"] for c in replayed["cards"]] == [{}, {}, {"unseen_categories": 0.3333}]
