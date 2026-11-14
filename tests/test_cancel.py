"""Cancelling a replay or a live agent run in fake mode: server.replay.cancel and DELETE /api/jobs/<job_id>.

Run: SENSE_FAKE=1 .venv/bin/python -m pytest -q tests/test_cancel.py
"""

from __future__ import annotations

import json
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

CANCELLED = "The replay was cancelled before it finished."
NOT_YET = "Cancelling a running agent run is not supported yet: wait for it to finish."


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("SENSE_FAKE", "1")
    monkeypatch.setenv("SENSE_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(jobs, "_jobs", {})
    monkeypatch.setattr(jobs, "_latest_id", None)
    monkeypatch.setattr(planner_mod, "_preferred", None)  # the rule planner, even if an LLM planner is registered
    monkeypatch.setattr(replay, "_replays", {})
    monkeypatch.setattr(replay, "_current_id", None)
    monkeypatch.setattr(replay, "_workers", {})
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


def run_job(client: TestClient) -> dict:
    r = client.post("/api/fit", json={"model": "fake", "dataset": "gtzan", "modality": "audio", "scope": "closed"})
    assert r.status_code == 200, r.text
    return finished(client, r.json()["job_id"])


def recorded_listing(client: TestClient, run_id: str) -> dict:
    """The GET /api/runs entry of a run, once the server's watcher has recorded it."""
    return wait_for(lambda: next((r for r in client.get("/api/runs").json() if r["id"] == run_id), None))


def synthetic(run_id: str = "r1", last_t: float = 0.2) -> JobState:
    return JobState(
        id=run_id,
        request=FitRequest(model="fake", dataset="gtzan"),
        status="done",
        events=[
            TraceEvent(step="inspect", status="start", message="Loading.", t=0.0),
            TraceEvent(step="done", status="ok", message="Nothing installed.", data={"installed": None}, t=last_t),
        ],
    )


def lines(events: list[dict]) -> list[tuple]:
    return [(e["step"], e["status"], e["message"]) for e in events]


def playing_with(client: TestClient, job_id: str, n_events: int) -> dict:
    """The replay's state once it shows n_events events and is still running."""
    def poll():
        state = client.get(f"/api/jobs/{job_id}").json()
        return state if len(state["events"]) >= n_events else None

    state = wait_for(poll)
    assert state["status"] == "running", state
    return state


# ----------------------------------------------------------------------------- replays
def test_a_cancelled_replay_stops_at_once_ends_failed_and_frees_the_server(client):
    runs.save_run(synthetic("slow", last_t=60.0))  # the second event is a minute away at speed 1
    r = client.post("/api/replay", json={"run": "slow", "speed": 1})
    assert r.status_code == 200, r.text
    job_id = r.json()["job_id"]
    playing_with(client, job_id, 1)
    assert replay.running()

    began = time.monotonic()
    assert client.delete(f"/api/jobs/{job_id}").json() == {"cancelled": True}
    assert time.monotonic() - began < 2.0  # the wait for the next event was cut short
    assert not replay.running()  # already when DELETE returns

    state = client.get(f"/api/jobs/{job_id}").json()
    assert state["status"] == "failed" and state["error"] == "Cancelled."
    assert state["cards"] == [] and state["installed"] is None
    assert lines(state["events"]) == [("inspect", "start", "Loading."), ("error", "fail", CANCELLED)]
    last = state["events"][-1]
    assert last["data"] == {"replay": True, "cancelled": True}
    assert 0 <= last["t"] < 60.0
    assert state["replay"]["of"] == "slow" and state["replay"]["speed"] == 1.0

    assert client.delete(f"/api/jobs/{job_id}").json() == {"cancelled": False}  # already over
    assert replay.cancel(job_id) is False
    assert client.get(f"/api/jobs/{job_id}").json() == state  # a second cancel changes nothing

    r = client.post("/api/replay", json={"run": "slow", "speed": 1000})  # a new replay may start
    assert r.status_code == 200, r.text
    assert finished(client, r.json()["job_id"])["status"] == "done"


def test_cancelling_a_replay_frees_the_server_for_a_fit(client):
    runs.save_run(synthetic("slow", last_t=60.0))
    job_id = client.post("/api/replay", json={"run": "slow", "speed": 1}).json()["job_id"]
    playing_with(client, job_id, 1)
    assert client.post("/api/fit", json={"model": "fake", "dataset": "gtzan"}).status_code == 409
    assert client.delete(f"/api/jobs/{job_id}").json() == {"cancelled": True}
    state = run_job(client)  # a live run may start straight away
    assert state["status"] == "done"
    recorded_listing(client, state["id"])  # let the watcher finish before the test's home goes away
    assert {r["id"] for r in client.get("/api/runs").json()} == {state["id"], "slow"}  # the replay is not recorded


def test_a_sense_reinstalled_before_the_cancel_stays_installed(client):
    state = run_job(client)
    recorded_listing(client, state["id"])
    path = runs.runs_dir() / state["id"] / "run.json"
    record = json.loads(path.read_text())
    at = next(i for i, e in enumerate(record["events"]) if (e["step"], e["status"]) == ("install", "ok"))
    assert at < len(record["events"]) - 1
    for event in record["events"][at + 1:]:  # the events after [install] a minute later
        event["t"] = record["events"][at]["t"] + 60.0
    path.write_text(json.dumps(record))
    assert client.delete("/api/senses", params={"model": "fake", "modality": "audio"}).json() == {"removed": True}

    job_id = client.post("/api/replay", json={"run": state["id"], "speed": 1}).json()["job_id"]
    playing = playing_with(client, job_id, at + 1)
    assert (playing["events"][-1]["step"], playing["events"][-1]["status"]) == ("install", "ok")
    assert client.delete(f"/api/jobs/{job_id}").json() == {"cancelled": True}

    cancelled = client.get(f"/api/jobs/{job_id}").json()
    assert cancelled["status"] == "failed" and cancelled["error"] == "Cancelled."
    assert cancelled["cards"] == [] and cancelled["installed"] is None  # only set when a replay reaches its end
    assert lines(cancelled["events"]) == lines(record["events"][: at + 1]) + [("error", "fail", CANCELLED)]
    assert [s["bridge"] for s in client.get("/api/senses", params={"model": "fake"}).json()] == ["vector:gtzan"]


def test_cancelling_a_finished_or_unknown_replay_returns_false(client):
    runs.save_run(synthetic("r1"))
    job_id = client.post("/api/replay", json={"run": "r1", "speed": 1000}).json()["job_id"]
    state = finished(client, job_id)
    assert state["status"] == "done" and state["error"] is None
    assert client.delete(f"/api/jobs/{job_id}").json() == {"cancelled": False}
    assert client.get(f"/api/jobs/{job_id}").json() == state  # untouched
    assert replay.cancel(job_id) is False
    assert replay.cancel("replay-000000000000") is False
    assert replay.cancel("nope") is False


def test_cancels_spread_around_a_replay_s_end_agree_with_how_it_ended(client):
    """A smoke test: the window around the end is too narrow to hit on purpose, so the tests that hold the worker
    there (test_a_cancel_during_the_last_reinstall_is_applied_at_the_end and
    test_a_cancel_after_the_replay_ended_but_before_its_worker_exits_is_refused) are what guard it."""
    runs.save_run(synthetic("quick", last_t=0.05))
    for i in range(20):  # cancels from just after the start to well after the end, around the last event at 0.05 s
        job_id = client.post("/api/replay", json={"run": "quick", "speed": 1}).json()["job_id"]
        time.sleep(0.005 * i)
        cancelled = client.delete(f"/api/jobs/{job_id}").json()["cancelled"]
        state = finished(client, job_id)
        if cancelled:
            assert state["status"] == "failed" and state["error"] == "Cancelled."
            assert state["events"][-1]["message"] == CANCELLED
        else:
            assert state["status"] == "done" and state["error"] is None
            assert CANCELLED not in [e["message"] for e in state["events"]]
        assert not replay.running()


class HeldPop(dict):
    """A stand-in for replay._workers whose pop, the worker's very last step, blocks until `gate` is set."""

    def __init__(self):
        super().__init__()
        self.entered = threading.Event()
        self.gate = threading.Event()

    def pop(self, *args):
        self.entered.set()
        assert self.gate.wait(10)
        return super().pop(*args)


def test_a_cancel_after_the_replay_ended_but_before_its_worker_exits_is_refused(client, monkeypatch):
    """The worker sets the end status, then drops itself from _workers. A cancel in between still finds the worker,
    so only the status check in cancel refuses it: cancel must not answer True for a replay that ends "done"."""
    workers = HeldPop()
    monkeypatch.setattr(replay, "_workers", workers)
    runs.save_run(synthetic("r1"))
    job_id = replay.start("r1", speed=1000)
    answer: dict = {}
    try:
        assert workers.entered.wait(10)
        assert job_id in workers and replay.get(job_id)[0].status == "done"
        canceller = threading.Thread(target=lambda: answer.update(cancelled=replay.cancel(job_id)))
        canceller.start()
        canceller.join(2.0)  # a refused cancel returns at once; an accepted one would wait for the held worker
    finally:
        workers.gate.set()
    canceller.join(10)
    assert answer == {"cancelled": False}
    wait_for(lambda: job_id not in workers)
    state = client.get(f"/api/jobs/{job_id}").json()
    assert state["status"] == "done" and state["error"] is None
    assert CANCELLED not in [e["message"] for e in state["events"]]


def test_cancel_returns_once_the_replay_has_stopped(client):
    runs.save_run(synthetic("slow", last_t=60.0))
    job_id = replay.start("slow", speed=1)
    playing_with(client, job_id, 1)
    assert replay.cancel(job_id) is True
    assert not replay.running()  # at once, in-process: cancel waited for the worker
    assert replay.get(job_id)[0].status == "failed"


def ends_with_install(run_id: str) -> None:
    """Save a record whose last event is [install] "ok", so the replay's end comes straight after the reinstall."""
    runs.save_run(JobState(
        id=run_id,
        request=FitRequest(model="fake", dataset="gtzan"),
        status="done",
        events=[
            TraceEvent(step="inspect", status="start", message="Loading.", t=0.0),
            TraceEvent(step="install", status="ok", message="Installed.", t=0.0),
        ],
    ))


class BlockedRestore:
    """A stand-in for runs.restore_sense that blocks until `gate` is set, then returns True or raises `error`."""

    def __init__(self, error: Exception | None = None):
        self.entered = threading.Event()
        self.gate = threading.Event()
        self.error = error

    def __call__(self, run_id: str) -> bool:
        self.entered.set()
        assert self.gate.wait(10)
        if self.error is not None:
            raise self.error
        return True


def cancel_during_restore(job_id: str, restore: BlockedRestore) -> tuple[bool, bool]:
    """Cancel while the replay is blocked in the reinstall, then let the reinstall finish. Returns what cancel
    returned and whether replay.running() was still True the moment it returned."""
    assert restore.entered.wait(10)
    answer: dict = {}

    def call():
        answer["cancelled"] = replay.cancel(job_id)
        answer["running"] = replay.running()

    canceller = threading.Thread(target=call)
    canceller.start()
    wait_for(lambda: job_id in replay._workers and replay._workers[job_id][0].is_set())  # the cancel flag is up
    time.sleep(0.05)
    assert canceller.is_alive()  # cancel waits for the worker, which is still reinstalling
    restore.gate.set()
    canceller.join(10)
    assert not canceller.is_alive()
    return answer["cancelled"], answer["running"]


def test_a_cancel_during_the_last_reinstall_is_applied_at_the_end(client, monkeypatch):
    """The cancel lands after the last wait: only the check at the end can apply it, so the replay must not end
    "done" once cancel has answered True."""
    ends_with_install("inst")
    restore = BlockedRestore()
    monkeypatch.setattr(runs, "restore_sense", restore)
    job_id = replay.start("inst", speed=1000)
    assert cancel_during_restore(job_id, restore) == (True, False)

    state = client.get(f"/api/jobs/{job_id}").json()
    assert state["status"] == "failed" and state["error"] == "Cancelled."
    assert state["cards"] == [] and state["installed"] is None
    assert lines(state["events"]) == [
        ("inspect", "start", "Loading."), ("install", "ok", "Installed."), ("error", "fail", CANCELLED)
    ]


def test_a_replay_that_fails_after_a_cancel_still_ends_cancelled(client, monkeypatch):
    ends_with_install("inst")
    restore = BlockedRestore(OSError("No space left on device"))
    monkeypatch.setattr(runs, "restore_sense", restore)
    job_id = replay.start("inst", speed=1000)
    assert cancel_during_restore(job_id, restore) == (True, False)

    state = client.get(f"/api/jobs/{job_id}").json()
    assert state["status"] == "failed" and state["error"] == "Cancelled."
    assert lines(state["events"])[0] == ("inspect", "start", "Loading.")
    failure, last = state["events"][-2:]
    assert (failure["step"], failure["status"]) == ("error", "fail")
    assert "OSError: No space left on device" in failure["message"] and failure["data"] == {"replay": True}
    assert lines([last]) == [("error", "fail", CANCELLED)] and last["data"] == {"replay": True, "cancelled": True}

    restore = BlockedRestore(OSError("No space left on device"))  # the same failure without a cancel
    monkeypatch.setattr(runs, "restore_sense", restore)
    job_id = replay.start("inst", speed=1000)
    restore.gate.set()
    state = finished(client, job_id)
    assert state["status"] == "failed" and state["error"] == "Replay stopped: OSError: No space left on device"
    assert CANCELLED not in [e["message"] for e in state["events"]]


def test_a_reinstall_that_hangs_does_not_hang_the_cancel(client, monkeypatch):
    """cancel waits for the worker only up to CANCEL_WAIT_SECONDS: then DELETE answers true while the replay still
    runs, and the replay ends cancelled once the reinstall returns."""
    monkeypatch.setattr(replay, "CANCEL_WAIT_SECONDS", 0.2)
    ends_with_install("inst")
    restore = BlockedRestore()
    monkeypatch.setattr(runs, "restore_sense", restore)
    job_id = replay.start("inst", speed=1000)
    answer: dict = {}

    def call():
        began = time.monotonic()
        answer["response"] = client.delete(f"/api/jobs/{job_id}").json()
        answer["took"] = time.monotonic() - began

    try:
        assert restore.entered.wait(10)
        canceller = threading.Thread(target=call)
        canceller.start()
        canceller.join(2.0)
        assert not canceller.is_alive(), "DELETE kept waiting for a reinstall that does not return"
        assert answer["response"] == {"cancelled": True} and 0.1 <= answer["took"] < 2.0
        assert replay.running()  # still reinstalling, so a fit must wait
        assert client.post("/api/fit", json={"model": "fake", "dataset": "gtzan"}).status_code == 409
    finally:
        restore.gate.set()
    canceller.join(10)

    state = finished(client, job_id)
    assert state["status"] == "failed" and state["error"] == "Cancelled."
    assert state["events"][-1]["message"] == CANCELLED
    assert not replay.running()


# ----------------------------------------------------------------------------- the endpoint
def test_delete_on_an_unknown_job_is_a_400(client):
    for job_id in ("missing", "replay-000000000000"):
        r = client.delete(f"/api/jobs/{job_id}")
        assert r.status_code == 400
        assert r.json()["detail"].startswith(f"Unknown job '{job_id}': there is nothing to cancel.")
    assert client.get("/api/jobs/missing").status_code == 400  # as GET reports it


def live_job(monkeypatch, status: str = "running") -> JobState:
    job = JobState(id="live", request=FitRequest(model="fake", dataset="gtzan"), status=status)
    get = jobs.get
    monkeypatch.setattr(jobs, "get", lambda job_id: job if job_id == "live" else get(job_id))
    return job


def test_a_running_agent_run_cannot_be_cancelled_before_agent_jobs_supports_it(client, monkeypatch):
    monkeypatch.delattr(jobs, "cancel", raising=False)  # agent.jobs as it was before jobs.cancel
    live_job(monkeypatch)
    r = client.delete("/api/jobs/live")
    assert r.status_code == 409 and r.json()["detail"] == NOT_YET
    monkeypatch.setattr(jobs, "cancel", "not a function", raising=False)
    assert client.delete("/api/jobs/live").status_code == 409


def test_a_running_agent_run_is_cancelled_through_agent_jobs(client, monkeypatch):
    live_job(monkeypatch)
    calls = []
    answer = {"value": True}

    def cancel(job_id):
        calls.append(job_id)
        return answer["value"]

    monkeypatch.setattr(jobs, "cancel", cancel, raising=False)
    r = client.delete("/api/jobs/live")
    assert r.status_code == 200 and r.json() == {"cancelled": True}
    assert calls == ["live"]
    answer["value"] = False  # e.g. the run ended between the check and the cancel
    assert client.delete("/api/jobs/live").json() == {"cancelled": False}
    assert calls == ["live", "live"]


@pytest.mark.parametrize("has_cancel", [False, True])
@pytest.mark.parametrize("status", ["done", "failed"])
def test_a_finished_agent_run_returns_false(client, monkeypatch, has_cancel, status):
    live_job(monkeypatch, status)
    calls = []
    if has_cancel:
        monkeypatch.setattr(jobs, "cancel", lambda job_id: calls.append(job_id) or True, raising=False)
    else:
        monkeypatch.delattr(jobs, "cancel", raising=False)
    r = client.delete("/api/jobs/live")
    assert r.status_code == 200 and r.json() == {"cancelled": False}
    assert calls == []


def test_a_cancelled_agent_run_is_still_recorded(client, monkeypatch):
    """A stand-in for agent.jobs.cancel: it lets the run end, failed, at its next trace event. The server's watcher
    must record that run like any other finished one."""
    release = threading.Event()

    def slow(request, emit, planner=None):
        emit(TraceEvent(step="inspect", status="start", message="Loading.", t=0.0))
        release.wait(10)
        emit(TraceEvent(step="error", status="fail", message="The run was cancelled.", t=0.1))
        raise RuntimeError("Cancelled.")

    calls = []

    def cancel(job_id):
        calls.append(job_id)
        release.set()
        return True

    monkeypatch.setattr(loop, "run_fit", slow)
    monkeypatch.setattr(jobs, "cancel", cancel, raising=False)
    r = client.post("/api/fit", json={"model": "fake", "dataset": "gtzan"})
    assert r.status_code == 200, r.text
    job_id = r.json()["job_id"]
    assert client.delete(f"/api/jobs/{job_id}").json() == {"cancelled": True}
    assert calls == [job_id]
    state = finished(client, job_id)
    assert state["status"] == "failed" and state["error"] == "RuntimeError: Cancelled."
    listed = recorded_listing(client, job_id)
    assert listed["status"] == "failed" and listed["has_sense"] is False
    assert lines(runs.load_run(job_id)["events"]) == lines(state["events"])
    assert client.delete(f"/api/jobs/{job_id}").json() == {"cancelled": False}
    assert calls == [job_id]
