"""agent.jobs approvals: a running job asks the user to approve something and waits for the answer.

Run: SENSE_FAKE=1 .venv/bin/python -m pytest -q tests/test_approvals.py
"""

from __future__ import annotations

import contextvars
import json
import threading
import time

import pytest

from agent import jobs
from sense.contracts import FitRequest, TraceEvent

WAIT = 10.0  # the most any test waits for something that should happen at once


@pytest.fixture(autouse=True)
def fresh_jobs(monkeypatch):
    monkeypatch.setattr(jobs, "_jobs", {})
    monkeypatch.setattr(jobs, "_latest_id", None)
    monkeypatch.setattr(jobs, "_approvals", {})
    monkeypatch.setattr(jobs, "_cli_handler", None)
    monkeypatch.setattr(jobs, "_listeners", [])


class Watch:
    """A listener that records a job's events and signals when an approval request opens."""

    def __init__(self):
        self.events: list[TraceEvent] = []
        self.asked = threading.Event()
        self.finished = threading.Event()
        jobs.add_listener(self)

    def __call__(self, job_id, event, cards):
        if event is None:
            self.finished.set()
            return
        self.events.append(event)
        if event.step == "approval" and event.status == "start":
            self.asked.set()

    def approvals(self) -> list[TraceEvent]:
        return [e for e in self.events if e.step == "approval"]


def start(run) -> tuple[str, Watch, dict]:
    """Start a job whose worker runs run(emit) -> result; returns (job id, watch, outcome dict)."""
    watch, outcome = Watch(), {}

    def worker(job, emit):
        emit(TraceEvent("design", "start", "Working."))
        outcome["result"] = run(emit)
        emit(TraceEvent("done", "ok", "Finished.", {"installed": None}))
        return [], None

    job_id = jobs._start(FitRequest(model="fake", dataset=""), worker)
    return job_id, watch, outcome


def ask(kind="install", timeout=WAIT):
    payload = {"summary": "OK to install meshkit.", "plan": {"verdict": "ok"}}
    return lambda emit: jobs.request_approval(kind, payload, timeout=timeout)


def test_an_approved_request_returns_true_with_its_events():
    job_id, watch, outcome = start(ask())
    assert watch.asked.wait(WAIT)
    open_request = jobs.pending(job_id)
    assert open_request["kind"] == "install" and open_request["payload"]["summary"] == "OK to install meshkit."
    assert open_request["timeout"] == WAIT and isinstance(open_request["asked"], float)
    assert jobs.answer(job_id, "not-the-id", True) is False and jobs.answer("no-such-job", open_request["id"], True) is False
    assert jobs.answer(job_id, open_request["id"], True) is True
    assert jobs.answer(job_id, open_request["id"], False) is False  # answered already
    assert watch.finished.wait(WAIT)
    assert outcome["result"] is True and jobs.get(job_id).status == "done" and jobs.pending(job_id) is None
    first, second = watch.approvals()
    assert (first.status, second.status) == ("start", "ok")
    assert first.data == {"id": open_request["id"], "kind": "install", "timeout": WAIT,
                          "payload": {"summary": "OK to install meshkit.", "plan": {"verdict": "ok"}}}
    assert "OK to install meshkit." in first.message
    assert second.data == {"id": open_request["id"], "kind": "install", "answer": True, "reason": "approved"}
    assert [e.step for e in jobs.get(job_id).events] == ["design", "approval", "approval", "done"]
    json.dumps([e.to_dict() for e in jobs.get(job_id).events])


def test_a_declined_request_returns_false():
    job_id, watch, outcome = start(ask())
    assert watch.asked.wait(WAIT)
    assert jobs.answer(job_id, jobs.pending(job_id)["id"], "yes") is True  # only True itself approves
    assert watch.finished.wait(WAIT)
    assert outcome["result"] is False
    end = watch.approvals()[-1]
    assert end.status == "fail" and end.data["answer"] is False and end.data["reason"] == "declined"


def test_an_unanswered_request_times_out_as_declined():
    started = time.monotonic()
    job_id, watch, outcome = start(ask(timeout=0.2))
    assert watch.finished.wait(WAIT)
    assert outcome["result"] is False and time.monotonic() - started < WAIT
    end = watch.approvals()[-1]
    assert end.status == "fail" and end.data["reason"] == "timed out" and "0.2 s" in end.message
    assert jobs.pending(job_id) is None


def test_cancel_releases_a_waiting_request_and_stops_the_run():
    job_id, watch, outcome = start(ask())
    assert watch.asked.wait(WAIT)
    request_id = jobs.pending(job_id)["id"]
    assert jobs.cancel(job_id) is True
    assert watch.finished.wait(WAIT)
    assert outcome["result"] is False
    state = jobs.get(job_id)
    assert state.status == "failed" and state.error == "Cancelled."
    steps = [(e.step, e.status) for e in state.events]
    assert steps == [("design", "start"), ("approval", "start"), ("approval", "fail"), ("error", "fail")]
    assert state.events[2].data == {"id": request_id, "kind": "install", "answer": False, "reason": "cancelled"}
    assert jobs.answer(job_id, request_id, True) is False


def test_one_open_request_per_job():
    seen = {}

    def two(emit):
        context = contextvars.copy_context()  # a helper thread in the worker's context may ask too
        helper = threading.Thread(target=context.run, args=(jobs.request_approval, "install", {}),
                                  kwargs={"timeout": WAIT})
        helper.start()
        deadline = time.monotonic() + WAIT
        while jobs.pending(jobs.latest().id) is None and time.monotonic() < deadline:
            time.sleep(0.005)
        try:
            jobs.request_approval("install", {})
        except RuntimeError as exc:
            seen["error"] = str(exc)
        jobs.answer(jobs.latest().id, jobs.pending(jobs.latest().id)["id"], True)
        helper.join(WAIT)
        return True

    job_id, watch, _ = start(two)
    assert watch.finished.wait(WAIT)
    assert "already waits for an answer" in seen["error"] and jobs.get(job_id).status == "done"


def test_a_thread_the_run_starts_has_no_job_context():
    def from_a_plain_thread(emit):
        found = {}
        helper = threading.Thread(target=lambda: found.update(answer=jobs.request_approval("install", {})))
        helper.start()
        helper.join(WAIT)
        return found["answer"]

    job_id, watch, outcome = start(from_a_plain_thread)
    assert watch.finished.wait(WAIT)
    assert outcome["result"] is False and watch.approvals() == []


def test_outside_a_job_the_cli_handler_answers():
    assert jobs.request_approval("install", {"summary": "x"}) is False  # no job, no handler: declined, no events
    asked = []
    jobs.set_cli_handler(lambda kind, payload: asked.append((kind, payload)) or True)
    assert jobs.request_approval("install", {"summary": "x"}) is True and asked == [("install", {"summary": "x"})]
    jobs.set_cli_handler(lambda kind, payload: "yes")
    assert jobs.request_approval("install", {}) is False  # only True approves

    def broken(kind, payload):
        raise EOFError("no terminal")

    jobs.set_cli_handler(broken)
    assert jobs.request_approval("install", {}) is False
    jobs.set_cli_handler(None)
    assert jobs.request_approval("install", {}) is False
    with pytest.raises(ValueError, match="positive number"):
        jobs.request_approval("install", {}, timeout=0)
