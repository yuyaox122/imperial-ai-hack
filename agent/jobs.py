"""Background agent runs, one at a time.

Public surface (do not rename):
  start(request) -> str            job id; runs agent.loop.run_fit in a worker thread
  get(job_id) -> JobState | None   live state: events appended as they happen, cards and installed at the end
  latest() -> JobState | None
  add_listener(fn) -> None         fn(job_id: str, event: TraceEvent | None, cards: list[ReportCard] | None)
  remove_listener(fn) -> None      stops calling fn; a function never added is ignored

Listeners are module-level and shared by all jobs, and are called on the job's worker thread: once per
TraceEvent as fn(job_id, event, None), right after it is appended to the job's events, then once at the end
as fn(job_id, None, cards), after the status is set (cards is [] when the job failed). A listener that
raises is logged and skipped; it never stops the run or the other listeners.

Only one job runs at a time (the models share the laptop's memory): a second start while one is running
raises RuntimeError("an agent run is already in progress").

Jobs live in memory only and are lost when the server restarts.
"""

from __future__ import annotations

import logging
import threading
import uuid
from typing import Callable

from agent import loop
from sense.contracts import FitRequest, JobState, ReportCard, TraceEvent

Listener = Callable[[str, TraceEvent | None, list[ReportCard] | None], None]

_lock = threading.Lock()
_jobs: dict[str, JobState] = {}
_latest_id: str | None = None
_listeners: list[Listener] = []
_log = logging.getLogger(__name__)


def start(request: FitRequest) -> str:
    global _latest_id
    with _lock:
        if any(job.status == "running" for job in _jobs.values()):
            raise RuntimeError("an agent run is already in progress")
        job = JobState(id=uuid.uuid4().hex[:12], request=request)
        _jobs[job.id] = job
        _latest_id = job.id
    threading.Thread(target=_run, args=(job,), name=f"agent-run-{job.id}", daemon=True).start()
    return job.id


def get(job_id: str) -> JobState | None:
    return _jobs.get(job_id)


def latest() -> JobState | None:
    return _jobs.get(_latest_id) if _latest_id else None


def add_listener(fn: Listener) -> None:
    _listeners.append(fn)


def remove_listener(fn: Listener) -> None:
    if fn in _listeners:
        _listeners.remove(fn)


def _notify(job_id: str, event: TraceEvent | None, cards: list[ReportCard] | None) -> None:
    for fn in list(_listeners):
        try:
            fn(job_id, event, cards)
        except Exception:
            _log.warning("agent job listener %r failed on job %s", fn, job_id, exc_info=True)


def _run(job: JobState) -> None:
    """Worker thread body. Sets cards and installed before the status flips, so a poller that sees
    "done" also sees the results."""

    def emit(event: TraceEvent) -> None:
        job.events.append(event)
        _notify(job.id, event, None)

    try:
        cards, installed = loop.run_fit(job.request, emit)
    except Exception as exc:  # run_fit has already put an [error] event in the trace
        job.error = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
        job.status = "failed"
        _notify(job.id, None, [])
    else:
        job.cards = cards
        job.installed = installed
        job.status = "done"
        _notify(job.id, None, cards)
