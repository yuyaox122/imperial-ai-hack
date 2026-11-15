"""Background agent runs, one at a time.

Public surface (do not rename):
  start_auto(model, file, goal, examples=None) -> str  detect and research, then fit in the same worker queue
  start(request) -> str            job id; runs agent.loop.run_fit in a worker thread
  cancel(job_id) -> bool          request cooperative stop; False for unknown or terminal jobs
  get(job_id) -> JobState | None   live state: events appended as they happen, cards and installed at the end
  latest() -> JobState | None
  add_listener(fn) -> None         fn(job_id: str, event: TraceEvent | None, cards: list[ReportCard] | None)
  remove_listener(fn) -> None      stops calling fn; removing a function that is not registered does nothing

Listeners are module-level and shared by all jobs, and are called on the job's worker thread: once per
TraceEvent as fn(job_id, event, None), right after it is appended to the job's events, then once at the end
as fn(job_id, None, cards), after the status is set (cards is [] when the job failed). A listener that
raises is logged and skipped; it never stops the run or the other listeners.
Listener calls are serialised by a lock, so no two calls (even from two jobs) run at the same time.
Listeners receive the same TraceEvent and ReportCard objects the job stores: they must not mutate them.
A listener removed while a notification is in progress may get one more call.

Only one job runs at a time (the models share the laptop's memory): a second start while one is running
raises RuntimeError("an agent run is already in progress").

Cancellation stops at the next trace event (or worker return), marks the job failed with error
"Cancelled.", and publishes one final error event followed by the normal final listener call with [].
It cannot interrupt a model call already in progress or undo side effects already completed.

Jobs live in memory only and are lost when the server restarts.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from typing import Callable
from pathlib import Path

from agent import auto, loop
from sense.contracts import FitRequest, JobState, ReportCard, TraceEvent

_Listener = Callable[[str, TraceEvent | None, list[ReportCard] | None], None]

_lock = threading.Lock()
_jobs: dict[str, JobState] = {}
_cancel_requested: set[str] = set()
_terminal_published: set[str] = set()  # closes cancellation before terminal listeners run
_latest_id: str | None = None
_listeners: list[_Listener] = []
_notify_lock = threading.Lock()  # serialises listener calls across jobs
_log = logging.getLogger(__name__)


def start(request: FitRequest) -> str:
    return _start(request, lambda job, emit: loop.run_fit(job.request, emit))


def start_auto(model: str, file: Path, goal: str, examples: Path | None = None) -> str:
    def run(job: JobState, emit: Callable[[TraceEvent], None]):
        def on_request(request: FitRequest) -> None:
            job.request = request
        return auto.run_auto(model, file, goal, emit, examples=examples, on_request=on_request)

    return _start(FitRequest(model=model, dataset="", modality="", scope="closed"), run)


def _start(request: FitRequest, run: Callable) -> str:
    global _latest_id
    with _lock:
        if any(job.status == "running" for job in _jobs.values()):
            raise RuntimeError("an agent run is already in progress")
        job = JobState(id=uuid.uuid4().hex[:12], request=request)
        _jobs[job.id] = job
        _latest_id = job.id
    threading.Thread(target=_run, args=(job, run), name=f"agent-run-{job.id}", daemon=True).start()
    return job.id


def cancel(job_id: str) -> bool:
    """Request a cooperative stop at the next trace event or worker return.

    Return True while the job is running before its terminal trace event, including
    repeated requests, or False for unknown jobs or once error/done is published. Cancellation ends as failed/"Cancelled.";
    it does not interrupt ongoing computation or undo completed side effects.
    """
    with _lock:
        job = _jobs.get(job_id)
        if job is None or job.status != "running" or job_id in _terminal_published:
            return False
        _cancel_requested.add(job_id)
        return True


class _Cancelled(RuntimeError):
    """Unwind both fit and auto error emitters after a cancellation request."""


def get(job_id: str) -> JobState | None:
    return _jobs.get(job_id)


def latest() -> JobState | None:
    return _jobs.get(_latest_id) if _latest_id else None


def add_listener(fn: _Listener) -> None:
    _listeners.append(fn)


def remove_listener(fn: _Listener) -> None:
    try:
        _listeners.remove(fn)
    except ValueError:  # not registered
        pass


def _notify(job_id: str, event: TraceEvent | None, cards: list[ReportCard] | None) -> None:
    with _notify_lock:
        for fn in list(_listeners):
            try:
                fn(job_id, event, cards)
            except Exception:
                _log.warning("agent job listener %r failed on job %s", fn, job_id, exc_info=True)


def _run(job: JobState, run: Callable) -> None:
    """Finalize results before status, then notify once, including when cancellation
    also interrupts the runner's own exception-reporting trace events."""
    started = time.monotonic()

    def emit(event: TraceEvent) -> None:
        with _lock:
            if job.id in _cancel_requested:
                raise _Cancelled("Cancelled.")
            if event.step in ("error", "done"):
                # Reserve the terminal outcome atomically with publication. Listeners
                # run outside this lock and may call cancel, which must now refuse.
                _terminal_published.add(job.id)
            job.events.append(event)
        _notify(job.id, event, None)

    error = None
    cards, installed = [], None
    try:
        cards, installed = run(job, emit)
    except Exception as exc:  # ordinary failures were already traced by the runner
        error = exc

    cancellation_event = None
    with _lock:
        if job.id in _cancel_requested or isinstance(error, _Cancelled):
            job.error = "Cancelled."
            previous = job.events[-1] if job.events else None
            cancellation_event = TraceEvent(
                step="error", status="fail", message="Cancelled.",
                data={"step": previous.step if previous else "start", "error": "Cancelled"},
                t=max(previous.t if previous else 0, round(time.monotonic() - started, 3)),
            )
            # Bypass the cancellation-aware emitter so cancellation cannot swallow
            # its own final event, even when run_fit tried to report the exception.
            job.events.append(cancellation_event)
            job.status = "failed"
        elif error is not None:
            job.error = f"{type(error).__name__}: {error}" if str(error) else type(error).__name__
            job.status = "failed"
        else:
            job.cards = cards
            job.installed = installed
            job.status = "done"
        _cancel_requested.discard(job.id)
        _terminal_published.discard(job.id)

    if cancellation_event is not None:
        _notify(job.id, cancellation_event, None)
    _notify(job.id, None, job.cards if job.status == "done" else [])
