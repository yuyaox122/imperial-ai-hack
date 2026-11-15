"""Background agent runs, one at a time.

Public surface (do not rename):
  start_auto(model, file, goal, examples=None) -> str  agent.auto.run_auto in the same worker queue: detect, then
                                   the native, specialist (audio) or adapter flow; job.request becomes the run's
                                   FitRequest once its data step has picked a set
  start(request) -> str            job id; runs agent.loop.run_fit in a worker thread
  cancel(job_id) -> bool          request cooperative stop; False for unknown or terminal jobs
  get(job_id) -> JobState | None   live state: events appended as they happen, cards and installed at the end
  latest() -> JobState | None
  add_listener(fn) -> None         fn(job_id: str, event: TraceEvent | None, cards: list[ReportCard] | None)
  remove_listener(fn) -> None      stops calling fn; removing a function that is not registered does nothing
  request_approval(kind, payload, *, timeout=600) -> bool
                                   asks the user to approve something (kind "install": packages for the toolbox)
                                   from inside a running job; blocks until answered (see Approvals)
  answer(job_id, approval_id, approved) -> bool
                                   True when it answered the job's open request; False for an unknown job or id, or
                                   one already answered, timed out or cancelled
  pending(job_id) -> dict | None   the open request: {"id", "kind", "payload", "timeout", "asked"} (asked: seconds
                                   since the job started), or None
  set_cli_handler(fn) -> None      fn(kind, payload) -> bool answers requests made outside any job (a script's
                                   terminal prompt); None removes it

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

Approvals. request_approval works only on a job's worker thread (the job context is set by the worker for the
run's duration; a thread the run starts does not inherit it unless it runs in a copy of the worker's context,
contextvars.copy_context()). It emits TraceEvent(step="approval", status="start", message, data={"id", "kind",
"payload", "timeout"}), where the message is the payload's "summary" when it has one; then it waits, at most
`timeout` seconds, for answer(job_id, id, approved). It then emits status "ok" when approved, or "fail" when
declined, timed out or cancelled, with data {"id", "kind", "answer": bool, "reason": "approved" | "declined" |
"timed out" | "cancelled"}, and returns the answer. Only one request may be open per job (a second raises
RuntimeError). cancel(job_id) also releases an open request as declined (reason "cancelled"): its "fail" event is
recorded, and the run stops at its next trace event as usual. `approved` counts only when it is True itself.
Outside a job (a CLI run) nothing is emitted: request_approval returns the CLI handler's answer (only True
approves; a handler that raises declines), or False when no handler is set.

Jobs live in memory only and are lost when the server restarts.
"""

from __future__ import annotations

import contextvars
import logging
import math
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable
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


@dataclass
class _Approval:
    id: str
    kind: str
    payload: dict
    timeout: float
    asked: float  # seconds since the job started
    done: threading.Event = field(default_factory=threading.Event)
    answer: bool | None = None
    reason: str = ""


@dataclass
class _JobContext:
    job: JobState
    publish: Callable[[TraceEvent, bool], None]  # (event, check_cancel)
    started: float


_context: contextvars.ContextVar[_JobContext | None] = contextvars.ContextVar("agent_job", default=None)
_approvals: dict[str, _Approval] = {}  # job id -> its open request
_cli_handler: Callable[[str, dict], bool] | None = None


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
        _settle(_approvals.get(job_id), False, "cancelled")  # a run waiting for an answer stops waiting
        return True


def _settle(request: _Approval | None, answer: bool, reason: str) -> bool:
    """Give an open request its answer and wake the run waiting for it; False when it was already settled. Call
    with _lock held."""
    if request is None or request.answer is not None:
        return False
    request.answer, request.reason = answer, reason
    request.done.set()
    return True


def request_approval(kind: str, payload: dict, *, timeout: float = 600) -> bool:
    """Ask the user to approve `kind` (described by `payload`, JSON-ready) and wait for the answer (Approvals in
    the module docstring). Returns True only when approved."""
    if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError(f"timeout must be a positive number of seconds, not {timeout!r}.")
    context = _context.get()
    if context is None:
        handler = _cli_handler
        if handler is None:
            return False
        try:
            return handler(kind, payload) is True
        except Exception:
            _log.warning("the approval handler failed on a %s request; treated as declined", kind, exc_info=True)
            return False
    job = context.job
    request = _Approval(
        id=uuid.uuid4().hex[:12], kind=str(kind), payload=payload, timeout=float(timeout),
        asked=round(time.monotonic() - context.started, 3),
    )
    with _lock:
        if job.id in _approvals:
            raise RuntimeError(f"job {job.id} already waits for an answer to request {_approvals[job.id].id}")
        _approvals[job.id] = request
    summary = payload.get("summary") if isinstance(payload, dict) else None
    try:
        context.publish(TraceEvent(
            step="approval", status="start",
            message=f"Waiting for your approval ({kind}): {summary}" if summary else f"Waiting for your approval ({kind}).",
            data={"id": request.id, "kind": request.kind, "payload": payload, "timeout": request.timeout},
            t=request.asked,
        ), True)
        if not request.done.wait(request.timeout):
            with _lock:
                _settle(request, False, "timed out")
    finally:
        with _lock:
            _settle(request, False, "cancelled")  # only when the start event itself was refused
            _approvals.pop(job.id, None)
    approved = request.answer is True
    message = {
        "approved": "Approved.",
        "declined": "Declined.",
        "timed out": f"No answer within {request.timeout:g} s, so it counts as declined.",
        "cancelled": "The run was cancelled, so it counts as declined.",
    }[request.reason]
    context.publish(TraceEvent(
        step="approval", status="ok" if approved else "fail", message=message,
        data={"id": request.id, "kind": request.kind, "answer": approved, "reason": request.reason},
        t=round(time.monotonic() - context.started, 3),
    ), False)  # recorded even after a cancel; the run stops at its next event
    return approved


def answer(job_id: str, approval_id: str, approved: bool) -> bool:
    """Answer the job's open request; False for an unknown job or id, or one no longer open."""
    with _lock:
        request = _approvals.get(job_id)
        if request is None or request.id != approval_id:
            return False
        return _settle(request, approved is True, "approved" if approved is True else "declined")


def pending(job_id: str) -> dict[str, Any] | None:
    """The job's open request, or None."""
    with _lock:
        request = _approvals.get(job_id)
        if request is None or request.answer is not None:
            return None
        return {"id": request.id, "kind": request.kind, "payload": request.payload, "timeout": request.timeout,
                "asked": request.asked}


def set_cli_handler(fn: Callable[[str, dict], bool] | None) -> None:
    """fn(kind, payload) -> bool answers requests made outside a job; None removes it."""
    global _cli_handler
    _cli_handler = fn


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

    def publish(event: TraceEvent, check_cancel: bool = True) -> None:
        with _lock:
            if check_cancel and job.id in _cancel_requested:
                raise _Cancelled("Cancelled.")
            if event.step in ("error", "done"):
                # Reserve the terminal outcome atomically with publication. Listeners
                # run outside this lock and may call cancel, which must now refuse.
                _terminal_published.add(job.id)
            job.events.append(event)
        _notify(job.id, event, None)

    def emit(event: TraceEvent) -> None:
        publish(event, True)

    error = None
    cards, installed = [], None
    token = _context.set(_JobContext(job=job, publish=publish, started=started))  # for request_approval
    try:
        cards, installed = run(job, emit)
    except Exception as exc:  # ordinary failures were already traced by the runner
        error = exc
    finally:
        _context.reset(token)
        with _lock:
            _settle(_approvals.pop(job.id, None), False, "cancelled")  # nothing is left waiting for this job

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
