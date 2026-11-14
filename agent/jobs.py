"""Background agent runs, one at a time.

Public surface (do not rename):
  start(request) -> str            job id; runs agent.loop.run_fit in a worker thread
  get(job_id) -> JobState | None   live state: events appended as they happen, cards and installed at the end
  latest() -> JobState | None

Only one job runs at a time (the models share the laptop's memory): a second start while one is running
raises RuntimeError("an agent run is already in progress").

Jobs live in memory only and are lost when the server restarts.
"""

from __future__ import annotations

import threading
import uuid

from agent import loop
from sense.contracts import FitRequest, JobState

_lock = threading.Lock()
_jobs: dict[str, JobState] = {}
_latest_id: str | None = None


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


def _run(job: JobState) -> None:
    """Worker thread body. Sets cards and installed before the status flips, so a poller that sees
    "done" also sees the results."""
    try:
        cards, installed = loop.run_fit(job.request, job.events.append)
    except Exception as exc:  # run_fit has already put an [error] event in the trace
        job.error = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
        job.status = "failed"
    else:
        job.cards = cards
        job.installed = installed
        job.status = "done"
