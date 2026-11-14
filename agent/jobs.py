"""Background agent runs, one at a time.

Public surface (do not rename):
  start(request) -> str            job id; runs agent.loop.run_fit in a worker thread
  get(job_id) -> JobState | None   live state: events appended as they happen, cards and installed at the end
  latest() -> JobState | None

Only one job runs at a time (the models share the laptop's memory): a second start while one is running
raises RuntimeError("an agent run is already in progress").
"""

from __future__ import annotations

from sense.contracts import FitRequest, JobState


def start(request: FitRequest) -> str:
    raise NotImplementedError


def get(job_id: str) -> JobState | None:
    raise NotImplementedError


def latest() -> JobState | None:
    raise NotImplementedError
