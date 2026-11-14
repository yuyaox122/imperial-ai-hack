"""Replay mode: play a recorded agent run (sense.runs) back as a job, so the demo survives a bad network or a
slow model.

Public surface (do not rename):
  start(run_id, speed=1.0) -> str                 job id "replay-<12 hex chars>"; plays the record in a worker thread
  get(job_id) -> tuple[JobState, dict] | None     live state of a replay, and {"of": run_id, "recorded", "speed"}
  running() -> bool                               True while a replay is playing

The replay job's request is the recorded one and it starts "running". Each recorded TraceEvent, unchanged, is
appended when the time since the replay started reaches event.t / speed: speed 1 plays in real time, 1000 almost
at once; it must be from 0.1 to 1000, else ValueError. Just before the recorded [install] "ok" event, the sense
the run installed is reinstalled from the record (sense.runs.restore_sense), so chat answers through it after
the replay as it did after the run. When the record holds no copy of the sense, an extra [install] "info" event
is added just before the recorded [install] "ok" event, saying the sense was not reinstalled; the recorded events
themselves are replayed unchanged. At the end the recorded cards, installed and error are set, and
then the status flips to the recorded one (as agent/jobs.py does, so a poller that sees the end sees the
results). If the replay itself fails, it ends "failed" with an [error] event saying why.

Only one replay plays at a time: a second start raises RuntimeError("a replay is already in progress").
server/app.py also keeps replays and live agent runs apart. Replays live in memory only.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid

from sense import runs
from sense.contracts import JobState, ReportCard, TraceEvent

MIN_SPEED, MAX_SPEED = 0.1, 1000.0

_lock = threading.Lock()
_replays: dict[str, tuple[JobState, dict]] = {}
_current_id: str | None = None

log = logging.getLogger(__name__)


def start(run_id: str, speed: float = 1.0) -> str:
    global _current_id
    speed = _check_speed(speed)
    record = runs.load_run(run_id)
    request, events, cards = runs.parse_record(record)
    with _lock:
        if _playing():
            raise RuntimeError("a replay is already in progress")
        job = JobState(id=f"replay-{uuid.uuid4().hex[:12]}", request=request)
        _replays[job.id] = (job, {"of": run_id, "recorded": record.get("recorded"), "speed": speed})
        _current_id = job.id
    threading.Thread(
        target=_play, args=(job, run_id, record, events, cards, speed), name=f"replay-{job.id}", daemon=True
    ).start()
    return job.id


def get(job_id: str) -> tuple[JobState, dict] | None:
    found = _replays.get(job_id)
    return None if found is None else (found[0], dict(found[1]))


def running() -> bool:
    with _lock:
        return _playing()


def _playing() -> bool:
    found = _replays.get(_current_id) if _current_id else None
    return found is not None and found[0].status == "running"


def _check_speed(speed: object) -> float:
    try:
        value = float(speed)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        value = float("nan")
    if not MIN_SPEED <= value <= MAX_SPEED:  # also refuses nan
        raise ValueError(
            f"The replay speed {speed!r} is out of range. Use a number from {MIN_SPEED:g} to {MAX_SPEED:g}: "
            "1 plays the run in real time, 1000 almost at once."
        )
    return value


def _play(
    job: JobState, run_id: str, record: dict, events: list[TraceEvent], cards: list[ReportCard], speed: float
) -> None:
    """Worker thread body."""
    t0 = time.monotonic()
    try:
        for event in events:
            delay = event.t / speed - (time.monotonic() - t0)
            if delay > 0:
                time.sleep(delay)
            if event.step == "install" and event.status == "ok" and not runs.restore_sense(run_id):
                job.events.append(TraceEvent(
                    step="install",
                    status="info",
                    message=(
                        f"Replay: the record of run {run_id} holds no copy of {record.get('installed') or 'this sense'}, "
                        "so it was not reinstalled; chat keeps the sense installed now, if any."
                    ),
                    data={"replay": True},
                    t=event.t,
                ))
            job.events.append(event)
        job.cards = cards
        job.installed = record.get("installed")
        job.error = record.get("error")
        job.status = record["status"]
    except Exception as exc:
        log.exception("The replay of run %s stopped.", run_id)
        detail = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
        job.events.append(TraceEvent(
            step="error",
            status="fail",
            message=f"The replay of run {run_id} stopped. {detail} Record the run again, or replay another one.",
            data={"replay": True},
            t=round((time.monotonic() - t0) * speed, 3),
        ))
        job.error = f"Replay stopped: {detail}"
        job.status = "failed"
