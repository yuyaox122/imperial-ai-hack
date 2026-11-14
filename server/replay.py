"""Replay mode: play a recorded agent run (sense.runs) back as a job, so the demo survives a bad network or a
slow model.

Public surface (do not rename):
  start(run_id, speed=1.0) -> str                 job id "replay-<12 hex chars>"; plays the record in a worker thread
  get(job_id) -> tuple[JobState, dict] | None     live state of a replay, and {"of": run_id, "recorded", "speed"}
  running() -> bool                               True while a replay is playing
  cancel(job_id) -> bool                          True when the replay was running and is now asked to stop;
                                                  False when the id is unknown or the replay had already finished

The replay job's request is the recorded one and it starts "running". Each recorded TraceEvent, unchanged, is
appended when the time since the replay started reaches event.t / speed: speed 1 plays in real time, 1000 almost
at once; it must be from 0.1 to 1000, else ValueError. Just before the recorded [install] "ok" event, the sense
the run installed is reinstalled from the record (sense.runs.restore_sense), so chat answers through it after
the replay as it did after the run. When the record holds no copy of the sense, an extra [install] "info" event
is added just before the recorded [install] "ok" event, saying the sense was not reinstalled; the recorded events
themselves are replayed unchanged. At the end the recorded cards, installed and error are set, and
then the status flips to the recorded one (as agent/jobs.py does, so a poller that sees the end sees the
results). If the replay itself fails, it ends "failed" with an [error] event saying why.

A cancelled replay stops at once: the worker checks for a cancel between events and while it waits for the next
event's time. It then appends a last [error] "fail" event, "The replay was cancelled before it finished." (data
{"replay": True, "cancelled": True}), sets error "Cancelled." and ends "failed". Cards and installed stay as they
were, empty and None, since they are only set at the end. A sense already reinstalled at the install step stays
installed. cancel waits up to CANCEL_WAIT_SECONDS for the worker to stop, which it does at once unless the
cancel comes while a sense is being reinstalled, so running() is normally False when cancel returns and a new
replay or agent run can start. If the reinstall outlasts that wait, cancel still returns True while running() is
True, and the replay ends cancelled once the reinstall returns. A cancel that comes as the replay reaches its end
is either applied or reported False.
So is one that comes as the replay fails: if the replay fails after a cancel was accepted (say the reinstall in
progress raises), its [error] event about the failure comes just before the cancel event and it still ends
"Cancelled.".

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
CANCEL_WAIT_SECONDS = 5.0  # how long cancel waits for the worker to stop

_lock = threading.Lock()
_replays: dict[str, tuple[JobState, dict]] = {}
_current_id: str | None = None
_workers: dict[str, tuple[threading.Event, threading.Thread]] = {}  # job id -> (cancel flag, worker) while playing

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
        stop = threading.Event()
        worker = threading.Thread(
            target=_play, args=(job, run_id, record, events, cards, speed, stop), name=f"replay-{job.id}", daemon=True
        )
        _workers[job.id] = (stop, worker)
        worker.start()  # under the lock, so that cancel never joins a worker that has not started
    return job.id


def get(job_id: str) -> tuple[JobState, dict] | None:
    found = _replays.get(job_id)
    return None if found is None else (found[0], dict(found[1]))


def running() -> bool:
    with _lock:
        return _playing()


def cancel(job_id: str) -> bool:
    with _lock:
        found = _replays.get(job_id)
        worker = _workers.get(job_id)
        if found is None or worker is None or found[0].status != "running":
            return False
        stop, thread = worker
        stop.set()
    thread.join(CANCEL_WAIT_SECONDS)
    return True


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
    job: JobState,
    run_id: str,
    record: dict,
    events: list[TraceEvent],
    cards: list[ReportCard],
    speed: float,
    stop: threading.Event,
) -> None:
    """Worker thread body. Ends the replay early once `stop` is set (see cancel)."""
    t0 = time.monotonic()
    try:
        for event in events:
            delay = event.t / speed - (time.monotonic() - t0)
            if stop.wait(max(0.0, delay)):  # returns at once when cancelled
                _end_cancelled(job, t0, speed)
                return
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
        with _lock:  # cancel reads the status under the same lock, so a late cancel is applied here or refused
            if stop.is_set():
                _end_cancelled(job, t0, speed)
                return
            job.cards = cards
            job.installed = record.get("installed")
            job.error = record.get("error")
            job.status = record["status"]
    except Exception as exc:
        log.exception("The replay of run %s stopped.", run_id)
        detail = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
        with _lock:  # as at the end: a cancel that came first is applied here, a later one is refused
            job.events.append(TraceEvent(
                step="error",
                status="fail",
                message=f"The replay of run {run_id} stopped. {detail} Record the run again, or replay another one.",
                data={"replay": True},
                t=round((time.monotonic() - t0) * speed, 3),
            ))
            if stop.is_set():  # cancel has already answered True: end as cancelled, this failure just before
                _end_cancelled(job, t0, speed)
                return
            job.error = f"Replay stopped: {detail}"
            job.status = "failed"
    finally:
        _workers.pop(job.id, None)


def _end_cancelled(job: JobState, t0: float, speed: float) -> None:
    """End a cancelled replay: a last [error] event, then error and status. Cards and installed are left as they are."""
    job.events.append(TraceEvent(
        step="error",
        status="fail",
        message="The replay was cancelled before it finished.",
        data={"replay": True, "cancelled": True},
        t=round((time.monotonic() - t0) * speed, 3),
    ))
    job.error = "Cancelled."
    job.status = "failed"
