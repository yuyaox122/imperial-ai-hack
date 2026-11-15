"""Recorded agent runs, kept under config.home() / "runs" so a finished run can be shown again after a server
restart (replay mode: server/replay.py plays a record back).

Public surface (do not rename):
  runs_dir() -> Path                          config.home() / "runs"
  save_run(state, origin="server", seconds=None) -> str
      Records a finished JobState, or a JobState-shaped dict (id, request, status, events, cards, installed,
      error), and returns its id. `seconds` defaults to the t of the last event. When the run installed a sense
      and the registry still holds that very sense, a copy of it is kept with the record. Saving an id that
      already exists replaces it. ValueError for a bad id, a run that is still running, or a malformed state.
  import_record(path, with_sense=True) -> str
      Records a JSON run record written by the CLI (scripts/demo_cli.py --save), keeping its origin, its
      recorded time and any extra keys such as "machine". With `with_sense`, the matching sense is copied from
      the registry when it is there.
  list_runs() -> list[dict]                   newest first: {"id", "request", "status", "installed", "recorded",
                                              "origin", "seconds", "has_sense"}; unreadable records are skipped
                                              with a logged warning
  load_run(run_id) -> dict                    the whole record; ValueError for an unknown or malformed id
  restore_sense(run_id) -> bool               reinstalls the recorded sense with registry.install; False when
                                              the record holds no sense
  parse_record(record) -> tuple[FitRequest, list[TraceEvent], list[ReportCard]]
      The contract objects of a record (keys the contracts do not know are ignored, and a card recorded before
      ReportCard.control_chance existed, or with a value that is not an object there, gets {}); ValueError when
      the record cannot be replayed.

Layout: <runs_dir>/<id>/run.json, plus <runs_dir>/<id>/sense/ (card.json and the saved bridge, as in the
registry) when a sense was recorded. run.json holds id, origin ("server" or "cli"), recorded (ISO 8601 UTC),
request, status, events, cards, installed, error, seconds and format (1). Ids are 1 to 64 letters, digits, '_'
or '-'. A record is written to a staging directory first and then renamed into place.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import threading
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sense import config, registry
from sense.bridges import load_bridge
from sense.contracts import FitRequest, ReportCard, TraceEvent

FORMAT = 1
ORIGINS = ("server", "cli")
_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")
_KEYS = ("id", "request", "status", "events", "cards", "installed", "error")
_FINISHED = ("done", "failed")

log = logging.getLogger(__name__)


# ----------------------------------------------------------------------------- public
def runs_dir() -> Path:
    return config.home() / "runs"


def save_run(state: Any, origin: str = "server", seconds: float | None = None) -> str:
    if origin not in ORIGINS:
        raise ValueError(f"Unknown run origin {origin!r}. Use 'server' or 'cli'.")
    raw = state.to_dict() if hasattr(state, "to_dict") else state
    record = _record(raw, keep_extra=False)
    record["origin"] = origin
    record["recorded"] = _now()
    record["seconds"] = seconds if seconds is not None else _last_t(record["events"])
    return _write(record, with_sense=True)


def import_record(path: str | Path, with_sense: bool = True) -> str:
    path = Path(path).expanduser()
    try:
        raw = json.loads(path.read_text())
    except FileNotFoundError:
        raise ValueError(f"{path} does not exist. Give the path of a run record written by scripts/demo_cli.py --save.") from None
    except (OSError, ValueError) as exc:
        raise ValueError(f"{path} is not a readable run record ({exc}). Save the run again with scripts/demo_cli.py --save.") from None
    if not isinstance(raw, dict):
        raise ValueError(f"{path} does not hold a run record (a JSON object). Save the run again with scripts/demo_cli.py --save.")
    record = _record(raw, keep_extra=True)
    origin, recorded, seconds = raw.get("origin"), raw.get("recorded"), raw.get("seconds")
    record["origin"] = origin if isinstance(origin, str) and origin else "cli"
    record["recorded"] = recorded if isinstance(recorded, str) and recorded else _now()
    has_number = isinstance(seconds, (int, float)) and not isinstance(seconds, bool)
    record["seconds"] = seconds if has_number else _last_t(record["events"])
    return _write(record, with_sense=with_sense)


def list_runs() -> list[dict]:
    root = runs_dir()
    if not root.is_dir():
        return []
    found = []
    for directory in root.iterdir():
        if directory.name.startswith(".") or not directory.is_dir():
            continue  # staging and removal directories of a save in progress
        try:
            record = json.loads((directory / "run.json").read_text())
            if not isinstance(record, dict) or record.get("id") != directory.name:
                raise ValueError(f"run.json does not describe the run {directory.name}")
            summary = {
                "id": record["id"],
                "request": record["request"],
                "status": record["status"],
                "installed": record.get("installed"),
                "recorded": record.get("recorded"),
                "origin": record.get("origin"),
                "seconds": record.get("seconds"),
                "has_sense": _has_sense(directory),
            }
            mtime = (directory / "run.json").stat().st_mtime
        except Exception as exc:  # one bad record must not hide the others
            log.warning("Skipping the recorded run in %s: it cannot be read (%s). Save it again or remove it.", directory, exc)
            continue
        found.append(((_when(summary["recorded"]), mtime), summary))
    found.sort(key=lambda pair: pair[0], reverse=True)
    return [summary for _, summary in found]


def load_run(run_id: str) -> dict:
    run_id = _check_id(run_id)
    path = runs_dir() / run_id / "run.json"
    if not path.is_file():
        raise ValueError(f"No recorded run '{run_id}'. List the recorded runs (GET /api/runs) and pick one of them.")
    try:
        record = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(f"The recorded run '{run_id}' cannot be read ({exc}). Record it again, or remove {path.parent}.") from None
    if not isinstance(record, dict):
        raise ValueError(f"The recorded run '{run_id}' is not a JSON object. Record it again, or remove {path.parent}.")
    return record


def restore_sense(run_id: str) -> bool:
    load_run(run_id)  # ValueError for an unknown run
    directory = runs_dir() / run_id / "sense"
    if not _has_sense(directory.parent):
        return False
    bridge = load_bridge(directory)
    card = _card(json.loads((directory / "card.json").read_text()))
    registry.install(bridge, card)
    return True


def parse_record(record: dict) -> tuple[FitRequest, list[TraceEvent], list[ReportCard]]:
    run_id = record.get("id", "?") if isinstance(record, dict) else "?"
    fix = "Record the run again."
    if not isinstance(record, dict):
        raise ValueError(f"A run record must be a JSON object. {fix}")
    missing = [key for key in _KEYS if key not in record]
    if missing:
        raise ValueError(f"The run record {run_id} lacks {', '.join(missing)}. {fix}")
    if record["status"] not in _FINISHED:
        raise ValueError(
            f"The run {run_id} is '{record['status']}'; only a finished run (done or failed) can be recorded "
            "and replayed. Save it when it has finished."
        )
    try:
        request = FitRequest(**_known(FitRequest, record["request"]))
        events = [TraceEvent(**_known(TraceEvent, e)) for e in record["events"]]
        cards = [_card(c) for c in record["cards"]]
        for event in events:
            event.t = float(event.t)
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError(f"The run record {run_id} is malformed ({exc}). {fix}") from None
    return request, events, cards


# ----------------------------------------------------------------------------- helpers
def _check_id(run_id: Any) -> str:
    if not isinstance(run_id, str) or not _ID.fullmatch(run_id):
        raise ValueError(f"{run_id!r} is not a valid run id: use 1 to 64 letters, digits, '_' or '-'.")
    return run_id


def _known(cls: type, data: Any) -> dict:
    """The keys of `data` that the dataclass `cls` takes, so a record written by a newer version still loads."""
    if not isinstance(data, dict):
        raise TypeError(f"expected an object for {cls.__name__}, got {type(data).__name__}")
    names = {f.name for f in fields(cls)}
    return {k: v for k, v in data.items() if k in names}


def _card(data: Any) -> ReportCard:
    """A ReportCard from a recorded card dict. A card recorded before control_chance existed (or holding null
    there) gets {}, so every reader can call card.control_chance.get(...)."""
    known = _known(ReportCard, data)
    if not isinstance(known.get("control_chance"), dict):
        known["control_chance"] = {}
    return ReportCard(**known)


def _plain(value: Any) -> Any:
    """JSON fallback for numpy values in event data."""
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return str(value)


def _record(raw: Any, keep_extra: bool) -> dict:
    """A validated, JSON-ready copy of a JobState-shaped dict, with its keys in a readable order."""
    if not isinstance(raw, dict):
        raise ValueError("A run to record must be a JobState or a dict with the keys of one.")
    plain = json.loads(json.dumps(raw, default=_plain))  # dataclasses and numpy values become plain JSON
    parse_record(plain)
    record = {"id": _check_id(plain["id"]), "origin": None, "recorded": None}
    record.update({key: plain[key] for key in _KEYS})
    record.update({"seconds": None, "format": FORMAT})
    if keep_extra:
        record.update({k: v for k, v in plain.items() if k not in record})
    return record


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _last_t(events: list[dict]) -> float | None:
    return float(events[-1].get("t", 0.0)) if events else None


def _when(recorded: Any) -> datetime:
    try:
        when = datetime.fromisoformat(recorded)
    except (TypeError, ValueError):
        return datetime.min.replace(tzinfo=timezone.utc)
    return when if when.tzinfo else when.replace(tzinfo=timezone.utc)


def _has_sense(run_directory: Path) -> bool:
    sense = run_directory / "sense"
    return (sense / "card.json").is_file() and (sense / "bridge.json").is_file()


def _read_card(directory: Path) -> dict | None:
    try:
        card = json.loads((directory / "card.json").read_text())
    except (OSError, ValueError):
        return None
    return card if isinstance(card, dict) else None


def _matches(card: dict | None, record: dict) -> bool:
    """True when an installed card is the one this run installed: same bridge and, when the run's card has a
    creation time, the same one (a later run may have installed a bridge of the same name)."""
    if card is None or card.get("bridge") != record["installed"]:
        return False
    created = next((c.get("created") for c in record["cards"] if c.get("bridge") == record["installed"]), "")
    return not created or card.get("created") == created


def _copy_sense(record: dict, destination: Path) -> bool:
    """Copy the sense this run installed from the registry, if the registry still holds it."""
    if not record["installed"]:
        return False
    request = record["request"]
    try:
        source = registry.sense_dir(request["model"], request.get("modality", "audio"))
    except (KeyError, ValueError):
        return False
    if not _matches(_read_card(source), record):
        return False
    try:
        shutil.copytree(source, destination)
    except OSError as exc:  # e.g. replaced by another install while copying
        shutil.rmtree(destination, ignore_errors=True)
        log.warning("Recording run %s without its sense: copying %s failed (%s).", record["id"], source, exc)
        return False
    if not _matches(_read_card(destination), record):
        shutil.rmtree(destination, ignore_errors=True)
        log.warning("Recording run %s without its sense: %s was replaced while it was copied.", record["id"], source)
        return False
    return True


def _write(record: dict, with_sense: bool) -> str:
    run_id = record["id"]
    root = runs_dir()
    root.mkdir(parents=True, exist_ok=True)
    tag = f"{os.getpid()}-{threading.get_ident()}"
    target = root / run_id
    staging = root / f".{run_id}.saving-{tag}"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir()
    old = None
    try:
        if with_sense:
            _copy_sense(record, staging / "sense")
        (staging / "run.json").write_text(json.dumps(record, indent=2, default=_plain))
        if target.exists():
            old = root / f".{run_id}.removing-{tag}"
            target.rename(old)
        staging.rename(target)
    except BaseException:
        if old is not None and old.exists() and not target.exists():
            old.rename(target)  # keep the earlier record rather than none
            old = None
        shutil.rmtree(staging, ignore_errors=True)
        raise
    if old is not None:
        shutil.rmtree(old, ignore_errors=True)
    return run_id
