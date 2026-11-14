"""Writes agent runs to Databricks: three Delta tables, the run record in a Volume, one MLflow run per card.

Tables, in <catalog>.<schema> (see conn.py):
  runs          one row per agent run: run_id, recorded, origin, model, dataset, modality, scope, status,
                installed, seconds, n_events, error
  report_cards  one row per verified bridge: run_id, recorded, model, dataset, scope, bridge, kind, n, chance,
                accuracy, interval_low, interval_high, the six controls as columns, controls_json, verdict,
                reason, installed (this card's bridge was the one installed)
  trace_events  one row per trace event: run_id, i, t, step, status, message, data_json

Writing a run replaces that run's rows, so backfilling twice gives the same tables.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from integrations.databricks import conn

log = logging.getLogger(__name__)

CONTROLS = ("swapped", "no_input", "lookup", "specialist_alone", "text_bridge", "unseen_classes")
VOLUME = "runs"

DDL = {
    "runs": """run_id STRING, recorded TIMESTAMP, origin STRING, model STRING, dataset STRING, modality STRING,
        scope STRING, status STRING, installed STRING, seconds DOUBLE, n_events BIGINT, error STRING""",
    "report_cards": """run_id STRING, recorded TIMESTAMP, model STRING, dataset STRING, modality STRING,
        scope STRING, bridge STRING, kind STRING, n BIGINT, chance DOUBLE, accuracy DOUBLE,
        interval_low DOUBLE, interval_high DOUBLE, """
    + ", ".join(f"{c} DOUBLE" for c in CONTROLS)
    + """, controls_json STRING, verdict STRING, reason STRING, installed BOOLEAN""",
    "trace_events": """run_id STRING, i BIGINT, t DOUBLE, step STRING, status STRING, message STRING,
        data_json STRING""",
}


def column_types(name: str) -> dict[str, str]:
    return dict(c.split() for c in " ".join(DDL[name].split()).split(", "))


def setup(sql: conn.SQL) -> None:
    """Creates the schema, the tables and the Volume when they are missing."""
    sql.run(f"CREATE SCHEMA IF NOT EXISTS {conn.catalog()}.{conn.schema()}")
    for name, cols in DDL.items():
        sql.run(f"CREATE TABLE IF NOT EXISTS {conn.table(name)} ({' '.join(cols.split())})")
    sql.run(f"CREATE VOLUME IF NOT EXISTS {conn.catalog()}.{conn.schema()}.{VOLUME}")


# ----------------------------------------------------------------------------- rows
def run_row(record: dict) -> dict[str, Any]:
    req = record.get("request") or {}
    return {
        "run_id": record["id"],
        "recorded": record.get("recorded"),
        "origin": record.get("origin", "server"),
        "model": req.get("model"),
        "dataset": req.get("dataset"),
        "modality": req.get("modality", "audio"),
        "scope": req.get("scope", "closed"),
        "status": record.get("status"),
        "installed": record.get("installed"),
        "seconds": _float(record.get("seconds")),
        "n_events": len(record.get("events") or []),
        "error": record.get("error"),
    }


def card_rows(record: dict) -> list[dict[str, Any]]:
    rows = []
    for card in record.get("cards") or []:
        controls = card.get("controls") or {}
        interval = card.get("interval") or [None, None]
        row = {
            "run_id": record["id"],
            "recorded": record.get("recorded") or card.get("created"),
            "model": card.get("model"),
            "dataset": card.get("dataset"),
            "modality": card.get("modality", "audio"),
            "scope": card.get("scope"),
            "bridge": card.get("bridge"),
            "kind": card.get("kind"),
            "n": int(card.get("n") or 0),
            "chance": _float(card.get("chance")),
            "accuracy": _float(card.get("accuracy")),
            "interval_low": _float(interval[0]),
            "interval_high": _float(interval[1]),
        }
        row.update({c: _float(controls.get(c)) for c in CONTROLS})
        row.update(
            controls_json=json.dumps(controls, sort_keys=True),
            verdict=card.get("verdict"),
            reason=card.get("reason", ""),
            installed=card.get("bridge") == record.get("installed"),
        )
        rows.append(row)
    return rows


def event_rows(record: dict) -> list[dict[str, Any]]:
    return [
        {
            "run_id": record["id"],
            "i": i,
            "t": _float(e.get("t")),
            "step": e.get("step"),
            "status": e.get("status"),
            "message": e.get("message", ""),
            "data_json": json.dumps(e.get("data") or {}, sort_keys=True, default=str),
        }
        for i, e in enumerate(record.get("events") or [])
    ]


# ----------------------------------------------------------------------------- writing
def write_run(record: dict, sql: conn.SQL, *, upload: bool = True, mlflow_runs: bool = True) -> dict[str, int]:
    """Writes one run record (the run.json shape of sense.runs) to the tables, the Volume and MLflow."""
    run_id = record["id"]
    counts = {}
    for name, rows in (("runs", [run_row(record)]), ("report_cards", card_rows(record)), ("trace_events", event_rows(record))):
        sql.run(f"DELETE FROM {conn.table(name)} WHERE run_id = :run_id", {"run_id": run_id})
        _insert(sql, name, rows)
        counts[name] = len(rows)
    if upload:
        upload_record(record)
    if mlflow_runs:
        counts["mlflow"] = log_mlflow(record)
    return counts


def _insert(sql: conn.SQL, name: str, rows: list[dict[str, Any]], batch: int = 50) -> None:
    for start in range(0, len(rows), batch):
        chunk = rows[start : start + batch]
        cols = list(chunk[0])
        types = column_types(name)
        params: dict[str, Any] = {}
        values = []
        for r, row in enumerate(chunk):
            names = []
            for c in cols:
                key = f"{c}_{r}"
                params[key] = row[c]
                names.append(f"CAST(:{key} AS {types[c]})")  # typed, so a NULL in one row matches numbers in another
            values.append(f"({', '.join(names)})")
        sql.run(f"INSERT INTO {conn.table(name)} ({', '.join(cols)}) VALUES {', '.join(values)}", params)


def upload_record(record: dict) -> str:
    import io

    path = conn.volume_path(VOLUME, record["id"], "run.json")
    data = json.dumps(record, indent=1, default=str).encode()
    conn.client().files.upload(path, io.BytesIO(data), overwrite=True)
    return path


def log_mlflow(record: dict) -> int:
    """One MLflow run per report card, tagged with the agent run id. Earlier runs for the same id are deleted."""
    import mlflow

    mlflow.set_tracking_uri("databricks")
    exp = mlflow.set_experiment(conn.experiment_path())
    client = mlflow.MlflowClient()
    for old in client.search_runs([exp.experiment_id], f"tags.sensorium_run_id = '{record['id']}'"):
        client.delete_run(old.info.run_id)
    rows = card_rows(record)
    for row in rows:
        with mlflow.start_run(run_name=f"{row['bridge']} · {record['id']}"):
            mlflow.set_tags({"sensorium_run_id": record["id"], "verdict": row["verdict"], "installed": str(row["installed"])})
            mlflow.log_params({k: row[k] for k in ("model", "dataset", "modality", "scope", "bridge", "kind", "n")})
            metrics = {k: row[k] for k in ("accuracy", "chance", "interval_low", "interval_high", *CONTROLS)}
            mlflow.log_metrics({k: v for k, v in metrics.items() if v is not None})
            mlflow.log_text(row["reason"] or "", "reason.txt")
    return len(rows)


# ----------------------------------------------------------------------------- live runs
class Listener:
    """For agent.jobs.add_listener: fn(job_id, event, None) per trace event, then fn(job_id, None, cards) once.

    Writes the run from a background thread when the job ends, so a slow or failing Databricks call never holds
    up an agent run. Errors are logged.
    """

    def __init__(self, sql: conn.SQL | None = None, **write_options: Any):
        self._sql = sql
        self._opts = write_options

    def __call__(self, job_id: str, event: Any, cards: Any) -> None:
        if cards is None:
            return
        import threading

        threading.Thread(target=self._write, args=(job_id,), name=f"dbx-{job_id}", daemon=True).start()

    def _write(self, job_id: str) -> None:
        try:
            from agent import jobs

            state = jobs.get(job_id)
            if state is None:
                log.warning("Databricks: job %s is gone; nothing written", job_id)
                return
            record = record_from_state(state)
            sql = self._sql or conn.warehouse()
            counts = write_run(record, sql, **self._opts)
            log.info("Databricks: wrote run %s %s", job_id, counts)
        except Exception:
            log.exception("Databricks: could not write run %s", job_id)


def record_from_state(state: Any, origin: str = "server") -> dict:
    """A finished JobState (or a dict with its keys) as a run record, in the run.json shape of sense.runs."""
    import dataclasses
    from datetime import datetime, timezone

    raw = dataclasses.asdict(state) if dataclasses.is_dataclass(state) else dict(state)
    plain = json.loads(json.dumps(raw, default=lambda v: v.tolist() if hasattr(v, "tolist") else str(v)))
    events = plain.get("events") or []
    return {
        "id": plain["id"],
        "origin": origin,
        "recorded": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "request": plain.get("request") or {},
        "status": plain.get("status"),
        "events": events,
        "cards": plain.get("cards") or [],
        "installed": plain.get("installed"),
        "error": plain.get("error"),
        "seconds": _float(events[-1].get("t")) if events else None,
    }


def _float(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None
