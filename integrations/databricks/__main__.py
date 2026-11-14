"""Databricks for Sensorium (workload 4).

  .venv/bin/python -m integrations.databricks smoke              sign in, Volume upload, table insert, one MLflow run
  .venv/bin/python -m integrations.databricks setup              create the schema, tables and Volume
  .venv/bin/python -m integrations.databricks backfill [FILE..]  write recorded runs (sense.runs, or CLI --save files)
  .venv/bin/python -m integrations.databricks serve [--port 8000] the usual server (server.app), and every agent run
                                                                  it finishes is written to Databricks

Set up auth first: `databricks configure`, or DATABRICKS_HOST and DATABRICKS_TOKEN. See conn.py for the names.
Flags: --no-mlflow skips MLflow, --no-upload skips the Volume copy of run.json.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from integrations.databricks import conn, sink


def smoke(args: argparse.Namespace) -> int:
    ok = True

    def step(name: str, fn):
        nonlocal ok
        t0 = time.monotonic()
        try:
            out = fn()
            print(f"  ok    {name} ({time.monotonic() - t0:.1f} s){': ' + str(out) if out else ''}")
        except Exception as exc:  # report every step, then exit non-zero
            ok = False
            print(f"  FAIL  {name} ({time.monotonic() - t0:.1f} s): {type(exc).__name__}: {exc}")

    print("Databricks smoke test")
    step("sign in", lambda: f"{conn.client().config.host} as {conn.client().current_user.me().user_name}")
    wh = {}
    step("SQL warehouse", lambda: (wh.setdefault("w", conn.warehouse()), wh["w"].warehouse_id)[1])
    if "w" not in wh:
        return 1
    sql = wh["w"]
    step("schema, tables, Volume", lambda: sink.setup(sql))
    probe = f"/Volumes/{conn.catalog()}/{conn.schema()}/{sink.VOLUME}/_smoke/hello.txt"
    step("Volume upload", lambda: (conn.client().files.upload(probe, io.BytesIO(b"sensorium smoke\n"), overwrite=True), probe)[1])
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    record = {
        "id": "smoke-test", "origin": "cli", "recorded": now,
        "request": {"model": "fake", "dataset": "smoke", "modality": "audio", "scope": "closed"},
        "status": "done", "installed": "vector:smoke", "error": None, "seconds": 0.1,
        "events": [{"step": "done", "status": "ok", "message": "smoke test", "data": {}, "t": 0.1}],
        "cards": [{
            "model": "fake", "modality": "audio", "dataset": "smoke", "scope": "closed", "bridge": "vector:smoke",
            "kind": "vector", "n": 10, "chance": 0.2, "accuracy": 0.9, "interval": [0.6, 0.98],
            "controls": {"swapped": 0.2, "no_input": 0.2}, "verdict": "installed", "reason": "smoke test", "created": now,
        }],
    }
    step("table insert", lambda: sink.write_run(record, sql, upload=False, mlflow_runs=False))
    step("table read back", lambda: sql.run(f"SELECT run_id, accuracy FROM {conn.table('report_cards')} WHERE run_id = 'smoke-test'"))
    if not args.no_mlflow:
        step("MLflow run", lambda: (sink.log_mlflow(record), conn.experiment_path())[1])
    print("All steps passed." if ok else "Some steps failed; see above.")
    return 0 if ok else 1


def backfill(args: argparse.Namespace) -> int:
    records = []
    if args.files:
        records = [json.loads(Path(f).read_text()) for f in args.files]
    else:
        from sense import runs

        records = [runs.load_run(r["id"]) for r in runs.list_runs()]
    if not records:
        print("No recorded runs to write.")
        return 0
    sql = conn.warehouse()
    sink.setup(sql)
    failed = 0
    for record in records:
        try:
            counts = sink.write_run(record, sql, upload=not args.no_upload, mlflow_runs=not args.no_mlflow)
            print(f"  ok    {record['id']}: {counts}")
        except Exception as exc:
            failed += 1
            print(f"  FAIL  {record.get('id')}: {type(exc).__name__}: {exc}")
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m integrations.databricks", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["smoke", "setup", "backfill", "serve"])
    ap.add_argument("files", nargs="*", help="backfill: run records written by scripts/demo_cli.py --save")
    ap.add_argument("--no-mlflow", action="store_true")
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args(argv)
    if args.command == "smoke":
        return smoke(args)
    if args.command == "setup":
        sink.setup(conn.warehouse())
        print(f"Ready: {conn.catalog()}.{conn.schema()} (runs, report_cards, trace_events, Volume {sink.VOLUME})")
        return 0
    if args.command == "serve":
        return serve(args)
    return backfill(args)


def serve(args: argparse.Namespace) -> int:
    import uvicorn

    from agent import jobs

    sql = conn.warehouse()
    sink.setup(sql)
    jobs.add_listener(sink.Listener(sql, upload=not args.no_upload, mlflow_runs=not args.no_mlflow))
    print(f"Writing every finished agent run to Databricks ({conn.catalog()}.{conn.schema()}).")
    uvicorn.run("server.app:app", host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
