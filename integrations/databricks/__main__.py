"""Databricks for Sensorium (workload 4).

  .venv/bin/python -m integrations.databricks smoke              sign in, Volume upload, table insert and read back,
                                                                  one MLflow run; then the smoke run's rows and MLflow
                                                                  run are deleted again, so the tables hold real runs
  .venv/bin/python -m integrations.databricks setup              create the schema, tables and Volume (and add any
                                                                  column an older table lacks)
  .venv/bin/python -m integrations.databricks backfill [FILE..]  write recorded runs (sense.runs, or CLI --save files)
  .venv/bin/python -m integrations.databricks serve [--port 8000] the usual server (server.app), and every agent run
                                                                  it finishes is written to Databricks
  .venv/bin/python -m integrations.databricks triage [--model M] [--question Q] [--fold F] [--from FILE ...]
                                                                  run the forest-monitoring triage (eval.triage) with
                                                                  the installed sense, or take saved records, and
                                                                  write them to triage_batches and triage_clips

Set up auth first: `databricks configure`, or DATABRICKS_HOST and DATABRICKS_TOKEN (in the environment or the
repository's .env). See conn.py for the names.
Flags: --no-mlflow skips MLflow, --no-upload skips the Volume copy of run.json.
FILE arguments go with backfill (run records) and with triage --from (triage records; `--from a.json b.json`
and a shell glob after --from work). Any other command given files, --from outside triage, and --from with the
flags that shape a new batch (--model, --folder, --limit, --save, --question, --fold, --seed) are refused with
exit status 2. Files that are missing or not records of the right kind are named, nothing is written, and the
exit status is 1.
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

SMOKE_ID = "smoke-test"
SMOKE_ACCURACY = 0.9


def _read_back(sql: conn.SQL) -> str:
    """The smoke card's row, read back from report_cards; raises unless exactly that row comes back."""
    rows = sql.run(f"SELECT run_id, accuracy FROM {conn.table('report_cards')} WHERE run_id = :run_id",
                   {"run_id": SMOKE_ID})
    try:
        ok = len(rows) == 1 and rows[0][0] == SMOKE_ID and abs(float(rows[0][1]) - SMOKE_ACCURACY) < 1e-9
    except (TypeError, ValueError, IndexError):
        ok = False
    if not ok:
        raise RuntimeError(f"expected the one smoke row ({SMOKE_ID}, {SMOKE_ACCURACY}) back, got {rows!r}")
    return f"{rows[0][0]}, accuracy {rows[0][1]}"


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
        "id": SMOKE_ID, "origin": "cli", "recorded": now,
        "request": {"model": "fake", "dataset": "smoke", "modality": "audio", "scope": "closed"},
        "status": "done", "installed": "vector:smoke", "error": None, "seconds": 0.1,
        "events": [{"step": "done", "status": "ok", "message": "smoke test", "data": {}, "t": 0.1}],
        "cards": [{
            "model": "fake", "modality": "audio", "dataset": "smoke", "scope": "closed", "bridge": "vector:smoke",
            "kind": "vector", "n": 10, "chance": 0.2, "accuracy": SMOKE_ACCURACY, "interval": [0.6, 0.98],
            "controls": {"swapped": 0.2, "no_input": 0.2}, "verdict": "installed", "reason": "smoke test", "created": now,
        }],
    }
    step("table insert", lambda: sink.write_run(record, sql, upload=False, mlflow_runs=False))
    step("table read back", lambda: _read_back(sql))
    if not args.no_mlflow:
        step("MLflow run", lambda: (sink.log_mlflow(record), conn.experiment_path())[1])
    # The smoke card is made up; it must not stay in report_cards or the experiment beside real runs.
    step("clean up", lambda: (sink.delete_run(SMOKE_ID, sql, mlflow_runs=not args.no_mlflow),
                              f"{SMOKE_ID} rows{'' if args.no_mlflow else ' and MLflow runs'} deleted")[1])
    print("All steps passed." if ok else "Some steps failed; see above.")
    return 0 if ok else 1


def read_records(paths: list[str], check, what: str) -> list[dict] | None:
    """The JSON record in each path, each passed through check (which raises ValueError for a wrong shape).
    None, after naming every file that is missing, unreadable or not a record, if any is."""
    records, problems = [], []
    for path in paths:
        try:
            record = json.loads(Path(path).read_text())
            check(record)
            records.append(record)
        except FileNotFoundError:
            problems.append(f"{path}: no such file")
        except (OSError, ValueError) as exc:  # unreadable, not JSON, or not this kind of record
            problems.append(f"{path}: not {what} ({exc})")
    for problem in problems:
        print(problem, file=sys.stderr)
    if problems:
        print("Nothing was written.", file=sys.stderr)
        return None
    return records


def _check_run_record(record) -> None:
    from sense import runs

    runs.parse_record(record)


def backfill(args: argparse.Namespace) -> int:
    records = []
    if args.files:
        records = read_records(args.files, _check_run_record, "a run record (sense.runs, or scripts/demo_cli.py --save)")
        if records is None:
            return 1
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


def triage(args: argparse.Namespace) -> int:
    from eval import triage as tri

    saved: list[Path] = []
    if args.from_file:
        # `--from a b c` (or a shell glob after --from) puts b and c in the positional files; take them too.
        paths = [*args.from_file, *args.files]
        records = read_records(paths, tri.check_record, "a triage record (written by eval.triage)")
        if records is None:
            return 1
    else:
        try:
            records = tri.run_all(args)
        except (RuntimeError, ValueError, OSError) as exc:  # no sense installed, no such folder, no audio in it
            print(f"Triage failed: {exc}", file=sys.stderr)
            return 1
        for record in records:
            saved.append(tri.save(record, args.save if len(records) == 1 else None))
            print(f"Saved {saved[-1]}")
    for record in records:
        print(tri.table(record), "\n")
    if len(records) > 1:
        print(tri.compare(records), "\n")
    sql = conn.warehouse()
    sink.setup(sql)
    for k, record in enumerate(records):
        try:
            counts = sink.write_triage(record, sql, mlflow_runs=not args.no_mlflow)
        except Exception as exc:
            again = f" Write the saved records later with --from {' '.join(map(str, saved[k:]))}." if saved else ""
            print(f"Could not write triage {record['id']} to Databricks: {type(exc).__name__}: {exc}.{again}",
                  file=sys.stderr)
            return 1
        print(f"Wrote triage {record['id']} to {conn.catalog()}.{conn.schema()}: {counts}")
    return 0


def argument_problem(args: argparse.Namespace) -> str | None:
    """Arguments the command would otherwise ignore without a word, or None."""
    from eval import triage as tri

    if args.from_file and args.command != "triage":
        return "--from goes with triage. backfill takes its run records as FILE arguments."
    if args.files and args.command != "backfill" and not (args.command == "triage" and args.from_file):
        hint = (" To write saved triage records, put them after --from; without --from, triage runs a new batch."
                if args.command == "triage" else "")
        return f"{args.command} takes no FILE arguments (got {' '.join(args.files)}).{hint}"
    if args.command != "triage":
        return None
    if args.from_file:
        shaping = [f"--{k}" for k in ("model", "folder", "limit", "save") if getattr(args, k) is not None]
        shaping += [f"--{k}" for k, default in (("question", tri.DEFAULT_QUESTION), ("fold", "test"), ("seed", 0))
                    if getattr(args, k) != default]
        if shaping:
            return f"{', '.join(shaping)} shape a new batch; --from writes saved records as they are."
        return None
    return tri.argument_problem(args)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m integrations.databricks", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["smoke", "setup", "backfill", "serve", "triage"])
    ap.add_argument("files", nargs="*", help="backfill: run records written by scripts/demo_cli.py --save")
    ap.add_argument("--no-mlflow", action="store_true")
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--from", dest="from_file", action="append", help="triage: write this saved eval.triage record instead of running one")
    from eval import triage as tri

    tri.add_arguments(ap)
    args = ap.parse_args(argv)
    problem = argument_problem(args)
    if problem:
        ap.error(problem)
    if args.command == "smoke":
        return smoke(args)
    if args.command == "setup":
        sink.setup(conn.warehouse())
        print(f"Ready: {conn.catalog()}.{conn.schema()} ({', '.join(sink.DDL)}, Volume {sink.VOLUME})")
        return 0
    if args.command == "serve":
        return serve(args)
    if args.command == "triage":
        return triage(args)
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
