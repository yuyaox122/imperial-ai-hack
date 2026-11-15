"""integrations.databricks without a workspace: rows from a run record and the SQL it sends, the CLI's checks,
the smoke test's clean-up, MLflow replacement, column migration and statement polling. Nothing goes over the
network: the SDK client, the warehouse and mlflow are fakes."""

import contextlib
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

from integrations.databricks import __main__ as cli
from integrations.databricks import conn, sink

REPO = Path(__file__).resolve().parent.parent
ARTIFACTS = REPO / "socket_pilot" / "artifacts"
TRIAGE_RECORDS = sorted(ARTIFACTS.glob("triage_*.json"))

RECORD = {
    "id": "run1",
    "origin": "cli",
    "recorded": "2026-11-14T18:55:33+00:00",
    "request": {"model": "fake", "dataset": "gtzan", "modality": "audio", "scope": "closed"},
    "status": "done",
    "installed": "vector:gtzan",
    "error": None,
    "seconds": 0.3,
    "events": [
        {"step": "plan", "status": "ok", "message": "Bridges to try", "data": {"kinds": ["vector", "text"]}, "t": 0.026},
        {"step": "done", "status": "ok", "message": "done", "data": {}, "t": 0.3},
    ],
    "cards": [
        {"model": "fake", "modality": "audio", "dataset": "gtzan", "scope": "closed", "bridge": "vector:gtzan",
         "kind": "vector", "n": 60, "chance": 0.2, "accuracy": 1.0, "interval": [0.94, 1.0],
         "controls": {"swapped": 0.1, "no_input": 0.2333}, "verdict": "installed", "reason": "r", "created": "x"},
        {"model": "fake", "modality": "audio", "dataset": "gtzan", "scope": "closed", "bridge": "text:gtzan",
         "kind": "text", "n": 60, "chance": 0.2, "accuracy": 0.5, "interval": None,
         "controls": {}, "verdict": "rejected", "reason": "r2", "created": "x"},
    ],
}


class FakeSQL:
    def __init__(self):
        self.calls = []

    def run(self, statement, params=None):
        self.calls.append((statement, params or {}))
        return []


def test_card_rows_flatten_controls_and_mark_the_installed_bridge():
    rows = sink.card_rows(RECORD)
    assert [r["bridge"] for r in rows] == ["vector:gtzan", "text:gtzan"]
    assert rows[0]["installed"] is True and rows[1]["installed"] is False
    assert rows[0]["swapped"] == 0.1 and rows[0]["lookup"] is None
    assert rows[0]["interval_low"] == 0.94 and rows[1]["interval_low"] is None
    assert set(rows[0]) == {c.split()[0] for c in " ".join(sink.DDL["report_cards"].split()).split(", ")}


def test_run_and_event_rows_match_their_tables():
    for name, rows in (("runs", [sink.run_row(RECORD)]), ("trace_events", sink.event_rows(RECORD))):
        cols = {c.split()[0] for c in " ".join(sink.DDL[name].split()).split(", ")}
        assert all(set(r) == cols for r in rows)
    assert sink.event_rows(RECORD)[0]["data_json"] == '{"kinds": ["vector", "text"]}'


def test_write_run_replaces_the_run_rows_with_bound_parameters():
    sql = FakeSQL()
    counts = sink.write_run(RECORD, sql, upload=False, mlflow_runs=False)
    assert counts == {"runs": 1, "report_cards": 2, "trace_events": 2}
    statements = [s for s, _ in sql.calls]
    assert statements[0].startswith(f"DELETE FROM {conn.table('runs')} WHERE run_id = :run_id")
    inserts = [(s, p) for s, p in sql.calls if s.startswith("INSERT")]
    assert len(inserts) == 3
    s, p = inserts[1]
    assert "CAST(:recorded_0 AS TIMESTAMP)" in s and p["bridge_1"] == "text:gtzan"
    assert "CAST(:lookup_1 AS DOUBLE)" in s and p["lookup_1"] is None  # a NULL control keeps the column's type
    assert "vector:gtzan" not in s  # values travel as parameters, never inside the SQL text


def test_names_refuse_anything_but_identifiers(monkeypatch):
    monkeypatch.setenv("SENSE_DBX_SCHEMA", "x; DROP TABLE y")
    with pytest.raises(ValueError):
        conn.table("runs")


def test_record_from_state_matches_the_run_record_shape():
    from sense.contracts import FitRequest, JobState, ReportCard, TraceEvent

    state = JobState(
        id="j1", request=FitRequest(model="fake", dataset="gtzan"), status="done",
        events=[TraceEvent(step="done", status="ok", message="m", t=1.5)],
        cards=[ReportCard(model="fake", modality="audio", dataset="gtzan", scope="closed", bridge="vector:gtzan",
                          kind="vector", n=10, chance=0.2, accuracy=0.9, controls={"swapped": 0.1})],
        installed="vector:gtzan",
    )
    record = sink.record_from_state(state)
    assert record["seconds"] == 1.5 and record["request"]["dataset"] == "gtzan"
    assert sink.card_rows(record)[0]["installed"] is True


def test_listener_writes_a_finished_fake_job(monkeypatch):
    import threading

    from agent import jobs
    from sense.contracts import FitRequest

    monkeypatch.setenv("SENSE_FAKE", "1")
    sql = FakeSQL()
    done = threading.Event()
    listener = sink.Listener(sql, upload=False, mlflow_runs=False)
    original = listener._write
    monkeypatch.setattr(listener, "_write", lambda job_id: (original(job_id), done.set()))
    jobs.add_listener(listener)
    try:
        job_id = jobs.start(FitRequest(model="fake", dataset="gtzan"))
        assert done.wait(60)
    finally:
        jobs.remove_listener(listener)
    inserts = [p for s, p in sql.calls if s.startswith(f"INSERT INTO {conn.table('report_cards')}")]
    assert inserts and inserts[0]["run_id_0"] == job_id
    assert any(p.get("installed_0") or p.get("installed_1") for p in inserts)


# ----------------------------------------------------------------------------- fakes for the CLI, MLflow, the SDK
class AnsweringSQL(FakeSQL):
    """FakeSQL that answers statements starting with a given prefix (rows, or a function of the params)."""

    def __init__(self, answers=None):
        super().__init__()
        self.answers = answers or {}
        self.warehouse_id = "wh-test"

    def run(self, statement, params=None):
        super().run(statement, params)
        for prefix, rows in self.answers.items():
            if statement.startswith(prefix):
                return rows(params or {}) if callable(rows) else rows
        return []


class FakeMlflow(types.ModuleType):
    """Just enough of mlflow for sink: runs with tags in one experiment, searchable by one tag."""

    def __init__(self):
        super().__init__("mlflow")
        self.runs = []  # dicts: id, name, tags, deleted
        self._active = None
        fake = self

        class Client:
            def search_runs(self, experiment_ids, filter_string):
                key, _, value = filter_string.partition(" = ")
                tag, value = key.removeprefix("tags."), value.strip("'")
                return [types.SimpleNamespace(info=types.SimpleNamespace(run_id=r["id"]))
                        for r in fake.runs if not r["deleted"] and r["tags"].get(tag) == value]

            def delete_run(self, run_id):
                next(r for r in fake.runs if r["id"] == run_id)["deleted"] = True

        self.MlflowClient = Client

    def live(self, tag, value):
        return [r for r in self.runs if not r["deleted"] and r["tags"].get(tag) == value]

    def set_tracking_uri(self, uri):
        assert uri == "databricks"

    def set_experiment(self, path):
        return types.SimpleNamespace(experiment_id="exp1")

    def get_experiment_by_name(self, path):
        return types.SimpleNamespace(experiment_id="exp1")

    def start_run(self, run_name=None):
        run = {"id": f"r{len(self.runs)}", "name": run_name, "tags": {}, "deleted": False}
        self.runs.append(run)

        @contextlib.contextmanager
        def active():
            self._active = run
            yield run
            self._active = None

        return active()

    def set_tags(self, tags):
        self._active["tags"].update(tags)

    def log_params(self, params):
        pass

    def log_metrics(self, metrics):
        pass

    def log_text(self, text, name):
        pass


@pytest.fixture
def mlflow(monkeypatch):
    fake = FakeMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    monkeypatch.setattr(conn, "experiment_path", lambda: "/Users/someone/sensorium")
    return fake


@pytest.fixture
def workspace(monkeypatch):
    """conn.client and conn.warehouse replaced by fakes: nothing goes over the network."""
    sql = AnsweringSQL()
    user = types.SimpleNamespace(user_name="someone@example.com")
    client = types.SimpleNamespace(
        config=types.SimpleNamespace(host="https://workspace.example"),
        current_user=types.SimpleNamespace(me=lambda: user),
        files=types.SimpleNamespace(upload=lambda path, data, overwrite: None),
    )
    monkeypatch.setattr(conn, "client", lambda: client)
    monkeypatch.setattr(conn, "warehouse", lambda: sql)
    return sql


def no_workspace():
    pytest.fail("nothing may reach Databricks")


def smoke_row(params):
    return [[cli.SMOKE_ID, "0.9"]] if params.get("run_id") == cli.SMOKE_ID else []


# ----------------------------------------------------------------------------- smoke
def test_smoke_reads_its_row_back_then_deletes_it(workspace, capsys):
    workspace.answers["SELECT run_id, accuracy"] = smoke_row
    assert cli.main(["smoke", "--no-mlflow"]) == 0
    out = capsys.readouterr().out
    assert "ok    table read back" in out and "smoke-test, accuracy 0.9" in out and "All steps passed." in out
    statements = [s for s, _ in workspace.calls]
    last_insert = max(i for i, s in enumerate(statements) if s.startswith("INSERT"))
    after = workspace.calls[last_insert + 1 :]
    deleted = {s.split()[2] for s, p in after if s.startswith("DELETE FROM") and p == {"run_id": cli.SMOKE_ID}}
    assert deleted == {conn.table(name) for name in sink.RUN_TABLES}  # no made-up card stays in report_cards


def test_smoke_fails_when_the_row_does_not_come_back(workspace, capsys):
    assert cli.main(["smoke", "--no-mlflow"]) == 1  # the fake warehouse returns no rows
    out = capsys.readouterr().out
    assert "FAIL  table read back" in out and "Some steps failed" in out
    assert "ok    clean up" in out  # the rows are deleted all the same


def test_smoke_deletes_its_mlflow_run(workspace, mlflow, capsys):
    workspace.answers["SELECT run_id, accuracy"] = smoke_row
    mlflow.runs.append({"id": "old", "name": "from an earlier smoke test", "tags": {"sensorium_run_id": cli.SMOKE_ID},
                        "deleted": False})
    assert cli.main(["smoke"]) == 0, capsys.readouterr().out
    assert any(r["tags"].get("sensorium_run_id") == cli.SMOKE_ID for r in mlflow.runs[1:])  # it did log one
    assert mlflow.live("sensorium_run_id", cli.SMOKE_ID) == []


# ----------------------------------------------------------------------------- MLflow on rewrite
def test_rewriting_a_run_or_a_triage_batch_replaces_its_mlflow_runs(mlflow):
    for _ in range(2):
        sink.write_run(RECORD, FakeSQL(), upload=False)
    assert len(mlflow.live("sensorium_run_id", "run1")) == len(RECORD["cards"])
    record = json.loads(TRIAGE_RECORDS[0].read_text())
    for _ in range(2):
        sink.write_triage(record, FakeSQL())
    assert len(mlflow.live("sensorium_triage_id", record["id"])) == 1


def test_delete_run_removes_rows_and_mlflow_runs(mlflow):
    sink.write_run(RECORD, FakeSQL(), upload=False)
    sql = FakeSQL()
    assert sink.delete_run("run1", sql) == {"mlflow": 2}
    assert [s.split()[2] for s, _ in sql.calls] == [conn.table(n) for n in sink.RUN_TABLES]
    assert mlflow.live("sensorium_run_id", "run1") == []
    assert sink.delete_run("run1", FakeSQL(), mlflow_runs=False) == {}


# ----------------------------------------------------------------------------- setup on older tables
def describe(columns):
    rows = [[c, t, None] for c, t in columns.items()]
    return rows + [["", "", ""], ["# Partition Information", "", ""], ["# col_name", "data_type", "comment"]]


def test_setup_adds_the_columns_an_older_table_lacks():
    old = {name: sink.column_types(name) for name in sink.DDL}
    del old["triage_batches"]["question"]  # the triage tables as d71318b created them
    del old["triage_clips"]["reply"]
    sql = AnsweringSQL({f"DESCRIBE TABLE {conn.table(n)}": describe(cols) for n, cols in old.items()})
    sink.setup(sql)
    alters = [s for s, _ in sql.calls if s.startswith("ALTER")]
    assert alters == [f"ALTER TABLE {conn.table('triage_batches')} ADD COLUMNS (question STRING)",
                      f"ALTER TABLE {conn.table('triage_clips')} ADD COLUMNS (reply STRING)"]


def test_setup_leaves_current_tables_and_unreadable_ones_alone():
    sql = AnsweringSQL({f"DESCRIBE TABLE {conn.table(n)}": describe(sink.column_types(n)) for n in sink.DDL})
    sink.setup(sql)
    assert not any(s.startswith("ALTER") for s, _ in sql.calls)
    sql = FakeSQL()  # DESCRIBE gives nothing: nothing to compare with
    sink.setup(sql)
    assert not any(s.startswith("ALTER") for s, _ in sql.calls)


# ----------------------------------------------------------------------------- Warehouse polling
class Statements:
    """statement_execution whose statement is still running for `polls` polls, then ends in `final`."""

    def __init__(self, polls, final="SUCCEEDED"):
        from databricks.sdk.service.sql import StatementState

        self.State = StatementState
        self.left, self.final = polls, final
        self.gets, self.cancelled = 0, []

    def _resp(self, state):
        return types.SimpleNamespace(statement_id="st1", status=types.SimpleNamespace(state=self.State(state), error=None),
                                     result=types.SimpleNamespace(data_array=[["ok"]]))

    def execute_statement(self, **kwargs):
        return self._resp("RUNNING" if self.left else self.final)

    def get_statement(self, statement_id):
        self.gets += 1
        self.left -= 1
        return self._resp("RUNNING" if self.left > 0 else self.final)

    def cancel_execution(self, statement_id):
        self.cancelled.append(statement_id)


def test_warehouse_waits_longer_between_polls(monkeypatch):
    pauses = []
    monkeypatch.setattr(conn, "_sleep", pauses.append)
    api = Statements(polls=7)
    wh = conn.Warehouse(types.SimpleNamespace(statement_execution=api), "wh1")
    assert wh.run("SELECT 1") == [["ok"]]
    assert api.gets == 7 and pauses == [0.5, 1.0, 2.0, 4.0, 5.0, 5.0, 5.0]


def test_warehouse_cancels_a_statement_past_its_deadline(monkeypatch):
    clock = iter(range(0, 10_000, 100))
    monkeypatch.setattr(conn, "_clock", lambda: next(clock))
    monkeypatch.setattr(conn, "_sleep", lambda s: None)
    api = Statements(polls=10**6)  # never ends
    wh = conn.Warehouse(types.SimpleNamespace(statement_execution=api), "wh1", deadline=600)
    with pytest.raises(RuntimeError, match="wh1 was still RUNNING after 600 s, so it was cancelled"):
        wh.run("SELECT 1")
    assert api.cancelled == ["st1"] and api.gets < 10


# ----------------------------------------------------------------------------- .env
def test_conn_loads_the_dotenv_without_the_cli(tmp_path):
    """Library use (conn.client() without the CLI) sees DATABRICKS_HOST from .env, as the harness does."""
    (tmp_path / ".env").write_text("DATABRICKS_HOST=https://dotenv-probe.example\n")
    env = {k: v for k, v in os.environ.items() if not k.startswith("DATABRICKS_")}
    env["PYTHONPATH"] = str(REPO)
    code = ("import os; from integrations.databricks import conn; "
            "print(os.environ.get('DATABRICKS_HOST') == 'https://dotenv-probe.example')")
    out = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "True"


# ----------------------------------------------------------------------------- CLI arguments and files
@pytest.mark.parametrize("argv, says", [
    (["triage", "x.json"], "put them after --from"),
    (["smoke", "x.json"], "smoke takes no FILE arguments"),
    (["serve", "x.json"], "serve takes no FILE arguments"),
    (["setup", "x.json"], "setup takes no FILE arguments"),
    (["backfill", "--from", "x.json"], "--from goes with triage"),
    (["triage", "--from", "x.json", "--model", "fake"], "--model shape a new batch"),
    (["triage", "--from", "x.json", "--question", "chat"], "--question shape a new batch"),
    (["triage", "--question", "all", "--save", "x.json"], "--save writes one record"),
    (["triage", "--limit", "0"], "--limit must be at least 1"),
])
def test_arguments_a_command_would_ignore_are_refused(monkeypatch, capsys, argv, says):
    monkeypatch.setattr(conn, "warehouse", no_workspace)
    with pytest.raises(SystemExit) as stop:
        cli.main(argv)
    assert stop.value.code == 2 and says in capsys.readouterr().err


def test_backfill_names_files_that_are_missing_or_not_run_records(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(conn, "warehouse", no_workspace)
    garbage = tmp_path / "garbage.json"
    garbage.write_text("{not json")
    files = [str(ARTIFACTS / "tool_openai-chat_closed.json"), str(tmp_path / "missing.json"), str(garbage)]
    assert cli.main(["backfill", "--no-mlflow", "--no-upload", *files]) == 1
    err = capsys.readouterr().err
    assert f"{files[0]}: not a run record" in err and f"{files[1]}: no such file" in err
    assert f"{files[2]}: not a run record" in err and "Nothing was written." in err


def test_backfill_writes_saved_run_records(workspace, capsys):
    files = sorted(str(p) for p in ARTIFACTS.glob("run_*.json"))
    assert files
    assert cli.main(["backfill", "--no-mlflow", "--no-upload", *files]) == 0
    assert capsys.readouterr().out.count("  ok    ") == len(files)


def test_triage_from_refuses_a_run_record(monkeypatch, capsys):
    monkeypatch.setattr(conn, "warehouse", no_workspace)
    run_record = str(ARTIFACTS / "run_qwen3.5-0.8b_gtzan_closed.json")
    assert cli.main(["triage", "--no-mlflow", "--from", run_record]) == 1
    assert f"{run_record}: not a triage record" in capsys.readouterr().err


def test_triage_from_a_glob_writes_every_record_and_names_each_model(workspace, capsys):
    assert len(TRIAGE_RECORDS) > 1
    paths = [str(p) for p in TRIAGE_RECORDS]
    assert cli.main(["triage", "--no-mlflow", "--from", *paths]) == 0  # `--from socket_pilot/artifacts/triage_*.json`
    out = capsys.readouterr().out
    assert out.count("Wrote triage ") == len(paths)
    compare = out.split("more than one model or batch")[1]
    models = {json.loads(p.read_text())["model"] for p in TRIAGE_RECORDS}
    assert all(m in compare for m in models) and "esc50 fold 4 (dev)" in compare and "esc50 fold 5" in compare
