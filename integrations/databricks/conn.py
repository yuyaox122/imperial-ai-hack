"""Connection to a Databricks workspace: auth, the SQL warehouse, the Unity Catalog names.

Auth is whatever the Databricks SDK finds: DATABRICKS_HOST and DATABRICKS_TOKEN, or a profile in
~/.databrickscfg (DATABRICKS_CONFIG_PROFILE picks one, DEFAULT otherwise). Install the client with
`uv sync --group databricks`.

Names (environment overrides):
  SENSE_DBX_CATALOG     catalog, default "workspace" (the one Free Edition gives every user)
  SENSE_DBX_SCHEMA      schema, default "sensorium"
  SENSE_DBX_WAREHOUSE   SQL warehouse id; default: the first warehouse in the workspace
  SENSE_DBX_EXPERIMENT  MLflow experiment path; default /Users/<you>/sensorium
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Protocol

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,127}")


def _ident(name: str, what: str) -> str:
    if not _IDENT.fullmatch(name):
        raise ValueError(f"{what} {name!r} is not a plain identifier (letters, digits, '_'). Fix {what.upper()}.")
    return name


def catalog() -> str:
    return _ident(os.environ.get("SENSE_DBX_CATALOG", "workspace"), "SENSE_DBX_CATALOG")


def schema() -> str:
    return _ident(os.environ.get("SENSE_DBX_SCHEMA", "sensorium"), "SENSE_DBX_SCHEMA")


def table(name: str) -> str:
    return f"{catalog()}.{schema()}.{_ident(name, 'table')}"


def volume_path(volume: str, *parts: str) -> str:
    return "/".join(["/Volumes", catalog(), schema(), _ident(volume, "volume"), *parts])


class SQL(Protocol):
    """What the sink needs from a warehouse; tests pass a fake."""

    def run(self, statement: str, params: dict[str, Any] | None = None) -> list[list[Any]]: ...


@dataclass
class Warehouse:
    """Runs SQL on a Databricks SQL warehouse through the Statement Execution API."""

    client: Any
    warehouse_id: str
    timeout: str = "50s"
    log: list[str] = field(default_factory=list)

    def run(self, statement: str, params: dict[str, Any] | None = None) -> list[list[Any]]:
        from databricks.sdk.service.sql import StatementParameterListItem, StatementState

        items = [_param(StatementParameterListItem, k, v) for k, v in (params or {}).items()]
        resp = self.client.statement_execution.execute_statement(
            statement=statement, warehouse_id=self.warehouse_id, parameters=items or None, wait_timeout=self.timeout
        )
        while resp.status and resp.status.state in (StatementState.PENDING, StatementState.RUNNING):
            resp = self.client.statement_execution.get_statement(resp.statement_id)
        state = resp.status.state if resp.status else None
        if state != StatementState.SUCCEEDED:
            err = resp.status.error.message if resp.status and resp.status.error else state
            raise RuntimeError(f"Databricks SQL failed ({err}): {statement.splitlines()[0][:120]}")
        self.log.append(statement.split("(")[0].strip()[:80])
        return (resp.result.data_array if resp.result else None) or []


def _param(cls: Any, name: str, value: Any) -> Any:
    if value is None:
        return cls(name=name, value=None, type="STRING")
    if isinstance(value, bool):
        return cls(name=name, value="true" if value else "false", type="BOOLEAN")
    if isinstance(value, int):
        return cls(name=name, value=str(value), type="BIGINT")
    if isinstance(value, float):
        return cls(name=name, value=repr(value), type="DOUBLE")
    return cls(name=name, value=str(value), type="STRING")


@lru_cache(maxsize=1)
def client() -> Any:
    try:
        from databricks.sdk import WorkspaceClient
    except ImportError:
        raise RuntimeError("The Databricks client is not installed. Run: uv sync --group databricks") from None
    return WorkspaceClient()


def warehouse() -> Warehouse:
    w = client()
    wid = os.environ.get("SENSE_DBX_WAREHOUSE")
    if not wid:
        found = list(w.warehouses.list())
        if not found:
            raise RuntimeError("The workspace has no SQL warehouse. Create one (SQL Warehouses) or set SENSE_DBX_WAREHOUSE.")
        wid = found[0].id
    return Warehouse(w, wid)


def experiment_path() -> str:
    path = os.environ.get("SENSE_DBX_EXPERIMENT")
    if path:
        return path
    return f"/Users/{client().current_user.me().user_name}/sensorium"
