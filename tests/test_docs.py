"""The documentation agrees with the code and with the recorded runs it cites.

These checks read files only: README.md, docs/, demo/script.md, socket_pilot/README.md, the HTTP contract in the
server/app.py docstring, the scripts/setup_machine.py guide and socket_pilot/artifacts/*.json.

Run: SENSE_FAKE=1 .venv/bin/python -m pytest -q tests/test_docs.py
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
ARTIFACTS = REPO / "socket_pilot" / "artifacts"
README = REPO / "README.md"
DOCS = [README, REPO / "docs" / "devpost.md", REPO / "docs" / "pitch.md", REPO / "demo" / "script.md",
        REPO / "socket_pilot" / "README.md"]
CODE_DIRS = ["sense", "agent", "server", "integrations", "scripts", "eval"]
# Every environment variable the code reads is written as one of these quoted names.
ENV_NAME = re.compile(r"""["'](SENSE_[A-Z0-9_]+|LINKUP_API_KEY|OPENAI_API_KEY|DATABRICKS_[A-Z_]+)["']""")
RECORD_PREFIXES = ("run_", "triage_", "tool_", "free_text_", "music_", "results_", "textplug_", "cascade_",
                   "transplant_")


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def docstring(path: Path) -> str:
    return ast.get_docstring(ast.parse(read(path))) or ""


def code_files() -> list[Path]:
    return [path for name in CODE_DIRS for path in sorted((REPO / name).rglob("*.py"))]


def records() -> list[Path]:
    return sorted(ARTIFACTS.glob("*.json"))


def test_the_settings_table_names_every_environment_variable_the_code_reads():
    names = {match for path in code_files() for match in ENV_NAME.findall(read(path))}
    assert {"SENSE_FAKE", "LINKUP_API_KEY", "OPENAI_API_KEY"} <= names  # the scan itself works
    readme = read(README)
    missing = sorted(name for name in names if f"`{name}`" not in readme)
    assert not missing, f"README.md does not document {missing}; add them to the settings table"


def test_every_record_the_docs_name_exists():
    pattern = re.compile(r"(?<![\w*-])((?:%s)[\w.*-]*\.json)" % "|".join(map(re.escape, RECORD_PREFIXES)))
    named = {(doc.relative_to(REPO), name) for doc in DOCS + [REPO / "scripts" / "setup_machine.py"]
             for name in pattern.findall(read(doc))}
    assert len(named) > 20  # the scan itself works
    missing = sorted((str(doc), name) for doc, name in named if not list(ARTIFACTS.glob(name)))
    assert not missing, f"no file in socket_pilot/artifacts matches {missing}"


@pytest.mark.parametrize("record", records(), ids=lambda path: path.name)
def test_every_record_is_described_in_the_socket_pilot_readme(record):
    assert record.name in read(REPO / "socket_pilot" / "README.md"), (
        f"socket_pilot/README.md does not describe {record.name}; add it to the table of records"
    )


@pytest.mark.parametrize("record", sorted(ARTIFACTS.glob("run_*.json")), ids=lambda path: path.name)
def test_the_readme_quotes_the_card_each_recorded_run_installed(record):
    run = json.loads(read(record))
    # A refused run (installed None) is quoted by its best card: the score that was not enough.
    card = next((card for card in run["cards"] if card["bridge"] == run["installed"]), None) or max(
        run["cards"], key=lambda card: card["accuracy"])
    assert f"{card['accuracy'] * 100:.1f}%" in read(README), (
        f"README.md does not quote the {card['bridge']} card that {record.name} installed"
    )


def test_the_setup_guide_compares_against_the_recorded_gtzan_card():
    run = json.loads(read(ARTIFACTS / "run_qwen3.5-0.8b_gtzan_closed.json"))
    card = next(card for card in run["cards"] if card["bridge"] == "vector:gtzan")
    guide = docstring(REPO / "scripts" / "setup_machine.py")
    found = re.search(r"vector:gtzan card with the ([\d.]+)%", guide)
    assert found, "step 7 of the setup guide no longer names a reference accuracy"
    assert float(found.group(1)) == round(card["accuracy"] * 100, 1)
    assert "run_qwen3.5-0.8b_gtzan_closed.json" in guide


def _api_routes() -> list[tuple[str, str]]:
    from server.app import app

    out = []
    for route in app.routes:
        path = getattr(route, "path", "")
        for method in sorted(getattr(route, "methods", None) or ()):
            if path.startswith("/api/") and method in ("GET", "POST", "DELETE"):
                out.append((method, path))
    return out


def _route_pattern(method: str, path: str) -> re.Pattern:
    parts = re.split(r"\{[^}]+\}", path)
    body = r"<[^>]+>".join(re.escape(part) for part in parts)
    return re.compile(rf"\b{method}\s+{body}(?![\w/<])")


def test_the_http_contract_lists_every_api_route():
    routes = _api_routes()
    assert ("POST", "/api/auto") in routes  # the scan itself works
    contract = docstring(REPO / "server" / "app.py")
    missing = [f"{method} {path}" for method, path in routes if not _route_pattern(method, path).search(contract)]
    assert not missing, f"the server/app.py docstring (the HTTP contract) does not list {missing}"


def test_the_docs_name_only_routes_that_exist():
    routes = _api_routes()
    named = set()
    for doc in DOCS:
        for method, path in re.findall(r"\b(GET|POST|DELETE)\s+`?(/api/[\w/<>-]*\w>?)", read(doc)):
            named.add((doc.relative_to(REPO), method, path))
    assert named  # the scan itself works
    unknown = sorted((str(doc), f"{method} {path}") for doc, method, path in named
                     if not any(m == method and _route_pattern(m, p).fullmatch(f"{method} {path}")
                                for m, p in routes))
    assert not unknown, f"the docs name routes the server does not have: {unknown}"


def test_every_path_in_the_readme_layout_table_exists():
    readme = read(README)
    table = readme.split("## Layout", 1)[1].split("\n\n", 2)[1]
    paths = [path for row in table.splitlines()[2:] for path in re.findall(r"`([^`]+)`", row.split("|")[1])]
    assert len(paths) > 10  # the scan itself works
    missing = [path for path in paths if not (REPO / path.rstrip("/")).exists()]
    assert not missing, f"README.md's Layout table names paths that do not exist: {missing}"
