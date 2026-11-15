"""Shared test setup: every test gets its own empty SENSE_HOME, so no test reads or writes the developer's
<repo>/.sense (installed senses, recorded runs, folder sets, the API answer cache, research and plan caches).

A test or a test file's own fixture may still set SENSE_HOME or SENSE_SETS_DIR; those run after this one and win.
SENSE_SETS_DIR is unset here, so folder sets default to <SENSE_HOME>/datasets.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def isolated_sense_home(monkeypatch, tmp_path_factory):
    home = tmp_path_factory.mktemp("sense-home")
    monkeypatch.setenv("SENSE_HOME", str(home))
    monkeypatch.delenv("SENSE_SETS_DIR", raising=False)
    return home
