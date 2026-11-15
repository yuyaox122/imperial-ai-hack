"""tests/conftest.py keeps every test out of the developer's <repo>/.sense.

Run: SENSE_FAKE=1 .venv/bin/python -m pytest -q tests/test_conftest.py
"""

from __future__ import annotations

from sense import config, registry, runs


def test_every_test_gets_its_own_empty_sense_home(isolated_sense_home):
    assert config.home() == isolated_sense_home != config.REPO / ".sense"
    assert list(isolated_sense_home.iterdir()) == []
    assert config.sets_dir() == isolated_sense_home / "datasets"
    assert registry.sense_dir("fake", "audio").is_relative_to(isolated_sense_home)
    assert runs.runs_dir().is_relative_to(isolated_sense_home)
