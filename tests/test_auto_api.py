"""POST /api/auto: an agent run from one file and a goal (agent.jobs.start_auto), in fake mode.

Run: SENSE_FAKE=1 .venv/bin/python -m pytest -q tests/test_auto_api.py
"""

from __future__ import annotations

import io
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from agent import jobs
from agent import planner as planner_mod
from sense import config
from sense.specialists import FAKE_CONCEPTS, FakeSpecialist
from server import app as server_app
from server import replay

GOAL = "Make it recognise music genres"


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("SENSE_FAKE", "1")
    monkeypatch.setenv("SENSE_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("SENSE_SETS_DIR", raising=False)
    monkeypatch.setattr(jobs, "_jobs", {})
    monkeypatch.setattr(jobs, "_latest_id", None)
    monkeypatch.setattr(planner_mod, "_preferred", None)  # the rule planner, even if an LLM planner is registered
    monkeypatch.setattr(server_app, "_recorder", None)
    assert config.fake()
    return TestClient(server_app.app)


@pytest.fixture
def rock_clip(tmp_path) -> bytes:
    path = tmp_path / "clip.wav"
    FakeSpecialist.write_sample(FAKE_CONCEPTS.index("rock"), path, seed=1)
    return path.read_bytes()


def post_auto(client, data: dict, clip: bytes | None, name: str = "my clip.wav"):
    files = {"file": (name, io.BytesIO(clip), "audio/wav")} if clip is not None else None
    return client.post("/api/auto", data=data, files=files)


def wait_done(client, job_id: str) -> dict:
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        state = client.get(f"/api/jobs/{job_id}").json()
        if state["status"] != "running":
            return state
        time.sleep(0.02)
    raise AssertionError("the agent run did not finish within 15 s")


@pytest.fixture
def started(monkeypatch):
    """jobs.start_auto replaced by a recorder of its arguments; returns the list of calls."""
    calls = []

    def start_auto(model, file, goal, examples=None):
        calls.append({"model": model, "file": Path(file), "goal": goal, "examples": examples,
                      "exists": Path(file).is_file(), "bytes": Path(file).read_bytes()})
        return "auto-job"

    monkeypatch.setattr(jobs, "start_auto", start_auto)
    return calls


def test_the_fields_reach_start_auto_and_the_upload_is_kept(client, started, rock_clip, tmp_path):
    examples = tmp_path / "examples"
    examples.mkdir()
    r = post_auto(client, {"model": "fake", "goal": f"  {GOAL}  ", "examples": str(examples)}, rock_clip)
    assert r.status_code == 200, r.text
    assert r.json() == {"job_id": "auto-job"}
    (call,) = started
    assert call["model"] == "fake" and call["goal"] == GOAL and call["examples"] == examples
    assert call["exists"] and call["bytes"] == rock_clip
    assert call["file"].name == "my clip.wav"
    assert call["file"].is_relative_to(config.home() / server_app.UPLOADS)
    assert call["file"].is_file()  # still there after the request returned: the run reads it later


def test_examples_is_optional(client, started, rock_clip):
    assert post_auto(client, {"model": "fake", "goal": GOAL}, rock_clip).status_code == 200
    assert started[0]["examples"] is None


def test_an_unsafe_file_name_is_cleaned(client, started, rock_clip):
    assert post_auto(client, {"model": "fake", "goal": GOAL}, rock_clip, name="../../evil;name.wav").status_code == 200
    path = started[0]["file"]
    assert path.name == "evil_name.wav"
    assert path.parent.parent == config.home() / server_app.UPLOADS


@pytest.mark.parametrize(
    "data, clip, words",
    [
        ({"model": "nope", "goal": GOAL}, b"x", "Unknown model"),
        ({"model": "fake", "goal": "   "}, b"x", "goal is empty"),
        ({"model": "fake", "goal": GOAL}, None, "No file"),
        ({"model": "fake", "goal": GOAL, "examples": "/no/such/folder"}, b"x", "does not exist"),
    ],
)
def test_bad_requests_are_400_and_keep_nothing(client, started, data, clip, words):
    r = post_auto(client, data, clip)
    assert r.status_code == 400 and words in r.json()["detail"]
    assert not started
    uploads = config.home() / server_app.UPLOADS
    assert not uploads.exists() or not any(uploads.iterdir())


def test_409_while_a_run_is_going_and_the_upload_is_removed(client, monkeypatch, rock_clip):
    def busy(*args, **kwargs):
        raise RuntimeError("an agent run is already in progress")

    monkeypatch.setattr(jobs, "start_auto", busy)
    r = post_auto(client, {"model": "fake", "goal": GOAL}, rock_clip)
    assert r.status_code == 409 and "already in progress" in r.json()["detail"]
    assert not any((config.home() / server_app.UPLOADS).iterdir())


def test_409_while_a_replay_is_going(client, started, monkeypatch, rock_clip):
    monkeypatch.setattr(replay, "running", lambda: True)
    r = post_auto(client, {"model": "fake", "goal": GOAL}, rock_clip)
    assert r.status_code == 409 and "replay" in r.json()["detail"]
    assert not started
    assert not any((config.home() / server_app.UPLOADS).iterdir())


def test_end_to_end_a_file_and_a_goal_install_a_sense_and_the_run_is_recorded(client, rock_clip):
    r = post_auto(client, {"model": "fake", "goal": GOAL}, rock_clip)
    assert r.status_code == 200, r.text
    job_id = r.json()["job_id"]
    state = wait_done(client, job_id)
    assert state["status"] == "done", state["error"]
    assert state["installed"] == "vector:gtzan" and state["request"]["dataset"] == "gtzan"
    steps = [e["step"] for e in state["events"]]
    order = ["detect", "research", "data", "inspect", "plan", "fit", "verify", "claims", "decide", "install", "done"]
    assert [s for s in dict.fromkeys(steps)] == order
    senses = client.get("/api/senses", params={"model": "fake"}).json()
    assert [s["bridge"] for s in senses] == ["vector:gtzan"]
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:  # the watcher records the finished run
        recorded = [run for run in client.get("/api/runs").json() if run["id"] == job_id]
        if recorded:
            break
        time.sleep(0.05)
    assert recorded and recorded[0]["installed"] == "vector:gtzan" and recorded[0]["has_sense"]


def test_end_to_end_a_running_auto_job_can_be_cancelled(client, monkeypatch, rock_clip):
    from agent import loop

    entered, release = [], []
    original = loop.run_fit

    def slow_fit(*args, **kwargs):  # hold the run inside its fit step until the test cancels it
        entered.append(True)
        deadline = time.monotonic() + 5
        while not release and time.monotonic() < deadline:
            time.sleep(0.01)
        return original(*args, **kwargs)

    monkeypatch.setattr(loop, "run_fit", slow_fit)
    job_id = post_auto(client, {"model": "fake", "goal": GOAL}, rock_clip).json()["job_id"]
    deadline = time.monotonic() + 5
    while not entered and time.monotonic() < deadline:
        time.sleep(0.01)
    assert client.delete(f"/api/jobs/{job_id}").json() == {"cancelled": True}
    release.append(True)
    state = wait_done(client, job_id)
    assert state["status"] == "failed" and state["error"] == "Cancelled."
    assert state["events"][-1]["step"] == "error"
