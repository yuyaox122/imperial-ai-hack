"""End to end in fake mode, nothing mocked: HTTP API -> agent.jobs -> agent.loop -> sense.* -> chat.

Run: SENSE_FAKE=1 .venv/bin/python -m pytest -q tests/test_integration.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from agent import jobs
from agent import planner as planner_mod
from sense import config
from server import app as server_app


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("SENSE_FAKE", "1")
    monkeypatch.setenv("SENSE_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(jobs, "_jobs", {})
    monkeypatch.setattr(jobs, "_latest_id", None)
    monkeypatch.setattr(planner_mod, "_preferred", None)  # the rule planner, even if an LLM planner is registered
    server_app._sample_paths.clear()
    assert config.fake()
    return TestClient(server_app.app)


def run_job(client: TestClient, scope: str) -> dict:
    r = client.post("/api/fit", json={"model": "fake", "dataset": "gtzan", "modality": "audio", "scope": scope})
    assert r.status_code == 200, r.text
    job_id = r.json()["job_id"]
    for _ in range(400):
        state = client.get(f"/api/jobs/{job_id}").json()
        if state["status"] != "running":
            json.dumps(state)  # everything the interface receives must be plain JSON
            return state
        time.sleep(0.02)
    raise AssertionError("the agent run did not finish within 8 seconds")


def test_closed_scope_installs_the_vector_bridge_and_chats(client):
    state = run_job(client, "closed")
    assert state["status"] == "done", state["error"]
    assert state["installed"] == "vector:gtzan"
    steps = [e["step"] for e in state["events"]]
    for step in ("inspect", "plan", "fit", "verify", "decide", "install", "done"):
        assert step in steps
    verify_cards = [e["data"]["card"] for e in state["events"] if e["step"] == "verify" and e["status"] == "ok"]
    assert {c["bridge"] for c in verify_cards} == {"vector:gtzan", "text:gtzan"}
    assert all(c["reason"] for c in state["cards"])

    senses = client.get("/api/senses", params={"model": "fake"}).json()
    assert [s["bridge"] for s in senses] == ["vector:gtzan"]
    assert senses[0]["card"]["verdict"] == "installed"

    samples = client.get("/api/samples", params={"dataset": "gtzan"}).json()
    rock = next(s for s in samples if s["label"] == "rock")
    assert client.get(f"/api/samples/{rock['id']}").status_code == 200

    r = client.post("/api/chat", data={"model": "fake", "message": "Which genre?", "sample_ids": [rock["id"]]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert "rock" in body["reply"]
    assert body["used"][0] | {"detail": ""} == {"bridge": "vector:gtzan", "kind": "vector", "source": rock["name"], "detail": ""}


def test_open_scope_rejects_the_vector_bridge_and_installs_captions(client):
    state = run_job(client, "open")
    assert state["status"] == "done", state["error"]
    assert state["installed"] == "text:captions"
    vector = next(c for c in state["cards"] if c["bridge"] == "vector:gtzan")
    assert vector["verdict"] == "rejected"
    assert "unseen_classes" in vector["controls"]
    assert "not fitted on" in vector["reason"]


def test_upload_is_reported_under_its_own_name(client):
    assert run_job(client, "closed")["installed"] == "vector:gtzan"
    samples = client.get("/api/samples", params={"dataset": "gtzan"}).json()
    jazz = next(s for s in samples if s["label"] == "jazz")
    audio = client.get(f"/api/samples/{jazz['id']}").content
    r = client.post("/api/chat", data={"model": "fake", "message": "What is this?"},
                    files=[("files", ("my track.wav", audio, "audio/wav"))])
    assert r.status_code == 200, r.text
    used = r.json()["used"]
    assert used[0]["source"] == "my track.wav"
    assert "jazz" in r.json()["reply"]


def test_files_that_cannot_reach_the_model_are_reported(client):
    samples = client.get("/api/samples", params={"dataset": "gtzan"}).json()
    r = client.post("/api/chat", data={"model": "fake", "message": "Hello", "sample_ids": [samples[0]["id"]]},
                    files=[("files", ("notes.txt", b"not audio", "text/plain"))])
    assert r.status_code == 200, r.text
    upload, sample = r.json()["used"]  # uploads are listed before samples
    assert upload["kind"] == "none" and "only audio files" in upload["detail"]
    assert sample["kind"] == "none" and "has no installed audio sense" in sample["detail"]
    assert "No audio reached me" in r.json()["reply"]


def test_second_run_while_one_is_running_is_refused(client, monkeypatch):
    from agent import loop

    release = __import__("threading").Event()

    def slow(request, emit, planner=None):
        release.wait(5)
        return [], None

    monkeypatch.setattr(loop, "run_fit", slow)
    body = {"model": "fake", "dataset": "gtzan", "scope": "closed"}
    assert client.post("/api/fit", json=body).status_code == 200
    second = client.post("/api/fit", json=body)
    release.set()
    assert second.status_code == 409
    assert "already in progress" in second.json()["detail"]


def test_index_serves_the_interface(client):
    page = client.get("/")
    assert page.status_code == 200 and "/static/app.js" in page.text
    assert client.get("/static/app.js").status_code == 200
    assert Path(server_app.WEB / "style.css").is_file()
