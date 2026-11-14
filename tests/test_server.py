"""HTTP API tests with fake models, datasets, registry and jobs."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from agent import jobs
from sense import datasets, models, registry
from sense.contracts import FitRequest, JobState, ModelInfo, SenseInput
from server import app as server_app

INFO = ModelInfo(name="m1", label="Model one", native=["text"], open_weights=True)


class FakeBridge:
    kind = "vector"
    modality = "audio"
    model_name = "m1"
    name = "vector:gtzan"

    def prepare(self, path: Path) -> SenseInput:
        assert path.is_file()  # the upload must exist while the bridge reads it
        self.seen = path
        return SenseInput(kind="vector", modality="audio", source="", detail="fake embedding")


class FakeModel:
    info = INFO

    def chat(self, message, inputs, max_new_tokens=96):
        return f"{message}|{len(inputs)}"


@pytest.fixture
def env(monkeypatch, tmp_path):
    bridge = FakeBridge()
    sample = tmp_path / "rock.wav"
    sample.write_bytes(b"RIFFfake")
    state = {"jobs": {}, "removed": [], "bridges": [bridge]}

    def start(request: FitRequest) -> str:
        if state.get("busy"):
            raise RuntimeError("an agent run is already in progress")
        state["jobs"]["j1"] = JobState(id="j1", request=request)
        return "j1"

    monkeypatch.setattr(models, "available_models", lambda: [INFO])
    monkeypatch.setattr(models, "load_model", lambda name: FakeModel())
    monkeypatch.setattr(datasets, "available_sets",
                        lambda: [{"name": "gtzan", "title": "Music", "modality": "audio", "labels": ["rock"], "n": 1}])
    monkeypatch.setattr(datasets, "sample_files", lambda name, per_label=1: [{"path": str(sample), "label": "rock"}])
    monkeypatch.setattr(registry, "installed", lambda name: [{"bridge": "vector:gtzan", "kind": "vector",
                                                              "modality": "audio", "dataset": "gtzan", "card": {}}])
    monkeypatch.setattr(registry, "get_bridges", lambda name: state["bridges"])
    monkeypatch.setattr(registry, "uninstall", lambda name, modality: state["removed"].append((name, modality)) or True)
    monkeypatch.setattr(jobs, "start", start)
    monkeypatch.setattr(jobs, "get", lambda job_id: state["jobs"].get(job_id))
    server_app._sample_paths.clear()
    return TestClient(server_app.app), state, bridge


def test_models_and_datasets(env):
    client, _, _ = env
    assert client.get("/api/models").json()[0]["name"] == "m1"
    assert client.get("/api/datasets").json()[0]["name"] == "gtzan"


def test_senses_list_and_delete(env):
    client, state, _ = env
    assert client.get("/api/senses", params={"model": "m1"}).json()[0]["bridge"] == "vector:gtzan"
    assert client.delete("/api/senses", params={"model": "m1", "modality": "audio"}).json() == {"removed": True}
    assert state["removed"] == [("m1", "audio")]
    assert client.get("/api/senses", params={"model": "nope"}).status_code == 400


def test_fit_and_job(env):
    client, _, _ = env
    r = client.post("/api/fit", json={"model": "m1", "dataset": "gtzan"})
    assert r.json() == {"job_id": "j1"}
    job = client.get("/api/jobs/j1").json()
    assert job["status"] == "running" and job["request"]["scope"] == "closed"


def test_fit_errors(env):
    client, state, _ = env
    assert client.post("/api/fit", json={"model": "x", "dataset": "gtzan"}).status_code == 400
    assert client.post("/api/fit", json={"model": "m1", "dataset": "x"}).status_code == 400
    assert client.post("/api/fit", json={"model": "m1", "dataset": "gtzan", "scope": "wide"}).status_code == 400
    state["busy"] = True
    r = client.post("/api/fit", json={"model": "m1", "dataset": "gtzan"})
    assert r.status_code == 409 and "already in progress" in r.json()["detail"]
    assert client.get("/api/jobs/missing").status_code == 400


def test_samples(env):
    client, _, _ = env
    listing = client.get("/api/samples", params={"dataset": "gtzan"}).json()
    assert listing[0]["label"] == "rock" and listing[0]["name"] == "rock.wav"
    sid = listing[0]["id"]
    assert sid == client.get("/api/samples", params={"dataset": "gtzan"}).json()[0]["id"]  # stable
    assert client.get(f"/api/samples/{sid}").content == b"RIFFfake"
    assert client.get("/api/samples/deadbeef").status_code == 404
    assert client.get("/api/samples", params={"dataset": "x"}).status_code == 400


def test_sample_served_after_restart(env):
    client, _, _ = env
    sid = client.get("/api/samples", params={"dataset": "gtzan"}).json()[0]["id"]
    server_app._sample_paths.clear()
    assert client.get(f"/api/samples/{sid}").status_code == 200


def test_chat_with_upload(env):
    client, _, bridge = env
    r = client.post("/api/chat", data={"model": "m1", "message": "hi"},
                    files=[("files", ("song.WAV", b"abc", "audio/wav"))])
    assert r.status_code == 200
    body = r.json()
    assert body["reply"] == "hi|1"
    assert body["used"] == [{"bridge": "vector:gtzan", "kind": "vector", "source": "song.WAV",
                             "detail": "fake embedding"}]
    assert not bridge.seen.exists()  # temporary file deleted


def test_chat_with_sample_and_unsupported_file(env):
    client, _, _ = env
    sid = client.get("/api/samples", params={"dataset": "gtzan"}).json()[0]["id"]
    r = client.post("/api/chat", data={"model": "m1", "message": "go", "sample_ids": [sid]},
                    files=[("files", ("notes.txt", b"x", "text/plain"))])
    body = r.json()
    assert body["reply"] == "go|1"
    assert [u["kind"] for u in body["used"]] == ["none", "vector"]
    assert "notes.txt" in body["used"][0]["detail"]


def test_chat_without_installed_sense(env):
    client, state, _ = env
    state["bridges"] = []
    r = client.post("/api/chat", data={"model": "m1", "message": "hi"},
                    files=[("files", ("a.wav", b"abc", "audio/wav"))])
    body = r.json()
    assert body["reply"] == "hi|0"
    assert body["used"][0]["kind"] == "none" and "Run the agent" in body["used"][0]["detail"]


def test_chat_errors(env):
    client, _, _ = env
    assert client.post("/api/chat", data={"model": "x", "message": "hi"}).status_code == 400
    r = client.post("/api/chat", data={"model": "m1", "message": "hi", "sample_ids": ["nope"]})
    assert r.status_code == 400


def test_index(env, monkeypatch, tmp_path):
    client, _, _ = env
    monkeypatch.setattr(server_app, "WEB", tmp_path)
    assert "missing" in client.get("/").text
    (tmp_path / "index.html").write_text("<h1>hello</h1>")
    assert client.get("/").text == "<h1>hello</h1>"
