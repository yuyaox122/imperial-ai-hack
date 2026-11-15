"""Files sent to the server: how chat gives each one to the model (as it is, through a sense, or not at all, with
the reason), decided from the file's bytes as POST /api/auto decides it; and how uploads are saved, limited and
removed. Model doubles first, then end to end in fake mode.

No weights, no downloads and no network.
Run: SENSE_FAKE=1 .venv/bin/python -m pytest -q tests/test_server_files.py
"""

from __future__ import annotations

import io
import json
import os
import struct
import tempfile
import time
import zlib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from agent import jobs
from agent import planner as planner_mod
from sense import config, models, registry
from sense.contracts import ModelInfo, SenseInput
from sense.specialists import FAKE_CONCEPTS, FakeSpecialist
from server import app as server_app

API = ModelInfo(name="api", label="API model", native=["text", "image"], open_weights=False, tools=True)
LOCAL = ModelInfo(name="local", label="Local model", native=["text", "image"], open_weights=True)
PLAIN = ModelInfo(name="plain", label="Plain model", native=["text"], open_weights=True)
ENDPOINTS = ["/api/chat", "/api/chat/stream"]


def png() -> bytes:
    """A valid 1x1 PNG."""
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(b"\x00\xff\x00\x00")) + chunk(b"IEND", b""))


WAV = b"RIFF\x24\x00\x00\x00WAVEfmt " + bytes(28)  # the magic bytes of a WAV file; the bridge double reads nothing


class AudioBridge:
    kind = "vector"
    modality = "audio"
    name = "vector:gtzan"

    def prepare(self, path: Path) -> SenseInput:
        assert path.is_file()
        return SenseInput(kind="vector", modality="audio", detail="fake embedding")


class Recorder:
    """A model double that remembers what each chat call was given."""

    def __init__(self, info: ModelInfo):
        self.info = info
        self.calls: list[tuple[str, list[SenseInput]]] = []

    def chat(self, message, inputs, max_new_tokens=96):
        self.calls.append((message, list(inputs)))
        return "ok"


@pytest.fixture
def doubles(monkeypatch, tmp_path):
    monkeypatch.setenv("SENSE_HOME", str(tmp_path / "home"))
    state = {"models": {info.name: Recorder(info) for info in (API, LOCAL, PLAIN)}, "bridge_calls": 0, "temps": []}

    def get_bridges(name):
        state["bridge_calls"] += 1
        return [AudioBridge()]

    real_mkstemp = tempfile.mkstemp

    def mkstemp(*args, **kwargs):
        fd, path = real_mkstemp(*args, **kwargs)
        state["temps"].append(Path(path))
        return fd, path

    monkeypatch.setattr(models, "available_models", lambda: [API, LOCAL, PLAIN])
    monkeypatch.setattr(models, "load_model", lambda name: state["models"][name])
    monkeypatch.setattr(jobs, "latest", lambda: None)
    monkeypatch.setattr(registry, "get_bridges", get_bridges)
    monkeypatch.setattr(server_app.tempfile, "mkstemp", mkstemp)
    server_app._sample_paths.clear()
    return TestClient(server_app.app), state


def chat(client, endpoint: str, model: str, files: list[tuple[str, bytes]], message: str = "What is this?"):
    """The `used` list of a chat answer, from either endpoint; asserts that it succeeded."""
    r = client.post(endpoint, data={"model": model, "message": message},
                    files=[("files", (name, data, "application/octet-stream")) for name, data in files])
    assert r.status_code == 200, r.text
    if endpoint == "/api/chat":
        return r.json()["used"]
    last = r.text.strip().split("\n\n")[-1]
    assert last.startswith("event: done\n")
    return json.loads(last.split("\n", 1)[1][len("data: "):])["used"]


# ----------------------------------------------------------------------------- what chat gives the model
@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_a_text_file_is_given_to_the_model_with_the_message_and_needs_no_sense(doubles, endpoint):
    client, state = doubles
    used = chat(client, endpoint, "local", [("notes.txt", "The meeting moved to noon. Café at 1.".encode())])
    assert used == [{"bridge": "", "kind": "native", "source": "notes.txt",
                     "detail": "its text was given to the model with the message"}]
    message, inputs = state["models"]["local"].calls[-1]
    assert message.startswith("What is this?\n\n") and 'The attached file "notes.txt"' in message
    assert "The meeting moved to noon. Café at 1." in message and inputs == []
    assert state["bridge_calls"] == 0  # a broken sense must not stop a chat that needs none


def test_a_long_text_file_is_cut_and_utf16_is_read(doubles):
    client, state = doubles
    long = "x" * (server_app.TEXT_FILE_CHARS + 50)
    utf16 = "﻿hello in UTF-16".encode("utf-16-le")
    used = chat(client, "/api/chat", "plain", [("long.txt", long.encode()), ("wide.txt", utf16)])
    assert used[0]["kind"] == "native" and f"first {server_app.TEXT_FILE_CHARS} characters" in used[0]["detail"]
    assert used[1]["kind"] == "native"
    message, _ = state["models"]["plain"].calls[-1]
    assert "x" * server_app.TEXT_FILE_CHARS in message and "x" * (server_app.TEXT_FILE_CHARS + 1) not in message
    assert "only its start is given" in message and "hello in UTF-16" in message


@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_an_image_is_shown_to_a_model_that_takes_images(doubles, endpoint):
    client, state = doubles
    picture = png()
    used = chat(client, endpoint, "api", [("photo.png", picture)])
    assert used == [{"bridge": "", "kind": "native", "source": "photo.png", "detail": "shown to the model as an image"}]
    message, inputs = state["models"]["api"].calls[-1]
    assert message == "What is this?"
    (image,) = inputs
    assert (image.kind, image.modality, image.image, image.source) == ("image", "image", picture, "photo.png")
    assert state["bridge_calls"] == 0


def test_a_model_that_takes_no_images_is_told_why_it_was_not_shown_one(doubles):
    client, state = doubles
    local = chat(client, "/api/chat", "local", [("photo.png", png())])
    assert local[0]["kind"] == "none" and "image inputs are wired up for API models only" in local[0]["detail"]
    plain = chat(client, "/api/chat", "plain", [("photo.png", png())])
    assert plain[0]["kind"] == "none" and "Plain model does not read images" in plain[0]["detail"]
    assert all(inputs == [] for _, inputs in state["models"]["local"].calls + state["models"]["plain"].calls)


@pytest.mark.parametrize("name", ["voice-memo", "clip.bin", "recording.M4A"])
def test_audio_is_found_from_its_bytes_whatever_the_file_is_called(doubles, name):
    client, state = doubles
    used = chat(client, "/api/chat", "local", [(name, WAV)])
    assert used == [{"bridge": "vector:gtzan", "kind": "vector", "source": name, "detail": "fake embedding"}]
    assert len(state["models"]["local"].calls[-1][1]) == 1


def test_bytes_decide_over_the_name(doubles):
    client, state = doubles
    used = chat(client, "/api/chat", "api", [("song.mp3", png())])  # a PNG with an audio name is an image
    assert used[0]["kind"] == "native" and state["models"]["api"].calls[-1][1][0].kind == "image"


def test_files_that_cannot_be_given_say_why(doubles):
    client, state = doubles
    used = chat(client, "/api/chat", "api", [("empty.txt", b""), ("table.csv", b"a,b,c\n1,2,3\n"),
                                             ("blob.bin", bytes(range(256)))])
    empty, table, blob = used
    assert empty["kind"] == "none" and empty["detail"] == "'empty.txt' was not shown to the model: it is empty."
    for entry in (table, blob):
        assert entry["kind"] == "none"
        assert "only text files, images (PNG or JPEG) and audio files in these formats" in entry["detail"]
    local = chat(client, "/api/chat", "local", [("blob.bin", bytes(range(256)))])[0]
    assert "only text files and audio files in these formats" in local["detail"]  # no images offered to it
    assert state["models"]["api"].calls[-1] == ("What is this?", [])


# ----------------------------------------------------------------------------- saving uploads
@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_a_long_file_name_or_extension_still_reaches_the_model(doubles, endpoint):
    client, state = doubles
    names = ["clip." + "x" * 300, "a" * 300 + ".wav", "é" * 200 + ".wav"]
    used = chat(client, endpoint, "local", [(name, WAV) for name in names])
    assert [u["kind"] for u in used] == ["vector"] * 3 and [u["source"] for u in used] == names
    assert all(len(p.name.encode()) < 64 for p in state["temps"])  # the client's name never reaches the disk
    assert not any(p.exists() for p in state["temps"])


@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_an_upload_over_the_limit_is_a_413_and_leaves_nothing(doubles, monkeypatch, endpoint):
    client, state = doubles
    monkeypatch.setattr(server_app, "MAX_UPLOAD_BYTES", 10)
    monkeypatch.setattr(server_app, "UPLOAD_CHUNK", 4)
    r = client.post(endpoint, data={"model": "local", "message": "hi"},
                    files=[("files", ("song.wav", WAV, "audio/wav"))])
    assert r.status_code == 413 and r.headers["content-type"] == "application/json"
    assert "'song.wav' is larger than" in r.json()["detail"]
    assert state["temps"] and not any(p.exists() for p in state["temps"])
    assert state["models"]["local"].calls == []


def test_a_file_the_server_cannot_save_is_a_json_500(doubles, monkeypatch):
    client, state = doubles

    def full(*args, **kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(server_app.tempfile, "mkstemp", full)
    r = client.post("/api/chat", data={"model": "local", "message": "hi"},
                    files=[("files", ("song.wav", WAV, "audio/wav"))])
    assert r.status_code == 500
    assert "Could not save 'song.wav' on the server (No space left on device)" in r.json()["detail"]


def test_safe_names_are_cut_by_bytes_and_keep_their_extension():
    assert server_app._safe_name("a" * 300 + ".wav") == "a" * (server_app.NAME_BYTES - 4) + ".wav"
    cut = server_app._safe_name("é" * 200 + ".wav")
    assert cut.endswith(".wav") and len(cut.encode()) <= server_app.NAME_BYTES
    long_ext = server_app._safe_name("clip." + "x" * 300)
    assert len(long_ext.encode()) <= server_app.NAME_BYTES and long_ext.startswith("clip.")
    assert server_app._safe_name("../../evil;name.wav") == "evil_name.wav"
    assert server_app._safe_name("." * 300) == "upload" and server_app._safe_name(None) == "upload"


# ----------------------------------------------------------------------------- end to end in fake mode
GOAL = "Make it recognise music genres"


@pytest.fixture
def fake(monkeypatch, tmp_path):
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


def auto(client, model: str, name: str, data: bytes, goal: str = GOAL) -> dict:
    """Run POST /api/auto to its end and return the finished job."""
    r = client.post("/api/auto", data={"model": model, "goal": goal}, files={"file": (name, io.BytesIO(data), "x/y")})
    assert r.status_code == 200, r.text
    job_id = r.json()["job_id"]
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        state = client.get(f"/api/jobs/{job_id}").json()
        if state["status"] != "running":
            return state
        time.sleep(0.02)
    raise AssertionError("the agent run did not finish within 15 s")


def uploads(home: Path) -> list[Path]:
    root = home / server_app.UPLOADS
    return sorted(root.iterdir()) if root.is_dir() else []


def wait_for(condition, seconds: float = 5.0) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.02)
    return condition()


def test_a_file_auto_reads_without_its_extension_also_reaches_the_model_in_chat(fake, rock_clip):
    state = auto(fake, "fake", "voice-memo", rock_clip)
    assert state["status"] == "done" and state["installed"] == "vector:gtzan", state
    r = fake.post("/api/chat", data={"model": "fake", "message": "What is this?"},
                  files=[("files", ("voice-memo", rock_clip, "application/octet-stream"))])
    assert r.status_code == 200, r.text
    (used,) = r.json()["used"]
    assert used["kind"] == "vector" and used["bridge"] == "vector:gtzan" and used["source"] == "voice-memo"
    assert "rock" in r.json()["reply"]


@pytest.mark.parametrize("name, data, words", [
    ("photo.png", png(), "reads image natively"),
    ("notes.txt", b"Buy milk and eggs.", "reads text natively"),
], ids=["image", "text"])
def test_a_file_auto_calls_native_reaches_the_api_model_in_chat(fake, name, data, words):
    state = auto(fake, "fake-api", name, data, goal="describe it")
    assert state["status"] == "done" and state["installed"] is None and words in state["events"][-1]["message"]
    r = fake.post("/api/chat", data={"model": "fake-api", "message": "Describe it."},
                  files=[("files", (name, data, "application/octet-stream"))])
    assert r.status_code == 200, r.text
    assert [u["kind"] for u in r.json()["used"]] == ["native"]


def test_the_upload_is_removed_once_the_run_has_ended(fake, rock_clip):
    home = config.home()
    state = auto(fake, "fake", "my clip.wav", rock_clip)
    assert state["status"] == "done"
    assert wait_for(lambda: not uploads(home)), uploads(home)


def test_a_long_upload_name_is_cut_and_a_large_one_is_a_413(fake, rock_clip, monkeypatch):
    seen = []

    def start_auto(model, file, goal, examples=None):
        seen.append(Path(file))
        return "auto-job"

    monkeypatch.setattr(jobs, "start_auto", start_auto)
    r = fake.post("/api/auto", data={"model": "fake", "goal": GOAL},
                  files={"file": ("a" * 300 + ".wav", io.BytesIO(rock_clip), "audio/wav")})
    assert r.status_code == 200, r.text
    assert seen[0].name == "a" * (server_app.NAME_BYTES - 4) + ".wav" and seen[0].read_bytes() == rock_clip
    monkeypatch.setattr(server_app, "MAX_UPLOAD_BYTES", len(rock_clip) - 1)
    before = uploads(config.home())
    r = fake.post("/api/auto", data={"model": "fake", "goal": GOAL},
                  files={"file": ("big.wav", io.BytesIO(rock_clip), "audio/wav")})
    assert r.status_code == 413 and "'big.wav' is larger than" in r.json()["detail"]
    assert uploads(config.home()) == before and len(seen) == 1  # nothing kept, no run started


def test_an_upload_the_server_cannot_save_is_a_json_500_that_leaves_nothing(fake, rock_clip, monkeypatch):
    monkeypatch.setattr(jobs, "start_auto", lambda *a, **k: pytest.fail("no run may start"))
    config.home().mkdir(parents=True, exist_ok=True)
    (config.home() / server_app.UPLOADS).write_text("a file where the uploads folder should be")
    r = fake.post("/api/auto", data={"model": "fake", "goal": GOAL},
                  files={"file": ("clip.wav", io.BytesIO(rock_clip), "audio/wav")})
    assert r.status_code == 500 and "Could not save 'clip.wav' on the server" in r.json()["detail"]


def test_upload_folders_left_a_day_ago_are_removed(fake, rock_clip, monkeypatch):
    monkeypatch.setattr(jobs, "start_auto", lambda *a, **k: "auto-job")
    root = config.home() / server_app.UPLOADS
    old, recent = root / "0123456789ab", root / "ba9876543210"
    for folder in (old, recent):
        folder.mkdir(parents=True)
        (folder / "clip.wav").write_bytes(b"x")
    two_days_ago = time.time() - 2 * 24 * 3600
    os.utime(old, (two_days_ago, two_days_ago))
    r = fake.post("/api/auto", data={"model": "fake", "goal": GOAL},
                  files={"file": ("clip.wav", io.BytesIO(rock_clip), "audio/wav")})
    assert r.status_code == 200, r.text
    assert not old.exists() and recent.exists()


def test_only_png_and_jpeg_images_are_shown(doubles, monkeypatch):
    client, state = doubles
    gif = b"GIF89a" + bytes(32)
    monkeypatch.setattr(server_app, "_modality_of", lambda name, path: "image")  # as if agent.detect knew GIF
    used = chat(client, "/api/chat", "api", [("anim.gif", gif), ("photo.jpg", b"\xff\xd8\xff\xe0" + bytes(16))])
    assert used[0]["kind"] == "none" and "only PNG and JPEG images" in used[0]["detail"]
    assert used[1]["kind"] == "native"
    assert [item.source for item in state["models"]["api"].calls[-1][1]] == ["photo.jpg"]
