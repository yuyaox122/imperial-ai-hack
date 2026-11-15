"""The adapter flow end to end: agent.auto sends every input the model cannot take, other than audio, to
agent.synth.run_synth, which finds labelled probe files on the web (agent.hub_probes), verifies and installs an
agent-written adapter; POST /api/chat then carries such files through it; install requests are answered through
POST /api/jobs/<id>/approval or on the terminal (python -m agent).

Fake mode, offline: agent.llm.ask_json gives canned answers (a design and an adapter's code; None for the Hub's own
questions, so its rules decide), and agent.hub_probes' HTTP is a small in-memory Hub serving meshes made with
trimesh. The adapters really run, in sense.adapters' sandbox, and are verified with FakeModel.
Run: SENSE_FAKE=1 .venv/bin/python -m pytest -q tests/test_adapter_flow.py
"""

from __future__ import annotations

import copy
import io
import json
import time
import urllib.parse
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from agent import auto, hub_probes, installer, jobs, llm, synth
from agent import planner as planner_mod
from sense import adapters, config, registry, runs
from sense.adapters import AdapterBridge
from sense.contracts import ReportCard, TraceEvent
from server import app as server_app

ASSETS = "https://datasets-server.huggingface.co/assets"
GOAL = "Tell which shape this 3D model is"
WAIT = 60.0  # the most a test waits for an agent run

# Counts what is in the file and says what its surface is like. It names the shape it measured, which FakeModel
# (it picks the option a text names) needs; a real adapter reports measurements and the base model decides.
SHAPE_ADAPTER = '''
def adapt(path):
    vertices = faces = 0
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("v "):
                vertices += 1
            elif line.startswith("f "):
                faces += 1
    if vertices == 0:
        raise ValueError("no vertices in this file")
    surface = "flat faces meeting at sharp corners, like a box" if vertices <= 24 else "a smoothly curved surface, like a ball"
    return {"text": f"The mesh has {vertices} vertices and {faces} faces, with {surface}."}
'''


def mesh(label: str, k: int) -> bytes:
    """An OBJ file made with trimesh: a box or an icosphere ("ball"), its size varied by k."""
    import trimesh

    if label == "box":
        shape = trimesh.creation.box(extents=[1 + 0.11 * k, 1 + 0.07 * k, 0.5 + 0.05 * k])
    else:
        shape = trimesh.creation.icosphere(subdivisions=1, radius=0.5 + 0.09 * k)
    return shape.export(file_type="obj").encode()


def shape_design(**changes) -> dict:
    answer = {
        "modality": "3d-model", "format": "obj", "route": "text",
        "description": "Counts the mesh's vertices and faces and says how its surface curves.",
        "libraries": [], "question": "What shape is this 3D model?", "label_hint": "simple solid shapes",
        "labels": ["box", "ball"], "probe_plan": "dataset",
        "probe_reason": "Public mesh datasets label simple shapes.", "install": [],
    }
    return answer | changes


class FakeLLM:
    """agent.llm.ask_json: the design and the adapter's code by schema; None for anything else (the Hub's search
    queries and column choice, so its rules decide). Records every call."""

    def __init__(self, design: dict, code: str = SHAPE_ADAPTER):
        self.design, self.code, self.calls = design, code, []

    def __call__(self, instruction, data, schema, *, max_tokens=400):
        json.dumps(data, allow_nan=False)
        self.calls.append((schema, copy.deepcopy(data)))
        if schema is synth.DESIGN_SCHEMA:
            return copy.deepcopy(self.design)
        if schema is synth.CODE_SCHEMA:
            return {"code": self.code}
        return None

    def of(self, schema) -> list[dict]:
        return [data for s, data in self.calls if s is schema]


class Hub:
    """agent.hub_probes._http_get and _table_get: every Hub search finds every dataset held here; the dataset viewer
    answers /splits, /first-rows and /rows for them, and their files are served from ASSETS."""

    def __init__(self):
        self.datasets: dict[str, tuple[list, list[dict]]] = {}
        self.files: dict[str, bytes] = {}
        self.calls: list[str] = []

    def add_meshes(self, hub_id: str, labels: list[str], n: int) -> None:
        rows = []
        for i in range(n):
            label = labels[i % len(labels)]
            url = f"{ASSETS}/{hub_id}/{i}.obj"
            self.files[url] = mesh(label, i)
            rows.append({"mesh": {"src": url}, "shape": label})
        features = [{"name": "mesh", "type": {"_type": "Mesh"}},
                    {"name": "shape", "type": {"dtype": "string", "_type": "Value"}}]
        self.datasets[hub_id] = (features, rows)

    @staticmethod
    def item(hub_id: str) -> dict:
        return {"id": hub_id, "downloads": 100, "likes": 0, "gated": False, "private": False, "disabled": False,
                "tags": ["modality:3d"], "cardData": {"license": "cc-by-4.0"}}

    def __call__(self, url, *, timeout=20.0, max_bytes=50_000_000):
        self.calls.append(url)
        parts = urllib.parse.urlsplit(url)
        query = {k: v[0] for k, v in urllib.parse.parse_qs(parts.query).items()}
        if parts.netloc == "huggingface.co" and parts.path == "/api/datasets":
            return json.dumps([self.item(hub_id) for hub_id in self.datasets]).encode()
        if parts.netloc == "huggingface.co" and parts.path.startswith("/api/datasets/"):
            hub_id = parts.path[len("/api/datasets/"):]
            if hub_id in self.datasets:
                return json.dumps(self.item(hub_id)).encode()
        if parts.netloc == "datasets-server.huggingface.co" and parts.path in ("/splits", "/first-rows", "/rows"):
            if query.get("dataset") in self.datasets:
                features, rows = self.datasets[query["dataset"]]

                def page(offset, length):
                    return [{"row_idx": i, "row": rows[i], "truncated_cells": []}
                            for i in range(offset, min(offset + length, len(rows)))]

                if parts.path == "/splits":
                    return json.dumps({"splits": [{"dataset": query["dataset"], "config": "default",
                                                   "split": "train"}], "pending": [], "failed": []}).encode()
                if parts.path == "/first-rows":
                    return json.dumps({"dataset": query["dataset"], "features": features, "rows": page(0, 100),
                                       "truncated": len(rows) > 100}).encode()
                return json.dumps({"features": features, "rows": page(int(query["offset"]), int(query["length"])),
                                   "num_rows_total": len(rows), "num_rows_per_page": 100, "partial": False}).encode()
        if url in self.files:
            return self.files[url]
        raise hub_probes._HttpError("http_404", f"HTTP 404 from {parts.netloc}")


@pytest.fixture
def hub(monkeypatch, tmp_path):
    monkeypatch.setenv("SENSE_FAKE", "1")
    monkeypatch.setenv("SENSE_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(adapters, "_cache", type(adapters._cache)())
    if not adapters.sandboxed():  # inside another sandbox (an agent's shell, say): run the adapters without one
        monkeypatch.setenv(adapters.UNSANDBOXED_ENV, "1")
    fake = Hub()
    monkeypatch.setattr(hub_probes, "_http_get", fake)
    monkeypatch.setattr(hub_probes, "_table_get", fake)
    monkeypatch.setattr(hub_probes, "_WEB_MEMO", {})
    monkeypatch.setattr(hub_probes, "_sleep", lambda s: None)
    monkeypatch.setattr(hub_probes, "_ffmpeg", lambda: None)
    monkeypatch.setattr(installer, "ask_llm_review", lambda plan: "A small, well-known mesh library.")
    return fake


@pytest.fixture
def client(hub, monkeypatch):
    monkeypatch.setattr(jobs, "_jobs", {})
    monkeypatch.setattr(jobs, "_latest_id", None)
    monkeypatch.setattr(jobs, "_approvals", {})
    monkeypatch.setattr(jobs, "_cli_handler", None)
    monkeypatch.setattr(planner_mod, "_preferred", None)
    monkeypatch.setattr(server_app, "_recorder", None)
    return TestClient(server_app.app)


def post_auto(client, name: str, data: bytes, **form) -> str:
    r = client.post("/api/auto", data={"model": "fake", "goal": GOAL} | form,
                    files={"file": (name, io.BytesIO(data), "model/obj")})
    assert r.status_code == 200, r.text
    return r.json()["job_id"]


def wait_until(client, job_id: str, condition) -> dict:
    deadline = time.monotonic() + WAIT
    while time.monotonic() < deadline:
        state = client.get(f"/api/jobs/{job_id}").json()
        if condition(state):
            return state
        time.sleep(0.02)
    raise AssertionError(f"the condition was not met within {WAIT:g} s: {state['status']}, {state['events'][-1:]}")


def finished(state: dict) -> bool:
    return state["status"] != "running"


def last(events: list[dict], step: str, status: str | None = None) -> dict:
    return next(e for e in reversed(events) if e["step"] == step and (status is None or e["status"] == status))


def chat(client, model: str, files: list[tuple[str, bytes]]) -> dict:
    r = client.post("/api/chat", data={"model": model, "message": "What shape is this?"},
                    files=[("files", (name, data, "application/octet-stream")) for name, data in files])
    assert r.status_code == 200, r.text
    return r.json()


# ----------------------------------------------------------------------------- end to end over HTTP
def test_a_mesh_gets_an_adapter_sense_from_web_probes_and_chat_uses_it(client, hub, monkeypatch):
    hub.add_meshes("shapes/solids", ["box", "ball"], 40)
    fake = FakeLLM(shape_design())
    monkeypatch.setattr(llm, "ask_json", fake)
    job_id = post_auto(client, "my box.obj", mesh("box", 3))
    state = wait_until(client, job_id, finished)
    assert state["status"] == "done", state["error"]
    events = state["events"]
    assert list(dict.fromkeys(e["step"] for e in events)) == [
        "detect", "design", "build", "data", "verify", "decide", "install", "done"]
    assert last(events, "detect", "ok")["data"]["flow"] == "adapter"
    assert [e["t"] for e in events] == sorted(e["t"] for e in events)  # one clock for the whole run
    assert state["installed"] == "adapter:3d-obj" and state["pending_approval"] is None
    assert state["request"]["modality"] == "3d" and state["request"]["dataset"].startswith("3d-dataset-")

    # the probe files came from the web, with the Hub's events as data sub-steps
    sub = [(e["data"].get("purpose"), e["data"].get("action"), e["status"]) for e in events if e["step"] == "data"]
    assert ("probes", "search", "start") in sub and ("probes", "fetch", "ok") in sub
    assert ("probes", "candidate", "ok") in sub
    # the step's own summary has no action; the sub-steps of the unseen search come after it
    data = next(e for e in reversed(events) if e["step"] == "data" and e["status"] == "ok" and "action" not in e["data"])
    assert data["data"]["provenance"] == "dataset" and data["data"]["labels"] == ["ball", "box"]
    assert data["data"]["fetched"]["hub_id"] == "shapes/solids" and data["data"]["licence"] == "cc-by-4.0"
    # the only other dataset is the same source with the same labels, so unseen categories cannot be measured
    assert ("unseen", "search", "start") in sub
    unseen_note = last(events, "data", "info")
    assert unseen_note["data"]["purpose"] == "unseen" and unseen_note["message"].startswith(
        "Unseen categories not measured:")
    verify = last(events, "verify", "start")["data"]
    assert verify["native"] is True and verify["unseen"] is None  # an OBJ file is text: the native control runs
    (card,) = state["cards"]
    assert card["verdict"] == "installed" and card["accuracy"] >= 0.9 and card["n"] == 40
    assert "native" in card["controls"] and "Unseen categories not measured:" in card["reason"]
    # the adapter was written from the user's file alone, before any probe existed
    (asked,) = fake.of(synth.CODE_SCHEMA)
    assert "box" not in json.dumps(asked["design"]) and "ball" not in json.dumps(asked["design"])

    senses = client.get("/api/senses", params={"model": "fake"}).json()
    assert [(s["bridge"], s["kind"], s["modality"]) for s in senses] == [("adapter:3d-obj", "text", "3d")]
    manifest = registry.get_bridges("fake")[0].manifest
    assert manifest["detected"] == {"modality": "3d", "format": "obj", "suffix": ".obj", "formats": ["obj"]}
    assert manifest["probes"]["fetched"]["source"] == "shapes/solids"

    # chat: another mesh goes through the installed adapter, not as text
    reply = chat(client, "fake", [("ball.obj", mesh("ball", 7)), ("notes.txt", b"Buy milk.")])
    adapter_used, text_used = reply["used"]
    assert adapter_used["bridge"].startswith("adapter:") and adapter_used["kind"] == "text"
    assert adapter_used["source"] == "ball.obj" and "like a ball" in adapter_used["detail"]
    assert text_used["kind"] == "native"

    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and not any(r["id"] == job_id for r in runs.list_runs()):
        time.sleep(0.05)
    (recorded,) = [r for r in runs.list_runs() if r["id"] == job_id]
    assert recorded["installed"] == "adapter:3d-obj" and recorded["has_sense"]  # the request named the modality


def wait_for_approval(client, job_id: str) -> dict:
    return wait_until(client, job_id, lambda s: s["pending_approval"] is not None or finished(s))


def plan(verdict: str = "ok") -> installer.InstallPlan:
    return installer.InstallPlan(requested=[{"name": "meshkit", "version": None, "why": "reads meshes"}], packages=[],
                                 verdict=verdict, requirements="", created="2026-11-15T00:00:00+00:00",
                                 problems=[], target={})


def examples_folder(root: Path) -> Path:
    for label in ("box", "ball"):
        (root / label).mkdir(parents=True)
        for k in range(6):
            (root / label / f"m{k}.obj").write_bytes(mesh(label, k))
    return root


@pytest.mark.parametrize("approve", [False, True])
def test_an_install_request_is_answered_over_http_and_the_run_goes_on(client, monkeypatch, tmp_path, approve):
    installs = []

    def install(plan_, *, approved):
        assert approved is True
        site = installer.toolbox_dir()
        (site / "meshkit").mkdir(parents=True)
        (site / "meshkit" / "__init__.py").write_text("VALUE = 42\n")
        info = site / "meshkit-1.0.dist-info"
        info.mkdir()
        (info / "METADATA").write_text("Name: meshkit\nVersion: 1.0\n\n")
        (info / "top_level.txt").write_text("meshkit\n")
        (site.parent / "installed.json").write_text(json.dumps([{"name": "meshkit", "version": "1.0"}]))
        installs.append(plan_)
        return {"installed": [{"name": "meshkit", "version": "1.0", "sha256": "0" * 64}], "target": str(site),
                "seconds": 0.1, "log_tail": ""}

    monkeypatch.setattr(installer, "plan", lambda requests: plan())
    monkeypatch.setattr(installer, "install", install)
    fake = FakeLLM(shape_design(install=[{"name": "meshkit", "why": "reads meshes"}]))
    monkeypatch.setattr(llm, "ask_json", fake)
    job_id = post_auto(client, "box.obj", mesh("box", 2), examples=str(examples_folder(tmp_path / "shapes")))

    state = wait_for_approval(client, job_id)
    pending = state["pending_approval"]
    assert state["status"] == "running" and pending["kind"] == "install"
    assert pending["payload"]["summary"].startswith("OK to install meshkit")
    assert pending["payload"]["review"] == "A small, well-known mesh library."
    assert last(state["events"], "approval", "start")["data"]["id"] == pending["id"]
    # a tab that did not start the run can find it, and its open request, while it holds the one run slot
    assert client.get("/api/jobs/latest").json() == {"job_id": job_id}

    # bad answers change nothing
    answer = f"/api/jobs/{job_id}/approval"
    assert client.post("/api/jobs/no-such-job/approval", json={"id": pending["id"], "approve": True}).status_code == 404
    for body in ({"id": pending["id"], "approve": "yes"}, {"id": pending["id"], "approve": 1}, {"approve": True},
                 ["not", "an", "object"]):
        r = client.post(answer, json=body)
        assert r.status_code == 400 and "approve" in r.json()["detail"], (body, r.text)
    assert client.post(answer, content=b"{not json", headers={"content-type": "application/json"}).status_code == 400
    wrong = client.post(answer, json={"id": "not-the-id", "approve": True})
    assert wrong.status_code == 409 and pending["id"] in wrong.json()["detail"]
    assert client.get(f"/api/jobs/{job_id}").json()["pending_approval"]["id"] == pending["id"]

    r = client.post(answer, json={"id": pending["id"], "approve": approve})
    assert r.status_code == 200 and r.json() == {"answered": True, "approve": approve}
    assert client.post(answer, json={"id": pending["id"], "approve": approve}).status_code == 409  # not open now
    state = wait_until(client, job_id, finished)
    assert state["status"] == "done", state["error"]
    assert state["pending_approval"] is None and state["installed"] == "adapter:3d-obj"
    assert client.get("/api/jobs/latest").json() == {"job_id": None}
    steps = list(dict.fromkeys(e["step"] for e in state["events"]))
    assert steps[:5] == ["detect", "design", "install_request", "approval", "build"]
    answered = last(state["events"], "approval")
    outcome = last(state["events"], "install_request")
    (code_request,) = fake.of(synth.CODE_SCHEMA)
    if approve:
        assert answered["status"] == "ok" and answered["data"]["reason"] == "approved"
        assert outcome["status"] == "ok" and len(installs) == 1 and "meshkit" in code_request["toolbox"]
    else:
        assert answered["status"] == "fail" and answered["data"]["reason"] == "declined"
        assert outcome["status"] == "info" and installs == [] and "meshkit" not in code_request["toolbox"]
    record = registry.get_bridges("fake")[0].manifest["install"]
    assert record["approved"] is approve


# ----------------------------------------------------------------------------- agent.auto, without the server
def run_auto(file: Path, goal: str = GOAL, **kwargs):
    events: list[TraceEvent] = []
    cards, installed = auto.run_auto("fake", file, goal, events.append, **kwargs)
    return cards, installed, events


def test_no_probes_anywhere_asks_the_user_for_examples(hub, monkeypatch, tmp_path):
    # The Hub has nothing, and the design cannot build its labels into generated files.
    monkeypatch.setattr(llm, "ask_json", FakeLLM(shape_design(probe_plan="user")))
    user_file = tmp_path / "box.obj"
    user_file.write_bytes(mesh("box", 1))
    cards, installed, events = run_auto(user_file)
    assert (cards, installed) == ([], None)
    assert list(dict.fromkeys(e.step for e in events)) == ["detect", "design", "build", "data", "done"]
    searched = [e for e in events if e.step == "data" and e.data.get("action") == "search"]
    assert [e.status for e in searched] == ["start", "ok"] and searched[1].data["found"] == 0
    assert searched[0].data["purpose"] == "probes" and searched[0].data["provenance"] == "dataset"
    told = next(e for e in events if e.step == "data" and e.status == "info")
    assert "the search for labelled obj files found none" in told.message and told.data["web_reason"]
    done = events[-1]
    assert done.data["needs_user"] is True and done.data["installed"] is None
    assert done.message.startswith("I found no labelled examples on the web.")
    assert "one sub-folder per label" in done.message and done.data["wanted"]["labels"] == ["box", "ball"]
    assert registry.installed("fake") == []


@pytest.mark.parametrize("model, images", [("fake", False), ("fake-api", True)])
def test_the_designer_is_offered_images_only_for_a_model_chat_can_show_them_to(hub, monkeypatch, tmp_path, model,
                                                                                images):
    # The local model's ModelInfo.native lists "image", but chat cannot show it one: an image adapter would never be
    # used, so the designer must describe the image as text. An image sense is installed under "image", the name
    # chat looks it up by.
    fake = FakeLLM(shape_design(modality="photo", format="png", probe_plan="user"))
    monkeypatch.setattr(llm, "ask_json", fake)
    picture = tmp_path / "photo.png"
    picture.write_bytes(png())
    events: list[TraceEvent] = []
    synth.run_synth(model, picture, "Tell what is in the photo", events.append, web=True)
    (asked,) = fake.of(synth.DESIGN_SCHEMA)
    assert asked["images_allowed"] is images and ("image" in asked["models"][0]["native"]) is images
    designed = last_event(events, "design", "ok").data
    assert designed["modality"] == "image" and designed["designed_modality"] == "photo"


def test_generated_probes_come_after_the_web(hub, monkeypatch, tmp_path):
    # With nothing on the web, a design that builds its labels into the content falls back to generated files.
    generator = '''
import os

def generate(out_dir, labels, n):
    for label in labels:
        os.makedirs(os.path.join(out_dir, label), exist_ok=True)
        for k in range(n):
            with open(os.path.join(out_dir, label, f"sample_{k:03d}.obj"), "w") as f:
                count = 8 if label == "box" else 42
                f.write("".join(f"v {k + i} {i} {k}\\n" for i in range(count)) + "f 1 2 3\\n")
'''
    fake = FakeLLM(shape_design(probe_plan="generate"))
    original = fake.__call__

    def answer(instruction, data, schema, *, max_tokens=400):
        return {"code": generator} if schema is synth.GENERATOR_SCHEMA else original(instruction, data, schema,
                                                                                     max_tokens=max_tokens)

    monkeypatch.setattr(llm, "ask_json", answer)
    monkeypatch.setattr(synth, "GENERATED_PER_LABEL", 6)
    user_file = tmp_path / "box.obj"
    user_file.write_bytes(mesh("box", 1))
    cards, installed, events = run_auto(user_file)
    data = next(e for e in reversed(events) if e.step == "data" and e.status == "ok" and "action" not in e.data)
    assert data.data["provenance"] == "generated"
    actions = [e.data.get("action") for e in events if e.step == "data"]
    assert actions.index("search") < actions.index(None)  # the web was searched first
    assert "consistency check" in cards[0].reason


UNSEEN_ADAPTER = SHAPE_ADAPTER.replace(
    'surface = "flat', 'surface = "a pointed tip, like a cone" if vertices == 5 else "flat')


def test_unseen_categories_come_from_another_source_with_other_labels(hub, monkeypatch, tmp_path):
    calls = []
    sets = {"probes": ("shapes/solids", {"box": 10, "ball": 10}), "unseen": ("shapes/more", {"cone": 4, "box2": 4})}

    def find_probes(goal, modality, dest_root, *, emit=None, exclude_sources=(), disjoint_from=(), **kwargs):
        purpose = "unseen" if str(dest_root).endswith("-unseen") else "probes"
        calls.append({"purpose": purpose, "modality": modality, "dest_root": Path(dest_root),
                      "exclude_sources": list(exclude_sources), "disjoint_from": list(disjoint_from)})
        emit({"action": "search", "status": "start", "goal": goal, "modality": modality})
        source, counts = sets[purpose]
        folder = Path(dest_root) / source.replace("/", "__")
        for label, n in counts.items():
            (folder / label).mkdir(parents=True)
            for k in range(n):
                shape = {"box": "box", "ball": "ball"}.get(label)
                data = mesh(shape, k) if shape else ("v 0 0 0\n" * (5 if label == "cone" else 8) + f"v {k} 0 0\nf 1 2 3\n").encode()
                (folder / label / f"{k}.obj").write_bytes(data)
        emit({"action": "candidate", "status": "ok", "hub_id": source, "source": source, "kind": "hub",
              "labels": list(counts), "licence": "mit"})
        return folder, {"hub_id": source, "url": f"https://huggingface.co/datasets/{source}", "title": source,
                        "licence": "mit", "kind": "hub", "source": source, "labels": list(counts),
                        "label_values": {label: label.title() for label in counts}, "counts": counts}

    monkeypatch.setattr(hub_probes, "find_probes", find_probes)
    monkeypatch.setattr(llm, "ask_json", FakeLLM(shape_design()))
    user_file = tmp_path / "box.obj"
    user_file.write_bytes(mesh("box", 1))
    cards, installed, events = run_auto(user_file)
    first, second = calls
    assert first["purpose"] == "probes" and first["exclude_sources"] == [] and first["modality"] == "3d"
    assert first["dest_root"] == config.home() / "fetched" / "probes" / "3d"
    assert second["dest_root"] == config.home() / "fetched" / "probes" / "3d-unseen"
    assert second["exclude_sources"] == ["shapes/solids"]
    assert {"Box", "Ball"} <= set(second["disjoint_from"])  # the dataset's own label values
    told = [e for e in events if e.step == "data" and e.status == "info"]
    assert told[-1].data["unseen"]["labels"] == ["box2", "cone"]
    assert last_event(events, "verify", "start").data["unseen"].startswith("3d-unseen-")
    (card,) = cards
    assert "unseen_categories" in card.controls and card.control_chance == {"unseen_categories": 0.5}
    assert "Unseen categories:" in card.reason and "source: https://huggingface.co/datasets/shapes/more" in card.reason


def last_event(events: list[TraceEvent], step: str, status: str | None = None) -> TraceEvent:
    return next(e for e in reversed(events) if e.step == step and (status is None or e.status == status))


def test_audio_keeps_the_specialist_flow(hub, monkeypatch, tmp_path):
    from sense.specialists import FAKE_CONCEPTS, FakeSpecialist

    monkeypatch.setattr(planner_mod, "_preferred", None)
    monkeypatch.setattr(synth, "run_synth", lambda *a, **k: pytest.fail("audio must not reach the adapter flow"))
    clip = tmp_path / "clip.wav"
    FakeSpecialist.write_sample(FAKE_CONCEPTS.index("rock"), clip, seed=1)
    cards, installed, events = run_auto(clip, "Make it recognise music genres")
    assert installed == "vector:gtzan" and last_event(events, "detect", "ok").data["flow"] == "specialist"
    assert last_event(events, "detect", "ok").message == last_event(events, "detect", "ok").data["detail"]
    assert list(dict.fromkeys(e.step for e in events)) == [
        "detect", "research", "data", "inspect", "plan", "fit", "verify", "claims", "decide", "install", "done"]
    assert hub.calls == []  # no probe search for audio


def test_an_error_in_the_adapter_flow_is_reported_once_and_raised(hub, monkeypatch, tmp_path):
    def broken(*args, **kwargs):
        raise RuntimeError("the probe store is not writable")

    monkeypatch.setattr(llm, "ask_json", FakeLLM(shape_design()))
    monkeypatch.setattr(hub_probes, "find_probes", broken)
    user_file = tmp_path / "box.obj"
    user_file.write_bytes(mesh("box", 1))
    events: list[TraceEvent] = []
    with pytest.raises(RuntimeError, match="not writable"):
        auto.run_auto("fake", user_file, GOAL, events.append)
    assert [e.step for e in events].count("error") == 1
    assert events[-2].step == "data" and events[-2].status == "fail"
    assert events[-1].data == {"step": "data", "error": "RuntimeError"}


# ----------------------------------------------------------------------------- chat through installed adapters
def install_adapter(model: str, modality: str, code: str, *, kind: str = "text", **manifest) -> AdapterBridge:
    bridge = AdapterBridge(kind=kind, modality=modality, model_name=model, code=code, manifest=manifest)
    card = ReportCard(model, modality, f"{modality}-examples", "closed", bridge.name, kind, 20, 0.5, 0.95,
                      {"swapped": 0.4}, verdict="installed")
    registry.install(bridge, card)
    return bridge


ECHO = 'def adapt(path):\n    return {"text": "Read: " + open(path, "rb").read()[:40].decode("utf-8", "replace")}\n'
PICTURE = ('def adapt(path):\n    import numpy as np\n    return {"images": [np.zeros((8, 8), dtype="uint8")]}\n')


def png() -> bytes:
    import struct
    import zlib

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(b"\x00\xff\x00\x00")) + chunk(b"IEND", b""))


def test_an_image_adapter_needs_a_model_that_takes_images(client):
    install_adapter("fake", "image", PICTURE, kind="image", format="png")
    (used,) = chat(client, "fake", [("photo.png", png())])["used"]
    assert used["kind"] == "none" and "turns it into images" in used["detail"]
    install_adapter("fake", "image", ECHO, format="png")  # a text adapter for images: a local model reads it
    (used,) = chat(client, "fake", [("photo.png", png())])["used"]
    assert used["bridge"] == "adapter:image-png" and used["kind"] == "text" and "Read:" in used["detail"]


def test_files_detect_calls_other_find_their_adapter_by_format_or_extension(client):
    install_adapter("fake", "dna-sequence", ECHO, format="fasta",
                    detected={"modality": "other", "format": "fasta", "suffix": ".fa"})
    install_adapter("fake", "sensor-blob", ECHO, format="xyz",
                    detected={"modality": "other", "format": "unknown", "suffix": ".xyz"})
    blob = bytes(range(256))
    used = chat(client, "fake", [("gene.fasta", b">seq\nACGT\n"), ("table.csv", b"a,b,c\n1,2,3\n"),
                                 ("reading.xyz", blob), ("reading.bin", blob)])["used"]
    fasta, table, xyz, other = used
    assert fasta["bridge"] == "adapter:dna-sequence-fasta" and "Read: >seq" in fasta["detail"]
    assert table["kind"] == "none" and "Run the agent on this file" in table["detail"]
    assert xyz["bridge"] == "adapter:sensor-blob-xyz"  # the upload kept its extension, so the adapter is found
    assert other["kind"] == "none"


GLB = b"glTF" + (2).to_bytes(4, "little") + bytes(12)  # agent.detect: 3d, glb (binary, so never given as text)
TRI_PLY = (b"ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\n"
           b"element face 1\nproperty list uchar int vertex_indices\nend_header\n0 0 0\n1 0 0\n0 1 0\n3 0 1 2\n")
OBJ_ONLY = {"modality": "3d", "format": "obj", "suffix": ".obj", "formats": ["obj"]}


def test_an_adapter_that_fails_on_a_file_leaves_it_out_and_says_why(client):
    # A text mesh falls back to its text, as it went before any sense, and says why; a binary one is left out.
    install_adapter("fake", "3d", 'def adapt(path):\n    raise ValueError("cannot parse this mesh")\n', format="obj")
    used = chat(client, "fake", [("broken.obj", mesh("box", 1)), ("scene.glb", GLB), ("notes.txt", b"hello")])["used"]
    as_text, left_out, notes = used
    assert as_text["kind"] == "native" and as_text["detail"].startswith("its text was given to the model")
    assert "could not read it" in as_text["detail"] and "cannot parse this mesh" in as_text["detail"]
    assert left_out["kind"] == "none" and "cannot parse this mesh" in left_out["detail"]
    assert notes["kind"] == "native"


def test_a_text_mesh_with_no_sense_is_still_given_as_text(client):
    (used,) = chat(client, "fake", [("tri.obj", b"v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")])["used"]
    assert used["kind"] == "native" and used["bridge"] == ""


def test_chat_gives_an_adapter_only_the_formats_it_was_verified_on(client):
    # An adapter written and scored on OBJ files is not handed a PLY or a GLB: its card covers neither. The PLY,
    # which is text, still goes as text, as it did before the adapter was installed.
    lenient = 'def adapt(path):\n    return {"text": "a smoothly curved surface, like a ball"}\n'
    install_adapter("fake", "3d", lenient, format="obj", detected=OBJ_ONLY)
    ply, glb, obj = chat(client, "fake", [("tri.ply", TRI_PLY), ("scene.glb", GLB), ("ball.obj", mesh("ball", 2))])["used"]
    assert ply["kind"] == "native" and ply["bridge"] == ""
    assert "adapter:3d-obj was built and checked on OBJ files only" in ply["detail"]
    assert glb["kind"] == "none" and "built and checked on OBJ files only, so it is not given a GLB file" in glb["detail"]
    assert obj["bridge"] == "adapter:3d-obj" and obj["kind"] == "text"
    # the formats of its probe files count too
    install_adapter("fake", "3d", lenient, format="obj", detected=OBJ_ONLY | {"formats": ["obj", "ply"]})
    (ply,) = chat(client, "fake", [("tri.ply", TRI_PLY)])["used"]
    assert ply["bridge"] == "adapter:3d-obj"


def m4a(size: int, seed: int = 0) -> bytes:
    """An M4A header (agent.detect: other, unknown) and `size` bytes."""
    return b"\x00\x00\x00\x20ftypM4A \x00\x00\x02\x00M4A mp42isom\x00\x00\x00\x00" + bytes([seed % 251]) * size


# The adapter tells a voice memo from music by its length; FakeModel picks the option its text names.
MEMO_ADAPTER = '''
import os

def adapt(path):
    size = os.path.getsize(path)
    return {"text": f"A clip of {size} bytes: short, like speech." if size < 600 else f"A clip of {size} bytes: long, like music."}
'''


def test_an_adapter_for_a_file_detect_calls_other_never_takes_an_audio_sense_or_wavs(client, monkeypatch, tmp_path):
    from sense.specialists import FAKE_CONCEPTS, FakeSpecialist

    # A verified audio sense first: CLAP's vector bridge, from the specialist flow.
    monkeypatch.setattr(planner_mod, "_preferred", None)
    clip = tmp_path / "song.wav"
    FakeSpecialist.write_sample(FAKE_CONCEPTS.index("rock"), clip, seed=1)
    _, installed, _ = run_auto(clip, "Make it recognise music genres")
    assert installed == "vector:gtzan"

    # Then an M4A voice memo, which agent.detect calls other: the designer names its modality "audio".
    folder = tmp_path / "memos"
    for label, size in (("speech", 200), ("music", 2000)):
        (folder / label).mkdir(parents=True)
        for k in range(6):
            (folder / label / f"c{k}.m4a").write_bytes(m4a(size + 10 * k, k))
    memo = tmp_path / "memo.m4a"
    memo.write_bytes(m4a(250))
    design = shape_design(modality="audio", format="m4a", question="Is this clip speech or music?",
                          label_hint="speech or music", labels=["speech", "music"], probe_plan="user",
                          description="Measures the clip's length.")
    monkeypatch.setattr(llm, "ask_json", FakeLLM(design, MEMO_ADAPTER))
    cards, installed, events = run_auto(memo, "Tell speech from music", examples=folder)
    designed = last_event(events, "design", "ok").data
    assert last_event(events, "detect", "ok").data["modality"] == "other"
    assert designed["modality"] == "audio-m4a" and designed["designed_modality"] == "audio"
    assert installed == "adapter:audio-m4a", cards and cards[0].reason
    assert sorted((s["bridge"], s["modality"]) for s in registry.installed("fake")) == [
        ("adapter:audio-m4a", "audio-m4a"), ("vector:gtzan", "audio")]  # nothing was replaced

    # Chat: a WAV still goes through the audio sense, and an M4A, alone or with a WAV, through its adapter.
    (wav,) = chat(client, "fake", [("song.wav", clip.read_bytes())])["used"]
    assert wav["bridge"] == "vector:gtzan"
    (alone,) = chat(client, "fake", [("memo2.m4a", m4a(300))])["used"]
    assert alone["bridge"] == "adapter:audio-m4a" and "like speech" in alone["detail"]
    with_wav = chat(client, "fake", [("memo3.m4a", m4a(3000)), ("song.wav", clip.read_bytes())])["used"]
    assert [u["bridge"] for u in with_wav] == ["adapter:audio-m4a", "vector:gtzan"]


def still_gif() -> bytes:
    """A 1x1 GIF89a with one frame: agent.detect calls it an image."""
    return (b"GIF89a" + (1).to_bytes(2, "little") * 2 + b"\x00\x00\x00" + b"\x2c" + bytes(4)
            + (1).to_bytes(2, "little") * 2 + b"\x00" + b"\x02\x02\x44\x01\x00" + b"\x3b")


def test_a_still_gif_needs_a_sense_even_for_a_model_that_takes_images(client, monkeypatch, tmp_path):
    # Chat shows an API model PNG and JPEG images only, so the agent must not say a GIF needs no sense.
    started = []
    monkeypatch.setattr(synth, "run_synth", lambda *args, **kwargs: started.append(args) or ([], None))
    picture = tmp_path / "pic.gif"
    picture.write_bytes(still_gif())
    events: list[TraceEvent] = []
    auto.run_auto("fake-api", picture, "Tell what is in the picture", events.append)
    detected = last_event(events, "detect", "ok")
    assert detected.data["flow"] == "adapter" and len(started) == 1
    assert "can be shown only PNG and JPEG images in chat, and this is a GIF image" in detected.message
    png_file = tmp_path / "pic.png"
    png_file.write_bytes(png())
    events = []
    auto.run_auto("fake-api", png_file, "Tell what is in the picture", events.append)
    assert last_event(events, "detect", "ok").data["flow"] == "native"

    # Chat: no sense, so the GIF is not shown and says why; with a GIF adapter installed, it goes through it.
    (gif,) = chat(client, "fake-api", [("pic.gif", still_gif())])["used"]
    assert gif["kind"] == "none" and "only PNG and JPEG images can be shown to it" in gif["detail"]
    install_adapter("fake-api", "image", ECHO, format="gif",
                    detected={"modality": "image", "format": "gif", "suffix": ".gif", "formats": ["gif"]})
    gif, photo = chat(client, "fake-api", [("pic.gif", still_gif()), ("pic.png", png())])["used"]
    assert gif["bridge"] == "adapter:image-gif" and gif["kind"] == "text"
    assert photo["kind"] == "native" and photo["detail"] == "shown to the model as an image"


# ----------------------------------------------------------------------------- python -m agent
def test_cli_saves_a_run_record_that_sense_runs_imports(hub, monkeypatch, tmp_path, capsys):
    from agent.__main__ import main

    hub.add_meshes("shapes/solids", ["box", "ball"], 40)
    monkeypatch.setattr(llm, "ask_json", FakeLLM(shape_design()))
    user_file = tmp_path / "box.obj"
    user_file.write_bytes(mesh("box", 4))
    saved = tmp_path / "records" / "box.json"
    assert main(["--model", "fake", "--file", str(user_file), "--goal", GOAL, "--save", str(saved)]) == 0
    record = json.loads(saved.read_text())
    assert record["origin"] == "cli" and record["status"] == "done" and record["installed"] == "adapter:3d-obj"
    assert record["request"]["modality"] == "3d" and record["events"][0]["step"] == "detect"
    assert set(record["machine"]) == {"platform", "python", "machine"} and record["seconds"] >= 0
    run_id = runs.import_record(saved)
    assert runs.list_runs()[0]["id"] == run_id and runs.list_runs()[0]["has_sense"]
    assert f"Saved the run record (done, {len(record['events'])} events)" in capsys.readouterr().out

    failed = tmp_path / "records" / "failed.json"
    assert main(["--model", "fake", "--file", str(tmp_path / "missing.obj"), "--goal", GOAL,
                 "--save", str(failed)]) == 1
    record = json.loads(failed.read_text())
    assert record["status"] == "failed" and record["error"].startswith("ValueError")


@pytest.mark.parametrize("typed, approved", [("y", True), ("yes", True), ("", False), ("n", False), (EOFError, False)])
def test_cli_asks_on_the_terminal_and_defaults_to_no(monkeypatch, tmp_path, capsys, typed, approved):
    from agent import __main__ as cli

    answers = []

    def fake_run_auto(model, file, goal, emit, examples=None, on_request=None):
        payload = {"summary": "OK to install meshkit 1.0.", "plan": {"packages": [
            {"name": "meshkit", "version": "1.0", "verdict": "ok", "reasons": []}]}, "review": None}
        answers.append(jobs.request_approval("install", payload, timeout=5))  # outside a job: the CLI handler
        return [], None

    def fake_input(prompt):
        assert prompt == "Install? [y/N] "
        if typed is EOFError:
            raise EOFError
        return typed

    monkeypatch.setattr(cli, "run_auto", fake_run_auto)
    monkeypatch.setattr("builtins.input", fake_input)
    assert cli.main(["--model", "fake", "--file", str(tmp_path / "x.obj"), "--goal", GOAL]) == 0
    assert answers == [approved]
    out = capsys.readouterr().out
    assert "OK to install meshkit 1.0." in out and "meshkit 1.0: ok" in out
    assert jobs._cli_handler is None  # removed after the run


def test_cli_prints_the_checks_verdict_and_labels_the_llm_review_as_an_opinion(monkeypatch, capsys):
    from agent.__main__ import ask_on_terminal

    monkeypatch.setattr("builtins.input", lambda prompt: "n")
    payload = {"summary": "Caution: meshkit 1.0 was first released 3 days ago.",
               "plan": {"verdict": "caution", "packages": [
                   {"name": "meshkit", "version": "1.0", "verdict": "caution", "reasons": ["first release 3 days ago"]}]},
               "review": "Looks safe to install; a well-known library."}
    assert ask_on_terminal("install", payload) is False
    lines = capsys.readouterr().out.splitlines()
    assert "What the checks found: caution" in lines
    assert "  meshkit 1.0: caution (first release 3 days ago)" in lines
    assert "An LLM's opinion, not one of the checks: Looks safe to install; a well-known library." in lines
    assert not any(line.strip().startswith("Review:") for line in lines)


def test_cli_approve_installs_approves_without_asking_and_says_so(monkeypatch, tmp_path, capsys):
    from agent import __main__ as cli

    answers = []

    def fake_run_auto(model, file, goal, emit, examples=None, on_request=None):
        answers.append(jobs.request_approval("install", {"summary": "OK to install meshkit 1.0."}, timeout=5))
        return [], None

    monkeypatch.setattr(cli, "run_auto", fake_run_auto)
    monkeypatch.setattr("builtins.input", lambda prompt: pytest.fail("--approve-installs must not ask"))
    assert cli.main(["--model", "fake", "--file", str(tmp_path / "x.obj"), "--goal", GOAL,
                     "--approve-installs"]) == 0
    assert answers == [True]
    out = capsys.readouterr().out
    assert out.count("APPROVED WITHOUT ASKING") == 1 and "WITHOUT ASKING." in out.splitlines()[1]
