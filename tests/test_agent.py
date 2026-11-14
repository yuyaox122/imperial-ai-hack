"""Agent tests: planner rules and registry, the fit loop, and background jobs.

Every sense.* call is replaced with a fake, so no model, data or disk is touched.
Run: SENSE_FAKE=1 .venv/bin/python -m pytest -q tests/test_agent.py
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from agent import jobs, loop
from agent import planner as planner_mod
from agent.planner import RulePlanner, get_planner, register_planner
from sense import bridges, datasets, models, registry, verify
from sense.contracts import (
    Decision,
    FitRequest,
    LabelledSet,
    ModelInfo,
    ReportCard,
    SenseInput,
    TraceEvent,
)

STEPS = ["inspect", "plan", "fit", "verify", "decide", "install", "done"]


# ----------------------------------------------------------------------------- fakes
@dataclass
class FakeBridge:
    kind: str
    name: str
    modality: str = "audio"
    model_name: str = "fake"

    def from_embedding(self, emb: np.ndarray, source: str = "") -> SenseInput:
        return SenseInput(kind="text", modality=self.modality, source=source, text="a sound")

    def prepare(self, path: Path) -> SenseInput:
        return self.from_embedding(np.zeros(4), path.name)

    def save(self, directory: Path) -> None:
        pass


class FakeModel:
    def __init__(self, open_weights: bool = True):
        self.info = ModelInfo(name="fake", label="Fake model", native=["text"], open_weights=open_weights)

    def chat(self, message: str, inputs: list[SenseInput], max_new_tokens: int = 96) -> str:
        return "ok"

    def choose(self, items: list) -> list[int]:
        return [0 for _ in items]


def toy_set() -> LabelledSet:
    n = 25
    return LabelledSet(
        name="toy",
        title="Toy sounds",
        modality="audio",
        question="Which sound is in the clip?",
        labels=["dog", "rain", "siren", "bell", "car"],
        y=np.arange(n) % 5,
        emb=np.zeros((n, 4)),
        label_text_emb=np.zeros((5, 4)),
        fit_idx=np.arange(15),
        dev_idx=np.arange(15, 20),
        test_idx=np.arange(20, 25),
    )


def make_card(bridge: str, kind: str, accuracy: float, scope: str = "closed", **controls: float) -> ReportCard:
    return ReportCard(
        model="fake", modality="audio", dataset="toy", scope=scope, bridge=bridge, kind=kind,
        n=200, chance=0.2, accuracy=accuracy, controls=controls,
    )


@pytest.fixture
def fake_sense(monkeypatch):
    """Replace every sense.* function the loop calls. Set `state.results[kind] = (accuracy, controls)`."""
    state = SimpleNamespace(
        model=FakeModel(), results={}, installs=[], verify_calls=[], text_calls=[], vector_fits=0,
    )

    def fit_vector(model, dataset, idx, layer=None):
        state.vector_fits += 1
        assert list(idx) == list(dataset.fit_idx), "the vector bridge must be fitted on dataset.fit_idx"
        return FakeBridge("vector", f"vector:{dataset.name}")

    def make_text(model, dataset=None):
        state.text_calls.append(dataset)
        return FakeBridge("text", f"text:{dataset.name}" if dataset is not None else "text:captions")

    def fake_verify(model, bridge, dataset, request, text_bridge=None, max_items=200, seed=0):
        state.verify_calls.append((bridge.name, text_bridge.name if text_bridge else None))
        accuracy, controls = state.results[bridge.kind]
        controls = dict(controls)
        if text_bridge is not None:
            controls["text_bridge"] = state.results["text"][0]
        return ReportCard(
            model=model.info.name, modality=dataset.modality, dataset=dataset.name, scope=request.scope,
            bridge=bridge.name, kind=bridge.kind, n=200, chance=0.2, accuracy=accuracy, controls=controls,
        )

    monkeypatch.setattr(models, "load_model", lambda name: state.model)
    monkeypatch.setattr(datasets, "load_labelled_set", lambda name: toy_set())
    monkeypatch.setattr(bridges, "fit_vector_bridge", fit_vector)
    monkeypatch.setattr(bridges, "make_text_bridge", make_text)
    monkeypatch.setattr(verify, "verify", fake_verify)
    monkeypatch.setattr(registry, "install", lambda bridge, card: state.installs.append((bridge.name, card.verdict)))
    return state


@pytest.fixture
def fresh_jobs(monkeypatch):
    monkeypatch.setattr(jobs, "_jobs", {})
    monkeypatch.setattr(jobs, "_latest_id", None)


def run(request: FitRequest) -> tuple[list[ReportCard], str | None, list[TraceEvent]]:
    events: list[TraceEvent] = []
    cards, installed = loop.run_fit(request, events.append, planner=RulePlanner())
    return cards, installed, events


def steps_in_order(events: list[TraceEvent]) -> list[str]:
    seen: list[str] = []
    for e in events:
        if not seen or seen[-1] != e.step:
            seen.append(e.step)
    return seen


def wait_for(condition, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.01)
    return False


# ----------------------------------------------------------------------------- planner
def test_plan_depends_on_open_weights():
    request, ds = FitRequest("fake", "toy"), toy_set()
    assert RulePlanner().plan(request, FakeModel(True).info, ds) == ["vector", "text"]
    assert RulePlanner().plan(request, FakeModel(False).info, ds) == ["text"]


def test_decide_prefers_vector_within_three_points():
    cards = [make_card("vector:toy", "vector", 0.84, swapped=0.2), make_card("text:toy", "text", 0.86)]
    decision = RulePlanner().decide(FitRequest("fake", "toy"), cards)
    assert decision.install == "vector:toy"
    assert "84.0%" in decision.reason and "86.0%" in decision.reason


def test_decide_takes_text_when_clearly_better():
    cards = [make_card("vector:toy", "vector", 0.80, swapped=0.2), make_card("text:toy", "text", 0.90)]
    decision = RulePlanner().decide(FitRequest("fake", "toy"), cards)
    assert decision.install == "text:toy"
    assert "90.0%" in decision.reason and "80.0%" in decision.reason


def test_decide_rejects_vector_close_to_swapped():
    cards = [make_card("vector:toy", "vector", 0.50, swapped=0.30)]
    decision = RulePlanner().decide(FitRequest("fake", "toy"), cards)
    assert decision.install is None
    assert "30.0%" in decision.reason and "swapped" in decision.reason


def test_decide_open_scope_needs_unseen_classes():
    request = FitRequest("fake", "toy", scope="open")
    cards = [
        make_card("vector:toy", "vector", 0.95, "open", swapped=0.2, unseen_classes=0.22),
        make_card("text:captions", "text", 0.70, "open"),
    ]
    decision = RulePlanner().decide(request, cards)
    assert decision.install == "text:captions"
    assert "22.0%" in decision.reason and "45.0%" in decision.reason


def test_decide_text_bar_is_inclusive():
    decision = RulePlanner().decide(FitRequest("fake", "toy"), [make_card("text:toy", "text", 0.35)])
    assert decision.install == "text:toy"


def test_planner_registry(monkeypatch):
    monkeypatch.setattr(planner_mod, "_factories", dict(planner_mod._factories))
    monkeypatch.setattr(planner_mod, "_preferred", None)
    assert isinstance(get_planner(), RulePlanner)

    class Other(RulePlanner):
        name = "other"

    register_planner("other", Other, preferred=True)
    assert isinstance(get_planner(), Other)
    assert type(get_planner("rules")) is RulePlanner
    with pytest.raises(ValueError, match="No planner is registered as 'nope'"):
        get_planner("nope")


# ----------------------------------------------------------------------------- loop
def test_closed_scope_installs_vector(fake_sense):
    fake_sense.results = {
        "vector": (0.90, {"swapped": 0.2, "no_input": 0.2, "lookup": 0.9}),
        "text": (0.60, {"swapped": 0.2, "no_input": 0.2, "lookup": 0.9}),
    }
    cards, installed, events = run(FitRequest("fake", "toy"))

    assert installed == "vector:toy"
    assert fake_sense.installs == [("vector:toy", "installed")]
    assert fake_sense.verify_calls == [("vector:toy", "text:toy"), ("text:toy", None)]
    assert fake_sense.text_calls[0].name == "toy"  # closed scope: text bridge names the dataset's labels

    by_name = {c.bridge: c for c in cards}
    assert by_name["vector:toy"].verdict == "installed"
    assert by_name["vector:toy"].controls["text_bridge"] == 0.60
    assert by_name["text:toy"].verdict == "rejected"
    assert "60.0%" in by_name["text:toy"].reason and "vector:toy" in by_name["text:toy"].reason

    assert steps_in_order(events) == STEPS
    assert [e.t for e in events] == sorted(e.t for e in events) and events[0].t >= 0
    verify_ok = [e for e in events if e.step == "verify" and e.status == "ok"]
    assert [e.data["card"]["bridge"] for e in verify_ok] == ["vector:toy", "text:toy"]
    assert events[-1].step == "done" and events[-1].status == "ok" and "90.0%" in events[-1].message


def test_open_scope_vector_fails_unseen_classes_and_text_installed(fake_sense):
    fake_sense.results = {
        "vector": (0.95, {"swapped": 0.2, "no_input": 0.2, "lookup": 0.95, "unseen_classes": 0.22}),
        "text": (0.70, {"swapped": 0.2, "no_input": 0.2, "lookup": 0.95}),
    }
    cards, installed, events = run(FitRequest("fake", "toy", scope="open"))

    assert installed == "text:captions"
    assert fake_sense.text_calls == [None]  # open scope: the caption bank, not the labels
    assert fake_sense.installs == [("text:captions", "installed")]
    vector = next(c for c in cards if c.kind == "vector")
    assert vector.verdict == "rejected"
    assert "22.0%" in vector.reason and "not fitted on" in vector.reason
    decide_ok = next(e for e in events if e.step == "decide" and e.status == "ok")
    assert decide_ok.data["install"] == "text:captions"


def test_nothing_acceptable_installs_nothing(fake_sense):
    fake_sense.results = {
        "vector": (0.30, {"swapped": 0.2, "no_input": 0.2, "lookup": 0.3}),
        "text": (0.25, {"swapped": 0.2, "no_input": 0.2, "lookup": 0.3}),
    }
    cards, installed, events = run(FitRequest("fake", "toy"))

    assert installed is None
    assert fake_sense.installs == []
    assert all(c.verdict == "rejected" and c.reason for c in cards)
    install_events = [e for e in events if e.step == "install"]
    assert [e.status for e in install_events] == ["info"]
    assert events[-1].step == "done" and "No audio sense was installed" in events[-1].message


def test_closed_weight_model_skips_vector(fake_sense):
    fake_sense.model = FakeModel(open_weights=False)
    fake_sense.results = {"text": (0.60, {"swapped": 0.2, "no_input": 0.2, "lookup": 0.9})}
    cards, installed, _ = run(FitRequest("fake", "toy"))
    assert fake_sense.vector_fits == 0
    assert installed == "text:toy" and [c.bridge for c in cards] == ["text:toy"]


def test_planner_choosing_unknown_bridge_is_an_error(fake_sense):
    class Confused(RulePlanner):
        def decide(self, request, cards):
            return Decision("vector:other", "made up")

    fake_sense.results = {"vector": (0.9, {"swapped": 0.2}), "text": (0.6, {})}
    events: list[TraceEvent] = []
    with pytest.raises(ValueError, match="not one of the verified candidates"):
        loop.run_fit(FitRequest("fake", "toy"), events.append, planner=Confused())
    assert fake_sense.installs == []
    assert events[-1].step == "error"


def test_failing_step_emits_error_and_reraises(fake_sense, monkeypatch):
    def broken(model, dataset, idx, layer=None):
        raise MemoryError("not enough memory to fit; close other models and retry")

    monkeypatch.setattr(bridges, "fit_vector_bridge", broken)
    events: list[TraceEvent] = []
    with pytest.raises(MemoryError):
        loop.run_fit(FitRequest("fake", "toy"), events.append, planner=RulePlanner())

    assert (events[-2].step, events[-2].status) == ("fit", "fail")
    assert (events[-1].step, events[-1].status) == ("error", "fail")
    assert "fit" in events[-1].message and "not enough memory" in events[-1].message
    assert events[-1].data["step"] == "fit"


# ----------------------------------------------------------------------------- jobs
def test_job_runs_and_reports_results(fake_sense, fresh_jobs, monkeypatch):
    monkeypatch.setattr(planner_mod, "_preferred", None)  # the rule planner, even if an LLM one is registered
    fake_sense.results = {
        "vector": (0.90, {"swapped": 0.2, "no_input": 0.2, "lookup": 0.9}),
        "text": (0.60, {}),
    }
    job_id = jobs.start(FitRequest("fake", "toy"))
    assert wait_for(lambda: jobs.get(job_id).status != "running")

    state = jobs.get(job_id)
    assert state.status == "done" and state.error is None
    assert state.installed == "vector:toy" and len(state.cards) == 2
    assert jobs.latest() is state
    assert state.to_dict()["events"][-1]["step"] == "done"
    assert jobs.get("missing") is None


def test_failing_job_is_marked_failed(fake_sense, fresh_jobs, monkeypatch):
    def broken(name):
        raise FileNotFoundError("gtzan_clap.npz is missing; run socket_pilot/clap_embed_gtzan.py first")

    monkeypatch.setattr(datasets, "load_labelled_set", broken)
    job_id = jobs.start(FitRequest("fake", "gtzan"))
    assert wait_for(lambda: jobs.get(job_id).status != "running")

    state = jobs.get(job_id)
    assert state.status == "failed"
    assert "gtzan_clap.npz is missing" in state.error
    assert state.events[-1].step == "error" and state.events[-1].status == "fail"
    assert state.installed is None and state.cards == []


def test_one_job_at_a_time_with_live_events(fresh_jobs, monkeypatch):
    release = threading.Event()

    def slow_run(request, emit, planner=None):
        emit(TraceEvent(step="inspect", status="start", message="Loading."))
        release.wait(5)
        return [], None

    monkeypatch.setattr(loop, "run_fit", slow_run)
    first = jobs.start(FitRequest("fake", "toy"))
    assert wait_for(lambda: len(jobs.get(first).events) == 1)  # a poller sees progress mid-run
    assert jobs.get(first).status == "running"

    with pytest.raises(RuntimeError, match="an agent run is already in progress"):
        jobs.start(FitRequest("fake", "toy"))

    release.set()
    assert wait_for(lambda: jobs.get(first).status == "done")
    second = jobs.start(FitRequest("fake", "toy"))
    assert second != first and jobs.latest().id == second
    assert wait_for(lambda: jobs.get(second).status == "done")
