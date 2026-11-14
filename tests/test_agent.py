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
from sense import bridges, datasets, models, registry, specialists, verify
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
        model=FakeModel(), results={}, installs=[], verify_calls=[], text_calls=[], vector_fits=0, earlier=[],
    )

    def fit_vector(model, dataset, idx, layer=None):
        state.vector_fits += 1
        assert list(idx) == list(dataset.fit_idx), "the vector bridge must be fitted on dataset.fit_idx"
        return FakeBridge("vector", f"vector:{dataset.name}")

    def make_text(model, dataset=None):
        state.text_calls.append(dataset)
        return FakeBridge("text", f"text:{dataset.name}" if dataset is not None else "text:captions")

    def build(kind, model, dataset, scope):
        if kind == "vector":
            return bridges.fit_vector_bridge(model, dataset, dataset.fit_idx)
        if kind == "text":
            return bridges.make_text_bridge(model, dataset if scope == "closed" else None)
        if kind == "tool":
            return FakeBridge("tool", f"tool:{dataset.name}")
        raise ValueError(f"Unsupported fake bridge kind: {kind}")

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
    monkeypatch.setattr(bridges, "build", build, raising=False)
    monkeypatch.setattr(bridges, "availability", lambda info, modality: [], raising=False)
    monkeypatch.setattr(verify, "verify", fake_verify)
    monkeypatch.setattr(registry, "install", lambda bridge, card: state.installs.append((bridge.name, card.verdict)))
    monkeypatch.setattr(registry, "installed", lambda name: state.earlier)  # senses installed before the run
    return state


EARLIER_GTZAN = {
    "bridge": "vector:gtzan", "kind": "vector", "modality": "audio", "dataset": "gtzan",
    "card": {"accuracy": 0.795, "dataset": "gtzan", "scope": "closed"},
}


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


def test_tool_card_uses_the_text_bars():
    request = FitRequest("fake", "toy")
    ok, why = RulePlanner().assess(request, make_card("tool:toy", "tool", 0.35, swapped=0.2))
    assert ok and "tool bridge" in why
    assert not RulePlanner().assess(request, make_card("tool:toy", "tool", 0.34, swapped=0.2))[0]
    assert not RulePlanner().assess(request, make_card("tool:toy", "tool", 0.60, swapped=0.55))[0]
    assert RulePlanner().assess(request, make_card("tool:toy", "tool", 0.60, swapped=0.50))[0]


def test_planner_registry(monkeypatch):
    monkeypatch.setattr(planner_mod, "_factories", dict(planner_mod._factories))
    monkeypatch.setattr(planner_mod, "_preferred", None)
    monkeypatch.setattr(planner_mod.config, "fake", lambda: False)
    assert isinstance(get_planner(), RulePlanner)

    class Other(RulePlanner):
        name = "other"

    register_planner("other", Other, preferred=True)
    assert isinstance(get_planner(), Other)
    assert type(get_planner("rules")) is RulePlanner
    with pytest.raises(ValueError, match="No planner is registered as 'nope'"):
        get_planner("nope")
    # in fake mode the default is always the rule planner, so tests never call an API
    monkeypatch.setattr(planner_mod.config, "fake", lambda: True)
    assert type(get_planner()) is RulePlanner
    assert isinstance(get_planner("other"), Other)


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


def test_refusal_names_the_earlier_sense_chat_keeps_using(fake_sense):
    fake_sense.earlier = [EARLIER_GTZAN]
    fake_sense.results = {
        "vector": (0.30, {"swapped": 0.2, "no_input": 0.2, "lookup": 0.3}),
        "text": (0.25, {"swapped": 0.2, "no_input": 0.2, "lookup": 0.3}),
    }
    _, installed, events = run(FitRequest("fake", "toy"))

    assert installed is None and fake_sense.installs == []
    [install] = [e for e in events if e.step == "install"]
    assert "still cannot take" not in install.message
    assert "keeps the earlier vector:gtzan sense (79.5% on gtzan, closed scope)" in install.message
    assert install.data["kept"] == "vector:gtzan"
    assert "Chat keeps using the earlier vector:gtzan sense" in events[-1].message
    assert events[-1].data == {"installed": None, "kept": "vector:gtzan"}


def test_install_names_the_sense_it_replaces(fake_sense):
    fake_sense.earlier = [EARLIER_GTZAN]
    fake_sense.results = {
        "vector": (0.95, {"swapped": 0.2, "no_input": 0.2, "lookup": 0.95, "unseen_classes": 0.22}),
        "text": (0.70, {"swapped": 0.2, "no_input": 0.2, "lookup": 0.95}),
    }
    _, installed, events = run(FitRequest("fake", "toy", scope="open"))

    assert installed == "text:captions"
    start = next(e for e in events if e.step == "install" and e.status == "start")
    assert "replacing the earlier vector:gtzan sense (79.5% on gtzan, closed scope)" in start.message
    assert start.data["replaced"] == "vector:gtzan"
    assert "It replaced the earlier vector:gtzan sense" in events[-1].message
    assert events[-1].data == {"installed": "text:captions", "replaced": "vector:gtzan"}


def test_closed_weight_model_skips_vector(fake_sense):
    fake_sense.model = FakeModel(open_weights=False)
    fake_sense.results = {"text": (0.60, {"swapped": 0.2, "no_input": 0.2, "lookup": 0.9})}
    cards, installed, _ = run(FitRequest("fake", "toy"))
    assert fake_sense.vector_fits == 0
    assert installed == "text:toy" and [c.bridge for c in cards] == ["text:toy"]


def test_closed_weight_model_tries_the_tool_bridge_when_available(fake_sense, monkeypatch):
    fake_sense.model = FakeModel(open_weights=False)
    calls = []

    def availability(info, modality):
        calls.append((info, modality))
        return [{"kind": "tool", "available": True, "why": ""}]

    monkeypatch.setattr(bridges, "availability", availability)
    fake_sense.results = {"text": (0.6, {"swapped": 0.2}), "tool": (0.9, {"swapped": 0.2})}
    cards, installed, _ = run(FitRequest("fake", "toy"))
    assert {c.kind for c in cards} == {"text", "tool"} and installed == "tool:toy"
    assert fake_sense.verify_calls == [("text:toy", None), ("tool:toy", None)]
    assert calls == [(fake_sense.model.info, "audio")]


def test_open_weight_model_does_not_try_the_tool_bridge(fake_sense, monkeypatch):
    monkeypatch.setattr(bridges, "availability", lambda info, modality: [
        {"kind": "tool", "available": True, "why": ""},
    ])
    fake_sense.results = {"vector": (0.9, {"swapped": 0.2}), "text": (0.6, {"swapped": 0.2})}
    cards, installed, _ = run(FitRequest("fake", "toy"))
    assert [c.kind for c in cards] == ["vector", "text"]
    assert installed == "vector:toy"


@pytest.mark.parametrize("availability_state", ["absent", "unavailable", "raises"])
def test_closed_weight_tool_requires_reported_availability(fake_sense, monkeypatch, availability_state):
    fake_sense.model = FakeModel(open_weights=False)
    if availability_state == "absent":
        monkeypatch.delattr(bridges, "availability")
    elif availability_state == "unavailable":
        monkeypatch.setattr(bridges, "availability", lambda info, modality: [
            {"kind": "tool", "available": False, "why": "no tool support"},
        ])
    else:
        def broken(info, modality):
            raise ValueError("availability unknown")

        monkeypatch.setattr(bridges, "availability", broken)
    fake_sense.results = {"text": (0.6, {"swapped": 0.2})}
    cards, installed, _ = run(FitRequest("fake", "toy"))
    assert [c.kind for c in cards] == ["text"] and installed == "text:toy"


@pytest.mark.parametrize("scope", ["closed", "open"])
def test_without_bridges_build_the_old_branches_still_work(fake_sense, monkeypatch, scope):
    monkeypatch.delattr(bridges, "build")
    fake_sense.results = {"vector": (0.9, {"swapped": 0.2, "unseen_classes": 0.8}),
                          "text": (0.6, {"swapped": 0.2})}
    cards, installed, _ = run(FitRequest("fake", "toy", scope=scope))
    assert [c.kind for c in cards] == ["vector", "text"] and installed == "vector:toy"
    assert fake_sense.vector_fits == 1
    assert (fake_sense.text_calls[0] is None) == (scope == "open")


def test_without_bridges_build_tool_is_skipped(fake_sense, monkeypatch):
    fake_sense.model = FakeModel(open_weights=False)
    monkeypatch.delattr(bridges, "build")
    monkeypatch.setattr(bridges, "availability", lambda info, modality: [
        {"kind": "tool", "available": True, "why": ""},
    ])
    fake_sense.results = {"text": (0.6, {"swapped": 0.2})}
    cards, installed, events = run(FitRequest("fake", "toy"))
    assert [c.kind for c in cards] == ["text"] and installed == "text:toy"
    assert any(e.step == "fit" and e.status == "fail" and "can build only vector and text" in e.message
               for e in events)


@pytest.mark.parametrize("scope", ["closed", "open"])
def test_build_receives_each_kind_and_scope_and_only_vector_gets_control(fake_sense, monkeypatch, scope):
    class ProposesTool(RulePlanner):
        def plan(self, request, model, dataset):
            return ["tool", "vector", "tool"]

    calls = []

    def build(kind, model, dataset, build_scope):
        calls.append((kind, model, dataset.name, build_scope))
        return FakeBridge(kind, f"{kind}:built")

    monkeypatch.setattr(bridges, "build", build)
    fake_sense.results = {"vector": (0.9, {"swapped": 0.2, "unseen_classes": 0.8}),
                          "text": (0.6, {"swapped": 0.2}), "tool": (0.7, {"swapped": 0.2})}
    cards, installed = loop.run_fit(FitRequest("fake", "toy", scope=scope), lambda event: None,
                                    planner=ProposesTool())
    assert calls == [(kind, fake_sense.model, "toy", scope) for kind in ["tool", "vector", "text"]]
    assert fake_sense.verify_calls == [("tool:built", None), ("vector:built", "text:built"), ("text:built", None)]
    assert installed == "vector:built" and len(cards) == 3


def test_refusal_distinguishes_passed_but_unselected_from_ineligible_cards(fake_sense):
    class Refuses(RulePlanner):
        def decide(self, request, cards):
            return Decision(None, "No bridge selected.")

    fake_sense.results = {"vector": (0.3, {"swapped": 0.2}), "text": (0.6, {"swapped": 0.2})}
    request = FitRequest("fake", "toy")
    events = []
    cards, installed = loop.run_fit(request, events.append, planner=Refuses())
    assert installed is None and fake_sense.installs == []
    vector, text = cards
    assert all(c.verdict == "rejected" for c in cards)
    assert vector.reason == RulePlanner().assess(request, vector)[1]
    assert "passed" in text.reason and "not selected" in text.reason and "no bridge selected" in text.reason.lower()
    assert "None was chosen instead" not in text.reason
    assert events[-1].step == "done" and "the planner selected no bridge" in events[-1].message
    assert "failed verification" not in events[-1].message and "passed verification" not in events[-1].message


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


def test_plan_event_lists_unavailable_bridges(fake_sense, monkeypatch):
    rows = [{"kind": "native", "available": False, "why": "Fake model does not read audio"},
            {"kind": "vector", "available": True, "why": ""},
            {"kind": "image", "available": False, "why": "no audio-to-image renderer"}]
    monkeypatch.setattr(bridges, "availability", lambda info, modality: rows, raising=False)
    fake_sense.results = {"vector": (0.85, {"swapped": 0.2}), "text": (0.6, {"swapped": 0.2})}
    events = run(FitRequest("fake", "toy"))[2]
    plan_ok = next(e for e in events if e.step == "plan" and e.status == "ok")
    assert plan_ok.data["unavailable"] == [{"kind": "native", "why": "Fake model does not read audio"},
                                           {"kind": "image", "why": "no audio-to-image renderer"}]


def test_fit_events_name_the_specialist_and_verify_events_are_timed(fake_sense, monkeypatch):
    spec = {"name": "fake-clap", "label": "Fake CLAP", "modality": "audio", "dim": 64, "embed_seconds_per_file": 0.0}
    monkeypatch.setattr(specialists, "describe", lambda modality: spec, raising=False)
    fake_sense.results = {"vector": (0.85, {"swapped": 0.2}), "text": (0.6, {"swapped": 0.2})}
    events = run(FitRequest("fake", "toy"))[2]
    fit_ok = [e for e in events if e.step == "fit" and e.status == "ok"]
    verify_ok = [e for e in events if e.step == "verify" and e.status == "ok"]
    assert len(fit_ok) == 2 and len(verify_ok) == 2
    assert all(e.data["specialist"] == spec for e in fit_ok)
    assert all(e.data["seconds"] >= 0 for e in verify_ok)


def test_trace_facts_default_when_sense_lacks_them(fake_sense, monkeypatch):
    monkeypatch.delattr(bridges, "availability", raising=False)
    monkeypatch.delattr(specialists, "describe", raising=False)
    fake_sense.results = {"vector": (0.85, {"swapped": 0.2}), "text": (0.6, {"swapped": 0.2})}
    events = run(FitRequest("fake", "toy"))[2]
    assert next(e for e in events if e.step == "plan" and e.status == "ok").data["unavailable"] == []
    fit_ok = [e for e in events if e.step == "fit" and e.status == "ok"]
    assert fit_ok and all(e.data["specialist"] is None for e in fit_ok)


def test_on_cards_runs_after_verify_and_t0_sets_the_clock(fake_sense):
    fake_sense.results = {"vector": (0.85, {"swapped": 0.2}), "text": (0.6, {"swapped": 0.2})}
    got = []
    events = []
    seen_at = []  # how many events had been emitted when on_cards ran

    def on_cards(cards):
        got.append(cards)
        seen_at.append(len(events))

    loop.run_fit(FitRequest("fake", "toy"), events.append, on_cards=on_cards, t0=time.monotonic() - 100)
    assert len(got) == 1 and len(got[0]) == 2
    [n] = seen_at
    assert (events[n - 1].step, events[n - 1].status) == ("verify", "ok")
    assert (events[n].step, events[n].status) == ("decide", "start")
    assert all(e.t >= 100 for e in events)


def test_a_raising_specialist_describe_gives_none_and_the_run_finishes(fake_sense, monkeypatch):
    def describe(modality):
        raise ValueError(f"No specialist is known for {modality}.")

    monkeypatch.setattr(specialists, "describe", describe, raising=False)
    fake_sense.results = {"vector": (0.85, {"swapped": 0.2}), "text": (0.6, {"swapped": 0.2})}
    cards, installed, events = run(FitRequest("fake", "toy"))
    fit_ok = [e for e in events if e.step == "fit" and e.status == "ok"]
    assert len(fit_ok) == 2 and all(e.data["specialist"] is None for e in fit_ok)
    assert events[-1].step == "done" and events[-1].status == "ok" and installed == "vector:toy"


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


def test_listener_gets_every_event_then_the_cards(fake_sense, fresh_jobs, monkeypatch):
    monkeypatch.setattr(planner_mod, "_preferred", None)
    monkeypatch.setattr(jobs, "_listeners", [])
    fake_sense.results = {"vector": (0.85, {"swapped": 0.2}), "text": (0.6, {"swapped": 0.2})}
    seen = []
    jobs.add_listener(lambda job_id, event, cards: seen.append((job_id, event, cards)))
    job_id = jobs.start(FitRequest("fake", "toy"))
    assert wait_for(lambda: jobs.get(job_id).status != "running")
    assert wait_for(lambda: bool(seen) and seen[-1][1] is None)  # the cards call follows the status flip
    job = jobs.get(job_id)
    events = [e for _, e, _ in seen[:-1]]
    assert all(c is None for _, _, c in seen[:-1]) and events == job.events
    assert seen[-1][1] is None and [c.bridge for c in seen[-1][2]] == [c.bridge for c in job.cards]
    assert {j for j, _, _ in seen} == {job.id}


def test_listener_gets_empty_cards_when_the_job_fails(fake_sense, fresh_jobs, monkeypatch):
    def broken(name):
        raise FileNotFoundError("gtzan_clap.npz is missing; run socket_pilot/clap_embed_gtzan.py first")

    monkeypatch.setattr(datasets, "load_labelled_set", broken)
    monkeypatch.setattr(jobs, "_listeners", [])
    seen = []
    jobs.add_listener(lambda job_id, event, cards: seen.append((job_id, event, cards)))
    job_id = jobs.start(FitRequest("fake", "gtzan"))
    assert wait_for(lambda: jobs.get(job_id).status != "running")
    assert wait_for(lambda: bool(seen) and seen[-1][1] is None)
    job = jobs.get(job_id)
    assert job.status == "failed"
    assert [e for _, e, _ in seen[:-1]] == job.events and job.events[-1].step == "error"
    assert seen[-1] == (job.id, None, [])


def test_a_raising_listener_does_not_stop_the_run(fake_sense, fresh_jobs, monkeypatch):
    monkeypatch.setattr(planner_mod, "_preferred", None)
    monkeypatch.setattr(jobs, "_listeners", [])
    fake_sense.results = {"vector": (0.85, {"swapped": 0.2}), "text": (0.6, {"swapped": 0.2})}
    jobs.add_listener(lambda *a: (_ for _ in ()).throw(RuntimeError("boom")))
    seen = []
    jobs.add_listener(lambda job_id, event, cards: seen.append((job_id, event, cards)))
    job_id = jobs.start(FitRequest("fake", "toy"))
    assert wait_for(lambda: jobs.get(job_id).status != "running")
    assert wait_for(lambda: bool(seen) and seen[-1][1] is None)  # a later listener still hears everything
    job = jobs.get(job_id)
    assert job.status == "done"
    assert job.error is None and len(job.cards) == 2 and job.events[-1].step == "done"
    assert [e for _, e, _ in seen[:-1]] == job.events


def test_remove_listener_stops_calls(fake_sense, fresh_jobs, monkeypatch):
    monkeypatch.setattr(planner_mod, "_preferred", None)
    monkeypatch.setattr(jobs, "_listeners", [])
    fake_sense.results = {"vector": (0.85, {"swapped": 0.2}), "text": (0.6, {"swapped": 0.2})}
    seen = []

    def listener(job_id, event, cards):
        seen.append((job_id, event, cards))

    jobs.add_listener(listener)
    jobs.remove_listener(listener)
    jobs.remove_listener(listener)  # removing one that is not registered is a no-op
    job_id = jobs.start(FitRequest("fake", "toy"))
    assert wait_for(lambda: jobs.get(job_id).status != "running")
    assert jobs.get(job_id).status == "done"
    assert seen == []


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


def test_open_scope_vector_must_hold_up_on_unseen_classes():
    """51% on unseen classes clears the chance bar but is far below 96.5% on fitted classes: not acceptable."""
    from sense.contracts import ReportCard

    card = ReportCard(
        model="m", modality="audio", dataset="esc50", scope="open", bridge="vector:esc50", kind="vector",
        n=200, chance=0.2, accuracy=0.965, controls={"swapped": 0.18, "unseen_classes": 0.512},
    )
    ok, sentence = RulePlanner().assess(FitRequest("m", "esc50", scope="open"), card)
    assert not ok and "more than 20 points below" in sentence
    card.controls["unseen_classes"] = 0.80
    assert RulePlanner().assess(FitRequest("m", "esc50", scope="open"), card)[0]
