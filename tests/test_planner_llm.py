"""Offline planner tests; registry stubs are patched before importing the adapter."""

import importlib
import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

import agent.planner
from sense import config
from sense.contracts import Decision, FitRequest, LabelledSet, ModelInfo, Planner, ReportCard


def completion(answer, *, finish="stop", refusal=None):
    content = answer if isinstance(answer, str) else json.dumps(answer)
    return SimpleNamespace(choices=[SimpleNamespace(
        finish_reason=finish, message=SimpleNamespace(content=content, refusal=refusal),
    )])


@pytest.fixture
def api(monkeypatch):
    registration = Mock()
    monkeypatch.setattr(agent.planner, "register_planner", registration)
    monkeypatch.setattr(config, "openai_key", lambda: "test-key")
    monkeypatch.delenv("SENSE_LLM_MODEL", raising=False)
    previous = sys.modules.pop("agent.planner_llm", None)
    previous_attribute = getattr(agent, "planner_llm", None)
    module = importlib.import_module("agent.planner_llm")
    client = Mock()
    factory = Mock(return_value=client)
    monkeypatch.setattr(module, "OpenAI", factory)
    rules = SimpleNamespace(
        plan=Mock(return_value=["text"]),
        decide=Mock(return_value=Decision("text:captions", "Rule result: 70% accuracy vs 20% swapped.")),
    )
    monkeypatch.setattr(module, "RulePlanner", lambda: rules)
    yield SimpleNamespace(module=module, client=client, factory=factory, rules=rules, registration=registration)
    sys.modules.pop("agent.planner_llm", None)
    if previous is not None:
        sys.modules["agent.planner_llm"] = previous
    if previous_attribute is not None:
        agent.planner_llm = previous_attribute
    else:
        delattr(agent, "planner_llm")


@pytest.fixture
def request_data():
    return FitRequest("open-model", "sounds")


@pytest.fixture
def dataset():
    return LabelledSet(
        name="sounds", title="Tiny sounds", modality="audio", question="Which sound?",
        labels=["dog", "cat"], y=np.array([0, 1] * 4), emb=np.ones((8, 3)),
        label_text_emb=np.ones((2, 3)), fit_idx=np.arange(4), dev_idx=np.array([4, 5]),
        test_idx=np.array([6, 7]),
    )


@pytest.fixture
def model():
    return ModelInfo("open-model", "Open model", ["text"], True)


@pytest.fixture
def cards():
    return [
        ReportCard("open-model", "audio", "sounds", "closed", "vector:sounds", "vector", 100, .2, .85, {"swapped": .2, "lookup": .84}),
        ReportCard("open-model", "audio", "sounds", "closed", "text:captions", "text", 100, .2, .7, {"swapped": .2}),
    ]


def test_registration_protocol_and_model_override(api, monkeypatch):
    api.registration.assert_called_once_with("llm", api.module.LLMPlanner, preferred=True)
    monkeypatch.setenv("SENSE_LLM_MODEL", "custom-model")
    planner = api.module.LLMPlanner()
    assert isinstance(planner, Planner)
    assert planner.model == "custom-model"
    api.factory.assert_not_called()


def test_no_key_does_not_register(api, monkeypatch):
    api.registration.reset_mock()
    monkeypatch.setattr(config, "openai_key", lambda: None)
    importlib.reload(api.module)
    api.registration.assert_not_called()


def test_plan_sends_model_and_short_dataset_summary(api, request_data, model, dataset):
    api.client.chat.completions.create.return_value = completion({"bridges": ["vector", "text"]})
    assert api.module.LLMPlanner().plan(request_data, model, dataset) == ["vector", "text"]
    kwargs = api.client.chat.completions.create.call_args.kwargs
    sent = json.loads(kwargs["messages"][1]["content"])
    assert sent["model"] == model.to_dict()
    assert sent["dataset"]["n_test"] == 2
    assert sent["dataset"]["labels"] == dataset.labels
    assert "emb" not in sent["dataset"]
    assert kwargs["model"] == "gpt-5.4-mini"
    assert kwargs["temperature"] == 0
    assert kwargs["reasoning_effort"] == "none"
    api.rules.plan.assert_not_called()


def test_api_model_can_only_plan_text(api, request_data, model, dataset):
    model.open_weights = False
    api.client.chat.completions.create.return_value = completion({"bridges": ["text"]})
    assert api.module.LLMPlanner().plan(request_data, model, dataset) == ["text"]
    kwargs = api.client.chat.completions.create.call_args.kwargs
    assert json.loads(kwargs["messages"][1]["content"])["allowed"] == ["text"]
    api.client.chat.completions.create.return_value = completion({"bridges": ["vector"]})
    assert api.module.LLMPlanner().plan(request_data, model, dataset) == ["text"]
    api.rules.plan.assert_called_once_with(request_data, model, dataset)


@pytest.mark.parametrize("answer", [
    "not json", [], {"bridges": []}, {"bridges": "text"}, {"bridges": ["tool"]},
    {"bridges": ["text", "text"]}, {"bridges": [1]}, {"bridges": [None]},
    {"bridges": ["text"], "extra": True}, {"wrong": ["text"]},
])
def test_invalid_plan_falls_back(api, request_data, model, dataset, answer):
    api.client.chat.completions.create.return_value = completion(answer)
    assert api.module.LLMPlanner().plan(request_data, model, dataset) == ["text"]
    api.rules.plan.assert_called_once_with(request_data, model, dataset)


@pytest.mark.parametrize("stage", ["client", "request", "missing_key"])
def test_plan_api_errors_fall_back(api, request_data, model, dataset, monkeypatch, stage):
    if stage == "client":
        api.factory.side_effect = RuntimeError("client error")
    elif stage == "request":
        api.client.chat.completions.create.side_effect = RuntimeError("API error")
    else:
        monkeypatch.setattr(config, "openai_key", lambda: None)
    assert api.module.LLMPlanner().plan(request_data, model, dataset) == ["text"]
    api.rules.plan.assert_called_once()


def test_decide_sends_complete_cards_and_returns_grounded_reason(api, request_data, cards):
    reason = "Vector scored 85% accuracy against 20% swapped input."
    api.client.chat.completions.create.return_value = completion({"install": "vector:sounds", "reason": reason})
    assert api.module.LLMPlanner().decide(request_data, cards) == Decision("vector:sounds", reason)
    sent = json.loads(api.client.chat.completions.create.call_args.kwargs["messages"][1]["content"])
    assert sent["cards"] == [card.to_dict() for card in cards]
    api.rules.decide.assert_not_called()


def test_decide_can_refuse(api, request_data, cards):
    reason = "Although accuracy was 85% vs 20% swapped, the request needs unseen-class evidence."
    api.client.chat.completions.create.return_value = completion({"install": None, "reason": reason})
    assert api.module.LLMPlanner().decide(request_data, cards) == Decision(None, reason)
    api.rules.decide.assert_not_called()


@pytest.mark.parametrize("answer", [
    "not json", [], {"install": "unknown", "reason": "85% vs 20%."},
    {"install": False, "reason": "85% vs 20%."}, {"install": "vector:sounds", "reason": ""},
    {"install": "vector:sounds", "reason": None}, {"install": "vector:sounds", "reason": "Looks good."},
    {"install": "vector:sounds", "reason": "99% vs 1%."},
    {"install": "vector:sounds", "reason": "85% vs 20%. Good. Install."},
    {"install": "vector:sounds", "reason": "85% vs 20%.", "extra": True},
    {"reason": "85% vs 20%."},
])
def test_invalid_decision_falls_back(api, request_data, cards, answer):
    api.client.chat.completions.create.return_value = completion(answer)
    assert api.module.LLMPlanner().decide(request_data, cards) == api.rules.decide.return_value
    api.rules.decide.assert_called_once_with(request_data, cards)


@pytest.mark.parametrize("response", [
    completion({"install": "vector:sounds", "reason": "85% vs 20%."}, finish="length"),
    completion({"install": "vector:sounds", "reason": "85% vs 20%."}, refusal="No"),
    SimpleNamespace(choices=[]),
])
def test_incomplete_or_refused_decision_falls_back(api, request_data, cards, response):
    api.client.chat.completions.create.return_value = response
    assert api.module.LLMPlanner().decide(request_data, cards) == api.rules.decide.return_value


def test_decide_api_error_falls_back(api, request_data, cards):
    api.client.chat.completions.create.side_effect = RuntimeError("API unavailable")
    assert api.module.LLMPlanner().decide(request_data, cards) == api.rules.decide.return_value


def test_floor_cannot_be_overridden_by_llm_or_rules(api, request_data, cards):
    cards[0].accuracy = .29
    api.client.chat.completions.create.return_value = completion({
        "install": "vector:sounds", "reason": "Vector accuracy is 29% against 20% swapped.",
    })
    assert api.module.LLMPlanner().decide(request_data, cards).install == "text:captions"
    api.rules.decide.assert_called_once_with(request_data, [cards[1]])


def test_floor_is_inclusive_at_exact_ten_points(api, request_data, cards):
    cards[0].accuracy = .3
    api.client.chat.completions.create.return_value = completion({
        "install": "vector:sounds", "reason": "Vector accuracy is 30% against 20% swapped.",
    })
    assert api.module.LLMPlanner().decide(request_data, cards).install == "vector:sounds"
    api.rules.decide.assert_not_called()


@pytest.mark.parametrize("swapped", [None, float("nan"), float("inf"), -.1, 1.1])
def test_invalid_swapped_control_is_never_installed(api, request_data, cards, swapped):
    cards[0].controls["swapped"] = swapped
    api.client.chat.completions.create.side_effect = RuntimeError("API unavailable")
    assert api.module.LLMPlanner().decide(request_data, cards).install == "text:captions"
    api.rules.decide.assert_called_once_with(request_data, [cards[1]])


def test_fallback_cannot_install_unsafe_text_card(api, request_data, cards):
    cards[1].accuracy = .25
    api.client.chat.completions.create.side_effect = RuntimeError("API unavailable")
    api.module.LLMPlanner().decide(request_data, cards)
    api.rules.decide.assert_called_once_with(request_data, [cards[0]])


def test_empty_and_all_unsafe_cards_refuse(api, request_data, cards):
    assert api.module.LLMPlanner().decide(request_data, []).install is None
    api.client.chat.completions.create.assert_not_called()
    for card in cards:
        card.accuracy = .2
    api.client.chat.completions.create.side_effect = RuntimeError("API unavailable")
    assert api.module.LLMPlanner().decide(request_data, cards).install is None


def test_duplicate_names_do_not_reach_api(api, request_data, cards):
    cards[1].bridge = cards[0].bridge
    api.module.LLMPlanner().decide(request_data, cards)
    api.client.chat.completions.create.assert_not_called()
