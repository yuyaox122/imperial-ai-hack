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
def api(monkeypatch, tmp_path):
    monkeypatch.setenv("SENSE_HOME", str(tmp_path))
    monkeypatch.setattr(config, "fake", lambda: False)  # Explicit adapter tests use a mocked client.
    registration = Mock()
    monkeypatch.setattr(agent.planner, "register_planner", registration)
    monkeypatch.setattr(config, "openai_key", lambda: "test-key")
    monkeypatch.delenv("SENSE_LLM_MODEL", raising=False)
    monkeypatch.delenv("SENSE_LLM_PROVIDER", raising=False)
    previous = sys.modules.pop("agent.planner_llm", None)
    previous_attribute = getattr(agent, "planner_llm", None)
    module = importlib.import_module("agent.planner_llm")
    client = Mock()
    factory = Mock(return_value=client)
    monkeypatch.setattr(module, "OpenAI", factory)
    rules = SimpleNamespace(
        plan=Mock(return_value=["text"]),
        decide=Mock(return_value=Decision("text:captions", "Rule result: 70% accuracy vs 20% swapped.")),
        assess=Mock(side_effect=agent.planner.RulePlanner().assess),  # the real shared acceptance gate
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
    assert kwargs["model"] == "gpt-6.1-terra"
    assert kwargs["temperature"] == 0
    assert kwargs["reasoning_effort"] == "none"
    api.rules.plan.assert_not_called()


def test_api_model_can_only_plan_text(api, request_data, model, dataset):
    model.open_weights = False
    api.client.chat.completions.create.return_value = completion({"bridges": ["text"]})
    assert api.module.LLMPlanner().plan(request_data, model, dataset) == ["text"]
    kwargs = api.client.chat.completions.create.call_args.kwargs
    assert json.loads(kwargs["messages"][1]["content"])["allowed"] == ["text"]
    request_data.scope = "open"  # A new request exercises API validation rather than a memo hit.
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
    planner = api.module.LLMPlanner()
    decision = planner.decide(request_data, cards)
    assert decision.install == "vector:sounds"
    assert planner.assess(request_data, cards[0])[1] in decision.reason
    sent = json.loads(api.client.chat.completions.create.call_args.kwargs["messages"][1]["content"])
    assert sent["cards"] == [{key: value for key, value in card.to_dict().items()
                              if key not in {"created", "verdict", "reason"}} for card in cards]
    api.rules.decide.assert_not_called()


def test_decide_can_refuse(api, request_data, cards):
    reason = "Although accuracy was 85% vs 20% swapped, the request needs unseen-class evidence."
    api.client.chat.completions.create.return_value = completion({"install": None, "reason": reason})
    decision = api.module.LLMPlanner().decide(request_data, cards)
    assert decision.install is None
    assert "chose not to install" in decision.reason
    assert "vector:sounds passed" in decision.reason
    assert "text:captions passed" in decision.reason
    assert "needs unseen-class evidence" not in decision.reason
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


def test_vector_below_the_shared_gate_is_not_installed(api, request_data, cards):
    """Ten points over swapped clears the floor but not the rule planner's 25-point vector margin."""
    cards[0].accuracy = .3
    api.client.chat.completions.create.return_value = completion({
        "install": "vector:sounds", "reason": "Vector accuracy is 30% against 20% swapped.",
    })
    assert api.module.LLMPlanner().decide(request_data, cards).install == "text:captions"
    api.rules.decide.assert_called_once()


def test_text_gate_is_inclusive_at_the_bar(api, request_data, cards):
    cards[1].accuracy = .35  # chance .20 + .15, and above swapped .20 + .10
    api.client.chat.completions.create.return_value = completion({
        "install": "text:captions", "reason": "Text accuracy is 35% against 20% swapped.",
    })
    assert api.module.LLMPlanner().decide(request_data, cards).install == "text:captions"
    api.rules.decide.assert_not_called()


def test_adapter_cards_are_held_to_native_and_unseen_categories(api, request_data, cards):
    """The LLM may not install an adapter that adds nothing to the raw file or fails on new categories."""
    native = ReportCard("open-model", "audio", "sounds", "closed", "adapter:a", "text", 100, .2, .9,
                        {"swapped": .2, "native": .85})
    narrow = ReportCard("open-model", "audio", "sounds", "closed", "adapter:b", "image", 100, .2, .95,
                        {"swapped": .2, "unseen_categories": .4}, control_chance={"unseen_categories": .3333})
    planner = api.module.LLMPlanner()
    assert not planner.assess(request_data, native)[0] and not planner.assess(request_data, narrow)[0]
    api.client.chat.completions.create.return_value = completion({
        "install": "adapter:b", "reason": "Adapter b scored 95% accuracy against 20% swapped input.",
    })
    assert planner.decide(request_data, [native, narrow, cards[1]]).install == "text:captions"
    instruction = api.client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
    assert "native + 0.10" in instruction and "unseen_categories" in instruction
    sent = json.loads(api.client.chat.completions.create.call_args.kwargs["messages"][1]["content"])
    assert sent["cards"][1]["control_chance"] == {"unseen_categories": .3333}


def test_open_scope_vector_needs_unseen_classes(api, cards):
    """For open scope the model may not install a vector bridge that fails on classes it was not fitted on."""
    request = FitRequest("open-model", "sounds", scope="open")
    cards[0].controls["unseen_classes"] = .21
    api.client.chat.completions.create.return_value = completion({
        "install": "vector:sounds", "reason": "Vector scored 85% accuracy against 20% swapped input.",
    })
    assert api.module.LLMPlanner().decide(request, cards).install == "text:captions"
    api.rules.decide.assert_called_once()


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


@pytest.mark.parametrize("install", [None, "text:gtzan"])
def test_false_llm_arithmetic_is_not_published(api, install):
    request = FitRequest("qwen", "gtzan")
    card = ReportCard("qwen", "audio", "gtzan", "closed", "text:gtzan", "text", 40, .1, .475, {"swapped": .2})
    false_reason = "Text accuracy 47.5% fails the required gain over 20% swapped input."
    api.client.chat.completions.create.return_value = completion({"install": install, "reason": false_reason})
    planner = api.module.LLMPlanner()
    decision = planner.decide(request, [card])
    assert decision.install == install  # A model may still decline an eligible bridge.
    assert false_reason not in decision.reason
    assert "text:gtzan passed" in decision.reason
    assert "47.5%" in decision.reason and "20.0%" in decision.reason
    assert planner.assess(request, card)[0] is True
    api.rules.decide.assert_not_called()


def test_llm_assessment_distinguishes_eligible_and_failed_cards(api, request_data, cards):
    from agent.loop import _apply_verdicts

    cards[0].accuracy = .4  # Clears ten points, but fails the vector's 25-point gate.
    api.client.chat.completions.create.return_value = completion({
        "install": None, "reason": "Text accuracy 70% is below 20% swapped input.",
    })
    planner = api.module.LLMPlanner()
    decision = planner.decide(request_data, cards)
    _apply_verdicts(planner, request_data, cards, decision)
    assert "vector:sounds failed" in cards[0].reason
    assert "text:captions passed" in cards[1].reason
    assert "below 20%" not in cards[1].reason


@pytest.mark.parametrize("swapped", [None, float("nan"), float("inf"), -.1, 1.1, .65])
def test_llm_assessment_includes_mandatory_floor(api, request_data, cards, swapped):
    cards[1].controls["swapped"] = swapped
    ok, reason = api.module.LLMPlanner().assess(request_data, cards[1])
    assert not ok
    assert "failed" in reason and "swapped" in reason


def test_plan_order_is_canonical(api, request_data, model, dataset):
    api.client.chat.completions.create.return_value = completion({"bridges": ["text", "vector"]})
    assert api.module.ORDER == ["vector", "text"]
    assert api.module.LLMPlanner().plan(request_data, model, dataset) == ["vector", "text"]


def test_identical_requests_share_a_persistent_plan(api, request_data, model, dataset, tmp_path):
    api.client.chat.completions.create.side_effect = [
        completion({"bridges": ["vector"]}), completion({"bridges": ["text"]}),
    ]
    first = api.module.LLMPlanner().plan(request_data, model, dataset)
    second = api.module.LLMPlanner().plan(request_data, model, dataset)
    assert first == second == ["vector"]
    assert api.client.chat.completions.create.call_count == 1
    assert len(list((tmp_path / "plans").glob("*.json"))) == 1


@pytest.mark.parametrize("install", ["vector:sounds", None])
def test_identical_cards_share_a_persistent_decision(api, request_data, cards, install):
    api.client.chat.completions.create.return_value = completion({"install": install, "reason": "85% vs 20%."})
    first = api.module.LLMPlanner().decide(request_data, cards)
    second = api.module.LLMPlanner().decide(request_data, cards)
    assert first == second
    assert first.install == install
    assert api.client.chat.completions.create.call_count == 1


@pytest.mark.parametrize("change", ["request", "model", "dataset", "llm_model"])
def test_changed_plan_inputs_miss_cache(api, request_data, model, dataset, monkeypatch, change):
    api.client.chat.completions.create.return_value = completion({"bridges": ["vector"]})
    api.module.LLMPlanner().plan(request_data, model, dataset)
    if change == "request":
        request_data.scope = "open"
    elif change == "model":
        model.label = "Updated model"
    elif change == "dataset":
        dataset.title = "Updated dataset"
    else:
        monkeypatch.setenv("SENSE_LLM_MODEL", "another-model")
    api.module.LLMPlanner().plan(request_data, model, dataset)
    assert api.client.chat.completions.create.call_count == 2


def test_changed_cards_miss_cache(api, request_data, cards):
    api.client.chat.completions.create.side_effect = [
        completion({"install": "vector:sounds", "reason": "85% vs 20%."}),
        completion({"install": "text:captions", "reason": "70% vs 20%."}),
    ]
    assert api.module.LLMPlanner().decide(request_data, cards).install == "vector:sounds"
    cards[0].accuracy = .84
    assert api.module.LLMPlanner().decide(request_data, cards).install == "text:captions"
    assert api.client.chat.completions.create.call_count == 2


@pytest.mark.parametrize("method", ["plan", "decide"])
@pytest.mark.parametrize("failure", [RuntimeError("API unavailable"), completion({"invalid": True})])
def test_fallback_answers_are_not_memoised(api, request_data, model, dataset, cards, tmp_path, method, failure):
    good = {"bridges": ["vector"]} if method == "plan" else {"install": "vector:sounds", "reason": "85% vs 20%."}
    api.client.chat.completions.create.side_effect = [failure, completion(good)]
    args = (request_data, model, dataset) if method == "plan" else (request_data, cards)
    getattr(api.module.LLMPlanner(), method)(*args)
    assert not list((tmp_path / "plans").glob("*.json"))
    result = getattr(api.module.LLMPlanner(), method)(*args)
    assert result == ["vector"] if method == "plan" else result.install == "vector:sounds"
    assert api.client.chat.completions.create.call_count == 2


def test_cached_decision_cannot_bypass_a_changed_gate(api, request_data, cards):
    api.client.chat.completions.create.return_value = completion({"install": "vector:sounds", "reason": "85% vs 20%."})
    assert api.module.LLMPlanner().decide(request_data, cards).install == "vector:sounds"
    real_assess = agent.planner.RulePlanner().assess
    api.rules.assess.side_effect = lambda request, card: (
        (False, "Vector no longer passes the current gate.") if card.kind == "vector" else real_assess(request, card)
    )
    assert api.module.LLMPlanner().decide(request_data, cards).install == "text:captions"


def test_json_mapping_order_does_not_change_memo_identity(api, request_data, cards):
    api.client.chat.completions.create.return_value = completion({"install": "vector:sounds", "reason": "85% vs 20%."})
    first = api.module.LLMPlanner().decide(request_data, cards)
    cards[0].controls = dict(reversed(list(cards[0].controls.items())))
    assert api.module.LLMPlanner().decide(request_data, cards) == first
    assert api.client.chat.completions.create.call_count == 1


def test_cache_write_failure_preserves_validated_decision(api, request_data, cards, tmp_path):
    (tmp_path / "plans").write_text("Not a directory")
    api.client.chat.completions.create.return_value = completion({"install": None, "reason": "85% vs 20%."})
    assert api.module.LLMPlanner().decide(request_data, cards).install is None
    api.rules.decide.assert_not_called()


def test_cache_hit_rechecks_gate_after_lookup(api, request_data, cards, monkeypatch):
    api.client.chat.completions.create.return_value = completion({"install": "vector:sounds", "reason": "85% vs 20%."})
    assert api.module.LLMPlanner().decide(request_data, cards).install == "vector:sounds"
    planner = api.module.LLMPlanner()
    lookup = planner._answer
    real_assess = agent.planner.RulePlanner().assess

    def change_gate_during_lookup(*args):
        answer = lookup(*args)
        api.rules.assess.side_effect = lambda request, card: (
            (False, "Changed gate rejects vector.") if card.kind == "vector" else real_assess(request, card)
        )
        return answer

    monkeypatch.setattr(planner, "_answer", change_gate_during_lookup)
    assert planner.decide(request_data, cards).install == "text:captions"
    assert api.client.chat.completions.create.call_count == 1


def test_corrupt_memo_is_a_cache_miss(api, request_data, model, dataset, tmp_path):
    api.client.chat.completions.create.return_value = completion({"bridges": ["vector"]})
    api.module.LLMPlanner().plan(request_data, model, dataset)
    memo = next((tmp_path / "plans").glob("*.json"))
    memo.write_text("{truncated")
    assert api.module.LLMPlanner().plan(request_data, model, dataset) == ["vector"]
    assert api.client.chat.completions.create.call_count == 2
    assert json.loads(memo.read_text()) == {"bridges": ["vector"]}


def test_fake_mode_uses_rules_even_with_live_memos(api, request_data, model, dataset, cards, monkeypatch):
    api.client.chat.completions.create.side_effect = [
        completion({"bridges": ["vector"]}),
        completion({"install": "vector:sounds", "reason": "85% vs 20%."}),
    ]
    planner = api.module.LLMPlanner()
    assert planner.plan(request_data, model, dataset) == ["vector"]
    assert planner.decide(request_data, cards).install == "vector:sounds"
    monkeypatch.setattr(config, "fake", lambda: True)
    api.client.chat.completions.create.reset_mock()
    api.factory.reset_mock()
    planner = api.module.LLMPlanner()
    assert planner.plan(request_data, model, dataset) == ["text"]
    assert planner.decide(request_data, cards) == api.rules.decide.return_value
    request_data.dataset = "uncached"
    planner.plan(request_data, model, dataset)
    planner.decide(request_data, cards)
    api.client.chat.completions.create.assert_not_called()
    api.factory.assert_not_called()


def test_changed_cache_version_misses(api, request_data, model, dataset, monkeypatch):
    api.client.chat.completions.create.return_value = completion({"bridges": ["vector"]})
    api.module.LLMPlanner().plan(request_data, model, dataset)
    monkeypatch.setattr(api.module, "_MEMO_VERSION", api.module._MEMO_VERSION + 1)
    api.module.LLMPlanner().plan(request_data, model, dataset)
    assert api.client.chat.completions.create.call_count == 2


@pytest.mark.parametrize("method,bad", [
    ("plan", {}), ("plan", {"bridges": []}),
    ("decide", {}), ("decide", {"install": "vector:sounds"}),
])
def test_invalid_memo_object_recovers_with_one_fresh_answer(
    api, request_data, model, dataset, cards, tmp_path, method, bad,
):
    good = {"bridges": ["vector"]} if method == "plan" else {"install": "vector:sounds", "reason": "85% vs 20%."}
    args = (request_data, model, dataset) if method == "plan" else (request_data, cards)
    api.client.chat.completions.create.return_value = completion(good)
    expected = getattr(api.module.LLMPlanner(), method)(*args)
    memo = next((tmp_path / "plans").glob("*.json"))
    memo.write_text(json.dumps(bad))
    assert getattr(api.module.LLMPlanner(), method)(*args) == expected
    assert api.client.chat.completions.create.call_count == 2
    assert json.loads(memo.read_text()) == good
    assert getattr(api.module.LLMPlanner(), method)(*args) == expected
    assert api.client.chat.completions.create.call_count == 2
    getattr(api.rules, method).assert_not_called()


@pytest.mark.parametrize("method", ["plan", "decide"])
def test_invalid_live_answer_after_invalid_memo_is_not_retried_or_stored(
    api, request_data, model, dataset, cards, tmp_path, method,
):
    good = {"bridges": ["vector"]} if method == "plan" else {"install": "vector:sounds", "reason": "85% vs 20%."}
    args = (request_data, model, dataset) if method == "plan" else (request_data, cards)
    api.client.chat.completions.create.return_value = completion(good)
    getattr(api.module.LLMPlanner(), method)(*args)
    memo = next((tmp_path / "plans").glob("*.json"))
    memo.write_text("{}")
    api.client.chat.completions.create.return_value = completion({"invalid": True})
    result = getattr(api.module.LLMPlanner(), method)(*args)
    assert result == getattr(api.rules, method).return_value
    assert api.client.chat.completions.create.call_count == 2
    assert json.loads(memo.read_text()) == {}
    getattr(api.rules, method).assert_called_once()



def test_decision_cache_survives_fresh_loop_verifications(api, monkeypatch, tmp_path):
    from datetime import datetime, timezone
    from agent import loop
    from sense import datasets, models, bridges, verify
    request = FitRequest("fake", "gtzan")
    # Real loop with fake base model/data and controlled verification cards/API.
    monkeypatch.setenv("SENSE_FAKE", "1")
    monkeypatch.setattr(config, "fake", lambda: True)
    model = models.load_model("fake")
    dataset = datasets.load_labelled_set("gtzan")
    monkeypatch.setattr(config, "fake", lambda: False)
    monkeypatch.setattr(models, "load_model", lambda _: model)
    monkeypatch.setattr(datasets, "load_labelled_set", lambda _: dataset)
    monkeypatch.setattr(bridges, "build", lambda kind, *a: SimpleNamespace(name=f"{kind}:gtzan", kind=kind))
    seen = []
    def measure(model, bridge, dataset, request, **kwargs):
        card = ReportCard("fake", "audio", "gtzan", "closed", bridge.name, bridge.kind, 100, .2, .85,
                          {"swapped": .2, "lookup": .84, "specialist_alone": .9})
        seen.append(card)
        return card
    monkeypatch.setattr(verify, "verify", measure)
    dates = iter([datetime(2026, 1, 1, tzinfo=timezone.utc), datetime(2026, 1, 1, tzinfo=timezone.utc),
                  datetime(2026, 1, 2, tzinfo=timezone.utc), datetime(2026, 1, 2, tzinfo=timezone.utc)])
    monkeypatch.setattr(loop, "datetime", SimpleNamespace(now=lambda *a: next(dates)))
    monkeypatch.setattr(loop, "_earlier_sense", lambda _: None)
    api.client.chat.completions.create.side_effect = lambda **kw: completion(
        {"bridges": ["vector", "text"]} if 'bridges' in kw['response_format']['json_schema']['schema']['properties']
        else {"install": None, "reason": "85% accuracy versus 20% swapped."})
    loop.run_fit(request, lambda _: None, api.module.LLMPlanner())
    loop.run_fit(request, lambda _: None, api.module.LLMPlanner())
    assert seen[0] is not seen[2] and seen[0].created != seen[2].created
    assert api.client.chat.completions.create.call_count == 2  # one plan and one decision



def test_card_presentation_does_not_change_decision_identity(api, request_data, cards):
    api.client.chat.completions.create.return_value = completion({"install": None, "reason": "85% vs 20%."})
    api.module.LLMPlanner().decide(request_data, cards)
    cards[0].verdict, cards[0].reason, cards[0].created = "installed", "old presentation", "tomorrow"
    api.module.LLMPlanner().decide(request_data, cards)
    assert api.client.chat.completions.create.call_count == 1


@pytest.mark.parametrize("field", ["controls", "interval", "n"])
def test_substantive_card_measurements_still_change_cache_key(api, request_data, cards, field):
    api.client.chat.completions.create.return_value = completion({"install": None, "reason": "85% vs 20%."})
    api.module.LLMPlanner().decide(request_data, cards)
    if field == "controls":
        cards[0].controls["specialist_alone"] = .91
    elif field == "interval":
        cards[0].interval = [.8, .9]
    else:
        cards[0].n = 101
    api.module.LLMPlanner().decide(request_data, cards)
    assert api.client.chat.completions.create.call_count == 2
