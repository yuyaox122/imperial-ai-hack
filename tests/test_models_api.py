"""Offline API adapter tests; no real client or API key is used."""

import importlib
import sys
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from sense import config
import sense.models
from sense.contracts import BaseModel, ChoiceItem, OpenModel, SenseInput


def completion(content="A", *, finish="stop", refusal=None):
    return SimpleNamespace(choices=[SimpleNamespace(
        finish_reason=finish, message=SimpleNamespace(content=content, refusal=refusal),
    )])


@pytest.fixture
def api(monkeypatch):
    registration = Mock()
    monkeypatch.setattr(sense.models, "register_model", registration)
    monkeypatch.setattr(config, "openai_key", lambda: "test-key")
    monkeypatch.delenv("SENSE_API_MODEL", raising=False)
    previous = sys.modules.pop("sense.models_api", None)
    previous_attribute = getattr(sense, "models_api", None)
    module = importlib.import_module("sense.models_api")
    client = Mock()
    client.chat.completions.create.return_value = completion("A")
    factory = Mock(return_value=client)
    monkeypatch.setattr(module, "OpenAI", factory)
    yield SimpleNamespace(module=module, client=client, factory=factory, registration=registration)
    sys.modules.pop("sense.models_api", None)
    if previous is not None:
        sys.modules["sense.models_api"] = previous
    if previous_attribute is not None:
        sense.models_api = previous_attribute
    else:
        delattr(sense, "models_api")


def test_registers_closed_api_model(api):
    info, loader = api.registration.call_args.args
    assert info.name == "openai-chat"
    assert info.native == ["text", "image"]
    assert info.available and not info.open_weights
    model = loader()
    assert isinstance(model, BaseModel)
    assert not isinstance(model, OpenModel)


def test_no_key_marks_unavailable_without_constructing_client(api, monkeypatch):
    monkeypatch.setattr(config, "openai_key", lambda: None)
    api.registration.reset_mock()
    importlib.reload(api.module)
    info, loader = api.registration.call_args.args
    assert not info.available
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        loader()


def test_chat_includes_text_captions_and_ignores_vectors(api, monkeypatch):
    monkeypatch.setenv("SENSE_API_MODEL", "custom-model")
    model = api.module.OpenAIChatModel()
    api.client.chat.completions.create.return_value = completion(" A dog is barking. ")
    inputs = [
        SenseInput("text", "audio", text="a dog barks"),
        SenseInput("vector", "audio", text="ignore vector text", vector=np.ones(4)),
        SenseInput("text", "audio", text="a bell rings"),
    ]
    assert model.chat("What can you hear?", inputs, max_new_tokens=42) == "A dog is barking."
    kwargs = api.client.chat.completions.create.call_args.kwargs
    assert kwargs["messages"][1]["content"] == "What can you hear?\n\nAudio clip: a dog barks\n\nAudio clip: a bell rings"
    assert kwargs["model"] == "custom-model"
    assert kwargs["max_completion_tokens"] == 42
    assert kwargs["temperature"] == 0
    assert kwargs["reasoning_effort"] == "none"


def test_native_images_are_sent_as_data_urls(api):
    model = api.module.OpenAIChatModel()
    inputs = [SenseInput("image", "image", image=b"\x89PNG\r\n\x1a\n")]
    model.chat("What is this?", inputs)
    content = api.client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
    assert content[0] == {"type": "text", "text": "What is this?"}
    assert content[1]["image_url"]["url"] == "data:image/png;base64,iVBORw0KGgo="


def test_chat_returns_available_text_at_token_limit(api):
    api.client.chat.completions.create.return_value = completion("A dog is barking", finish="length")
    assert api.module.OpenAIChatModel().chat("What sound?", []) == "A dog is barking"


def test_choose_sends_letter_options_and_returns_indices(api):
    model = api.module.OpenAIChatModel()
    item = ChoiceItem("Which sound?", ["cat", "dog"], [SenseInput("text", "audio", text="a dog barks")])
    api.client.chat.completions.create.return_value = completion(" B\n")
    assert model.choose([item]) == [1]
    kwargs = api.client.chat.completions.create.call_args.kwargs
    assert kwargs["messages"][1]["content"] == "Which sound?\nA. cat\nB. dog\n\nAudio clip: a dog barks"
    assert "exactly one uppercase" in kwargs["messages"][0]["content"]
    assert kwargs["temperature"] == 0
    assert kwargs["max_completion_tokens"] == 16


def test_choose_is_parallel_bounded_and_preserves_order(api):
    model = api.module.OpenAIChatModel()
    barrier = threading.Barrier(4)
    lock = threading.Lock()
    active = peak = 0

    def answer(**kwargs):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        barrier.wait(timeout=5)
        question = kwargs["messages"][1]["content"].splitlines()[0]
        if question == "item 0":
            result = "B"
        else:
            result = "A"
        with lock:
            active -= 1
        return completion(result)

    api.client.chat.completions.create.side_effect = answer
    items = [ChoiceItem(f"item {i}", ["cat", "dog"]) for i in range(8)]
    assert model.choose(items) == [1, 0, 0, 0, 0, 0, 0, 0]
    assert peak == 4


def test_empty_choice_batch_makes_no_request(api):
    assert api.module.OpenAIChatModel().choose([]) == []
    api.client.chat.completions.create.assert_not_called()


@pytest.mark.parametrize("options", [[], [str(i) for i in range(27)]])
def test_invalid_options_fail_before_network(api, options):
    with pytest.raises(ValueError, match="between 1 and 26"):
        api.module.OpenAIChatModel().choose([ChoiceItem("Question", options)])
    api.client.chat.completions.create.assert_not_called()


@pytest.mark.parametrize("answer", ["", "Z", "a", "1", "AB", "A.", "A because a dog barks"])
def test_malformed_letter_fails_verification(api, answer):
    api.client.chat.completions.create.return_value = completion(answer)
    with pytest.raises(ValueError):
        api.module.OpenAIChatModel().choose([ChoiceItem("Question", ["cat", "dog"])])


@pytest.mark.parametrize("response", [
    completion(None), completion("A", finish="length"), completion("A", refusal="No"),
])
def test_missing_truncated_or_refused_answers_fail(api, response):
    api.client.chat.completions.create.return_value = response
    with pytest.raises(ValueError):
        api.module.OpenAIChatModel().choose([ChoiceItem("Question", ["cat", "dog"])])


def test_api_failure_propagates_instead_of_inventing_score(api):
    api.client.chat.completions.create.side_effect = RuntimeError("API unavailable")
    with pytest.raises(RuntimeError, match="API unavailable"):
        api.module.OpenAIChatModel().choose([ChoiceItem("Question", ["cat", "dog"])])
