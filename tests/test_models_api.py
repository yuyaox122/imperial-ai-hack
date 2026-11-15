"""Offline API adapter tests; no real client or API key is used."""

import hashlib
import importlib
import json
import sys
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from sense import config
import sense.models
from sense.contracts import BaseModel, ChoiceItem, OpenModel, SenseInput, ToolHandle


def completion(content="A", *, finish="stop", refusal=None, usage=None):
    response = SimpleNamespace(choices=[SimpleNamespace(
        finish_reason=finish, message=SimpleNamespace(content=content, refusal=refusal),
    )])
    if usage is not None:
        response.usage = SimpleNamespace(prompt_tokens=usage[0], completion_tokens=usage[1])
    return response


@pytest.fixture
def api(monkeypatch, tmp_path):
    registration = Mock()
    monkeypatch.setattr(sense.models, "register_model", registration)
    monkeypatch.setattr(config, "openai_key", lambda: "test-key")
    monkeypatch.delenv("SENSE_API_MODEL", raising=False)
    monkeypatch.delenv("SENSE_API_CACHE", raising=False)
    monkeypatch.setenv("SENSE_HOME", str(tmp_path / "home"))  # the answer cache lives under SENSE_HOME
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


@pytest.mark.parametrize("answer", ["", "Z", "a", "1", "AB", "A because a dog barks", "A dog", "I can't hear it"])
def test_anything_but_an_option_letter_is_an_abstention_counted_wrong(api, answer):
    api.client.chat.completions.create.return_value = completion(answer)
    model = api.module.OpenAIChatModel()
    assert model.choose([ChoiceItem("Question", ["cat", "dog"])]) == [api.module.ABSTAIN] == [-1]
    assert model.abstained == 1


@pytest.mark.parametrize("answer,index", [("A.", 0), ("B) dog", 1), ("B\nbecause it barks", 1), (" A ", 0)])
def test_a_letter_followed_by_punctuation_counts_as_that_letter(api, answer, index):
    api.client.chat.completions.create.return_value = completion(answer)
    model = api.module.OpenAIChatModel()
    assert model.choose([ChoiceItem("Question", ["cat", "dog"])]) == [index] and model.abstained == 0


@pytest.mark.parametrize("response", [
    completion(None), completion("A", finish="length"), completion("A", refusal="No"),
])
def test_missing_truncated_or_refused_answers_are_abstentions(api, response):
    api.client.chat.completions.create.return_value = response
    model = api.module.OpenAIChatModel()
    assert model.choose([ChoiceItem("Question", ["cat", "dog"])]) == [-1] and model.abstained == 1


def test_api_failure_propagates_instead_of_inventing_score(api):
    api.client.chat.completions.create.side_effect = RuntimeError("API unavailable")
    with pytest.raises(RuntimeError, match="API unavailable"):
        api.module.OpenAIChatModel().choose([ChoiceItem("Question", ["cat", "dog"])])


# ----------------------------------------------------------------------------- answer cache and usage
def cache_files(tmp_path):
    root = tmp_path / "home" / "api_cache"
    return sorted(p for p in root.rglob("*") if p.is_file()) if root.exists() else []


def test_repeat_choose_is_answered_from_the_disk_cache(api):
    def answer(**kwargs):
        return completion("B" if "dog barks" in kwargs["messages"][1]["content"] else "A")

    api.client.chat.completions.create.side_effect = answer
    items = [
        ChoiceItem("Which sound?", ["cat", "dog"], [SenseInput("text", "audio", text="a dog barks")]),
        ChoiceItem("Which sound?", ["cat", "dog"], [SenseInput("text", "audio", text="a cat meows")]),
    ]
    first = api.module.OpenAIChatModel().choose(items)
    assert api.client.chat.completions.create.call_count == 2
    second = api.module.OpenAIChatModel().choose(items)  # a new instance: the answers come from disk
    assert second == first == [1, 0]
    assert api.client.chat.completions.create.call_count == 2


def test_cache_file_is_keyed_on_the_request_as_sent(api, tmp_path):
    api.client.chat.completions.create.return_value = completion("B")
    api.module.OpenAIChatModel().choose([ChoiceItem("Which sound?", ["cat", "dog"])])
    messages = api.client.chat.completions.create.call_args.kwargs["messages"]
    request = {"model": "gpt-6.1-terra", "instruction": messages[0]["content"], "content": messages[1]["content"],
               "limit": 16}
    key = hashlib.sha256(json.dumps(request, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    path = tmp_path / "home" / "api_cache" / "gpt-6.1-terra" / key[:2] / f"{key}.json"
    assert cache_files(tmp_path) == [path]  # no temporary file is left behind
    record = json.loads(path.read_text())
    assert record["answer"] == "B" and record["model"] == "gpt-6.1-terra" and record["created"]


def test_a_different_prompt_input_image_or_model_misses_the_cache(api, monkeypatch, tmp_path):
    create = api.client.chat.completions.create
    create.return_value = completion("A")
    item = ChoiceItem("Which sound?", ["cat", "dog"])
    variants = [
        item,
        ChoiceItem("Which animal?", ["cat", "dog"]),
        ChoiceItem("Which sound?", ["cat", "dog", "bird"]),
        ChoiceItem("Which sound?", ["cat", "dog"], [SenseInput("text", "audio", text="a dog barks")]),
        ChoiceItem("Which sound?", ["cat", "dog"], [SenseInput("image", "image", image=b"\x89PNG\r\n\x1a\n one")]),
        ChoiceItem("Which sound?", ["cat", "dog"], [SenseInput("image", "image", image=b"\x89PNG\r\n\x1a\n two")]),
    ]
    for k, variant in enumerate(variants, start=1):
        api.module.OpenAIChatModel().choose([variant])
        assert create.call_count == k
    api.module.OpenAIChatModel().choose(variants)  # every variant is now cached
    assert create.call_count == len(variants)

    monkeypatch.setenv("SENSE_API_MODEL", "other/model")
    api.module.OpenAIChatModel().choose([item])
    assert create.call_count == len(variants) + 1
    api.module.OpenAIChatModel().choose([item])
    assert create.call_count == len(variants) + 1
    assert (tmp_path / "home" / "api_cache" / "other_model").is_dir()


@pytest.mark.parametrize("off", ["0", "false", "no", "off", "OFF"])
def test_turning_the_cache_off_sends_every_request(api, monkeypatch, tmp_path, off):
    monkeypatch.setenv("SENSE_API_CACHE", off)
    assert not config.api_cache_enabled()
    model = api.module.OpenAIChatModel()
    item = ChoiceItem("Which sound?", ["cat", "dog"])
    assert model.choose([item]) == model.choose([item]) == [0]
    assert api.client.chat.completions.create.call_count == 2
    assert model.usage()["cache_hits"] == 0
    assert not (tmp_path / "home" / "api_cache").exists()


def test_turning_the_cache_off_ignores_answers_already_on_disk(api, monkeypatch, tmp_path):
    create = api.client.chat.completions.create
    item = ChoiceItem("Which sound?", ["cat", "dog"])
    create.return_value = completion("B")
    assert api.module.OpenAIChatModel().choose([item]) == [1]
    [path] = cache_files(tmp_path)

    monkeypatch.setenv("SENSE_API_CACHE", "0")
    create.return_value = completion("A")  # a fresh answer that differs from the cached one
    model = api.module.OpenAIChatModel()
    assert model.choose([item]) == [0]
    assert create.call_count == 2
    assert model.usage()["cache_hits"] == 0
    assert json.loads(path.read_text())["answer"] == "B"  # the cache is neither read nor written while off


@pytest.mark.parametrize("value", [None, "", "1", "yes", "on", "true"])
def test_cache_is_on_by_default(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("SENSE_API_CACHE", raising=False)
    else:
        monkeypatch.setenv("SENSE_API_CACHE", value)
    assert config.api_cache_enabled()


@pytest.mark.parametrize("response", [completion("Z"), completion("A because"), completion("A", refusal="No")])
def test_an_invalid_answer_is_not_cached(api, tmp_path, response):
    item = ChoiceItem("Which sound?", ["cat", "dog"])
    api.client.chat.completions.create.return_value = response
    assert api.module.OpenAIChatModel().choose([item]) == [-1]
    assert cache_files(tmp_path) == []
    api.client.chat.completions.create.return_value = completion("B")
    assert api.module.OpenAIChatModel().choose([item]) == [1]
    assert api.client.chat.completions.create.call_count == 2


def test_an_unreadable_cache_file_is_asked_again_and_replaced(api, tmp_path):
    item = ChoiceItem("Which sound?", ["cat", "dog"])
    api.client.chat.completions.create.return_value = completion("B")
    api.module.OpenAIChatModel().choose([item])
    [path] = cache_files(tmp_path)
    path.write_text("{not json")
    assert api.module.OpenAIChatModel().choose([item]) == [1]
    assert api.client.chat.completions.create.call_count == 2
    assert json.loads(path.read_text())["answer"] == "B"


@pytest.mark.parametrize("stored", ["Z", "C", "b", "", None, 1, ["B"]])
def test_a_cached_answer_that_is_not_an_option_is_asked_again_and_replaced(api, tmp_path, stored):
    item = ChoiceItem("Which sound?", ["cat", "dog"])  # only A and B are options
    api.client.chat.completions.create.return_value = completion("B")
    api.module.OpenAIChatModel().choose([item])
    [path] = cache_files(tmp_path)
    path.write_text(json.dumps({"answer": stored, "model": "gpt-6.1-terra", "created": "2026-01-01T00:00:00+00:00"}))
    model = api.module.OpenAIChatModel()
    assert model.choose([item]) == [1]
    assert api.client.chat.completions.create.call_count == 2
    assert model.usage()["cache_hits"] == 0
    assert json.loads(path.read_text())["answer"] == "B"


def test_a_cache_that_cannot_be_written_only_warns(api, monkeypatch, tmp_path, caplog):
    blocked = tmp_path / "blocked"
    blocked.write_text("a file where the cache directory should be")
    monkeypatch.setenv("SENSE_HOME", str(blocked))
    model = api.module.OpenAIChatModel()
    items = [ChoiceItem(f"item {i}", ["cat", "dog"]) for i in range(3)]
    with caplog.at_level("WARNING", logger="sense.models_api"):
        assert model.choose(items) == [0, 0, 0]
    warnings = [r for r in caplog.records if "SENSE_API_CACHE=0" in r.getMessage()]
    assert len(warnings) == 1  # once per model, not once per answer


def test_usage_counts_requests_cache_hits_and_tokens(api):
    create = api.client.chat.completions.create
    create.return_value = completion("A", usage=(30, 1))
    model = api.module.OpenAIChatModel()
    assert model.usage() == {"calls": 0, "cache_hits": 0, "prompt_tokens": 0, "completion_tokens": 0}
    items = [ChoiceItem(f"question {i}", ["cat", "dog"]) for i in range(3)]
    model.choose(items)
    assert model.usage() == {"calls": 3, "cache_hits": 0, "prompt_tokens": 90, "completion_tokens": 3}
    model.choose(items)
    assert model.usage() == {"calls": 3, "cache_hits": 3, "prompt_tokens": 90, "completion_tokens": 3}
    create.return_value = completion("A dog.", usage=(12, 5))
    model.chat("What is it?", [])
    model.chat("What is it?", [])  # chat is never cached
    assert model.usage() == {"calls": 5, "cache_hits": 3, "prompt_tokens": 114, "completion_tokens": 13}
    create.return_value = completion("A dog.")  # no usage reported
    model.chat("What is it?", [])
    snapshot = model.usage()
    assert snapshot == {"calls": 6, "cache_hits": 3, "prompt_tokens": 114, "completion_tokens": 13}
    snapshot["calls"] = 0
    assert model.usage()["calls"] == 6  # usage() returns a copy
    create.side_effect = RuntimeError("API unavailable")
    with pytest.raises(RuntimeError):
        model.chat("What is it?", [])
    assert model.usage()["calls"] == 7  # a failed request was still sent


def test_concurrent_choose_writes_valid_cache_files(api, tmp_path):
    barrier = threading.Barrier(4)

    def answer(**kwargs):
        barrier.wait(timeout=5)  # four requests in flight at once
        number = int(kwargs["messages"][1]["content"].splitlines()[0].split()[-1])
        return completion("AB"[number % 2], usage=(10, 1))

    api.client.chat.completions.create.side_effect = answer
    items = [ChoiceItem(f"item {i}", ["cat", "dog"]) for i in range(8)]
    expected = [i % 2 for i in range(8)]
    model = api.module.OpenAIChatModel()
    assert model.choose(items) == expected
    assert model.usage()["calls"] == 8 and model.usage()["prompt_tokens"] == 80
    files = cache_files(tmp_path)
    assert len(files) == 8 and all(p.suffix == ".json" and not p.name.startswith(".") for p in files)
    assert sorted(json.loads(p.read_text())["answer"] for p in files) == ["A"] * 4 + ["B"] * 4

    api.client.chat.completions.create.side_effect = AssertionError("the cache should have answered")
    again = api.module.OpenAIChatModel()
    assert again.choose(items) == expected
    assert again.usage()["cache_hits"] == 8 and again.usage()["calls"] == 0


def test_concurrent_writers_of_one_answer_leave_one_valid_file(api, tmp_path):
    barrier = threading.Barrier(4)

    def answer(**kwargs):
        barrier.wait(timeout=5)
        return completion("B")

    api.client.chat.completions.create.side_effect = answer
    model = api.module.OpenAIChatModel()
    assert model.choose([ChoiceItem("Which sound?", ["cat", "dog"]) for _ in range(8)]) == [1] * 8
    assert model.usage()["calls"] == 4 and model.usage()["cache_hits"] == 4  # the first four raced, the rest hit
    [path] = cache_files(tmp_path)
    assert json.loads(path.read_text())["answer"] == "B"


# ----------------------------------------------------------------------------- tool calls
ANSWER = "Similarity of the clip to each candidate (CLAP): dog 0.40, cat 0.10. Best match: dog."


def tool_input(key="clip-key-1", answer=ANSWER, source="dog.00012.wav", name="listen"):
    """A tool input whose handle records its calls and always gives `answer`."""
    def call(arguments):
        handle.calls.append({"arguments": arguments, "result": answer})
        return answer

    handle = ToolHandle(
        name=name,
        description="Listen to an audio clip with CLAP.",
        parameters={"type": "object", "properties": {"candidates": {"type": "array", "items": {"type": "string"}}}},
        call=call,
        key=key,
    )
    return SenseInput("tool", "audio", source=source, detail="the model can call the listen tool", tool=handle)


def tool_call(arguments, call_id="call_1", name="listen"):
    text = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return SimpleNamespace(id=call_id, type="function", function=SimpleNamespace(name=name, arguments=text))


def calling(*calls, usage=None):
    """A response in which the model calls tools instead of answering."""
    response = SimpleNamespace(choices=[SimpleNamespace(
        finish_reason="tool_calls", message=SimpleNamespace(content=None, refusal=None, tool_calls=list(calls)),
    )])
    if usage is not None:
        response.usage = SimpleNamespace(prompt_tokens=usage[0], completion_tokens=usage[1])
    return response


def test_model_info_says_it_can_call_tools(api):
    info, _ = api.registration.call_args.args
    assert info.tools is True and info.to_dict()["tools"] is True


def test_a_tool_call_is_run_and_its_result_sent_back_before_the_answer(api):
    create = api.client.chat.completions.create
    create.side_effect = [calling(tool_call({"clip": 1, "candidates": ["cat", "dog"]})), completion("B")]
    inp = tool_input()
    model = api.module.OpenAIChatModel()
    assert model.choose([ChoiceItem("Which sound?", ["cat", "dog"], [inp])]) == [1]
    assert inp.tool.calls == [{"arguments": {"candidates": ["cat", "dog"]}, "result": ANSWER}]

    first, second = (c.kwargs for c in create.call_args_list)
    system, user = first["messages"]
    assert "listen tool" in system["content"] and "exactly one uppercase" in system["content"]
    assert user["content"] == (
        "Which sound?\nA. cat\nB. dog\n\n"
        "Audio clip 1: you cannot hear it directly; call the listen tool with clip=1 to hear it."
    )
    assert "dog.00012" not in json.dumps(first)  # the file name, which can give the answer away, is never sent
    [tool] = first["tools"]
    assert tool["type"] == "function" and tool["function"]["name"] == "listen"
    parameters = tool["function"]["parameters"]
    assert parameters["required"] == ["clip"]
    assert parameters["properties"]["clip"]["type"] == "integer" and parameters["properties"]["clip"]["maximum"] == 1
    assert parameters["properties"]["candidates"]["items"] == {"type": "string"}
    assert first["tool_choice"] == "auto"
    assert first["max_completion_tokens"] == api.module.TOOL_CALL_TOKENS  # room for the call, not just a letter

    assert second["messages"][:2] == first["messages"]
    assistant, result = second["messages"][2:]
    assert assistant == {"role": "assistant", "content": None, "tool_calls": [{
        "id": "call_1", "type": "function",
        "function": {"name": "listen", "arguments": json.dumps({"clip": 1, "candidates": ["cat", "dog"]})},
    }]}
    assert result == {"role": "tool", "tool_call_id": "call_1", "content": ANSWER}


def test_chat_with_a_tool_input(api):
    create = api.client.chat.completions.create
    create.side_effect = [calling(tool_call({"clip": 1})), completion("It sounds like a dog.", finish="length")]
    inp = tool_input()
    assert api.module.OpenAIChatModel().chat("What is it?", [inp], max_new_tokens=40) == "It sounds like a dog."
    assert inp.tool.calls == [{"arguments": {}, "result": ANSWER}]
    first, second = (c.kwargs for c in create.call_args_list)
    assert "listen tool" in first["messages"][0]["content"]
    assert first["max_completion_tokens"] == api.module.TOOL_CALL_TOKENS
    assert second["messages"][-1]["content"] == ANSWER


def budgets(create):
    return [(c.kwargs["tool_choice"], c.kwargs["max_completion_tokens"]) for c in create.call_args_list]


def test_a_reply_after_a_tool_call_keeps_to_the_callers_limit(api):
    """The answer round allows TOOL_CALL_TOKENS in case the model calls again; a longer reply is asked for again
    within max_new_tokens, so free_text compares the bridge and no-input replies at the same budget."""
    create = api.client.chat.completions.create
    create.side_effect = [
        calling(tool_call({"clip": 1}), usage=(30, 12)),
        completion("It is a dog barking twice in a quiet room, then a car.", usage=(50, 14)),
        completion("A dog barking", finish="length", usage=(50, 8)),
    ]
    model = api.module.OpenAIChatModel()
    assert model.chat("What is it?", [tool_input()], max_new_tokens=8) == "A dog barking"
    assert budgets(create) == [("auto", api.module.TOOL_CALL_TOKENS), ("auto", api.module.TOOL_CALL_TOKENS), ("none", 8)]
    second, third = (c.kwargs for c in create.call_args_list[1:])
    assert third["messages"] == second["messages"]  # the same question, asked again within the limit
    assert model.usage() == {"calls": 3, "cache_hits": 0, "prompt_tokens": 130, "completion_tokens": 34}


def test_a_reply_within_the_limit_is_kept(api):
    create = api.client.chat.completions.create
    create.side_effect = [calling(tool_call({"clip": 1}), usage=(30, 12)), completion("A dog.", usage=(50, 3))]
    assert api.module.OpenAIChatModel().chat("What is it?", [tool_input()], max_new_tokens=8) == "A dog."
    assert budgets(create) == [("auto", api.module.TOOL_CALL_TOKENS), ("auto", api.module.TOOL_CALL_TOKENS)]


def test_a_long_reply_without_a_tool_call_is_asked_for_again_within_the_limit(api):
    create = api.client.chat.completions.create
    create.side_effect = [completion("x " * 100, usage=(30, 100)), completion("A dog", finish="length", usage=(30, 8))]
    inp = tool_input()
    assert api.module.OpenAIChatModel().chat("What is it?", [inp], max_new_tokens=8) == "A dog"
    assert budgets(create) == [("auto", api.module.TOOL_CALL_TOKENS), ("none", 8)]
    assert inp.tool.calls == []


def test_without_tool_inputs_no_tools_are_offered(api):
    api.module.OpenAIChatModel().choose([ChoiceItem("Which sound?", ["cat", "dog"])])
    kwargs = api.client.chat.completions.create.call_args.kwargs
    assert "tools" not in kwargs and "tool_choice" not in kwargs
    assert "listen" not in kwargs["messages"][0]["content"]


def test_the_model_picks_the_clip_by_number(api):
    create = api.client.chat.completions.create
    create.side_effect = [
        calling(tool_call({"clip": 2, "candidates": ["cat"]}, "c2"), tool_call({"clip": 1.0}, "c1")),
        completion("A"),
    ]
    one, two = tool_input("key-1", "first clip"), tool_input("key-2", "second clip")
    api.module.OpenAIChatModel().choose([ChoiceItem("Which sound?", ["cat", "dog"], [one, two])])
    first, second = (c.kwargs for c in create.call_args_list)
    assert first["messages"][1]["content"].endswith(
        "Audio clip 1: you cannot hear it directly; call the listen tool with clip=1 to hear it.\n\n"
        "Audio clip 2: you cannot hear it directly; call the listen tool with clip=2 to hear it."
    )
    assert first["tools"][0]["function"]["parameters"]["properties"]["clip"]["maximum"] == 2
    assert [m["content"] for m in second["messages"][3:]] == ["second clip", "first clip"]
    assert [m["tool_call_id"] for m in second["messages"][3:]] == ["c2", "c1"]
    assert one.tool.calls == [{"arguments": {}, "result": "first clip"}]
    assert two.tool.calls == [{"arguments": {"candidates": ["cat"]}, "result": "second clip"}]


@pytest.mark.parametrize("arguments, says", [
    ({"clip": 3}, "clip must be the number of an audio clip, from 1 to 2"),
    ({"clip": 0}, "clip must be"),
    ({"clip": -1}, "clip must be"),
    ({"clip": "one"}, "clip must be"),
    ({"clip": True}, "clip must be"),
    ({"clip": 1.5}, "clip must be"),
    ({"candidates": ["cat"]}, "clip must be"),  # two clips: the model has to say which
    ("{not json", "must be a JSON object"),
    ("[1]", "must be a JSON object"),
])
def test_a_bad_tool_call_gets_an_error_text_and_no_clip_is_heard(api, arguments, says):
    create = api.client.chat.completions.create
    create.side_effect = [calling(tool_call(arguments)), completion("A")]
    one, two = tool_input("key-1"), tool_input("key-2")
    assert api.module.OpenAIChatModel().choose([ChoiceItem("Which sound?", ["cat", "dog"], [one, two])]) == [0]
    error = create.call_args_list[1].kwargs["messages"][-1]
    assert error["role"] == "tool" and error["content"].startswith("listen error: ") and says in error["content"]
    assert one.tool.calls == [] and two.tool.calls == []


def test_an_unknown_function_gets_an_error_text(api):
    create = api.client.chat.completions.create
    create.side_effect = [calling(tool_call({"clip": 1}, name="look")), completion("A")]
    inp = tool_input()
    api.module.OpenAIChatModel().choose([ChoiceItem("Which sound?", ["cat", "dog"], [inp])])
    assert create.call_args_list[1].kwargs["messages"][-1]["content"] == (
        "Error: there is no tool called 'look'; the only tool is listen."
    )
    assert inp.tool.calls == []


def test_one_clip_needs_no_clip_number(api):
    api.client.chat.completions.create.side_effect = [calling(tool_call({})), completion("B")]
    inp = tool_input()
    assert api.module.OpenAIChatModel().choose([ChoiceItem("Which sound?", ["cat", "dog"], [inp])]) == [1]
    assert inp.tool.calls == [{"arguments": {}, "result": ANSWER}]


def test_after_two_rounds_of_tool_calls_the_model_must_answer(api):
    create = api.client.chat.completions.create

    def answer(**kwargs):
        if kwargs["tool_choice"] == "none":
            return completion("B", usage=(20, 1))
        return calling(tool_call({"clip": 1}, f"call_{len(kwargs['messages'])}"), usage=(10, 8))

    create.side_effect = answer
    inp = tool_input()
    model = api.module.OpenAIChatModel()
    assert model.choose([ChoiceItem("Which sound?", ["cat", "dog"], [inp])]) == [1]
    requests = [c.kwargs for c in create.call_args_list]
    assert [r["tool_choice"] for r in requests] == ["auto", "auto", "none"]
    assert [len(r["messages"]) for r in requests] == [2, 4, 6]
    assert requests[-1]["max_completion_tokens"] == 16  # the forced answer is a letter
    assert all(r["tools"] == requests[0]["tools"] for r in requests)
    assert len(inp.tool.calls) == 2
    assert model.usage() == {"calls": 3, "cache_hits": 0, "prompt_tokens": 40, "completion_tokens": 17}


def test_a_model_that_still_calls_tools_when_it_must_answer_abstains(api):
    api.client.chat.completions.create.side_effect = lambda **kwargs: calling(tool_call({"clip": 1}))
    model = api.module.OpenAIChatModel()
    assert model.choose([ChoiceItem("Which sound?", ["cat", "dog"], [tool_input()])]) == [-1]
    assert model.abstained == 1
    assert api.client.chat.completions.create.call_count == 3


def test_usage_counts_every_request_of_a_tool_call(api):
    api.client.chat.completions.create.side_effect = [
        calling(tool_call({"clip": 1}), usage=(30, 12)), completion("A", usage=(45, 1)),
    ]
    model = api.module.OpenAIChatModel()
    model.choose([ChoiceItem("Which sound?", ["cat", "dog"], [tool_input()])])
    assert model.usage() == {"calls": 2, "cache_hits": 0, "prompt_tokens": 75, "completion_tokens": 13}


def test_the_cache_key_includes_each_tool_key(api, tmp_path):
    """Two clips with the same question have the same prompt; only the tool key tells them apart."""
    create = api.client.chat.completions.create
    create.side_effect = lambda **kwargs: (
        completion("B") if kwargs["messages"][-1]["role"] == "tool" else calling(tool_call({"clip": 1}))
    )

    def item(key):
        return ChoiceItem("Which sound?", ["cat", "dog"], [tool_input(key)])

    first = api.module.OpenAIChatModel()
    assert first.choose([item("clip-a"), item("clip-b")]) == [1, 1]
    assert create.call_count == 4 and len(cache_files(tmp_path)) == 2

    messages = create.call_args_list[0].kwargs["messages"]
    request = {"model": "gpt-6.1-terra", "instruction": messages[0]["content"], "content": messages[1]["content"],
               "limit": 16, "tools": ["clip-a"]}
    key = hashlib.sha256(json.dumps(request, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    assert (tmp_path / "home" / "api_cache" / "gpt-6.1-terra" / key[:2] / f"{key}.json") in cache_files(tmp_path)

    again = item("clip-a")
    second = api.module.OpenAIChatModel()
    assert second.choose([again]) == [1]  # answered from the cache: no request, and the tool is not called
    assert create.call_count == 4 and again.inputs[0].tool.calls == []
    assert second.usage() == {"calls": 0, "cache_hits": 1, "prompt_tokens": 0, "completion_tokens": 0}
    second.choose([item("clip-c")])  # a new clip asks again
    assert create.call_count == 6


def test_a_tool_input_with_another_tool_is_refused_before_any_request(api):
    with pytest.raises(ValueError, match="wires only the listen tool"):
        api.module.OpenAIChatModel().choose([ChoiceItem("Which?", ["cat", "dog"], [tool_input(name="look")])])
    with pytest.raises(ValueError, match="wires only the listen tool"):
        api.module.OpenAIChatModel().chat("Which?", [tool_input(name="look")])
    api.client.chat.completions.create.assert_not_called()


# ----------------------------------------------------------------------------- history and streaming
HISTORY = [{"role": "user", "content": "What is this?"}, {"role": "assistant", "content": "A dog barking."}]


class FakeStream:
    """What the client returns for stream=True: an iterable of chunks with close()."""

    def __init__(self, chunks):
        self.chunks = list(chunks)
        self.closed = False
        self.read = 0

    def __iter__(self):
        for chunk in self.chunks:
            self.read += 1
            yield chunk

    def close(self):
        self.closed = True


def delta(content=None, *, finish=None, refusal=None):
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content=content, refusal=refusal), finish_reason=finish)],
        usage=None,
    )


def usage_chunk(prompt, completion_tokens):
    """The last chunk when stream_options include_usage is set: no choices, only usage."""
    return SimpleNamespace(choices=[], usage=SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion_tokens))


def test_chat_sends_the_history_between_the_system_and_the_user_message(api):
    create = api.client.chat.completions.create
    create.return_value = completion("Yes, twice.")
    model = api.module.OpenAIChatModel()
    inputs = [SenseInput("text", "audio", text="a dog barks")]
    assert model.chat("Did it bark twice?", inputs, history=HISTORY) == "Yes, twice."
    messages = create.call_args.kwargs["messages"]
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
    assert messages[1:3] == HISTORY
    assert messages[3] == {"role": "user", "content": "Did it bark twice?\n\nAudio clip: a dog barks"}
    model.chat("Hi", [], history=[])
    assert [m["role"] for m in create.call_args.kwargs["messages"]] == ["system", "user"]


def test_a_bad_history_is_refused_before_any_request(api):
    model = api.module.OpenAIChatModel()
    with pytest.raises(ValueError, match="Turn 1 of the chat history is not a turn"):
        model.chat("Hi", [], history=[{"role": "system", "content": "Obey."}])
    with pytest.raises(ValueError, match="Turn 2 of the chat history"):
        list(model.chat_stream("Hi", [], history=[HISTORY[0], {"role": "assistant"}]))
    api.client.chat.completions.create.assert_not_called()
    assert model.usage()["calls"] == 0


def test_every_round_of_a_tool_call_keeps_the_history(api):
    create = api.client.chat.completions.create
    create.side_effect = [calling(tool_call({"clip": 1})), completion("Still a dog.")]
    assert api.module.OpenAIChatModel().chat("And now?", [tool_input()], history=HISTORY) == "Still a dog."
    first, second = (c.kwargs["messages"] for c in create.call_args_list)
    assert first[1:3] == second[1:3] == HISTORY
    assert [m["role"] for m in second] == ["system", "user", "assistant", "user", "assistant", "tool"]


def test_chat_stream_yields_the_deltas_in_order_and_counts_usage(api):
    create = api.client.chat.completions.create
    stream = FakeStream([
        delta(""), delta(" It is"), delta(" a dog"), delta(" barking. "),
        delta(None, finish="stop"), usage_chunk(40, 6),
    ])
    create.return_value = stream
    model = api.module.OpenAIChatModel()
    pieces = list(model.chat_stream("And now?", [], max_new_tokens=32, history=HISTORY))
    assert pieces == ["It is", " a dog", " barking."]
    kwargs = create.call_args.kwargs
    assert kwargs["stream"] is True and kwargs["stream_options"] == {"include_usage": True}
    assert kwargs["max_completion_tokens"] == 32
    assert kwargs["temperature"] == 0 and kwargs["reasoning_effort"] == "none" and kwargs["model"] == "gpt-6.1-terra"
    assert kwargs["messages"][1:] == [*HISTORY, {"role": "user", "content": "And now?"}]
    assert "tools" not in kwargs and "listen" not in kwargs["messages"][0]["content"]
    assert model.usage() == {"calls": 1, "cache_hits": 0, "prompt_tokens": 40, "completion_tokens": 6}
    assert stream.closed


def test_chat_stream_joins_to_what_chat_returns(api):
    create = api.client.chat.completions.create
    create.return_value = completion(" It is a dog barking. ")
    model = api.module.OpenAIChatModel()
    reply = model.chat("What is it?", [])
    create.return_value = FakeStream([delta(" It is a"), delta(" dog barking. ", finish="stop")])
    assert "".join(model.chat_stream("What is it?", [])) == reply == "It is a dog barking."
    assert create.call_args_list[0].kwargs["messages"] == create.call_args_list[1].kwargs["messages"]
    assert model.usage() == {"calls": 2, "cache_hits": 0, "prompt_tokens": 0, "completion_tokens": 0}  # none reported


def test_a_stream_cut_off_at_the_token_limit_is_kept(api):
    stream = FakeStream([delta("A dog is"), delta(" barking", finish="length")])
    api.client.chat.completions.create.return_value = stream
    assert "".join(api.module.OpenAIChatModel().chat_stream("What sound?", [], max_new_tokens=4)) == "A dog is barking"


@pytest.mark.parametrize("chunks", [
    [delta(None, refusal="I can't help with that."), delta(None, finish="stop")],
    [delta(None, finish="stop")],
    [delta("   "), delta(None, finish="stop")],
    [delta("Half an ans")],  # the stream ended without a finish reason
    [delta("Some text"), delta(None, finish="content_filter")],
])
def test_a_refused_empty_or_unfinished_stream_fails_after_its_pieces(api, chunks):
    stream = FakeStream(chunks)
    api.client.chat.completions.create.return_value = stream
    pieces = []
    with pytest.raises(ValueError, match="no complete answer"):
        for piece in api.module.OpenAIChatModel().chat_stream("What is it?", []):
            pieces.append(piece)
    assert "".join(pieces) == "".join(c.choices[0].delta.content or "" for c in chunks).strip()
    assert stream.closed


def test_a_failed_stream_request_is_still_counted(api):
    api.client.chat.completions.create.side_effect = RuntimeError("API unavailable")
    model = api.module.OpenAIChatModel()
    with pytest.raises(RuntimeError, match="API unavailable"):
        list(model.chat_stream("What is it?", []))
    assert model.usage()["calls"] == 1


def test_closing_a_stream_early_closes_the_response(api):
    stream = FakeStream([delta(f" word{k}") for k in range(50)] + [delta(None, finish="stop")])
    api.client.chat.completions.create.return_value = stream
    pieces = api.module.OpenAIChatModel().chat_stream("Talk", [])
    assert next(pieces) == "word0"
    pieces.close()
    assert stream.closed and stream.read < 10


def test_chat_stream_with_a_tool_input_runs_chat_and_yields_its_reply_once(api):
    create = api.client.chat.completions.create
    create.side_effect = [
        calling(tool_call({"clip": 1}), usage=(30, 12)), completion(" It sounds like a dog. ", usage=(50, 6)),
    ]
    inp = tool_input()
    model = api.module.OpenAIChatModel()
    pieces = list(model.chat_stream("What is it?", [inp], max_new_tokens=40, history=HISTORY))
    assert pieces == ["It sounds like a dog."]
    requests = [c.kwargs for c in create.call_args_list]
    assert all("stream" not in r for r in requests) and requests[0]["tool_choice"] == "auto"
    assert requests[0]["messages"][1:3] == HISTORY and "listen tool" in requests[0]["messages"][0]["content"]
    assert inp.tool.calls == [{"arguments": {}, "result": ANSWER}]
    assert model.usage() == {"calls": 2, "cache_hits": 0, "prompt_tokens": 80, "completion_tokens": 18}
