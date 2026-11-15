"""The claude-sonnet base model (sense/models_anthropic.py), offline: the Anthropic client is a mock and no key is real."""

from __future__ import annotations

import base64
import importlib
import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

import sense
import sense.models
from sense import config
from sense.contracts import BaseModel, ChoiceItem, OpenModel, SenseInput, ToolHandle
from sense.models_api import CHAT_INSTRUCTION

FORBIDDEN = {"temperature", "top_p", "top_k"}  # Claude Sonnet 5.5 rejects non-default sampling values


def text(value):
    return SimpleNamespace(type="text", text=value)


def reply(content="A", *, stop="end_turn", usage=(10, 2), model="claude-sonnet-5-5", category=None):
    blocks = [text(content)] if isinstance(content, str) else content
    details = SimpleNamespace(category=category) if stop == "refusal" else None
    return SimpleNamespace(content=blocks, stop_reason=stop, model=model, stop_details=details,
                           usage=SimpleNamespace(input_tokens=usage[0], output_tokens=usage[1]))


def tool_use(arguments, call_id="toolu_1", name="listen"):
    return SimpleNamespace(type="tool_use", id=call_id, name=name, input=arguments)


@pytest.fixture
def claude(monkeypatch, tmp_path):
    registration = Mock()
    monkeypatch.setattr(sense.models, "register_model", registration)
    monkeypatch.setattr(config, "anthropic_key", lambda: "test-key")
    monkeypatch.delenv("SENSE_CLAUDE_MODEL", raising=False)
    monkeypatch.delenv("SENSE_API_CACHE", raising=False)
    monkeypatch.setenv("SENSE_HOME", str(tmp_path / "home"))  # the answer cache lives under SENSE_HOME
    previous = sys.modules.pop("sense.models_anthropic", None)
    previous_attribute = getattr(sense, "models_anthropic", None)
    module = importlib.import_module("sense.models_anthropic")
    client = Mock()
    client.beta.messages.create.return_value = reply("A")
    factory = Mock(return_value=client)
    monkeypatch.setattr(module, "Anthropic", factory)
    yield SimpleNamespace(module=module, client=client, factory=factory, registration=registration,
                          create=client.beta.messages.create)
    sys.modules.pop("sense.models_anthropic", None)
    if previous is not None:
        sys.modules["sense.models_anthropic"] = previous
    if previous_attribute is not None:
        sense.models_anthropic = previous_attribute
    else:
        delattr(sense, "models_anthropic")


def assert_claude_request(kwargs):
    """What every request to Claude Sonnet 5.5 carries: the beta fallback, no sampling, no extended thinking."""
    assert not FORBIDDEN & set(kwargs)
    assert kwargs["thinking"] == {"type": "between_tools"}
    assert kwargs["output_config"] == {"effort": "low"}
    assert kwargs["betas"] == ["server-side-fallback-2026-07-01"] and kwargs["fallbacks"] == "default"


# ----------------------------------------------------------------------------- registration and listing
def test_importing_models_api_first_still_loads_claude_sonnet():
    """sense.models_api imported before sense.models used to leave claude-sonnet out with a circular-import
    warning (seen in a script); a fresh interpreter with warnings as errors must load it either way."""
    import subprocess
    from pathlib import Path

    repo = str(Path(__file__).resolve().parent.parent)
    code = ("import sys; sys.path.insert(0, {repo!r}); import sense.models_api, sense.models; "
            "assert 'sense.models_anthropic' in sys.modules").format(repo=repo)
    done = subprocess.run([sys.executable, "-I", "-W", "error::UserWarning", "-c", code], cwd=repo,
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr[-2000:]


def test_registers_a_closed_model_that_can_call_tools(claude):
    info, loader = claude.registration.call_args.args
    assert (info.name, info.label, info.native) == ("claude-sonnet", "Claude Sonnet 5.5", ["text", "image"])
    assert info.available and not info.open_weights and info.tools
    model = loader()
    assert isinstance(model, BaseModel) and not isinstance(model, OpenModel)
    claude.factory.assert_called_once_with(api_key="test-key", timeout=30.0, max_retries=2)


def test_no_key_marks_it_unavailable_without_constructing_a_client(claude, monkeypatch):
    monkeypatch.setattr(config, "anthropic_key", lambda: None)
    claude.registration.reset_mock()
    importlib.reload(claude.module)
    info, loader = claude.registration.call_args.args
    assert not info.available
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        loader()


@pytest.fixture
def listed(monkeypatch):
    """available_models() with sense.models_anthropic registered afresh into a copy of the registry."""
    monkeypatch.setattr(sense.models, "_registered", dict(sense.models._registered))
    previous = sys.modules.pop("sense.models_anthropic", None)
    previous_attribute = getattr(sense, "models_anthropic", None)

    def available(key):
        monkeypatch.setattr(config, "anthropic_key", lambda: key)
        sys.modules.pop("sense.models_anthropic", None)
        importlib.import_module("sense.models_anthropic")
        return {info.name: info for info in sense.models.available_models()}

    yield available
    sys.modules.pop("sense.models_anthropic", None)
    if previous is not None:
        sys.modules["sense.models_anthropic"] = previous
    if previous_attribute is not None:
        sense.models_anthropic = previous_attribute


def test_sense_models_registers_claude_sonnet_next_to_openai_chat():
    assert {"openai-chat", "claude-sonnet"} <= set(sense.models._registered)
    info, _ = sense.models._registered["claude-sonnet"]
    assert info.label == "Claude Sonnet 5.5" and info.tools and not info.open_weights


def test_claude_sonnet_is_offered_only_with_a_key_and_never_in_fake_mode(listed, monkeypatch):
    monkeypatch.setattr(config, "fake", lambda: False)
    assert listed("test-key")["claude-sonnet"].available
    assert not listed(None)["claude-sonnet"].available  # listed as unavailable, as openai-chat is without its key
    monkeypatch.setattr(config, "fake", lambda: True)
    assert list(listed("test-key")) == ["fake", "fake-api"]


# ----------------------------------------------------------------------------- chat
def test_text_chat(claude, monkeypatch):
    monkeypatch.setenv("SENSE_CLAUDE_MODEL", "claude-custom")
    model = claude.module.AnthropicChatModel()
    claude.create.return_value = reply(" A dog is barking. ", usage=(30, 6))
    inputs = [
        SenseInput("text", "audio", text="a dog barks"),
        SenseInput("vector", "audio", text="ignore vector text", vector=np.ones(4)),
        SenseInput("text", "audio", text="a bell rings"),
    ]
    assert model.chat("What can you hear?", inputs, max_new_tokens=42) == "A dog is barking."
    kwargs = claude.create.call_args.kwargs
    assert kwargs["model"] == "claude-custom" and kwargs["max_tokens"] == 42
    assert kwargs["system"] == CHAT_INSTRUCTION
    assert kwargs["messages"] == [
        {"role": "user", "content": "What can you hear?\n\nAudio clip: a dog barks\n\nAudio clip: a bell rings"},
    ]
    assert "tools" not in kwargs
    assert_claude_request(kwargs)
    claude.client.messages.create.assert_not_called()  # the beta endpoint, which takes the fallback
    assert model.usage() == {"calls": 1, "cache_hits": 0, "prompt_tokens": 30, "completion_tokens": 6}


def test_chat_history_comes_before_the_message(claude):
    history = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello."}]
    claude.module.AnthropicChatModel().chat("And now?", [], history=history)
    assert claude.create.call_args.kwargs["messages"] == [*history, {"role": "user", "content": "And now?"}]


@pytest.mark.parametrize("image,media_type", [(b"\x89PNG\r\n\x1a\n", "image/png"), (b"\xff\xd8\xff\xe0", "image/jpeg")])
def test_an_image_input_becomes_an_image_block(claude, image, media_type):
    claude.module.AnthropicChatModel().chat("What is this?", [SenseInput("image", "image", image=image)])
    [message] = claude.create.call_args.kwargs["messages"]
    assert message["content"] == [
        {"type": "image", "source": {"type": "base64", "media_type": media_type,
                                     "data": base64.b64encode(image).decode("ascii")}},
        {"type": "text", "text": "What is this?"},
    ]


def test_a_large_image_is_scaled_under_the_api_limits(claude):
    """Chat lets 20 MB images through; Claude takes at most 5 MB of base64 per image, so big ones are scaled."""
    from io import BytesIO

    from PIL import Image

    noisy = Image.fromarray(np.random.default_rng(0).integers(0, 256, (2600, 2600, 3), dtype=np.uint8))
    buf = BytesIO()
    noisy.save(buf, format="PNG")
    big = buf.getvalue()
    assert len(big) > 5_000_000
    claude.module.AnthropicChatModel().chat("What is this?", [SenseInput("image", "image", image=big)])
    [message] = claude.create.call_args.kwargs["messages"]
    sent = base64.b64decode(message["content"][0]["source"]["data"])
    assert len(message["content"][0]["source"]["data"]) < 5_000_000
    with Image.open(BytesIO(sent)) as img:
        assert max(img.size) == claude.module.IMAGE_SIDE
    assert message["content"][0]["source"]["media_type"] == ("image/jpeg" if sent[:3] == b"\xff\xd8\xff" else "image/png")


def test_chat_keeps_a_reply_cut_at_the_limit_and_fails_on_a_refusal(claude):
    claude.create.return_value = reply("A dog is", stop="max_tokens")
    assert claude.module.AnthropicChatModel().chat("What sound?", []) == "A dog is"
    claude.create.return_value = reply([], stop="refusal", category="cyber")
    with pytest.raises(ValueError, match="declined to answer.*cyber"):
        claude.module.AnthropicChatModel().chat("What sound?", [])


def test_usage_adds_every_attempt_of_a_fallback(claude):
    response = reply("A dog.")
    response.usage.iterations = [
        SimpleNamespace(type="message", input_tokens=40, output_tokens=0, cache_read_input_tokens=0),
        SimpleNamespace(type="fallback_message", input_tokens=42, output_tokens=7, cache_read_input_tokens=8),
    ]
    claude.create.return_value = response
    model = claude.module.AnthropicChatModel()
    model.chat("What sound?", [])
    assert model.usage() == {"calls": 1, "cache_hits": 0, "prompt_tokens": 90, "completion_tokens": 7}


# ----------------------------------------------------------------------------- choose and the answer cache
def cache_files(tmp_path):
    root = tmp_path / "home" / "api_cache"
    return sorted(p for p in root.rglob("*") if p.is_file()) if root.exists() else []


def test_choose_returns_option_indices_and_repeats_from_the_cache(claude, tmp_path):
    def answer(**kwargs):
        return reply('{"answer": "B"}' if "dog barks" in kwargs["messages"][0]["content"] else '{"answer": "A"}')

    claude.create.side_effect = answer
    items = [
        ChoiceItem("Which sound?", ["cat", "dog"], [SenseInput("text", "audio", text="a dog barks")]),
        ChoiceItem("Which sound?", ["cat", "dog"], [SenseInput("text", "audio", text="a cat meows")]),
    ]
    assert claude.module.AnthropicChatModel().choose(items) == [1, 0]
    assert claude.create.call_count == 2
    kwargs = claude.create.call_args.kwargs
    # A choice lets the model think, then constrains the reply to the option letters (16 tokens with thinking off
    # cut Claude off mid-reasoning on raw files).
    assert "exactly one uppercase" in kwargs["system"] and kwargs["max_tokens"] == claude.module.CHOICE_TOKENS
    assert kwargs["thinking"] == {"type": "adaptive"}
    assert kwargs["output_config"] == {"effort": "low", "format": {"type": "json_schema", "schema": {
        "type": "object", "properties": {"answer": {"type": "string", "enum": ["A", "B"]}},
        "required": ["answer"], "additionalProperties": False}}}
    assert not FORBIDDEN & set(kwargs)
    assert kwargs["betas"] == ["server-side-fallback-2026-07-01"] and kwargs["fallbacks"] == "default"
    again = claude.module.AnthropicChatModel()  # a new instance: the answers come from disk
    assert again.choose(items) == [1, 0]
    assert claude.create.call_count == 2
    assert again.usage()["cache_hits"] == 2 and again.usage()["calls"] == 0
    files = cache_files(tmp_path)
    assert len(files) == 2 and all(path.parts[-3] == "claude-sonnet-5-5" for path in files)
    assert json.loads(files[0].read_text())["model"] == "claude-sonnet-5-5"


@pytest.mark.parametrize("answer", ["", "C", "a", "AB", "A because a dog barks", "I can't see the 3D model"])
def test_anything_but_one_option_letter_is_an_uncached_abstention(claude, answer, tmp_path):
    claude.create.return_value = reply(answer)
    model = claude.module.AnthropicChatModel()
    assert model.choose([ChoiceItem("Question", ["cat", "dog"])]) == [-1] and model.abstained == 1
    assert cache_files(tmp_path) == []


@pytest.mark.parametrize("response", [reply("A", stop="max_tokens"), reply([], stop="refusal"), reply("A", stop="refusal"),
                                      reply("I can't see the 3D model, and no image or audio", stop="max_tokens")])
def test_a_cut_off_or_refused_choice_is_an_uncached_abstention(claude, response, tmp_path):
    claude.create.return_value = response
    model = claude.module.AnthropicChatModel()
    assert model.choose([ChoiceItem("Question", ["cat", "dog"])]) == [-1] and model.abstained == 1
    assert cache_files(tmp_path) == []


# ----------------------------------------------------------------------------- the listen tool
ANSWER = "Similarity of the clip to each candidate (CLAP): dog 0.40, cat 0.10. Best match: dog."


def tool_input(key="clip-key-1"):
    def call(arguments):
        handle.calls.append({"arguments": arguments, "result": ANSWER})
        return ANSWER

    handle = ToolHandle(
        name="listen", description="Listen to an audio clip with CLAP.",
        parameters={"type": "object", "properties": {"candidates": {"type": "array", "items": {"type": "string"}}}},
        call=call, key=key,
    )
    return SenseInput("tool", "audio", source="dog.00012.wav", tool=handle)


def test_a_tool_call_is_run_and_its_result_sent_back_before_the_answer(claude):
    calling = reply([text("Let me listen."), tool_use({"clip": 1, "candidates": ["cat", "dog"]})], stop="tool_use")
    claude.create.side_effect = [calling, reply("B")]
    inp = tool_input()
    assert claude.module.AnthropicChatModel().choose([ChoiceItem("Which sound?", ["cat", "dog"], [inp])]) == [1]
    assert inp.tool.calls == [{"arguments": {"candidates": ["cat", "dog"]}, "result": ANSWER}]
    first, second = (call.kwargs for call in claude.create.call_args_list)
    assert "listen tool" in first["system"]
    assert first["messages"] == [{"role": "user", "content": (
        "Which sound?\nA. cat\nB. dog\n\n"
        "Audio clip 1: you cannot hear it directly; call the listen tool with clip=1 to hear it.")}]
    assert "dog.00012" not in json.dumps(first, default=str)  # the file name can give the answer away
    [tool] = first["tools"]
    assert tool["name"] == "listen" and set(tool) == {"name", "description", "input_schema"}
    assert tool["input_schema"]["required"] == ["clip"] and tool["input_schema"]["properties"]["clip"]["maximum"] == 1
    assert first["tool_choice"] == {"type": "auto"}
    assert first["max_tokens"] == claude.module.TOOL_CALL_TOKENS
    assert_claude_request(first)
    assert second["messages"][0] == first["messages"][0]
    assert second["messages"][1] == {"role": "assistant", "content": calling.content}  # sent back unchanged
    assert second["messages"][2] == {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "toolu_1", "content": ANSWER}]}


def test_a_wrong_tool_call_comes_back_as_an_error(claude):
    claude.create.side_effect = [reply([tool_use({"clip": 7})], stop="tool_use"), reply("A")]
    inp = tool_input()
    claude.module.AnthropicChatModel().choose([ChoiceItem("Which sound?", ["cat", "dog"], [inp])])
    [result] = claude.create.call_args_list[1].kwargs["messages"][2]["content"]
    assert result["is_error"] is True and "clip must be the number" in result["content"]
    assert inp.tool.calls == []


def test_after_two_rounds_of_tool_calls_the_model_must_answer(claude):
    claude.create.side_effect = [
        reply([tool_use({"clip": 1}, "toolu_1")], stop="tool_use"),
        reply([tool_use({"clip": 1}, "toolu_2")], stop="tool_use"),
        reply("B"),
    ]
    inp = tool_input()
    assert claude.module.AnthropicChatModel().choose([ChoiceItem("Which sound?", ["cat", "dog"], [inp])]) == [1]
    budgets = [(c.kwargs["tool_choice"], c.kwargs["max_tokens"]) for c in claude.create.call_args_list]
    tokens = claude.module.TOOL_CALL_TOKENS
    assert budgets == [({"type": "auto"}, tokens), ({"type": "auto"}, tokens), ({"type": "none"}, 16)]
    assert len(inp.tool.calls) == 2


def test_a_long_reply_after_a_tool_call_is_asked_for_again_within_the_limit(claude):
    claude.create.side_effect = [
        reply([tool_use({"clip": 1})], stop="tool_use", usage=(30, 12)),
        reply("It is a dog barking twice in a quiet room, then a car.", usage=(50, 14)),
        reply("A dog barking", stop="max_tokens", usage=(50, 8)),
    ]
    model = claude.module.AnthropicChatModel()
    assert model.chat("What is it?", [tool_input()], max_new_tokens=8) == "A dog barking"
    second, third = (c.kwargs for c in claude.create.call_args_list[1:])
    assert third["messages"] == second["messages"] and third["tool_choice"] == {"type": "none"}
    assert third["max_tokens"] == 8
    assert model.usage() == {"calls": 3, "cache_hits": 0, "prompt_tokens": 130, "completion_tokens": 34}


# ----------------------------------------------------------------------------- streaming
class FakeStream:
    def __init__(self, pieces, final):
        self.text_stream = iter(pieces)
        self.final = final
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.closed = True
        return False

    def get_final_message(self):
        return self.final


def test_chat_stream_yields_the_text_as_it_arrives(claude):
    stream = FakeStream([" A dog", " is", " barking. "], reply("A dog is barking.", usage=(25, 5)))
    claude.client.beta.messages.stream.return_value = stream
    model = claude.module.AnthropicChatModel()
    pieces = list(model.chat_stream("What sound?", [], max_new_tokens=12))
    assert pieces == ["A dog", " is", " barking."] and stream.closed
    kwargs = claude.client.beta.messages.stream.call_args.kwargs
    assert kwargs["max_tokens"] == 12 and kwargs["system"] == CHAT_INSTRUCTION
    assert_claude_request(kwargs)
    claude.create.assert_not_called()
    assert model.usage() == {"calls": 1, "cache_hits": 0, "prompt_tokens": 25, "completion_tokens": 5}


def test_a_refused_stream_raises_after_its_pieces_and_an_early_stop_closes_it(claude):
    claude.client.beta.messages.stream.return_value = FakeStream(["Partial"], reply([], stop="refusal"))
    pieces = claude.module.AnthropicChatModel().chat_stream("What sound?", [])
    assert next(pieces) == "Partial"
    with pytest.raises(ValueError, match="declined"):
        next(pieces)
    stream = FakeStream(["One", " two", " three"], reply("One two three"))
    claude.client.beta.messages.stream.return_value = stream
    pieces = claude.module.AnthropicChatModel().chat_stream("Talk", [])
    assert next(pieces) == "One"
    pieces.close()
    assert stream.closed


def test_chat_stream_with_a_tool_input_runs_the_tool_loop(claude):
    claude.create.side_effect = [reply([tool_use({"clip": 1})], stop="tool_use"), reply("A dog.")]
    inp = tool_input()
    assert list(claude.module.AnthropicChatModel().chat_stream("What is it?", [inp])) == ["A dog."]
    assert len(inp.tool.calls) == 1
