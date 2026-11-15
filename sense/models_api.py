"""OpenAI text/image base model that can call tools. SENSE_API_MODEL defaults to gpt-6.1-terra.

Public surface (do not rename):
  MODEL_INFO                        registered as "openai-chat": not open-weight and tools=True, so it takes the
                                    text and tool bridges, not the vector bridge
  OpenAIChatModel                   contracts.BaseModel over the OpenAI chat API
  APIChatModel                      what an API base model shares (usage, chat, choose and the answer cache); a
                                    subclass sets model and implements _content, _complete and chat_stream.
                                    sense.models_anthropic.AnthropicChatModel is the other one
  OpenAIChatModel.usage() -> dict   {"calls", "cache_hits", "prompt_tokens", "completion_tokens"} since it was loaded
  OpenAIChatModel.chat_stream(message, inputs, max_new_tokens=96, history=None) -> Iterator[str]
                                    the reply in pieces, as it arrives (see "History and streaming" below)

Choice answers are single letters at temperature zero, returned as option indices
in input order. A reply that is not one option letter (an explanation, a refusal, a reply cut off at the token
limit; a leading letter followed by punctuation, such as "B.", counts as that letter) is an abstention: choose
returns ABSTAIN (-1) for that item, which never matches an answer, so it counts as wrong rather than inventing a
score, and the model's `abstained` counts them. Abstentions are not cached. A model asked with no input (the
no-input control) often abstains, which is the honest outcome. Every request sends
temperature 0 together with reasoning_effort "none": gpt-6.1-terra takes temperature 0 only with reasoning_effort
"none" (temperature 0 alone is a 400).

Inputs. A text input becomes a line "Audio clip: <text>" after the message. A tool input (SenseInput kind
"tool", from sense.bridges.make_tool_bridge) becomes a line "Audio clip N: you cannot hear it directly; call the
listen tool with clip=N to hear it.", numbering the tool inputs from 1. The clip's file name is left out on
purpose: some file names give the answer away (GTZAN's are "rock.00012.wav"). When any input is a tool input the
request offers one function, listen, with arguments {"clip": the clip number, "candidates": optional list of
strings}, and tool_choice "auto". Each tool call the model makes is run through that clip's ToolHandle.call; a
bad clip number, an unknown function or arguments that are not a JSON object come back to the model as an error
text. The assistant turn with its tool calls and one "tool" message per call are added, and the model is asked
again. After TOOL_ROUNDS rounds of tool calls the last request is sent with tool_choice "none", so the model
must answer. A request in which the model may call the tool allows up to TOOL_CALL_TOKENS completion tokens (a
tool call does not fit in the 16 tokens of a letter answer); the forced last request has the caller's limit
(max_new_tokens, or 16 for a letter). The answer still keeps to the caller's limit: when the model answers in
a request that allowed more, and the API reports more completion tokens than the limit, the same messages are
sent once more with tool_choice "none" and the caller's limit, and that reply is the answer. So a tool input
does not let chat(max_new_tokens=32) reply at greater length (if the API reported no token counts, the reply is
kept as it came). The final answer is checked as before.

Answer cache. Every multiple-choice answer that passes that check is kept on disk, so a repeat agent run gets
the same verification answers without asking the API again (the LLM planner's requests, which do not go through
this module, and chat() replies are not cached). Each answer is one file,
<SENSE_HOME>/api_cache/<model id>/<key[:2]>/<key>.json holding {"answer", "model", "created"}, where the key is
the sha256 of canonical JSON of the model id, the system instruction, the user message content exactly as sent
(image data included) and the token limit, plus, when there are tool inputs, the ToolHandle.key of each (which
identifies the clip, so two clips with the same prompt never share an answer); changing any of them asks the API
again. An answer served from the cache calls no tool, so the ToolHandle.calls of that question stay empty. Files
are written to a temporary name and renamed into place, because choose() runs four requests at once. Set
SENSE_API_CACHE=0 to turn the cache off, or delete the directory to forget the answers. chat() replies are never
cached.

usage() counts requests sent to the API by chat() and choose() (including any that failed, every round of a
question with tool calls, and a request sent again because the reply was too long), answers served from the
cache, and the prompt and completion tokens the API reported.

History and streaming. chat(history=...) sends the earlier turns (contracts.BaseModel.chat) as user and assistant
messages between the system message and the final user message, text only. Nothing with a history is cached:
only choose() answers are, and choose() has no history. chat_stream() takes the same arguments as chat(). Without
tool inputs it sends one request with stream=True and stream_options {"include_usage": True}, with
max_completion_tokens max_new_tokens, and yields the text as it arrives, stripped like chat()'s reply; usage()
counts it as one call, plus the tokens if the stream reports them. A refused, empty or unfinished stream raises
ValueError after the pieces already yielded, as chat() would. With tool inputs it runs chat(), tool loop and all,
and yields the whole reply once.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import re
import string
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from openai import OpenAI

from sense import config
from sense.contracts import ChoiceItem, ModelInfo, SenseInput, ToolHandle
from sense.models import chat_history, register_model, stripped_pieces

_log = logging.getLogger(__name__)

MODEL_INFO = ModelInfo(
    name="openai-chat",
    label="OpenAI chat (API)",
    native=["text", "image"],
    open_weights=False,
    available=bool(config.openai_key()),
    note="Requires OPENAI_API_KEY; text bridges convey audio captions, and the tool bridge lets it call CLAP.",
    tools=True,
)

DEFAULT_MODEL = "gpt-6.1-terra"
TOOL = "listen"  # the one function offered; sense.bridges.ToolBridge names its handles this way
TOOL_ROUNDS = 2  # rounds of tool calls before the model must answer
TOOL_CALL_TOKENS = 256  # completion tokens for a request that offers tools: room for the calls of several clips
TOOL_INSTRUCTION = (
    " An audio clip you cannot hear directly can be heard by calling the listen tool; call it before you answer."
)
CHAT_INSTRUCTION = "Answer the user's question using the supplied inputs. Audio clip text is a specialist's caption."


class APIChatModel:
    """What an API base model shares: usage(), chat(), choose() and the answer cache. A subclass sets self.model (the
    model id, also the cache directory) and implements _content(message, inputs) (the user message content as sent,
    part of the cache key), _complete(instruction, message, inputs, limit, *, allow_truncated=False, history=None)
    (the model's final text, tool loop and all) and chat_stream."""

    info: ModelInfo

    def __init__(self, model: str) -> None:
        self.model = model
        self._usage_lock = threading.Lock()
        self._usage = {"calls": 0, "cache_hits": 0, "prompt_tokens": 0, "completion_tokens": 0}
        self._cache_warned = False
        self.abstained = 0  # choice items answered with something other than one option letter (choose gave -1)

    def usage(self) -> dict[str, int]:
        """Requests sent to the API, answers served from the cache, and tokens the API reported, so far."""
        with self._usage_lock:
            return dict(self._usage)

    def _count(self, **amounts: int) -> None:
        with self._usage_lock:
            for name, amount in amounts.items():
                self._usage[name] += amount

    def _content(self, message: str, inputs: list[SenseInput]) -> object:
        raise NotImplementedError

    def _complete(
        self, instruction: str, message: str, inputs: list[SenseInput], limit: int,
        *, allow_truncated: bool = False, history: list[dict] | None = None,
    ) -> str:
        raise NotImplementedError

    def chat(
        self, message: str, inputs: list[SenseInput], max_new_tokens: int = 96, history: list[dict] | None = None,
    ) -> str:
        instruction = CHAT_INSTRUCTION
        if _tool_handles(inputs):
            instruction += TOOL_INSTRUCTION
        return self._complete(instruction, message, inputs, max_new_tokens, allow_truncated=True, history=history)

    def _choose(self, item: ChoiceItem) -> int:
        options = "\n".join(f"{letter}. {text}" for letter, text in zip(string.ascii_uppercase, item.options))
        instruction = (
            "Answer the multiple-choice question using the supplied inputs. Return exactly one uppercase "
            "option letter, with no explanation or punctuation. Audio clip text is a specialist's caption."
        )
        if _tool_handles(item.inputs):
            instruction += TOOL_INSTRUCTION
        message, limit = f"{item.question}\n{options}", 16
        valid = string.ascii_uppercase[:len(item.options)]
        cached_at = self._cache_path(instruction, message, item.inputs, limit) if config.api_cache_enabled() else None
        if cached_at is not None:
            cached = self._cache_read(cached_at, valid)
            if cached is not None:
                return string.ascii_uppercase.index(cached)
        try:
            answer = _letter(self._complete(instruction, message, item.inputs, limit), valid)
        except ValueError:  # cut off while explaining, refused, or no text at all
            answer = None
        if answer is None:
            # Not one option letter (for example "I can't see the 3D model..." when the question comes with no
            # input): an abstention. -1 never matches an answer, so it counts as wrong; it is not cached.
            with self._usage_lock:
                self.abstained += 1
            return ABSTAIN
        if cached_at is not None:
            self._cache_write(cached_at, answer)
        return string.ascii_uppercase.index(answer)

    # ------------------------------------------------------------------------- answer cache
    def _cache_path(self, instruction: str, message: str, inputs: list[SenseInput], limit: int) -> Path:
        content = self._content(message, inputs)
        request = {"model": self.model, "instruction": instruction, "content": content, "limit": limit}
        handles = _tool_handles(inputs)
        if handles:  # the prompt of a tool input does not say which clip it is; the tool's key does
            request["tools"] = [handle.key for handle in handles]
        key = hashlib.sha256(json.dumps(request, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        return config.home() / "api_cache" / _safe_model_id(self.model) / key[:2] / f"{key}.json"

    def _cache_read(self, path: Path, valid: str) -> str | None:
        """The cached letter, or None when there is none (or the file is unreadable or not a valid answer)."""
        try:
            answer = json.loads(path.read_text()).get("answer")
        except (OSError, ValueError, AttributeError):
            return None
        if not isinstance(answer, str) or len(answer) != 1 or answer not in valid:
            return None
        self._count(cache_hits=1)
        return answer

    def _cache_write(self, path: Path, answer: str) -> None:
        """Write atomically: a temporary file in the same directory, then os.replace. A failure only warns."""
        created = datetime.now(timezone.utc).isoformat(timespec="seconds")
        record = {"answer": answer, "model": self.model, "created": created}
        temporary = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.stem[:12]}.", suffix=".tmp")
            with os.fdopen(handle, "w") as out:
                json.dump(record, out)
            os.replace(temporary, path)
        except OSError as err:
            if temporary is not None:
                Path(temporary).unlink(missing_ok=True)
            with self._usage_lock:
                warned, self._cache_warned = self._cache_warned, True
            if not warned:
                _log.warning(
                    "Could not save an API answer under %s (%s), so a repeat run will ask the API again. Set "
                    "SENSE_HOME to a writable directory, or set SENSE_API_CACHE=0 to turn the cache off.",
                    path.parent, err,
                )

    def choose(self, items: list[ChoiceItem]) -> list[int]:
        if not items:
            return []
        if any(not 1 <= len(item.options) <= 26 for item in items):
            raise ValueError("Each choice item needs between 1 and 26 options")
        with ThreadPoolExecutor(max_workers=min(4, len(items))) as pool:
            return list(pool.map(self._choose, items))


class OpenAIChatModel(APIChatModel):
    info = MODEL_INFO

    def __init__(self) -> None:
        key = config.openai_key()
        if not key:
            raise RuntimeError("OPENAI_API_KEY is required for openai-chat")
        super().__init__(os.environ.get("SENSE_API_MODEL") or DEFAULT_MODEL)
        self._client = OpenAI(api_key=key, timeout=30.0, max_retries=0)

    @staticmethod
    def _content(message: str, inputs: list[SenseInput]) -> str | list[dict]:
        prompt = _prompt(message, inputs)
        images = [item.image for item in inputs if item.kind == "image" and item.image]
        if not images:
            return prompt
        content = [{"type": "text", "text": prompt}]
        for image in images:
            encoded = base64.b64encode(image).decode("ascii")
            content.append({"type": "image_url", "image_url": {"url": f"data:{_image_type(image)};base64,{encoded}"}})
        return content

    def _send(self, **request) -> object:
        """One request to the API, counted in usage() whether or not it succeeds."""
        self._count(calls=1)
        response = self._client.chat.completions.create(**request)
        usage = getattr(response, "usage", None)
        self._count(prompt_tokens=_tokens(usage, "prompt_tokens"), completion_tokens=_tokens(usage, "completion_tokens"))
        return response

    def _messages(
        self, instruction: str, message: str, inputs: list[SenseInput], history: list[dict] | None = None,
    ) -> list[dict]:
        """The system message, the earlier turns (oldest first, text only), then the user message with the inputs."""
        return [
            {"role": "system", "content": instruction},
            *chat_history(history),
            {"role": "user", "content": self._content(message, inputs)},
        ]

    def _complete(
        self, instruction: str, message: str, inputs: list[SenseInput], limit: int,
        *, allow_truncated: bool = False, history: list[dict] | None = None,
    ) -> str:
        handles = _tool_handles(inputs)
        # temperature 0 only together with reasoning_effort "none": gpt-6.1-terra rejects temperature 0 alone (400)
        request = {"model": self.model, "temperature": 0, "reasoning_effort": "none"}
        messages = self._messages(instruction, message, inputs, history)
        if not handles:
            return _answer(self._send(**request, max_completion_tokens=limit, messages=messages), allow_truncated)
        tools = [_listen_function(handles)]
        rounds = 0  # rounds of tool calls so far
        while True:
            last = rounds == TOOL_ROUNDS  # tool calls are used up (or the reply was too long): answer now
            response = self._send(
                **request,
                max_completion_tokens=limit if last else max(limit, TOOL_CALL_TOKENS),
                messages=list(messages),  # a copy, so each request keeps the messages it was sent with
                tools=tools,
                tool_choice="none" if last else "auto",
            )
            if last:
                break
            reply = response.choices[0].message
            calls = list(getattr(reply, "tool_calls", None) or [])
            if not calls:
                if _tokens(getattr(response, "usage", None), "completion_tokens") <= limit:
                    break
                rounds = TOOL_ROUNDS  # the reply ran past the caller's limit: ask again within it
                continue
            messages.append(_assistant_turn(reply, calls))
            for call in calls:
                messages.append({"role": "tool", "tool_call_id": call.id, "content": _run_tool(call, handles)})
            rounds += 1
        return _answer(response, allow_truncated)

    def chat_stream(
        self, message: str, inputs: list[SenseInput], max_new_tokens: int = 96, history: list[dict] | None = None,
    ) -> Iterator[str]:
        """chat(), streamed. Without tool inputs: one request with stream=True, its text yielded as it arrives.
        With tool inputs: chat() runs (with its tool loop) and its reply is yielded once."""
        if _tool_handles(inputs):
            yield self.chat(message, inputs, max_new_tokens, history)
            return
        messages = self._messages(CHAT_INSTRUCTION, message, inputs, history)
        self._count(calls=1)  # counted whether or not the request succeeds, as _send does
        stream = self._client.chat.completions.create(  # temperature 0 needs reasoning_effort "none", as above
            model=self.model, temperature=0, reasoning_effort="none", max_completion_tokens=max_new_tokens,
            messages=messages, stream=True, stream_options={"include_usage": True},
        )
        seen = {"finish": None, "refused": False, "text": False}

        def deltas() -> Iterator[str]:
            for chunk in stream:
                usage = getattr(chunk, "usage", None)
                if usage is not None:  # the last chunk, when the API reports usage
                    self._count(
                        prompt_tokens=_tokens(usage, "prompt_tokens"),
                        completion_tokens=_tokens(usage, "completion_tokens"),
                    )
                for choice in getattr(chunk, "choices", None) or []:
                    delta = getattr(choice, "delta", None)
                    if getattr(delta, "refusal", None):
                        seen["refused"] = True
                    text = getattr(delta, "content", None)
                    if isinstance(text, str) and text:
                        yield text
                    if getattr(choice, "finish_reason", None):
                        seen["finish"] = choice.finish_reason

        try:
            for piece in stripped_pieces(deltas()):
                seen["text"] = True
                yield piece
        finally:
            close = getattr(stream, "close", None)  # frees the HTTP response, also when the reader stops early
            if callable(close):
                close()
        if seen["refused"] or not seen["text"] or seen["finish"] not in ("stop", "length"):
            raise ValueError("The API model returned no complete answer")


def _prompt(message: str, inputs: list[SenseInput]) -> str:
    """The message, then one line per text input ("Audio clip: <text>") and per tool input (numbered from 1)."""
    lines, clip = [], 0
    for item in inputs:
        if item.kind == "text" and item.text is not None:
            lines.append(f"Audio clip: {item.text}")
        elif item.kind == "tool" and item.tool is not None:
            clip += 1  # the file name is left out: some give the answer away (GTZAN's "rock.00012.wav")
            lines.append(
                f"Audio clip {clip}: you cannot hear it directly; call the {TOOL} tool with clip={clip} to hear it."
            )
    return "\n\n".join([message, *lines])


ABSTAIN = -1  # what choose returns for an item the model did not answer with one option letter


def _letter(answer: str, valid: str) -> str | None:
    """The option letter in a reply: the reply itself when it is one valid letter, or its first character when that
    is a valid letter followed by punctuation or a line break ("B.", "B) chair", "C\\n..."); None for anything else,
    including a reply that starts with a word ("I can't see...", "A model of...", "A because...")."""
    answer = answer.strip()
    if len(answer) == 1:
        return answer if answer in valid else None
    if answer[:1] in valid and answer[1:2] in (".", ")", ":", ",", ";", "\n"):
        return answer[0]
    return None


def _image_type(image: bytes) -> str:
    """The media type of an image input: JPEG by its magic bytes, PNG otherwise."""
    return "image/jpeg" if image.startswith(b"\xff\xd8\xff") else "image/png"


def _tokens(usage: object, name: str) -> int:
    """A token count from response.usage, or 0 when the API did not report it."""
    value = getattr(usage, name, None)
    return value if isinstance(value, int) else 0


def _answer(response: object, allow_truncated: bool) -> str:
    """The model's final text, or ValueError when it gave none (refused, cut off, or still calling tools)."""
    choice = response.choices[0]
    finishes = ("stop", "length") if allow_truncated else ("stop",)
    if choice.finish_reason not in finishes or choice.message.refusal or not choice.message.content:
        raise ValueError("The API model returned no complete answer")
    return choice.message.content.strip()


# ----------------------------------------------------------------------------- tool calls
def _tool_handles(inputs: list[SenseInput]) -> list[ToolHandle]:
    """The ToolHandle of each tool input, in order: clip 1, clip 2, ..."""
    handles = [item.tool for item in inputs if item.kind == "tool" and item.tool is not None]
    for handle in handles:
        if handle.name != TOOL:
            raise ValueError(
                f"A tool input offers a tool called {handle.name!r}, but this adapter wires only the {TOOL} tool. "
                "Build tool inputs with sense.bridges.make_tool_bridge."
            )
    return handles


def _listen_function(handles: list[ToolHandle]) -> dict:
    """The one function offered to the model: listen, with the clip number and the tool's own arguments."""
    n = len(handles)
    properties = {
        "clip": {
            "type": "integer",
            "minimum": 1,
            "maximum": n,
            "description": f"Which audio clip to listen to: the N of 'Audio clip N', from 1 to {n}.",
        },
        **handles[0].parameters.get("properties", {}),
    }
    return {
        "type": "function",
        "function": {
            "name": TOOL,
            "description": f"{handles[0].description} Say which audio clip with clip.",
            "parameters": {"type": "object", "properties": properties, "required": ["clip"]},
        },
    }


def _assistant_turn(reply: object, calls: list) -> dict:
    """The assistant message that made `calls`, as it must be sent back before the tool results."""
    return {
        "role": "assistant",
        "content": getattr(reply, "content", None),
        "tool_calls": [
            {"id": call.id, "type": "function", "function": {"name": call.function.name, "arguments": call.function.arguments}}
            for call in calls
        ],
    }


def _run_tool(call: object, handles: list[ToolHandle]) -> str:
    """Run one tool call through the clip's ToolHandle. Anything the model got wrong comes back as an error text."""
    function = getattr(call, "function", None)
    try:
        arguments = json.loads(getattr(function, "arguments", None) or "{}")
    except (TypeError, ValueError):
        arguments = None
    return _listen(getattr(function, "name", None), arguments, handles)[0]


def _listen(name: object, arguments: object, handles: list[ToolHandle]) -> tuple[str, bool]:
    """Run one call of the listen tool with its parsed arguments: (the text the model sees, whether it is an error
    text). A wrong tool name, arguments that are not an object or a bad clip number come back as an error text."""
    if name != TOOL:
        return f"Error: there is no tool called {name!r}; the only tool is {TOOL}.", True
    if not isinstance(arguments, dict):
        return (
            f'{TOOL} error: the arguments must be a JSON object, such as {{"clip": 1, "candidates": ["rock", "jazz"]}}.',
            True,
        )
    arguments = dict(arguments)  # the clip number is taken out; the caller's object is left as it was
    clip = arguments.pop("clip", None)
    if clip is None and len(handles) == 1:
        clip = 1  # one clip: nothing to choose between
    if isinstance(clip, float) and clip.is_integer():
        clip = int(clip)
    if isinstance(clip, bool) or not isinstance(clip, int) or not 1 <= clip <= len(handles):
        return (
            f"{TOOL} error: clip must be the number of an audio clip, from 1 to {len(handles)}, "
            f"but it was {repr(clip)[:40]}."
        ), True
    return handles[clip - 1].call(arguments), False


def _safe_model_id(model: str) -> str:
    """The model id as one directory name."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", model)
    return cleaned if cleaned.strip(".") else cleaned.replace(".", "_") or "_"


register_model(MODEL_INFO, OpenAIChatModel)
