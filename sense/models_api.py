"""OpenAI text/image base model that can call tools. SENSE_API_MODEL defaults to gpt-5.4-mini.

Public surface (do not rename):
  MODEL_INFO                        registered as "openai-chat": not open-weight and tools=True, so it takes the
                                    text and tool bridges, not the vector bridge
  OpenAIChatModel                   contracts.BaseModel over the OpenAI chat API
  OpenAIChatModel.usage() -> dict   {"calls", "cache_hits", "prompt_tokens", "completion_tokens"} since it was loaded

Choice answers are single letters at temperature zero, returned as option indices
in input order. A malformed answer fails verification rather than inventing a score.

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

from openai import OpenAI

from sense import config
from sense.contracts import ChoiceItem, ModelInfo, SenseInput, ToolHandle
from sense.models import register_model

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

TOOL = "listen"  # the one function offered; sense.bridges.ToolBridge names its handles this way
TOOL_ROUNDS = 2  # rounds of tool calls before the model must answer
TOOL_CALL_TOKENS = 256  # completion tokens for a request that offers tools: room for the calls of several clips
TOOL_INSTRUCTION = (
    " An audio clip you cannot hear directly can be heard by calling the listen tool; call it before you answer."
)


class OpenAIChatModel:
    info = MODEL_INFO

    def __init__(self) -> None:
        key = config.openai_key()
        if not key:
            raise RuntimeError("OPENAI_API_KEY is required for openai-chat")
        self.model = os.environ.get("SENSE_API_MODEL") or "gpt-5.4-mini"
        self._client = OpenAI(api_key=key, timeout=30.0, max_retries=0)
        self._usage_lock = threading.Lock()
        self._usage = {"calls": 0, "cache_hits": 0, "prompt_tokens": 0, "completion_tokens": 0}
        self._cache_warned = False

    def usage(self) -> dict[str, int]:
        """Requests sent to the API, answers served from the cache, and tokens the API reported, so far."""
        with self._usage_lock:
            return dict(self._usage)

    def _count(self, **amounts: int) -> None:
        with self._usage_lock:
            for name, amount in amounts.items():
                self._usage[name] += amount

    @staticmethod
    def _content(message: str, inputs: list[SenseInput]) -> str | list[dict]:
        lines, clip = [], 0
        for item in inputs:
            if item.kind == "text" and item.text is not None:
                lines.append(f"Audio clip: {item.text}")
            elif item.kind == "tool" and item.tool is not None:
                clip += 1  # the file name is left out: some give the answer away (GTZAN's "rock.00012.wav")
                lines.append(
                    f"Audio clip {clip}: you cannot hear it directly; call the {TOOL} tool with clip={clip} to hear it."
                )
        prompt = "\n\n".join([message, *lines])
        images = [item.image for item in inputs if item.kind == "image" and item.image]
        if not images:
            return prompt
        content = [{"type": "text", "text": prompt}]
        for image in images:
            mime = "image/jpeg" if image.startswith(b"\xff\xd8\xff") else "image/png"
            encoded = base64.b64encode(image).decode("ascii")
            content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}})
        return content

    def _send(self, **request) -> object:
        """One request to the API, counted in usage() whether or not it succeeds."""
        self._count(calls=1)
        response = self._client.chat.completions.create(**request)
        usage = getattr(response, "usage", None)
        self._count(prompt_tokens=_tokens(usage, "prompt_tokens"), completion_tokens=_tokens(usage, "completion_tokens"))
        return response

    def _complete(
        self, instruction: str, message: str, inputs: list[SenseInput], limit: int,
        *, allow_truncated: bool = False,
    ) -> str:
        handles = _tool_handles(inputs)
        request = {"model": self.model, "temperature": 0, "reasoning_effort": "none"}
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": self._content(message, inputs)},
        ]
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

    def chat(self, message: str, inputs: list[SenseInput], max_new_tokens: int = 96) -> str:
        instruction = "Answer the user's question using the supplied inputs. Audio clip text is a specialist's caption."
        if _tool_handles(inputs):
            instruction += TOOL_INSTRUCTION
        return self._complete(instruction, message, inputs, max_new_tokens, allow_truncated=True)

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
        answer = self._complete(instruction, message, item.inputs, limit)
        if len(answer) != 1 or answer not in valid:
            raise ValueError("The API model must return one valid option letter")
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
    name = getattr(function, "name", None)
    if name != TOOL:
        return f"Error: there is no tool called {name!r}; the only tool is {TOOL}."
    try:
        arguments = json.loads(getattr(function, "arguments", None) or "{}")
    except (TypeError, ValueError):
        arguments = None
    if not isinstance(arguments, dict):
        return f'{TOOL} error: the arguments must be a JSON object, such as {{"clip": 1, "candidates": ["rock", "jazz"]}}.'
    clip = arguments.pop("clip", None)
    if clip is None and len(handles) == 1:
        clip = 1  # one clip: nothing to choose between
    if isinstance(clip, float) and clip.is_integer():
        clip = int(clip)
    if isinstance(clip, bool) or not isinstance(clip, int) or not 1 <= clip <= len(handles):
        return (
            f"{TOOL} error: clip must be the number of an audio clip, from 1 to {len(handles)}, "
            f"but it was {repr(clip)[:40]}."
        )
    return handles[clip - 1].call(arguments)


def _safe_model_id(model: str) -> str:
    """The model id as one directory name."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", model)
    return cleaned if cleaned.strip(".") else cleaned.replace(".", "_") or "_"


register_model(MODEL_INFO, OpenAIChatModel)
