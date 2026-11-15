"""Claude Sonnet 5.5 as a text/image base model that can call tools, through the Anthropic API.

Public surface (do not rename):
  MODEL_INFO                           registered as "claude-sonnet" (label "Claude Sonnet 5.5"): not open-weight, native
                                       text and image, tools=True, so it takes the text and tool bridges, not the
                                       vector bridge. available only when ANTHROPIC_API_KEY is set; fake mode lists
                                       only the fake models (sense.models), so it is never offered there
  AnthropicChatModel                   contracts.BaseModel over the Anthropic Messages API; the same interface as
                                       sense.models_api.OpenAIChatModel, whose base class APIChatModel it shares
  AnthropicChatModel.usage() -> dict   {"calls", "cache_hits", "prompt_tokens", "completion_tokens"} since it was loaded
  AnthropicChatModel.chat_stream(message, inputs, max_new_tokens=96, history=None) -> Iterator[str]

SENSE_CLAUDE_MODEL picks the model id (default claude-sonnet-5-5); the client has a 30 s timeout and no retries.

Everything sense.models_api says about inputs, the listen tool, the answer cache, usage(), history and streaming
holds here too, with these differences:
  Requests. client.beta.messages.create (the beta endpoint, for the refusal fallback below) with the instruction as
  the system prompt, thinking {"type": "between_tools"} and output_config {"effort": "low"}. Claude Sonnet 5.5 takes
  no temperature or top_p (a non-default value is a 400), so none is sent, and choose() cannot pin the answer to one
  value the way temperature 0 does: a repeat question can get another letter. The answer cache is what makes
  repeated verification repeatable: an answer, once given, is served from <SENSE_HOME>/api_cache/<model id>/ (unless
  SENSE_API_CACHE=0). between_tools is the model's setting without extended thinking (thinking {"type": "disabled"}
  is a 400 on it); thinking would count toward max_tokens, and the caller's limit (16 tokens for a letter, 96 for a
  chat reply by default) is meant for the answer, as with OpenAI's reasoning_effort "none".
  Images. An image input is an image block, base64 with media type image/png (image/jpeg for JPEG bytes), before
  the text block.
  Tools. The listen tool is offered as {"name", "description", "input_schema"} with tool_choice {"type": "auto"};
  the forced last request sends tool_choice {"type": "none"} (forced "any" or "tool" choices are a 400 on this
  model). Tool calls run only when stop_reason is "tool_use"; the assistant turn goes back unchanged (its content
  blocks as they came) followed by one user turn of tool_result blocks, a wrong call marked is_error. Rounds,
  token budgets and the re-ask within the caller's limit are as for OpenAI, comparing usage.output_tokens.
  Refusals. Server-side refusal fallback is on: the beta "server-side-fallback-2026-07-01" with fallbacks "default",
  so a request declined in some categories is answered by the fallback model Anthropic picks, in the same call. A
  refusal that still comes back (stop_reason "refusal") is an error: ValueError "Claude declined to answer
  (refusal ...)", like any other missing answer, so verification fails rather than inventing a score.
  Answers. The text blocks of the reply, joined and stripped. stop_reason "end_turn" is a complete answer;
  "max_tokens" is accepted for chat() only, as OpenAI's "length" is.
  Usage. prompt_tokens adds input, cache-creation and cache-read input tokens, completion_tokens the output tokens;
  when usage.iterations lists the attempts of a fallback, their message and fallback_message entries are summed.
  Streaming. chat_stream without tool inputs uses client.beta.messages.stream and yields stream.text_stream; its
  usage is counted from the final message. A refused, empty or unfinished stream raises ValueError after the pieces
  already yielded.
"""

from __future__ import annotations

import base64
import os
from typing import Iterator

from anthropic import Anthropic

from sense import config
from sense.contracts import ModelInfo, SenseInput, ToolHandle
from sense.models import chat_history, register_model, stripped_pieces
from sense.models_api import (
    CHAT_INSTRUCTION, TOOL_CALL_TOKENS, TOOL_ROUNDS, APIChatModel, _image_type, _listen, _listen_function, _prompt,
    _tokens, _tool_handles,
)

MODEL_INFO = ModelInfo(
    name="claude-sonnet",
    label="Claude Sonnet 5.5",
    native=["text", "image"],
    open_weights=False,
    available=bool(config.anthropic_key()),
    note="Requires ANTHROPIC_API_KEY; text bridges convey audio captions, and the tool bridge lets it call CLAP.",
    tools=True,
)

DEFAULT_MODEL = "claude-sonnet-5-5"
EFFORT = "low"
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AnthropicChatModel(APIChatModel):
    info = MODEL_INFO

    def __init__(self) -> None:
        key = config.anthropic_key()
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY is required for claude-sonnet")
        super().__init__(os.environ.get("SENSE_CLAUDE_MODEL") or DEFAULT_MODEL)
        self._client = Anthropic(api_key=key, timeout=30.0, max_retries=0)

    @staticmethod
    def _content(message: str, inputs: list[SenseInput]) -> str | list[dict]:
        prompt = _prompt(message, inputs)
        images = [item.image for item in inputs if item.kind == "image" and item.image]
        if not images:
            return prompt
        content = []
        for image in images:
            image = _fit_image(image)
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": _image_type(image), "data": base64.b64encode(image).decode("ascii")},
            })
        return content + [{"type": "text", "text": prompt}]

    def _request(self, instruction: str, messages: list[dict], max_tokens: int, **extra) -> dict:
        """The keyword arguments of one request. No temperature: Claude Sonnet 5.5 rejects non-default values."""
        return {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": instruction,
            "messages": messages,
            "thinking": {"type": "between_tools"},
            "output_config": {"effort": EFFORT},
            "betas": [FALLBACK_BETA],
            "fallbacks": "default",
            **extra,
        }

    def _send(self, **request) -> object:
        """One request to the API, counted in usage() whether or not it succeeds."""
        self._count(calls=1)
        response = self._client.beta.messages.create(**request)
        prompt, completion = _usage(getattr(response, "usage", None))
        self._count(prompt_tokens=prompt, completion_tokens=completion)
        return response

    def _messages(self, message: str, inputs: list[SenseInput], history: list[dict] | None = None) -> list[dict]:
        """The earlier turns (oldest first, text only), then the user message with the inputs."""
        return [*chat_history(history), {"role": "user", "content": self._content(message, inputs)}]

    def _complete(
        self, instruction: str, message: str, inputs: list[SenseInput], limit: int,
        *, allow_truncated: bool = False, history: list[dict] | None = None,
    ) -> str:
        handles = _tool_handles(inputs)
        messages = self._messages(message, inputs, history)
        if not handles:
            return _answer(self._send(**self._request(instruction, messages, limit)), allow_truncated)
        tools = [_listen_tool(handles)]
        rounds = 0  # rounds of tool calls so far
        while True:
            last = rounds == TOOL_ROUNDS  # tool calls are used up (or the reply was too long): answer now
            response = self._send(**self._request(
                instruction,
                list(messages),  # a copy, so each request keeps the messages it was sent with
                limit if last else max(limit, TOOL_CALL_TOKENS),
                tools=tools,
                tool_choice={"type": "none" if last else "auto"},
            ))
            if last:
                break
            calls = [block for block in response.content if block.type == "tool_use"]
            if response.stop_reason != "tool_use" or not calls:
                if _tokens(getattr(response, "usage", None), "output_tokens") <= limit:
                    break
                rounds = TOOL_ROUNDS  # the reply ran past the caller's limit: ask again within it
                continue
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": [_tool_result(call, handles) for call in calls]})
            rounds += 1
        return _answer(response, allow_truncated)

    def chat_stream(
        self, message: str, inputs: list[SenseInput], max_new_tokens: int = 96, history: list[dict] | None = None,
    ) -> Iterator[str]:
        """chat(), streamed. Without tool inputs: one streamed request, its text yielded as it arrives. With tool
        inputs: chat() runs (with its tool loop) and its reply is yielded once."""
        if _tool_handles(inputs):
            yield self.chat(message, inputs, max_new_tokens, history)
            return
        request = self._request(CHAT_INSTRUCTION, self._messages(message, inputs, history), max_new_tokens)
        self._count(calls=1)  # counted whether or not the request succeeds, as _send does
        texted = False
        # Leaving the with block (also when the reader stops early) closes the HTTP response.
        with self._client.beta.messages.stream(**request) as stream:
            for piece in stripped_pieces(stream.text_stream):
                texted = True
                yield piece
            final = stream.get_final_message()
        prompt, completion = _usage(getattr(final, "usage", None))
        self._count(prompt_tokens=prompt, completion_tokens=completion)
        if final.stop_reason == "refusal":
            raise ValueError(_refused(final))
        if not texted or final.stop_reason not in ("end_turn", "max_tokens"):
            raise ValueError("The API model returned no complete answer")


IMAGE_SIDE = 1568  # the longest side an image is sent at; larger images are scaled down first
IMAGE_BYTES = 3_700_000  # raw bytes, so the base64 form stays under the API's 5 MB per image


def _fit_image(image: bytes) -> bytes:
    """An image the Messages API accepts: unchanged when it is small enough, else scaled so its longest side is
    IMAGE_SIDE and re-encoded (PNG, or JPEG at quality 85 when the PNG is still over IMAGE_BYTES). Chat lets
    images of up to 20 MB through (the OpenAI limit), and Claude takes at most 5 MB of base64 per image."""
    from io import BytesIO

    from PIL import Image, UnidentifiedImageError

    try:
        opened = Image.open(BytesIO(image))
    except (UnidentifiedImageError, OSError):
        return image  # not an image PIL reads: sent as it is, and the API says what is wrong with it
    with opened as img:
        if len(image) <= IMAGE_BYTES and max(img.size) <= IMAGE_SIDE:
            return image
        img = img.convert("RGB") if img.mode not in ("RGB", "RGBA", "L") else img.copy()
        img.thumbnail((IMAGE_SIDE, IMAGE_SIDE))
        out = BytesIO()
        img.save(out, format="PNG", optimize=True)
        if out.tell() <= IMAGE_BYTES:
            return out.getvalue()
        out = BytesIO()
        img.convert("RGB").save(out, format="JPEG", quality=85)
        return out.getvalue()


def _usage(usage: object) -> tuple[int, int]:
    """(prompt tokens, completion tokens) the API reported: every attempt when usage.iterations lists them (a
    fallback), else the top-level counts; 0 for anything not reported."""
    iterations = getattr(usage, "iterations", None)
    entries = [usage]
    if isinstance(iterations, list):
        attempts = [entry for entry in iterations if getattr(entry, "type", None) in ("message", "fallback_message")]
        entries = attempts or entries
    prompt = sum(
        _tokens(entry, "input_tokens") + _tokens(entry, "cache_creation_input_tokens")
        + _tokens(entry, "cache_read_input_tokens")
        for entry in entries
    )
    return prompt, sum(_tokens(entry, "output_tokens") for entry in entries)


def _refused(response: object) -> str:
    category = getattr(getattr(response, "stop_details", None), "category", None)
    return f"Claude declined to answer (refusal{f', category {category}' if isinstance(category, str) else ''})"


def _answer(response: object, allow_truncated: bool) -> str:
    """The model's final text, or ValueError when it gave none (refused, cut off, or still calling tools)."""
    if response.stop_reason == "refusal":
        raise ValueError(_refused(response))
    finishes = ("end_turn", "max_tokens") if allow_truncated else ("end_turn",)
    text = "".join(block.text for block in response.content if block.type == "text").strip()
    if response.stop_reason not in finishes or not text:
        raise ValueError("The API model returned no complete answer")
    return text


def _listen_tool(handles: list[ToolHandle]) -> dict:
    """The listen tool as the Anthropic API takes it: the same name, description and schema as for OpenAI."""
    function = _listen_function(handles)["function"]
    return {"name": function["name"], "description": function["description"], "input_schema": function["parameters"]}


def _tool_result(call: object, handles: list[ToolHandle]) -> dict:
    """One tool_use block, run through the clip's ToolHandle, as the tool_result block that answers it."""
    text, error = _listen(getattr(call, "name", None), getattr(call, "input", None), handles)
    result = {"type": "tool_result", "tool_use_id": call.id, "content": text}
    if error:
        result["is_error"] = True
    return result


register_model(MODEL_INFO, AnthropicChatModel)
