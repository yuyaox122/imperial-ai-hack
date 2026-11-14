"""OpenAI text/image base model. SENSE_API_MODEL defaults to gpt-5.4-mini.

Public surface (do not rename):
  MODEL_INFO                        registered as "openai-chat": not open-weight, so it takes text bridges only
  OpenAIChatModel                   contracts.BaseModel over the OpenAI chat API
  OpenAIChatModel.usage() -> dict   {"calls", "cache_hits", "prompt_tokens", "completion_tokens"} since it was loaded

Choice answers are single letters at temperature zero, returned as option indices
in input order. A malformed answer fails verification rather than inventing a score.

Answer cache. Every multiple-choice answer that passes that check is kept on disk, so a repeat agent run gets
the same verification answers without asking the API again (the LLM planner's requests, which do not go through
this module, and chat() replies are not cached). Each answer is one file,
<SENSE_HOME>/api_cache/<model id>/<key[:2]>/<key>.json holding {"answer", "model", "created"}, where the key is
the sha256 of canonical JSON of the model id, the system instruction, the user message content exactly as sent
(image data included) and the token limit; changing any of them asks the API again. Files are written to a
temporary name and renamed into place, because choose() runs four requests at once. Set SENSE_API_CACHE=0 to
turn the cache off, or delete the directory to forget the answers. chat() replies are never cached.

usage() counts requests sent to the API by chat() and choose() (including any that failed), answers served
from the cache, and the prompt and completion tokens the API reported.
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
from sense.contracts import ChoiceItem, ModelInfo, SenseInput
from sense.models import register_model

_log = logging.getLogger(__name__)

MODEL_INFO = ModelInfo(
    name="openai-chat",
    label="OpenAI chat (API)",
    native=["text", "image"],
    open_weights=False,
    available=bool(config.openai_key()),
    note="Requires OPENAI_API_KEY; text bridges convey audio captions.",
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
        texts = [f"Audio clip: {item.text}" for item in inputs if item.kind == "text" and item.text is not None]
        prompt = "\n\n".join([message, *texts])
        images = [item.image for item in inputs if item.kind == "image" and item.image]
        if not images:
            return prompt
        content = [{"type": "text", "text": prompt}]
        for image in images:
            mime = "image/jpeg" if image.startswith(b"\xff\xd8\xff") else "image/png"
            encoded = base64.b64encode(image).decode("ascii")
            content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}})
        return content

    def _complete(
        self, instruction: str, message: str, inputs: list[SenseInput], limit: int,
        *, allow_truncated: bool = False,
    ) -> str:
        self._count(calls=1)
        response = self._client.chat.completions.create(
            model=self.model,
            temperature=0,
            reasoning_effort="none",
            max_completion_tokens=limit,
            messages=[
                {"role": "system", "content": instruction},
                {"role": "user", "content": self._content(message, inputs)},
            ],
        )
        usage = getattr(response, "usage", None)
        self._count(prompt_tokens=_tokens(usage, "prompt_tokens"), completion_tokens=_tokens(usage, "completion_tokens"))
        choice = response.choices[0]
        finishes = ("stop", "length") if allow_truncated else ("stop",)
        if choice.finish_reason not in finishes or choice.message.refusal or not choice.message.content:
            raise ValueError("The API model returned no complete answer")
        return choice.message.content.strip()

    def chat(self, message: str, inputs: list[SenseInput], max_new_tokens: int = 96) -> str:
        return self._complete(
            "Answer the user's question using the supplied inputs. Audio clip text is a specialist's caption.",
            message,
            inputs,
            max_new_tokens,
            allow_truncated=True,
        )

    def _choose(self, item: ChoiceItem) -> int:
        options = "\n".join(f"{letter}. {text}" for letter, text in zip(string.ascii_uppercase, item.options))
        instruction = (
            "Answer the multiple-choice question using the supplied inputs. Return exactly one uppercase "
            "option letter, with no explanation or punctuation. Audio clip text is a specialist's caption."
        )
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


def _safe_model_id(model: str) -> str:
    """The model id as one directory name."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", model)
    return cleaned if cleaned.strip(".") else cleaned.replace(".", "_") or "_"


register_model(MODEL_INFO, OpenAIChatModel)
