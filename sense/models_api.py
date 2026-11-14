"""OpenAI text/image base model. SENSE_API_MODEL defaults to gpt-5.4-mini.

Choice answers are single letters at temperature zero, returned as option indices
in input order. A malformed answer fails verification rather than inventing a score.
"""

from __future__ import annotations

import base64
import os
import string
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

from sense import config
from sense.contracts import ChoiceItem, ModelInfo, SenseInput
from sense.models import register_model


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
        answer = self._complete(
            "Answer the multiple-choice question using the supplied inputs. Return exactly one uppercase "
            "option letter, with no explanation or punctuation. Audio clip text is a specialist's caption.",
            f"{item.question}\n{options}",
            item.inputs,
            16,
        )
        if len(answer) != 1 or answer not in string.ascii_uppercase[:len(item.options)]:
            raise ValueError("The API model must return one valid option letter")
        return string.ascii_uppercase.index(answer)

    def choose(self, items: list[ChoiceItem]) -> list[int]:
        if not items:
            return []
        if any(not 1 <= len(item.options) <= 26 for item in items):
            raise ValueError("Each choice item needs between 1 and 26 options")
        with ThreadPoolExecutor(max_workers=min(4, len(items))) as pool:
            return list(pool.map(self._choose, items))


register_model(MODEL_INFO, OpenAIChatModel)
