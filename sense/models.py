"""Base models the user can chat with.

Public surface (do not rename):
  available_models() -> list[ModelInfo]     every known model, with available=False when it cannot be loaded
  load_model(name) -> BaseModel             cached: one instance per name; loading a second local model
                                            releases the first (the laptop has 24 GB)
  register_model(info, loader)              lets other modules (for example an API adapter) add a model

Local open-weight models implement contracts.OpenModel:
  "qwen3.5-0.8b", "qwen3.5-2b"   frozen Qwen3.5 through transformers; reference: socket_pilot/host.py.
  Vector inputs are written into placeholder "slot" tokens after block `SenseInput.layer`; text inputs are
  put into the prompt; several inputs may be given at once.
  "fake"                         deterministic model for tests and interface work (only in fake mode)
"""

from __future__ import annotations

from typing import Callable

from sense.contracts import BaseModel, ModelInfo


def available_models() -> list[ModelInfo]:
    raise NotImplementedError


def load_model(name: str) -> BaseModel:
    raise NotImplementedError


def register_model(info: ModelInfo, loader: Callable[[], BaseModel]) -> None:
    raise NotImplementedError
