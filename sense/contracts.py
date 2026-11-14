"""Shared contracts for the sense harness.

Every module in sense/, agent/ and server/ codes against these types. Change a signature here only
together with every caller, and say so on the team board.

Vocabulary
  base model  the model the user chats with (local Qwen3.5, or an API model)
  specialist  a frozen model that understands one modality (CLAP for audio)
  bridge      one way of getting a modality into a base model: native, text, tool, image or vector
  sense       an installed bridge together with the report card that justified installing it
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np

BridgeKind = Literal["native", "text", "tool", "image", "vector"]
Scope = Literal["closed", "open"]  # closed: only the labelled categories matter. open: any input of the modality.


# ----------------------------------------------------------------------------- inputs
@dataclass
class SenseInput:
    """What a bridge hands to a base model for one file."""

    kind: Literal["text", "vector", "image"]
    modality: str  # "audio"
    source: str = ""  # file name, for traces and the interface
    text: str | None = None  # kind "text": inserted into the prompt
    vector: np.ndarray | None = None  # kind "vector": [hidden_size], written into the model's slots
    layer: int | None = None  # kind "vector": the block after which the vector is written
    image: bytes | None = None  # kind "image": PNG or JPEG bytes
    detail: str = ""  # one line for the interface, e.g. 'nearest caption: "a dog barks"'


@dataclass
class ChoiceItem:
    """One multiple-choice question for verification."""

    question: str
    options: list[str]
    inputs: list[SenseInput] = field(default_factory=list)


# ----------------------------------------------------------------------------- models
@dataclass
class ModelInfo:
    name: str  # stable id, e.g. "qwen3.5-0.8b"
    label: str  # shown in the interface
    native: list[str]  # modalities the model accepts as-is, e.g. ["text", "image"]
    open_weights: bool  # True allows the vector bridge
    available: bool = True  # False when weights or an API key are missing
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@runtime_checkable
class BaseModel(Protocol):
    info: ModelInfo

    def chat(self, message: str, inputs: list[SenseInput], max_new_tokens: int = 96) -> str:
        """Free-text reply to `message`, with the bridged inputs available to the model."""

    def choose(self, items: list[ChoiceItem]) -> list[int]:
        """For each item, the index of the option the model picks. Deterministic. Used by verification."""


@runtime_checkable
class OpenModel(BaseModel, Protocol):
    """An open-weight base model. Adds what the vector bridge needs."""

    hidden_size: int
    n_layers: int

    def text_states(self, texts: list[str], layer: int) -> np.ndarray:
        """Mean-pooled hidden state after block `layer` for each text: [n_texts, hidden_size]."""


# ----------------------------------------------------------------------------- data
@dataclass
class LabelledSet:
    """Labelled examples of one modality, with cached specialist embeddings."""

    name: str  # "gtzan"
    title: str  # "Music genres (GTZAN)"
    modality: str  # "audio"
    question: str  # "Which genre of music is in the audio clip?"
    labels: list[str]
    y: np.ndarray  # [n] label index of each example
    emb: np.ndarray  # [n, dim] specialist embedding of each example
    label_text_emb: np.ndarray  # [n_labels, dim] specialist text embedding of each label
    fit_idx: np.ndarray  # examples the bridge may be fitted on
    dev_idx: np.ndarray  # examples for choosing settings
    test_idx: np.ndarray  # held-out examples, scored once
    files: list[str] | None = None  # absolute paths when the raw files are on disk


# ----------------------------------------------------------------------------- bridges
@runtime_checkable
class Bridge(Protocol):
    kind: BridgeKind
    modality: str
    model_name: str
    name: str  # unique per model, e.g. "vector:gtzan" or "text:captions"

    def from_embedding(self, emb: np.ndarray, source: str = "") -> SenseInput:
        """Turn one specialist embedding [dim] into the input for the base model."""

    def prepare(self, path: Path) -> SenseInput:
        """Embed the file with the specialist, then from_embedding."""

    def save(self, directory: Path) -> None:
        """Persist everything needed to reload the bridge into `directory`."""


# ----------------------------------------------------------------------------- verification
@dataclass
class ReportCard:
    """The checkable end state of an agent run for one candidate bridge."""

    model: str
    modality: str
    dataset: str
    scope: Scope
    bridge: str  # Bridge.name
    kind: BridgeKind
    n: int  # held-out items scored
    chance: float  # e.g. 0.2 for five options
    accuracy: float  # of this bridge
    controls: dict[str, float]  # "swapped", "no_input", "lookup", "text_bridge", and "unseen_classes" for scope "open"
    verdict: Literal["installed", "rejected", "candidate"] = "candidate"
    reason: str = ""
    created: str = ""  # ISO 8601

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ----------------------------------------------------------------------------- agent
@dataclass
class FitRequest:
    model: str  # ModelInfo.name
    dataset: str  # LabelledSet.name
    modality: str = "audio"
    scope: Scope = "closed"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TraceEvent:
    """One line of the agent's visible trace."""

    step: str  # "inspect" | "plan" | "fit" | "verify" | "decide" | "install" | "done" | "error"
    status: Literal["start", "ok", "fail", "info"]
    message: str
    data: dict[str, Any] = field(default_factory=dict)
    t: float = 0.0  # seconds since the job started

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Decision:
    install: str | None  # Bridge.name to install, or None to install nothing
    reason: str


@runtime_checkable
class Planner(Protocol):
    """Chooses which bridges to try and which one to install. A rule-based planner ships with the
    prototype; an LLM planner and run-time specialist discovery implement the same interface."""

    name: str

    def plan(self, request: FitRequest, model: ModelInfo, dataset: LabelledSet) -> list[BridgeKind]:
        """Bridge kinds to build and verify, in order."""

    def decide(self, request: FitRequest, cards: list[ReportCard]) -> Decision:
        """Pick the bridge to install from the verified candidates, with the reason."""


@dataclass
class JobState:
    id: str
    request: FitRequest
    status: Literal["running", "done", "failed"] = "running"
    events: list[TraceEvent] = field(default_factory=list)
    cards: list[ReportCard] = field(default_factory=list)
    installed: str | None = None  # Bridge.name that ended up installed
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "request": self.request.to_dict(),
            "status": self.status,
            "events": [e.to_dict() for e in self.events],
            "cards": [c.to_dict() for c in self.cards],
            "installed": self.installed,
            "error": self.error,
        }
