"""Shared contracts for the sense harness.

Every module in sense/, agent/ and server/ codes against these types. Change a signature here only
together with every caller, and say so on the team board.

Vocabulary
  base model  the model the user chats with (local Qwen3.5, or an API model)
  specialist  a frozen model that understands one modality (CLAP for audio)
  bridge      one way of getting a modality into a base model: native, text, tool, image or vector
  sense       an installed bridge together with the report card that justified installing it
  tool        a function the base model may call while it answers (ToolHandle); the tool bridge offers the
              specialist this way, to models that can call tools (ModelInfo.tools)

SenseInput is never sent over HTTP (a tool input holds a Python callable); the server reports its `detail` instead.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Protocol, runtime_checkable

import numpy as np

BridgeKind = Literal["native", "text", "tool", "image", "vector"]
Scope = Literal["closed", "open"]  # closed: only the labelled categories matter. open: any input of the modality.


# ----------------------------------------------------------------------------- inputs
@dataclass
class ToolHandle:
    """A tool the base model may call to ask a specialist about one input (a SenseInput of kind "tool").

    A tool bridge builds one per file. The model adapter offers it to the model and runs `call` with the
    arguments the model sent; the text `call` returns is what the model sees. A tool bridge's `call` answers bad
    arguments with an error text for the model instead of raising, and logs every call in `calls`."""

    name: str  # the function name the model sees, e.g. "listen"
    description: str  # what the tool does, for the model
    parameters: dict  # JSON schema of the arguments
    call: Callable[[dict], str]  # runs the tool with the model's parsed arguments; returns the text the model sees
    key: str  # what the tool would answer about (e.g. sha256 of the clip's embedding plus the bridge name), so
    #           that an answer cache never gives one clip's answer to another
    calls: list[dict] = field(default_factory=list)  # one {"arguments", "result"} per call, appended by call wrappers


@dataclass
class SenseInput:
    """What a bridge hands to a base model for one file. Never serialised: a tool input holds a callable."""

    kind: Literal["text", "vector", "image", "tool"]
    modality: str  # "audio"
    source: str = ""  # file name, for traces and the interface
    text: str | None = None  # kind "text": inserted into the prompt
    vector: np.ndarray | None = None  # kind "vector": [hidden_size], written into the model's slots
    layer: int | None = None  # kind "vector": the block after which the vector is written
    image: bytes | None = None  # kind "image": PNG or JPEG bytes
    detail: str = ""  # one line for the interface, e.g. 'nearest caption: "a dog barks"'
    tool: ToolHandle | None = None  # kind "tool": the model learns about the file only by calling this tool,
    #                                 which asks the specialist


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
    tools: bool = False  # the model can call tools; True allows the tool bridge

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@runtime_checkable
class BaseModel(Protocol):
    """A model the user chats with.

    Optional streaming. A model may also have
        chat_stream(message, inputs, max_new_tokens=96, history=None) -> Iterator[str]
    which takes the same arguments as chat() and yields pieces of the reply in order; their concatenation is the
    reply chat() would give. It is not part of this protocol: callers check for it (getattr) and fall back to
    chat() when a model lacks it. A consumer may stop reading early; the model must then free whatever it holds
    (a lock, a worker thread, an HTTP response) when the iterator is closed. An error during generation is raised
    to the consumer, after the pieces already yielded."""

    info: ModelInfo

    def chat(
        self, message: str, inputs: list[SenseInput], max_new_tokens: int = 96, history: list[dict] | None = None,
    ) -> str:
        """Free-text reply to `message`, with the bridged inputs available to the model.

        `history` holds the earlier turns of the conversation, oldest first: a list of {"role": "user" or
        "assistant", "content": str}. It is text only: the inputs of earlier turns are not sent again, only
        `inputs` (for this message) reach the model. None or [] starts a new conversation."""

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
    # Where the set came from and how it was built, for the agent and the interface. Empty for the built-in sets;
    # for a set built by sense.datasets.from_folder: source, licence, folder, counts of duplicates, skipped files...
    meta: dict[str, Any] = field(default_factory=dict)


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
    controls: dict[str, float]  # "swapped", "no_input", "lookup", "specialist_alone", "text_bridge", and "unseen_classes"
    #                             for scope "open"; an adapter card (sense.adapters.verify_files) may add "native"
    #                             (the model given the raw file as text) and "unseen_categories" (a second probe set
    #                             whose categories the adapter was not built around)
    verdict: Literal["installed", "rejected", "candidate"] = "candidate"
    reason: str = ""
    created: str = ""  # ISO 8601
    interval: list[float] | None = None  # 95% Wilson interval for `accuracy`, [low, high]
    control_chance: dict[str, float] = field(default_factory=dict)  # a control's own chance level, only for the
    #                             controls whose chance differs from `chance` (e.g. "unseen_categories", scored on a
    #                             set with another number of labels); records written before it existed load as {}

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

    # A fit run (agent/loop.py): "inspect" | "plan" | "fit" | "verify" | "decide" | "install" | "done" | "error".
    # A run from one file and a goal (agent/auto.py) adds "detect", "research" and "data" before "inspect", and
    # "claims" between "verify" and "decide". Each module's docstring says what its steps carry.
    step: str
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
