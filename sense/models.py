"""Base models the user can chat with.

Public surface (do not rename):
  available_models() -> list[ModelInfo]     every known model, with available=False when it cannot be loaded
  load_model(name) -> BaseModel             cached: one instance per name; loading a second local model
                                            releases the first (the laptop has 24 GB)
  register_model(info, loader)              lets other modules (for example an API adapter) add a model

Local open-weight models implement contracts.OpenModel:
  "qwen3.5-0.8b", "qwen3.5-2b"   frozen Qwen3.5 through transformers; reference: socket_pilot/host.py.
  Vector inputs are written into placeholder "slot" tokens after block `SenseInput.layer`; text inputs are
  put into the prompt; several inputs may be given at once. Tool inputs are refused: local tool calling is not
  wired, so their ModelInfo has tools=False.

Fake mode (SENSE_FAKE=1) lists two deterministic models for tests and interface work, both FakeModel:
  "fake"                         open-weight, no tools: takes the vector and text bridges
  "fake-api"                     not open-weight, can call tools: stands in for an API model, so it takes the
                                 text and tool bridges (FakeModel itself would also take vector inputs, but
                                 bridges.build refuses to build a vector bridge for it, as for a real API model)

Prompt shape: one line per input ("Audio clip: <4 slot tokens>" or "Audio clip: <text>"), then the message or
the question. API models may be added in sense/models_api.py, which is imported here if it exists.
"""

from __future__ import annotations

import gc
import hashlib
import re
import sys
import threading
import warnings
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Callable

import numpy as np

from sense import config
from sense.contracts import BaseModel, ChoiceItem, ModelInfo, SenseInput

LETTERS = "ABCDE"
SLOT_MARKER = "<|vision_pad|>"
N_SLOTS = 4

_LOCAL = {  # name -> (label, environment variable that overrides the source)
    "qwen3.5-0.8b": ("Qwen3.5 0.8B (local)", "SENSE_QWEN_0P8B"),
    "qwen3.5-2b": ("Qwen3.5 2B (local)", "SENSE_QWEN_2B"),
}
_registered: dict[str, tuple[ModelInfo, Callable[[], BaseModel]]] = {}
_loaded: dict[str, BaseModel] = {}
_lock = threading.RLock()


def available_models() -> list[ModelInfo]:
    if config.fake():
        return _fake_infos()
    infos = [_local_info(name) for name in _LOCAL if name not in _registered]
    return infos + [info for info, _ in _registered.values()]


def load_model(name: str) -> BaseModel:
    with _lock:
        if name in _loaded:
            return _loaded[name]
        info, loader = _entry(name)
        if not info.available:
            raise RuntimeError(f"{info.label} cannot be loaded. {info.note}".strip())
        if info.open_weights:
            _release_local()
        model = loader()
        _loaded[name] = model
        return model


def register_model(info: ModelInfo, loader: Callable[[], BaseModel]) -> None:
    with _lock:
        _registered[info.name] = (info, loader)
        _loaded.pop(info.name, None)


def _entry(name: str) -> tuple[ModelInfo, Callable[[], BaseModel]]:
    if config.fake():
        infos = {info.name: info for info in _fake_infos()}
        if name not in infos:
            raise ValueError(
                f"Unknown model {name!r}: in fake mode (SENSE_FAKE=1) the only model is 'fake' (open-weight) or "
                "'fake-api' (an API model that can call tools). Use one of those, or unset SENSE_FAKE for real models."
            )
        info = infos[name]
        return info, lambda: FakeModel(info)
    if name in _registered:
        return _registered[name]
    if name in _LOCAL:
        info = _local_info(name)
        path = _local_snapshot(config.model_source(name))
        return info, lambda: QwenModel(info, path)
    known = ", ".join(m.name for m in available_models())
    raise ValueError(f"Unknown model {name!r}. Known models: {known}.")


def _release_local() -> None:
    """Drop every loaded open-weight model and give its memory back."""
    for name in [n for n, m in _loaded.items() if m.info.open_weights]:
        del _loaded[name]
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None and torch.backends.mps.is_available():
        torch.mps.empty_cache()


# ----------------------------------------------------------------------------- finding local weights
def _local_info(name: str) -> ModelInfo:
    label, env = _LOCAL[name]
    source = config.model_source(name)
    found = _local_snapshot(source) is not None
    note = "" if found else (
        f"Weights for {source} were not found on this machine. Set SENSE_HF_CACHE to the Hugging Face cache "
        f"that holds them, or {env} to a local model directory."
    )
    return ModelInfo(
        name=name, label=label, native=["text", "image"], open_weights=True, available=found, note=note, tools=False,
    )


def _has_weights(directory: Path) -> bool:
    return (directory / "config.json").is_file() and any(p.exists() for p in directory.glob("*.safetensors"))


def _local_snapshot(source: str) -> Path | None:
    """A local directory holding the model's config and weights, found without any download."""
    path = Path(source).expanduser()
    if path.is_dir():
        return path if _has_weights(path) else None
    from huggingface_hub.constants import HF_HUB_CACHE

    for cache in [c for c in (config.hf_cache(), HF_HUB_CACHE) if c]:
        repo = Path(cache) / ("models--" + source.replace("/", "--"))
        snapshots = repo / "snapshots"
        if not snapshots.is_dir():
            continue
        candidates = []
        ref = repo / "refs" / "main"
        if ref.is_file():
            candidates.append(snapshots / ref.read_text().strip())
        candidates += sorted(snapshots.iterdir(), key=lambda d: d.stat().st_mtime, reverse=True)
        for snap in candidates:
            if snap.is_dir() and _has_weights(snap):
                return snap
    return None


# ----------------------------------------------------------------------------- prompts
def _input_label(inp: SenseInput, k: int, n: int) -> str:
    name = "Audio clip" if inp.modality == "audio" else f"{inp.modality.capitalize()} input"
    return name if n == 1 else f"{name} {k + 1}"


def _question_text(item: ChoiceItem) -> str:
    if not 1 <= len(item.options) <= len(LETTERS):
        raise ValueError(f"A choice item needs 1 to {len(LETTERS)} options, got {len(item.options)}.")
    lines = "\n".join(f"{LETTERS[k]}. {opt}" for k, opt in enumerate(item.options))
    return f"Question: {item.question}\n{lines}\nAnswer with the letter only."


def _same_length_batches(lengths: list[int], batch: int):
    """Batches of equal token length, so nothing is ever padded."""
    by_len: dict[int, list[int]] = {}
    for i, n in enumerate(lengths):
        by_len.setdefault(n, []).append(i)
    for idx in by_len.values():
        for s in range(0, len(idx), batch):
            yield idx[s : s + batch]


@dataclass
class _Rendered:
    ids: list[int]
    writes: list[tuple[list[int], np.ndarray, int]] = field(default_factory=list)  # (positions, vector, layer)


# ----------------------------------------------------------------------------- Qwen3.5
class QwenModel:
    """Frozen Qwen3.5 through transformers, with a socket: vectors are written into slot tokens after a block."""

    def __init__(self, info: ModelInfo, path: Path | None):
        import torch
        from transformers import AutoModelForImageTextToText, AutoTokenizer

        if path is None:
            raise RuntimeError(f"{info.label} cannot be loaded. {info.note}")
        self.info = info
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        dtype = torch.float16 if self.device == "mps" else torch.float32
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForImageTextToText.from_pretrained(path, dtype=dtype).to(self.device).eval()
        self.layers = self.model.model.language_model.layers
        self.n_layers = len(self.layers)
        self.hidden_size = int(self.model.config.text_config.hidden_size)
        self.slot_id = self._single_token(SLOT_MARKER)
        self.letter_ids = [self._single_token(letter) for letter in LETTERS]
        self._lock = threading.Lock()  # one forward pass at a time: a chat may arrive during an agent run

    def _single_token(self, text: str) -> int:
        ids = self.tok.encode(text, add_special_tokens=False)
        if len(ids) != 1:
            raise RuntimeError(f"{text!r} is not a single token for {self.info.label}; the socket cannot be used.")
        return ids[0]

    def _check_layer(self, layer: int) -> None:
        if not 0 <= layer < self.n_layers:
            raise ValueError(f"Layer {layer} does not exist: {self.info.label} has blocks 0 to {self.n_layers - 1}.")

    def _render(self, inputs: list[SenseInput], body: str) -> _Rendered:
        lines, vectors = [], []
        for k, inp in enumerate(inputs):
            label = _input_label(inp, k, len(inputs))
            if inp.kind == "vector":
                if inp.vector is None or inp.layer is None:
                    raise ValueError(f"The vector input for {inp.source or label} has no vector or no layer.")
                vec = np.asarray(inp.vector, dtype=np.float32).reshape(-1)
                if vec.shape[0] != self.hidden_size:
                    raise ValueError(
                        f"The vector input for {inp.source or label} has {vec.shape[0]} values, but "
                        f"{self.info.label} expects {self.hidden_size}; the bridge was fitted for another model."
                    )
                self._check_layer(inp.layer)
                lines.append(f"{label}: {SLOT_MARKER * N_SLOTS}")
                vectors.append((vec, inp.layer))
            elif inp.kind == "text":
                lines.append(f"{label}: {(inp.text or '').replace(SLOT_MARKER, '')}")
            elif inp.kind == "tool":
                raise ValueError(
                    f"{self.info.label} cannot call tools: tool calling is wired for API models only, not for local "
                    f"models. Give {inp.source or label} to it through a text or vector bridge instead."
                )
            else:
                raise ValueError(f"{self.info.label} takes text and vector inputs here; {inp.kind} inputs are not wired up.")
        user = "\n".join(lines + [body.replace(SLOT_MARKER, "")])
        ids = self.tok.encode(self._chat_template(user), add_special_tokens=False)
        slots = [i for i, t in enumerate(ids) if t == self.slot_id]
        if len(slots) != N_SLOTS * len(vectors):
            raise RuntimeError(f"Expected {N_SLOTS * len(vectors)} slot tokens in the prompt, found {len(slots)}.")
        writes = [(slots[N_SLOTS * k : N_SLOTS * (k + 1)], vec, layer) for k, (vec, layer) in enumerate(vectors)]
        return _Rendered(ids=ids, writes=writes)

    def _chat_template(self, user: str) -> str:
        msgs = [{"role": "user", "content": user}]
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    def _run_with_writes(self, ids, rows: list[_Rendered], fn):
        """Call fn() with forward hooks that overwrite the slot positions of each row after its block. The
        hook only fires on a pass over the whole prompt, so generation steps (one token each) are untouched."""
        import torch

        by_layer: dict[int, tuple[list[int], list[int], list[np.ndarray]]] = {}
        for row, rendered in enumerate(rows):
            for positions, vec, layer in rendered.writes:
                r, c, v = by_layer.setdefault(layer, ([], [], []))
                r += [row] * len(positions)
                c += positions
                v += [vec] * len(positions)
        full_len = ids.shape[1]
        handles = []
        for layer, (r, c, v) in by_layer.items():
            r_t = torch.tensor(r, device=self.device)
            c_t = torch.tensor(c, device=self.device)
            v_t = torch.tensor(np.stack(v), device=self.device)

            def hook(_module, _inputs, output, r_t=r_t, c_t=c_t, v_t=v_t):
                hs = output[0] if isinstance(output, tuple) else output
                if hs.shape[1] == full_len:
                    hs[r_t, c_t] = v_t.to(hs.dtype)
                return output

            handles.append(self.layers[layer].register_forward_hook(hook))
        try:
            return fn()
        finally:
            for handle in handles:
                handle.remove()

    def text_states(self, texts: list[str], layer: int) -> np.ndarray:
        import torch

        self._check_layer(layer)
        enc = [self.tok.encode(t, add_special_tokens=False) for t in texts]
        if any(len(e) == 0 for e in enc):
            raise ValueError("text_states was given an empty text.")
        out = np.zeros((len(texts), self.hidden_size), dtype=np.float32)
        with self._lock, torch.no_grad():
            for idx in _same_length_batches([len(e) for e in enc], 32):
                ids = torch.tensor([enc[i] for i in idx], device=self.device)
                hs = self.model.model(input_ids=ids, output_hidden_states=True).hidden_states[layer + 1]
                out[idx] = hs.float().mean(dim=1).cpu().numpy()
        return out

    def choose(self, items: list[ChoiceItem]) -> list[int]:
        import torch

        rendered = [self._render(item.inputs, _question_text(item)) for item in items]
        preds = [0] * len(items)
        with self._lock, torch.no_grad():
            for idx in _same_length_batches([len(r.ids) for r in rendered], 16):
                ids = torch.tensor([rendered[i].ids for i in idx], device=self.device)

                def forward(ids=ids):
                    last = self.model.model(input_ids=ids).last_hidden_state[:, -1]
                    return self.model.lm_head(last)[:, self.letter_ids]

                logits = self._run_with_writes(ids, [rendered[i] for i in idx], forward).float().cpu().numpy()
                for row, i in enumerate(idx):
                    preds[i] = int(logits[row, : len(items[i].options)].argmax())
        return preds

    def chat(self, message: str, inputs: list[SenseInput], max_new_tokens: int = 96) -> str:
        import torch

        rendered = self._render(inputs, message)
        ids = torch.tensor([rendered.ids], device=self.device)

        def generate():
            return self.model.generate(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        with self._lock, torch.no_grad():
            out = self._run_with_writes(ids, [rendered], generate)
        return self.tok.decode(out[0, ids.shape[1] :], skip_special_tokens=True).strip()


# ----------------------------------------------------------------------------- fake
def _fake_info() -> ModelInfo:
    return ModelInfo(
        name="fake",
        label="Fake model (no weights)",
        native=["text", "image"],
        open_weights=True,
        available=True,
        note="Deterministic stand-in for tests and interface work.",
    )


def _fake_api_info() -> ModelInfo:
    return ModelInfo(
        name="fake-api",
        label="Fake API model (no weights, tools)",
        native=["text", "image"],
        open_weights=False,
        available=True,
        note="Deterministic stand-in for an API model that can call tools, for tests and interface work.",
        tools=True,
    )


def _fake_infos() -> list[ModelInfo]:
    return [_fake_info(), _fake_api_info()]


_BEST_MATCH = re.compile(r"Best match: (.+)\.\s*$")


def _best_match(result: str) -> str | None:
    """The X of a tool result ending 'Best match: X.', or None."""
    found = _BEST_MATCH.search(result)
    return found.group(1).strip() if found else None


def _first_concept(text: str) -> str | None:
    """The fake concept named earliest in `text` (a tool lists its nearest match first), or None."""
    from sense.specialists import FAKE_CONCEPTS

    found = [(m.start(), c) for c in FAKE_CONCEPTS if (m := re.search(rf"\b{re.escape(c)}\b", text.lower()))]
    return min(found)[1] if found else None


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _hash_uniform(*parts: object) -> float:
    digest = hashlib.md5("\x1f".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "little") / 2**64


class FakeModel:
    """Deterministic open-weight stand-in. A text's state is the mean of fixed random word vectors (scaled per
    block). For a vector input it picks the option whose state is most similar, but only reads slots written
    after blocks 0 to READ_DEPTH, like Qwen3.5; for a text input it picks the option the text names; for a tool
    input it calls the tool with the options as candidates and picks the option the tool names in "Best match:
    X.". Without evidence it picks by a hash of the question, so it scores near chance. In chat it calls a tool
    without candidates and says what the tool returned.

    It takes every kind of input whatever its ModelInfo says (it is a test double): "fake" (open-weight, no
    tools) still answers tool inputs, and "fake-api" (not open-weight) still reads vector inputs."""

    READ_DEPTH = 3

    def __init__(self, info: ModelInfo):
        self.info = info
        self.hidden_size = 96
        self.n_layers = 8

    @staticmethod
    @lru_cache(maxsize=4096)
    def _word_vec(word: str, size: int) -> np.ndarray:
        seed = int.from_bytes(hashlib.md5(word.encode()).digest()[:8], "little")
        return np.random.default_rng(seed).normal(size=size) / np.sqrt(size)

    def _state(self, text: str, layer: int) -> np.ndarray:
        words = _words(text)
        if not words:
            return np.zeros(self.hidden_size)
        return np.mean([self._word_vec(w, self.hidden_size) for w in words], axis=0) * (1 + 0.25 * layer)

    def text_states(self, texts: list[str], layer: int) -> np.ndarray:
        if not 0 <= layer < self.n_layers:
            raise ValueError(f"Layer {layer} does not exist: {self.info.label} has blocks 0 to {self.n_layers - 1}.")
        return np.array([self._state(t, layer) for t in texts], dtype=np.float32).reshape(len(texts), self.hidden_size)

    def _evidence(self, inp: SenseInput, options: list[str]) -> np.ndarray:
        """How strongly the input supports each option, in [-1, 1]."""
        if inp.kind == "vector" and inp.vector is not None and inp.layer is not None:
            if not 0 <= inp.layer <= self.READ_DEPTH:
                return np.zeros(len(options))  # written too deep: the model does not read it
            v = np.asarray(inp.vector, dtype=np.float64).reshape(-1)
            if v.shape[0] != self.hidden_size:
                raise ValueError(f"The vector input has {v.shape[0]} values; {self.info.label} expects {self.hidden_size}.")
            states = np.array([self._state(o, inp.layer) for o in options])
            norms = np.linalg.norm(states, axis=1) * max(np.linalg.norm(v), 1e-12)
            return states @ v / np.maximum(norms, 1e-12)
        if inp.kind == "text" and inp.text:
            said = set(_words(inp.text))
            return np.array([np.mean([w in said for w in _words(o)]) if _words(o) else 0.0 for o in options])
        if inp.kind == "tool" and inp.tool is not None:
            best = _best_match(inp.tool.call({"candidates": list(options)}))
            return np.array([1.0 if o == best else 0.0 for o in options])
        return np.zeros(len(options))

    def choose(self, items: list[ChoiceItem]) -> list[int]:
        preds = []
        for item in items:
            scores = np.array([_hash_uniform(item.question, *item.options, k) for k in range(len(item.options))])
            for inp in item.inputs:
                scores += 4.0 * self._evidence(inp, item.options)
            preds.append(int(scores.argmax()))
        return preds

    def _heard(self, inp: SenseInput) -> str | None:
        from sense.specialists import FAKE_CONCEPTS, fake_concepts_in

        if inp.kind == "text" and inp.text:
            hits = fake_concepts_in(inp.text)
            return FAKE_CONCEPTS[hits[0]] if hits else None
        if inp.kind == "vector" and inp.vector is not None:
            sims = self._evidence(inp, FAKE_CONCEPTS)
            return FAKE_CONCEPTS[int(sims.argmax())] if sims.max() > 0.5 else None
        return None

    def chat(self, message: str, inputs: list[SenseInput], max_new_tokens: int = 96) -> str:
        if not inputs:
            reply = "No audio reached me, so I can only answer from your message."
        else:
            parts = []
            for k, inp in enumerate(inputs):
                name = "The audio clip" if len(inputs) == 1 else f"Audio clip {k + 1}"
                if inp.kind == "tool" and inp.tool is not None:
                    result = inp.tool.call({})
                    heard = _first_concept(result)
                    lead = f"{name} sounds like {heard}." if heard else f"I cannot make out what is in {name.lower()}."
                    parts.append(f"{lead} I called {inp.tool.name}, which returned: {result}")
                    continue
                heard = self._heard(inp)
                parts.append(f"{name} sounds like {heard}." if heard else f"I cannot make out what is in {name.lower()}.")
            reply = " ".join(parts)
        return " ".join(f"{reply} (Fake model reply to: {message.strip()})".split()[:max_new_tokens])


# API models, if a developer has added them. Imported last so that it can call register_model.
try:
    import sense.models_api  # noqa: F401,E402
except ModuleNotFoundError as err:
    if err.name != "sense.models_api":
        warnings.warn(f"sense.models_api was not loaded: {err}", stacklevel=1)
except ImportError as err:
    warnings.warn(f"sense.models_api was not loaded: {err}", stacklevel=1)
