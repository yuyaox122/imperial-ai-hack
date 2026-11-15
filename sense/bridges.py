"""Bridges: ways of getting a modality into a base model.

Public surface (do not rename):
  fit_vector_bridge(model, dataset, idx, layer=None) -> Bridge
      Closed-form ridge map from specialist embeddings to the model's own text representation of each label,
      with a norm match (method and reference code: socket_pilot/run_pilot.py, fit_plug). Open-weight models
      only. `idx` are the examples to fit on. If `layer` is None, pick it on dataset.dev_idx from blocks 0-4.
  make_text_bridge(model, dataset=None) -> Bridge
      Describes the input in words. With a dataset: the specialist's best-matching label
      ('audio: sounds like "rock"'). Without: the nearest caption from datasets.caption_bank().
  make_tool_bridge(model, dataset=None) -> Bridge
      Lets the model call the specialist as a tool named "listen" (a SenseInput of kind "tool" holding a
      contracts.ToolHandle). With candidates the tool returns the specialist's similarity of the clip to each;
      without, the 3 nearest labels of the dataset, or the 3 nearest captions of datasets.caption_bank().
      For models that can call tools (ModelInfo.tools); it does not check that itself, see availability.
  build(kind, model, dataset, scope) -> Bridge
      Build one bridge kind the way an agent run does: "vector" fits on dataset.fit_idx (ValueError when
      model.info.open_weights is False); "text" and "tool" use the dataset's labels for scope "closed" and the
      caption bank for scope "open". Other kinds: ValueError.
  load_bridge(directory) -> Bridge
      Reload a bridge saved with Bridge.save. A bridge.json with "adapter": true is an agent-written adapter and
      loads through sense.adapters.load_adapter_bridge (whatever its kind), so sense.registry.get_bridges returns
      an AdapterBridge for an adapter sense.
  describe(bridge) -> dict
      Facts for the agent's trace, as plain JSON values: {"name", "kind", "modality", "model", "specialist",
      "fit_seconds", "fit_seconds_by_layer", "layer", "dev_accuracy", "n_fit", "source", "n_phrases"}, with None
      where a bridge has no such fact. Layer-keyed dicts have string keys, so they survive a JSON round trip.
  availability(model_info, modality) -> list[dict]
      [{"kind", "available", "why"}] for native, text, tool, image and vector, in that order, without loading
      anything.

A vector bridge records fit_seconds (the whole fit, including the layer sweep and its dev scoring) and
fit_seconds_by_layer (the closed-form fit of each candidate block alone, keyed by block). Both keep 4 decimals:
at 2, the fast blocks of fake mode each rounded up to 0.01 s and added up to more than the whole fit.

Every bridge implements contracts.Bridge: from_embedding, prepare, save. `name` is "vector:<dataset>",
"text:<dataset>", "text:captions", "tool:<dataset>" or "tool:captions".

The listen tool. Its one optional argument, "candidates", is a list of at most 10 short descriptions or labels.
Each candidate is phrased as "<c> music" when the dataset's question mentions music and "the sound of <c>"
otherwise (always the latter for the caption bank), embedded as text by the specialist and compared with the
clip by cosine: 'Similarity of the clip to each candidate (CLAP): rock 0.41, jazz 0.22. Best match: rock.'
Without candidates (or with an empty list) it returns 'Nearest labels (CLAP): rock 0.41, metal 0.30, blues 0.12.'
or 'Nearest captions (CLAP): "...", "...", "...".' Bad arguments get a one-line error text the model can act
on. Each call is logged in ToolHandle.calls; ToolHandle.key is the sha256 of the clip's float32 embedding bytes
plus the bridge name.

Also used by sense.verify: five_options, the seeded five-option question builder.
On disk a bridge is bridge.json (settings) plus bridge.npz (arrays).
"""

from __future__ import annotations

import copy
import hashlib
import json
import threading
import time
from pathlib import Path

import numpy as np

from sense.contracts import BaseModel, Bridge, ChoiceItem, LabelledSet, ModelInfo, OpenModel, SenseInput, ToolHandle

LAMBDAS = [1e-2, 1e-1, 1.0, 10.0, 1e2, 1e3, 1e4, 1e5, 1e6]
LAYER_CANDIDATES = range(5)  # blocks 0-4: Qwen3.5 reads the slots up to about block 4
DEV_ITEMS = 200  # 100 was too noisy to separate blocks 0-2


# ----------------------------------------------------------------------------- shared helpers
def five_options(y: np.ndarray, n_labels: int, rng: np.random.Generator) -> tuple[list[list[int]], np.ndarray]:
    """For each true label in y: the true label and up to four others in a seeded random order, and the answer index."""
    k = min(5, n_labels)
    options, answers = [], []
    for t in y:
        wrong = rng.choice([c for c in range(n_labels) if c != t], k - 1, replace=False)
        opts = [int(t), *map(int, wrong)]
        rng.shuffle(opts)
        options.append(opts)
        answers.append(opts.index(int(t)))
    return options, np.array(answers, dtype=np.int64)


def _unit(a: np.ndarray) -> np.ndarray:
    return a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12)


def _phrasing(question: str) -> str:
    """"music" when the question is about music, else "sound": how a label is worded as a phrase."""
    return "music" if "music" in question.lower() else "sound"


def _phrased(names: list[str], style: str) -> list[str]:
    """"<n> music" for style "music", else "the sound of <n>"."""
    if style == "music":
        return [f"{n} music" for n in names]
    return [f"the sound of {n}" for n in names]


def _label_phrases(dataset: LabelledSet) -> list[str]:
    """A second wording of each label, averaged with the bare label to form the target (as in socket_pilot)."""
    return _phrased(list(dataset.labels), _phrasing(dataset.question))


def _choose_lambda(X: np.ndarray, T: np.ndarray, k: int = 5, seed: int = 0) -> float:
    """Ridge penalty by k-fold cross-validated R^2 on the fitting examples only."""
    order = np.random.default_rng(seed).permutation(len(X))
    folds = np.array_split(order, min(k, len(X)))
    eye = np.eye(X.shape[1])
    best, best_r2 = LAMBDAS[0], -np.inf
    for lam in LAMBDAS:
        ss_res = ss_tot = 0.0
        for f in folds:
            tr = np.setdiff1d(order, f)
            mu = T[tr].mean(0)
            W = np.linalg.solve(X[tr].T @ X[tr] + lam * eye, X[tr].T @ (T[tr] - mu))
            ss_res += ((T[f] - (mu + X[f] @ W)) ** 2).sum()
            ss_tot += ((T[f] - mu) ** 2).sum()
        r2 = 1 - ss_res / max(ss_tot, 1e-12)
        if r2 > best_r2:
            best, best_r2 = lam, r2
    return best


# ----------------------------------------------------------------------------- vector bridge
class VectorBridge:
    """Writes a mapped specialist embedding into the base model's slot tokens after block `layer`."""

    kind = "vector"

    def __init__(
        self,
        *,
        modality: str,
        model_name: str,
        dataset: str,
        layer: int,
        lam: float,
        labels: list[str],
        mu_x: np.ndarray,
        sd_x: np.ndarray,
        W: np.ndarray,
        mu_t: np.ndarray,
        rho: float,
        label_targets: np.ndarray,
        dev_accuracy: dict[int, float] | None = None,
        n_fit: int = 0,
        fit_seconds: float = 0.0,
        fit_seconds_by_layer: dict[int, float] | None = None,
    ):
        self.modality = modality
        self.model_name = model_name
        self.dataset = dataset
        self.name = f"vector:{dataset}"
        self.layer = layer
        self.lam = lam
        self.labels = labels  # the labels the map was fitted on, for `detail`
        self.mu_x, self.sd_x, self.W, self.mu_t, self.rho = mu_x, sd_x, W, mu_t, rho
        self.label_targets = label_targets  # [len(labels), hidden]
        self.dev_accuracy = dev_accuracy or {}
        self.n_fit = n_fit
        self.fit_seconds = fit_seconds
        self.fit_seconds_by_layer = fit_seconds_by_layer or {}  # closed-form fit of each candidate block, no scoring

    def vectors(self, emb: np.ndarray) -> np.ndarray:
        """[n, dim] specialist embeddings -> [n, hidden] vectors for the slots: ridge map, then norm match."""
        emb = np.atleast_2d(np.asarray(emb, dtype=np.float64))
        if emb.shape[1] != self.mu_x.shape[0]:
            raise ValueError(
                f"The embedding has {emb.shape[1]} values but {self.name} was fitted on {self.mu_x.shape[0]}; "
                "it was fitted with a different specialist (or in fake mode). Fit the bridge again."
            )
        dev = ((emb - self.mu_x) / self.sd_x) @ self.W
        dev = _unit(dev)
        return (self.mu_t + self.rho * dev).astype(np.float32)

    def from_embedding(self, emb: np.ndarray, source: str = "") -> SenseInput:
        vec = self.vectors(emb)[0]
        sims = _unit(self.label_targets.astype(np.float64)) @ _unit(vec.astype(np.float64))
        best = int(sims.argmax())
        return SenseInput(
            kind="vector",
            modality=self.modality,
            source=source,
            vector=vec,
            layer=self.layer,
            detail=(
                f'closest fitted label: "{self.labels[best]}" (cosine {sims[best]:.2f}); '
                f"this sense covers its {len(self.labels)} fitted categories only"
            ),
        )

    def prepare(self, path: Path) -> SenseInput:
        from sense.specialists import get_specialist

        path = Path(path)
        return self.from_embedding(get_specialist(self.modality).embed_files([path])[0], source=path.name)

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        meta = {
            "kind": self.kind,
            "name": self.name,
            "modality": self.modality,
            "model_name": self.model_name,
            "dataset": self.dataset,
            "layer": self.layer,
            "lam": self.lam,
            "labels": self.labels,
            "dev_accuracy": {str(k): v for k, v in self.dev_accuracy.items()},
            "n_fit": self.n_fit,
            "fit_seconds": self.fit_seconds,
            "fit_seconds_by_layer": {str(k): v for k, v in self.fit_seconds_by_layer.items()},
        }
        np.savez(
            directory / "bridge.npz",
            mu_x=self.mu_x, sd_x=self.sd_x, W=self.W, mu_t=self.mu_t, rho=np.array(self.rho), label_targets=self.label_targets,
        )
        (directory / "bridge.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, directory: Path, meta: dict) -> VectorBridge:
        arrays = np.load(Path(directory) / "bridge.npz", allow_pickle=False)
        return cls(
            modality=meta["modality"],
            model_name=meta["model_name"],
            dataset=meta["dataset"],
            layer=int(meta["layer"]),
            lam=float(meta["lam"]),
            labels=list(meta["labels"]),
            mu_x=arrays["mu_x"],
            sd_x=arrays["sd_x"],
            W=arrays["W"],
            mu_t=arrays["mu_t"],
            rho=float(arrays["rho"]),
            label_targets=arrays["label_targets"],
            dev_accuracy={int(k): v for k, v in meta.get("dev_accuracy", {}).items()},
            n_fit=int(meta.get("n_fit", 0)),
            fit_seconds=float(meta.get("fit_seconds", 0.0)),
            fit_seconds_by_layer={int(k): float(v) for k, v in meta.get("fit_seconds_by_layer", {}).items()},
        )


def _label_targets(model: OpenModel, dataset: LabelledSet, layer: int) -> np.ndarray:
    """The model's own representation of each label after block `layer`: [n_labels, hidden]."""
    bare = model.text_states(list(dataset.labels), layer)
    phrased = model.text_states(_label_phrases(dataset), layer)
    return ((bare.astype(np.float64) + phrased.astype(np.float64)) / 2)


def _fit_at(model: OpenModel, dataset: LabelledSet, idx: np.ndarray, layer: int) -> VectorBridge:
    start = time.time()
    emb = dataset.emb.astype(np.float64)
    mu_x, sd_x = emb[idx].mean(0), emb[idx].std(0) + 1e-8
    X = (emb[idx] - mu_x) / sd_x
    targets = _label_targets(model, dataset, layer)
    T = targets[dataset.y[idx]]
    lam = _choose_lambda(X, T)
    mu_t = T.mean(0)
    W = np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ (T - mu_t))
    rho = float(np.linalg.norm(T - mu_t, axis=1).mean())
    seen = sorted({int(c) for c in dataset.y[idx]})
    seconds = round(time.time() - start, 4)  # the closed-form fit of this block only, without dev scoring
    return VectorBridge(
        modality=dataset.modality,
        model_name=model.info.name,
        dataset=dataset.name,
        layer=layer,
        lam=lam,
        labels=[dataset.labels[c] for c in seen],
        mu_x=mu_x,
        sd_x=sd_x,
        W=W,
        mu_t=mu_t,
        rho=rho,
        label_targets=targets[seen].astype(np.float32),
        n_fit=len(idx),
        fit_seconds=seconds,
        fit_seconds_by_layer={layer: seconds},
    )


def _dev_accuracy(model: BaseModel, bridge: VectorBridge, dataset: LabelledSet) -> float:
    rng = np.random.default_rng(0)
    dev = dataset.dev_idx
    if len(dev) > DEV_ITEMS:
        dev = np.sort(rng.choice(dev, DEV_ITEMS, replace=False))
    options, answers = five_options(dataset.y[dev], len(dataset.labels), rng)
    vecs = bridge.vectors(dataset.emb[dev])
    items = [
        ChoiceItem(
            question=dataset.question,
            options=[dataset.labels[c] for c in opts],
            inputs=[SenseInput(kind="vector", modality=dataset.modality, vector=v, layer=bridge.layer)],
        )
        for opts, v in zip(options, vecs)
    ]
    return float((np.array(model.choose(items)) == answers).mean())


def fit_vector_bridge(model: BaseModel, dataset: LabelledSet, idx: np.ndarray, layer: int | None = None) -> Bridge:
    if not isinstance(model, OpenModel):
        raise ValueError(
            f"{model.info.label} is not an open-weight model, so it cannot take a vector bridge. Use the text bridge."
        )
    idx = np.asarray(idx, dtype=np.int64)
    if len(idx) < 5 or len(np.unique(dataset.y[idx])) < 2:
        raise ValueError(f"A vector bridge needs at least 5 examples of at least 2 labels to fit on; got {len(idx)}.")
    if layer is not None:
        return _fit_at(model, dataset, idx, layer)
    if len(dataset.dev_idx) == 0:
        raise ValueError(f"Dataset {dataset.name} has no dev examples to choose the layer on; pass a layer.")
    start = time.time()
    best, scores = None, {}
    fit_times: dict[int, float] = {}
    for candidate in [L for L in LAYER_CANDIDATES if L < model.n_layers]:
        bridge = _fit_at(model, dataset, idx, candidate)
        fit_times[candidate] = bridge.fit_seconds
        scores[candidate] = round(_dev_accuracy(model, bridge, dataset), 4)
        if best is None or scores[candidate] > scores[best.layer]:  # ties keep the earlier block
            best = bridge
    best.dev_accuracy = scores
    best.fit_seconds_by_layer = fit_times
    best.fit_seconds = round(time.time() - start, 4)
    return best


# ----------------------------------------------------------------------------- text bridge
class TextBridge:
    """Puts words into the prompt: the best-matching label of a dataset, or the nearest caption of the caption bank."""

    kind = "text"

    def __init__(self, *, modality: str, model_name: str, source: str, phrases: list[str], phrase_emb: np.ndarray):
        self.modality = modality
        self.model_name = model_name
        self.source = source  # a dataset name, or "captions"
        self.name = f"text:{source}"
        self.phrases = phrases
        self._phrase_unit = _unit(np.asarray(phrase_emb, dtype=np.float32))

    def from_embedding(self, emb: np.ndarray, source: str = "") -> SenseInput:
        emb = np.asarray(emb, dtype=np.float32).reshape(-1)
        if emb.shape[0] != self._phrase_unit.shape[1]:
            raise ValueError(
                f"The embedding has {emb.shape[0]} values but {self.name} compares against {self._phrase_unit.shape[1]}; "
                "it was built with a different specialist (or in fake mode). Build the bridge again."
            )
        sims = self._phrase_unit @ _unit(emb)
        best = int(sims.argmax())
        phrase = self.phrases[best]
        if self.source == "captions":
            text, detail = f'"{phrase}"', f'nearest caption: "{phrase}" (cosine {sims[best]:.2f})'
        else:
            text, detail = f'sounds like "{phrase}"', f'best-matching label: "{phrase}" (cosine {sims[best]:.2f})'
        return SenseInput(kind="text", modality=self.modality, source=source, text=text, detail=detail)

    def prepare(self, path: Path) -> SenseInput:
        from sense.specialists import get_specialist

        path = Path(path)
        return self.from_embedding(get_specialist(self.modality).embed_files([path])[0], source=path.name)

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        meta = {"kind": self.kind, "name": self.name, "modality": self.modality, "model_name": self.model_name, "source": self.source}
        if self.source != "captions":  # small: keep the labels with the bridge; the caption bank is reloaded instead
            meta["phrases"] = self.phrases
            np.savez(directory / "bridge.npz", phrase_emb=self._phrase_unit)
        (directory / "bridge.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, directory: Path, meta: dict) -> TextBridge:
        if meta["source"] == "captions":
            from sense.datasets import caption_bank

            emb, phrases = caption_bank()
        else:
            emb, phrases = np.load(Path(directory) / "bridge.npz", allow_pickle=False)["phrase_emb"], list(meta["phrases"])
        return cls(modality=meta["modality"], model_name=meta["model_name"], source=meta["source"], phrases=phrases, phrase_emb=emb)


def make_text_bridge(model: BaseModel, dataset: LabelledSet | None = None) -> Bridge:
    if dataset is not None:
        return TextBridge(
            modality=dataset.modality,
            model_name=model.info.name,
            source=dataset.name,
            phrases=list(dataset.labels),
            phrase_emb=dataset.label_text_emb,
        )
    from sense.datasets import caption_bank

    emb, texts = caption_bank()
    return TextBridge(modality="audio", model_name=model.info.name, source="captions", phrases=texts, phrase_emb=emb)


# ----------------------------------------------------------------------------- tool bridge
TOOL_NAME = "listen"
MAX_CANDIDATES = 10
MAX_CANDIDATE_CHARS = 120  # "a short description or label"; CLAP reads little more than this anyway
NEAREST = 3  # phrases the tool returns when it is called without candidates
_TEXT_CACHE_SIZE = 4096  # candidate phrases kept embedded per bridge; verification asks about the same labels often
_EXAMPLE = 'such as {"candidates": ["rock", "jazz"]}'


class ToolBridge:
    """Offers the specialist to the model as the listen tool, one ToolHandle per clip. The model asks how well the
    clip matches candidate descriptions it chooses, or what the clip is nearest to, and reads the answer as text."""

    kind = "tool"

    def __init__(
        self,
        *,
        modality: str,
        model_name: str,
        source: str,
        phrases: list[str],
        phrase_emb: np.ndarray,
        question_style: str = "sound",
    ):
        if question_style not in ("music", "sound"):
            raise ValueError(f"question_style must be 'music' or 'sound', not {question_style!r}.")
        self.modality = modality
        self.model_name = model_name
        self.source = source  # a dataset name, or "captions"
        self.name = f"tool:{source}"
        self.phrases = phrases  # the dataset's labels, or the caption bank: what the tool names without candidates
        self.question_style = question_style  # candidates are worded "<c> music" for "music", else "the sound of <c>"
        self._phrase_unit = _unit(np.asarray(phrase_emb, dtype=np.float32))
        self._text_cache: dict[str, np.ndarray] = {}  # phrase -> unit text embedding
        self._lock = threading.Lock()  # an API model may call the tools of several clips at once

    # ------------------------------------------------------------------------- what the model is offered
    def _specialist_label(self) -> str:
        """The specialist's display name without its parenthesis: "CLAP", or "Fake CLAP" in fake mode."""
        from sense import specialists

        try:
            return specialists.describe(self.modality)["label"].split(" (")[0]
        except ValueError:  # no specialist for this modality
            return "the specialist"

    def tool_description(self) -> str:
        nearest = "captions" if self.source == "captions" else f"{self.source} labels"
        return (
            f"Listen to an audio clip with {self._specialist_label()}, an audio-text model. With candidates (up to "
            f"{MAX_CANDIDATES} short descriptions or labels) it returns how similar the clip is to each, and the best "
            f"match; without candidates it returns the {NEAREST} nearest {nearest}."
        )

    @staticmethod
    def tool_parameters() -> dict:
        return {
            "type": "object",
            "properties": {
                "candidates": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": MAX_CANDIDATES,
                    "description": (
                        f"Optional: up to {MAX_CANDIDATES} short descriptions or labels to compare the clip with, "
                        'such as ["rock", "jazz"]. Leave it out to hear what the clip is nearest to.'
                    ),
                },
            },
        }

    # ------------------------------------------------------------------------- Bridge
    def from_embedding(self, emb: np.ndarray, source: str = "") -> SenseInput:
        emb = np.asarray(emb, dtype=np.float32).reshape(-1)
        if emb.shape[0] != self._phrase_unit.shape[1]:
            raise ValueError(
                f"The embedding has {emb.shape[0]} values but {self.name} compares against {self._phrase_unit.shape[1]}; "
                "it was built with a different specialist (or in fake mode). Build the bridge again."
            )
        clip = _unit(emb)

        def call(arguments: dict) -> str:
            result = self._answer(clip, arguments)
            handle.calls.append({"arguments": copy.deepcopy(arguments), "result": result})
            return result

        handle = ToolHandle(
            name=TOOL_NAME,
            description=self.tool_description(),
            parameters=self.tool_parameters(),
            call=call,
            key=hashlib.sha256(emb.tobytes() + self.name.encode("utf-8")).hexdigest(),
        )
        return SenseInput(
            kind="tool",
            modality=self.modality,
            source=source,
            detail=f"the model can call the {TOOL_NAME} tool to hear this clip",
            tool=handle,
        )

    def prepare(self, path: Path) -> SenseInput:
        from sense.specialists import get_specialist

        path = Path(path)
        return self.from_embedding(get_specialist(self.modality).embed_files([path])[0], source=path.name)

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        meta = {
            "kind": self.kind,
            "name": self.name,
            "modality": self.modality,
            "model_name": self.model_name,
            "source": self.source,
            "question_style": self.question_style,
        }
        if self.source != "captions":  # small: keep the labels with the bridge; the caption bank is reloaded instead
            meta["phrases"] = self.phrases
            np.savez(directory / "bridge.npz", phrase_emb=self._phrase_unit)
        (directory / "bridge.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, directory: Path, meta: dict) -> ToolBridge:
        if meta["source"] == "captions":
            from sense.datasets import caption_bank

            emb, phrases = caption_bank()
        else:
            emb, phrases = np.load(Path(directory) / "bridge.npz", allow_pickle=False)["phrase_emb"], list(meta["phrases"])
        return cls(
            modality=meta["modality"],
            model_name=meta["model_name"],
            source=meta["source"],
            phrases=phrases,
            phrase_emb=emb,
            question_style=meta.get("question_style", "sound"),
        )

    # ------------------------------------------------------------------------- the tool itself
    def _answer(self, clip: np.ndarray, arguments: dict) -> str:
        """The text the model sees for one call. Bad arguments get an error text, never an exception."""
        if not isinstance(arguments, dict):
            return f"{TOOL_NAME} error: the arguments must be a JSON object, {_EXAMPLE}."
        candidates = arguments.get("candidates")
        if candidates is None or (isinstance(candidates, list) and not candidates):
            return self._nearest(clip)
        problem = _candidates_problem(candidates)
        if problem:
            return f"{TOOL_NAME} error: {problem}"
        names = [c.strip() for c in candidates]
        sims = self._text_units(_phrased(names, self.question_style), len(clip)) @ clip
        order = sorted(range(len(names)), key=lambda k: -float(sims[k]))  # stable: ties keep the order given
        listed = ", ".join(f"{names[k]} {_sim(sims[k])}" for k in order)
        return (
            f"Similarity of the clip to each candidate ({self._specialist_label()}): {listed}. "
            f"Best match: {names[order[0]]}."
        )

    def _nearest(self, clip: np.ndarray) -> str:
        sims = self._phrase_unit @ clip
        top = [int(k) for k in np.argsort(-sims, kind="stable")[:NEAREST]]
        if self.source == "captions":
            listed = ", ".join(f'"{self.phrases[k]}"' for k in top)
            return f"Nearest captions ({self._specialist_label()}): {listed}."
        listed = ", ".join(f"{self.phrases[k]} {_sim(sims[k])}" for k in top)
        return f"Nearest labels ({self._specialist_label()}): {listed}."

    def _text_units(self, texts: list[str], dim: int) -> np.ndarray:
        """Unit text embeddings of `texts` by the specialist, [len(texts), dim], remembered between calls."""
        from sense.specialists import get_specialist

        with self._lock:
            found = {t: self._text_cache[t] for t in texts if t in self._text_cache}
        missing = [t for t in dict.fromkeys(texts) if t not in found]
        if missing:
            emb = np.asarray(get_specialist(self.modality).embed_texts(missing), dtype=np.float32)
            if emb.ndim != 2 or emb.shape[1] != dim:
                raise ValueError(
                    f"The specialist's text embeddings have {emb.shape[-1]} values but the clips of {self.name} have "
                    f"{dim}; the bridge was built with a different specialist (or in fake mode). Build it again."
                )
            fresh = dict(zip(missing, _unit(emb)))
            found.update(fresh)
            with self._lock:
                if len(self._text_cache) + len(fresh) > _TEXT_CACHE_SIZE:
                    self._text_cache.clear()
                self._text_cache.update(fresh)
        return np.stack([found[t] for t in texts])


def _sim(x: float) -> str:
    """A cosine with 2 decimals, never "-0.00"."""
    text = f"{float(x):.2f}"
    return "0.00" if text == "-0.00" else text


def _candidates_problem(candidates: object) -> str | None:
    """Why `candidates` cannot be used, as a sentence for the model, or None when it can."""
    if not isinstance(candidates, list):
        return f"candidates must be a list of short descriptions or labels, {_EXAMPLE}."
    if len(candidates) > MAX_CANDIDATES:
        return f"{len(candidates)} candidates were given, but at most {MAX_CANDIDATES} fit in one call; ask about fewer."
    for c in candidates:
        if not isinstance(c, str):
            return f"every candidate must be a string, but {repr(c)[:40]} is not; send a list {_EXAMPLE}."
        if not c.strip():
            return f"a candidate is empty; give each candidate a few words, {_EXAMPLE}."
        if len(c) > MAX_CANDIDATE_CHARS:
            return f"each candidate must be a short description or label of at most {MAX_CANDIDATE_CHARS} characters."
    return None


def make_tool_bridge(model: BaseModel, dataset: LabelledSet | None = None) -> Bridge:
    if dataset is not None:
        return ToolBridge(
            modality=dataset.modality,
            model_name=model.info.name,
            source=dataset.name,
            phrases=list(dataset.labels),
            phrase_emb=dataset.label_text_emb,
            question_style=_phrasing(dataset.question),
        )
    from sense.datasets import caption_bank

    emb, texts = caption_bank()
    return ToolBridge(modality="audio", model_name=model.info.name, source="captions", phrases=texts, phrase_emb=emb)


# ----------------------------------------------------------------------------- building by kind
_BUILDABLE = ("vector", "text", "tool")


def build(kind: str, model: BaseModel, dataset: LabelledSet, scope: str) -> Bridge:
    """Build one bridge of `kind` for `model`, as an agent run does: a vector bridge is fitted on
    dataset.fit_idx; a text or tool bridge uses the dataset's labels for scope "closed" and the caption bank
    for scope "open", so that it can describe any sound, not only the labelled ones. A vector bridge is refused
    when model.info.open_weights is False, as availability() says, even for a test double such as the fake-api
    FakeModel that could take vector inputs."""
    if kind == "vector":
        if not model.info.open_weights:
            raise ValueError(
                f"{model.info.label} is not an open-weight model, so it cannot take a vector bridge. "
                "Use the text bridge."
            )
        return fit_vector_bridge(model, dataset, dataset.fit_idx)
    if kind == "text":
        return make_text_bridge(model, dataset if scope == "closed" else None)
    if kind == "tool":
        return make_tool_bridge(model, dataset if scope == "closed" else None)
    raise ValueError(
        f"Cannot build a {kind!r} bridge: the kinds that can be built are {', '.join(_BUILDABLE)}. "
        "Ask for one of those."
    )


# ----------------------------------------------------------------------------- loading
def load_bridge(directory: Path) -> Bridge:
    meta_path = Path(directory) / "bridge.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"{meta_path} is missing: {directory} does not hold a saved bridge.")
    meta = json.loads(meta_path.read_text())
    if meta.get("adapter"):  # an agent-written adapter (kind "text" or "image"); imported here, not at the top
        from sense.adapters import load_adapter_bridge

        return load_adapter_bridge(directory)
    if meta.get("kind") == "vector":
        return VectorBridge.load(directory, meta)
    if meta.get("kind") == "text":
        return TextBridge.load(directory, meta)
    if meta.get("kind") == "tool":
        return ToolBridge.load(directory, meta)
    raise ValueError(f"{meta_path} names an unknown bridge kind {meta.get('kind')!r}.")


# ----------------------------------------------------------------------------- facts for the agent's trace
def _str_keys(scores: dict | None) -> dict[str, float] | None:
    """{2: 0.85} -> {"2": 0.85}, so that the dict comes back unchanged from a JSON round trip."""
    return None if scores is None else {str(k): float(v) for k, v in scores.items()}


def describe(bridge: Bridge) -> dict:
    """What the trace should say about a built bridge, as plain JSON values; None where the bridge has no such
    fact (a text or tool bridge has no layer and no fit timing, a vector bridge has no phrases). `source` is the
    data the bridge was built from: the dataset name, or "captions" for a caption-bank text or tool bridge;
    `n_phrases` counts the labels or captions a text or tool bridge compares a clip with."""
    from sense import specialists

    try:
        specialist = specialists.describe(bridge.modality)
    except ValueError:  # no specialist for this modality: the bridge does not use one
        specialist = None
    layer = getattr(bridge, "layer", None)
    n_fit = getattr(bridge, "n_fit", None)
    fit_seconds = getattr(bridge, "fit_seconds", None)
    phrases = getattr(bridge, "phrases", None)
    return {
        "name": bridge.name,
        "kind": bridge.kind,
        "modality": bridge.modality,
        "model": bridge.model_name,
        "specialist": specialist,
        "fit_seconds": None if fit_seconds is None else float(fit_seconds),
        "fit_seconds_by_layer": _str_keys(getattr(bridge, "fit_seconds_by_layer", None)),
        "layer": None if layer is None else int(layer),
        "dev_accuracy": _str_keys(getattr(bridge, "dev_accuracy", None)),
        "n_fit": None if n_fit is None else int(n_fit),
        "source": getattr(bridge, "source", None) or getattr(bridge, "dataset", None),
        "n_phrases": None if phrases is None else len(phrases),
    }


def availability(model_info: ModelInfo, modality: str) -> list[dict]:
    """For each bridge kind, in the order native, text, tool, image, vector: whether it could carry `modality` into
    the model, and why or why not in one plain clause. Loads no model and no specialist."""
    from sense import specialists

    label = model_info.label
    try:
        specialist = specialists.describe(modality)["label"]
    except ValueError:  # no specialist is known for this modality
        specialist = None
    native = modality in model_info.native
    if specialist is None:
        text_why = f"there is no {modality} specialist to describe the input in words"
    else:
        text_why = f"{specialist} can describe the {modality} in words for the prompt"
    if not model_info.open_weights:
        vector_why = f"{label} is not open-weight: an API exposes no hidden states to write into"
    elif specialist is None:
        vector_why = f"there is no {modality} specialist to give an embedding to write into {label}"
    else:
        vector_why = f"{specialist} embeddings can be written into the hidden states of open-weight {label}"
    tool = bool(model_info.tools and specialist is not None)
    if not model_info.tools:
        tool_why = f"{label} cannot call tools; tool calls are wired for API models only"
    elif specialist is None:
        tool_why = f"there is no {modality} specialist for {label} to call as a tool"
    else:
        tool_why = f"{label} can call {specialist} as a tool"
    return [
        {
            "kind": "native",
            "available": native,
            "why": f"{label} reads {modality} natively" if native else f"{label} does not read {modality} natively",
        },
        {"kind": "text", "available": specialist is not None, "why": text_why},
        {"kind": "tool", "available": tool, "why": tool_why},
        {
            "kind": "image",
            "available": False,
            "why": "the image bridge, which would show the input as a picture such as a spectrogram, is not built yet",
        },
        {"kind": "vector", "available": bool(model_info.open_weights and specialist is not None), "why": vector_why},
    ]
