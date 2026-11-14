"""Verification: score a candidate bridge on held-out examples against controls.

Public surface (do not rename):
  verify(model, bridge, dataset, request, text_bridge=None, max_items=200, seed=0) -> ReportCard

Each held-out example (dataset.test_idx, at most max_items) becomes a five-option question: dataset.question,
the true label and four other labels in a seeded random order. Chance is 0.2 (or 1/len(labels) if fewer than
five labels). The card's `accuracy` is the bridge's; `controls` holds:
  "swapped"         the same questions with each input taken from an example of a different label
  "no_input"        the same questions with no bridged input
  "lookup"          answer with the label of the nearest fitted example in specialist space (no base model)
  "text_bridge"     accuracy of `text_bridge` on the same questions, when one is given
  "unseen_classes"  only for request.scope == "open" and vector bridges: refit the bridge with a fifth of the
                    labels left out and score it on held-out examples of those labels
The card comes back with verdict "candidate"; the planner decides.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from sense.bridges import fit_vector_bridge, five_options
from sense.contracts import BaseModel, Bridge, ChoiceItem, FitRequest, LabelledSet, ReportCard, SenseInput


def verify(
    model: BaseModel,
    bridge: Bridge,
    dataset: LabelledSet,
    request: FitRequest,
    text_bridge: Bridge | None = None,
    max_items: int = 200,
    seed: int = 0,
) -> ReportCard:
    if bridge.model_name != model.info.name:
        raise ValueError(f"{bridge.name} was built for {bridge.model_name}, not {model.info.name}; build it for this model.")
    rng = np.random.default_rng(seed)
    test = _subsample(dataset.test_idx, max_items, rng)
    if len(test) == 0:
        raise ValueError(f"Dataset {dataset.name} has no held-out examples to verify on.")
    n_labels = len(dataset.labels)
    options, answers = five_options(dataset.y[test], n_labels, rng)
    swap = _swap_partners(test, dataset.y, rng)

    def score(inputs: list[list[SenseInput]]) -> float:
        return _accuracy(model, dataset, options, answers, inputs)

    accuracy = score([[bridge.from_embedding(dataset.emb[i], _source(dataset, i))] for i in test])
    controls = {
        "swapped": score([[bridge.from_embedding(dataset.emb[j], _source(dataset, j))] for j in swap]),
        "no_input": score([[] for _ in test]),
        "lookup": _lookup(dataset, test),
    }
    if text_bridge is not None:
        controls["text_bridge"] = score([[text_bridge.from_embedding(dataset.emb[i], _source(dataset, i))] for i in test])
    if request.scope == "open" and bridge.kind == "vector":
        controls["unseen_classes"] = _unseen_classes(model, bridge, dataset, max_items, rng)
    return ReportCard(
        model=model.info.name,
        modality=bridge.modality,
        dataset=dataset.name,
        scope=request.scope,
        bridge=bridge.name,
        kind=bridge.kind,
        n=len(test),
        chance=round(1 / min(5, n_labels), 4),
        accuracy=accuracy,
        controls=controls,
        verdict="candidate",
        reason="",
        created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


def _subsample(idx: np.ndarray, max_items: int, rng: np.random.Generator) -> np.ndarray:
    idx = np.asarray(idx, dtype=np.int64)
    return np.sort(rng.choice(idx, max_items, replace=False)) if len(idx) > max_items else idx


def _source(dataset: LabelledSet, i: int) -> str:
    return Path(dataset.files[i]).name if dataset.files else f"{dataset.name} example {i}"


def _swap_partners(idx: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """For each example, another example of the same split with a different label."""
    out = []
    for i in idx:
        pool = idx[y[idx] != y[i]]
        if len(pool) == 0:
            raise ValueError("The swapped-input control needs held-out examples of at least two labels.")
        out.append(int(pool[rng.integers(len(pool))]))
    return np.array(out, dtype=np.int64)


def _accuracy(
    model: BaseModel,
    dataset: LabelledSet,
    options: list[list[int]],
    answers: np.ndarray,
    inputs: list[list[SenseInput]],
) -> float:
    items = [
        ChoiceItem(question=dataset.question, options=[dataset.labels[c] for c in opts], inputs=inp)
        for opts, inp in zip(options, inputs)
    ]
    return round(float((np.array(model.choose(items)) == answers).mean()), 4)


def _lookup(dataset: LabelledSet, test: np.ndarray) -> float:
    """Label of the nearest fitted example (cosine on embeddings z-scored with the fitting statistics)."""
    emb = dataset.emb.astype(np.float64)
    fit = dataset.fit_idx
    mu, sd = emb[fit].mean(0), emb[fit].std(0) + 1e-8
    a = (emb[test] - mu) / sd
    b = (emb[fit] - mu) / sd
    a /= np.linalg.norm(a, axis=1, keepdims=True) + 1e-12
    b /= np.linalg.norm(b, axis=1, keepdims=True) + 1e-12
    nearest = fit[(a @ b.T).argmax(1)]
    return round(float((dataset.y[nearest] == dataset.y[test]).mean()), 4)


def _unseen_classes(model: BaseModel, bridge: Bridge, dataset: LabelledSet, max_items: int, rng: np.random.Generator) -> float:
    """Refit with a fifth of the labels left out; score on held-out examples of those labels."""
    n_labels = len(dataset.labels)
    held = rng.choice(n_labels, max(1, n_labels // 5), replace=False)
    fit = dataset.fit_idx[~np.isin(dataset.y[dataset.fit_idx], held)]
    test = _subsample(dataset.test_idx[np.isin(dataset.y[dataset.test_idx], held)], max_items, rng)
    if len(test) == 0:
        raise ValueError(f"Dataset {dataset.name} has no held-out examples of the left-out labels.")
    refit = fit_vector_bridge(model, dataset, fit, layer=getattr(bridge, "layer", None))
    options, answers = five_options(dataset.y[test], n_labels, rng)
    inputs = [[refit.from_embedding(dataset.emb[i], _source(dataset, i))] for i in test]
    return _accuracy(model, dataset, options, answers, inputs)
