"""Verification: score a candidate bridge on held-out examples against controls.

Public surface (do not rename):
  verify(model, bridge, dataset, request, text_bridge=None, max_items=None, seed=0) -> ReportCard
  free_text(model, bridge, dataset, max_items=100, seed=0, max_new_tokens=32) -> dict
      how often the model's free-text reply names the true label of a held-out clip, with controls
  labels_named(text, labels) -> list[int]
      indices of the labels a text names; the matching rule free_text uses

Each held-out example (dataset.test_idx, at most max_items) becomes a five-option question: dataset.question,
the true label and four other labels in a seeded random order. Chance is 0.2 (or 1/len(labels) if fewer than
five labels). The card's `accuracy` is the bridge's; `controls` holds:
  "swapped"         the same questions with each input taken from an example of a different label
  "no_input"        the same questions with no bridged input
  "lookup"          the same questions answered with the label of the nearest fitted example in specialist space,
                    searching only fitted examples of the question's five options (no base model)
  "text_bridge"     accuracy of `text_bridge` on the same questions, when one is given
  "unseen_classes"  only for request.scope == "open" and vector bridges: refit the bridge with a fifth of the
                    labels left out and score it on held-out examples of those labels
The card comes back with verdict "candidate"; the planner decides.

max_items None scores at most 200 held-out examples for an open-weight model, and at most
config.api_verify_items() (SENSE_API_VERIFY_ITEMS, default 50) for a model that is not open-weight, such as an
API model: there each question is asked three times (the bridge, swapped input and no input), each a paid
request, or up to three requests for a tool-bridge question, where the model may call the tool in two rounds
before it answers; the lookup and specialist-alone controls ask nothing (sense.models_api caches the answers, so
verifying the same questions again asks the API nothing). A tool bridge is scored like any other: its "swapped"
control hands the model the tool of a clip of another label. Fewer items give a wider interval: the card's `n`
says how many were scored and its `interval` how certain the accuracy is. An explicit max_items is used as given.

free_text asks held-out clips from the same split (dataset.test_idx) an open question instead (dataset.question
plus " Answer in a few words.") and counts how often the reply names the true label, under the same swapped and
no-input controls. It draws its own sample, so with different max_items its clips need not be the clips on a
report card. It returns a plain dict for a results file; nothing it measures goes on the ReportCard.
"""

from __future__ import annotations

import re
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from sense import config
from sense.bridges import fit_vector_bridge, five_options
from sense.contracts import BaseModel, Bridge, ChoiceItem, FitRequest, LabelledSet, ReportCard, SenseInput


def verify(
    model: BaseModel,
    bridge: Bridge,
    dataset: LabelledSet,
    request: FitRequest,
    text_bridge: Bridge | None = None,
    max_items: int | None = None,
    seed: int = 0,
) -> ReportCard:
    if bridge.model_name != model.info.name:
        raise ValueError(f"{bridge.name} was built for {bridge.model_name}, not {model.info.name}; build it for this model.")
    if max_items is None:
        max_items = 200 if model.info.open_weights else config.api_verify_items()
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
        "lookup": _lookup(dataset, test, options),
    }
    controls["specialist_alone"] = _specialist_alone(dataset, test, options, answers)
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
        interval=_wilson(accuracy, len(test)),
    )


def _wilson(p: float, n: int, z: float = 1.96) -> list[float]:
    """95% Wilson score interval for a proportion; unlike a bootstrap it has width at 0% and 100%."""
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(float(max(0.0, centre - half)), 4), round(float(min(1.0, centre + half)), 4)]


def _specialist_alone(dataset: LabelledSet, test: np.ndarray, options: list[list[int]], answers: np.ndarray) -> float:
    """The specialist answers the same questions with no base model: the option whose label text is closest
    to the example in the specialist's own space. Shows what the base model adds, or does not."""
    emb = dataset.emb[test].astype(np.float64)
    text = dataset.label_text_emb.astype(np.float64)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
    text /= np.linalg.norm(text, axis=1, keepdims=True) + 1e-12
    picks = [int(np.argmax(emb[k] @ text[opts].T)) for k, opts in enumerate(options)]
    return round(float((np.array(picks) == answers).mean()), 4)


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


def _lookup(dataset: LabelledSet, test: np.ndarray, options: list[list[int]]) -> float:
    """The bridge's own questions, answered with the label of the nearest fitted example among the offered
    options (cosine on embeddings z-scored with the fitting statistics)."""
    emb = dataset.emb.astype(np.float64)
    fit = dataset.fit_idx
    mu, sd = emb[fit].mean(0), emb[fit].std(0) + 1e-8
    a = (emb[test] - mu) / sd
    b = (emb[fit] - mu) / sd
    a /= np.linalg.norm(a, axis=1, keepdims=True) + 1e-12
    b /= np.linalg.norm(b, axis=1, keepdims=True) + 1e-12
    fit_y = dataset.y[fit]
    offered = (fit_y[None, None, :] == np.asarray(options)[:, :, None]).any(1)  # [n_test, n_fit]
    nearest = np.where(offered, a @ b.T, -np.inf).argmax(1)
    correct = offered.any(1) & (fit_y[nearest] == dataset.y[test])
    return round(float(correct.mean()), 4)


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


# ----------------------------------------------------------------------------- free-text answers
FREE_TEXT_SUFFIX = " Answer in a few words."
FREE_TEXT_EXAMPLES = 10


def free_text(
    model: BaseModel,
    bridge: Bridge,
    dataset: LabelledSet,
    max_items: int = 100,
    seed: int = 0,
    max_new_tokens: int = 32,
) -> dict:
    """How often the model's free-text reply names the true label of a held-out clip, through `bridge`.

    Each held-out clip (dataset.test_idx, at most max_items, drawn with `seed`) is sent with dataset.question
    plus " Answer in a few words." and no options, under three conditions:
      "bridge"    the clip through the bridge
      "swapped"   a clip of a different label through the bridge, in its place
      "no_input"  no clip at all
    Each condition reports, as shares of its n replies: names_true (the reply names the true label),
    names_true_only (the true label and no other label: the headline), names_other (at least one wrong label)
    and names_none (no label), with a 95% Wilson interval for names_true_only.

    This measures whether the reply names the label, not whether the rest of the reply is true: "jazz, played
    on a kazoo" counts as naming jazz. A label is named when labels_named finds it: every word of the label
    appears as a whole word of the reply, in any order, or as a simple plural ("dogs" names "dog"); case,
    hyphens and punctuation are ignored and "hiphop" equals "hip hop". A reply that describes the sound without
    using the label's words ("a ticking clock" for "clock tick") is counted as naming nothing.

    Returns a plain-JSON dict: model, dataset, bridge, kind, question, n, seed, max_new_tokens, result (the
    "bridge" condition), controls ("swapped", "no_input"), examples (up to 10 bridge-condition replies spread
    over the held-out clips), seconds and created. Nothing here goes on the ReportCard.
    """
    if bridge.model_name != model.info.name:
        raise ValueError(f"{bridge.name} was built for {bridge.model_name}, not {model.info.name}; build it for this model.")
    started = time.monotonic()
    rng = np.random.default_rng(seed)
    test = _subsample(dataset.test_idx, max_items, rng)
    if len(test) == 0:
        raise ValueError(f"Dataset {dataset.name} has no held-out examples to score free-text answers on.")
    truth = dataset.y[test]
    if len(np.unique(truth)) < 2:
        fix = (
            "Raise max_items or change the seed."
            if len(test) < len(dataset.test_idx)
            else "Use a dataset whose held-out split has at least two labels."
        )
        raise ValueError(
            f"The {len(test)} held-out clips drawn from {dataset.name} (max_items={max_items}, seed={seed}) are all "
            f"of one label, so the swapped-input control cannot run. {fix}"
        )
    swap = _swap_partners(test, dataset.y, rng)
    prompt = dataset.question + FREE_TEXT_SUFFIX

    def replies(inputs: list[list[SenseInput]]) -> list[str]:
        return [model.chat(prompt, inp, max_new_tokens=max_new_tokens) for inp in inputs]

    said = {
        "bridge": replies([[bridge.from_embedding(dataset.emb[i], _source(dataset, i))] for i in test]),
        "swapped": replies([[bridge.from_embedding(dataset.emb[j], _source(dataset, j))] for j in swap]),
        "no_input": replies([[] for _ in test]),
    }
    named = {condition: [labels_named(r, dataset.labels) for r in rs] for condition, rs in said.items()}
    stats = {condition: _naming_stats(hits, truth) for condition, hits in named.items()}
    picks = sorted({int(round(k)) for k in np.linspace(0, len(test) - 1, min(FREE_TEXT_EXAMPLES, len(test)))})
    examples = [
        {
            "source": _source(dataset, int(test[k])),
            "label": dataset.labels[int(truth[k])],
            "reply": said["bridge"][k],
            "named": [dataset.labels[c] for c in named["bridge"][k]],
        }
        for k in picks
    ]
    return {
        "model": model.info.name,
        "dataset": dataset.name,
        "bridge": bridge.name,
        "kind": bridge.kind,
        "question": prompt,
        "n": int(len(test)),
        "seed": seed,
        "max_new_tokens": max_new_tokens,
        "result": stats["bridge"],
        "controls": {"swapped": stats["swapped"], "no_input": stats["no_input"]},
        "examples": examples,
        "seconds": round(time.monotonic() - started, 2),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def labels_named(text: str, labels: list[str]) -> list[int]:
    """Indices of the labels that `text` names, in label order.

    Both sides are normalised: lowercase, "_" and "-" and every other non-alphanumeric character become spaces,
    whitespace is collapsed, and "hip hop" is read as "hiphop". A label is named when every word of the label
    appears as a whole word of the text, in any order, either as it is or as a simple plural variant: the word
    plus "s" or "es", or the word without a trailing "s". So "dogs barking" names "dog", "waves of the sea"
    names "sea waves" and "Hip-hop" names "hip hop", but "rockabilly" does not name "rock". A label with no
    letters or digits is never named.
    """
    said = set(_match_words(text))
    out = []
    for k, label in enumerate(labels):
        words = _match_words(label)
        if words and all(_word_forms(w) & said for w in words):
            out.append(k)
    return out


def _match_words(text: str) -> list[str]:
    """Lowercase words for labels_named: non-alphanumerics (including "_" and "-") split words; "hip hop" -> "hiphop"."""
    text = " ".join(re.sub(r"[\W_]+", " ", text.lower()).split())
    return re.sub(r"\bhip hop\b", "hiphop", text).split()


def _word_forms(word: str) -> set[str]:
    """A label word and the simple plural variants that also count as naming it."""
    forms = {word, word + "s", word + "es"}
    if len(word) > 1 and word.endswith("s"):
        forms.add(word[:-1])
    return forms


def _naming_stats(named: list[list[int]], truth: np.ndarray) -> dict:
    """Shares of replies that name the true label, only the true label, a wrong label, or no label."""
    n = len(named)
    true = np.array([int(t) in hits for hits, t in zip(named, truth)], dtype=bool)
    other = np.array([any(h != int(t) for h in hits) for hits, t in zip(named, truth)], dtype=bool)
    none = np.array([not hits for hits in named], dtype=bool)
    true_only = true & ~other
    return {
        "n": n,
        "names_true": round(float(true.mean()), 4),
        "names_true_only": round(float(true_only.mean()), 4),
        "names_other": round(float(other.mean()), 4),
        "names_none": round(float(none.mean()), 4),
        "interval": _wilson(float(true_only.mean()), n),
    }
