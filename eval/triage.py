"""Forest-monitoring triage: the installed audio sense, put to work on a batch of unlabelled clips.

The user is a forest-monitoring team that listens for illegal logging. They want the clips with a chainsaw, a
hand saw or an engine flagged, so a ranger listens to those and not to the whole batch. ESC-50 stands in for
field recordings: its held-out fold 5 (400 clips, 8 per sound, 24 of them chainsaw, hand saw or engine) is the
batch. The model never sees its labels; they are used afterwards, only to score the flags. Fold 4 (the dev fold,
also 400 clips) is for choosing how to ask, so the reported fold-5 score is not tuned on itself.

How the base model is asked (--question; the clip is flagged when the answer is a target):
  ranger   "A forest ranger is listening for illegal logging. Which of these is in the audio clip?" with the
           three targets in a seeded order, then "none of these". On Qwen3.5-0.8B (fold 5) the framing primes
           it: with no audio it answers "chainsaw" every time, and through the sense it flags 369 of 400.
  neutral  "Which of these is in the audio clip?" with the targets and "none of these", all four in a seeded order.
  yesno    one question per target, "Is there a <target> in the audio clip?" (yes/no, seeded order); the first
           target answered yes is the answer.
  chat     an open question, "What sound is in the audio clip? Answer in a few words."; the answer is the first
           target the reply names (with "chain saw", "handsaw", "motor" as variants), otherwise "none of these".
  all      every one of the above, one record each, and a comparison table (use with --fold dev).

Two controls answer the same clips:
  no_input  the same questions with no audio: what the model says from the question alone
  lookup    the label of the nearest labelled ESC-50 fit example in specialist (CLAP) space, mapped to a target
            or "none of these"; no base model. This is what the labelled data alone gives.

  .venv/bin/python -m eval.triage [--model M] [--question Q] [--fold test|dev] [--folder DIR] [--limit N] [--save FILE]
  .venv/bin/python -m integrations.databricks triage ...    the same, then written to Databricks

--folder triages a folder of audio files instead (real unlabelled recordings: no truth, so no scores).

Public surface:
  run(model_name=None, folder=None, limit=None, seed=0, question=DEFAULT_QUESTION, fold="test") -> dict
  table(record) -> str            a plain-text summary for the terminal
  compare(records) -> str         one line per record, for --question all
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from sense import config, datasets, models, registry
from sense.contracts import Bridge, ChoiceItem
from sense.specialists import AUDIO_EXTENSIONS, get_specialist

TASK = "forest-monitoring triage"
USER = "a forest-monitoring team listening for illegal logging"
TARGETS = ("chainsaw", "hand saw", "engine")
NONE = "none of these"
QUESTIONS = ("ranger", "neutral", "yesno", "chat")
DEFAULT_QUESTION = "neutral"
RANGER = "A forest ranger is listening for illegal logging. Which of these is in the audio clip?"
NEUTRAL = "Which of these is in the audio clip?"
OPEN = "What sound is in the audio clip? Answer in a few words."
VARIANTS = {"chainsaw": ("chainsaw", "chain saw"), "hand saw": ("hand saw", "handsaw"), "engine": ("engine", "motor")}
REFERENCE_SET = "esc50"
METHODS = ("sense", "no_input", "lookup")
STAND_IN = (
    "ESC-50 fold {fold} stands in for field recordings; its labels are hidden from the model and used only to "
    "score the flags afterwards."
)


def run(
    model_name: str | None = None,
    folder: str | Path | None = None,
    limit: int | None = None,
    seed: int = 0,
    question: str = DEFAULT_QUESTION,
    fold: str = "test",
) -> dict:
    if question not in QUESTIONS:
        raise ValueError(f"Unknown question {question!r}; choose one of {', '.join(QUESTIONS)}.")
    if fold not in ("test", "dev"):
        raise ValueError(f"fold must be 'test' (ESC-50 fold 5) or 'dev' (fold 4), not {fold!r}.")
    t0 = time.monotonic()
    model_name = model_name or ("fake" if config.fake() else "qwen3.5-0.8b")
    bridge, card = _installed_sense(model_name)
    reference = datasets.load_labelled_set(REFERENCE_SET)

    if folder is None:
        idx = reference.test_idx if fold == "test" else reference.dev_idx
        fold_no = 5 if fold == "test" else 4
        if limit:
            idx = _subsample_keep_targets(idx, reference, limit, seed)
        emb = reference.emb[idx]
        names = [Path(reference.files[i]).name if reference.files else f"esc50-fold{fold_no}-{i}" for i in idx]
        truth = [_group(reference.labels[reference.y[i]]) for i in idx]
        source = f"{REFERENCE_SET} fold {fold_no}" + (" (dev)" if fold == "dev" else "")
    else:
        paths = _audio_files(Path(folder))[: limit or None]
        emb = get_specialist(bridge.modality).embed_files(paths)
        names = [p.name for p in paths]
        truth = [None] * len(paths)
        source = str(folder)

    inputs = [bridge.from_embedding(e, source=n) for e, n in zip(emb, names)]
    model = models.load_model(model_name)
    sense, replies = _ask(model, question, inputs, seed)
    blind, _ = _ask(model, question, [None] * len(inputs), seed)
    lookup = _lookup(reference, emb)

    clips = [
        {
            "i": k,
            "clip": names[k],
            "truth": truth[k],
            "sense": sense[k],
            "no_input": blind[k],
            "lookup": lookup[k],
            "flagged": sense[k] != NONE,
            "detail": inputs[k].detail,
            **({"reply": replies[k]} if replies else {}),
        }
        for k in range(len(inputs))
    ]
    has_truth = folder is None
    return {
        "id": f"triage-{_safe(model_name)}-{question}-{fold if has_truth else 'folder'}-"
        f"{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}",
        "recorded": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "task": TASK,
        "user": USER,
        "note": STAND_IN.format(fold=4 if fold == "dev" else 5) if has_truth else "Unlabelled recordings: flags only, no scores.",
        "question": question,
        "question_text": {"ranger": RANGER, "neutral": NEUTRAL, "yesno": "Is there a <target> in the audio clip?", "chat": OPEN}[question],
        "targets": list(TARGETS),
        "model": model_name,
        "sense": {
            "bridge": card.get("bridge"),
            "kind": card.get("kind"),
            "dataset": card.get("dataset"),
            "scope": card.get("scope"),
            "accuracy": card.get("accuracy"),
            "n": card.get("n"),
        },
        "batch": {"source": source, "n": len(clips), "has_truth": has_truth},
        "summary": {m: summarise(clips, m, has_truth) for m in METHODS},
        "clips": clips,
        "seconds": round(time.monotonic() - t0, 2),
    }


def _ask(model, question: str, inputs: list, seed: int) -> tuple[list[str], list[str] | None]:
    """One answer per clip (a target or NONE); inputs[k] None means no audio. For "chat", also the replies."""
    n = len(inputs)
    given = [[inp] if inp is not None else [] for inp in inputs]
    if question in ("ranger", "neutral"):
        text = RANGER if question == "ranger" else NEUTRAL
        options = [_options(seed, k, shuffle_none=question == "neutral") for k in range(n)]
        picks = model.choose([ChoiceItem(text, opts, g) for opts, g in zip(options, given)])
        return [opts[p] for opts, p in zip(options, picks)], None
    if question == "yesno":
        items, yes_at = [], []
        for k in range(n):
            for j, t in enumerate(TARGETS):
                yes_first = bool(np.random.default_rng([seed, k, j]).integers(2))
                items.append(ChoiceItem(f"Is there a {t} in the audio clip?", ["yes", "no"] if yes_first else ["no", "yes"], given[k]))
                yes_at.append(0 if yes_first else 1)
        picks = model.choose(items)
        said_yes = [p == y for p, y in zip(picks, yes_at)]
        return [next((t for j, t in enumerate(TARGETS) if said_yes[k * len(TARGETS) + j]), NONE) for k in range(n)], None
    # chat; with no audio every prompt is the same, so ask once
    replies: list[str] = []
    cache: dict[int, str] = {}
    for k, g in enumerate(given):
        key = k if g else -1
        if key not in cache:
            cache[key] = model.chat(OPEN, g, max_new_tokens=24)
        replies.append(cache[key])
    return [_named_target(r) for r in replies], replies


def _named_target(reply: str) -> str:
    """The target the reply names first (whole words, a plural "s" allowed), or NONE."""
    words = " " + " ".join("".join(ch if ch.isalnum() else " " for ch in reply.lower()).split()) + " "
    found = []
    for t in TARGETS:
        at = [words.find(f" {form} ") for v in VARIANTS[t] for form in (v, v + "s")]
        at = [i for i in at if i >= 0]
        if at:
            found.append((min(at), t))
    return min(found)[1] if found else NONE


def summarise(clips: list[dict], method: str, has_truth: bool) -> dict:
    answers = [c[method] for c in clips]
    flagged = np.array([a != NONE for a in answers])
    out = {
        "n": len(clips),
        "flagged": int(flagged.sum()),
        "listen_fraction": round(float(flagged.mean()), 4) if clips else None,
        "answers": {a: answers.count(a) for a in (*TARGETS, NONE)},
    }
    if not has_truth:
        return out
    truth = [c["truth"] for c in clips]
    threat = np.array([t != NONE for t in truth])
    caught = flagged & threat
    named = np.array([a == t for a, t in zip(answers, truth)]) & threat
    out.update(
        threats=int(threat.sum()),
        caught=int(caught.sum()),
        named_right=int(named.sum()),
        recall=_ratio(caught.sum(), threat.sum()),
        recall_interval=_wilson(caught.sum(), threat.sum()),
        precision=_ratio(caught.sum(), flagged.sum()),
        false_alarm_rate=_ratio((flagged & ~threat).sum(), (~threat).sum()),
        per_target={
            t: {"clips": sum(x == t for x in truth), "caught": int(sum(f and x == t for f, x in zip(flagged, truth)))}
            for t in TARGETS
        },
    )
    return out


def table(record: dict) -> str:
    s = record["sense"]
    acc = s.get("accuracy")
    lines = [
        f"Triage for {record['user']}: {record['batch']['n']} clips from {record['batch']['source']}, "
        f"question '{_question_name(record)}'.",
        f"Model {record['model']} with its installed sense {s['bridge']} "
        f"({s['kind']}, {'%.1f%%' % (100 * acc) if acc is not None else '?'} on its report card). {record['note']}",
        "",
        f"{'method':<10} {'flagged':>8} {'listen':>7}" + (f" {'caught':>8} {'recall':>7} {'precision':>9} {'false alarm':>11}" if record["batch"]["has_truth"] else ""),
    ]
    for m in METHODS:
        r = record["summary"][m]
        row = f"{m:<10} {r['flagged']:>8} {_pct(r['listen_fraction']):>7}"
        if record["batch"]["has_truth"]:
            row += f" {str(r['caught']) + '/' + str(r['threats']):>8} {_pct(r['recall']):>7} {_pct(r['precision']):>9} {_pct(r['false_alarm_rate']):>11}"
        lines.append(row)
    flagged = [c for c in record["clips"] if c["flagged"]]
    if flagged:
        lines += ["", "Flagged for a ranger (first 12):"]
        lines += [
            f"  {c['clip']:<28} {c['sense']:<10}" + (f" truth: {c['truth']:<13}" if c["truth"] is not None else "") + f" {c['detail'][:70]}"
            for c in flagged[:12]
        ]
    lines.append(f"\n{record['seconds']} s.")
    return "\n".join(lines)


def compare(records: list[dict]) -> str:
    """One line per record: how each way of asking did, through the sense and with no input."""
    has_truth = all(r["batch"]["has_truth"] for r in records)
    head = f"{'question':<9} {'sense flags':>11} {'no-input flags':>14}"
    head += f" {'caught':>7} {'recall':>7} {'precision':>9} {'false alarm':>11}" if has_truth else ""
    lines = [f"{records[0]['batch']['n']} clips from {records[0]['batch']['source']}; lookup control flags "
             f"{records[0]['summary']['lookup']['flagged']}.", head]
    for r in records:
        s, b = r["summary"]["sense"], r["summary"]["no_input"]
        row = f"{_question_name(r):<9} {s['flagged']:>11} {b['flagged']:>14}"
        if has_truth:
            row += f" {str(s['caught']) + '/' + str(s['threats']):>7} {_pct(s['recall']):>7} {_pct(s['precision']):>9} {_pct(s['false_alarm_rate']):>11}"
        lines.append(row)
    return "\n".join(lines)


# ----------------------------------------------------------------------------- helpers
def _question_name(record: dict) -> str:
    q = record.get("question") or "ranger"
    return "ranger" if " " in q else q  # records from before the variants hold the ranger question's text


def _installed_sense(model_name: str) -> tuple[Bridge, dict]:
    cards = {e["modality"]: e["card"] for e in registry.installed(model_name)}
    for bridge in registry.get_bridges(model_name):
        if bridge.modality == "audio":
            return bridge, cards["audio"]
    raise RuntimeError(
        f"{model_name} has no audio sense installed. Run the agent first, for example "
        f"`.venv/bin/python scripts/demo_cli.py --model {model_name} --dataset esc50 --scope open`."
    )


def _group(label: str) -> str:
    return label if label in TARGETS else NONE


def _options(seed: int, k: int, shuffle_none: bool = False) -> list[str]:
    rng = np.random.default_rng([seed, k])
    if shuffle_none:
        allopts = [*TARGETS, NONE]
        return [allopts[j] for j in rng.permutation(len(allopts))]
    return [TARGETS[j] for j in rng.permutation(len(TARGETS))] + [NONE]


def _lookup(reference, emb: np.ndarray) -> list[str]:
    """Label of the nearest fitted example (cosine on embeddings z-scored with the fit statistics), as in
    sense.verify's lookup control, but over all labels."""
    ref = reference.emb.astype(np.float64)
    fit = reference.fit_idx
    mu, sd = ref[fit].mean(0), ref[fit].std(0) + 1e-8
    a = (np.asarray(emb, dtype=np.float64) - mu) / sd
    b = (ref[fit] - mu) / sd
    a /= np.linalg.norm(a, axis=1, keepdims=True) + 1e-12
    b /= np.linalg.norm(b, axis=1, keepdims=True) + 1e-12
    nearest = (a @ b.T).argmax(1)
    return [_group(reference.labels[reference.y[fit[j]]]) for j in nearest]


def _subsample_keep_targets(idx: np.ndarray, reference, limit: int, seed: int) -> np.ndarray:
    """At most `limit` clips; every target clip is kept, the rest is a seeded sample of the other sounds."""
    is_target = np.array([reference.labels[reference.y[i]] in TARGETS for i in idx])
    rest = np.random.default_rng(seed).permutation(idx[~is_target])[: max(0, limit - int(is_target.sum()))]
    return np.sort(np.concatenate([idx[is_target], rest]))


def _audio_files(folder: Path) -> list[Path]:
    if not folder.is_dir():
        raise FileNotFoundError(f"{folder} is not a folder.")
    files = sorted(p for p in folder.rglob("*") if p.suffix.lower() in AUDIO_EXTENSIONS and not p.name.startswith("."))
    if not files:
        raise ValueError(f"{folder} holds no audio files ({', '.join(sorted(AUDIO_EXTENSIONS))}).")
    return files


def _ratio(a, b) -> float | None:
    return round(float(a) / float(b), 4) if b else None


def _wilson(k, n, z: float = 1.96) -> list[float] | None:
    if not n:
        return None
    p, n = float(k) / float(n), float(n)
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]


def _pct(v) -> str:
    return "-" if v is None else f"{100 * v:.1f}%"


def _safe(name: str) -> str:
    return "".join(ch if ch.isalnum() else "-" for ch in name)


def save(record: dict, path: str | Path | None = None) -> Path:
    path = Path(path) if path else config.home() / "triage" / f"{record['id']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1))
    return path


def add_arguments(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--model", help="base model with an installed audio sense (default qwen3.5-0.8b; fake in fake mode)")
    ap.add_argument("--folder", help="triage the audio files in this folder instead of ESC-50 fold 5")
    ap.add_argument("--limit", type=int, help="at most this many clips (on ESC-50, every target clip is kept)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--question", default=DEFAULT_QUESTION, choices=[*QUESTIONS, "all"],
                    help=f"how the model is asked (default {DEFAULT_QUESTION}); 'all' compares them")
    ap.add_argument("--fold", default="test", choices=["test", "dev"],
                    help="ESC-50 fold 5 (test, the reported score) or fold 4 (dev, for choosing the question)")
    ap.add_argument("--save", help="write the record here (default <SENSE_HOME>/triage/<id>.json)")


def run_all(args: argparse.Namespace) -> list[dict]:
    """The records asked for on the command line: one, or one per question for --question all."""
    questions = QUESTIONS if args.question == "all" else (args.question,)
    return [run(args.model, args.folder, args.limit, args.seed, q, args.fold) for q in questions]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m eval.triage", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_arguments(ap)
    args = ap.parse_args(argv)
    records = run_all(args)
    for record in records:
        print(table(record), "\n")
        print(f"Saved {save(record, args.save if len(records) == 1 else None)}\n")
    if len(records) > 1:
        print(compare(records))
    return 0


if __name__ == "__main__":
    sys.exit(main())
