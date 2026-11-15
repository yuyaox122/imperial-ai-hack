"""The data step: decide what data the goal needs, find a labelled dataset on the Hugging Face Hub, fetch a small
capped sample of it into a folder, and build a labelled set from that folder, or fall back to a pre-wired set.

Public surface (do not rename):
  DataUnavailable                         raised when no usable sample can be had; the message says why
  DataNeed, DatasetCandidate, FetchReport plain dataclasses with to_dict()
  PER_LABEL, MAX_LABELS, MIN_PER_LABEL, MAX_PAGES, MAX_BYTES, DEADLINE_S
  NEED_SCHEMA, DATASET_SCHEMA             structured-output schemas (DATASET_SCHEMA's top-level property is "datasets")
  decide_need(goal, detection) -> DataNeed
  find_datasets(need, modality, goal) -> (candidates, claims, origin)
  fetch_sample(candidate, dest, scope, *, requested_labels=None) -> FetchReport
  choose_prewired(goal, modality, *, requested_labels=None) -> str | None
  safe_label(text, index) -> str
  build_set(folder, name, *, title, question, modality, source, licence) -> str

decide_need asks agent.llm.ask_json and accepts only a kind of "labelled" or "probe" with at most 20 string labels;
otherwise a word rule decides (any of "any ", "anything", "describe", "whatever", "open-ended" means probe).

fetch_sample owns `dest`: it replaces whatever is there, and every file it writes is
dest/<safe_label>/<row_idx>.<audio extension>, so no label text or URL can place a file outside dest. It lists up to
MAX_PAGES pages of the dataset viewer spread evenly over the whole split (so a split sorted by label still shows
several labels), reserves slots for requested labels before keeping optional labels up to MAX_LABELS, and downloads at most PER_LABEL
files per label, label by label in turn, each label's rows picked evenly across the split. Colliding normalized
labels are rejected. Every requested category must have enough rows and retained downloads, otherwise the
sample fails. Named coverage only accepts ASCII letters/numbers and space, underscore or hyphen separators;
unrepresentable requested or original source names fail before downloads. Fallback selection and the caller
also validate named categories against the built catalogue. A failure before body bytes arrive skips that row; a partially transferred failure aborts
and cleans up the sample so discarded bytes cannot fuel retries beyond the budget. Downloading stops when every
label is full, at MAX_BYTES (plus at most one overflow probe byte) or at DEADLINE_S. Labels with fewer than
MIN_PER_LABEL files are deleted; fewer than 2 labels left (3 for scope "open") deletes dest and raises
DataUnavailable. A ResearchUnavailable from the viewer (including fake mode, which has no network) becomes
DataUnavailable after dest is deleted. The viewer's audio URLs are signed and expire: they are used at once and
never stored. All network access goes through agent.research.
"""

from __future__ import annotations

import re
import shutil
import time
import urllib.parse
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath

from agent import llm, research
from agent.claims import Claim, citable, settle_dataset_claim
from agent.detect import Detection
from sense import datasets
from sense.specialists import AUDIO_EXTENSIONS

PER_LABEL = 30
MAX_LABELS = 10
MIN_PER_LABEL = 10
MAX_PAGES = 12
MAX_BYTES = 200_000_000
DEADLINE_S = 180
PAGE_ROWS = 100  # the most rows the dataset viewer returns in one page
FILE_MAX_BYTES = 20_000_000  # one audio file larger than this is skipped
MAX_NEED_LABELS = 20

_PROBE_WORDS = ("any ", "anything", "describe", "whatever", "open-ended")
_LABEL_COLUMNS = ("label", "category", "genre", "class", "target")
_SCOPE = {"labelled": "closed", "probe": "open"}
_now = time.monotonic  # tests replace it to run the deadline

NEED_SCHEMA = {
    "type": "object",
    "properties": {
        "kind": {"type": "string", "enum": ["labelled", "probe"]},
        "labels": {"type": "array", "items": {"type": "string"}},
        "reason": {"type": "string"},
    },
    "required": ["kind", "labels", "reason"],
    "additionalProperties": False,
}

DATASET_SCHEMA = {
    "type": "object",
    "properties": {
        "datasets": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "hub_id": {"type": "string"},
                    "title": {"type": "string"},
                    "licence": {"type": ["string", "null"]},
                    "n_examples": {"type": ["integer", "null"]},
                    "labels": {"type": "array", "items": {"type": "string"}},
                    "statement": {"type": "string"},
                    "source_url": {"type": "string"},
                },
                "required": ["hub_id", "source_url"],
            },
        }
    },
    "required": ["datasets"],
}

_NEED_INSTRUCTION = (
    "Decide what data is needed to verify a new sense for this goal. Answer kind 'labelled' when the goal is to "
    "recognise or sort inputs into categories, or 'probe' when it asks to describe any input in an open-ended way. "
    "List only the category names the goal itself names (an empty list when it names none), and give a one-sentence "
    "reason. Return JSON with kind, labels and reason."
)


class DataUnavailable(RuntimeError):
    """No usable labelled sample: the reason is in the message."""


@dataclass
class DataNeed:
    kind: str  # "labelled" | "probe"
    scope: str  # "closed" for labelled, "open" for probe
    labels: list[str]  # categories named in the goal, possibly empty
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DatasetCandidate:
    hub_id: str
    title: str
    source_url: str
    licence: str | None = None  # source assertion, not the independent Hub observation
    n_examples: int | None = None
    labels: list[str] = field(default_factory=list)
    observed: dict = field(default_factory=dict)  # independent Hub facts; fields above remain source assertions

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FetchReport:
    folder: Path
    hub_id: str
    labels: dict[str, int]  # label -> files kept
    n_files: int
    n_rows_total: int  # selected split only, not necessarily the entire dataset
    licence: str | None  # independent Hub licence or None
    bytes: int
    selected_split: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self) | {"folder": str(self.folder)}


# ----------------------------------------------------------------------------- what data is needed
def decide_need(goal: str, detection: Detection) -> DataNeed:
    payload = {"goal": goal, "modality": detection.modality, "format": detection.format}
    return _need_from(llm.ask_json(_NEED_INSTRUCTION, payload, NEED_SCHEMA)) or _rule_need(goal)


def _need_from(answer: object) -> DataNeed | None:
    if not isinstance(answer, dict):
        return None
    kind, labels, reason = answer.get("kind"), answer.get("labels"), answer.get("reason")
    if not isinstance(kind, str) or kind not in _SCOPE or not isinstance(labels, list) or len(labels) > MAX_NEED_LABELS:
        return None
    if not all(isinstance(label, str) for label in labels):
        return None
    reason = reason.strip()[:300] if isinstance(reason, str) and reason.strip() else "The language model chose it."
    return DataNeed(kind, _SCOPE[kind], [label.strip() for label in labels if label.strip()], reason)


def _rule_need(goal: str) -> DataNeed:
    if any(word in str(goal).lower() for word in _PROBE_WORDS):
        return DataNeed("probe", "open", [], "The goal asks for open-ended description, so the check is open scope.")
    return DataNeed("labelled", "closed", [], "The goal is to recognise categories, so labelled examples are needed.")


# ----------------------------------------------------------------------------- finding a dataset
def find_datasets(need: DataNeed, modality: str, goal: str) -> tuple[list[DatasetCandidate], list[Claim], str]:
    # `need` is part of the fixed signature; the query is fixed by the modality and the goal.
    query = (
        f"Which labelled {modality} datasets on Hugging Face fit this goal: {goal}? Give the dataset id, licence, "
        "number of examples and label names, with the page each fact comes from."
    )
    try:
        found = research.search_structured(query, DATASET_SCHEMA, include_domains=["huggingface.co"])
    except research.ResearchUnavailable:
        return [], [], "unavailable"
    items = found.data.get("datasets") if isinstance(found.data, dict) else None
    names = {s.get("url"): s.get("name") for s in found.sources if isinstance(s, dict)}
    candidates, claims = [], []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict) or not citable(item.get("hub_id"), item.get("source_url")):
            continue
        candidate, claim = _candidate(item, _text(names.get(item["source_url"])))
        if found.origin == "fake":
            claim.statement = "Fake claim for tests: " + claim.statement
        try:
            info = research.hub_dataset(candidate.hub_id)
        except research.ResearchUnavailable as exc:
            candidate.observed["hub_error"] = str(exc)
            claim.evidence = f"The dataset's Hub record could not be checked: {exc}"
        else:
            if info is None:
                candidate.observed["hub_exists"] = False
                claim.status, claim.evidence = "refuted", "no such dataset on the Hugging Face Hub"
            else:
                candidate.observed.update(hub_exists=True, licence=_text(info.get("licence")) or None)
                claim.evidence = f"The Hub has this dataset, with licence {candidate.observed['licence'] or 'not stated'}."
        claim.observed = dict(candidate.observed)
        # Contradictions can already be established before acquisition.
        if candidate.observed.get("licence") and candidate.licence and candidate.licence.casefold() != candidate.observed["licence"].casefold():
            settle_dataset_claim(claim)
        candidates.append(candidate)
        claims.append(claim)
    return candidates, claims, found.origin


def _candidate(item: dict, source_name: str) -> tuple[DatasetCandidate, Claim]:
    hub_id, url = item["hub_id"], item["source_url"]
    licence = _text(item.get("licence")) or None
    n = item.get("n_examples")
    n = n if isinstance(n, int) and not isinstance(n, bool) and n >= 0 else None
    raw = item.get("labels") if isinstance(item.get("labels"), list) else []
    labels = [label.strip() for label in raw if isinstance(label, str) and label.strip()][:100]
    parts = [f"licence {licence}"] if licence else []
    parts += [f"{n} examples"] if n is not None else []
    if labels:
        more = f" and {len(labels) - 10} more" if len(labels) > 10 else ""
        parts.append("labels " + ", ".join(labels[:10]) + more)
    statement = "; ".join(parts) or _text(item.get("statement")) or "the source states no licence, size or labels"
    candidate = DatasetCandidate(hub_id, _text(item.get("title")) or hub_id, url, licence, n, labels)
    assertions = {}
    if licence:
        assertions["licence"] = licence
    if n is not None:
        assertions["n_examples"] = n
    if labels:
        assertions["labels"] = list(labels)
    return candidate, Claim(hub_id, "dataset", statement, url, source_name=source_name, assertions=assertions)


def _text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


# ----------------------------------------------------------------------------- fetching a sample
def fetch_sample(candidate: DatasetCandidate, dest: Path, scope: str, *,
                 requested_labels: list[str] | None = None) -> FetchReport:
    dest = Path(dest)
    shutil.rmtree(dest, ignore_errors=True)  # dest belongs to this fetch: an older sample is replaced
    try:
        dest.mkdir(parents=True, exist_ok=True)
        requested = _label_keys(requested_labels or [])
        if len(requested) > MAX_LABELS:
            raise DataUnavailable(f"{len(requested)} requested labels exceed the {MAX_LABELS}-label sample cap")
        return _fetch(candidate, dest, scope, requested)
    except research.ResearchUnavailable as exc:
        shutil.rmtree(dest, ignore_errors=True)
        raise DataUnavailable(str(exc)) from exc
    except DataUnavailable:
        shutil.rmtree(dest, ignore_errors=True)
        raise
    except OSError as exc:
        shutil.rmtree(dest, ignore_errors=True)
        raise DataUnavailable(f"could not write the sample under {dest}: {exc}") from exc


def _fetch(candidate: DatasetCandidate, dest: Path, scope: str, requested: list[str]) -> FetchReport:
    start, hub_id = _now(), candidate.hub_id
    split = _pick_split(research.viewer("splits", dataset=hub_id, timeout=min(25.0, _remaining(start))), hub_id)
    params = {"dataset": hub_id, "config": split["config"], "split": split["split"]}
    first = research.viewer("rows", **params, offset=0, length=PAGE_ROWS, timeout=min(25.0, _remaining(start)))
    audio_col, label_col, names = _columns(first.get("features"), hub_id)
    total = first.get("num_rows_total")
    total = total if isinstance(total, int) and total >= 0 else len(_row_entries(first))
    # Reserve capacity and download priority for named categories, even if they appear last.
    rows: dict[str, list[tuple[int, str]]] = {label: [] for label in requested}
    seen: set[int] = set()
    texts: dict[str, object] = {}  # safe folder -> original label identity
    _collect(first, audio_col, label_col, names, rows, seen, texts, named=bool(requested))
    for offset in _offsets(total)[1:]:
        if _now() - start > DEADLINE_S:
            break
        page = research.viewer("rows", **params, offset=offset, length=PAGE_ROWS, timeout=min(25.0, _remaining(start)))
        _collect(page, audio_col, label_col, names, rows, seen, texts, named=bool(requested))
    short = [label for label in requested if len(rows[label]) < MIN_PER_LABEL]
    if short:
        raise DataUnavailable(f"requested labels absent or too small in bounded viewer sample: {short}")
    queues = {label: _spread(sorted(found), PER_LABEL) for label, found in rows.items()}
    counts, sizes = _download_all(queues, dest, start)
    report = _finish(candidate, dest, scope, counts, sizes, total)
    require_coverage(requested, list(report.labels))
    report.selected_split = {"config": split["config"], "split": split["split"], "n_rows": total}
    return report


def _pick_split(answer: dict, hub_id: str) -> dict:
    raw = answer.get("splits")
    if not isinstance(raw, list):
        raise DataUnavailable(f"the dataset viewer returned invalid splits for {hub_id}")
    splits = [
        s for s in raw
        if isinstance(s, dict) and isinstance(s.get("config"), str) and isinstance(s.get("split"), str)
    ]
    if not splits:
        raise DataUnavailable(f"the dataset viewer lists no splits for {hub_id}")
    return next((s for s in splits if s["split"] == "train"), splits[0])


def _columns(features: object, hub_id: str) -> tuple[str, str, list[str] | None]:
    """(audio column, label column, ClassLabel names or None for a string label column)."""
    features = features if isinstance(features, list) else []
    typed = [f for f in features if isinstance(f, dict) and isinstance(f.get("type"), dict)]
    typed = [f for f in typed if isinstance(f.get("name"), str)]
    audio = next((f["name"] for f in typed if f["type"].get("_type") == "Audio"), None)
    if audio is None:
        raise DataUnavailable(f"{hub_id} has no audio column (no feature of type Audio) in the dataset viewer")
    for f in typed:
        names = f["type"].get("names")
        if f["type"].get("_type") == "ClassLabel" and isinstance(names, list):
            return audio, f["name"], [str(n) for n in names]
    for f in typed:
        if f["name"] in _LABEL_COLUMNS and f["type"].get("dtype") in ("string", "large_string"):
            return audio, f["name"], None
    raise DataUnavailable(
        f"{hub_id} has no label column: no ClassLabel feature and no text column named "
        + ", ".join(_LABEL_COLUMNS)
    )


def _offsets(total: int) -> list[int]:
    """Page offsets spread evenly over the split; one page when it fits in one."""
    if total <= PAGE_ROWS:
        return [0]
    return sorted({int(k * total / MAX_PAGES) for k in range(MAX_PAGES)})


def _collect(page: dict, audio_col: str, label_col: str, names: list[str] | None,
             rows: dict[str, list[tuple[int, str]]], seen: set[int], texts: dict[str, object], *, named: bool = False) -> None:
    for entry in _row_entries(page):
        if not isinstance(entry, dict):
            continue
        idx, row = entry.get("row_idx"), entry.get("row")
        if not isinstance(idx, int) or isinstance(idx, bool) or idx < 0 or idx in seen or not isinstance(row, dict):
            continue
        seen.add(idx)
        label, src = _label(row.get(label_col), names, texts, named=named), _src(row.get(audio_col))
        if label is None or src is None:
            continue
        if label not in rows:
            if len(rows) >= MAX_LABELS:
                continue
            rows[label] = []
        rows[label].append((idx, src))


def _row_entries(page: dict) -> list:
    entries = page.get("rows")
    if not isinstance(entries, list):
        raise DataUnavailable("the dataset viewer returned invalid rows (expected a list)")
    return entries


def _label(cell: object, names: list[str] | None, texts: dict[str, object], *, named: bool = False) -> str | None:
    if names is not None and isinstance(cell, int) and not isinstance(cell, bool):
        if not 0 <= cell < len(names):
            return None
        raw, index = names[cell], cell
    elif isinstance(cell, str) and cell.strip():
        raw = cell
        index = next((i for i, value in enumerate(texts.values()) if value == raw), len(texts))
    else:
        return None
    label = safe_label(raw, index)
    # A filesystem-safe name is not evidence of the original category. For named
    # goals, reject source identities the folder representation would discard.
    if named and _label_keys([raw])[0] != label:
        raise DataUnavailable(f"requested coverage cannot represent source label {raw!r} without information loss")
    # Distinct source labels must never silently share a folder, including after truncation.
    identity = (cell if names is not None else raw)
    previous = texts.setdefault(label, identity)
    if previous != identity:
        raise DataUnavailable(f"dataset labels collide in sample folder {label!r}")
    return label


def _src(cell: object) -> str | None:
    """The audio URL of a viewer cell: a dict with "src", or a one-item list of one."""
    if isinstance(cell, list) and len(cell) == 1:
        cell = cell[0]
    src = cell.get("src") if isinstance(cell, dict) else None
    return src if isinstance(src, str) and src.startswith(("http://", "https://")) else None


def _spread(items: list, k: int) -> list:
    """k items evenly spaced through the list first, then the rest as spares for failed downloads."""
    if len(items) <= k:
        return list(items)
    picks = sorted({int(i * len(items) / k) for i in range(k)})
    chosen = set(picks)
    return [items[j] for j in picks] + [x for j, x in enumerate(items) if j not in chosen]


def _download_all(queues: dict[str, list[tuple[int, str]]], dest: Path, start: float) -> tuple[dict, dict]:
    """One file per label in turn until each label is full or out of rows, or the byte or time budget is spent."""
    pending = {label: deque(queue) for label, queue in queues.items()}
    counts, sizes = dict.fromkeys(pending, 0), dict.fromkeys(pending, 0)
    spent = 0
    while True:
        progressed = False
        for label, queue in pending.items():
            if counts[label] >= PER_LABEL or not queue:
                continue
            if spent >= MAX_BYTES or _now() - start > DEADLINE_S:
                return counts, sizes
            idx, src = queue.popleft()
            progressed = True
            n = _download_one(src, dest, label, idx, MAX_BYTES - spent, min(20.0, _remaining(start)))
            if n is not None:
                counts[label], sizes[label], spent = counts[label] + 1, sizes[label] + n, spent + n
        if not progressed:
            return counts, sizes


def _remaining(start: float) -> float:
    remaining = DEADLINE_S - (_now() - start)
    if remaining <= 0:
        raise DataUnavailable("the sample deadline was reached")
    return remaining


def _download_one(src: str, dest: Path, label: str, idx: int, budget: int, timeout: float) -> int | None:
    try:
        target = dest / label / f"{idx}.{_extension(src)}"
        if not target.resolve().is_relative_to(dest.resolve()):
            return None
        return research.download(src, target, timeout=timeout, max_bytes=min(FILE_MAX_BYTES, budget))
    except research.ResearchUnavailable as exc:
        if getattr(exc, "bytes_received", 0) or getattr(exc, "transfer_incomplete", False):
            # Abort after any partially transferred failure: discarded bytes cannot fuel
            # repeated oversized or broken transfers outside the sample's byte budget.
            raise DataUnavailable(f"sample transfer stopped after {getattr(exc, "bytes_received", 0)} discarded bytes: {exc}") from exc
        return None
    except (OSError, ValueError):
        return None


def _extension(url: str) -> str:
    suffix = PurePosixPath(urllib.parse.urlsplit(url).path).suffix.lower()
    return suffix[1:] if suffix in AUDIO_EXTENSIONS else "wav"


def _finish(candidate: DatasetCandidate, dest: Path, scope: str, counts: dict, sizes: dict, total: int) -> FetchReport:
    kept = {}
    for label, n in counts.items():
        if n >= MIN_PER_LABEL:
            kept[label] = n
        else:
            shutil.rmtree(dest / label, ignore_errors=True)
    need = 3 if scope == "open" else 2
    if len(kept) < need:
        fetched = sum(counts.values())
        raise DataUnavailable(
            f"only {_count(len(kept), 'label')} of {candidate.hub_id} had at least {MIN_PER_LABEL} files "
            f"({_count(fetched, 'file')} fetched in all), and scope {scope} needs {need}"
        )
    return FetchReport(dest, candidate.hub_id, kept, sum(kept.values()), total, candidate.observed.get("licence"),
                       sum(sizes[label] for label in kept))


def _count(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


# ----------------------------------------------------------------------------- labels, pre-wired sets, building a set
def safe_label(text: str, index: int) -> str:
    """Only a-z, 0-9 and inner "_", at most 40 characters, never empty: so never ".", ".." or a path."""
    out = re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")[:40].strip("_")
    return out or f"label_{index}"


def _label_keys(labels: list[str]) -> list[str]:
    keys, originals = [], {}
    for label in labels:
        # Only case and the documented word separators are interchangeable.
        # Arbitrary punctuation and non-ASCII names cannot be safely compared
        # through safe_label's lossy replacement or generated label_N fallback.
        if (not isinstance(label, str) or not re.fullmatch(r"[a-zA-Z0-9_\s-]+", label)
                or not re.search(r"[a-zA-Z0-9]", label)):
            raise DataUnavailable(f"requested or available category is not representable without information loss: {label!r}")
        # Coverage also must not collapse long names through the folder length cap.
        key = re.sub(r"[_\s-]+", "_", label.lower()).strip("_")
        previous = originals.setdefault(key, label)
        if previous != label:
            raise DataUnavailable(f"requested or available labels collide after normalization: {previous!r}, {label!r}")
        if key not in keys:
            keys.append(key)
    return keys


def require_coverage(requested: list[str], actual: list[str]) -> None:
    if not requested:
        return
    missing = set(_label_keys(requested)) - set(_label_keys(actual))
    if missing:
        raise DataUnavailable(f"requested labels missing from usable set: {', '.join(sorted(missing))}")


def choose_prewired(goal: str, modality: str, *, requested_labels: list[str] | None = None) -> str | None:
    wanted = _words(goal)
    best, best_score = None, 0
    for entry in datasets.available_sets():
        if entry.get("modality") != modality:
            continue
        try:
            require_coverage(requested_labels or [], entry.get("labels") or [])
        except DataUnavailable:
            continue
        text = " ".join([str(entry.get("title") or "")] + [str(label) for label in entry.get("labels") or []])
        score = len(wanted & _words(text))
        if score > best_score:
            best, best_score = entry.get("name"), score
    return best


def _words(text: str) -> set[str]:
    """Distinct lower-case words of 3 or more letters, with a trailing "s" dropped."""
    return {w[:-1] if w.endswith("s") else w for w in re.findall(r"[a-z0-9]+", str(text).lower()) if len(w) >= 3}


def build_set(folder: Path, name: str, *, title: str, question: str, modality: str, source: str, licence: str) -> str:
    from_folder = getattr(datasets, "from_folder", None)
    if from_folder is None:
        raise DataUnavailable("this build cannot build a set from a folder yet")
    try:
        ds = from_folder(folder, name, title=title, question=question, modality=modality,
                         max_per_label=PER_LABEL, min_per_label=MIN_PER_LABEL, source=source, licence=licence)
    except ValueError as exc:
        raise DataUnavailable(str(exc)) from exc
    return ds.name
