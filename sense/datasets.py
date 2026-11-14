"""Labelled example sets with cached specialist embeddings.

Public surface (do not rename):
  available_sets() -> list[dict]                 [{"name", "title", "modality", "labels", "n"}], only sets present on disk
  load_labelled_set(name) -> LabelledSet         "gtzan" (10 music genres) or "esc50" (50 environmental sounds)
  caption_bank() -> tuple[np.ndarray, list[str]] specialist text embeddings and the caption texts, for the text bridge
  sample_files(name, per_label=1) -> list[dict]  [{"path", "label"}] from the held-out split, when raw files are on disk

Embeddings come from config.data_dir(): esc50_clap.npz, gtzan_clap.npz, audiocaps_text_clap.npz, written by the
scripts in socket_pilot/ (keys documented in socket_pilot/clap_embed.py, clap_embed_gtzan.py, clap_text.py).
Splits: esc50 uses its folds (1-3 fit, 4 dev, 5 test); gtzan uses a seeded 60/20/20 split per genre, as in
socket_pilot/run_music.py. In fake mode both sets are small synthetic ones with separable classes.
"""

from __future__ import annotations

import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np

from sense import config
from sense.contracts import LabelledSet
from sense.specialists import FAKE_CONCEPTS, FAKE_LABELS, FakeSpecialist  # the shared fake world

_SETS = {
    "gtzan": {
        "file": "gtzan_clap.npz",
        "title": "Music genres (GTZAN)",
        "question": "Which genre of music is in the audio clip?",
        "script": "socket_pilot/clap_embed_gtzan.py",
    },
    "esc50": {
        "file": "esc50_clap.npz",
        "title": "Environmental sounds (ESC-50)",
        "question": "Which sound is in the audio clip?",
        "script": "socket_pilot/clap_embed.py",
    },
}
_DISPLAY = {"hiphop": "hip hop"}
_CAPTIONS_FILE = "audiocaps_text_clap.npz"
_FAKE_PER_LABEL = 30


def available_sets() -> list[dict]:
    out = []
    for name in _SETS:
        if not config.fake() and not (config.data_dir() / _SETS[name]["file"]).is_file():
            continue
        ds = load_labelled_set(name)
        out.append({"name": ds.name, "title": ds.title, "modality": ds.modality, "labels": list(ds.labels), "n": len(ds.y)})
    return out


def load_labelled_set(name: str) -> LabelledSet:
    if name not in _SETS:
        raise ValueError(f"Unknown dataset {name!r}. Choose one of: {', '.join(_SETS)}.")
    if config.fake():
        return _fake_set(name)
    return _real_set(name, str(config.data_dir()), tuple(map(str, config.sample_dirs())))


def caption_bank() -> tuple[np.ndarray, list[str]]:
    if config.fake():
        return _fake_captions()
    return _real_captions(str(config.data_dir()))


def sample_files(name: str, per_label: int = 1) -> list[dict]:
    ds = load_labelled_set(name)
    picks: list[int] = []
    for c in range(len(ds.labels)):
        picks += [int(i) for i in ds.test_idx[ds.y[ds.test_idx] == c][:per_label]]
    if config.fake():
        return [{"path": str(_fake_sample(ds, i)), "label": ds.labels[ds.y[i]]} for i in picks]
    if ds.files is None:
        return []
    return [{"path": ds.files[i], "label": ds.labels[ds.y[i]]} for i in picks]


# ----------------------------------------------------------------------------- real data
@lru_cache(maxsize=None)
def _real_set(name: str, data_dir: str, sample_dirs: tuple[str, ...]) -> LabelledSet:
    meta = _SETS[name]
    path = Path(data_dir) / meta["file"]
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is missing. Write it with {meta['script']}, or set SENSE_DATA_DIR to the directory that holds it."
        )
    data = np.load(path, allow_pickle=False)
    raw_names = [str(n) for n in data["names"]]
    y = data["y"].astype(np.int64)
    if name == "esc50":
        fold = data["fold"]
        fit_idx, dev_idx, test_idx = (np.where(fold <= 3)[0], np.where(fold == 4)[0], np.where(fold == 5)[0])
    else:
        fit_idx, dev_idx, test_idx = _split_per_label(y, len(raw_names), seed=0)
    files = _resolve_files(
        [str(f) for f in data["files"]],
        [raw_names[c] if name == "gtzan" else "" for c in y],
        [Path(d) for d in sample_dirs],
    )
    return LabelledSet(
        name=name,
        title=meta["title"],
        modality="audio",
        question=meta["question"],
        labels=[_DISPLAY.get(n, n) for n in raw_names],
        y=y,
        emb=data["emb"].astype(np.float32),
        label_text_emb=data["text_emb"].astype(np.float32),
        fit_idx=fit_idx,
        dev_idx=dev_idx,
        test_idx=test_idx,
        files=files,
    )


def _split_per_label(y: np.ndarray, n_labels: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Seeded 60/20/20 split within each label, as in socket_pilot/run_music.py."""
    rng = np.random.default_rng(seed)
    fit, dev, test = [], [], []
    for c in range(n_labels):
        order = rng.permutation(np.where(y == c)[0])
        n_fit, n_dev = int(round(0.6 * len(order))), int(round(0.2 * len(order)))
        fit += list(order[:n_fit])
        dev += list(order[n_fit : n_fit + n_dev])
        test += list(order[n_fit + n_dev :])
    return np.array(fit, dtype=np.int64), np.array(dev, dtype=np.int64), np.array(test, dtype=np.int64)


def _resolve_files(names: list[str], subdirs: list[str], roots: list[Path]) -> list[str] | None:
    """Absolute paths of the raw files, looked up as <root>/<subdir>/<name> or <root>/<name>; None unless all exist."""
    out = []
    for name, sub in zip(names, subdirs):
        for root in roots:
            hit = next((p for p in (root / sub / name, root / name) if p.is_file()), None)
            if hit is not None:
                out.append(str(hit.resolve()))
                break
        else:
            return None
    return out


@lru_cache(maxsize=None)
def _real_captions(data_dir: str) -> tuple[np.ndarray, list[str]]:
    path = Path(data_dir) / _CAPTIONS_FILE
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is missing. Write it with socket_pilot/clap_text.py, or set SENSE_DATA_DIR to the directory "
            "that holds it."
        )
    data = np.load(path, allow_pickle=False)
    return data["text_emb"].astype(np.float32), [str(t) for t in data["texts"]]


# ----------------------------------------------------------------------------- fake data
@lru_cache(maxsize=None)
def _fake_set(name: str) -> LabelledSet:
    labels = FAKE_LABELS[name]
    spec = FakeSpecialist()
    rng = np.random.default_rng(7 if name == "gtzan" else 11)
    y = np.repeat(np.arange(len(labels)), _FAKE_PER_LABEL)
    emb = np.stack([spec.example(FAKE_CONCEPTS.index(labels[c]), rng) for c in y])
    if name == "esc50":  # five folds per label, as in the real set
        fold = np.tile(np.repeat(np.arange(1, 6), _FAKE_PER_LABEL // 5), len(labels))
        fit_idx, dev_idx, test_idx = (np.where(fold <= 3)[0], np.where(fold == 4)[0], np.where(fold == 5)[0])
        phrases = [f"the sound of {n}" for n in labels]
    else:
        fit_idx, dev_idx, test_idx = _split_per_label(y, len(labels), seed=0)
        phrases = [f"{n} music" for n in labels]
    meta = _SETS[name]
    return LabelledSet(
        name=name,
        title=meta["title"].replace(")", ", synthetic)"),
        modality="audio",
        question=meta["question"],
        labels=list(labels),
        y=y,
        emb=emb,
        label_text_emb=spec.embed_texts(phrases),
        fit_idx=fit_idx,
        dev_idx=dev_idx,
        test_idx=test_idx,
        files=None,
    )


@lru_cache(maxsize=1)
def _fake_captions() -> tuple[np.ndarray, list[str]]:
    """Captions for every fake sound but only three music genres: like AudioCaps, the bank is weak on genre."""
    covered = FAKE_LABELS["esc50"] + ["classical", "jazz", "rock"]
    texts = []
    for c in covered:
        texts += [f"the sound of {c}", f"{c} heard from across a quiet room"]
    texts += [
        "people talking in a crowded hall",
        "wind blowing through trees",
        "a car engine idling",
        "footsteps on gravel",
    ]
    return FakeSpecialist().embed_texts(texts), texts


def _fake_sample(ds: LabelledSet, i: int) -> Path:
    """A fake audio file for example i of a fake set, written once into the temporary directory."""
    label = ds.labels[ds.y[i]]
    path = Path(tempfile.gettempdir()) / "sense-fake-samples" / ds.name / f"{label.replace(' ', '_')}.{i:04d}.wav"
    if not path.is_file():
        FakeSpecialist.write_sample(FAKE_CONCEPTS.index(label), path, seed=i)
    return path
