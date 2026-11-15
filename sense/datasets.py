"""Labelled example sets with cached specialist embeddings.

Public surface (do not rename):
  available_sets() -> list[dict]                 [{"name", "title", "modality", "labels", "n", "origin", "meta"}]:
                                                 the built-in sets present on disk ("origin": "builtin"; "meta" is {}
                                                 except for real gtzan, which notes its duplicates),
                                                 then every folder set ("origin": "folder", "meta": LabelledSet.meta)
  load_labelled_set(name) -> LabelledSet         "gtzan" (10 music genres), "esc50" (50 environmental sounds), or
                                                 the name of a folder set built earlier with from_folder
  from_folder(path, name=None, *, title=None, question=None, modality="audio", max_per_label=50,
              min_per_label=5, seed=0, source="", licence="") -> LabelledSet
                                                 build a set from a folder with one sub-folder of audio files per
                                                 label, save it under config.sets_dir() and return it
  caption_bank() -> tuple[np.ndarray, list[str]] specialist text embeddings and the caption texts, for the text bridge
  sample_files(name, per_label=1) -> list[dict]  [{"path", "label"}] from the held-out split, when raw files are on disk
  fit_problem(ds, scope="closed", *, open_weights=True) -> str | None
                                                 why an agent run on this set and scope would stop before verifying
                                                 anything, or None; the server and scripts/demo_cli.py check it first.
                                                 open_weights=False (ModelInfo.open_weights of an API model, which
                                                 gets no vector bridge) skips the vector bridge's fit checks

Built-in sets. Embeddings come from config.data_dir(): esc50_clap.npz, gtzan_clap.npz, audiocaps_text_clap.npz,
written by the scripts in socket_pilot/ (keys documented in socket_pilot/clap_embed.py, clap_embed_gtzan.py,
clap_text.py). Splits: esc50 uses its folds (1-3 fit, 4 dev, 5 test); gtzan uses a seeded 60/20/20 split per
genre, as in socket_pilot/run_music.py, except that a recording GTZAN holds twice stays in one split: its 14
duplicate recordings (found by file hash and by embedding cosine >= 0.999) go to one split whole, and the one
filed under two genres (metal.00058 and rock.00016) is left out of every split. meta["left_out"] names it and
meta["duplicate_groups_kept_in_one_split"] counts the rest. ESC-50 has no duplicates. In fake mode both sets are
small synthetic ones with separable classes.

Folder sets (from_folder). Each immediate sub-folder of the folder is one label; sub-folders whose name starts
with "." or "_" (such as __MACOSX) are ignored. The label is the folder name with "_" and "-" read as spaces
("sea_waves" is "sea waves"); two folders that give the same label are an error. Only files directly inside a
label folder are read: audio files (specialists.AUDIO_EXTENSIONS) are used, other files are counted in meta and
ignored, and hidden files are skipped. Then, in order:
  1. Exact duplicates, by SHA-1 of the bytes. Within a label the first copy by path is kept and the rest are
     counted. The same bytes under two or more labels is a labelling conflict: every copy is dropped and listed.
  2. At most max_per_label files per label, a seeded pick, so the embedding pass is bounded.
  3. The specialist embeds the files in batches; a file it cannot read is skipped and listed in meta.
  4. Every label needs at least min_per_label usable files; min_per_label is at least 4, so that each label can
     have fit, dev and test examples. A set needs at least 2 labels. A short label is reported with the reasons
     its other files were not used, and this is checked before embedding too, so no time is spent on a folder
     that cannot make a set.
  5. Near duplicates: clips of one label whose specialist embeddings have cosine >= 0.999 form a group (joined
     transitively), and a group always goes to one split whole. Near duplicates across labels are counted, not
     dropped.
  6. Split per label, 60/20/20 by groups: a seeded permutation of the groups fills fit up to round(0.6 n) clips,
     then dev up to round(0.2 n), and the rest go to test, always leaving at least one group each for dev and
     test. A label with fewer than 3 distinct groups cannot be split, which is an error. So is a split that
     leaves fewer than 5 examples to fit on in all, the least a vector bridge (sense.bridges) fits on.
Storage: <config.sets_dir()>/<name>/set.json (name, title, question, modality, labels, absolute file paths, y,
the three splits, meta, fingerprint, format) and set.npz (emb, label_text_emb). config.sets_dir() is
SENSE_SETS_DIR, by default <SENSE_HOME>/datasets. A set is written to a temporary directory and then moved into
place, so a reader never sees a half-written set. Saving replaces only a saved set: a source folder inside
config.sets_dir() is refused, and so is a name whose directory there holds anything but set.json and set.npz.
The fingerprint covers the folder, the names, sizes and modification times of the audio files in its label
folders, max_per_label, min_per_label, seed, question, modality and the specialist ("fake" in fake mode, else
config.model_source("clap")). Calling from_folder again with the same fingerprint loads the saved set without
embedding again, and a title, source or licence left out of the call keeps the saved one, so a later call from
the interface does not erase what the agent recorded. A different fingerprint rebuilds and replaces the set: a
source or licence left out still keeps the saved one when the saved set was built from the same folder, but a
title left out becomes the default "<folder> (folder, N labels)" again, since the label folders may have changed;
pass the title again to keep it. Folder sets load in fake and real mode alike;
meta["specialist"] says which specialist embedded them, and fit_problem refuses a run on a set embedded in the
other mode.

Scope "open" needs a set with at least 3 labels when the run builds a vector bridge (an open-weight model): its
unseen-classes control (sense.verify) refits the vector bridge with a fifth of the labels, at least one, left out,
and a bridge needs at least 2 labels and 5 examples to fit on. fit_problem checks both, assuming the labels with
the most fit examples are the ones left out. A model that is not open-weight gets only text and tool bridges,
which have no unseen-classes control, so fit_problem(..., open_weights=False) skips these checks.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
import threading
import time
import uuid
from dataclasses import replace
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np

from sense import config
from sense.contracts import LabelledSet
from sense.specialists import FAKE_CONCEPTS, FAKE_LABELS, FakeSpecialist  # the shared fake world
from sense.specialists import AUDIO_EXTENSIONS, Specialist, get_specialist

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
        out.append(_entry(ds, "builtin"))
    with _swap_lock:  # a set that from_folder is replacing is listed, not briefly missing
        for name in _folder_set_names():
            try:
                ds = _load_folder_set(name)
            except Exception as err:  # one damaged folder set must not hide the others
                _log.warning("Leaving the folder set %r out of the list: %s", name, err)
                continue
            out.append(_entry(ds, "folder"))
    return out


def load_labelled_set(name: str) -> LabelledSet:
    if name not in _SETS:
        with _swap_lock:  # a set that from_folder is replacing is read whole, not briefly missing
            if _is_folder_set(name):
                return _load_folder_set(name)
        raise _unknown(name)
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
    if ds.files is not None:  # real built-in sets with their raw files, and every folder set, in either mode
        return [{"path": ds.files[i], "label": ds.labels[ds.y[i]]} for i in picks if Path(ds.files[i]).is_file()]
    if config.fake():  # the synthetic built-in sets: write a fake audio file for each pick
        return [{"path": str(_fake_sample(ds, i)), "label": ds.labels[ds.y[i]]} for i in picks]
    return []


def fit_problem(ds: LabelledSet, scope: str = "closed", *, open_weights: bool = True) -> str | None:
    """Why an agent run on this set and scope would stop before it can verify anything, or None.

    It checks that a folder set was embedded by the specialist of this mode (fake or real). When the run builds a
    vector bridge, that is for an open-weight model (`open_weights`, ModelInfo.open_weights; the default), it also
    checks that the set has the 5 fit examples a vector bridge needs (sense.bridges.fit_vector_bridge), and for
    scope "open" the unseen-classes control (sense.verify), which refits the vector bridge with a fifth of the
    labels, at least one, left out: that needs 3 labels or more, and 5 fit examples even when the labels left out
    are the ones with the most fit examples. A model that is not open-weight gets only text and tool bridges,
    which fit on nothing and have no unseen-classes control, so neither check applies to it."""
    specialist = ds.meta.get("specialist")
    if specialist and (specialist == "fake") != config.fake():
        return (
            f"The set {ds.name} was embedded by the {specialist} specialist, but the harness is in "
            f"{'fake' if config.fake() else 'real'} mode, so its embeddings do not match the specialist in use. "
            "Build it again in this mode (POST /api/datasets, or scripts/demo_cli.py --folder)."
        )
    if not open_weights:
        return None
    n_fit = np.bincount(np.asarray(ds.y)[ds.fit_idx], minlength=len(ds.labels))
    if n_fit.sum() < _MIN_FIT:
        return (
            f"The set {ds.name} has {int(n_fit.sum())} examples to fit on, but a vector bridge needs at least "
            f"{_MIN_FIT}. Build it again from a folder with more distinct recordings."
        )
    if scope == "open":
        if len(ds.labels) < 3:
            return (
                "Scope 'open' needs a set with at least 3 labels, because its unseen-classes check refits the bridge "
                f"with some labels left out; {ds.name} has {len(ds.labels)}. Use scope 'closed', or add label "
                "folders and build the set again."
            )
        held = max(1, len(ds.labels) // 5)
        left = int(n_fit.sum() - np.sort(n_fit)[::-1][:held].sum())
        if left < _MIN_FIT:
            return (
                f"Scope 'open' needs more examples to fit on in {ds.name}: its unseen-classes check refits the bridge "
                f"with {held} label{'' if held == 1 else 's'} left out, which can leave {left} examples, and a vector "
                f"bridge needs at least {_MIN_FIT}. Use scope 'closed', or add recordings to the label folders and "
                "build the set again."
            )
    return None


def _unknown(name: str) -> ValueError:
    choices = ", ".join([*_SETS, *_folder_set_names()])
    return ValueError(
        f"Unknown dataset {name!r}. Choose one of: {choices}; or build a set from a folder of labelled audio "
        "files with sense.datasets.from_folder."
    )


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
        set_meta: dict = {}
    else:
        fit_idx, dev_idx, test_idx, set_meta = _split_without_duplicates(data["emb"], y, len(raw_names), seed=0)
        set_meta["left_out"] = [str(data["files"][i]) for i in set_meta.pop("left_out_idx")]
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
        meta=set_meta,
    )


def _split_without_duplicates(
    emb: np.ndarray, y: np.ndarray, n_labels: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """_split_per_label, but a recording that occurs twice cannot sit in two splits. Clips whose embeddings are
    near-identical (cosine >= _NEAR_DUPLICATE: the same recording, as from_folder treats it) form a group that goes
    to one split whole, and a clip near-identical to a clip of another label is left out of every split, since its
    label is in doubt. With no such clips this returns exactly what _split_per_label returns for the same seed.
    GTZAN holds 14 duplicate recordings, one of them filed under two genres (metal.00058 and rock.00016)."""
    unit = _unit_rows(emb)
    y = np.asarray(y)
    across = np.triu((unit @ unit.T >= _NEAR_DUPLICATE) & (y[:, None] != y[None, :]), k=1)
    left_out = sorted({int(i) for pair in zip(*np.nonzero(across)) for i in pair})
    rng = np.random.default_rng(seed)
    fit: list[int] = []
    dev: list[int] = []
    test: list[int] = []
    kept_together = 0
    for c in range(n_labels):
        members = np.array([i for i in np.where(y == c)[0] if i not in left_out], dtype=np.int64)
        groups = [[int(members[k]) for k in g] for g in _near_duplicate_groups(unit[members])]
        kept_together += sum(len(g) > 1 for g in groups)
        f, d, t = _split_groups(groups, len(members), rng)
        fit += f
        dev += d
        test += t
    info = {"duplicate_groups_kept_in_one_split": kept_together, "left_out_idx": left_out}
    return np.array(fit, dtype=np.int64), np.array(dev, dtype=np.int64), np.array(test, dtype=np.int64), info


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


# ----------------------------------------------------------------------------- folder sets
_FOLDER_FORMAT = 1  # version of set.json and set.npz; a saved set of another version is built again
_FOLDER_QUESTION = "Which sound is in the audio clip?"
_MIN_PER_LABEL = 4  # one fit, one dev and one test example need at least this many (60/20/20 of 4 is 2/1/1)
_MIN_FIT = 5  # the fewest fit examples sense.bridges.fit_vector_bridge accepts
_SAVED_FILES = {"set.json", "set.npz"}  # all a saved folder set's directory holds
_READ_TYPES = ", ".join(sorted(AUDIO_EXTENSIONS))
_EMBED_BATCH = 16
_NEAR_DUPLICATE = 0.999  # cosine of L2-normalised specialist embeddings at or above which two clips are one group
_MAX_LISTED = 20  # entries kept in meta["skipped"]; meta["n_skipped"] keeps the full count
_SET_NAME = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_RESERVED = {"captions"}  # "text:captions" names the caption-bank text bridge; a set of that name would clash
_log = logging.getLogger(__name__)
# Held while a set is moved into place and while folder sets are found and read, so that within this process a
# reader never finds a set missing while it is being replaced.
_swap_lock = threading.RLock()


def from_folder(
    path: str | Path,
    name: str | None = None,
    *,
    title: str | None = None,
    question: str | None = None,
    modality: str = "audio",
    max_per_label: int = 50,
    min_per_label: int = 5,
    seed: int = 0,
    source: str = "",
    licence: str = "",
) -> LabelledSet:
    """Build a labelled set from a folder with one sub-folder of audio files per label, save it under
    config.sets_dir()/<name> and return it. If a set of that name was saved from the same files with the same
    settings, it is loaded instead of embedded again; a title, source or licence given here replaces the saved
    one, and one left out keeps it. A rebuild (other files or settings) keeps a saved source and licence from the
    same folder but not the title, which is the default unless given. Raises ValueError, saying what to change,
    when the folder cannot make a usable set. The rules are in the module docstring; `source` and `licence` are
    recorded in meta for whoever reads the set later."""
    if not str(path).strip():
        raise ValueError(
            "No folder was given. Pass the path of a folder that holds one sub-folder of audio files per label."
        )
    root = Path(path).expanduser()
    if not root.is_dir():
        raise ValueError(
            f"{path} is not an existing folder. Pass the path of a folder that holds one sub-folder of audio "
            "files per label."
        )
    root = root.resolve()
    sets = config.sets_dir().expanduser().resolve()
    if root == sets or sets in root.parents:
        raise ValueError(
            f"{root} lies inside SENSE_SETS_DIR ({sets}), where the harness saves the sets it builds, so saving a "
            "set could overwrite these files. Move the folder out of it, or point SENSE_SETS_DIR somewhere else."
        )
    name = _set_name(root.name if name is None else name)
    if _not_a_saved_set(config.sets_dir() / name):
        raise _foreign_error(config.sets_dir() / name)
    max_per_label, min_per_label, seed = int(max_per_label), int(min_per_label), int(seed)
    if min_per_label < _MIN_PER_LABEL:
        raise ValueError(
            f"min_per_label is {min_per_label}, but every label needs at least {_MIN_PER_LABEL} usable files so that "
            f"it has fit, dev and test examples. Use min_per_label={_MIN_PER_LABEL} or more."
        )
    if max_per_label < min_per_label:
        raise ValueError(
            f"max_per_label ({max_per_label}) is below min_per_label ({min_per_label}), so no label could have "
            "enough files. Raise max_per_label or lower min_per_label."
        )
    if seed < 0:
        raise ValueError(f"seed is {seed}. Use a whole number of 0 or more.")
    specialist = get_specialist(modality)  # ValueError for a modality without a specialist; loads no weights
    question = (question or "").strip() or _FOLDER_QUESTION
    folders = _label_folders(root)
    listings = [_list_label_folder(directory) for _, directory in folders]
    settings = {
        "max_per_label": max_per_label,
        "min_per_label": min_per_label,
        "seed": seed,
        "question": question,
        "modality": modality,
        "specialist": _specialist_id(),
    }
    fingerprint = _fingerprint(root, folders, listings, settings)
    title = (title or "").strip()
    previous = _saved_meta(name, root)  # what was recorded for this folder under this name, if anything
    source, licence = source or previous.get("source", ""), licence or previous.get("licence", "")

    saved = _saved_set(name, fingerprint)
    if saved is not None:
        title = title or saved.title
        if (saved.title, saved.meta.get("source", ""), saved.meta.get("licence", "")) == (title, source, licence):
            return saved
        updated = replace(saved, title=title, meta={**saved.meta, "source": source, "licence": licence})
        _save_folder_set(updated, fingerprint)
        return updated

    title = title or f"{root.name} (folder, {len(folders)} labels)"
    ds = _build_folder_set(
        root, name, title, question, modality, specialist, folders, listings, settings, source, licence
    )
    _save_folder_set(ds, fingerprint)
    return ds


def _entry(ds: LabelledSet, origin: str) -> dict:
    """One item of available_sets()."""
    return {
        "name": ds.name,
        "title": ds.title,
        "modality": ds.modality,
        "labels": list(ds.labels),
        "n": len(ds.y),
        "origin": origin,
        "meta": copy.deepcopy(ds.meta),
    }


def _set_name(raw: str) -> str:
    """Lowercase; characters outside [a-z0-9-] become "-", runs of "-" collapse, and "-" is stripped from the ends."""
    name = re.sub(r"-+", "-", re.sub(r"[^a-z0-9-]", "-", str(raw).lower())).strip("-")
    if not name:
        raise ValueError(f"{raw!r} gives an empty set name. Pass a name made of letters, digits and '-'.")
    if name in _SETS:
        raise ValueError(f"The name {name!r} is taken by a built-in set. Pass another name.")
    if name in _RESERVED:
        raise ValueError(f"The name {name!r} is taken by the caption bank of the text bridge. Pass another name.")
    return name


def _is_folder_set(name: str) -> bool:
    return (
        isinstance(name, str)
        and _SET_NAME.fullmatch(name) is not None
        and name not in _SETS
        and name not in _RESERVED
        and (config.sets_dir() / name / "set.json").is_file()
    )


def _folder_set_names() -> list[str]:
    root = config.sets_dir()
    if not root.is_dir():
        return []
    return sorted(d.name for d in root.iterdir() if _is_folder_set(d.name))


def _specialist_id() -> str:
    return "fake" if config.fake() else config.model_source("clap")


def _entries(directory: Path) -> list[Path]:
    try:
        return sorted(directory.iterdir())
    except OSError as err:
        raise ValueError(
            f"Could not read the folder {directory} ({err.strerror or err}). Check that it exists and can be read."
        ) from err


def _label_folders(root: Path) -> list[tuple[str, Path]]:
    """(label, folder) for every label sub-folder of root, sorted by folder name."""
    out: list[tuple[str, Path]] = []
    seen: dict[str, str] = {}  # label, case ignored -> folder name
    for directory in _entries(root):
        if directory.name.startswith((".", "_")) or not directory.is_dir():
            continue
        label = " ".join(directory.name.replace("_", " ").replace("-", " ").split())
        if not label:
            raise ValueError(
                f"The folder {directory.name!r} in {root} gives an empty label. Rename it after its label."
            )
        key = label.casefold()
        if key in seen:
            raise ValueError(
                f"The folders {seen[key]!r} and {directory.name!r} in {root} both give the label {label!r}. "
                "Merge them into one folder, or rename one of them."
            )
        seen[key] = directory.name
        out.append((label, directory))
    if len(out) < 2:
        found = f" ({out[0][1].name!r})" if out else ""
        raise ValueError(
            f"{root} has {len(out)} label folder{'' if len(out) == 1 else 's'}{found}, but a labelled set needs at "
            "least 2. Put the audio files of each label in their own sub-folder, named after the label."
        )
    return out


def _list_label_folder(directory: Path) -> dict:
    """The audio files directly inside one label folder, the number and types of other files, and each audio file's
    (name, size, mtime_ns) for the fingerprint. Hidden files (such as .DS_Store) are skipped."""
    files = [p for p in _entries(directory) if not p.name.startswith(".") and p.is_file()]
    audio = [p for p in files if p.suffix.lower() in AUDIO_EXTENSIONS]
    other_types = sorted({p.suffix.lower() or "no extension" for p in files if p not in audio})
    stats = []
    for p in audio:
        st = p.stat()
        stats.append([p.name, st.st_size, st.st_mtime_ns])
    return {"audio": audio, "non_audio": len(files) - len(audio), "non_audio_types": other_types, "stats": stats}


def _fingerprint(root: Path, folders: list[tuple[str, Path]], listings: list[dict], settings: dict) -> str:
    files = sorted(
        [f"{directory.name}/{file_name}", size, mtime_ns]
        for (_, directory), listing in zip(folders, listings)
        for file_name, size, mtime_ns in listing["stats"]
    )
    payload = {
        "format": _FOLDER_FORMAT,
        "folder": str(root),
        "label_folders": [directory.name for _, directory in folders],
        "files": files,
        **settings,
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _saved_set(name: str, fingerprint: str) -> LabelledSet | None:
    """The saved folder set of this name if it was built with this fingerprint and can be read, else None."""
    meta_path = config.sets_dir() / name / "set.json"
    if not meta_path.is_file():
        return None
    try:
        if json.loads(meta_path.read_text()).get("fingerprint") != fingerprint:
            return None
        return _load_folder_set(name)
    except Exception as err:  # damaged: build it again, which replaces it
        _log.warning("The saved folder set %r cannot be read and will be built again: %s", name, err)
        return None


def _saved_meta(name: str, root: Path) -> dict:
    """meta of the set saved under this name if it was built from this folder and its set.json can be read, else {}."""
    try:
        meta = json.loads((config.sets_dir() / name / "set.json").read_text()).get("meta")
    except (OSError, ValueError, AttributeError):
        return {}
    return meta if isinstance(meta, dict) and meta.get("folder") == str(root) else {}


def _not_a_saved_set(target: Path) -> bool:
    """True if target exists and is not the directory of a saved folder set (one holding only set.json and set.npz),
    so saving a set there would delete something else."""
    if target.is_symlink():
        return True
    if not target.exists():
        return False
    try:
        return not target.is_dir() or any(p.name not in _SAVED_FILES for p in target.iterdir())
    except OSError:
        return True


def _foreign_error(target: Path) -> ValueError:
    return ValueError(
        f"{target} already exists and is not a set saved by from_folder, so it is left alone. Pass another name."
    )


def _build_folder_set(
    root: Path,
    name: str,
    title: str,
    question: str,
    modality: str,
    specialist: Specialist,
    folders: list[tuple[str, Path]],
    listings: list[dict],
    settings: dict,
    source: str,
    licence: str,
) -> LabelledSet:
    labels = [label for label, _ in folders]

    def rel(p: Path) -> str:
        return p.relative_to(root).as_posix()

    kept, hashes, duplicates, conflicts, skipped = _drop_exact_duplicates([x["audio"] for x in listings], rel)
    minimum = settings["min_per_label"]
    # before embedding, so no time is spent on a folder that cannot make a set
    _check_enough(folders, listings, [len(files) for files in kept], duplicates, conflicts, skipped, minimum)
    kept, n_capped = _cap(kept, settings["max_per_label"], settings["seed"])
    to_embed = [p for files in kept for p in files]
    rows, unreadable, seconds = _embed(specialist, to_embed, rel)
    skipped += unreadable

    usable = [[p for p in files if p in rows] for files in kept]
    _check_enough(folders, listings, [len(files) for files in usable], duplicates, conflicts, skipped, minimum)

    paths = [p for files in usable for p in files]
    y = np.array([c for c, files in enumerate(usable) for _ in files], dtype=np.int64)
    emb = np.stack([rows[p] for p in paths]).astype(np.float32)
    unit = _unit_rows(emb)

    rng = np.random.default_rng(settings["seed"])
    splits: list[list[int]] = [[], [], []]  # fit, dev, test
    group_of = np.zeros(len(paths), dtype=np.int64)
    n_groups = near_groups = 0
    too_few = []
    for c, label in enumerate(labels):
        members = np.where(y == c)[0]
        groups = [[int(members[k]) for k in g] for g in _near_duplicate_groups(unit[members])]
        near_groups += sum(len(g) > 1 for g in groups)
        for g in groups:
            group_of[g] = n_groups
            n_groups += 1
        parts = _split_groups(groups, len(members), rng)
        if not all(parts):
            too_few.append(f"{label} ({len(members)} usable files, {len(groups)} distinct)")
        for split, part in zip(splits, parts):
            split += part
    if too_few:
        raise ValueError(
            f"Too few distinct clips to give every label fit, dev and test examples: {', '.join(too_few)}. Clips "
            f"whose specialist embeddings have cosine {_NEAR_DUPLICATE} or more count as one. Add more distinct "
            "recordings to those folders."
        )
    if len(splits[0]) < _MIN_FIT:
        per_label = ", ".join(f"{label} {int((y[splits[0]] == c).sum())}" for c, label in enumerate(labels))
        raise ValueError(
            f"The split leaves {len(splits[0])} examples to fit on ({per_label}), but a vector bridge needs at least "
            f"{_MIN_FIT}. Add more distinct recordings to the label folders."
        )
    split_of = np.zeros(len(paths), dtype=np.int64)
    for k, split in enumerate(splits):
        split_of[split] = k
    _check_no_leak(split_of, group_of, [hashes[p] for p in paths])

    # one phrase per label, by the same rule as bridges._label_phrases
    phrases = [f"{n} music" for n in labels] if "music" in question.lower() else [f"the sound of {n}" for n in labels]
    label_text_emb = np.asarray(specialist.embed_texts(phrases), dtype=np.float32)

    fit_idx, dev_idx, test_idx = (np.sort(np.array(split, dtype=np.int64)) for split in splits)
    meta = {
        "origin": "folder",
        "source": source,
        "licence": licence,
        "folder": str(root),
        "specialist": settings["specialist"],
        "max_per_label": settings["max_per_label"],
        "min_per_label": settings["min_per_label"],
        "seed": settings["seed"],
        "n_found": sum(len(x["audio"]) for x in listings),
        "n_used": len(paths),
        "per_label": {label: int((y == c).sum()) for c, label in enumerate(labels)},
        "splits": {"fit": len(fit_idx), "dev": len(dev_idx), "test": len(test_idx)},
        "non_audio": sum(x["non_audio"] for x in listings),
        "duplicates": sum(duplicates),
        "conflicts": conflicts,
        "n_capped": n_capped,
        "skipped": skipped[:_MAX_LISTED],
        "n_skipped": len(skipped),
        "near_duplicate_groups": near_groups,
        "near_duplicates_across_labels": _near_duplicates_across_labels(unit, y),
        "embed_seconds": round(seconds, 3),
        "embed_seconds_per_file": round(seconds / max(len(to_embed), 1), 4),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    return LabelledSet(
        name=name,
        title=title,
        modality=modality,
        question=question,
        labels=labels,
        y=y,
        emb=emb,
        label_text_emb=label_text_emb,
        fit_idx=fit_idx,
        dev_idx=dev_idx,
        test_idx=test_idx,
        files=[str(p) for p in paths],
        meta=meta,
    )


def _drop_exact_duplicates(
    audio: list[list[Path]], rel
) -> tuple[list[list[Path]], dict[Path, str], list[int], list[str], list[dict]]:
    """Per label, the files to keep (sorted by path), the SHA-1 of each kept file, per label the number of extra
    copies dropped, the relative paths of files whose bytes appear under two or more labels (all dropped), and the
    files that could not be opened."""
    copies: dict[str, list[tuple[int, Path]]] = {}
    skipped: list[dict] = []
    for c, files in enumerate(audio):
        for p in files:
            try:
                with open(p, "rb") as handle:
                    digest = hashlib.file_digest(handle, "sha1").hexdigest()
            except OSError as err:
                skipped.append({"file": rel(p), "why": f"it could not be opened ({err.strerror or err})"})
                continue
            copies.setdefault(digest, []).append((c, p))
    kept: list[list[Path]] = [[] for _ in audio]
    hashes: dict[Path, str] = {}
    duplicates, conflicts = [0 for _ in audio], []
    for digest, found in copies.items():
        if len({c for c, _ in found}) > 1:
            conflicts += [rel(p) for _, p in found]
            continue
        c, first = min(found, key=lambda cp: cp[1])
        kept[c].append(first)
        hashes[first] = digest
        duplicates[c] += len(found) - 1
    return [sorted(files) for files in kept], hashes, duplicates, sorted(conflicts), skipped


def _check_enough(
    folders: list[tuple[str, Path]],
    listings: list[dict],
    counts: list[int],
    duplicates: list[int],
    conflicts: list[str],
    skipped: list[dict],
    minimum: int,
) -> None:
    """ValueError naming every label with fewer than `minimum` usable files, with why its other files were not used."""
    short = []
    for (label, directory), listing, count, n_duplicates in zip(folders, listings, counts, duplicates):
        if count >= minimum:
            continue
        prefix = f"{directory.name}/"  # skipped and conflicting files are listed by path relative to the folder
        reasons = []
        if listing["non_audio"]:
            types = ", ".join(listing["non_audio_types"])
            reasons.append(f"{listing['non_audio']} of a type that is not read ({types})")
        if n_duplicates:
            reasons.append(f"{n_duplicates} exact duplicate{'' if n_duplicates == 1 else 's'}")
        n_conflicts = sum(f.startswith(prefix) for f in conflicts)
        if n_conflicts:
            reasons.append(f"{n_conflicts} with the same bytes as a file under another label")
        unreadable = [s for s in skipped if s["file"].startswith(prefix)]
        if unreadable:
            first = unreadable[0]
            reasons.append(f"{len(unreadable)} could not be read (e.g. {first['file']}: {first['why'].rstrip('.')})")
        total = len(listing["audio"]) + listing["non_audio"]
        short.append(f"{label} has {count}" + (f" of {total} files: {', '.join(reasons)}" if reasons else ""))
    if short:
        raise ValueError(
            f"Too few usable audio files: {'; '.join(short)}. Every label needs at least {minimum} (min_per_label). "
            f"Add audio files ({_READ_TYPES}) to those folders, or lower min_per_label (4 at least); a folder whose "
            "name starts with '_' is left out."
        )


def _cap(kept: list[list[Path]], max_per_label: int, seed: int) -> tuple[list[list[Path]], int]:
    """At most max_per_label files per label: a seeded permutation of the sorted list, cut, then sorted again."""
    rng = np.random.default_rng(seed)
    out, n_capped = [], 0
    for files in kept:
        if len(files) > max_per_label:
            n_capped += len(files) - max_per_label
            files = sorted(files[k] for k in rng.permutation(len(files))[:max_per_label])
        out.append(files)
    return out, n_capped


def _embed(specialist: Specialist, paths: list[Path], rel) -> tuple[dict[Path, np.ndarray], list[dict], float]:
    """Specialist embeddings in batches; a batch that fails on a file is retried file by file, and files that
    cannot be read are returned as skipped entries {"file", "why"}."""
    rows: dict[Path, np.ndarray] = {}
    skipped: list[dict] = []

    def keep(p: Path, row: np.ndarray) -> None:
        if np.all(np.isfinite(row)):
            rows[p] = row
        else:
            skipped.append({"file": rel(p), "why": "the specialist returned an embedding that is not finite"})

    started = time.perf_counter()
    for start in range(0, len(paths), _EMBED_BATCH):
        batch = paths[start : start + _EMBED_BATCH]
        try:
            out = np.asarray(specialist.embed_files(batch), dtype=np.float32)
        except (ValueError, FileNotFoundError):
            for p in batch:
                try:
                    keep(p, np.asarray(specialist.embed_files([p]), dtype=np.float32)[0])
                except (ValueError, FileNotFoundError) as err:
                    skipped.append({"file": rel(p), "why": str(err).replace(str(p), rel(p))})
            continue
        for p, row in zip(batch, out):
            keep(p, row)
    return rows, skipped, time.perf_counter() - started


def _unit_rows(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)


def _near_duplicate_groups(unit: np.ndarray) -> list[list[int]]:
    """Groups of rows (indices into `unit`) joined by cosine >= _NEAR_DUPLICATE, transitively (union-find),
    ordered by their first row."""
    parent = list(range(len(unit)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    close = np.triu(unit @ unit.T >= _NEAR_DUPLICATE, k=1)
    for i, j in zip(*np.nonzero(close)):
        a, b = find(int(i)), find(int(j))
        if a != b:
            parent[max(a, b)] = min(a, b)
    groups: dict[int, list[int]] = {}
    for i in range(len(unit)):
        groups.setdefault(find(i), []).append(i)
    return sorted(groups.values(), key=lambda g: g[0])


def _split_groups(groups: list[list[int]], n: int, rng: np.random.Generator) -> tuple[list[int], list[int], list[int]]:
    """60/20/20 of one label's n clips by whole groups: a seeded permutation of the groups fills fit up to
    round(0.6 n), then dev up to round(0.2 n), then test, always leaving at least one group each for dev and test
    while they are empty. With all groups single clips this is the per-label split of the built-in gtzan set."""
    n_fit, n_dev = int(round(0.6 * n)), int(round(0.2 * n))
    fit: list[int] = []
    dev: list[int] = []
    test: list[int] = []
    order = rng.permutation(len(groups))
    for k, g in enumerate(order):
        members, left = groups[g], len(order) - k - 1  # groups still to place after this one
        if (len(fit) < n_fit or not fit) and left >= (not dev) + (not test):
            fit += members
        elif (len(dev) < n_dev or not dev) and left >= (not test):
            dev += members
        else:
            test += members
    return fit, dev, test


def _near_duplicates_across_labels(unit: np.ndarray, y: np.ndarray) -> int:
    """Pairs of clips with different labels whose embeddings have cosine >= _NEAR_DUPLICATE."""
    count = 0
    for c in range(int(y.max()) + 1 if len(y) else 0):
        later = unit[y > c]
        if len(later):
            count += int((unit[y == c] @ later.T >= _NEAR_DUPLICATE).sum())
    return count


def _check_no_leak(split_of: np.ndarray, group_of: np.ndarray, hashes: list[str]) -> None:
    """Internal check: no exact-duplicate bytes and no near-duplicate group may sit in two splits."""
    for what, keys in (("exact-duplicate", hashes), ("near-duplicate", group_of.tolist())):
        first: dict = {}
        for i, key in enumerate(keys):
            if first.setdefault(key, int(split_of[i])) != int(split_of[i]):
                raise RuntimeError(
                    f"Internal error in sense.datasets.from_folder: a {what} group spans two splits. "
                    "This is a bug in the split; report it with the folder that caused it."
                )


def _save_folder_set(ds: LabelledSet, fingerprint: str) -> None:
    """Write set.json and set.npz to a staging directory, then move it into place under config.sets_dir(). Only a
    saved folder set is replaced; anything else at that path raises ValueError and is left alone."""
    root = config.sets_dir()
    root.mkdir(parents=True, exist_ok=True)
    target = root / ds.name
    tag = f"{os.getpid()}-{uuid.uuid4().hex[:8]}"
    staging = root / f".{ds.name}.building-{tag}"
    old = None
    record = {
        "format": _FOLDER_FORMAT,
        "name": ds.name,
        "title": ds.title,
        "question": ds.question,
        "modality": ds.modality,
        "labels": list(ds.labels),
        "files": list(ds.files or []),
        "y": [int(c) for c in ds.y],
        "fit_idx": [int(i) for i in ds.fit_idx],
        "dev_idx": [int(i) for i in ds.dev_idx],
        "test_idx": [int(i) for i in ds.test_idx],
        "meta": ds.meta,
        "fingerprint": fingerprint,
    }
    try:
        staging.mkdir()
        np.savez(staging / "set.npz", emb=ds.emb, label_text_emb=ds.label_text_emb, fingerprint=np.array(fingerprint))
        (staging / "set.json").write_text(json.dumps(record, indent=1))
        with _swap_lock:
            if target.exists() or target.is_symlink():
                if _not_a_saved_set(target):
                    raise _foreign_error(target)
                old = root / f".{ds.name}.removing-{tag}"
                target.rename(old)
            try:
                staging.rename(target)
            except OSError:
                if old is not None:  # put the replaced set back rather than lose it
                    old.rename(target)
                    old = None
                raise
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        if old is not None:  # the replaced set, removed after the swap so readers wait only for two renames
            shutil.rmtree(old, ignore_errors=True)


def _load_folder_set(name: str) -> LabelledSet:
    directory = config.sets_dir() / name
    try:
        st = (directory / "set.json").stat()
    except FileNotFoundError:  # removed by another process since it was found
        raise _unknown(name) from None
    return _read_folder_set(str(directory), st.st_mtime_ns, st.st_ino, st.st_size)


@lru_cache(maxsize=32)
def _read_folder_set(directory: str, mtime_ns: int, inode: int, size: int) -> LabelledSet:
    """A saved folder set. The cache key holds set.json's mtime, inode and size, so a rebuilt set is read again."""
    folder = Path(directory)
    try:
        record = json.loads((folder / "set.json").read_text())
        if record.get("format") != _FOLDER_FORMAT:
            raise ValueError(f"it is in format {record.get('format')!r}, and this version reads {_FOLDER_FORMAT}")
        with np.load(folder / "set.npz", allow_pickle=False) as arrays:
            emb = arrays["emb"].astype(np.float32)
            label_text_emb = arrays["label_text_emb"].astype(np.float32)
            if str(arrays["fingerprint"]) != record["fingerprint"]:
                raise ValueError("set.json and set.npz come from different builds")
        labels = [str(label) for label in record["labels"]]
        y = np.asarray(record["y"], dtype=np.int64)
        fit_idx, dev_idx, test_idx = (np.asarray(record[k], dtype=np.int64) for k in ("fit_idx", "dev_idx", "test_idx"))
        files = [str(f) for f in record["files"]]
        n = len(y)
        if emb.ndim != 2 or emb.shape[0] != n or len(files) != n or label_text_emb.shape != (len(labels), emb.shape[1]):
            raise ValueError("its arrays do not match its list of examples")
        if n and (y.min() < 0 or y.max() >= len(labels)):
            raise ValueError("an example has a label index outside its labels")
        every = np.concatenate([fit_idx, dev_idx, test_idx])
        if len(np.unique(every)) != len(every) or (len(every) and (every.min() < 0 or every.max() >= n)):
            raise ValueError("its fit, dev and test splits overlap or point outside the set")
        ds = LabelledSet(
            name=folder.name,
            title=str(record["title"]),
            modality=str(record["modality"]),
            question=str(record["question"]),
            labels=labels,
            y=y,
            emb=emb,
            label_text_emb=label_text_emb,
            fit_idx=fit_idx,
            dev_idx=dev_idx,
            test_idx=test_idx,
            files=files,
            meta=dict(record.get("meta") or {}),
        )
    except Exception as err:
        raise ValueError(
            f"The folder set in {folder} cannot be read ({type(err).__name__}: {err}). Build it again with "
            "sense.datasets.from_folder, or delete that directory."
        ) from err
    if (ds.meta.get("specialist") == "fake") != config.fake():
        _log.warning(
            "The folder set %r was embedded by the %s specialist, but the harness is in %s mode; its embeddings do "
            "not match the specialist in use. Build it again with sense.datasets.from_folder in this mode.",
            ds.name, ds.meta.get("specialist"), "fake" if config.fake() else "real",
        )
    return ds
