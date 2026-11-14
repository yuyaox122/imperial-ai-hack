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

import numpy as np

from sense.contracts import LabelledSet


def available_sets() -> list[dict]:
    raise NotImplementedError


def load_labelled_set(name: str) -> LabelledSet:
    raise NotImplementedError


def caption_bank() -> tuple[np.ndarray, list[str]]:
    raise NotImplementedError


def sample_files(name: str, per_label: int = 1) -> list[dict]:
    raise NotImplementedError
