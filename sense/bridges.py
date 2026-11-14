"""Bridges: ways of getting a modality into a base model.

Public surface (do not rename):
  fit_vector_bridge(model, dataset, idx, layer=None) -> Bridge
      Closed-form ridge map from specialist embeddings to the model's own text representation of each label,
      with a norm match (method and reference code: socket_pilot/run_pilot.py, fit_plug). Open-weight models
      only. `idx` are the examples to fit on. If `layer` is None, pick it on dataset.dev_idx from blocks 0-4.
  make_text_bridge(model, dataset=None) -> Bridge
      Describes the input in words. With a dataset: the specialist's best-matching label
      ('audio: sounds like "rock"'). Without: the nearest caption from datasets.caption_bank().
  load_bridge(directory) -> Bridge
      Reload a bridge saved with Bridge.save.

Every bridge implements contracts.Bridge: from_embedding, prepare, save. `name` is "vector:<dataset>",
"text:<dataset>" or "text:captions".
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sense.contracts import BaseModel, Bridge, LabelledSet


def fit_vector_bridge(model: BaseModel, dataset: LabelledSet, idx: np.ndarray, layer: int | None = None) -> Bridge:
    raise NotImplementedError


def make_text_bridge(model: BaseModel, dataset: LabelledSet | None = None) -> Bridge:
    raise NotImplementedError


def load_bridge(directory: Path) -> Bridge:
    raise NotImplementedError
