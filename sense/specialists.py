"""Specialist models: frozen encoders that understand one modality.

Public surface (do not rename):
  get_specialist(modality) -> Specialist      cached; honours config.fake()
  Specialist.dim                              embedding size
  Specialist.embed_files(paths) -> [n, dim]   one embedding per file
  Specialist.embed_texts(texts) -> [n, dim]   text embeddings in the same space

The audio specialist is CLAP (config.model_source("clap")); it takes 48 kHz mono and uses the middle
10 seconds of longer files. Reference implementation: socket_pilot/clap_embed_gtzan.py.
In fake mode the specialist is deterministic and loads nothing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np


class Specialist(Protocol):
    modality: str
    dim: int

    def embed_files(self, paths: list[Path]) -> np.ndarray: ...

    def embed_texts(self, texts: list[str]) -> np.ndarray: ...


def get_specialist(modality: str) -> Specialist:
    raise NotImplementedError
