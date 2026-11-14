"""Specialist models: frozen encoders that understand one modality.

Public surface (do not rename):
  get_specialist(modality) -> Specialist      cached; honours config.fake()
  Specialist.dim                              embedding size
  Specialist.embed_files(paths) -> [n, dim]   one embedding per file
  Specialist.embed_texts(texts) -> [n, dim]   text embeddings in the same space

The audio specialist is CLAP (config.model_source("clap")); it takes 48 kHz mono and uses the middle
10 seconds of longer files. Reference implementation: socket_pilot/clap_embed_gtzan.py.
In fake mode the specialist is deterministic and loads nothing.

The fake world (FAKE_LABELS, FAKE_CONCEPTS, fake_concepts_in, FakeSpecialist.write_sample) is shared with the
fake datasets and the fake base model, so that fake mode behaves like the real thing end to end.
"""

from __future__ import annotations

import hashlib
import re
import threading
from functools import lru_cache
from math import gcd
from pathlib import Path
from typing import Protocol

import numpy as np

from sense import config


class Specialist(Protocol):
    modality: str
    dim: int

    def embed_files(self, paths: list[Path]) -> np.ndarray: ...

    def embed_texts(self, texts: list[str]) -> np.ndarray: ...


def get_specialist(modality: str) -> Specialist:
    if modality != "audio":
        raise ValueError(f"No specialist is known for modality {modality!r}; the only one available is 'audio'.")
    return _specialist(modality, config.fake())


@lru_cache(maxsize=None)
def _specialist(modality: str, fake: bool) -> Specialist:
    return FakeSpecialist() if fake else ClapSpecialist()


# ----------------------------------------------------------------------------- CLAP
class ClapSpecialist:
    """Frozen CLAP (laion/clap-htsat-unfused by default). Loads on the first embed call and keeps one instance."""

    modality = "audio"
    SR = 48_000
    WINDOW_S = 10
    BATCH = 8

    def __init__(self) -> None:
        self.dim = 512
        self._model = None
        self._proc = None
        self._device = "cpu"
        self._lock = threading.Lock()

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import ClapProcessor

        source = config.model_source("clap")
        try:
            model = _load_clap_model(source)
            proc = ClapProcessor.from_pretrained(source)
        except Exception as err:
            raise RuntimeError(
                f"Could not load the CLAP audio specialist from {source!r} ({type(err).__name__}: {err}). "
                "Connect to the internet once so Hugging Face can download it, or set SENSE_CLAP to a local "
                "directory holding the model."
            ) from err
        self._device = "mps" if torch.backends.mps.is_available() else "cpu"
        self._model = model.to(self._device).eval()
        self._proc = proc
        self.dim = int(getattr(model.config, "projection_dim", self.dim))

    def _features(self, out) -> np.ndarray:
        import torch

        tensor = out if torch.is_tensor(out) else out.pooler_output  # depends on the transformers version
        return tensor.float().cpu().numpy()

    def embed_files(self, paths: list[Path]) -> np.ndarray:
        import torch

        wavs = [_load_middle(Path(p), self.SR, self.WINDOW_S) for p in paths]
        with self._lock:
            self._ensure_loaded()
            out = []
            for start in range(0, len(wavs), self.BATCH):
                batch = wavs[start : start + self.BATCH]
                try:
                    inputs = self._proc(audio=batch, sampling_rate=self.SR, return_tensors="pt", padding=True)
                except TypeError:  # older processors call the argument `audios`
                    inputs = self._proc(audios=batch, sampling_rate=self.SR, return_tensors="pt", padding=True)
                with torch.no_grad():
                    feats = self._model.get_audio_features(**{k: v.to(self._device) for k, v in inputs.items()})
                out.append(self._features(feats))
        return np.concatenate(out).astype(np.float32) if out else np.zeros((0, self.dim), np.float32)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        import torch

        with self._lock:
            self._ensure_loaded()
            out = []
            for start in range(0, len(texts), 64):
                inputs = self._proc(text=texts[start : start + 64], return_tensors="pt", padding=True, truncation=True)
                with torch.no_grad():
                    feats = self._model.get_text_features(**{k: v.to(self._device) for k, v in inputs.items()})
                out.append(self._features(feats))
        return np.concatenate(out).astype(np.float32) if out else np.zeros((0, self.dim), np.float32)


def _load_clap_model(source: str):
    """ClapModel from a hub id or a local path. laion/clap-htsat-unfused keeps its safetensors weights in a
    conversion pull request (refs/pr/3), not on main, so offline the hub id alone cannot find them: fall back to
    any cached snapshot that holds model.safetensors, with the config from main."""
    from transformers import ClapConfig, ClapModel

    try:
        return ClapModel.from_pretrained(source, use_safetensors=True)
    except Exception:
        weights = _cached_safetensors_dir(source)
        if weights is not None:
            return ClapModel.from_pretrained(weights, config=ClapConfig.from_pretrained(source), use_safetensors=True)
        return ClapModel.from_pretrained(source)  # a repo that ships only pytorch_model.bin


def _cached_safetensors_dir(source: str) -> Path | None:
    """The newest cached snapshot of `source` that holds model.safetensors, or None."""
    from huggingface_hub.constants import HF_HUB_CACHE

    for cache in [c for c in (config.hf_cache(), HF_HUB_CACHE) if c]:
        snapshots = Path(cache) / ("models--" + source.replace("/", "--")) / "snapshots"
        if not snapshots.is_dir():
            continue
        for snap in sorted(snapshots.iterdir(), key=lambda d: d.stat().st_mtime, reverse=True):
            if (snap / "model.safetensors").exists():
                return snap
    return None


def _load_middle(path: Path, sr: int, window_s: int) -> np.ndarray:
    """Mono float32 at `sr`, cut to the middle `window_s` seconds."""
    import soundfile as sf
    from scipy.signal import resample_poly

    if not path.is_file():
        raise FileNotFoundError(f"Audio file {path} does not exist.")
    try:
        wav, fs = sf.read(path, dtype="float32", always_2d=True)
    except Exception as err:
        raise ValueError(f"Could not read {path.name} as audio ({err}). Use a WAV, FLAC or OGG file.") from err
    wav = wav.mean(axis=1)
    if len(wav) == 0:
        raise ValueError(f"{path.name} contains no audio samples.")
    half = window_s * fs // 2
    mid = len(wav) // 2
    wav = wav[max(0, mid - half) : mid + half]
    if fs != sr:
        g = gcd(sr, fs)
        wav = resample_poly(wav, sr // g, fs // g)
    return wav.astype(np.float32)


# ----------------------------------------------------------------------------- the fake world
FAKE_LABELS: dict[str, list[str]] = {
    "gtzan": ["blues", "classical", "country", "disco", "hip hop", "jazz", "metal", "pop", "reggae", "rock"],
    "esc50": [
        "dog", "rooster", "rain", "sea waves", "crying baby", "clock tick", "sneezing", "helicopter",
        "chainsaw", "church bells", "keyboard typing", "door knock", "thunderstorm", "cat", "siren",
    ],
}
FAKE_CONCEPTS: list[str] = [c for labels in FAKE_LABELS.values() for c in labels]
FAKE_DIM = 64
_FAKE_SR = 16_000


def _seed(*parts: object) -> int:
    digest = hashlib.md5("\x1f".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "little")


@lru_cache(maxsize=1)
def _prototypes() -> np.ndarray:
    protos = np.random.default_rng(20261114).normal(size=(len(FAKE_CONCEPTS), FAKE_DIM))
    return protos / np.linalg.norm(protos, axis=1, keepdims=True)


def fake_concepts_in(text: str) -> list[int]:
    """Indices into FAKE_CONCEPTS of every concept named in `text` (whole words, case ignored)."""
    low = text.lower()
    return [k for k, c in enumerate(FAKE_CONCEPTS) if re.search(rf"\b{re.escape(c)}\b", low)]


def _concept_freq(k: int) -> float:
    return 300.0 + 25.0 * k


class FakeSpecialist:
    """Deterministic stand-in for CLAP. Each fake concept has a prototype vector; a fake audio file is a pure
    tone whose frequency names its concept, and text is embedded by the concepts it mentions."""

    modality = "audio"
    dim = FAKE_DIM
    NOISE = 0.5  # norm of the per-example noise, against unit-norm prototypes

    def example(self, concept: int, rng: np.random.Generator) -> np.ndarray:
        """One synthetic example embedding of a concept."""
        noise = rng.normal(size=FAKE_DIM) * self.NOISE / np.sqrt(FAKE_DIM)
        return (_prototypes()[concept] + noise).astype(np.float32)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        out = []
        for text in texts:
            rng = np.random.default_rng(_seed("text", text))
            hits = fake_concepts_in(text)
            if hits:
                vec = _prototypes()[hits].sum(0) + rng.normal(size=FAKE_DIM) * 0.1 / np.sqrt(FAKE_DIM)
            else:
                vec = rng.normal(size=FAKE_DIM) / np.sqrt(FAKE_DIM)
            out.append(vec)
        return np.array(out, dtype=np.float32).reshape(len(texts), FAKE_DIM)

    def embed_files(self, paths: list[Path]) -> np.ndarray:
        return np.array([self._embed_file(Path(p)) for p in paths], dtype=np.float32).reshape(len(paths), FAKE_DIM)

    def _embed_file(self, path: Path) -> np.ndarray:
        import soundfile as sf

        if not path.is_file():
            raise FileNotFoundError(f"Audio file {path} does not exist.")
        try:
            wav, fs = sf.read(path, dtype="float32", always_2d=True)
        except Exception as err:
            raise ValueError(f"Could not read {path.name} as audio ({err}). Use a WAV, FLAC or OGG file.") from err
        wav = wav.mean(axis=1)
        rng = np.random.default_rng(_seed("file", hashlib.md5(wav.tobytes()).hexdigest()))
        concept = self._tone_concept(wav, fs)
        if concept is None:  # not a fake sample: an embedding no concept is close to
            return rng.normal(size=FAKE_DIM) / np.sqrt(FAKE_DIM)
        return self.example(concept, rng)

    @staticmethod
    def _tone_concept(wav: np.ndarray, fs: int) -> int | None:
        if len(wav) < fs // 10:
            return None
        spectrum = np.abs(np.fft.rfft(wav))
        freqs = np.fft.rfftfreq(len(wav), 1 / fs)
        peak = freqs[int(spectrum.argmax())]
        k = int(round((peak - 300.0) / 25.0))
        if 0 <= k < len(FAKE_CONCEPTS) and abs(peak - _concept_freq(k)) <= 3.0:
            return k
        return None

    @staticmethod
    def write_sample(concept: int, path: Path, seed: int) -> None:
        """Write a one-second fake audio file for a concept: a tone plus a little seeded noise."""
        import soundfile as sf

        rng = np.random.default_rng(seed)
        t = np.arange(_FAKE_SR) / _FAKE_SR
        wav = 0.3 * np.sin(2 * np.pi * _concept_freq(concept) * t) + 0.02 * rng.normal(size=t.shape)
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, wav.astype(np.float32), _FAKE_SR)
