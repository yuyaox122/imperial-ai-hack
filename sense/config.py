"""Settings, read from the environment and from .env files found by walking up from the working directory.

  SENSE_FAKE=1          use fake models, specialist and data: no weights, no downloads (tests, interface work)
  SENSE_HOME            where installed senses are stored (default: <repo>/.sense)
  SENSE_DATA_DIR        cached specialist embeddings as .npz (default: <repo>/socket_pilot/artifacts)
  SENSE_HF_CACHE        extra Hugging Face cache directory holding the base-model weights
  SENSE_QWEN_0P8B       Hugging Face id or local path (default: Qwen/Qwen3.5-0.8B)
  SENSE_QWEN_2B         Hugging Face id or local path (default: Qwen/Qwen3.5-2B)
  SENSE_CLAP            Hugging Face id or local path (default: laion/clap-htsat-unfused)
  SENSE_SAMPLE_DIRS     os.pathsep-separated directories of audio files offered as samples in the interface
  OPENAI_API_KEY        optional: enables the LLM planner and API base models
"""

from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _load_env_files() -> None:
    """Minimal .env loader: KEY=VALUE lines, no export, no interpolation. Existing variables win."""
    seen: set[Path] = set()
    for start in (Path.cwd(), REPO):
        for directory in (start, *start.parents):
            for name in (".env", ".env.local"):
                path = directory / name
                if path in seen or not path.is_file():
                    continue
                seen.add(path)
                for line in path.read_text().splitlines():
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, value = line.partition("=")
                    os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_env_files()


def fake() -> bool:
    return os.environ.get("SENSE_FAKE", "") not in ("", "0", "false")


def home() -> Path:
    return Path(os.environ.get("SENSE_HOME", REPO / ".sense"))


def data_dir() -> Path:
    return Path(os.environ.get("SENSE_DATA_DIR", REPO / "socket_pilot" / "artifacts"))


def hf_cache() -> str | None:
    return os.environ.get("SENSE_HF_CACHE") or None


def model_source(name: str) -> str:
    """Hugging Face id or local path for a known model name."""
    defaults = {
        "qwen3.5-0.8b": ("SENSE_QWEN_0P8B", "Qwen/Qwen3.5-0.8B"),
        "qwen3.5-2b": ("SENSE_QWEN_2B", "Qwen/Qwen3.5-2B"),
        "clap": ("SENSE_CLAP", "laion/clap-htsat-unfused"),
    }
    env, default = defaults[name]
    return os.environ.get(env, default)


def sample_dirs() -> list[Path]:
    raw = os.environ.get("SENSE_SAMPLE_DIRS", "")
    return [Path(p) for p in raw.split(os.pathsep) if p]


def openai_key() -> str | None:
    return os.environ.get("OPENAI_API_KEY") or None
