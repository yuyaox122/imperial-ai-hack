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
  SENSE_API_VERIFY_ITEMS  held-out items verification scores for an API (not open-weight) base model (default 50,
                        at least 10); each item costs three paid requests (the bridge, swapped input, no input),
                        so this is lower than the 200 used locally
  SENSE_API_CACHE       on by default: an API model's multiple-choice answers are kept under <SENSE_HOME>/api_cache
                        so repeat verification asks the API model nothing again; "0", "false", "no" or "off" turn
                        it off
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


API_VERIFY_MIN_ITEMS = 10  # fewer can leave the swapped-input control with held-out examples of one label only


def api_verify_items() -> int:
    """Held-out items to score when verifying an API base model: SENSE_API_VERIFY_ITEMS, default 50, at least 10."""
    raw = os.environ.get("SENSE_API_VERIFY_ITEMS", "").strip()
    if not raw:
        return 50
    try:
        value = int(raw)
    except ValueError:
        value = 0
    if value <= 0:
        raise ValueError(
            f"SENSE_API_VERIFY_ITEMS is {raw!r}, but it must be a positive whole number of held-out items. "
            "Set it to, for example, 50, or unset it to use the default of 50."
        )
    if value < API_VERIFY_MIN_ITEMS:
        raise ValueError(
            f"SENSE_API_VERIFY_ITEMS is {raw!r}, but verification needs at least {API_VERIFY_MIN_ITEMS} held-out "
            "items so the swapped-input control has examples of two labels. "
            f"Set it to {API_VERIFY_MIN_ITEMS} or more, or unset it to use the default of 50."
        )
    return value


def api_cache_enabled() -> bool:
    """Whether an API model's multiple-choice answers are cached on disk (SENSE_API_CACHE, on by default)."""
    return os.environ.get("SENSE_API_CACHE", "").strip().lower() not in ("0", "false", "no", "off")
