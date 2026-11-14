"""Installed senses, persisted under config.home().

Public surface (do not rename):
  install(bridge, card) -> None                 saves the bridge and its card; replaces the same model+modality
  installed(model_name) -> list[dict]           [{"bridge", "kind", "modality", "dataset", "card": ReportCard dict}]
  get_bridges(model_name) -> list[Bridge]       loaded bridges for chat, one per modality
  uninstall(model_name, modality) -> bool
  sense_dir(model_name, modality) -> Path       where the sense of that model and modality lives (it may not exist)

Layout: <home>/<model_name>/<modality>/card.json plus whatever Bridge.save writes.
The saved card always has verdict "installed". A sense is written to a temporary directory first and then
moved into place, so a chat never sees a half-written bridge.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

from sense import config
from sense.bridges import load_bridge
from sense.contracts import Bridge, ReportCard


def install(bridge: Bridge, card: ReportCard) -> None:
    if card.model != bridge.model_name or card.modality != bridge.modality or card.bridge != bridge.name:
        raise ValueError(
            f"The report card is for {card.bridge} on {card.model} ({card.modality}), but the bridge is "
            f"{bridge.name} on {bridge.model_name} ({bridge.modality}); install a bridge with its own card."
        )
    target = _sense_dir(bridge.model_name, bridge.modality)
    staging = target.with_name(f".{target.name}.installing-{os.getpid()}")
    if staging.exists():
        shutil.rmtree(staging)
    bridge.save(staging)
    saved = card.to_dict()
    saved["verdict"] = "installed"
    saved["created"] = saved["created"] or datetime.now(timezone.utc).isoformat(timespec="seconds")
    (staging / "card.json").write_text(json.dumps(saved, indent=2))
    if target.exists():
        old = target.with_name(f".{target.name}.removing-{os.getpid()}")
        target.rename(old)
        shutil.rmtree(old)
    staging.rename(target)


def installed(model_name: str) -> list[dict]:
    out = []
    for directory in _installed_dirs(model_name):
        card = json.loads((directory / "card.json").read_text())
        out.append(
            {"bridge": card["bridge"], "kind": card["kind"], "modality": card["modality"], "dataset": card["dataset"], "card": card}
        )
    return out


def get_bridges(model_name: str) -> list[Bridge]:
    bridges = []
    for directory in _installed_dirs(model_name):
        try:
            bridges.append(load_bridge(directory))
        except Exception as err:
            raise RuntimeError(
                f"The installed {directory.name} sense for {model_name} could not be loaded ({err}). "
                f"Uninstall it and run the agent again."
            ) from err
    return bridges


def uninstall(model_name: str, modality: str) -> bool:
    target = _sense_dir(model_name, modality)
    if not target.exists():
        return False
    shutil.rmtree(target)
    return True


def sense_dir(model_name: str, modality: str) -> Path:
    return _sense_dir(model_name, modality)


def _safe(part: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", part)
    if not cleaned or set(cleaned) == {"."}:
        raise ValueError(f"{part!r} cannot be used as a model or modality name.")
    return cleaned


def _sense_dir(model_name: str, modality: str) -> Path:
    return config.home() / _safe(model_name) / _safe(modality)


def _installed_dirs(model_name: str) -> list[Path]:
    root = config.home() / _safe(model_name)
    if not root.is_dir():
        return []
    return sorted(d for d in root.iterdir() if d.is_dir() and not d.name.startswith(".") and (d / "card.json").is_file())
