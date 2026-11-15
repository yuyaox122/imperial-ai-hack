"""Installed senses, persisted under config.home().

Public surface (do not rename):
  install(bridge, card) -> None                 saves the bridge and its card; replaces the same model+modality
  installed(model_name) -> list[dict]           [{"bridge", "kind", "modality", "dataset", "card": ReportCard dict}]
  get_bridges(model_name) -> list[Bridge]       loaded bridges for chat, one per modality
  uninstall(model_name, modality) -> bool
  sense_dir(model_name, modality) -> Path       where the sense of that model and modality lives (it may not exist)

Layout: <home>/<model_name>/<modality>/card.json plus whatever Bridge.save writes.
The saved card always has verdict "installed". A sense is written to a temporary directory first and then
moved into place, so a chat never sees a half-written bridge. Replacing a sense moves the old one aside, moves the
new one in, and only then deletes the old one (put back if the new one cannot be moved in); uninstall also moves
the sense aside before deleting it. In one process (the server and its agent runs) these moves hold a lock that
installed() and get_bridges() also take, so a reader finds the old sense or the new one, never none and never a
half-deleted one. Temporary directories start with "." and are not listed.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from sense import config
from sense.bridges import load_bridge
from sense.contracts import Bridge, ReportCard

_swap_lock = threading.RLock()  # held while a sense is moved; readers take it too, so they never find it missing


def install(bridge: Bridge, card: ReportCard) -> None:
    if card.model != bridge.model_name or card.modality != bridge.modality or card.bridge != bridge.name:
        raise ValueError(
            f"The report card is for {card.bridge} on {card.model} ({card.modality}), but the bridge is "
            f"{bridge.name} on {bridge.model_name} ({bridge.modality}); install a bridge with its own card."
        )
    target = _sense_dir(bridge.model_name, bridge.modality)
    tag = _tag()
    staging = target.with_name(f".{target.name}.installing-{tag}")
    old = None
    try:
        bridge.save(staging)
        saved = card.to_dict()
        saved["verdict"] = "installed"
        saved["created"] = saved["created"] or datetime.now(timezone.utc).isoformat(timespec="seconds")
        (staging / "card.json").write_text(json.dumps(saved, indent=2))
        with _swap_lock:
            if target.exists():
                old = target.with_name(f".{target.name}.removing-{tag}")
                target.rename(old)
            try:
                staging.rename(target)
            except OSError:
                if old is not None:  # put the replaced sense back rather than lose it; never delete it from here on
                    restore, old = old, None
                    restore.rename(target)
                raise
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        if old is not None:  # the replaced sense, removed after the swap so readers wait only for two renames
            shutil.rmtree(old, ignore_errors=True)


def installed(model_name: str) -> list[dict]:
    out = []
    with _swap_lock:  # a sense that install is replacing is listed, not briefly missing
        for directory in _installed_dirs(model_name):
            card = json.loads((directory / "card.json").read_text())
            out.append(
                {"bridge": card["bridge"], "kind": card["kind"], "modality": card["modality"], "dataset": card["dataset"], "card": card}
            )
    return out


def get_bridges(model_name: str) -> list[Bridge]:
    bridges = []
    with _swap_lock:  # a sense that install is replacing is loaded whole, not briefly missing
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
    with _swap_lock:  # moved aside first, so a reader never finds it half deleted
        if not target.exists():
            return False
        old = target.with_name(f".{target.name}.removing-{_tag()}")
        target.rename(old)
    shutil.rmtree(old, ignore_errors=True)
    return True


def sense_dir(model_name: str, modality: str) -> Path:
    return _sense_dir(model_name, modality)


def _tag() -> str:
    """A suffix no other install or uninstall uses, in this process or another, for staging and removed directories."""
    return f"{os.getpid()}-{threading.get_ident()}-{uuid.uuid4().hex[:8]}"


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
