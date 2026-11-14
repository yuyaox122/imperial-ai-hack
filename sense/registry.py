"""Installed senses, persisted under config.home().

Public surface (do not rename):
  install(bridge, card) -> None                 saves the bridge and its card; replaces the same model+modality
  installed(model_name) -> list[dict]           [{"bridge", "kind", "modality", "dataset", "card": ReportCard dict}]
  get_bridges(model_name) -> list[Bridge]       loaded bridges for chat, one per modality
  uninstall(model_name, modality) -> bool

Layout: <home>/<model_name>/<modality>/card.json plus whatever Bridge.save writes.
"""

from __future__ import annotations

from sense.contracts import Bridge, ReportCard


def install(bridge: Bridge, card: ReportCard) -> None:
    raise NotImplementedError


def installed(model_name: str) -> list[dict]:
    raise NotImplementedError


def get_bridges(model_name: str) -> list[Bridge]:
    raise NotImplementedError


def uninstall(model_name: str, modality: str) -> bool:
    raise NotImplementedError
