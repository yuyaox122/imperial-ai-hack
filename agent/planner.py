"""Planners decide which bridges to try and which one to install.

Public surface (do not rename):
  RulePlanner                      needs no API key; implements contracts.Planner
  get_planner(name=None) -> Planner   the registered planner called `name`, else the preferred one
  register_planner(name, factory, preferred=False)   lets another module add a planner (LLM, run-time discovery)

RulePlanner.plan: ["vector", "text"] for an open-weight model, ["text"] otherwise.
RulePlanner.decide, over the verified candidate cards:
  - a vector card is acceptable when accuracy >= swapped + 0.25 and accuracy >= chance + 0.25, and, for
    scope "open", when controls["unseen_classes"] >= chance + 0.25 as well
  - a text card is acceptable when accuracy >= chance + 0.15
  - install the acceptable card with the highest accuracy, preferring vector when within 0.03 of the best;
    if none is acceptable install nothing
  - the reason names the numbers that decided it, in one or two plain sentences
"""

from __future__ import annotations

from typing import Callable

from sense.contracts import Planner


class RulePlanner:
    name = "rules"


def get_planner(name: str | None = None) -> Planner:
    raise NotImplementedError


def register_planner(name: str, factory: Callable[[], Planner], preferred: bool = False) -> None:
    raise NotImplementedError
