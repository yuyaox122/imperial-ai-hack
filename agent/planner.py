"""Planners decide which bridges to try and which one to install.

Public surface (do not rename):
  RulePlanner                      needs no API key; implements contracts.Planner
  get_planner(name=None) -> Planner   the registered planner called `name`, else the preferred one
  register_planner(name, factory, preferred=False)   lets another module add a planner (LLM, run-time discovery)

RulePlanner.plan: ["vector", "text"] for an open-weight model, ["text"] otherwise.
RulePlanner.decide, over the verified candidate cards:
  - a vector card is acceptable when accuracy >= swapped + 0.25 and accuracy >= chance + 0.25, and, for
    scope "open", when controls["unseen_classes"] >= chance + 0.25 and is within 0.20 of the accuracy
  - a text card is acceptable when accuracy >= chance + 0.15 and accuracy >= swapped + 0.10
  - install the acceptable card with the highest accuracy, preferring vector when within 0.03 of the best;
    if none is acceptable install nothing
  - the reason names the numbers that decided it, in one or two plain sentences
RulePlanner.assess(request, card) -> (acceptable, sentence) is the per-card check behind decide; agent.loop
uses it, when a planner has it, to write the reason on each rejected card.
"""

from __future__ import annotations

import logging
from typing import Callable

from sense import config
from sense.contracts import BridgeKind, Decision, FitRequest, LabelledSet, ModelInfo, Planner, ReportCard

VECTOR_MARGIN = 0.25  # a vector bridge must beat chance and swapped input by this much
TEXT_MARGIN = 0.15  # a text bridge must beat chance by this much
TEXT_SWAPPED_MARGIN = 0.10  # and swapped input by this much, or it is not using the input
OPEN_DROP = 0.20  # for open scope, accuracy on unseen classes may trail fitted classes by at most this much
PREFER_VECTOR_WITHIN = 0.03  # install the vector bridge when it is this close to the best acceptable card
_EPS = 1e-9  # accuracies sitting exactly on a bar pass it


def pct(x: float) -> str:
    """0.849 -> '84.9%'."""
    return f"{x * 100:.1f}%"


class RulePlanner:
    """Fixed thresholds; no API key needed."""

    name = "rules"

    def plan(self, request: FitRequest, model: ModelInfo, dataset: LabelledSet) -> list[BridgeKind]:
        return ["vector", "text"] if model.open_weights else ["text"]

    def assess(self, request: FitRequest, card: ReportCard) -> tuple[bool, str]:
        """Whether `card` passes its checks, and one sentence with the numbers that decided it."""
        ok, detail = self._check(request, card)
        return ok, f"{card.bridge} {'passed' if ok else 'failed'}: {detail}."

    def _check(self, request: FitRequest, card: ReportCard) -> tuple[bool, str]:
        """Acceptability and a clause (no subject, no full stop) with the deciding numbers."""
        if card.kind == "vector":
            return self._check_vector(request, card)
        if card.kind == "text":
            bar = card.chance + TEXT_MARGIN
            swapped = card.controls.get("swapped")
            if swapped is not None and card.accuracy + _EPS < swapped + TEXT_SWAPPED_MARGIN:
                return False, (
                    f"its {pct(card.accuracy)} is not 10 points above the {pct(swapped)} it scored with swapped input"
                )
            if card.accuracy + _EPS >= bar:
                return True, (
                    f"it scored {pct(card.accuracy)} against {pct(card.chance)} chance, "
                    f"above the {pct(bar)} bar for a text bridge"
                )
            return False, (
                f"it scored {pct(card.accuracy)}, below the {pct(bar)} bar for a text bridge "
                f"({pct(card.chance)} chance plus 15 points)"
            )
        return False, f"the rule planner only judges vector and text bridges, not {card.kind} bridges"

    def _check_vector(self, request: FitRequest, card: ReportCard) -> tuple[bool, str]:
        bar = card.chance + VECTOR_MARGIN
        swapped = card.controls.get("swapped")
        unseen = card.controls.get("unseen_classes")
        problems: list[str] = []
        if card.accuracy + _EPS < bar:
            problems.append(
                f"it scored {pct(card.accuracy)}, below the {pct(bar)} bar ({pct(card.chance)} chance plus 25 points)"
            )
        if swapped is None:
            problems.append("the swapped-input control is missing")
        elif card.accuracy + _EPS < swapped + VECTOR_MARGIN:
            problems.append(
                f"its {pct(card.accuracy)} is not 25 points above the {pct(swapped)} it scored with swapped input"
            )
        if request.scope == "open":
            if unseen is None:
                problems.append("the unseen-classes control is missing")
            elif unseen + _EPS < bar:
                problems.append(
                    f"it scored only {pct(unseen)} on classes it was not fitted on, below the {pct(bar)} bar"
                )
            elif unseen + _EPS < card.accuracy - OPEN_DROP:
                problems.append(
                    f"it scored {pct(unseen)} on classes it was not fitted on, more than 20 points below its "
                    f"{pct(card.accuracy)} on fitted classes, so it does not hold up on new kinds of input"
                )
        if problems:
            return False, "; ".join(problems)
        detail = f"it scored {pct(card.accuracy)} against {pct(card.chance)} chance and {pct(swapped)} with swapped input"
        if request.scope == "open":
            detail += f", and {pct(unseen)} on classes it was not fitted on"
        return True, detail

    def decide(self, request: FitRequest, cards: list[ReportCard]) -> Decision:
        if not cards:
            return Decision(None, "Installed nothing: no candidate bridge was built and verified.")
        checks = [(card, *self._check(request, card)) for card in cards]
        good = [card for card, ok, _ in checks if ok]
        failed = [f"{card.bridge} failed: {detail}." for card, ok, detail in checks if not ok]
        if not good:
            return Decision(None, "Installed nothing, because no bridge passed its checks. " + " ".join(failed))

        best = max(good, key=lambda c: c.accuracy)
        close_vectors = [
            c for c in good if c.kind == "vector" and c.accuracy + PREFER_VECTOR_WITHIN + _EPS >= best.accuracy
        ]
        chosen = max(close_vectors, key=lambda c: c.accuracy) if close_vectors else best

        reason = f"Installed {chosen.bridge}: " + next(detail for card, _, detail in checks if card is chosen)
        if chosen is not best:
            reason += f", within 3 points of {best.bridge} at {pct(best.accuracy)}, so the vector bridge is preferred"
        elif len(good) > 1:
            runner_up = max((c for c in good if c is not chosen), key=lambda c: c.accuracy)
            if runner_up.accuracy + _EPS >= chosen.accuracy:
                reason += f", level with {runner_up.bridge} at {pct(runner_up.accuracy)}"
                if chosen.kind == "vector":
                    reason += "; on a tie the vector bridge is preferred"
            else:
                reason += f", ahead of {runner_up.bridge} at {pct(runner_up.accuracy)}"
        return Decision(chosen.bridge, " ".join([reason + ".", *failed]))


# ----------------------------------------------------------------------------- registry
_factories: dict[str, Callable[[], Planner]] = {RulePlanner.name: RulePlanner}
_preferred: str | None = None


def register_planner(name: str, factory: Callable[[], Planner], preferred: bool = False) -> None:
    """Add a planner under `name`. With preferred=True, get_planner() returns it by default."""
    global _preferred
    _factories[name] = factory
    if preferred:
        _preferred = name


def get_planner(name: str | None = None) -> Planner:
    """The planner registered as `name`; with no name, the preferred planner, else RulePlanner."""
    if name is None:
        # In fake mode the default is always the rule planner, so tests and the fake dev server never call an API.
        name = RulePlanner.name if config.fake() else (_preferred or RulePlanner.name)
    if name not in _factories:
        raise ValueError(f"No planner is registered as {name!r}. Registered planners: {', '.join(sorted(_factories))}.")
    return _factories[name]()


try:  # an LLM planner registers itself there (preferred=True) when its module is present
    import agent.planner_llm  # noqa: F401
except ImportError as exc:
    if exc.name != "agent.planner_llm":
        logging.getLogger(__name__).warning("The LLM planner could not be loaded (%s); using the rule planner.", exc)
