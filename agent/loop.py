"""One agent run: give a base model a sense, or refuse.

Public surface (do not rename):
  run_fit(request, emit, planner=None, *, on_cards=None, t0=None) -> tuple[list[ReportCard], str | None]
      Returns every candidate's card (with final verdicts and reasons) and the installed Bridge.name, if any.
      `emit(TraceEvent)` is called as the run proceeds; it must be safe to call from a worker thread.
      `on_cards(list[ReportCard])`, if given, is called once, right after the verify step, with the cards
      (verdicts are set later, by the decide step); an exception raised by on_cards propagates and fails
      the run at the verify step. `t0`, if given, is the time.monotonic() value every
      event's `t` counts from, so a caller's earlier events share one clock; by default the run's own start.

Steps, each emitting a "start" event and then "ok" or "fail" (TraceEvent.step in brackets):
  [inspect] load the model and dataset; report what the model accepts natively and whether it is open-weight
  [plan]    planner.plan -> bridge kinds to try. The "ok" event data has "unavailable": the bridges
            bridges.availability(model_info, modality) reports as not available, each {"kind", "why"}
            ([] when sense.bridges has no availability function or it raises). Always verify text, vector
            for open-weight models, and tool for closed-weight models when availability reports it available.
  [fit]     build each candidate with bridges.build(kind, model, dataset, scope), when present; otherwise
            use bridges.fit_vector_bridge on dataset.fit_idx and bridges.make_text_bridge, skipping other kinds.
            Each "ok" event data has "specialist": specialists.describe(modality), read just after that bridge
            is built, or None when sense.specialists has no describe function or it raises
  [verify]  verify.verify for each candidate, the text bridge serving as a control for the vector bridge;
            the "ok" event data carries the ReportCard dict ("card") and the wall time of that verify.verify
            call in seconds, rounded to 2 places ("seconds")
  [decide]  planner.decide -> which bridge, and why
  [install] registry.install for the chosen bridge; other cards get verdict "rejected" with the reason.
            The message names the sense already installed for this model and modality, which the new one
            replaces or, when nothing is installed, chat keeps using (event data "replaced" or "kept")
  [done]    one-line summary, with "installed" and "replaced" or "kept" in its data
Any exception becomes an [error] event with status "fail" and is re-raised.

The text bridge is built from the dataset's labels for scope "closed" and from the caption bank for scope
"open", so that an installed text bridge can describe any sound, not only the labelled ones.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Callable

from agent.planner import get_planner, pct
from sense import bridges, datasets, models, registry, specialists, verify
from sense.contracts import (
    Bridge, BridgeKind, Decision, FitRequest, LabelledSet, ModelInfo, Planner, ReportCard, TraceEvent,
)

_log = logging.getLogger(__name__)

_CONTROL_NAMES = {
    "swapped": "swapped input",
    "no_input": "no input",
    "lookup": "nearest-example lookup",
    "specialist_alone": "specialist alone",
    "text_bridge": "text bridge",
    "unseen_classes": "unseen classes",
}


def run_fit(
    request: FitRequest,
    emit: Callable[[TraceEvent], None],
    planner: Planner | None = None,
    *,
    on_cards: Callable[[list[ReportCard]], None] | None = None,
    t0: float | None = None,
) -> tuple[list[ReportCard], str | None]:
    if t0 is None:
        t0 = time.monotonic()
    step = "inspect"

    def say(status: str, message: str, **data: Any) -> None:
        emit(TraceEvent(step=step, status=status, message=message, data=data, t=round(time.monotonic() - t0, 3)))

    try:
        # [inspect]
        say("start", f"Loading the base model {request.model} and the labelled set {request.dataset}.")
        model = models.load_model(request.model)
        dataset = datasets.load_labelled_set(request.dataset)
        info = model.info
        if dataset.modality != request.modality:
            raise ValueError(
                f"The set {dataset.name} holds {dataset.modality} examples, but the request is for "
                f"{request.modality}. Pick a {request.modality} set."
            )
        reads = " and ".join(info.native) or "nothing"
        native = "including" if request.modality in info.native else "but not"
        weights = "open-weight" if info.open_weights else "not open-weight, so the vector bridge is unavailable"
        say(
            "ok",
            f"{info.label} reads {reads} natively, {native} {request.modality}; it is {weights}. "
            f"{dataset.title}: {len(dataset.labels)} labels, {len(dataset.fit_idx)} examples to fit on, "
            f"{len(dataset.test_idx)} held out.",
            model=info.to_dict(),
            dataset={
                "name": dataset.name,
                "title": dataset.title,
                "labels": list(dataset.labels),
                "n_fit": len(dataset.fit_idx),
                "n_dev": len(dataset.dev_idx),
                "n_test": len(dataset.test_idx),
            },
        )

        # [plan]
        step = "plan"
        planner = planner or get_planner()
        say("start", f"Asking the {planner.name} planner which bridges to try for {request.scope}-scope {request.modality}.")
        kinds = list(planner.plan(request, info, dataset))
        available = _availability(info, request.modality)
        # Every bridge this model can take is verified, whatever the planner proposes: the text bridge is the
        # fallback and a control, and a refused vector bridge must be refused with numbers, not skipped.
        required = (["vector"] if info.open_weights else []) + ["text"]
        if not info.open_weights and any(row["kind"] == "tool" and row["available"] for row in available):
            required.append("tool")
        added = [kind for kind in required if kind not in kinds]
        kinds += added
        note = f" The planner proposed {', '.join(kinds[: len(kinds) - len(added)]) or 'none'}; {', '.join(added)} added so every available bridge is verified." if added else ""
        say("ok", f"Bridges to try, in order: {', '.join(kinds) or 'none'}.{note}", planner=planner.name, kinds=kinds,
            unavailable=[{"kind": row["kind"], "why": row["why"]} for row in available if not row["available"]])

        # [fit]
        step = "fit"
        built: dict[BridgeKind, Bridge] = {}
        build = getattr(bridges, "build", None)
        for kind in kinds:
            if kind in built:
                continue
            say("start", _fit_message(kind, request, dataset), kind=kind)
            started = time.monotonic()
            if build is not None:
                bridge = build(kind, model, dataset, request.scope)
            elif kind == "vector":
                bridge = bridges.fit_vector_bridge(model, dataset, dataset.fit_idx)
            elif kind == "text":
                bridge = bridges.make_text_bridge(model, dataset if request.scope == "closed" else None)
            else:
                say("fail", f"Skipped the {kind} bridge: this prototype can build only vector and text bridges.", kind=kind)
                continue
            built[kind] = bridge
            seconds = round(time.monotonic() - started, 2)
            say("ok", f"Built {bridge.name} in {seconds:.1f} s.{_layer_note(bridge)}", kind=kind, bridge=bridge.name,
                seconds=seconds, layer=getattr(bridge, "layer", None), dev_accuracy=getattr(bridge, "dev_accuracy", None),
                specialist=_specialist(request.modality))

        # [verify]
        step = "verify"
        cards: list[ReportCard] = []
        by_name: dict[str, Bridge] = {}
        for kind, bridge in built.items():
            control = built.get("text") if kind == "vector" else None
            extra = f", including {control.name}" if control is not None else ""
            say("start", f"Scoring {bridge.name} on held-out {dataset.title} examples against controls{extra}.", bridge=bridge.name)
            started = time.monotonic()
            card = verify.verify(model, bridge, dataset, request, text_bridge=control)
            seconds = round(time.monotonic() - started, 2)
            card.created = card.created or datetime.now(timezone.utc).isoformat(timespec="seconds")
            cards.append(card)
            by_name[card.bridge] = bridge
            say("ok", _card_summary(card), card=card.to_dict(), seconds=seconds)
        if on_cards is not None:
            on_cards(cards)

        # [decide]
        step = "decide"
        say("start", f"Asking the {planner.name} planner to choose among {len(cards)} verified candidates.")
        decision = planner.decide(request, cards)
        if decision.install is not None and decision.install not in by_name:
            raise ValueError(
                f"The planner chose {decision.install}, which is not one of the verified candidates "
                f"({', '.join(by_name) or 'none'})."
            )
        chosen = _apply_verdicts(planner, request, cards, decision)
        say("ok", decision.reason, install=decision.install, cards=[c.to_dict() for c in cards])

        # [install]
        step = "install"
        earlier = _earlier_sense(request)  # (bridge name, description) of the sense chat uses now, if any
        if chosen is None and earlier is None:
            say("info", f"Installed nothing; {info.label} still cannot take {request.modality} input.")
        elif chosen is None:
            say("info", f"Installed nothing; {info.label} keeps the earlier {earlier[1]}, which chat still uses.",
                kept=earlier[0])
        else:
            replacing = f", replacing the earlier {earlier[1]}" if earlier else ""
            say("start", f"Installing {chosen.bridge} for {info.label}{replacing}.", bridge=chosen.bridge,
                replaced=earlier[0] if earlier else None)
            registry.install(by_name[chosen.bridge], chosen)
            say("ok", f"Installed {chosen.bridge}; {info.label} can now take {request.modality} input in chat.",
                bridge=chosen.bridge, replaced=earlier[0] if earlier else None)

        # [done]
        step = "done"
        if chosen is None:
            if not cards:
                why = "no candidate bridge could be built"
            else:
                why = "the planner selected no bridge"
            if earlier is None:
                summary = f"No {request.modality} sense was installed for {info.label}: {why}."
            else:
                summary = (
                    f"No new {request.modality} sense was installed for {info.label}: {why}. "
                    f"Chat keeps using the earlier {earlier[1]}."
                )
        else:
            summary = (
                f"{info.label} can now take {request.modality} input through {chosen.bridge}: it scored "
                f"{pct(chosen.accuracy)} on {chosen.n} held-out questions, against {pct(chosen.chance)} chance."
            )
            if chosen.kind == "vector" and request.scope == "closed":
                summary += (
                    f" This sense covers the {len(dataset.labels)} categories of {dataset.title} only; "
                    "a clip outside them will be given the nearest of those labels."
                )
            if earlier is not None:
                summary += f" It replaced the earlier {earlier[1]}."
        outcome = {}
        if earlier is not None:
            outcome["replaced" if chosen else "kept"] = earlier[0]
        say("ok", summary, installed=decision.install, **outcome)
        return cards, decision.install

    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
        failed = step
        say("fail", detail)
        step = "error"
        say("fail", f"The agent run stopped during the {failed} step. {detail}", step=failed, error=type(exc).__name__)
        raise


def _availability(info: ModelInfo, modality: str) -> list[dict[str, Any]]:
    """Read bridge availability once for both required candidates and trace facts; return [] if unavailable."""
    availability = getattr(bridges, "availability", None)
    if availability is None:
        return []
    try:
        return [{"kind": row["kind"], "available": row["available"], "why": row["why"]}
                for row in availability(info, modality)]
    except Exception:
        _log.warning("bridges.availability failed for %s; reporting no availability facts", modality, exc_info=True)
        return []


def _specialist(modality: str) -> dict[str, Any] | None:
    """specialists.describe(modality), or None when that function does not exist or fails (it raises ValueError
    for a modality with no specialist): a trace fact must never fail a run."""
    describe = getattr(specialists, "describe", None)
    if describe is None:
        return None
    try:
        return describe(modality)
    except Exception:
        _log.warning("specialists.describe failed for %s; reporting no specialist", modality, exc_info=True)
        return None


def _earlier_sense(request: FitRequest) -> tuple[str, str] | None:
    """The installed sense of this model and modality, as (bridge name, 'vector:gtzan sense (79.5% on gtzan,
    closed scope)'), or None. An unreadable card does not stop the run: it is described without numbers."""
    try:
        entries = [e for e in registry.installed(request.model) if e.get("modality") == request.modality]
    except (OSError, ValueError, KeyError):
        return None
    if not entries:
        return None
    entry, card = entries[0], entries[0].get("card") or {}
    try:
        numbers = f" ({pct(float(card['accuracy']))} on {card['dataset']}, {card['scope']} scope)"
    except (KeyError, TypeError, ValueError):
        numbers = ""
    return entry["bridge"], f"{entry['bridge']} sense{numbers}"


def _fit_message(kind: str, request: FitRequest, dataset: LabelledSet) -> str:
    if kind == "vector":
        return f"Fitting the vector bridge on {len(dataset.fit_idx)} labelled examples of {dataset.title}."
    if kind == "text" and request.scope == "closed":
        return f"Building the text bridge, which names the closest of the {len(dataset.labels)} labels."
    if kind == "text":
        return "Building the text bridge, which describes each clip with its nearest caption."
    return f"Building the {kind} bridge."


def _layer_note(bridge: Bridge) -> str:
    """' It writes after block 2, chosen on dev questions (block 0: 70.0%, ...).' for a vector bridge, else ''."""
    layer, dev = getattr(bridge, "layer", None), getattr(bridge, "dev_accuracy", None)
    if layer is None:
        return ""
    if not dev:
        return f" It writes after block {layer}."
    scores = ", ".join(f"block {k}: {pct(v)}" for k, v in sorted(dev.items()))
    return f" It writes after block {layer}, chosen on dev questions ({scores})."


def _card_summary(card: ReportCard) -> str:
    """'vector:gtzan scored 84.9% on 200 held-out questions (chance 20.0%); controls: swapped input 19.5%, ...'"""
    controls = ", ".join(f"{_CONTROL_NAMES.get(k, k)} {pct(v)}" for k, v in card.controls.items())
    return (
        f"{card.bridge} scored {pct(card.accuracy)} on {card.n} held-out questions (chance {pct(card.chance)})"
        + (f"; controls: {controls}." if controls else ".")
    )


def _apply_verdicts(
    planner: Planner, request: FitRequest, cards: list[ReportCard], decision: Decision
) -> ReportCard | None:
    """Mark the chosen card "installed" and every other card "rejected", each with a reason. Returns the chosen card."""
    assess = getattr(planner, "assess", None)
    chosen = None
    for card in cards:
        if card.bridge == decision.install:
            card.verdict, card.reason = "installed", decision.reason
            chosen = card
            continue
        card.verdict = "rejected"
        if callable(assess):
            ok, why = assess(request, card)
            if not ok:
                card.reason = why
            elif decision.install is None:
                card.reason = f"{why} It was not selected; no bridge selected. {decision.reason}"
            else:
                card.reason = f"{why} It was not installed because {decision.install} was chosen instead."
        elif decision.install is None:
            card.reason = f"{_card_summary(card)} Nothing was installed: {decision.reason}"
        else:
            card.reason = f"{_card_summary(card)} {decision.install} was installed instead: {decision.reason}"
    return chosen
