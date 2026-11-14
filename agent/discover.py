"""Specialist discovery: ask the web which pretrained encoders fit the goal, check each id on the Hugging Face Hub,
and pick one this harness can actually run.

Public surface (do not rename):
  RUNNABLE                                      hub id -> modality, for what sense.specialists can load today
  SPECIALIST_SCHEMA                             the structured-output schema; top-level property "specialists"
  Discovery                                     claims, chosen, fallback, origin, note; to_dict() for the trace
  find_specialists(modality, fmt, goal) -> Discovery

Each search result becomes a specialist Claim when its hub_id has the form owner/name and its source_url is an
http(s) page; anything else is dropped. A claim whose id the Hub does not have is refuted. `chosen` is the first
claim the Hub has whose RUNNABLE modality is `modality` (the Hub's pipeline_tag is never used: CLAP is tagged
"feature-extraction"). With nothing runnable, or with research unavailable, `chosen` is None and `fallback` is
True when RUNNABLE holds a pre-wired specialist for the modality. All network access goes through agent.research.
"""

from __future__ import annotations

from dataclasses import dataclass

from agent import research
from agent.claims import Claim, citable, fraction

RUNNABLE = {"laion/clap-htsat-unfused": "audio"}  # what sense.specialists can load today

_STR = {"type": "string"}
_NULLABLE_STR = {"type": ["string", "null"]}
SPECIALIST_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "specialists": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "hub_id": _STR,
                    "statement": _STR,
                    "source_url": _STR,
                    "source_name": _STR,
                    "metric": _NULLABLE_STR,
                    "value": {"type": ["number", "null"]},
                },
                "required": ["hub_id", "statement", "source_url"],
            },
        }
    },
    "required": ["specialists"],
}


@dataclass
class Discovery:
    claims: list[Claim]
    chosen: str | None  # hub id of the runnable specialist picked, or None
    fallback: bool  # True when the pre-wired specialist is used because research gave nothing runnable
    origin: str  # "live" | "cache" | "fake" | "unavailable"
    note: str  # one sentence for the trace

    def to_dict(self) -> dict:
        return {
            "claims": [c.to_dict() for c in self.claims],
            "chosen": self.chosen,
            "fallback": self.fallback,
            "origin": self.origin,
            "note": self.note,
        }


def find_specialists(modality: str, fmt: str, goal: str) -> Discovery:
    # fmt is part of the fixed signature; the query is about the modality, so it is not used here.
    query = (
        f"Which open pretrained {modality} encoder models on Hugging Face are best for: {goal}? For each, give the "
        "Hugging Face model id and one benchmark result the source states, with the page it comes from."
    )
    prewired = _prewired(modality)
    try:
        found = research.search_structured(query, SPECIALIST_SCHEMA)
    except research.ResearchUnavailable as exc:
        tail = f"so the pre-wired specialist {prewired} is used" if prewired else f"and no {modality} model loads here"
        return Discovery([], None, prewired is not None, "unavailable", f"Research was unavailable ({exc}), {tail}.")
    claims = _specialist_claims(found)
    chosen = None
    for claim in claims:
        if research.hub_model(claim.subject) is None:
            claim.status, claim.evidence = "refuted", "no such model on the Hugging Face Hub"
        elif chosen is None and RUNNABLE.get(claim.subject) == modality:
            chosen = claim.subject
    return Discovery(claims, chosen, chosen is None and prewired is not None, found.origin,
                     _note(len(claims), chosen, prewired, modality))


def _specialist_claims(found: research.Research) -> list[Claim]:
    """One Claim per usable item of the answer, in order."""
    items = found.data.get("specialists") if isinstance(found.data, dict) else None
    names = {s.get("url"): s.get("name") for s in found.sources if isinstance(s, dict)}
    claims = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict) or not citable(item.get("hub_id"), item.get("source_url")):
            continue
        url = item["source_url"]
        metric = item.get("metric")
        claims.append(Claim(
            subject=item["hub_id"],
            kind="specialist",
            statement=_text(item.get("statement")),
            source_url=url,
            source_name=_text(item.get("source_name")) or _text(names.get(url)),
            metric=metric.strip() if isinstance(metric, str) and metric.strip() else None,
            value=fraction(item.get("value")),
        ))
    return claims


def _prewired(modality: str) -> str | None:
    return next((hub_id for hub_id, m in RUNNABLE.items() if m == modality), None)


def _note(n: int, chosen: str | None, prewired: str | None, modality: str) -> str:
    named = f"Research named {n} candidate model{'' if n == 1 else 's'}"
    if chosen:
        return f"{named}; {chosen} exists on the Hub and can be loaded here, so it is used."
    if prewired:
        return f"{named}, none both on the Hub and loadable here, so the pre-wired specialist {prewired} is used."
    return f"{named}, and no {modality} specialist loads here."


def _text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""
