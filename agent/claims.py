"""The claims ledger: what a web source claimed about a model or dataset, and whether our own measurement
confirmed it.

Public surface (do not rename):
  Claim                                            one claim; to_dict() for the trace
  MARGIN = 0.25                                    how far above chance the specialist alone must score
  settle_specialist_claims(claims, used, cards)    sets status and evidence of the specialist claims in place
  citable(hub_id, source_url) -> bool              the id and URL checks shared by agent.discover and agent.data
  fraction(value) -> float | None                  a claimed score as a fraction 0..1, or None

A claim starts "unverified". settle_specialist_claims settles each specialist claim that is still unverified
(a claim already refuted, such as an id the Hub does not have, keeps its status and evidence):
  - its subject is not the specialist `used` in this run -> "not_testable";
  - no card has a "specialist_alone" control -> "not_testable";
  - else, with s that control, and the chance and n of the same card: s >= chance + MARGIN -> "confirmed",
    otherwise "refuted". The evidence quotes our number, and the source's number when it has one, noting that
    the source measured it on a different test.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass

from sense.contracts import ReportCard

MARGIN = 0.25
_HUB_ID = re.compile(r"^[\w.-]+/[\w.-]+$")


@dataclass
class Claim:
    subject: str  # hub id
    kind: str  # "specialist" | "dataset"
    statement: str  # the claim as the source states it
    source_url: str
    source_name: str = ""
    metric: str | None = None
    value: float | None = None  # 0..1
    status: str = "unverified"  # "unverified" | "confirmed" | "refuted" | "not_testable"
    evidence: str = ""  # one sentence with our own number

    def to_dict(self) -> dict:
        return asdict(self)


def citable(hub_id: object, source_url: object) -> bool:
    """A Hub id of the form owner/name, and a source page reached over http or https."""
    return (
        isinstance(hub_id, str)
        and _HUB_ID.match(hub_id) is not None
        and isinstance(source_url, str)
        and source_url.startswith(("http://", "https://"))
    )


def fraction(value: object) -> float | None:
    """A finite number in 0..1 as given, or in (1, 100] read as a percentage; anything else is None."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    if 0 <= value <= 1:
        return float(value)
    if 1 < value <= 100:
        return value / 100
    return None


def settle_specialist_claims(claims: list[Claim], used: str | None, cards: list[ReportCard]) -> None:
    card = next((c for c in cards if "specialist_alone" in (c.controls or {})), None)
    for claim in claims:
        if claim.kind != "specialist" or claim.status != "unverified":
            continue
        if claim.subject != used:
            claim.status, claim.evidence = "not_testable", "not loaded in this harness, so not run on our data"
        elif card is None:
            claim.status, claim.evidence = "not_testable", "the specialist-alone control was not run"
        else:
            _settle(claim, card)


def _settle(claim: Claim, card: ReportCard) -> None:
    s, c, n = float(card.controls["specialist_alone"]), float(card.chance), card.n
    # Rounded so that a score exactly on the margin (0.45 with chance 0.2) is not lost to float error.
    claim.status = "confirmed" if round(s - c, 9) >= MARGIN else "refuted"
    evidence = f"On our data the specialist alone scored {s:.1%} on {n} held-out five-option questions (chance {c:.1%})"
    if claim.value is not None:
        evidence += f"; the source claims {claim.value:.1%} on {claim.metric or 'its benchmark'}, a different test"
    claim.evidence = evidence + "."
