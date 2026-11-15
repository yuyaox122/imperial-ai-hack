"""The claims ledger: what a web source claimed about a model or dataset, and whether our own measurement
confirmed it.

Public surface (do not rename):
  Claim                                            one claim; to_dict() for the trace
  MARGIN = 0.25                                    how far above chance the specialist alone must score
  settle_specialist_claims(claims, used, cards, *, synthetic_subject=None)
                                                   sets specialist status/evidence against measured identity
  settle_dataset_claim(claim)                       compares structured assertions with observed facts
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
from dataclasses import asdict, dataclass, field

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

    assertions: dict = field(default_factory=dict)  # structured source facts, never inferred from prose
    observed: dict = field(default_factory=dict)  # independent facts with their scope

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


def settle_specialist_claims(claims: list[Claim], used: str | None, cards: list[ReportCard], *,
                             synthetic_subject: str | None = None) -> None:
    card = next((c for c in cards if "specialist_alone" in (c.controls or {})), None)
    for claim in claims:
        if claim.kind != "specialist" or claim.status != "unverified":
            continue
        synthetic_match = used == "fake-clap" and claim.subject == synthetic_subject
        if claim.subject != used and not synthetic_match:
            claim.status, claim.evidence = "not_testable", "not loaded in this harness, so not run on our data"
        elif card is None:
            claim.status, claim.evidence = "not_testable", "the specialist-alone control was not run"
        else:
            _settle(claim, card)
            if synthetic_match:
                claim.evidence = "Synthetic fixture checked with fake-clap, not its named live model. " + claim.evidence


def _settle(claim: Claim, card: ReportCard) -> None:
    s, c, n = float(card.controls["specialist_alone"]), float(card.chance), card.n
    # Rounded so that a score exactly on the margin (0.45 with chance 0.2) is not lost to float error.
    claim.status = "confirmed" if round(s - c, 9) >= MARGIN else "refuted"
    evidence = f"On our data the specialist alone scored {s:.1%} on {n} held-out questions (chance {c:.1%})"
    if claim.value is not None:
        evidence += f"; the source claims {claim.value:.1%} on {claim.metric or 'its benchmark'}, a different test"
    claim.evidence = evidence + "."


def settle_dataset_claim(claim: Claim) -> None:
    """A compound source claim is confirmed only when every asserted fact is established.

    Hub licence is comparable to a source licence. A bounded sample and the selected
    split's size do not establish the whole dataset's size or complete label vocabulary.
    Keep those measurements visible without silently promoting them to source truth.
    """
    if claim.kind != "dataset":
        return
    facts = claim.observed
    evidence, outcomes = [], []
    if facts.get("hub_exists") is False:
        claim.status, claim.evidence = "refuted", "no such dataset on the Hugging Face Hub"
        return
    if "hub_error" in facts:
        evidence.append("The dataset's Hub record could not be checked: " + facts["hub_error"])
    if facts.get("hub_exists"):
        evidence.append("The Hub has this dataset")
    for key, asserted in claim.assertions.items():
        if key == "licence" and facts.get("licence"):
            observed = facts["licence"]
            outcomes.append("confirmed" if asserted.casefold() == observed.casefold() else "refuted")
            evidence.append(f"source licence {asserted}; Hub licence {observed}")
        else:
            outcomes.append("not_testable")
            evidence.append(f"source {key} {asserted!r} is not independently established for the whole dataset")
    if "sample" in facts:
        evidence.append(f"built sample: {facts['sample']!r}")
    if "selected_split" in facts:
        evidence.append(f"viewer selected split (not whole dataset): {facts['selected_split']!r}")
    claim.status = ("refuted" if "refuted" in outcomes else
                    "confirmed" if outcomes and all(s == "confirmed" for s in outcomes) else "not_testable")
    if not claim.assertions:
        evidence.append("No structured source facts to test; acquisition does not establish free-prose assertions")
    claim.evidence = "; ".join(evidence) + "."
