"""Verification: score a candidate bridge on held-out examples against controls.

Public surface (do not rename):
  verify(model, bridge, dataset, request, text_bridge=None, max_items=200, seed=0) -> ReportCard

Each held-out example (dataset.test_idx, at most max_items) becomes a five-option question: dataset.question,
the true label and four other labels in a seeded random order. Chance is 0.2 (or 1/len(labels) if fewer than
five labels). The card's `accuracy` is the bridge's; `controls` holds:
  "swapped"         the same questions with each input taken from an example of a different label
  "no_input"        the same questions with no bridged input
  "lookup"          answer with the label of the nearest fitted example in specialist space (no base model)
  "text_bridge"     accuracy of `text_bridge` on the same questions, when one is given
  "unseen_classes"  only for request.scope == "open" and vector bridges: refit the bridge with a fifth of the
                    labels left out and score it on held-out examples of those labels
The card comes back with verdict "candidate"; the planner decides.
"""

from __future__ import annotations

from sense.contracts import BaseModel, Bridge, FitRequest, LabelledSet, ReportCard


def verify(
    model: BaseModel,
    bridge: Bridge,
    dataset: LabelledSet,
    request: FitRequest,
    text_bridge: Bridge | None = None,
    max_items: int = 200,
    seed: int = 0,
) -> ReportCard:
    raise NotImplementedError
