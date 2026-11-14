"""One agent run: give a base model a sense, or refuse.

Public surface (do not rename):
  run_fit(request, emit, planner=None) -> tuple[list[ReportCard], str | None]
      Returns every candidate's card (with final verdicts and reasons) and the installed Bridge.name, if any.
      `emit(TraceEvent)` is called as the run proceeds; it must be safe to call from a worker thread.

Steps, each emitting a "start" event and then "ok" or "fail" (TraceEvent.step in brackets):
  [inspect] load the model and dataset; report what the model accepts natively and whether it is open-weight
  [plan]    planner.plan -> bridge kinds to try
  [fit]     build each candidate (bridges.fit_vector_bridge on dataset.fit_idx, bridges.make_text_bridge)
  [verify]  verify.verify for each candidate, the text bridge serving as a control for the vector bridge;
            the event data carries the ReportCard dict
  [decide]  planner.decide -> which bridge, and why
  [install] registry.install for the chosen bridge; other cards get verdict "rejected" with the reason
  [done]    one-line summary
Any exception becomes an [error] event with status "fail" and is re-raised.
"""

from __future__ import annotations

from typing import Callable

from sense.contracts import FitRequest, Planner, ReportCard, TraceEvent


def run_fit(
    request: FitRequest,
    emit: Callable[[TraceEvent], None],
    planner: Planner | None = None,
) -> tuple[list[ReportCard], str | None]:
    raise NotImplementedError
