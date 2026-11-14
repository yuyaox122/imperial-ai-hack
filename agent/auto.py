"""Detect an input, research a specialist and labelled data, then verify a bridge.

Public surface: run_auto(model, file, goal, emit, planner=None, *, examples=None,
                         on_request=None) -> (cards, installed).

Trace contract for the interface (all times use the same monotonic start):
  [detect]   start, ok: file, goal, modality, format, how, detail.
  [research] start, ok: origin, chosen, fallback, note, claims.
  [data]     start then ok/fail/info for need, search, fetch, build and fallback;
             substep events carry action and relevant need/candidate/reason facts.
             Final ok: dataset, origin (user/fetched/prewired), source, licence,
             labels, n_files, need, claims, meta (authoritative built-set metadata).
  [inspect], [plan], [fit], [verify] delegate to agent.loop.run_fit; see its contract.
  [claims]   start, ok after verification: claims, measured_specialist; synthetic marks fake-mode evidence.
  [decide], [install], [done] delegate to run_fit after claims are settled.
             Early done ok: installed=None and native=True or needs_user=True;
             unsupported/no-data refusals also include the accumulated claims.
  [error]    fail: step, error, preceded by a failure at that step; re-raises.

on_request publishes the resolved FitRequest before fitting. A supplied folder is
closed scope. Research and sample failures allow a goal-matching prewired fallback;
invalid user folders fail visibly instead of silently using unrelated examples.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

from agent import data, discover, loop, research
from agent.claims import Claim, citable, settle_dataset_claim, settle_specialist_claims
from agent.detect import detect
from sense import config, datasets
from sense import models
from sense.contracts import FitRequest, Planner, ReportCard, TraceEvent


def run_auto(model: str, file: Path, goal: str, emit: Callable[[TraceEvent], None],
             planner: Planner | None = None, *, examples: Path | None = None,
             on_request: Callable[[FitRequest], None] | None = None) -> tuple[list[ReportCard], str | None]:
    t0 = time.monotonic()
    step = 'detect'
    error_emitted = False
    claims: list[Claim] = []

    def forward(event: TraceEvent) -> None:
        nonlocal error_emitted
        if event.step == 'error':
            error_emitted = True
        emit(event)

    def say(status: str, message: str, **facts: Any) -> None:
        forward(TraceEvent(step, status, message, facts, round(time.monotonic() - t0, 3)))

    def ledger() -> list[dict]:
        return [claim.to_dict() for claim in claims]

    def metadata(name: str) -> dict:
        # Public catalogue includes the actual retained files after folder validation.
        entry = next((entry for entry in datasets.available_sets() if entry['name'] == name), None)
        if (entry is None or not isinstance(entry.get('labels'), list) or len(entry['labels']) < 2
                or not isinstance(entry.get('n'), int) or entry['n'] <= 0):
            raise data.DataUnavailable(f'No authoritative usable-set metadata for {name}.')
        return entry

    try:
        file = Path(file)
        say('start', f'Detecting the contents of {file.name}.')
        detected = detect(file)
        info = next((info for info in models.available_models() if info.name == model), None)
        if info is None:
            raise ValueError(f'Unknown model {model!r}; choose a model from available_models().')
        say('ok', detected.detail, file=file.name, goal=goal, **detected.to_dict())
        modality = detected.modality
        if modality in info.native:
            step = 'done'
            say('ok', f'{info.label} reads {modality} natively, so no sense is needed.', installed=None, native=True)
            return [], None

        step = 'research'
        say('start', f'Researching a specialist for {modality}: {goal}')
        discovery = discover.find_specialists(modality, detected.format, goal)
        claims.extend(discovery.claims)
        say('ok', ('Synthetic research fixtures. ' if discovery.origin == 'fake' else '') + discovery.note,
            **discovery.to_dict())
        if discovery.chosen is None and not discovery.fallback:
            step = 'done'
            say('ok', f'No verified bridge for {detected.format} files. {discovery.note} '
                f'To verify one I need about {data.MIN_PER_LABEL} labelled examples for each of at least two '
                'categories, and a specialist model that can be loaded here.',
                installed=None, needs_user=True, claims=ledger())
            return [], None

        step = 'data'
        name = None
        source = licence = ''
        labels: list[str] = []
        n_files = 0
        if examples is not None:
            examples = Path(examples)
            need = data.DataNeed('labelled', 'closed', [], 'The user supplied labelled categories to verify.')
            origin, source, licence = 'user', str(examples), 'user supplied'
            say('start', f'Building the labelled set from {examples}.', action='build', need=need.to_dict())
            name = data.build_set(examples, data.safe_label(examples.name, 0), title=examples.name,
                                  question=f'Which of these is in the {modality} clip?', modality=modality,
                                  source=source, licence=licence)
            entry = metadata(name)
            labels, n_files = entry.get('labels', []), entry.get('n', 0)
        else:
            say('start', f'Deciding which labelled examples the goal needs: {goal}', action='need')
            need = data.decide_need(goal, detected)
            say('ok', need.reason, action='need', need=need.to_dict())
            say('start', 'Searching for labelled datasets.', action='search')
            try:
                candidates, dataset_claims, search_origin = data.find_datasets(need, modality, goal)
            except (data.DataUnavailable, research.ResearchUnavailable) as exc:
                candidates, dataset_claims = [], []
                say('fail', f'Dataset search was unavailable: {exc}', action='search', reason=str(exc))
            else:
                say('ok', f'Found {len(candidates)} dataset candidates.', action='search',
                    origin=search_origin, claims=[claim.to_dict() for claim in dataset_claims])
            claims.extend(dataset_claims)
            attempts = 0
            for candidate in candidates:
                matching = [claim for claim in dataset_claims if claim.subject == candidate.hub_id]
                if any(claim.status == 'refuted' for claim in matching) and not candidate.observed.get('hub_exists'):
                    continue
                if attempts == 2:
                    break
                attempts += 1
                say('start', f'Fetching labelled examples from {candidate.hub_id}.', action='fetch',
                    candidate=candidate.to_dict())
                try:
                    report = data.fetch_sample(candidate, config.home() / 'fetched' / data.safe_label(candidate.hub_id, 0), need.scope,
                                               requested_labels=need.labels)
                    say('ok', f'Fetched {report.n_files} files across {len(report.labels)} labels.', action='fetch',
                        report=report.to_dict())
                    say('start', f'Building the labelled set {candidate.title}.', action='build')
                    name = data.build_set(report.folder, data.safe_label(candidate.hub_id, 0), title=candidate.title,
                                          question=f'Which of these is in the {modality} clip?', modality=modality,
                                          source=candidate.source_url, licence=report.licence or '')
                    entry = metadata(name)
                    data.require_coverage(need.labels, entry['labels'])
                except (data.DataUnavailable, research.ResearchUnavailable) as exc:
                    name = None
                    for claim in matching:
                        if claim.status != 'refuted':
                            claim.status, claim.evidence = 'not_testable', str(exc)
                    say('fail', f'Could not use {candidate.hub_id}: {exc}', action='candidate', reason=str(exc), claims=ledger())
                    continue
                origin = 'fetched'
                meta = entry.get('meta') or {}
                source, licence = meta.get('source', ''), meta.get('licence', '')
                labels, n_files = entry['labels'], entry['n']
                for claim in matching:
                    claim.observed.update(candidate.observed)
                    claim.observed['sample'] = {'n_files': n_files, 'labels': list(labels),
                                                'downloaded_files': report.n_files}
                    claim.observed['selected_split'] = report.selected_split or {'n_rows': report.n_rows_total}
                    settle_dataset_claim(claim)
                break
            for claim in dataset_claims:
                if claim.status == 'unverified':
                    claim.status, claim.evidence = 'not_testable', 'Not sampled: an earlier set succeeded or the two-candidate limit was reached.'
            if name is None:
                say('start', 'Looking for a prewired set matching the goal.', action='fallback')
                name = data.choose_prewired(goal, modality, requested_labels=need.labels)
                if name is not None:
                    try:
                        entry = metadata(name)
                        data.require_coverage(need.labels, entry['labels'])
                    except data.DataUnavailable as exc:
                        say('fail', str(exc), action='fallback', reason=str(exc))
                        name = None
                if name is None:
                    say('info', 'No matching prewired labelled set is available.', action='fallback')
                    step = 'done'
                    say('ok', f'I could not find labelled {modality} examples for this goal. Put at least '
                        f'{data.MIN_PER_LABEL} clips per category in a folder with one sub-folder per label '
                        'and run again with that folder.', installed=None, needs_user=True, claims=ledger())
                    return [], None
                origin = 'prewired'
                meta = entry.get('meta') or {}
                source, licence = meta.get('source', ''), meta.get('licence', '')
                labels, n_files = entry.get('labels', []), entry.get('n', 0)
        say('ok', f'Using {name} ({origin}) to verify the goal.', dataset=name, origin=origin, source=source,
            licence=licence, labels=labels, n_files=n_files, need=need.to_dict(), claims=ledger(),
            meta=entry.get('meta') or {})
        request = FitRequest(model=model, dataset=name, modality=modality, scope=need.scope)
        if on_request is not None:
            on_request(request)

        def on_cards(cards: list[ReportCard]) -> None:
            nonlocal step
            step = 'claims'
            say('start', 'Checking research claims against the specialist-alone control.')
            synthetic = config.fake()
            measured = (entry.get('meta') or {}).get('specialist')
            if synthetic and (measured == 'fake' or entry.get('origin') == 'builtin'):
                measured = 'fake-clap'
            # Folder metadata identifies the actual cached embeddings. Built-in real sets
            # carry no such provenance; the currently configured encoder cannot prove it.
            used = measured if (isinstance(measured, str) and citable(measured, 'https://huggingface.co')
                                and not Path(measured).exists()) else None
            if synthetic:
                used = 'fake-clap' if measured == 'fake-clap' else None
            settle_specialist_claims(claims, used, cards,
                                     synthetic_subject=discovery.chosen if synthetic and discovery.origin == 'fake' else None)
            if synthetic:
                for claim in claims:
                    if claim.kind == 'specialist':
                        claim.evidence = (f'Synthetic fake-mode check; measured identity {measured!r}; '
                                          'no live specialist was measured. ' + claim.evidence)
            counts = {status: sum(claim.status == status for claim in claims)
                      for status in ('confirmed', 'refuted', 'not_testable')}
            say('ok', f'Claims: {counts["confirmed"]} confirmed, {counts["refuted"]} refuted, '
                f'{counts["not_testable"]} not testable.' + (' Synthetic verification only.' if synthetic else ''),
                claims=ledger(), synthetic=synthetic, measured_specialist=measured)

        step = 'inspect'
        return loop.run_fit(request, forward, planner, on_cards=on_cards, t0=t0)
    except Exception as exc:
        if not error_emitted:
            detail = f'{type(exc).__name__}: {exc}' if str(exc) else type(exc).__name__
            failed = step
            say('fail', detail)
            step = 'error'
            say('fail', f'The agent run stopped during the {failed} step. {detail}', step=failed, error=type(exc).__name__)
        raise
