"""Command line for a complete autonomous sense-fitting run: python -m agent --model M --file F --goal G.

  python -m agent --model qwen3.5-0.8b --file clip.wav --goal "recognise music genres"
  python -m agent --model openai-chat --file chair.obj --goal "what object is this?" --save runs/chair.json

Prints each trace event as it happens, then every report card and every research claim.
--examples PATH   a folder with one sub-folder of labelled files per label, used instead of searching for data.
--save PATH       when the run ends, finished or failed, write it to PATH as JSON in the sense.runs format (as
                  scripts/demo_cli.py --save writes it): "id", "request", "status" ("done" or "failed"), "events"
                  (each with "t", seconds since the run started), "cards", "installed" and "error" exactly as in
                  JobState.to_dict(), plus "origin" ("cli"), "recorded" (when the run started, ISO 8601 UTC),
                  "seconds" (the run's wall time) and "machine" ({"platform", "python", "machine"}).
                  sense.runs.import_record reads it back for replay.
Approvals. When the agent asks to install packages into the adapter toolbox (agent.jobs.request_approval outside a
job), the plan is printed and the terminal asks "Install? [y/N]"; anything but y or yes, or no terminal input,
declines. --approve-installs approves every such request without asking, and says so loudly each time.
Exit status: 0 when the run finished (installed, refused or needing the user), 1 when it failed or --save could not
write its file.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from agent import jobs
from agent.auto import run_auto
from sense.contracts import FitRequest, JobState, ReportCard, TraceEvent

LOUD = "!" * 78


def _print_plan(payload: dict) -> None:
    plan = payload.get("plan") if isinstance(payload, dict) else None
    print(f"\nThe agent asks to install packages into the adapter toolbox: {payload.get('summary') or ''}", flush=True)
    for package in (plan or {}).get("packages") or []:
        reasons = "; ".join(package.get("reasons") or []) or "no concerns"
        print(f"  {package.get('name')} {package.get('version')}: {package.get('verdict')} ({reasons})")
    if payload.get("review"):
        print(f"  Review: {payload['review']}")


def ask_on_terminal(kind: str, payload: dict) -> bool:
    """agent.jobs' CLI handler: show the request and ask y/N on the terminal; No unless the answer is y or yes."""
    if kind == "install":
        _print_plan(payload)
    else:
        print(f"\nThe agent asks for your approval ({kind}): {payload.get('summary') or payload}", flush=True)
    try:
        answer = input("Install? [y/N] " if kind == "install" else "Approve? [y/N] ")
    except EOFError:  # no terminal input: nobody approved it
        print("\nNo answer, so it counts as declined.")
        return False
    approved = answer.strip().lower() in ("y", "yes")
    print("Approved." if approved else "Declined.", flush=True)
    return approved


def approve_without_asking(kind: str, payload: dict) -> bool:
    """agent.jobs' CLI handler for --approve-installs: approve, and say so loudly."""
    if kind == "install":
        _print_plan(payload)
    print(f"{LOUD}\n!! --approve-installs: APPROVED WITHOUT ASKING ({kind}): {payload.get('summary') or ''}\n{LOUD}",
          flush=True)
    return True


def save_record(path: Path, request: FitRequest, events: list[TraceEvent], cards: list[ReportCard],
                installed: str | None, error: str | None, recorded: str, seconds: float) -> bool:
    """Write the run record to `path` (module docstring). Returns False, after saying why, if it cannot."""
    job = JobState(id=uuid.uuid4().hex[:12], request=request, status="failed" if error else "done",
                   events=events, cards=cards, installed=installed, error=error)
    machine = {"platform": platform.platform(), "python": platform.python_version(), "machine": platform.machine()}
    record = {"id": job.id, "origin": "cli", "recorded": recorded} | job.to_dict() | {
        "seconds": round(seconds, 2), "machine": machine}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=2) + "\n")
    except OSError as exc:
        print(f"Could not save the run record to {path} ({exc}). Pass a writable path to --save.", file=sys.stderr)
        return False
    print(f"Saved the run record ({job.status}, {len(events)} events) to {path}.")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Detect, research and verify a new sense for a model.')
    parser.add_argument('--model', required=True)
    parser.add_argument('--file', required=True, type=Path)
    parser.add_argument('--goal', required=True)
    parser.add_argument('--examples', type=Path)
    parser.add_argument('--save', metavar='PATH', type=Path,
                        help='when the run ends, write it (request, timed trace events, report cards) to PATH as JSON')
    parser.add_argument('--approve-installs', action='store_true',
                        help='approve every package install the agent asks for, without asking')
    args = parser.parse_args(argv)
    claims: dict[tuple[str, str, str], dict] = {}
    events: list[TraceEvent] = []
    request = FitRequest(model=args.model, dataset='', modality='', scope='closed')

    def emit(event: TraceEvent) -> None:
        events.append(event)
        print(f'[{event.step}] {event.status}  {event.message}', flush=True)
        for claim in event.data.get('claims', []):
            claims[(claim['kind'], claim['subject'], claim['source_url'])] = claim

    def on_request(resolved: FitRequest) -> None:
        nonlocal request
        request = resolved

    if args.approve_installs:
        print(f'{LOUD}\n!! --approve-installs: every package install the agent asks for is approved WITHOUT ASKING.'
              f'\n{LOUD}', flush=True)
    jobs.set_cli_handler(approve_without_asking if args.approve_installs else ask_on_terminal)
    recorded = datetime.now(timezone.utc).isoformat(timespec='seconds')
    started = time.monotonic()
    result, cards, installed, error = 0, [], None, None
    try:
        cards, installed = run_auto(args.model, args.file, args.goal, emit, examples=args.examples,
                                    on_request=on_request)
    except Exception as exc:
        # run_auto has already printed the contextual error pair.
        result, error = 1, (f'{type(exc).__name__}: {exc}' if str(exc) else type(exc).__name__)
    else:
        for card in cards:
            print(f'{card.bridge}: {card.verdict} — {card.reason}')
    finally:
        jobs.set_cli_handler(None)
    for claim in claims.values():
        print(f'Claim [{claim["status"]}] {claim["subject"]}: {claim["statement"]} '
              f'Source: {claim["source_url"]} {claim["evidence"]}')
    if args.save and not save_record(args.save, request, events, cards, installed, error, recorded,
                                     time.monotonic() - started):
        return 1
    return result


if __name__ == '__main__':
    raise SystemExit(main())
