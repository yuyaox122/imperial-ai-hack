"""Command line for a complete autonomous sense-fitting run."""
from __future__ import annotations

import argparse
from pathlib import Path

from agent.auto import run_auto
from sense.contracts import TraceEvent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Detect, research and verify a new sense for a model.')
    parser.add_argument('--model', required=True)
    parser.add_argument('--file', required=True, type=Path)
    parser.add_argument('--goal', required=True)
    parser.add_argument('--examples', type=Path)
    args = parser.parse_args(argv)
    claims: dict[tuple[str, str, str], dict] = {}

    def emit(event: TraceEvent) -> None:
        print(f'[{event.step}] {event.status}  {event.message}', flush=True)
        for claim in event.data.get('claims', []):
            claims[(claim['kind'], claim['subject'], claim['source_url'])] = claim

    result = 0
    try:
        cards, _ = run_auto(args.model, args.file, args.goal, emit, examples=args.examples)
    except Exception:
        # run_auto has already printed the contextual error pair.
        result = 1
    else:
        for card in cards:
            print(f'{card.bridge}: {card.verdict} — {card.reason}')
    for claim in claims.values():
        print(f'Claim [{claim["status"]}] {claim["subject"]}: {claim["statement"]} '
              f'Source: {claim["source_url"]} {claim["evidence"]}')
    return result


if __name__ == '__main__':
    raise SystemExit(main())
