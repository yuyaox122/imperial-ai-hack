"""Run the agent from the command line: fit, verify, install or refuse, then optionally ask about one file.

  .venv/bin/python scripts/demo_cli.py --model qwen3.5-0.8b --dataset gtzan --scope closed
  .venv/bin/python scripts/demo_cli.py --model qwen3.5-0.8b --dataset gtzan --scope closed \\
      --ask /path/to/clip.wav "Which genre of music is this?"
  SENSE_FAKE=1 .venv/bin/python scripts/demo_cli.py --model fake --dataset gtzan --scope open
  .venv/bin/python scripts/demo_cli.py --model qwen3.5-0.8b --folder ~/sounds --name my-sounds --scope closed
  .venv/bin/python scripts/demo_cli.py --model qwen3.5-0.8b --dataset gtzan --save runs/gtzan_closed.json

Give either --dataset (a built-in set, or a folder set built earlier) or --folder, a folder with one sub-folder of
audio files per label: the labelled set is built from it with sense.datasets.from_folder (or the saved one is
reused if the files are unchanged), a one-line summary of it is printed, and the run uses it.

Prints each trace event as it happens, then every report card. With --ask, the file goes through the sense the
run installed (or the one already installed, if the run installed nothing) and the model's reply is printed.
With --save PATH, the run is also written to PATH as JSON when it ends, whether it finished or failed: "id",
"request", "status" ("done" or "failed"), "events" (each with "t", seconds since the run started), "cards",
"installed" and "error" exactly as in JobState.to_dict(), so the file can be served like a finished job, plus
"origin" ("cli"), "recorded" (when the run started, ISO 8601 UTC), "seconds" (the run's wall time) and
"machine" ({"platform", "python", "machine"}). The file is written before --ask is answered.
Exit status: 0 when the run finished (installed or refused), 1 when it failed, --save could not write its file,
the folder could not make a set, or the set cannot carry this run (sense.datasets.fit_problem, checked before the
run starts: for example scope open on a set with fewer than 3 labels).
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run from anywhere

from agent.loop import run_fit  # noqa: E402
from agent.planner import pct  # noqa: E402
from sense import config, datasets, models, registry  # noqa: E402
from sense.contracts import FitRequest, JobState, LabelledSet, ReportCard, TraceEvent  # noqa: E402

CONTROL_NAMES = {
    "swapped": "swapped input",
    "no_input": "no input",
    "lookup": "nearest-example lookup",
    "specialist_alone": "specialist alone",
    "text_bridge": "text bridge",
    "unseen_classes": "unseen classes",
}


def print_event(event: TraceEvent) -> None:
    print(f"[{event.t:7.1f} s] {event.step:<8} {event.status:<5} {event.message}", flush=True)


def print_card(card: ReportCard) -> None:
    print(f"\n{card.bridge} ({card.kind} bridge): {card.verdict.upper()}")
    print(f"  {card.model}, {card.dataset}, scope {card.scope}, {card.n} held-out questions, chance {pct(card.chance)}")
    print(f"  {'accuracy':<24}{pct(card.accuracy):>7}")
    for key, value in card.controls.items():
        print(f"  {CONTROL_NAMES.get(key, key):<24}{pct(value):>7}")
    print(f"  reason: {card.reason}")


def folder_summary(ds: LabelledSet) -> str:
    """One line from a folder set's meta: what was used, what was left out and why, and the embedding time."""
    m = ds.meta
    left_out = [
        f"{n} {why}"
        for n, why in (
            (m.get("duplicates", 0), "exact duplicates"),
            (len(m.get("conflicts", [])), "files whose bytes sit under two labels"),
            (m.get("n_skipped", 0), "unreadable"),
            (m.get("n_capped", 0), f"over the cap of {m.get('max_per_label')} per label"),
        )
        if n
    ]
    return (
        f"Folder set {ds.name}: {len(ds.labels)} labels, {m.get('n_used', len(ds.y))} of {m.get('n_found', '?')} "
        f"audio files used ({len(ds.fit_idx)} fit, {len(ds.dev_idx)} dev, {len(ds.test_idx)} test); "
        f"left out: {', '.join(left_out) or 'none'}; {m.get('near_duplicate_groups', 0)} near-duplicate groups "
        f"kept within one split; embedded in {m.get('embed_seconds', 0):.1f} s at {m.get('created', '?')}."
    )


def ask(model_name: str, path: Path, question: str) -> int:
    """Send one file and a question through the model's installed sense; print what reached the model and the reply."""
    model = models.load_model(model_name)
    bridges = {b.modality: b for b in registry.get_bridges(model_name)}
    inputs = []
    print(f"\nYou: {question}  [{path.name}]")
    if "audio" in bridges:
        bridge = bridges["audio"]
        started = time.monotonic()
        item = bridge.prepare(path)
        item.source = path.name
        inputs.append(item)
        print(f"  {path.name} reached the model through {bridge.name} ({bridge.kind} bridge) "
              f"in {time.monotonic() - started:.1f} s: {item.detail}")
    else:
        print(f"  {model.info.label} has no installed audio sense, so {path.name} was not shown to the model.")
    started = time.monotonic()
    reply = model.chat(question, inputs, max_new_tokens=256)
    print(f"{model.info.label} ({time.monotonic() - started:.1f} s): {reply}")
    return 0


def save_run(
    path: Path,
    request: FitRequest,
    events: list[TraceEvent],
    cards: list[ReportCard],
    installed: str | None,
    error: str | None,
    recorded: str,
    seconds: float,
) -> bool:
    """Write the run record to `path` (see the module docstring). Returns False, after saying why, if it cannot."""
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


def main() -> int:
    ap = argparse.ArgumentParser(description="Give a base model an audio sense, verify it, and install it or refuse.")
    ap.add_argument("--model", required=True, help="qwen3.5-0.8b or qwen3.5-2b; 'fake' with SENSE_FAKE=1")
    data = ap.add_mutually_exclusive_group(required=True)
    data.add_argument("--dataset", help="gtzan (music genres), esc50 (environmental sounds), or the name of a "
                                        "folder set built earlier")
    data.add_argument("--folder", metavar="PATH", help="a folder with one sub-folder of audio files per label: build "
                                                       "a labelled set from it, then run on it")
    ap.add_argument("--name", help="with --folder: the set's name (default: from the folder's name)")
    ap.add_argument("--question", help="with --folder: the question asked about each clip "
                                       "(default: 'Which sound is in the audio clip?')")
    ap.add_argument("--scope", choices=["closed", "open"], default="closed",
                    help="closed: only the labelled categories matter; open: any sound (needs 3 labels or more)")
    ap.add_argument("--ask", nargs=2, metavar=("FILE", "QUESTION"), help="afterwards, ask about one audio file")
    ap.add_argument("--save", metavar="PATH", type=Path,
                    help="when the run ends, write it (request, timed trace events, report cards) to PATH as JSON")
    args = ap.parse_args()
    if args.ask and not Path(args.ask[0]).expanduser().is_file():
        ap.error(f"--ask: {args.ask[0]} does not exist. Give the path of an audio file.")
    if (args.name or args.question) and not args.folder:
        ap.error("--name and --question go with --folder; a set given with --dataset already has both.")

    dataset = args.dataset
    if args.folder:
        try:
            ds = datasets.from_folder(args.folder, args.name, question=args.question)
        except (ValueError, FileNotFoundError) as exc:
            print(f"Could not build a labelled set from {args.folder}: {exc}", file=sys.stderr)
            return 1
        print(folder_summary(ds), flush=True)
        dataset = ds.name
    try:
        problem = datasets.fit_problem(datasets.load_labelled_set(dataset), args.scope)
    except (ValueError, FileNotFoundError):
        problem = None  # an unknown or missing set: the run's [inspect] step says so
    if problem:
        print(problem, file=sys.stderr)
        return 1

    mode ="fake mode (SENSE_FAKE=1)" if config.fake() else "real models"
    print(f"Agent run: {args.model}, {dataset}, scope {args.scope}, {mode}\n", flush=True)
    request = FitRequest(model=args.model, dataset=dataset, scope=args.scope)
    events: list[TraceEvent] = []

    def record(event: TraceEvent) -> None:
        events.append(event)
        print_event(event)

    recorded = datetime.now(timezone.utc).isoformat(timespec="seconds")
    started = time.monotonic()
    try:
        cards, installed = run_fit(request, record)
    except Exception as exc:  # run_fit has already printed the [error] event with the reason
        if args.save:
            error = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
            save_run(args.save, request, events, [], None, error, recorded, time.monotonic() - started)
        return 1
    seconds = time.monotonic() - started
    for card in cards:
        print_card(card)
    print(f"\nInstalled: {installed or 'nothing'}. Run took {seconds:.1f} s.")
    if args.save and not save_run(args.save, request, events, cards, installed, None, recorded, seconds):
        return 1
    if args.ask:
        return ask(args.model, Path(args.ask[0]).expanduser(), args.ask[1])
    return 0


if __name__ == "__main__":
    sys.exit(main())
