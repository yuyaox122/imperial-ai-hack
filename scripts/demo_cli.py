"""Run the agent from the command line: fit, verify, install or refuse, then optionally ask about one file.

  .venv/bin/python scripts/demo_cli.py --model qwen3.5-0.8b --dataset gtzan --scope closed
  .venv/bin/python scripts/demo_cli.py --model qwen3.5-0.8b --dataset gtzan --scope closed \\
      --ask /path/to/clip.wav "Which genre of music is this?"
  SENSE_FAKE=1 .venv/bin/python scripts/demo_cli.py --model fake --dataset gtzan --scope open

Prints each trace event as it happens, then every report card. With --ask, the file goes through the sense the
run installed (or the one already installed, if the run installed nothing) and the model's reply is printed.
Exit status: 0 when the run finished (installed or refused), 1 when it failed.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run from anywhere

from agent.loop import run_fit  # noqa: E402
from agent.planner import pct  # noqa: E402
from sense import config, models, registry  # noqa: E402
from sense.contracts import FitRequest, ReportCard, TraceEvent  # noqa: E402

CONTROL_NAMES = {
    "swapped": "swapped input",
    "no_input": "no input",
    "lookup": "nearest-example lookup",
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


def main() -> int:
    ap = argparse.ArgumentParser(description="Give a base model an audio sense, verify it, and install it or refuse.")
    ap.add_argument("--model", required=True, help="qwen3.5-0.8b or qwen3.5-2b; 'fake' with SENSE_FAKE=1")
    ap.add_argument("--dataset", required=True, help="gtzan (music genres) or esc50 (environmental sounds)")
    ap.add_argument("--scope", choices=["closed", "open"], default="closed",
                    help="closed: only the labelled categories matter; open: any sound")
    ap.add_argument("--ask", nargs=2, metavar=("FILE", "QUESTION"), help="afterwards, ask about one audio file")
    args = ap.parse_args()
    if args.ask and not Path(args.ask[0]).expanduser().is_file():
        ap.error(f"--ask: {args.ask[0]} does not exist. Give the path of an audio file.")

    mode ="fake mode (SENSE_FAKE=1)" if config.fake() else "real models"
    print(f"Agent run: {args.model}, {args.dataset}, scope {args.scope}, {mode}\n", flush=True)
    started = time.monotonic()
    try:
        cards, installed = run_fit(FitRequest(model=args.model, dataset=args.dataset, scope=args.scope), print_event)
    except Exception:  # run_fit has already printed the [error] event with the reason
        return 1
    for card in cards:
        print_card(card)
    print(f"\nInstalled: {installed or 'nothing'}. Run took {time.monotonic() - started:.1f} s.")
    if args.ask:
        return ask(args.model, Path(args.ask[0]).expanduser(), args.ask[1])
    return 0


if __name__ == "__main__":
    sys.exit(main())
