"""Score free-text answers through a bridge: how often the base model's reply names the true label of a clip.

  .venv/bin/python scripts/score_free_text.py --model qwen3.5-0.8b --dataset gtzan --bridge vector
  .venv/bin/python scripts/score_free_text.py --model qwen3.5-0.8b --dataset esc50 --installed
  SENSE_FAKE=1 .venv/bin/python scripts/score_free_text.py --model fake --dataset gtzan --out /tmp/free_text.json

Builds the bridge the way an agent run does (vector: fitted on the dataset's fit split, with the block chosen on
its dev split; text: the specialist's best-matching label), or with --installed uses the sense already installed
for the model. Then sense.verify.free_text sends each held-out clip (at most --max-items) with the dataset's
question plus " Answer in a few words." three ways: through the bridge, swapped for a clip of another label, and
with no clip. It prints how often each reply names the true label, and writes free_text's dict plus a "build"
block (seconds to build or load the bridge, fit_seconds, layer, dev_accuracy, installed) and a "machine" block
to --out, by default socket_pilot/artifacts/free_text_<model>_<dataset>_<bridge>.json.

This counts whether a reply names the label, not whether the rest of the reply is true. It changes no report
card and no installed sense.
Exit status: 0 when the results file was written, 1 with a message when anything failed.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run from anywhere

from sense import bridges, config, datasets, models, registry, verify  # noqa: E402
from sense.contracts import BaseModel, Bridge, LabelledSet  # noqa: E402

KINDS = ("vector", "text")
CONDITIONS = {"bridge": "bridge", "swapped": "swapped input", "no_input": "no input"}
SHOW_EXAMPLES = 3


def pct(x: float) -> str:
    """0.849 -> '84.9%'."""
    return f"{x * 100:.1f}%"


def machine() -> dict:
    return {"platform": platform.platform(), "python": platform.python_version(), "machine": platform.machine()}


def default_out(model_name: str, dataset_name: str, kind: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", model_name)
    return config.REPO / "socket_pilot" / "artifacts" / f"free_text_{safe}_{dataset_name}_{kind}.json"


def build_bridge(model: BaseModel, dataset: LabelledSet, kind: str) -> Bridge:
    """The bridge an agent run would build for scope "closed"."""
    if kind == "vector":
        return bridges.fit_vector_bridge(model, dataset, dataset.fit_idx)
    return bridges.make_text_bridge(model, dataset)


def installed_bridge(model_name: str, dataset: LabelledSet, kind: str | None) -> Bridge:
    """The sense installed for this model and the dataset's modality, checked against the dataset and --bridge."""
    entries = [e for e in registry.installed(model_name) if e["modality"] == dataset.modality]
    if not entries:
        raise RuntimeError(
            f"No {dataset.modality} sense is installed for {model_name} (in {config.home()}). Run the agent first, "
            f"for example scripts/demo_cli.py --model {model_name} --dataset {dataset.name}, or drop --installed "
            "to build a bridge here."
        )
    entry = entries[0]
    if entry["dataset"] != dataset.name:
        raise ValueError(
            f"The {dataset.modality} sense installed for {model_name} ({entry['bridge']}) was verified on "
            f"{entry['dataset']}, not {dataset.name}. Pass --dataset {entry['dataset']}, or drop --installed to "
            f"build a bridge for {dataset.name}."
        )
    if kind is not None and entry["kind"] != kind:
        raise ValueError(
            f"The installed sense {entry['bridge']} is a {entry['kind']} bridge, not a {kind} bridge. Leave out "
            f"--bridge when using --installed, or drop --installed to build a {kind} bridge."
        )
    found = [b for b in registry.get_bridges(model_name) if b.modality == dataset.modality]
    if not found:
        raise RuntimeError(f"The installed {dataset.modality} sense for {model_name} could not be found on disk.")
    return found[0]


def print_results(record: dict) -> None:
    rows = [(CONDITIONS["bridge"], record["result"])]
    rows += [(CONDITIONS[name], stats) for name, stats in record["controls"].items()]
    header = f"{'condition':<15}{'names true only':>16}  {'95% interval':<17}{'names true':>11}{'names other':>13}{'names none':>12}"
    print(header)
    for name, s in rows:
        interval = f"[{pct(s['interval'][0])}, {pct(s['interval'][1])}]"
        print(
            f"{name:<15}{pct(s['names_true_only']):>16}  {interval:<17}{pct(s['names_true']):>11}"
            f"{pct(s['names_other']):>13}{pct(s['names_none']):>12}"
        )
    examples = record["examples"][:SHOW_EXAMPLES]
    if examples:
        print("\nExample replies through the bridge:")
    for ex in examples:
        reply = " ".join(ex["reply"].split())
        reply = reply if len(reply) <= 90 else reply[:87] + "..."
        named = ", ".join(ex["named"]) or "no label"
        print(f"  {ex['label']}: names {named}: \"{reply}\"")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Score how often a base model's free-text reply names the true label of a held-out clip."
    )
    ap.add_argument("--model", default="qwen3.5-0.8b", help="qwen3.5-0.8b or qwen3.5-2b; 'fake' with SENSE_FAKE=1 "
                    "(default: qwen3.5-0.8b)")
    ap.add_argument("--dataset", default="gtzan", help="gtzan (music genres) or esc50 (environmental sounds) "
                    "(default: gtzan)")
    ap.add_argument("--bridge", choices=KINDS, default=None, help="the bridge to build (default: vector); with "
                    "--installed, leave it out or give the installed sense's kind")
    ap.add_argument("--installed", action="store_true", help="use the sense installed for the model instead of "
                    "building one")
    ap.add_argument("--max-items", type=int, default=100, help="most held-out clips to score (default: 100)")
    ap.add_argument("--tokens", type=int, default=32, help="most new tokens in each reply (default: 32)")
    ap.add_argument("--seed", type=int, default=0, help="seed for choosing clips and swap partners (default: 0)")
    ap.add_argument("--out", type=Path, default=None, help="where to write the JSON (default: "
                    "socket_pilot/artifacts/free_text_<model>_<dataset>_<bridge>.json)")
    args = ap.parse_args(argv)
    if args.max_items < 2:
        ap.error("--max-items must be at least 2: the swapped-input control needs clips of two labels.")
    if args.tokens < 1:
        ap.error("--tokens must be at least 1.")

    mode = "fake mode (SENSE_FAKE=1)" if config.fake() else "real models"
    source = "the installed sense" if args.installed else f"a new {args.bridge or 'vector'} bridge"
    print(f"Free-text answers: {args.model}, {args.dataset}, {source}, {mode}", flush=True)
    try:
        model = models.load_model(args.model)
        dataset = datasets.load_labelled_set(args.dataset)
        started = time.monotonic()
        if args.installed:
            bridge = installed_bridge(args.model, dataset, args.bridge)
        else:
            bridge = build_bridge(model, dataset, args.bridge or "vector")
        build_seconds = round(time.monotonic() - started, 2)
        verb = "Loaded" if args.installed else "Built"
        print(f"{verb} {bridge.name} ({bridge.kind} bridge) in {build_seconds:.1f} s. Asking the model...", flush=True)
        result = verify.free_text(
            model, bridge, dataset, max_items=args.max_items, seed=args.seed, max_new_tokens=args.tokens
        )
    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
        print(f"Free-text scoring failed. {detail}", file=sys.stderr)
        return 1

    dev = getattr(bridge, "dev_accuracy", None)
    record = {
        **result,
        "build": {
            "seconds": build_seconds,
            "fit_seconds": getattr(bridge, "fit_seconds", None),
            "fit_seconds_by_layer": bridges.describe(bridge)["fit_seconds_by_layer"],
            "layer": getattr(bridge, "layer", None),
            "dev_accuracy": {str(k): v for k, v in dev.items()} if isinstance(dev, dict) else dev,
            "installed": args.installed,
        },
        "machine": machine(),
    }
    print(
        f"Scored {result['n']} held-out clips in three conditions in {result['seconds']:.1f} s, at most "
        f"{result['max_new_tokens']} new tokens per reply.\nQuestion: \"{result['question']}\"\n"
        "A reply counts when it uses the label's words; the rest of the reply is not checked.\n"
    )
    print_results(record)

    out = args.out or default_out(args.model, args.dataset, bridge.kind)
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(record, indent=2) + "\n")
    except OSError as exc:
        print(f"Could not write the results to {out} ({exc}). Pass a writable path with --out.", file=sys.stderr)
        return 1
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
