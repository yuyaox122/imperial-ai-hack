"""Set up a second machine for real mode: check what is there, fetch the weights, copy or rebuild the embeddings.

Second-machine guide
--------------------
Nobody has run this guide on a second machine yet. It was written and tested offline on Michael's laptop
(macOS, Apple silicon), so expect gaps, and post on the board when you hit one. Run every command from the
repository root.

1. Clone the repository.
       git clone https://github.com/yuyaox122/imperial-ai-hack.git
       cd imperial-ai-hack

2. Install the dependencies with uv (https://docs.astral.sh/uv/). This creates .venv with Python 3.12.
       uv sync

3. Get the three embedding files, gtzan_clap.npz, esc50_clap.npz and audiocaps_text_clap.npz, into one
   directory. Step 5 points SENSE_DATA_DIR at it. Either:

   a. Copy them (fast). Michael keeps them in his SENSE_DATA_DIR, ~/.cache/sense-harness/artifacts on his
      laptop, 39 MB in total. Copy the three files by AirDrop, a USB stick or scp, for example
          mkdir -p ~/.cache/sense-harness/artifacts
          scp "<michael's laptop>:.cache/sense-harness/artifacts/*.npz" ~/.cache/sense-harness/artifacts/
      Step 6 compares them with scripts/embeddings.sha256, so a broken copy shows up there.

   b. Rebuild them (slow). This runs CLAP over the datasets, so do step 4 first. Download each dataset into
      its own directory:
          GTZAN      https://huggingface.co/datasets/marsyas/gtzan, unpacked so that one directory holds a
                     sub-directory of .wav files per genre (blues/, classical/, ..., rock/)
          ESC-50     https://github.com/karolpiczak/ESC-50, the ESC-50-master directory (meta/ and audio/)
          AudioCaps  dataset/train.csv from https://github.com/cdjkim/audiocaps (caption text only)
      Then run the following, with --dry-run first to see the three commands:
          .venv/bin/python scripts/setup_machine.py rebuild --gtzan <genres dir> --esc50 <ESC-50-master> \\
              --captions <train.csv> --out ~/.cache/sense-harness/artifacts
      A rebuilt file need not match Michael's checksum byte for byte (other hardware, library versions or
      dataset copies can change the embeddings slightly), so step 6 checks rebuilt files with --rebuilt.

4. Fetch the model weights. This needs a network connection; --dry-run lists what would be fetched.
       .venv/bin/python scripts/setup_machine.py fetch --dry-run
       .venv/bin/python scripts/setup_machine.py fetch
       .venv/bin/python scripts/setup_machine.py fetch --model qwen3.5-2b
   The second command fetches Qwen3.5-0.8B and CLAP; the third adds the optional Qwen3.5-2B. Qwen goes into
   SENSE_HF_CACHE when it is set (set it first, as in step 5, to keep the weights on another disk), otherwise
   into the default Hugging Face cache (~/.cache/huggingface/hub, unless HF_HOME or HF_HUB_CACHE says
   otherwise). CLAP always goes into the default cache, because transformers reads its config and processor
   from there. CLAP comes in two parts: the config, processor and tokenizer files from revision main, and
   model.safetensors (about 600 MB) from revision refs/pr/3, the only place laion/clap-htsat-unfused keeps
   safetensors weights.

5. Write .env.local in the repository root (git ignores it): one KEY=VALUE per line, with no comment after a
   value. Write full paths: ~ is not expanded in .env.local. SENSE_DATA_DIR is the directory from step 3. Add
   SENSE_HF_CACHE only if Qwen is in a cache other than the default one. SENSE_SAMPLE_DIRS is optional (only
   for sample clips in the interface): the GTZAN genres directory and the ESC-50 audio directory, joined by
   ':'. Leave SENSE_FAKE unset.
       SENSE_DATA_DIR=/Users/<you>/.cache/sense-harness/artifacts
       SENSE_HF_CACHE=/Volumes/<disk>/hf-cache
       SENSE_SAMPLE_DIRS=/Users/<you>/data/gtzan/genres:/Users/<you>/data/ESC-50-master/audio

6. Check. Every line names one requirement and says OK, MISSING or MISMATCH; a line that is not OK ends with
   a fix: hint. The exit status is 0 when everything real mode needs is present.
       .venv/bin/python scripts/setup_machine.py check              after copying the embeddings (3a)
       .venv/bin/python scripts/setup_machine.py check --rebuilt    after rebuilding them (3b)
   Add --load-clap to also load CLAP the way the harness does and embed one caption. It imports torch and
   takes longer, and with a network connection transformers may download CLAP files that are missing.

7. Confirm with a real agent run:
       .venv/bin/python scripts/demo_cli.py --model qwen3.5-0.8b --dataset gtzan --scope closed
   Compare the accuracy on the vector:gtzan card with the 81.0% measured on Michael's laptop (README.md,
   "What works today"). With copied embeddings, expect a number close to it. On a machine without Apple's
   MPS the model runs on the CPU in float32, which is slower and may move the number a little. If the gap is
   large, something differs between the machines: post both cards on the board.

Commands (exit status 0 on success, 1 otherwise):
  check [--rebuilt] [--load-clap]   one line per requirement; without --load-clap it only looks at files on disk
                                    and does not import torch
  fetch [--model M] [--dry-run]     M is qwen3.5-0.8b (default) or qwen3.5-2b; CLAP is fetched as well
  rebuild --gtzan DIR --esc50 DIR --captions CSV [--out DIR] [--dry-run] [--force]
                                    runs the three socket_pilot embedding scripts, then prints the checksums
  checksums [--write]               sha256 of the three files in SENSE_DATA_DIR; --write records them in
                                    scripts/embeddings.sha256 (for whoever regenerates the reference files)

Public surface (do not rename):
  main(argv=None) -> int
  check(load_clap=False, rebuilt=False) -> int
  fetch(model="qwen3.5-0.8b", dry_run=False) -> int
  rebuild(gtzan, esc50, captions, out=None, dry_run=False, force=False) -> int
  checksums(write=False) -> int
  EMBEDDING_FILES, CHECKSUM_FILE
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))  # run from anywhere

from sense import config  # noqa: E402

CHECKSUM_FILE = REPO / "scripts" / "embeddings.sha256"
EMBEDDING_FILES = {  # file in SENSE_DATA_DIR -> the keys sense/datasets.py reads from it
    "gtzan_clap.npz": ("emb", "text_emb", "y", "names", "files"),
    "esc50_clap.npz": ("emb", "text_emb", "y", "fold", "names", "files"),
    "audiocaps_text_clap.npz": ("text_emb", "texts"),
}
QWEN = {  # model name -> (label, environment variable that overrides the source)
    "qwen3.5-0.8b": ("Qwen3.5 0.8B", "SENSE_QWEN_0P8B"),
    "qwen3.5-2b": ("Qwen3.5 2B", "SENSE_QWEN_2B"),
}
QWEN_PATTERNS = ["*.json", "*.safetensors", "*.txt", "*.jinja"]
CLAP_DEFAULT = "laion/clap-htsat-unfused"
CLAP_PATTERNS = ["*.json", "*.txt", "tokenizer*", "vocab*", "merges*"]  # config, processor, tokenizer; no weights
CLAP_WEIGHTS = "model.safetensors"
CLAP_WEIGHTS_REVISION = "refs/pr/3"  # laion/clap-htsat-unfused keeps model.safetensors only in this conversion PR
CAPTION_LIMIT = 12_000  # the caption bank in README.md holds 12,000 AudioCaps captions; clap_text.py defaults to 20,000


# ----------------------------------------------------------------------------- check
@dataclass
class _Report:
    failed: list[str] = field(default_factory=list)

    def line(self, status: str, what: str, detail: str, fix: str = "", required: bool = True) -> None:
        text = f"{status:<9}{what}: {detail}"
        if status != "OK" and fix:
            text += f"  fix: {fix}"
        print(text, flush=True)
        if required and status != "OK":
            self.failed.append(what)


def check(load_clap: bool = False, rebuilt: bool = False) -> int:
    """Print one line per requirement of real mode. 0 when everything required is present, else 1."""
    report = _Report()
    print(f"Checking this machine for real mode. Embeddings directory (SENSE_DATA_DIR): {config.data_dir()}\n")
    _check_python(report)
    _check_fake(report)
    if config.hf_cache():
        _unexpanded(report, "SENSE_HF_CACHE", config.hf_cache())
    try:
        from sense import models, specialists

        _check_qwen(report, models, "qwen3.5-0.8b", required=True)
        _check_qwen(report, models, "qwen3.5-2b", required=False)
        _check_clap(report, specialists, load_clap)
    except ImportError as err:
        report.line("MISSING", "Python packages", f"{err}", "run `uv sync`, then run this script with .venv/bin/python")
    _check_embeddings(report, rebuilt)
    _check_samples(report)
    if report.failed:
        print(f"\nReal mode is not ready: fix {', '.join(report.failed)} (see the fix: hints above).")
        return 1
    print(
        "\nReal mode is ready on this machine. Confirm with:\n"
        "  .venv/bin/python scripts/demo_cli.py --model qwen3.5-0.8b --dataset gtzan --scope closed"
    )
    return 0


def _check_python(report: _Report) -> None:
    version = ".".join(map(str, sys.version_info[:3]))
    fix = "run `uv sync` in the repository, then run this script with .venv/bin/python"
    if sys.version_info[:2] == (3, 12):
        report.line("OK", "Python 3.12", f"Python {version} at {sys.executable}")
    else:
        report.line("MISMATCH", "Python 3.12", f"this is Python {version}; the project needs 3.12", fix)
    if sys.prefix != sys.base_prefix:
        report.line("OK", "virtual environment (optional)", sys.prefix, required=False)
    else:
        report.line("MISSING", "virtual environment (optional)", f"{sys.executable} is not in one", fix, required=False)


def _check_fake(report: _Report) -> None:
    if config.fake():
        report.line(
            "MISMATCH", "fake mode off", "SENSE_FAKE is set, so the harness would use the fake model and data",
            "unset SENSE_FAKE in the shell and remove it from .env and .env.local",
        )
    else:
        report.line("OK", "fake mode off", "SENSE_FAKE is not set")


def _hf_caches() -> list[str]:
    from huggingface_hub import constants

    return list(dict.fromkeys(c for c in (config.hf_cache(), constants.HF_HUB_CACHE) if c))


def _looks_like_path(source: str) -> bool:
    """True when a model setting names a local path rather than a Hugging Face repo id."""
    if source.startswith(("~", ".")) or Path(source).is_absolute():
        return True
    from huggingface_hub.utils import HFValidationError, validate_repo_id

    try:
        validate_repo_id(source)
    except HFValidationError:
        return True
    return False


def _unexpanded(report: _Report, what: str, value: str, required: bool = True) -> bool:
    """Report a path setting that starts with ~ and return True. The .env files are read without shell expansion
    and the harness does not expand ~ in SENSE_DATA_DIR, SENSE_HF_CACHE, SENSE_CLAP or SENSE_SAMPLE_DIRS, so such
    a path points nowhere. (SENSE_QWEN_* are expanded, so they are not checked here.)"""
    if not value.startswith("~"):
        return False
    report.line(
        "MISMATCH", what, f"{value} starts with ~, which the harness does not expand",
        "write the full path (/Users/<you>/... or /home/<you>/...) in .env.local; ~ is not expanded there",
        required=required,
    )
    return True


def _check_qwen(report: _Report, models, name: str, required: bool) -> None:
    label, env = QWEN[name]
    what = f"{label} weights" + ("" if required else " (optional)")
    source = config.model_source(name)
    snapshot = models._local_snapshot(source)
    if snapshot is not None:
        report.line("OK", what, str(snapshot), required=required)
        return
    fetch = f"`.venv/bin/python scripts/setup_machine.py fetch --model {name}`"
    if Path(source).expanduser().is_dir():
        detail = f"{env} points at {source}, which holds no config.json and *.safetensors"
        fix = f"put the model's config.json and *.safetensors files in {source}, or remove {env} and run {fetch}"
    elif _looks_like_path(source):
        detail = f"{env} points at {source}, which is not a directory on this machine"
        fix = f"correct {env} in .env.local, or remove it and run {fetch}"
    else:
        detail = f"no snapshot of {source} with config.json and *.safetensors in {' or '.join(_hf_caches())}"
        fix = (
            f"run {fetch}, or set SENSE_HF_CACHE to the Hugging Face cache that holds it, or {env} to a local "
            "model directory"
        )
    if not required:
        fix += " (only needed to offer the 2B model)"
    report.line("MISSING", what, detail, fix, required=required)


def _missing_clap_files(present) -> list[str]:
    """The config, processor and tokenizer files ClapProcessor needs that `present(name)` does not find."""
    missing = [f for f in ("config.json", "preprocessor_config.json", "tokenizer_config.json") if not present(f)]
    if not present("tokenizer.json") and not (present("vocab.json") and present("merges.txt")):
        missing.append("tokenizer.json (or vocab.json and merges.txt)")
    return missing


def _check_clap(report: _Report, specialists, load_clap: bool) -> None:
    source = config.model_source("clap")
    if _unexpanded(report, "SENSE_CLAP", source):
        return
    local = Path(source)  # not expanduser: transformers loads SENSE_CLAP as written
    fetch = "run `.venv/bin/python scripts/setup_machine.py fetch` with a network connection"
    if not local.is_dir() and _looks_like_path(source):
        report.line(
            "MISSING", "SENSE_CLAP", f"points at {source}, which is not a directory on this machine",
            f"correct SENSE_CLAP in .env.local, or remove it to use {CLAP_DEFAULT} and {fetch}",
        )
        return
    if local.is_dir():
        where = str(local)
        fix = f"put them in {local}, or unset SENSE_CLAP and {fetch}"

        def found(name: str) -> bool:
            return (local / name).is_file()
    else:
        from huggingface_hub import constants, try_to_load_from_cache

        where = f"the cached main revision of {source} in {constants.HF_HUB_CACHE}"
        fix = fetch

        def found(name: str) -> bool:  # where transformers looks for CLAP's config and processor
            return isinstance(try_to_load_from_cache(source, name), str)
    missing = _missing_clap_files(found)
    if missing:
        report.line("MISSING", "CLAP config and processor", f"{', '.join(missing)} not found in {where}", fix)
    else:
        report.line("OK", "CLAP config and processor", where)

    if local.is_dir():
        weights = next((local / f for f in (CLAP_WEIGHTS, "pytorch_model.bin") if (local / f).is_file()), None)
    else:
        snapshot = specialists._cached_safetensors_dir(source)
        weights = snapshot / CLAP_WEIGHTS if snapshot is not None else None
    if weights is not None:
        report.line("OK", "CLAP weights", str(weights))
    else:
        searched = str(local) if local.is_dir() else " or ".join(_hf_caches())
        report.line(
            "MISSING", "CLAP weights", f"no {CLAP_WEIGHTS} for {source} in {searched}",
            f"{fetch}; it takes {CLAP_WEIGHTS} from revision {CLAP_WEIGHTS_REVISION}",
        )

    if load_clap:
        ok, detail = _try_load_clap()
        report.line(
            "OK" if ok else "MISMATCH", "CLAP loads", detail,
            f"read the error; if it names a missing file, {fetch} again",
        )


def _try_load_clap() -> tuple[bool, str]:
    """Load CLAP the way the harness does and embed one caption. Imports torch."""
    from sense.specialists import ClapSpecialist

    try:
        emb = ClapSpecialist().embed_texts(["a dog barking"])
    except Exception as err:
        return False, f"{type(err).__name__}: {err}"
    return True, f"loaded and embedded a test caption into {emb.shape[1]} numbers"


def _check_embeddings(report: _Report, rebuilt: bool) -> None:
    data = config.data_dir()
    if _unexpanded(report, "SENSE_DATA_DIR", str(data)):
        return
    expected = _read_checksums(CHECKSUM_FILE)
    reference = _shown(CHECKSUM_FILE)
    for name, keys in EMBEDDING_FILES.items():
        path = data / name
        if not path.is_file():
            report.line(
                "MISSING", name, f"not in {data}",
                f"copy it from Michael's SENSE_DATA_DIR into {data}, rebuild it with "
                "`.venv/bin/python scripts/setup_machine.py rebuild`, or set SENSE_DATA_DIR to the directory "
                "that holds it",
            )
            continue
        size = _size(path)
        if rebuilt:
            problem, rows = _npz_problem(path, keys)
            if problem:
                report.line(
                    "MISMATCH", name, f"{size}, {problem}",
                    "rebuild it with `.venv/bin/python scripts/setup_machine.py rebuild --force ...`",
                )
            else:
                report.line("OK", name, f"{size}, {rows} rows, every key present (rebuilt: checksum not compared)")
            continue
        digest = _sha256(path)
        want = expected.get(name)
        if want is None:
            report.line(
                "MISMATCH", name, f"{size}, but {reference} has no checksum for it",
                f"restore {reference} with `git checkout {reference}`",
            )
        elif digest == want:
            report.line("OK", name, f"{size}, sha256 matches {reference}")
        else:
            report.line(
                "MISMATCH", name, f"{size}, sha256 {digest[:12]}... differs from {want[:12]}... in {reference}",
                "copy the file again from Michael's SENSE_DATA_DIR (this copy may be incomplete); if you rebuilt "
                "it on this machine, run `check --rebuilt` instead",
            )


def _npz_problem(path: Path, keys: tuple[str, ...]) -> tuple[str, int]:
    """(what is wrong with a rebuilt .npz file or "", number of rows of its first key)."""
    import numpy as np

    try:
        with np.load(path, allow_pickle=False) as data:
            absent = [k for k in keys if k not in data.files]
            if absent:
                return f"key(s) {', '.join(absent)} missing", 0
            first = data[keys[0]]
            if first.ndim != 2 or first.shape[0] == 0:
                return f"{keys[0]} has shape {first.shape}, expected a non-empty 2-D array", 0
            return "", int(first.shape[0])
    except Exception as err:
        return f"could not be read as .npz ({type(err).__name__}: {err})", 0


def _check_samples(report: _Report) -> None:
    dirs = config.sample_dirs()
    if not dirs:
        report.line(
            "MISSING", "sample audio (optional)", "SENSE_SAMPLE_DIRS is not set",
            f"set SENSE_SAMPLE_DIRS in .env.local to the GTZAN genres directory and the ESC-50 audio directory, "
            f"joined by {os.pathsep!r} (only needed for sample clips in the interface)",
            required=False,
        )
    for directory in dirs:
        if _unexpanded(report, f"sample audio {directory} (optional)", str(directory), required=False):
            continue
        if directory.is_dir():
            report.line("OK", f"sample audio {directory} (optional)", "directory exists", required=False)
        else:
            report.line(
                "MISSING", f"sample audio {directory} (optional)", "no such directory",
                "correct or remove this entry of SENSE_SAMPLE_DIRS", required=False,
            )


# ----------------------------------------------------------------------------- fetch
@dataclass
class _Fetch:
    what: str
    repo_id: str
    revision: str
    allow_patterns: list[str]
    cache_dir: str | None  # None: the default Hugging Face cache
    size: str = ""


def _fetch_plan(model: str) -> tuple[list[_Fetch], list[str]]:
    """What `fetch` downloads, and notes about what it skips."""
    plan, skipped = [], []
    label, env = QWEN[model]
    qwen = config.model_source(model)
    if Path(qwen).expanduser().is_dir():
        skipped.append(f"{label} weights: {env} points at the local directory {qwen}, so there is nothing to fetch.")
    elif _looks_like_path(qwen):
        skipped.append(f"{label} weights: skipped. {env} points at {qwen}, which is not a directory on this machine. "
                       f"Correct {env} in .env.local, or remove it and run fetch again.")
    else:
        plan.append(_Fetch(f"{label} weights", qwen, "main", QWEN_PATTERNS, config.hf_cache()))
    clap = config.model_source("clap")
    if Path(clap).expanduser().is_dir():
        skipped.append(f"CLAP: SENSE_CLAP points at the local directory {clap}, so there is nothing to fetch.")
    elif _looks_like_path(clap):
        skipped.append(f"CLAP: skipped. SENSE_CLAP points at {clap}, which is not a directory on this machine. "
                       "Correct SENSE_CLAP in .env.local, or remove it and run fetch again.")
    else:
        weights_revision = CLAP_WEIGHTS_REVISION if clap == CLAP_DEFAULT else "main"
        plan.append(_Fetch("CLAP config, processor and tokenizer", clap, "main", CLAP_PATTERNS, None))
        plan.append(_Fetch("CLAP weights", clap, weights_revision, [CLAP_WEIGHTS], None,
                           "about 600 MB" if clap == CLAP_DEFAULT else ""))
    return plan, skipped


def _describe(item: _Fetch) -> str:
    from huggingface_hub import constants

    where = f"{item.cache_dir} (SENSE_HF_CACHE)" if item.cache_dir else f"{constants.HF_HUB_CACHE} (default cache)"
    size = f" ({item.size})" if item.size else ""
    return (f"{item.what}: {item.repo_id} at revision {item.revision}, files matching "
            f"{' '.join(item.allow_patterns)}{size}, into {where}")


def fetch(model: str = "qwen3.5-0.8b", dry_run: bool = False) -> int:
    """Download the Qwen weights and CLAP into the Hugging Face cache. With dry_run, only print the plan."""
    if model not in QWEN:
        raise ValueError(f"Unknown model {model!r}. Choose one of: {', '.join(QWEN)}.")
    plan, skipped = _fetch_plan(model)
    for note in skipped:
        print(note)
    if dry_run:
        print("Dry run: nothing is downloaded. Without --dry-run this fetches:")
        for item in plan:
            print(f"  {_describe(item)}")
        return 0
    from huggingface_hub import snapshot_download

    for item in plan:
        print(f"Fetching {_describe(item)}", flush=True)
        try:
            path = snapshot_download(
                repo_id=item.repo_id,
                revision=item.revision,
                allow_patterns=item.allow_patterns,
                cache_dir=item.cache_dir,
            )
        except Exception as err:
            print(
                f"Could not fetch {item.repo_id} at revision {item.revision} ({type(err).__name__}: {err}). "
                "Check the network connection and run fetch again; files already fetched are kept. If Hugging "
                "Face refuses access, log in with `.venv/bin/hf auth login` and retry."
            )
            return 1
        print(f"  done: {path}", flush=True)
    if any(item.cache_dir for item in plan):
        print(f"Qwen is in SENSE_HF_CACHE ({config.hf_cache()}): keep SENSE_HF_CACHE in .env.local so the harness "
              "finds it.")
    print("Next: .venv/bin/python scripts/setup_machine.py check")
    return 0


# ----------------------------------------------------------------------------- rebuild
def _rebuild_commands(gtzan: Path, esc50: Path, captions: Path, out_dir: Path) -> dict[str, list[str]]:
    """file name -> the command that writes it. Each script reads a downloaded dataset, so each runs isolated (-I)."""
    clap = config.model_source("clap")
    pilot = REPO / "socket_pilot"
    python = [sys.executable, "-I"]
    return {
        "gtzan_clap.npz": python + [
            str(pilot / "clap_embed_gtzan.py"), "--gtzan", str(gtzan),
            "--out", str(out_dir / "gtzan_clap.npz"), "--model", clap,
        ],
        "esc50_clap.npz": python + [
            str(pilot / "clap_embed.py"), "--esc50", str(esc50),
            "--out", str(out_dir / "esc50_clap.npz"), "--model", clap,
        ],
        "audiocaps_text_clap.npz": python + [
            str(pilot / "clap_text.py"), "--captions", str(captions),
            "--out", str(out_dir / "audiocaps_text_clap.npz"), "--model", clap, "--limit", str(CAPTION_LIMIT),
        ],
    }


def _rebuild_problems(gtzan: Path, esc50: Path, captions: Path) -> list[str]:
    problems = []
    if not gtzan.is_dir():
        problems.append(f"--gtzan {gtzan} is not a directory. Give the directory that holds one sub-directory "
                        "of .wav files per genre (blues/, classical/, ..., rock/).")
    elif not any(d.is_dir() and any(d.glob("*.wav")) for d in gtzan.iterdir()):
        if any(gtzan.glob("*/*.au")):
            problems.append(f"--gtzan {gtzan} holds .au files, but clap_embed_gtzan.py reads only .wav. Convert "
                            "them to .wav, or use a copy of GTZAN that has .wav files.")
        else:
            problems.append(f"--gtzan {gtzan} has no sub-directory holding .wav files. Give the directory that "
                            "holds blues/, classical/, ..., rock/.")
    if not (esc50 / "meta" / "esc50.csv").is_file() or not (esc50 / "audio").is_dir():
        problems.append(f"--esc50 {esc50} has no meta/esc50.csv and audio/ directory. Give the unpacked "
                        "ESC-50-master directory.")
    if not captions.is_file():
        problems.append(f"--captions {captions} does not exist. Give AudioCaps' dataset/train.csv.")
    else:
        with open(captions, newline="") as fh:
            header = next(csv.reader(fh), [])
        if "caption" not in header:
            problems.append(f"--captions {captions} has no 'caption' column. Give AudioCaps' dataset/train.csv.")
    return problems


def rebuild(gtzan, esc50, captions, out=None, dry_run: bool = False, force: bool = False) -> int:
    """Write the three embedding files with the socket_pilot scripts, then print their checksums."""
    gtzan, esc50, captions = (Path(p).expanduser().absolute() for p in (gtzan, esc50, captions))
    out_dir = Path(out).expanduser().absolute() if out else config.data_dir()
    commands = _rebuild_commands(gtzan, esc50, captions, out_dir)
    problems = _rebuild_problems(gtzan, esc50, captions)
    if not out and str(out_dir).startswith("~"):
        problems.append(f"SENSE_DATA_DIR is {out_dir}, and ~ is not expanded in .env.local. Write the full path "
                        "there, or pass --out.")
    clap = config.model_source("clap")
    if not Path(clap).is_dir() and _looks_like_path(clap):
        problems.append(f"SENSE_CLAP is {clap}, which is not a directory on this machine. Correct SENSE_CLAP in "
                        ".env.local (with a full path: ~ is not expanded there), or remove it.")
    if dry_run:
        print(f"Dry run: nothing is run. These commands would write the three embedding files into {out_dir}:")
        for command in commands.values():
            print(f"  {shlex.join(command)}")
        for problem in problems:
            print(f"Note: {problem}")
        return 0
    if problems:
        for problem in problems:
            print(f"Cannot rebuild: {problem}")
        return 1
    existing = [name for name in commands if (out_dir / name).exists()]
    if existing and not force:
        print(f"Cannot rebuild: {', '.join(existing)} already in {out_dir}. Move them away, choose another --out, "
              "or pass --force to overwrite them.")
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, command in commands.items():
        print(f"\n$ {shlex.join(command)}", flush=True)
        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            print(f"Writing {name} failed with exit status {result.returncode}; see the output above. Fix the cause "
                  "and run rebuild again with --force.")
            return 1
    print(f"\nWrote the three embedding files into {out_dir}. Their checksums:")
    for name in commands:
        print(f"{_sha256(out_dir / name)}  {name}")
    data_dir = "" if out_dir == config.data_dir() else f"SENSE_DATA_DIR={out_dir} "
    print(f"They need not match {_shown(CHECKSUM_FILE)} (Michael's files) byte for byte. Check them with:\n"
          f"  {data_dir}.venv/bin/python scripts/setup_machine.py check --rebuilt")
    return 0


# ----------------------------------------------------------------------------- checksums
def checksums(write: bool = False) -> int:
    """Print the sha256 of the three files in SENSE_DATA_DIR; with write, record them in CHECKSUM_FILE."""
    data = config.data_dir()
    lines, missing = [], []
    for name in EMBEDDING_FILES:
        path = data / name
        if path.is_file():
            lines.append(f"{_sha256(path)}  {name}")
        else:
            missing.append(name)
    print(f"sha256 of the embedding files in {data}:")
    for line in lines:
        print(line)
    for name in missing:
        print(f"MISSING  {name}: not in {data}")
    if write:
        if missing:
            print(f"Did not write {CHECKSUM_FILE}: it must list all three files, and {', '.join(missing)} "
                  "is missing.")
            return 1
        CHECKSUM_FILE.write_text("\n".join(lines) + "\n")
        print(f"Wrote {CHECKSUM_FILE}. Commit it so that other machines check their copies against these files.")
    return 1 if missing else 0


def _read_checksums(path: Path) -> dict[str, str]:
    """file name -> sha256 from sha256sum-style lines ("<hex>  <file name>"); empty when the file is absent."""
    if not path.is_file():
        return {}
    out = {}
    for line in path.read_text().splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2 and not parts[0].startswith("#"):
            out[parts[1].lstrip("*")] = parts[0].lower()
    return out


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _size(path: Path) -> str:
    return f"{path.stat().st_size / 2**20:.1f} MB"


def _shown(path: Path) -> str:
    """A path relative to the repository when it is inside it."""
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


# ----------------------------------------------------------------------------- command line
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="setup_machine.py",
        description=__doc__.split("Public surface")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("check", help="report what real mode needs and what is missing")
    p.add_argument("--rebuilt", action="store_true",
                   help="the embeddings were rebuilt on this machine: check their contents instead of their checksums")
    p.add_argument("--load-clap", action="store_true",
                   help="also load CLAP and embed one caption (imports torch; slower)")
    p = sub.add_parser("fetch", help="download the Qwen3.5 weights and CLAP into the Hugging Face cache")
    p.add_argument("--model", choices=list(QWEN), default="qwen3.5-0.8b", help="which Qwen3.5 to fetch")
    p.add_argument("--dry-run", action="store_true", help="print what would be fetched and fetch nothing")
    p = sub.add_parser("rebuild", help="write the three embedding files from the datasets with CLAP")
    p.add_argument("--gtzan", required=True, metavar="DIR",
                   help="directory with one sub-directory of .wav files per genre")
    p.add_argument("--esc50", required=True, metavar="DIR", help="the unpacked ESC-50-master directory")
    p.add_argument("--captions", required=True, metavar="CSV", help="AudioCaps' dataset/train.csv")
    p.add_argument("--out", metavar="DIR", help="where to write the files (default: SENSE_DATA_DIR)")
    p.add_argument("--dry-run", action="store_true", help="print the three commands and run nothing")
    p.add_argument("--force", action="store_true", help="overwrite embedding files already in --out")
    p = sub.add_parser("checksums", help="print the sha256 of the three embedding files in SENSE_DATA_DIR")
    p.add_argument("--write", action="store_true", help=f"record them in {_shown(CHECKSUM_FILE)}")
    args = ap.parse_args(argv)

    if args.command == "check":
        return check(load_clap=args.load_clap, rebuilt=args.rebuilt)
    if args.command == "fetch":
        return fetch(model=args.model, dry_run=args.dry_run)
    if args.command == "rebuild":
        return rebuild(args.gtzan, args.esc50, args.captions, out=args.out, dry_run=args.dry_run, force=args.force)
    return checksums(write=args.write)


if __name__ == "__main__":
    sys.exit(main())
