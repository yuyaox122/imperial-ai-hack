"""scripts/setup_machine.py, fully offline: a fake Hugging Face cache and small dummy .npz files in tmp_path.

Nothing here downloads, loads weights or imports torch; snapshot_download, subprocess.run and the CLAP loader are
replaced, and the Hugging Face hub is set offline so that an accidental download fails fast.
"""

from __future__ import annotations

import fnmatch
import hashlib
import importlib.util
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import huggingface_hub
import huggingface_hub.constants as hf_constants
import numpy as np
import pytest

from sense import models, specialists

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "setup_machine.py"
_spec = importlib.util.spec_from_file_location("setup_machine", SCRIPT)
sm = importlib.util.module_from_spec(_spec)
sys.modules["setup_machine"] = sm  # dataclasses look their module up here
_spec.loader.exec_module(sm)

STATUSES = ("OK", "MISSING", "MISMATCH")
SENSE_VARS = ("SENSE_FAKE", "SENSE_HF_CACHE", "SENSE_QWEN_0P8B", "SENSE_QWEN_2B", "SENSE_CLAP", "SENSE_SAMPLE_DIRS")
CLAP_MAIN = ["config.json", "preprocessor_config.json", "tokenizer_config.json", "special_tokens_map.json",
             "tokenizer.json", "vocab.json", "merges.txt"]


# ----------------------------------------------------------------------------- helpers
@pytest.fixture
def machine(tmp_path, monkeypatch):
    """A blank second machine: empty Hugging Face cache and data directory, no SENSE_* settings."""
    hub, data = tmp_path / "hub", tmp_path / "data"
    hub.mkdir()
    data.mkdir()
    for var in SENSE_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("SENSE_DATA_DIR", str(data))
    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(hub))
    monkeypatch.setattr(hf_constants, "HF_HUB_OFFLINE", True)  # any hub access by accident fails instead of downloading
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(sm, "_try_load_clap", lambda: pytest.fail("check loaded CLAP without --load-clap"))
    monkeypatch.setattr(sm, "CHECKSUM_FILE", tmp_path / "embeddings.sha256")
    return SimpleNamespace(tmp=tmp_path, hub=hub, data=data, sums=tmp_path / "embeddings.sha256")


def snapshot(hub: Path, repo_id: str, files: list[str], ref: str = "main") -> Path:
    """One cached revision of a repo, laid out as huggingface_hub does it."""
    repo = hub / ("models--" + repo_id.replace("/", "--"))
    sha = hashlib.sha1(f"{repo_id}@{ref}".encode()).hexdigest()
    snap = repo / "snapshots" / sha
    snap.mkdir(parents=True, exist_ok=True)
    for name in files:
        (snap / name).write_bytes(b"x")
    (repo / "refs" / ref).parent.mkdir(parents=True, exist_ok=True)
    (repo / "refs" / ref).write_text(sha)
    return snap


def install_weights(hub: Path, qwen: bool = True, clap_main: bool = True, clap_weights: bool = True) -> None:
    if qwen:
        snapshot(hub, "Qwen/Qwen3.5-0.8B", ["config.json", "model.safetensors", "tokenizer.json"])
    if clap_main:
        snapshot(hub, "laion/clap-htsat-unfused", CLAP_MAIN)
    if clap_weights:
        snapshot(hub, "laion/clap-htsat-unfused", ["model.safetensors"], ref="refs/pr/3")


def _arrays(name: str, rows: int = 4) -> dict[str, np.ndarray]:
    emb = np.arange(rows * 8, dtype=np.float32).reshape(rows, 8)
    if name == "audiocaps_text_clap.npz":
        return {"text_emb": emb, "texts": np.array([f"caption {i}" for i in range(rows)])}
    out = {
        "emb": emb,
        "text_emb": np.ones((2, 8), np.float32),
        "y": np.arange(rows) % 2,
        "names": np.array(["blues", "rock"]),
        "files": np.array([f"clip{i}.wav" for i in range(rows)]),
    }
    if name == "esc50_clap.npz":
        out["fold"] = np.arange(rows) % 5 + 1
    return out


def write_embeddings(data: Path, sums: Path | None = None, skip: tuple[str, ...] = ()) -> None:
    """Small .npz files with the keys sense/datasets.py reads; `sums` records their checksums."""
    lines = []
    for name in sm.EMBEDDING_FILES:
        if name in skip:
            continue
        np.savez(data / name, **_arrays(name))
        lines.append(f"{hashlib.sha256((data / name).read_bytes()).hexdigest()}  {name}\n")
    if sums is not None:
        sums.write_text("".join(lines))


def status_of(out: str, what: str) -> str:
    """The status word of the check line that names `what`."""
    for line in out.splitlines():
        word = line.split(" ", 1)[0]
        if word in STATUSES and what in line:
            return word
    raise AssertionError(f"no check line names {what!r}")


def check_lines(out: str) -> list[str]:
    return [line for line in out.splitlines() if line.split(" ", 1)[0] in STATUSES]


def line_named(out: str, what: str) -> str:
    """The check line for requirement `what` exactly (lines read "<STATUS> <what>: <detail>")."""
    for line in check_lines(out):
        if line[9:].startswith(f"{what}:"):
            return line
    raise AssertionError(f"no check line for {what!r}")


# ----------------------------------------------------------------------------- check
def test_check_passes_when_everything_required_is_present(machine, capsys):
    install_weights(machine.hub)
    write_embeddings(machine.data, machine.sums)
    assert sm.main(["check"]) == 0
    out = capsys.readouterr().out
    for name in sm.EMBEDDING_FILES:
        assert status_of(out, name) == "OK"
    assert status_of(out, "Qwen3.5 0.8B weights") == "OK"
    assert status_of(out, "CLAP config and processor") == "OK"
    assert status_of(out, "CLAP weights") == "OK"
    assert status_of(out, "fake mode off") == "OK"
    assert status_of(out, "Qwen3.5 2B weights (optional)") == "MISSING"  # optional: does not fail the check
    assert status_of(out, "sample audio (optional)") == "MISSING"
    assert "MISMATCH" not in out
    assert "Real mode is ready" in out
    for line in check_lines(out):
        assert line.startswith("OK") or "fix:" in line


def test_check_reports_match_mismatch_and_missing_embeddings(machine, capsys):
    install_weights(machine.hub)
    write_embeddings(machine.data, machine.sums, skip=("audiocaps_text_clap.npz",))
    np.savez(machine.data / "esc50_clap.npz", **_arrays("esc50_clap.npz", rows=5))  # differs from its checksum
    assert sm.main(["check"]) == 1
    out = capsys.readouterr().out
    assert status_of(out, "gtzan_clap.npz") == "OK"
    assert status_of(out, "esc50_clap.npz") == "MISMATCH"
    assert status_of(out, "audiocaps_text_clap.npz") == "MISSING"
    for line in check_lines(out):
        if not line.startswith("OK"):
            assert "fix:" in line
    assert "Real mode is not ready" in out


def test_check_without_a_checksum_file_cannot_match(machine, capsys):
    install_weights(machine.hub)
    write_embeddings(machine.data)  # no checksum file written
    assert sm.main(["check"]) == 1
    assert status_of(capsys.readouterr().out, "gtzan_clap.npz") == "MISMATCH"


def test_check_rebuilt_checks_contents_instead_of_checksums(machine, capsys):
    install_weights(machine.hub)
    write_embeddings(machine.data, machine.sums)
    machine.sums.write_text("")  # rebuilt files never match Michael's checksums
    assert sm.main(["check", "--rebuilt"]) == 0
    out = capsys.readouterr().out
    for name in sm.EMBEDDING_FILES:
        assert status_of(out, name) == "OK"

    broken = _arrays("esc50_clap.npz")
    del broken["fold"]
    np.savez(machine.data / "esc50_clap.npz", **broken)
    assert sm.main(["check", "--rebuilt"]) == 1
    out = capsys.readouterr().out
    assert status_of(out, "esc50_clap.npz") == "MISMATCH"
    assert "fold" in out


def test_check_fails_when_the_weights_are_not_found(machine, capsys, monkeypatch):
    write_embeddings(machine.data, machine.sums)
    install_weights(machine.hub, qwen=False, clap_weights=False)
    monkeypatch.setattr(models, "_local_snapshot", lambda source: None)
    monkeypatch.setattr(specialists, "_cached_safetensors_dir", lambda source: None)
    assert sm.main(["check"]) == 1
    out = capsys.readouterr().out
    assert status_of(out, "Qwen3.5 0.8B weights") == "MISSING"
    assert status_of(out, "CLAP weights") == "MISSING"
    assert status_of(out, "CLAP config and processor") == "OK"
    qwen_line = next(line for line in out.splitlines() if "Qwen3.5 0.8B weights" in line)
    assert "fix:" in qwen_line and "fetch" in qwen_line


REQUIRED = [
    ("Qwen3.5 0.8B weights", {"qwen": False}, ()),
    ("CLAP config and processor", {"clap_main": False}, ()),
    ("CLAP weights", {"clap_weights": False}, ()),
    *[(name, {}, (name,)) for name in sm.EMBEDDING_FILES],
]


@pytest.mark.parametrize("what, weights, skip", REQUIRED, ids=[r[0] for r in REQUIRED])
def test_check_fails_when_any_one_required_item_is_missing(machine, capsys, what, weights, skip):
    install_weights(machine.hub, **weights)
    write_embeddings(machine.data, machine.sums, skip=skip)
    assert sm.main(["check"]) == 1
    out = capsys.readouterr().out
    line = line_named(out, what)
    assert line.startswith("MISSING") and "fix:" in line
    assert f"Real mode is not ready: fix {what} (see" in out  # the only required item that failed
    assert "(optional)" not in line_named(out, "Qwen3.5 0.8B weights")


def test_check_rebuilt_wants_a_non_empty_matrix_first(machine, capsys):
    install_weights(machine.hub)
    write_embeddings(machine.data)
    for emb in (np.arange(8, dtype=np.float32), np.zeros((0, 8), np.float32)):  # 1-D, then empty
        np.savez(machine.data / "gtzan_clap.npz", **{**_arrays("gtzan_clap.npz"), "emb": emb})
        assert sm.main(["check", "--rebuilt"]) == 1
        line = line_named(capsys.readouterr().out, "gtzan_clap.npz")
        assert line.startswith("MISMATCH") and "non-empty 2-D" in line


@pytest.mark.parametrize("drop, ok", [
    (("tokenizer.json",), True),
    (("vocab.json", "merges.txt"), True),
    (("tokenizer.json", "vocab.json"), False),
    (("tokenizer.json", "merges.txt"), False),
])
def test_check_needs_tokenizer_json_or_vocab_and_merges(machine, capsys, drop, ok):
    install_weights(machine.hub, clap_main=False)
    snapshot(machine.hub, "laion/clap-htsat-unfused", [f for f in CLAP_MAIN if f not in drop])
    write_embeddings(machine.data, machine.sums)
    assert sm.main(["check"]) == (0 if ok else 1)
    line = line_named(capsys.readouterr().out, "CLAP config and processor")
    assert line.startswith("OK" if ok else "MISSING")
    if not ok:
        assert "tokenizer.json (or vocab.json and merges.txt)" in line


def test_check_reads_a_local_clap_directory(machine, capsys, monkeypatch):
    install_weights(machine.hub, clap_main=False, clap_weights=False)
    write_embeddings(machine.data, machine.sums)
    local = machine.tmp / "clap"
    local.mkdir()
    monkeypatch.setenv("SENSE_CLAP", str(local))
    assert sm.main(["check"]) == 1  # an empty directory
    out = capsys.readouterr().out
    assert line_named(out, "CLAP config and processor").startswith("MISSING")
    assert line_named(out, "CLAP weights").startswith("MISSING")

    for name in [*CLAP_MAIN, "model.safetensors"]:
        (local / name).write_text("x")
    assert sm.main(["check"]) == 0
    out = capsys.readouterr().out
    assert line_named(out, "CLAP config and processor") == f"{'OK':<9}CLAP config and processor: {local}"
    assert line_named(out, "CLAP weights") == f"{'OK':<9}CLAP weights: {local / 'model.safetensors'}"


def test_model_settings_that_point_nowhere_are_reported_not_fetched(machine, capsys, monkeypatch):
    install_weights(machine.hub)
    write_embeddings(machine.data, machine.sums)
    gone = machine.tmp / "missing"
    monkeypatch.setenv("SENSE_CLAP", str(gone / "clap"))
    monkeypatch.setenv("SENSE_QWEN_0P8B", str(gone / "qwen"))
    assert sm.main(["check"]) == 1  # no traceback from the Hugging Face repo id validation
    out = capsys.readouterr().out
    qwen, clap = line_named(out, "Qwen3.5 0.8B weights"), line_named(out, "SENSE_CLAP")
    for line, env in ((qwen, "SENSE_QWEN_0P8B"), (clap, "SENSE_CLAP")):
        assert line.startswith("MISSING") and "not a directory" in line and f"correct {env}" in line
    for name in sm.EMBEDDING_FILES:  # check went on after the model lines
        assert line_named(out, name).startswith("OK")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", _no_download)
    assert sm.main(["fetch", "--dry-run"]) == 0
    assert sm.main(["fetch"]) == 0
    out = capsys.readouterr().out
    assert out.count("which is not a directory on this machine") == 4
    assert all("skipped" in line for line in out.splitlines() if str(gone) in line)  # the plan lists neither


def test_a_qwen_directory_without_weights_is_not_sent_to_fetch(machine, capsys, monkeypatch):
    install_weights(machine.hub)
    write_embeddings(machine.data, machine.sums)
    monkeypatch.setenv("SENSE_QWEN_0P8B", str(machine.tmp))
    assert sm.main(["check"]) == 1
    line = line_named(capsys.readouterr().out, "Qwen3.5 0.8B weights")
    assert "holds no config.json" in line and "put the model's config.json and *.safetensors files in" in line


def test_check_flags_paths_that_start_with_a_tilde(machine, capsys, monkeypatch):
    install_weights(machine.hub)
    write_embeddings(machine.data, machine.sums)
    settings = {"SENSE_DATA_DIR": "~/sense-data", "SENSE_HF_CACHE": "~/hf-cache", "SENSE_CLAP": "~/clap",
                "SENSE_SAMPLE_DIRS": "~/genres"}
    for var, value in settings.items():
        monkeypatch.setenv(var, value)
    assert sm.main(["check"]) == 1
    out = capsys.readouterr().out
    for what in ("SENSE_DATA_DIR", "SENSE_HF_CACHE", "SENSE_CLAP", "sample audio ~/genres (optional)"):
        line = line_named(out, what)
        assert line.startswith("MISMATCH") and "~ is not expanded" in line and "full path" in line
    assert "not in ~/" not in out  # no MISSING lines that say the files are not where they are
    assert "Real mode is not ready: fix SENSE_HF_CACHE, SENSE_CLAP, SENSE_DATA_DIR (see" in out

    d = datasets(machine.tmp)
    assert sm.main(["rebuild", "--gtzan", str(d.gtzan), "--esc50", str(d.esc50), "--captions", str(d.captions),
                    "--dry-run"]) == 0
    notes = [line for line in capsys.readouterr().out.splitlines() if line.startswith("Note:")]
    assert len(notes) == 2 and "SENSE_DATA_DIR is ~/sense-data" in notes[0] and "SENSE_CLAP is ~/clap" in notes[1]


def test_check_passes_when_the_weight_lookups_find_them(machine, capsys, monkeypatch):
    write_embeddings(machine.data, machine.sums)
    install_weights(machine.hub, qwen=False, clap_weights=False)
    found = machine.tmp / "found"
    monkeypatch.setattr(models, "_local_snapshot", lambda source: found if source == "Qwen/Qwen3.5-0.8B" else None)
    monkeypatch.setattr(specialists, "_cached_safetensors_dir", lambda source: found)
    assert sm.main(["check"]) == 0
    out = capsys.readouterr().out
    assert status_of(out, "Qwen3.5 0.8B weights") == "OK"
    assert status_of(out, "CLAP weights") == "OK"


def test_check_needs_the_clap_config_files_from_main(machine, capsys):
    write_embeddings(machine.data, machine.sums)
    install_weights(machine.hub, clap_main=False)
    assert sm.main(["check"]) == 1
    out = capsys.readouterr().out
    assert status_of(out, "CLAP config and processor") == "MISSING"
    assert status_of(out, "CLAP weights") == "OK"


def test_check_finds_weights_in_sense_hf_cache_and_local_directories(machine, capsys, monkeypatch):
    write_embeddings(machine.data, machine.sums)
    install_weights(machine.hub, qwen=False)
    other = machine.tmp / "other-cache"
    snapshot(other, "Qwen/Qwen3.5-0.8B", ["config.json", "model.safetensors"])
    monkeypatch.setenv("SENSE_HF_CACHE", str(other))
    local_2b = machine.tmp / "qwen2b"
    local_2b.mkdir()
    (local_2b / "config.json").write_text("{}")
    (local_2b / "model.safetensors").write_bytes(b"x")
    monkeypatch.setenv("SENSE_QWEN_2B", str(local_2b))
    assert sm.main(["check"]) == 0
    out = capsys.readouterr().out
    assert status_of(out, "Qwen3.5 0.8B weights") == "OK"
    assert status_of(out, "Qwen3.5 2B weights (optional)") == "OK"
    assert str(other) in out and str(local_2b) in out


def test_check_reports_each_sample_dir_without_failing(machine, capsys, monkeypatch):
    install_weights(machine.hub)
    write_embeddings(machine.data, machine.sums)
    there, gone = machine.tmp / "genres", machine.tmp / "gone"
    there.mkdir()
    monkeypatch.setenv("SENSE_SAMPLE_DIRS", f"{there}{os.pathsep}{gone}")
    assert sm.main(["check"]) == 0
    out = capsys.readouterr().out
    assert status_of(out, f"sample audio {there}") == "OK"
    assert status_of(out, f"sample audio {gone}") == "MISSING"


def test_check_fails_in_fake_mode(machine, capsys, monkeypatch):
    install_weights(machine.hub)
    write_embeddings(machine.data, machine.sums)
    monkeypatch.setenv("SENSE_FAKE", "1")
    assert sm.main(["check"]) == 1
    assert status_of(capsys.readouterr().out, "fake mode off") == "MISMATCH"


def test_check_loads_clap_only_when_asked(machine, capsys, monkeypatch):
    install_weights(machine.hub)
    write_embeddings(machine.data, machine.sums)
    calls = []

    def fake_load():
        calls.append(1)
        return False, "OSError: model.safetensors is broken"

    monkeypatch.setattr(sm, "_try_load_clap", fake_load)
    assert sm.main(["check"]) == 0
    assert calls == []
    assert sm.main(["check", "--load-clap"]) == 1
    assert calls == [1]
    assert status_of(capsys.readouterr().out, "CLAP loads") == "MISMATCH"


def test_check_does_not_import_torch(tmp_path):
    """Run check (and a fetch dry run) in a fresh interpreter and look at what it imported."""
    hub = tmp_path / "hub"
    hub.mkdir()
    code = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('setup_machine', {str(SCRIPT)!r})\n"
        "sm = importlib.util.module_from_spec(spec)\n"
        "sys.modules['setup_machine'] = sm\n"
        "spec.loader.exec_module(sm)\n"
        "sm.main(['check'])\n"
        "sm.main(['fetch', '--dry-run'])\n"
        "print('TORCH_IMPORTED', 'torch' in sys.modules)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, env=_hermetic_env(tmp_path, hub),
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "TORCH_IMPORTED False" in result.stdout
    assert "Qwen3.5 0.8B weights" in result.stdout


def _hermetic_env(tmp_path: Path, hub: Path) -> dict[str, str]:
    """Environment for a child process: explicit settings win over any .env file the child finds."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(("SENSE_", "HF_"))}
    env.update(
        SENSE_FAKE="", SENSE_DATA_DIR=str(tmp_path / "data"), SENSE_HF_CACHE=str(hub), SENSE_SAMPLE_DIRS="",
        SENSE_QWEN_0P8B="Qwen/Qwen3.5-0.8B", SENSE_QWEN_2B="Qwen/Qwen3.5-2B", SENSE_CLAP="laion/clap-htsat-unfused",
        HF_HUB_CACHE=str(hub), HF_HUB_OFFLINE="1",
    )
    return env


# ----------------------------------------------------------------------------- fetch
def _no_download(*args, **kwargs):
    raise AssertionError("snapshot_download must not be called")


def test_fetch_dry_run_fetches_nothing_and_lists_both_clap_revisions(machine, capsys, monkeypatch):
    monkeypatch.setattr(huggingface_hub, "snapshot_download", _no_download)
    assert sm.main(["fetch", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "Qwen/Qwen3.5-0.8B" in out
    clap_lines = [line for line in out.splitlines() if "laion/clap-htsat-unfused" in line]
    assert any("revision main" in line for line in clap_lines)
    assert any("revision refs/pr/3" in line and "model.safetensors" in line for line in clap_lines)
    assert "pytorch_model.bin" not in out

    assert sm.main(["fetch", "--model", "qwen3.5-2b", "--dry-run"]) == 0
    assert "Qwen/Qwen3.5-2B" in capsys.readouterr().out


def test_fetch_offers_only_the_0p8b_and_2b(machine, capsys):
    with pytest.raises(SystemExit):
        sm.main(["fetch", "--model", "qwen3.5-4b", "--dry-run"])
    assert "4b" not in " ".join(sm.QWEN).lower()


def test_fetch_downloads_qwen_into_sense_hf_cache_and_clap_twice(machine, capsys, monkeypatch):
    calls = []

    def fake_download(repo_id, **kwargs):
        calls.append({"repo_id": repo_id, **kwargs})
        return str(machine.tmp / "snapshot")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_download)
    monkeypatch.setenv("SENSE_HF_CACHE", str(machine.tmp / "hf"))
    assert sm.main(["fetch"]) == 0
    qwen, clap_main, clap_weights = calls
    assert qwen["repo_id"] == "Qwen/Qwen3.5-0.8B" and qwen["cache_dir"] == str(machine.tmp / "hf")
    assert any(fnmatch.fnmatch("model.safetensors", p) for p in qwen["allow_patterns"])
    assert clap_main["repo_id"] == "laion/clap-htsat-unfused" and clap_main["revision"] == "main"
    for name in CLAP_MAIN:
        assert any(fnmatch.fnmatch(name, p) for p in clap_main["allow_patterns"]), name
    assert not any(fnmatch.fnmatch("pytorch_model.bin", p) for p in clap_main["allow_patterns"])
    assert clap_weights["revision"] == "refs/pr/3" and clap_weights["allow_patterns"] == ["model.safetensors"]
    assert clap_main["cache_dir"] is None and clap_weights["cache_dir"] is None  # where transformers looks
    assert "check" in capsys.readouterr().out


def test_fetch_skips_a_model_given_as_a_local_directory(machine, capsys, monkeypatch):
    calls = []
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda repo_id, **kw: calls.append(repo_id) or "x")
    monkeypatch.setenv("SENSE_QWEN_0P8B", str(machine.tmp))
    assert sm.main(["fetch"]) == 0
    assert calls == ["laion/clap-htsat-unfused", "laion/clap-htsat-unfused"]
    assert "nothing to fetch" in capsys.readouterr().out


def test_fetch_reports_a_failed_download(machine, capsys, monkeypatch):
    def offline(repo_id, **kwargs):
        raise OSError("no network")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", offline)
    assert sm.main(["fetch"]) == 1
    out = capsys.readouterr().out
    assert "Could not fetch Qwen/Qwen3.5-0.8B" in out and "no network" in out


# ----------------------------------------------------------------------------- rebuild
def datasets(tmp: Path) -> SimpleNamespace:
    gtzan, esc50, captions = tmp / "gtzan" / "genres", tmp / "ESC-50-master", tmp / "train.csv"
    (gtzan / "blues").mkdir(parents=True)
    (gtzan / "blues" / "blues.00000.wav").write_bytes(b"")
    (esc50 / "meta").mkdir(parents=True)
    (esc50 / "audio").mkdir()
    (esc50 / "meta" / "esc50.csv").write_text("filename,fold,target,category\n")
    captions.write_text("audiocap_id,youtube_id,start_time,caption\n1,abc,0,A dog barks\n")
    return SimpleNamespace(gtzan=gtzan, esc50=esc50, captions=captions, out=tmp / "out")


def _rebuild_args(d: SimpleNamespace, *extra: str) -> list[str]:
    return ["rebuild", "--gtzan", str(d.gtzan), "--esc50", str(d.esc50), "--captions", str(d.captions),
            "--out", str(d.out), *extra]


def test_rebuild_dry_run_prints_three_isolated_commands(machine, capsys, monkeypatch):
    monkeypatch.setattr(sm.subprocess, "run", _no_download)
    d = datasets(machine.tmp)
    assert sm.main(_rebuild_args(d, "--dry-run")) == 0
    out = capsys.readouterr().out
    commands = [shlex.split(line) for line in out.splitlines() if " -I " in line]
    assert len(commands) == 3
    scripts = ["clap_embed_gtzan.py", "clap_embed.py", "clap_text.py"]
    for command, script, data_arg, data_path, name in zip(
        commands, scripts, ["--gtzan", "--esc50", "--captions"], [d.gtzan, d.esc50, d.captions], sm.EMBEDDING_FILES
    ):
        assert command[:2] == [sys.executable, "-I"]
        assert Path(command[2]).name == script
        assert command[command.index(data_arg) + 1] == str(data_path)
        assert command[command.index("--out") + 1] == str(d.out / name)
        assert command[command.index("--model") + 1] == "laion/clap-htsat-unfused"
    assert commands[2][commands[2].index("--limit") + 1] == str(sm.CAPTION_LIMIT) == "12000"
    assert not d.out.exists()


def test_rebuild_dry_run_defaults_to_the_data_dir_and_notes_missing_inputs(machine, capsys):
    tmp = machine.tmp
    args = ["rebuild", "--gtzan", str(tmp / "nope"), "--esc50", str(tmp / "nope"), "--captions",
            str(tmp / "nope.csv"), "--dry-run"]
    assert sm.main(args) == 0
    out = capsys.readouterr().out
    assert str(machine.data / "gtzan_clap.npz") in out
    assert out.count("Note:") == 3


def test_rebuild_runs_the_scripts_and_prints_checksums(machine, capsys, monkeypatch):
    calls = []

    def fake_run(command, check=False):
        calls.append(command)
        Path(command[command.index("--out") + 1]).write_bytes(Path(command[2]).name.encode())
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(sm.subprocess, "run", fake_run)
    d = datasets(machine.tmp)
    assert sm.main(_rebuild_args(d)) == 0
    out = capsys.readouterr().out
    assert len(calls) == 3
    for name in sm.EMBEDDING_FILES:
        digest = hashlib.sha256((d.out / name).read_bytes()).hexdigest()
        assert f"{digest}  {name}" in out
    assert "check --rebuilt" in out

    assert sm.main(_rebuild_args(d)) == 1  # the files are there now: refuse to overwrite them
    assert "--force" in capsys.readouterr().out
    assert len(calls) == 3
    assert sm.main(_rebuild_args(d, "--force")) == 0
    assert len(calls) == 6


def test_rebuild_stops_at_the_first_failing_script(machine, capsys, monkeypatch):
    calls = []
    monkeypatch.setattr(sm.subprocess, "run", lambda command, check=False: calls.append(command)
                        or subprocess.CompletedProcess(command, 2))
    d = datasets(machine.tmp)
    assert sm.main(_rebuild_args(d)) == 1
    assert len(calls) == 1
    assert "gtzan_clap.npz failed" in capsys.readouterr().out


def test_rebuild_refuses_inputs_in_the_wrong_shape(machine, capsys, monkeypatch):
    monkeypatch.setattr(sm.subprocess, "run", _no_download)
    d = datasets(machine.tmp)
    (d.gtzan / "blues" / "blues.00000.wav").rename(d.gtzan / "blues" / "blues.00000.au")
    d.captions.write_text("audiocap_id,text\n1,A dog barks\n")
    assert sm.main(_rebuild_args(d)) == 1
    out = capsys.readouterr().out
    assert ".au files" in out and "'caption' column" in out


# ----------------------------------------------------------------------------- checksums
def test_checksums_write_records_the_three_files(machine, capsys):
    write_embeddings(machine.data)
    assert sm.main(["checksums", "--write"]) == 0
    expected = "".join(
        f"{hashlib.sha256((machine.data / name).read_bytes()).hexdigest()}  {name}\n" for name in sm.EMBEDDING_FILES
    )
    assert machine.sums.read_text() == expected
    assert sm._read_checksums(machine.sums) == {
        name: line.split()[0] for name, line in zip(sm.EMBEDDING_FILES, expected.splitlines())
    }


def test_checksums_refuses_to_write_an_incomplete_list(machine, capsys):
    write_embeddings(machine.data, skip=("esc50_clap.npz",))
    machine.sums.write_text("keep me\n")
    assert sm.main(["checksums"]) == 1
    assert sm.main(["checksums", "--write"]) == 1
    assert machine.sums.read_text() == "keep me\n"
    assert "MISSING  esc50_clap.npz" in capsys.readouterr().out


def test_committed_checksum_file_lists_the_three_files():
    lines = (SCRIPT.parent / "embeddings.sha256").read_text().splitlines()
    assert [re.fullmatch(r"[0-9a-f]{64}  (\S+)", line).group(1) for line in lines] == list(sm.EMBEDDING_FILES)


def test_script_runs_from_the_command_line(tmp_path):
    hub = tmp_path / "hub"
    hub.mkdir()
    result = subprocess.run(
        [sys.executable, "-I", str(SCRIPT), "fetch", "--dry-run"], cwd=tmp_path,
        env=_hermetic_env(tmp_path, hub), capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "refs/pr/3" in result.stdout
