"""sense.adapters: running agent-written adapters in a sandbox, probe folders, adapter bridges and verify_files.

Fake mode, offline: no LLM is asked anything; the adapters are written here by hand."""

from __future__ import annotations

import importlib.util
import io
import json
import os
import shutil
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from sense import adapters
from sense.adapters import (
    AdapterBridge,
    AdapterError,
    ProbeSet,
    code_imports,
    load_adapter_bridge,
    probe_folder,
    run_adapter,
    toolbox,
    verify_files,
)
from sense.contracts import Bridge, FitRequest

SANDBOXED = adapters.sandboxed()
COLOURS = ["red", "green", "blue", "yellow", "purple", "orange"]

FIRST_LINE = '''
def adapt(path):
    with open(path, "rb") as f:
        return {"text": f.readline().decode("utf-8", "replace").strip()}
'''
USELESS = '''
def adapt(path):
    return {"text": "a file of some kind"}
'''


@pytest.fixture(autouse=True)
def _fake_home(tmp_path, monkeypatch):
    monkeypatch.setenv("SENSE_FAKE", "1")
    monkeypatch.setenv("SENSE_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(adapters, "_cache", type(adapters._cache)())


@pytest.fixture
def a_file(tmp_path) -> Path:
    path = tmp_path / "input.dat"
    path.write_bytes(b"hello world\nsecond line\n")
    return path


def _png_size(data: bytes) -> tuple[int, int]:
    img = Image.open(io.BytesIO(data))
    assert img.format == "PNG"
    return img.size


# ----------------------------------------------------------------------------- toolbox
def test_toolbox_lists_the_allow_list_that_can_be_imported():
    tools = toolbox()
    for name in ["numpy", "scipy", "PIL", "json", "csv", "math", "struct", "zipfile", "gzip", "io", "re",
                 "xml.etree", "statistics"]:
        assert name in tools
    for name in ["trimesh", "rdkit"]:
        assert (name in tools) == (importlib.util.find_spec(name) is not None)
    assert ("ffmpeg (via subprocess)" in tools) == (adapters._binary("ffmpeg") is not None)
    assert "os" not in tools and "socket" not in tools


def test_code_imports_names_top_level_modules_and_commands():
    code = "import numpy as np\nfrom PIL import Image\nimport xml.etree.ElementTree as ET\nimport subprocess\n" \
           "def adapt(p):\n    subprocess.run(['ffprobe', p])\n"
    assert code_imports(code) == ["PIL", "ffprobe", "numpy", "subprocess", "xml"]
    assert code_imports("def broken(:") == []


# ----------------------------------------------------------------------------- run_adapter
def test_run_adapter_returns_text_and_keeps_prints_out_of_the_result(a_file):
    code = '''
import os
def adapt(path):
    print("chatter on stdout")
    os.system("echo from a child process")
    with open(path) as f:
        return {"text": f.readline().strip()}
'''
    out = run_adapter(code, a_file)
    assert out["text"] == "hello world"
    assert out["notes"] == []
    assert out["seconds"] >= 0


def test_run_adapter_gets_the_path_as_a_str(a_file):
    out = run_adapter('def adapt(path):\n    return {"text": type(path).__name__ + " " + path}\n', a_file)
    assert out["text"] == f"str {a_file.resolve()}"


def test_run_adapter_normalises_images_to_png_of_at_most_512_px(a_file):
    code = '''
import io
import numpy as np
from PIL import Image
def adapt(path):
    wide = Image.new("RGB", (1200, 300), "red")
    jpeg = io.BytesIO()
    Image.new("L", (40, 60), 128).save(jpeg, "JPEG")
    Image.new("RGBA", (800, 800), (0, 0, 255, 128)).save("drawn.png")
    return {"images": [wide, jpeg.getvalue(), "drawn.png", np.zeros((20, 30, 3), np.uint8)]}
'''
    out = run_adapter(code, a_file)
    assert [_png_size(b) for b in out["images"]] == [(512, 128), (40, 60), (512, 512), (30, 20)]


def test_run_adapter_caps_text_and_images(a_file):
    out = run_adapter('def adapt(path):\n    return {"text": "x" * 10000}\n', a_file)
    assert len(out["text"]) <= adapters.MAX_TEXT_CHARS
    assert "cut by the harness" in out["text"] and "10000" in out["text"]
    assert out["notes"] and "cut" in out["notes"][0]

    code = '''
from PIL import Image
def adapt(path):
    return {"images": [Image.new("RGB", (16, 16), (k * 20, 0, 0)) for k in range(12)]}
'''
    out = run_adapter(code, a_file)
    assert len(out["images"]) == adapters.MAX_IMAGES
    assert "12 images" in out["notes"][0]


def test_run_adapter_stops_an_adapter_that_runs_too_long(a_file):
    started = time.monotonic()
    with pytest.raises(AdapterError) as caught:
        run_adapter("import time\ndef adapt(path):\n    time.sleep(30)\n", a_file, timeout=1)
    assert caught.value.cause == "timeout"
    assert "longer than 1 s" in caught.value.reason
    assert time.monotonic() - started < 10


def test_run_adapter_reports_the_error_with_the_traceback_tail(a_file):
    code = '''
def read_header(path):
    raise ValueError("bad header")

def adapt(path):
    return read_header(path)
'''
    with pytest.raises(AdapterError) as caught:
        run_adapter(code, a_file)
    err = caught.value
    assert err.cause == "exception"
    assert err.reason == "adapt(path) raised ValueError: bad header"
    assert 'File "adapter.py"' in err.traceback and "read_header" in err.traceback
    assert "runner.py" not in err.traceback and "sense-adapter-" not in err.traceback
    assert "bad header" in str(err) and "read_header" in str(err)


@pytest.mark.parametrize(
    "code, expected",
    [
        ("def adapt(path):\n    return 42\n", "returned a int"),
        ('def adapt(path):\n    return {"text": 1}\n', '"text" is a int'),
        ('def adapt(path):\n    return {"txt": "hi"}\n', "keys ['txt']"),
        ('def adapt(path):\n    return {"text": "a", "images": []}\n', "keys ['images', 'text']"),
        ('def adapt(path):\n    return {"images": []}\n', "no images"),
        ('def adapt(path):\n    return {"images": [3.5]}\n', "image 1 is a float"),
        ("x = 1\n", "defines no function adapt"),
    ],
)
def test_run_adapter_refuses_output_that_breaks_the_contract(a_file, code, expected):
    with pytest.raises(AdapterError) as caught:
        run_adapter(code, a_file)
    assert caught.value.cause == "output"
    assert expected in caught.value.reason


def test_run_adapter_reports_syntax_errors_and_hard_exits(a_file):
    with pytest.raises(AdapterError) as caught:
        run_adapter("def adapt(path)\n    return {}\n", a_file)
    assert "SyntaxError" in caught.value.reason
    with pytest.raises(AdapterError) as caught:
        run_adapter('import os, sys\ndef adapt(path):\n    print("dying", file=sys.stderr)\n    os._exit(3)\n', a_file)
    assert caught.value.cause == "exit" and "code 3" in caught.value.reason
    assert "dying" in caught.value.traceback


def test_run_adapter_refuses_a_missing_file(tmp_path):
    with pytest.raises(ValueError, match="not a file"):
        run_adapter(FIRST_LINE, tmp_path / "nothing.bin")


def test_run_adapter_strips_secrets_from_the_environment(a_file, monkeypatch):
    for name, value in {
        "OPENAI_API_KEY": "sk-test-not-real",
        "OPENAI_BASE_URL": "https://example.invalid",
        "LINKUP_API_KEY": "lk-test",
        "DATABRICKS_HOST": "https://example.invalid",
        "MY_SECRET_THING": "s",
        "SOME_TOKEN": "t",
        "DB_PASSWORD": "p",
        "SENSE_HARMLESS_SETTING": "kept",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("NO_PROXY", "localhost")
    code = '''
import json, os
def adapt(path):
    keys = sorted(os.environ)
    shown = [k for k in keys if k.startswith(("SENSE_", "OPENAI", "LINKUP", "DATABRICKS")) or "PROXY" in k.upper()
             or k in ("HOME", "MY_SECRET_THING", "SOME_TOKEN", "DB_PASSWORD")]
    return {"text": json.dumps({"keys": keys, "env": {k: os.environ[k] for k in shown}})}
'''
    out = json.loads(run_adapter(code, a_file)["text"])
    env = out["env"]
    for name in ["OPENAI_API_KEY", "OPENAI_BASE_URL", "LINKUP_API_KEY", "DATABRICKS_HOST", "MY_SECRET_THING",
                 "SOME_TOKEN", "DB_PASSWORD"]:
        assert name not in out["keys"]
    assert not [k for k in out["keys"] if any(w in k.upper() for w in ("KEY", "TOKEN", "SECRET", "PASSWORD"))]
    assert env["SENSE_HARMLESS_SETTING"] == "kept"
    for name in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy"]:
        assert env[name] == adapters.BLOCKED_PROXY
    assert env["NO_PROXY"] == "" and env["no_proxy"] == ""
    assert env["HOME"] != str(Path.home())


def test_run_adapter_blocks_the_network(a_file):
    proxies = '''
import json, urllib.request
def adapt(path):
    return {"text": json.dumps(urllib.request.getproxies())}
'''
    seen = json.loads(run_adapter(proxies, a_file)["text"])
    assert seen.get("http") == seen.get("https") == adapters.BLOCKED_PROXY
    assert "no" not in seen or seen["no"] == ""
    if not SANDBOXED:
        pytest.skip("sandbox-exec is not available: only the proxy settings stand in the way of HTTP here")
    code = '''
import urllib.request
def adapt(path):
    with urllib.request.urlopen("http://example.com", timeout=5) as r:
        return {"text": r.read(100).decode("utf-8", "replace")}
'''
    with pytest.raises(AdapterError) as caught:
        run_adapter(code, a_file, timeout=20)
    assert caught.value.cause in ("exception", "timeout")
    direct = '''
import socket
def adapt(path):
    s = socket.create_connection(("1.1.1.1", 80), timeout=5)
    s.close()
    return {"text": "connected"}
'''
    with pytest.raises(AdapterError) as caught:
        run_adapter(direct, a_file, timeout=20)
    assert "Operation not permitted" in caught.value.reason or "PermissionError" in caught.value.reason


@pytest.mark.skipif(not SANDBOXED, reason="needs macOS sandbox-exec")
def test_sandbox_hides_env_files_and_allows_writes_only_to_temporary_places(a_file, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("NOT_A_REAL_KEY=1\n")
    with pytest.raises(AdapterError) as caught:
        run_adapter(f'def adapt(path):\n    return {{"text": open({str(env_file)!r}).read()}}\n', a_file)
    assert "Operation not permitted" in caught.value.reason

    outside = Path(adapters.__file__).resolve().parent.parent / f".adapter-write-probe-{uuid.uuid4().hex}"
    try:
        with pytest.raises(AdapterError) as caught:
            code = f'def adapt(path):\n    open({str(outside)!r}, "w").write("x")\n    return {{"text": "wrote"}}\n'
            run_adapter(code, a_file)
        assert "Operation not permitted" in caught.value.reason
        assert not outside.exists()
    finally:
        if outside.exists():
            outside.unlink()
    code = '''
import os, tempfile
def adapt(path):
    open("in_work_dir.txt", "w").write("x")
    with tempfile.NamedTemporaryFile(suffix=".txt") as f:
        f.write(b"x")
    return {"text": "ok"}
'''
    assert run_adapter(code, a_file)["text"] == "ok"


# ----------------------------------------------------------------------------- probe folders
def _probe_tree(root: Path, labels: list[str], per_label: int, *, line=lambda label, k: f"colour: {label}") -> Path:
    for label in labels:
        folder = root / label
        folder.mkdir(parents=True, exist_ok=True)
        for k in range(per_label):
            (folder / f"{label}_{k:02d}.thing").write_bytes(f"{line(label, k)}\nsample {k}\n".encode())
    return root


def test_probe_folder_splits_dedups_and_anonymises(tmp_path):
    root = _probe_tree(tmp_path / "examples", ["sea_waves", "dog-bark"], 6)
    (root / "sea_waves" / "copy_of_00.thing").write_bytes((root / "sea_waves" / "sea_waves_00.thing").read_bytes())
    shared = b"the same bytes under two labels\n"
    (root / "sea_waves" / "shared.thing").write_bytes(shared)
    (root / "dog-bark" / "shared.thing").write_bytes(shared)
    (root / "dog-bark" / ".DS_Store").write_bytes(b"hidden")
    (root / "dog-bark" / "empty.thing").write_bytes(b"")
    (root / "_ignored").mkdir()
    (root / "_ignored" / "x.thing").write_bytes(b"not a label")

    probes = probe_folder(root, question="What does the file show?", modality="mystery", source="test", seed=1)
    assert probes.name == "examples"
    assert probes.labels == ["dog bark", "sea waves"]
    assert len(probes.items) == 12
    assert np.bincount(probes.y).tolist() == [6, 6]
    assert probes.meta["duplicates_collapsed"] == 1
    assert sorted(probes.meta["conflicts_dropped"]) == ["dog-bark/shared.thing", "sea_waves/shared.thing"]
    assert probes.meta["skipped"] == {"hidden": 1, "empty": 1, "subfolders": 0}
    assert probes.provenance == "user" and probes.source == "test"

    splits = [set(probes.fit_idx.tolist()), set(probes.dev_idx.tolist()), set(probes.test_idx.tolist())]
    assert not (splits[0] & splits[1]) and not (splits[0] & splits[2]) and not (splits[1] & splits[2])
    assert set.union(*splits) == set(range(12))
    for c in range(2):
        for split in (probes.fit_idx, probes.dev_idx, probes.test_idx):
            assert (probes.y[split] == c).sum() >= 1

    copies = Path(os.environ["SENSE_HOME"]) / "probes" / "examples"
    originals = {p.read_bytes(): p for p in root.rglob("*.thing")}
    for path, label in probes.items:
        assert path.parent == copies
        assert len(path.stem) == 16 and path.suffix == ".thing"
        assert not any(word in path.name for word in ["sea", "waves", "dog", "bark"])
        assert originals[path.read_bytes()].parent.name.replace("_", " ").replace("-", " ") == probes.labels[label]

    again = probe_folder(root, question="What does the file show?", modality="mystery", seed=1)
    assert [p for p, _ in again.items] == [p for p, _ in probes.items]
    assert again.test_idx.tolist() == probes.test_idx.tolist()
    assert json.dumps(probes.to_dict())


def test_probe_folder_caps_each_label(tmp_path):
    root = _probe_tree(tmp_path / "many", ["a", "b"], 9)
    probes = probe_folder(root, question="q?", modality="m", max_per_label=5)
    assert np.bincount(probes.y).tolist() == [5, 5]
    assert probes.meta["capped"] == 8


def test_probe_folder_explains_what_to_fix(tmp_path):
    one = _probe_tree(tmp_path / "one", ["only"], 6)
    with pytest.raises(ValueError, match="at least 2"):
        probe_folder(one, question="q?", modality="m")
    short = _probe_tree(tmp_path / "short", ["plenty", "few"], 6)
    for extra in list((short / "few").iterdir())[:4]:
        extra.unlink()
    with pytest.raises(ValueError, match=r"'few' has 2.*min_per_label"):
        probe_folder(short, question="q?", modality="m")
    with pytest.raises(ValueError, match="at least 3"):
        probe_folder(short, question="q?", modality="m", min_per_label=2)


# ----------------------------------------------------------------------------- the bridge
def test_adapter_bridge_is_a_file_bridge_that_saves_and_loads(tmp_path, a_file):
    bridge = AdapterBridge(
        kind="text", modality="mystery", model_name="fake", code=FIRST_LINE,
        manifest={"format": "dat", "description": "first line", "written_by": "test", "attempts": 2},
    )
    assert isinstance(bridge, Bridge)
    assert bridge.name == "adapter:mystery-dat"
    assert bridge.manifest["route"] == "text" and bridge.manifest["libraries"] == []
    with pytest.raises(ValueError, match="file-based"):
        bridge.from_embedding(np.zeros(4))
    inp = bridge.prepare(a_file)
    assert inp.kind == "text" and inp.text == "hello world" and inp.modality == "mystery"
    assert "hello world" in inp.detail

    bridge.save(tmp_path / "saved")
    meta = json.loads((tmp_path / "saved" / "bridge.json").read_text())
    assert meta["adapter"] is True and meta["kind"] == "text" and meta["name"] == bridge.name
    assert meta["manifest"]["attempts"] == 2 and meta["manifest"]["written_by"] == "test"
    assert (tmp_path / "saved" / "adapter.py").read_text() == FIRST_LINE

    loaded = load_adapter_bridge(tmp_path / "saved")
    assert (loaded.name, loaded.kind, loaded.modality, loaded.model_name) == (bridge.name, "text", "mystery", "fake")
    assert loaded.manifest == bridge.manifest
    assert loaded.prepare(a_file).text == "hello world"


def test_adapter_bridge_caches_per_file_content(a_file, tmp_path, monkeypatch):
    calls = []
    real = adapters.run_adapter

    def counting(code, path, *, timeout=60):
        calls.append(Path(path))
        return real(code, path, timeout=timeout)

    monkeypatch.setattr(adapters, "run_adapter", counting)
    bridge = AdapterBridge(kind="text", modality="mystery", model_name="fake", code=FIRST_LINE)
    twin = tmp_path / "twin.dat"
    twin.write_bytes(a_file.read_bytes())
    assert bridge.prepare(a_file).text == bridge.prepare(twin).text == "hello world"
    assert len(calls) == 1
    failing = AdapterBridge(kind="text", modality="mystery", model_name="fake", code="def adapt(p):\n    1 / 0\n")
    for _ in range(2):
        with pytest.raises(AdapterError, match="ZeroDivisionError"):
            failing.prepare(a_file)
    assert len(calls) == 2


def test_image_bridge_tiles_the_images_into_one_contact_sheet(a_file):
    code = '''
from PIL import Image
def adapt(path):
    return {"images": [Image.new("RGB", (512, 512), c) for c in ("red", "green", "blue", "white")]}
'''
    bridge = AdapterBridge(kind="image", modality="3d model", model_name="fake", code=code, slug="views")
    assert bridge.name == "adapter:views"
    inp = bridge.prepare(a_file)
    assert inp.kind == "image" and inp.detail == "4 images from the agent-written adapter"
    width, height = _png_size(inp.image)
    assert width <= adapters.SHEET_WIDTH and height <= adapters.SHEET_WIDTH

    one_image = 'from PIL import Image\ndef adapt(p):\n    return {"images": Image.new("L", (64, 32))}\n'
    single = AdapterBridge(kind="image", modality="m", model_name="fake", code=one_image)
    inp = single.prepare(a_file)
    assert _png_size(inp.image) == (64, 32) and inp.detail == "1 image from the agent-written adapter"
    with pytest.raises(AdapterError, match="returned text"):
        AdapterBridge(kind="image", modality="m", model_name="fake", code=FIRST_LINE).prepare(a_file)


# ----------------------------------------------------------------------------- verify_files
def _colour_probes(tmp_path: Path, per_label: int = 10) -> ProbeSet:
    root = _probe_tree(tmp_path / "colours", COLOURS, per_label)
    return probe_folder(root, question="Which colour does the file describe?", modality="colour file",
                        provenance="generated", source="generated by the test")


def _fake_model(name: str = "fake"):
    from sense.models import load_model

    return load_model(name)


def test_verify_files_scores_a_working_adapter_high_and_swapped_input_low(tmp_path):
    probes = _colour_probes(tmp_path)
    model = _fake_model()
    bridge = AdapterBridge(kind="text", modality="colour file", model_name="fake", code=FIRST_LINE)
    card = verify_files(model, bridge, probes, FitRequest(model="fake", dataset=probes.name, modality="colour file"))
    assert card.dataset == probes.name and card.kind == "text" and card.bridge == "adapter:colour-file"
    assert card.verdict == "candidate" and card.reason == ""
    assert card.n == len(probes.test_idx) and card.chance == 0.2
    assert card.accuracy >= 0.9
    assert card.controls["swapped"] <= 0.3
    assert set(card.controls) == {"swapped", "no_input"}
    assert card.interval[0] <= card.accuracy <= card.interval[1]

    from agent.planner import RulePlanner

    assert RulePlanner().assess(FitRequest(model="fake", dataset=probes.name), card)[0]


def test_verify_files_scores_a_useless_adapter_near_chance(tmp_path):
    probes = _colour_probes(tmp_path, per_label=20)
    bridge = AdapterBridge(kind="text", modality="colour file", model_name="fake-api", code=USELESS)
    card = verify_files(_fake_model("fake-api"), bridge, probes, FitRequest(model="fake-api", dataset=probes.name))
    assert card.n == len(probes.test_idx)
    assert card.accuracy <= 0.45
    assert abs(card.accuracy - card.controls["swapped"]) <= 0.25

    from agent.planner import RulePlanner

    assert not RulePlanner().assess(FitRequest(model="fake-api", dataset=probes.name), card)[0]


def test_verify_files_counts_adapter_failures_as_wrong(tmp_path):
    probes = _colour_probes(tmp_path)
    code = FIRST_LINE.replace(
        "    with open", '    if "blue" in open(path).read():\n        raise RuntimeError("cannot read blue")\n    with open'
    )
    bridge = AdapterBridge(kind="text", modality="colour file", model_name="fake", code=code)
    card = verify_files(_fake_model(), bridge, probes, FitRequest(model="fake", dataset=probes.name))
    n_blue = int((probes.y[probes.test_idx] == probes.labels.index("blue")).sum())
    assert card.accuracy <= 1 - n_blue / card.n + 1e-9
    assert f"failed on {n_blue} of {card.n}" in card.reason and "cannot read blue" in card.reason

    broken = AdapterBridge(kind="text", modality="colour file", model_name="fake", code="def adapt(p):\n    1 / 0\n")
    with pytest.raises(AdapterError) as caught:
        verify_files(_fake_model(), broken, probes, FitRequest(model="fake", dataset=probes.name))
    assert caught.value.cause == "verify" and "more than half" in caught.value.reason


def test_verify_files_respects_max_items_and_the_model(tmp_path):
    probes = _colour_probes(tmp_path)
    bridge = AdapterBridge(kind="text", modality="colour file", model_name="fake", code=FIRST_LINE)
    card = verify_files(_fake_model(), bridge, probes, FitRequest(model="fake", dataset=probes.name), max_items=7)
    assert card.n == 7
    with pytest.raises(ValueError, match="built for fake"):
        verify_files(_fake_model("fake-api"), bridge, probes, FitRequest(model="fake-api", dataset=probes.name))
