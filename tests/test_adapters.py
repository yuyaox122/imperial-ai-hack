"""sense.adapters: running agent-written adapters in a sandbox, probe folders, adapter bridges and verify_files.

Fake mode, offline: no LLM is asked anything; the adapters are written here by hand."""

from __future__ import annotations

import importlib.util
import io
import json
import os
import shutil
import subprocess
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
    if not SANDBOXED:  # inside another sandbox (an agent's shell, say): run the adapters without one
        monkeypatch.setenv(adapters.UNSANDBOXED_ENV, "1")


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _gone(pids: list[int], within: float = 5.0) -> bool:
    deadline = time.monotonic() + within
    while time.monotonic() < deadline:
        if not any(_alive(p) for p in pids):
            return True
        time.sleep(0.1)
    return False


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


# ----------------------------------------------------------------------------- the kit and the toolbox on the path
def _fake_font_cache() -> Path:
    """A stand-in for matplotlib's font cache in the harness's cache folder, so no test pays the 12 s scan."""
    from sense import config

    folder = config.home() / "cache" / "matplotlib"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "fontlist-v0.0.0.json").write_text('{"_version": "fake"}')
    return folder


def _toolbox_package(name: str = "meshkit", *, top_level: bool = True, recorded: bool = True) -> Path:
    """A package installed into the toolbox the way agent.installer leaves it: files, dist-info, installed.json."""
    from agent import installer

    site_dir = installer.toolbox_dir()
    (site_dir / name).mkdir(parents=True)
    (site_dir / name / "__init__.py").write_text("VALUE = 42\n")
    info = site_dir / f"{name}-1.0.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: 1.0\n\n")
    if top_level:
        (info / "top_level.txt").write_text(f"{name}\n")
    (info / "RECORD").write_text(f"{name}/__init__.py,,\n{info.name}/METADATA,,\n")
    (site_dir / "evil.pth").write_text("import builtins; builtins.PTH_RAN = True\n")  # must never run
    ledger = site_dir.parent / "installed.json"
    rows = json.loads(ledger.read_text()) if ledger.is_file() else []
    if recorded:
        rows.append({"name": name, "version": "1.0", "sha256": "0" * 64})
    ledger.write_text(json.dumps(rows))
    return site_dir


def test_toolbox_lists_the_kit_and_the_approved_packages():
    tools = toolbox()
    kit = [t for t in tools if t.startswith("adapter_kit (")]
    assert len(kit) == 1 and "mesh_views" in kit[0] and "video_frames" in kit[0] and "\n" not in kit[0]
    assert kit[0] == f"adapter_kit ({adapters.KIT_SUMMARY})"
    assert "meshkit" not in tools  # no toolbox yet

    _toolbox_package("meshkit")
    _toolbox_package("pointkit", top_level=False)  # no top_level.txt: the RECORD names its modules
    _toolbox_package("stray", recorded=False)  # files in the toolbox that installed.json does not record
    tools = toolbox()
    assert tools[-2:] == ["meshkit", "pointkit"] and "stray" not in tools


def test_adapters_import_the_kit_and_the_toolbox_without_running_pth_files(a_file):
    _fake_font_cache()
    site_dir = _toolbox_package("meshkit")
    code = f'''
import builtins, json, os, sys
import adapter_kit
import meshkit

def adapt(path):
    try:
        open({str(site_dir.parent / "installed.json")!r}).read()
        ledger = "read"
    except PermissionError:
        ledger = "denied"
    return {{"text": json.dumps({{
        "kit": os.path.basename(adapter_kit.__file__), "has_views": hasattr(adapter_kit, "mesh_views"),
        "lib_first": sys.path[0] == os.path.dirname(adapter_kit.__file__), "toolbox_last": sys.path[-1],
        "value": meshkit.VALUE, "pth": hasattr(builtins, "PTH_RAN"), "ledger": ledger,
        "statistics": statistics.__file__, "stdlib_mean": hasattr(statistics, "mean"),
    }})}}
'''
    # A wheel in the toolbox with a module of a standard-library name must not replace that module for adapters.
    (site_dir / "statistics.py").write_text("SHADOWED = True\n")
    code = code.replace("import meshkit", "import meshkit\nimport statistics")
    seen = json.loads(run_adapter(code, a_file)["text"])
    assert seen["kit"] == "adapter_kit.py" and seen["has_views"] and seen["lib_first"]
    assert seen["toolbox_last"] == str(site_dir.resolve()) and seen["value"] == 42
    assert seen["stdlib_mean"] and not seen["statistics"].startswith(str(site_dir.resolve()))
    assert seen["pth"] is False  # sys.path.insert, never site.addsitedir
    if SANDBOXED:  # exactly the toolbox folder is readable, not the rest of config.home()
        assert seen["ledger"] == "denied"


def test_matplotlib_cache_is_built_once_and_copied_into_each_run(a_file, monkeypatch):
    pytest.importorskip("matplotlib")
    from sense import config

    monkeypatch.setenv("SOME_API_KEY", "sk-secret-value-123456789")
    monkeypatch.setattr(adapters, "_mpl_failed", set())
    builds = []
    real_run = subprocess.run

    def fake_scan(cmd, **kwargs):  # what `import matplotlib.font_manager` leaves in MPLCONFIGDIR, without 12 s
        if cmd[-1] != "import matplotlib.font_manager":
            return real_run(cmd, **kwargs)
        builds.append(kwargs["env"])
        Path(kwargs["env"]["MPLCONFIGDIR"], "fontlist-v9.9.9.json").write_text('{"_version": "9.9.9"}')
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(adapters.subprocess, "run", fake_scan)
    folder = adapters.matplotlib_cache()
    assert folder == config.home() / "cache" / "matplotlib" and (folder / "fontlist-v9.9.9.json").is_file()
    assert len(builds) == 1 and "SOME_API_KEY" not in builds[0] and builds[0]["MPLBACKEND"] == "Agg"
    assert adapters.matplotlib_cache() == folder and len(builds) == 1  # built once
    assert [p.name for p in folder.parent.iterdir()] == ["matplotlib"]  # no staging folder left behind

    code = '''
import os
import matplotlib

def adapt(path):
    here = os.environ["MPLCONFIGDIR"]
    inside = os.path.dirname(os.getcwd()) == os.path.dirname(here)
    return {"text": f"{inside} {sorted(os.listdir(here))}"}
'''
    assert run_adapter(code, a_file)["text"] == "True ['fontlist-v9.9.9.json']"
    assert len(builds) == 1

    # a build that fails is remembered, and the adapter still runs (matplotlib then scans on its own)
    monkeypatch.setenv("SENSE_HOME", str(Path(config.home()).parent / "other-home"))
    monkeypatch.setattr(adapters.subprocess, "run",
                        lambda cmd, **kw: builds.append(kw["env"]) or subprocess.CompletedProcess(cmd, 1))
    assert adapters.matplotlib_cache() is None and adapters.matplotlib_cache() is None and len(builds) == 2
    monkeypatch.setattr(adapters.subprocess, "run", real_run)
    assert run_adapter(code, a_file)["text"] == "True []"


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


def test_run_adapter_gets_a_private_copy_as_a_str(a_file):
    code = '''
import json, os
def adapt(path):
    return {"text": json.dumps([type(path).__name__, path, sorted(os.listdir(os.path.dirname(path))),
                                open(path).read()])}
'''
    kind, seen, neighbours, content = json.loads(run_adapter(code, a_file)["text"])
    assert kind == "str" and os.path.isabs(seen) and seen.endswith("/input/input.dat")
    assert str(a_file.parent) not in seen  # the original's folder and name are never shown
    assert neighbours == ["input.dat"]  # nothing beside it to compare with
    assert content == a_file.read_text()
    run_adapter('def adapt(path):\n    open(path, "a").write("changed")\n    return {"text": "x"}\n', a_file)
    assert a_file.read_text() == "hello world\nsecond line\n"  # the adapter wrote to its copy only


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


def test_run_adapter_timeout_holds_when_a_child_keeps_the_result_pipe(a_file):
    # The forked child inherits the runner's result pipe and leaves the process group: reading until the pipe
    # closes would wait for it, not for the timeout.
    code = '''
import os, time
def adapt(path):
    if os.fork() == 0:
        os.setsid()
        time.sleep(30)
        os._exit(0)
    time.sleep(30)
'''
    started = time.monotonic()
    with pytest.raises(AdapterError) as caught:
        run_adapter(code, a_file, timeout=2)
    assert caught.value.cause == "timeout"
    assert time.monotonic() - started < 8


def test_run_adapter_kills_what_the_adapter_left_running(a_file):
    code = '''
import json, os, subprocess, time
def adapt(path):
    forked = os.fork()
    if forked == 0:
        os.setsid()
        os.closerange(0, 256)
        time.sleep(30)
        os._exit(0)
    started = subprocess.Popen(["/bin/sleep", "30"], start_new_session=True, close_fds=True,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return {"text": json.dumps([forked, started.pid])}
'''
    started = time.monotonic()
    pids = json.loads(run_adapter(code, a_file, timeout=20)["text"])
    assert time.monotonic() - started < 8
    assert _gone(pids), [p for p in pids if _alive(p)]


def test_run_adapter_caps_what_it_reads_from_the_runner(a_file, monkeypatch):
    monkeypatch.setattr(adapters, "MAX_RESULT_BYTES", 1 << 20)
    code = '''
import sys
def adapt(path):
    out = sys._getframe(1).f_locals["out"]  # the runner's own result stream
    for _ in range(8):
        out.write("x" * (1 << 20))
    out.flush()
    return {"text": "fine"}
'''
    started = time.monotonic()
    with pytest.raises(AdapterError) as caught:
        run_adapter(code, a_file, timeout=30)
    assert caught.value.cause == "output" and "more than 1 MB of result data" in caught.value.reason
    assert time.monotonic() - started < 10


def test_run_adapter_applies_the_caps_again_to_a_forged_result(a_file):
    forge = '''
import base64, io, json, os, sys
from PIL import Image
def adapt(path):
    out = sys._getframe(1).f_locals["out"]
    buf = io.BytesIO()
    Image.new("RGB", SIZE, "red").save(buf, "PNG")
    image = base64.b64encode(buf.getvalue()).decode()
    out.write(json.dumps(PAYLOAD))
    out.flush()
    os._exit(0)
'''
    many = forge.replace("SIZE", "(2000, 1000)").replace("PAYLOAD", '{"ok": True, "images": [image] * 20}')
    out = run_adapter(many, a_file)
    assert len(out["images"]) == adapters.MAX_IMAGES and "20 images" in out["notes"][0]
    assert all(_png_size(b) == (512, 256) for b in out["images"])

    long_text = forge.replace("SIZE", "(1, 1)").replace("PAYLOAD", '{"ok": True, "text": "y" * 100000}')
    out = run_adapter(long_text, a_file)
    assert len(out["text"]) <= adapters.MAX_TEXT_CHARS and "cut by the harness" in out["text"]

    huge = forge.replace("SIZE", "(5000, 5000)").replace("PAYLOAD", '{"ok": True, "images": [image]}')
    with pytest.raises(AdapterError, match="too large to decode") as caught:
        run_adapter(huge, a_file)
    assert caught.value.cause == "output"


def test_run_adapter_stops_an_adapter_that_writes_or_holds_too_much(a_file, monkeypatch):
    monkeypatch.setattr(adapters, "CHECK_EVERY", 0.1)
    monkeypatch.setattr(adapters, "DISK_BUDGET", 5 << 20)
    writes = '''
import time
def adapt(path):
    with open("big.bin", "wb") as f:
        for _ in range(8):
            f.write(b"x" * (1 << 20))
    time.sleep(20)
'''
    with pytest.raises(AdapterError) as caught:
        run_adapter(writes, a_file, timeout=30)
    assert caught.value.cause == "limit" and "more than 5 MB of files" in caught.value.reason

    monkeypatch.setattr(adapters, "MEMORY_BUDGET", 300 << 20)
    holds = 'import time\ndef adapt(path):\n    data = b"x" * (700 << 20)\n    time.sleep(20)\n'
    with pytest.raises(AdapterError) as caught:
        run_adapter(holds, a_file, timeout=30)
    assert caught.value.cause == "limit" and "more than 300 MB of memory" in caught.value.reason

    monkeypatch.setattr(adapters, "FILE_SIZE_LIMIT", 1 << 20)
    one_file = 'def adapt(path):\n    open("big.bin", "wb").write(b"x" * (2 << 20))\n    return {"text": "x"}\n'
    with pytest.raises(AdapterError, match="File too large"):
        run_adapter(one_file, a_file)


def test_run_adapter_redacts_secrets_and_cuts_long_errors(a_file, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-canary-0123456789abcdefghij")
    monkeypatch.setenv("MY_SERVICE_TOKEN", "plain-canary-value")
    secrets = "sk-canary-0123456789abcdefghij plain-canary-value ghp_ABCDEFGHIJKLMNOPQRSTUV12"
    code = f'def adapt(path):\n    return {{"text": "{secrets}"}}\n'
    text = run_adapter(code, a_file)["text"]
    assert "canary" not in text and "ghp_" not in text and text.count("[redacted]") == 3
    with pytest.raises(AdapterError) as caught:
        run_adapter('def adapt(path):\n    raise ValueError("the key is plain-canary-value")\n', a_file)
    assert "canary" not in str(caught.value) and "[redacted]" in caught.value.reason

    with pytest.raises(AdapterError) as caught:
        run_adapter('def adapt(path):\n    raise ValueError("x" * 3_000_000)\n', a_file)
    err = caught.value
    assert len(err.reason) <= adapters.REASON_CHARS and "more characters cut" in err.reason
    assert len(err.traceback) <= adapters.TRACEBACK_CHARS + 10
    assert len(json.dumps(err.to_dict())) < 20_000


@pytest.mark.skipif(sys.platform != "darwin" or not os.path.exists(adapters.SANDBOX_EXEC), reason="macOS only")
def test_run_adapter_refuses_to_run_where_the_sandbox_cannot_be_applied(a_file, monkeypatch):
    monkeypatch.setattr(adapters, "sandboxed", lambda: False)
    monkeypatch.delenv(adapters.UNSANDBOXED_ENV, raising=False)
    assert "sandbox" in adapters.sandbox_problem()
    with pytest.raises(AdapterError) as caught:
        run_adapter(FIRST_LINE, a_file)
    assert caught.value.cause == "sandbox" and adapters.UNSANDBOXED_ENV in caught.value.reason
    monkeypatch.setenv(adapters.UNSANDBOXED_ENV, "1")
    assert adapters.sandbox_problem() is None
    assert run_adapter(FIRST_LINE, a_file)["text"] == "hello world"


def test_sandboxed_tries_again_after_a_failed_check(monkeypatch):
    answers = iter([False, True])
    monkeypatch.setattr(adapters, "_sandbox_works", False)
    monkeypatch.setattr(adapters, "_probe_sandbox", lambda: next(answers))
    assert adapters.sandboxed() is False
    assert adapters.sandboxed() is True
    assert adapters.sandboxed() is True  # a working sandbox is remembered: no third check


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

    elsewhere = Path("/private/tmp") / f"sense-adapter-escape-{uuid.uuid4().hex}"
    try:
        with pytest.raises(AdapterError, match="Operation not permitted"):
            run_adapter(f'def adapt(path):\n    open({str(elsewhere)!r}, "w").write("x")\n', a_file)
        assert not elsewhere.exists()
    finally:
        if elsewhere.exists():
            elsewhere.unlink()


@pytest.mark.skipif(not SANDBOXED, reason="needs macOS sandbox-exec")
def test_sandbox_hides_the_probe_store_the_examples_and_the_users_files(tmp_path, a_file):
    root = _probe_tree(tmp_path / "colours", ["red", "blue"], 5)
    probes = probe_folder(root, question="Which colour?", modality="colour file")
    targets = {
        "probe store": ("listdir", probes.files[0].parent),
        "a probe copy": ("open", probes.files[0]),
        "examples folder": ("listdir", root),
        "an example": ("open", next((root / "red").iterdir())),
        "the repository": ("open", Path(adapters.__file__).resolve()),
        "home": ("listdir", Path.home()),
        "codex credentials": ("open", Path.home() / ".codex" / "auth.json"),
        "shell profile": ("open", Path.home() / ".zshrc"),
    }
    code = f'''
import json, os
TARGETS = {json.dumps({k: [how, str(p)] for k, (how, p) in targets.items()})}
def adapt(path):
    seen = {{}}
    for name, (how, target) in TARGETS.items():
        try:
            os.listdir(target) if how == "listdir" else open(target, "rb").close()
            seen[name] = "read"
        except PermissionError:
            seen[name] = "denied"
        except FileNotFoundError:
            seen[name] = "missing"
    return {{"text": json.dumps(seen)}}
'''
    seen = json.loads(run_adapter(code, a_file)["text"])
    assert "read" not in seen.values(), seen
    for name in ["probe store", "a probe copy", "examples folder", "an example", "the repository", "home"]:
        assert seen[name] == "denied", seen


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
    assert probes.meta["skipped"] == {"hidden": 1, "empty": 1, "subfolders": 0, "other_types": 0}
    assert probes.meta["warnings"] == []
    assert probes.provenance == "user" and probes.source == "test"
    # every copy has the same times, so neither times nor names line up with labels
    assert {p.stat().st_mtime for p in probes.files} == {adapters._PROBE_TIME}

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


def test_probe_folder_keeps_one_file_type_and_warns_when_extensions_differ(tmp_path):
    shapes = tmp_path / "shapes"
    for label in ["chair", "table"]:
        (shapes / label).mkdir(parents=True)
        for k in range(5):
            (shapes / label / f"m{k}.obj").write_text(f"mtllib m{k}.mtl\nv {k} 0 0\n# {label}\n")
            (shapes / label / f"m{k}.mtl").write_text(f"newmtl {label}_{k}\n")
    probes = probe_folder(shapes, question="Which piece of furniture?", modality="3d-model", suffixes=[".OBJ"])
    assert len(probes.items) == 10 and {p.suffix for p in probes.files} == {".obj"}
    assert probes.meta["skipped"]["other_types"] == 10 and probes.meta["suffixes"] == [".obj"]
    assert probes.meta["warnings"] == []

    clips = tmp_path / "clips"
    for label, ext in [("walk", ".mov"), ("run", ".mp4")]:
        (clips / label).mkdir(parents=True)
        for k in range(5):
            (clips / label / f"c{k}{ext}").write_bytes(f"{label} clip {k}".encode())
    warned = probe_folder(clips, question="What motion?", modality="video")
    assert len(warned.meta["warnings"]) == 1 and ".mov" in warned.meta["warnings"][0]


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


@pytest.mark.parametrize("kind", ["text", "image"])
def test_the_registry_loads_an_installed_adapter_sense(a_file, kind):
    from sense import registry
    from sense.bridges import load_bridge
    from sense.contracts import ReportCard

    code = FIRST_LINE if kind == "text" else 'from PIL import Image\ndef adapt(p):\n    return {"images": [Image.new("RGB", (8, 8))]}\n'
    bridge = AdapterBridge(kind=kind, modality="mystery", model_name="fake", code=code)
    card = ReportCard("fake", "mystery", "p", "closed", bridge.name, kind, 20, 0.2, 0.9, {"swapped": 0.2})
    registry.install(bridge, card)
    (loaded,) = registry.get_bridges("fake")  # what chat calls for every attached file
    assert isinstance(loaded, AdapterBridge) and (loaded.name, loaded.kind, loaded.code) == (bridge.name, kind, code)
    assert isinstance(load_bridge(registry.sense_dir("fake", "mystery")), AdapterBridge)
    assert loaded.prepare(a_file).kind == kind


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


def test_adapter_bridge_does_not_cache_a_crash(a_file, monkeypatch):
    outcomes = [AdapterError("killed by SIGKILL", cause="exit"), AdapterError("too slow", cause="timeout"),
                {"text": "fine", "notes": [], "seconds": 0.1}]

    def flaky(code, path, *, timeout=60):
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(adapters, "run_adapter", flaky)
    bridge = AdapterBridge(kind="text", modality="mystery", model_name="fake", code=FIRST_LINE)
    for cause in ["exit", "timeout"]:
        with pytest.raises(AdapterError) as caught:
            bridge.prepare(a_file)
        assert caught.value.cause == cause
    assert bridge.prepare(a_file).text == "fine"  # the crash was not remembered
    assert bridge.prepare(a_file).text == "fine" and outcomes == []  # the success was


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
    assert card.controls["no_input"] <= 0.5  # with no input the model guesses
    assert set(card.controls) == {"swapped", "no_input"}
    assert card.interval[0] <= card.accuracy <= card.interval[1]

    every = verify_files(model, bridge, probes, FitRequest(model="fake", dataset=probes.name),
                         idx=np.arange(len(probes.items)))
    assert every.n == len(probes.items) == 60 and every.accuracy >= 0.9

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


def _fails_on(colours: list[str]) -> str:
    return (
        f"FAIL = {colours!r}\n"
        "def adapt(path):\n"
        "    line = open(path).readline().strip()\n"
        "    if any(c in line for c in FAIL):\n"
        "        raise RuntimeError('cannot read this one')\n"
        "    return {'text': line}\n"
    )


def test_verify_files_raises_only_when_more_than_half_of_the_files_fail(tmp_path):
    root = _probe_tree(tmp_path / "four", ["red", "green", "blue", "yellow"], 5)
    probes = probe_folder(root, question="Which colour?", modality="colour file")
    every = np.arange(len(probes.items))
    request = FitRequest(model="fake", dataset=probes.name)

    def card_for(colours):
        bridge = AdapterBridge(kind="text", modality="colour file", model_name="fake", code=_fails_on(colours))
        return verify_files(_fake_model(), bridge, probes, request, idx=every)

    assert "failed on 5 of 20" in card_for(["red"]).reason  # a quarter fails: a card
    assert "failed on 10 of 20" in card_for(["red", "green"]).reason  # half fails: still a card
    with pytest.raises(AdapterError) as caught:
        card_for(["red", "green", "blue"])  # three quarters fail: no card
    assert caught.value.cause == "verify" and "15 of 20" in caught.value.reason


def _skewed(root: Path, counts: dict[str, int], *, header=lambda label: "") -> Path:
    for label, n in counts.items():
        (root / label).mkdir(parents=True)
        for k in range(n):
            (root / label / f"n{k:02d}.txt").write_text(f"{header(label)}{label} note {k} of {n}\n")
    return root


def test_verify_files_swapped_control_catches_a_constant_label_on_a_skewed_set(tmp_path):
    root = _skewed(tmp_path / "skewed", {"red": 40, "green": 4, "blue": 4, "yellow": 4})
    probes = probe_folder(root, question="Which colour is the note about?", modality="colour note")
    says_red = 'def adapt(path):\n    return {"text": "This note is about the colour red."}\n'
    bridge = AdapterBridge(kind="text", modality="colour note", model_name="fake", code=says_red)
    request = FitRequest(model="fake", dataset=probes.name)
    card = verify_files(_fake_model(), bridge, probes, request, idx=np.arange(len(probes.items)))
    assert card.n == 52 and card.accuracy >= card.chance + 0.15  # the majority label alone clears the chance bar
    assert card.controls["swapped"] == card.accuracy  # but any other file's input does as well
    assert card.controls["no_input"] <= 0.5

    from agent.planner import RulePlanner

    assert not RulePlanner().assess(request, card)[0]


def test_verify_files_never_swaps_in_a_file_the_adapter_failed_on(tmp_path):
    # A constant answer plus a crash on every file of the minority label: with failed files as swap partners
    # (asked with no input), the swapped control would fall to a guess and the constant answer would pass.
    root = _skewed(tmp_path / "crashy", {"red": 40, "blue": 6}, header=lambda label: "V2 " if label == "blue" else "")
    probes = probe_folder(root, question="Which colour is the note about?", modality="colour note")
    code = '''
def adapt(path):
    if open(path).read().startswith("V2"):
        raise ValueError("version 2 header")
    return {"text": "This note is about the colour red."}
'''
    bridge = AdapterBridge(kind="text", modality="colour note", model_name="fake", code=code)
    request = FitRequest(model="fake", dataset=probes.name)
    card = verify_files(_fake_model(), bridge, probes, request, idx=np.arange(len(probes.items)))
    assert card.n == 46 and "failed on 6 of 46" in card.reason
    assert card.controls["swapped"] == card.accuracy == round(40 / 46, 4)

    from agent.planner import RulePlanner

    assert not RulePlanner().assess(request, card)[0]


def test_verify_files_respects_max_items_and_the_model(tmp_path):
    probes = _colour_probes(tmp_path)
    bridge = AdapterBridge(kind="text", modality="colour file", model_name="fake", code=FIRST_LINE)
    card = verify_files(_fake_model(), bridge, probes, FitRequest(model="fake", dataset=probes.name), max_items=7)
    assert card.n == 7
    with pytest.raises(ValueError, match="built for fake"):
        verify_files(_fake_model("fake-api"), bridge, probes, FitRequest(model="fake-api", dataset=probes.name))


# ----------------------------------------------------------------------------- native and unseen categories
KNOWS_FOUR = '''
KNOWN = ["red", "green", "blue", "yellow"]

def adapt(path):
    line = open(path, encoding="utf-8").readline()
    for colour in KNOWN:  # an adapter built around the tested colours only
        if colour in line:
            return {"text": f"The file is about the colour {colour}."}
    return {"text": "The file is about some colour."}
'''


def _four_and_unseen(tmp_path, unseen_labels=("Red", "purple", "orange", "pink"), per_label=5):
    main = probe_folder(_probe_tree(tmp_path / "main", ["red", "green", "blue", "yellow"], 6),
                        question="Which colour?", modality="colour file")
    unseen = probe_folder(_probe_tree(tmp_path / "later", list(unseen_labels), per_label), name="later-colours",
                          question="Which colour is it?", modality="colour file", source="a second folder of examples")
    return main, unseen


def test_verify_files_native_control_is_the_model_reading_the_file_as_it_is(tmp_path):
    probes = _colour_probes(tmp_path)
    bridge = AdapterBridge(kind="text", modality="colour file", model_name="fake", code=FIRST_LINE)
    request = FitRequest(model="fake", dataset=probes.name)
    card = verify_files(_fake_model(), bridge, probes, request, native=True)
    assert list(card.controls) == ["swapped", "no_input", "native"]
    assert card.controls["native"] >= 0.9  # the raw file names its colour: the model reads it without the adapter
    assert card.control_chance == {} and card.reason == ""
    plain = verify_files(_fake_model(), bridge, probes, request)
    assert plain.accuracy == card.accuracy and plain.controls == {k: card.controls[k] for k in ("swapped", "no_input")}

    long_file = tmp_path / "long.txt"
    long_file.write_text("colour: red\n" + "x" * 9000)
    text = adapters._native_text(long_file)
    assert len(text) <= adapters.MAX_TEXT_CHARS and text.startswith("colour: red") and text.endswith("goes on]")


def test_verify_files_skips_native_for_files_that_are_not_text(tmp_path):
    root = tmp_path / "binary"
    for label in ["red", "blue"]:
        (root / label).mkdir(parents=True)
        for k in range(6):
            (root / label / f"{k}.bin").write_bytes(f"colour: {label}\n".encode() + bytes([0, 255, k]) * 20)
    probes = probe_folder(root, question="Which colour?", modality="colour file")
    bridge = AdapterBridge(kind="text", modality="colour file", model_name="fake", code=FIRST_LINE)
    card = verify_files(_fake_model(), bridge, probes, FitRequest(model="fake", dataset=probes.name), native=True)
    assert "native" not in card.controls
    assert f"Native not measured: {card.n} of the {card.n} scored files are not text" in card.reason


ECHO = '''
def adapt(path):
    with open(path, "rb") as f:
        return {"text": f.read().decode("utf-8", "replace")}
'''


def test_one_odd_file_does_not_switch_the_native_control_off(tmp_path):
    # An adapter that only echoes the file adds nothing to the model reading it as it is. One Latin-1 byte, or one
    # binary file, among many text notes must not drop the native control and let the echo pass the text bars.
    from agent.planner import RulePlanner

    root = _probe_tree(tmp_path / "notes", ["red", "green", "blue", "yellow"], 8)
    latin1 = root / "red" / "red_00.thing"
    latin1.write_bytes(latin1.read_bytes() + b" caf\xe9\n")
    binary = root / "green" / "green_00.thing"
    binary.write_bytes(b"colour: green\n" + bytes([0, 255, 7]) * 40)
    probes = probe_folder(root, question="Which colour?", modality="colour file")
    bridge = AdapterBridge(kind="text", modality="colour file", model_name="fake", code=ECHO)
    request = FitRequest(model="fake", dataset=probes.name)
    every = np.arange(len(probes.items))
    card = verify_files(_fake_model(), bridge, probes, request, idx=every, native=True)

    assert adapters._native_text(latin1).endswith("caf\ufffd\n")  # a few bad bytes still read as text
    assert adapters._native_text(binary) is None
    assert card.controls["native"] >= 0.9  # measured: the binary file alone was asked with no input
    assert "Native: 1 of the 32 scored files are not text" in card.reason and "with no input" in card.reason
    ok, sentence = RulePlanner().assess(request, card)
    assert not ok and "the model already reads this file as it is" in sentence


def test_verify_files_unseen_categories_control(tmp_path):
    main, unseen = _four_and_unseen(tmp_path)
    request = FitRequest(model="fake", dataset=main.name)
    every = np.arange(len(main.items))

    reads = AdapterBridge(kind="text", modality="colour file", model_name="fake", code=FIRST_LINE)
    card = verify_files(_fake_model(), reads, main, request, idx=every, unseen=unseen)
    assert card.controls["unseen_categories"] >= 0.9 and card.control_chance == {"unseen_categories": 0.3333}
    assert card.chance == 0.25  # the card's own chance is the main set's
    assert ("Unseen categories: 100.0% on 15 files of 3 categories the adapter was not built around (orange, "
            "pink, purple; source: a second folder of examples) (dropped as already probed: Red), against 33.3% "
            "chance.") in card.reason

    narrow = AdapterBridge(kind="text", modality="colour file", model_name="fake", code=KNOWS_FOUR)
    card = verify_files(_fake_model(), narrow, main, request, idx=every, unseen=unseen, unseen_max_items=9)
    assert card.accuracy >= 0.9 and card.controls["unseen_categories"] <= 0.67
    assert "on 9 files of 3 categories" in card.reason

    fails = AdapterBridge(kind="text", modality="colour file", model_name="fake", code=_fails_on(["purple"]))
    card = verify_files(_fake_model(), fails, main, request, idx=every, unseen=unseen)
    assert card.controls["unseen_categories"] <= 0.67  # the purple files count as wrong
    assert "the adapter failed on 5 of them, counted as wrong" in card.reason


def test_verify_files_says_why_unseen_categories_were_not_measured(tmp_path):
    main, overlapping = _four_and_unseen(tmp_path, unseen_labels=("RED", "Green", "purple"))
    bridge = AdapterBridge(kind="text", modality="colour file", model_name="fake", code=FIRST_LINE)
    request = FitRequest(model="fake", dataset=main.name)
    card = verify_files(_fake_model(), bridge, main, request, unseen=overlapping)
    assert "unseen_categories" not in card.controls and card.control_chance == {}
    assert ("Unseen categories not measured: later-colours has 1 label the probe set does not already have "
            "(dropped as already probed: Green, RED); at least 2 are needed.") in card.reason

    tiny = ProbeSet(name="tiny", title="t", modality="colour file", question="Which colour?",
                    labels=["purple", "pink"], items=[(overlapping.files[0], 0),
                    (overlapping.files[1], 1), (overlapping.files[2], 1)],
                    fit_idx=np.array([], dtype=np.int64), dev_idx=np.array([], dtype=np.int64),
                    test_idx=np.arange(3))
    card = verify_files(_fake_model(), bridge, main, request, unseen=tiny)
    assert "unseen_categories" not in card.controls and "has 3 files" in card.reason and "at least 4" in card.reason
