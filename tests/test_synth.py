"""agent.synth: the agent designs, writes, repairs and verifies an adapter.

Fake mode, offline: agent.llm.ask_json is replaced by canned answers (a design, adapter code, generator code), so
no LLM is asked anything. The adapters really run, in sense.adapters' sandbox, and are verified with FakeModel."""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import numpy as np
import pytest

from agent import llm, synth
from sense import adapters, config, registry
from sense.adapters import AdapterError, load_adapter_bridge
from sense.contracts import FitRequest, ReportCard

COLOURS = ["red", "green", "blue", "yellow"]
SIZES = ["small", "medium", "large"]

READS_CONTENT = '''
def adapt(path):
    with open(path, encoding="utf-8") as f:
        return {"text": "The file says: " + f.read().strip()}
'''
CONSTANT = '''
def adapt(path):
    return {"text": "a short text file"}
'''
READS_NAME = '''
import os

def adapt(path):
    return {"text": "The file is called " + os.path.basename(path)}
'''
RAISES = '''
def adapt(path):
    raise ValueError("first problem: cannot parse the header")
'''
SIZE_ADAPTER = '''
def adapt(path):
    with open(path, encoding="utf-8") as f:
        value = float(f.read().split()[0])
    size = "small" if value < 100 else "medium" if value < 1000 else "large"
    return {"text": f"The file holds the number {value:g}, which is {size}."}
'''
NUMBER_GENERATOR = '''
import os
import random

RANGES = {"small": (1, 99), "medium": (100, 999), "large": (1000, 9999)}


def generate(out_dir, labels, n):
    rng = random.Random(7)
    for label in labels:
        folder = os.path.join(out_dir, label)
        os.makedirs(folder, exist_ok=True)
        lo, hi = RANGES[label]
        for k in range(n):
            with open(os.path.join(folder, f"sample_{k:03d}.txt"), "w") as f:
                f.write(f"{rng.uniform(lo, hi):.4f}\\n")
'''
LEAKY_GENERATOR = NUMBER_GENERATOR.replace('f.write(f"{rng', 'f.write(f"{label} {rng')


@pytest.fixture(autouse=True)
def _fake_home(tmp_path, monkeypatch):
    monkeypatch.setenv("SENSE_FAKE", "1")
    monkeypatch.setenv("SENSE_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(adapters, "_cache", type(adapters._cache)())
    if not adapters.sandboxed():  # inside another sandbox (an agent's shell, say): run the adapters without one
        monkeypatch.setenv(adapters.UNSANDBOXED_ENV, "1")


class FakeLLM:
    """Canned answers for agent.llm.ask_json, chosen by schema; records every call."""

    def __init__(self, design=None, codes=(), generators=()):
        self.design = design if isinstance(design, list) else [design]
        self.codes = list(codes)
        self.generators = list(generators)
        self.calls: list[tuple[dict, dict, str]] = []

    def __call__(self, instruction, data, schema, *, max_tokens=400):
        json.dumps(data, allow_nan=False)  # what the real ask_json sends must be JSON
        self.calls.append((schema, copy.deepcopy(data), instruction))
        if schema is synth.DESIGN_SCHEMA:
            answer = self.design[0] if len(self.design) == 1 else self.design.pop(0)
            return copy.deepcopy(answer)
        if schema is synth.CODE_SCHEMA:
            return {"code": self.codes.pop(0)} if self.codes else None
        if schema is synth.GENERATOR_SCHEMA:
            return {"code": self.generators.pop(0)} if self.generators else None
        raise AssertionError(f"unexpected schema {schema}")

    def of(self, schema) -> list[dict]:
        return [data for s, data, _ in self.calls if s is schema]


def colour_design(**changes) -> dict:
    answer = {
        "modality": "colour note",
        "format": "txt",
        "route": "text",
        "description": "Reads the note and states what it says.",
        "libraries": [],
        "question": "Which colour does the note describe?",
        "label_hint": "colour names",
        "labels": COLOURS,
        "probe_plan": "user",
        "probe_reason": "Only the user knows which notes they mean.",
        "install": [],
    }
    answer.update(changes)
    return answer


def size_design(**changes) -> dict:
    answer = colour_design(
        modality="number file", description="Reads the number and says how large it is.",
        question="How large is the number in the file?", label_hint="sizes", labels=SIZES, probe_plan="generate",
        probe_reason="Files holding numbers in known ranges can be written by a script.",
    )
    answer.update(changes)
    return answer


def colour_tree(root: Path, per_label: int = 10, *, name_after_label: bool = False) -> Path:
    rng = np.random.default_rng(0)
    for colour in COLOURS:
        folder = root / colour
        folder.mkdir(parents=True)
        for k in range(per_label):
            name = f"{colour}_{k:02d}.txt" if name_after_label else f"note_{k:02d}.txt"
            (folder / name).write_text(f"This note is about the colour {colour}. Serial {rng.integers(10**9)}\n")
    return root


@pytest.fixture
def note(tmp_path) -> Path:
    path = tmp_path / "mystery.txt"
    path.write_text("This note is about the colour blue. Serial 1\n")
    return path


def design_of(answer: dict, models=None) -> synth.Design:
    found, problems = synth.validate_design(answer, ["numpy", "PIL", "json"], models or [{"name": "fake", "native": ["text"]}])
    assert found is not None, problems
    return found


def run(model, file, goal, **kwargs):
    events = []
    cards, installed = synth.run_synth(model, file, goal, events.append, **kwargs)
    return cards, installed, events


def steps(events) -> list[str]:
    return list(dict.fromkeys(e.step for e in events))


def last(events, step, status=None):
    return next(e for e in reversed(events) if e.step == step and (status is None or e.status == status))


# ----------------------------------------------------------------------------- schemas and heads
def test_schemas_are_strict():
    for schema in (synth.DESIGN_SCHEMA, synth.CODE_SCHEMA, synth.GENERATOR_SCHEMA):
        assert schema["type"] == "object" and schema["additionalProperties"] is False
        assert sorted(schema["required"]) == sorted(schema["properties"])
    assert synth.DESIGN_SCHEMA["properties"]["route"]["enum"] == ["text", "images"]
    assert synth.DESIGN_SCHEMA["properties"]["probe_plan"]["enum"] == ["user", "dataset", "generate"]
    assert synth.CODE_SCHEMA["required"] == ["code"] and synth.GENERATOR_SCHEMA["required"] == ["code"]


def test_file_head_shows_text_or_a_hex_dump(tmp_path):
    text = tmp_path / "a.obj"
    text.write_text("v 0 0 0\n" * 1000)
    head = synth.file_head(text)
    assert head.startswith("v 0 0 0\n") and len(head.encode()) < synth.HEAD_TEXT_BYTES + 100
    assert "more bytes" in head

    binary = tmp_path / "b.bin"
    binary.write_bytes(bytes(range(256)) * 20)
    dump = synth.file_head(binary)
    assert dump.startswith(f"[hex dump of the first {synth.HEAD_HEX_BYTES} of 5120 bytes]")
    assert "00000000  00 01 02 03" in dump and len(dump.splitlines()) == 1 + synth.HEAD_HEX_BYTES // 16


# ----------------------------------------------------------------------------- design
def test_validate_design_accepts_and_cleans_a_good_answer():
    answer = colour_design(
        modality="3D Model", format=".OBJ", libraries=["numpy", "pillow", "os", "xml.etree.ElementTree"],
        labels=["sci_fi", "Sci-Fi", "cube", "", 3], probe_plan="generate",
    )
    found, problems = synth.validate_design(answer, ["numpy", "PIL", "xml.etree", "ffmpeg (via subprocess)"],
                                            [{"name": "m", "native": ["text"]}])
    assert problems == []
    assert found.modality == "3d-model" and found.format == "obj" and found.kind == "text"
    assert found.libraries == ["numpy", "PIL", "xml.etree"]  # "os" is standard library: no listing needed
    assert found.labels == ["sci fi", "cube"]  # folder-safe, distinct
    assert json.loads(json.dumps(found.to_dict())) == found.to_dict()

    ffmpeg, _ = synth.validate_design(colour_design(libraries=["ffmpeg"]), ["ffmpeg (via subprocess)"], [])
    assert ffmpeg.libraries == ["ffmpeg (via subprocess)"]


def test_validate_design_cleans_the_install_requests():
    box, models_ = ["numpy", "PIL", "trimesh"], [{"name": "m", "native": ["text"]}]
    wants = [{"name": "open3d", "why": "  point   clouds "}, {"name": "pillow", "why": "in the toolbox"},
             {"name": "json", "why": "standard library"}, {"name": "OPEN3D", "why": "twice"},
             {"name": "pyvista", "why": "b"}, {"name": "meshio", "why": "c"}, {"name": "pymeshlab", "why": "d"}]
    found, problems = synth.validate_design(colour_design(install=wants), box, models_)
    assert problems == [] and found.install == [
        {"name": "open3d", "why": "point clouds"}, {"name": "pyvista", "why": "b"}, {"name": "meshio", "why": "c"}]
    assert synth.validate_design(colour_design(install=None), box, models_)[0].install == []  # an older answer
    old = colour_design()
    del old["install"]
    assert synth.validate_design(old, box, models_)[0].install == []
    for bad, problem in [("open3d", "must be a list"), ([{"why": "x"}], "needs a package name"),
                         ([{"name": "requests", "why": "x"}], "network")]:
        found, problems = synth.validate_design(colour_design(install=bad), box, models_)
        assert found is None and any(problem in p for p in problems), problems
    assert "install" in synth.DESIGN_SCHEMA["required"] and "never name them in libraries" in synth.DESIGN_INSTRUCTION


@pytest.mark.parametrize(
    "changes, models, problem",
    [
        ({"route": "audio"}, [{"name": "m", "native": ["text", "image"]}], "route is 'audio'"),
        ({"libraries": ["numpy", "tensorflow"]}, [{"name": "m", "native": ["text"]}], "'tensorflow' is not in the toolbox"),
        ({"libraries": ["requests"]}, [{"name": "m", "native": ["text"]}], "network"),
        ({"route": "images"}, [{"name": "qwen", "native": ["text"]}], "no base model here reads images"),
        ({"probe_plan": "generate", "labels": ["only one"]}, [{"name": "m", "native": ["text"]}], "1 usable label"),
        ({"probe_plan": "somewhere"}, [{"name": "m", "native": ["text"]}], "probe_plan is 'somewhere'"),
        ({"question": "  "}, [{"name": "m", "native": ["text"]}], "question is missing"),
        ({"modality": "!!"}, [{"name": "m", "native": ["text"]}], "modality is missing"),
    ],
)
def test_design_refuses_unusable_answers(monkeypatch, changes, models, problem):
    fake = FakeLLM(design=colour_design(**changes))
    monkeypatch.setattr(llm, "ask_json", fake)
    assert synth.design({"modality": "text", "format": "text"}, "goal", "head", ["numpy", "json"], models) is None
    asks = fake.of(synth.DESIGN_SCHEMA)
    assert len(asks) == 2  # one repair round, told what was wrong
    assert "problems_with_last_answer" not in asks[0]
    assert any(problem in p for p in asks[1]["problems_with_last_answer"]), asks[1]["problems_with_last_answer"]


def test_design_images_route_needs_a_model_that_sees_images(monkeypatch):
    monkeypatch.setattr(llm, "ask_json", FakeLLM(design=colour_design(route="images", libraries=["PIL"])))
    sees = synth.design({}, "goal", "head", ["PIL"], [{"name": "openai-chat", "native": ["text", "image"]}])
    assert sees is not None and sees.route == "images" and sees.kind == "image"


def test_design_repairs_after_problems_and_sends_the_file_facts(monkeypatch):
    fake = FakeLLM(design=[colour_design(route="audio"), colour_design()])
    monkeypatch.setattr(llm, "ask_json", fake)
    found = synth.design(
        {"modality": "text", "format": "text", "how": "content", "detail": "d"}, "know the colour", "x" * 9000,
        ["numpy"], [{"name": "fake", "native": ["text"], "label": "ignored"}], name="Note.TXT", size=12,
    )
    assert found is not None and found.route == "text"
    first = fake.of(synth.DESIGN_SCHEMA)[0]
    assert first["goal"] == "know the colour" and first["file"] == {"suffix": ".txt", "size_bytes": 12}
    assert len(first["head"]) <= 6000 and first["images_allowed"] is False
    assert first["models"] == [{"name": "fake", "native": ["text"]}]


def test_design_is_none_in_fake_mode_without_an_llm():
    assert synth.design({}, "goal", "head", ["numpy"], [{"name": "fake", "native": ["text"]}]) is None


# ----------------------------------------------------------------------------- write_adapter
def test_write_adapter_repairs_after_a_failing_first_code(monkeypatch, note):
    fake = FakeLLM(codes=[RAISES, READS_CONTENT])
    monkeypatch.setattr(llm, "ask_json", fake)
    seen = []
    code, output, used = synth.write_adapter(design_of(colour_design()), note, on_attempt=lambda k, e: seen.append((k, e)))
    assert code.strip() == READS_CONTENT.strip() and used == 2
    assert output["text"] == "The file says: This note is about the colour blue. Serial 1"
    assert [k for k, _ in seen] == [1, 2] and isinstance(seen[0][1], AdapterError) and seen[1][1] is None

    first, second = fake.of(synth.CODE_SCHEMA)
    assert first["earlier_attempts"] == [] and "last_code" not in first
    assert "first problem" in second["earlier_attempts"][0]["reason"]
    assert second["last_code"].strip() == RAISES.strip()
    assert second["sample"]["head"].startswith("This note is about")


def test_write_adapter_prompt_holds_the_contract_and_the_rules(monkeypatch, note):
    fake = FakeLLM(codes=[READS_CONTENT])
    monkeypatch.setattr(llm, "ask_json", fake)
    synth.write_adapter(design_of(colour_design()), note)
    (schema, data, instruction), = fake.calls
    assert "adapt(path)" in instruction and "no network" in instruction
    assert data["toolbox"] == adapters.toolbox() and data["design"]["question"].startswith("Which colour")
    # the adapter's author never sees the candidate labels or where the probes will come from
    assert not {"labels", "probe_plan", "probe_reason"} & set(data["design"])
    prompt = json.dumps(data)
    assert not [c for c in COLOURS if c != "blue" and re.search(rf"\b{c}\b", prompt)]  # blue: the sample says it
    assert '{"text": str}' in data["contract"]
    rules = " ".join(data["rules"])
    assert "file name" in rules and "Never guess or state a label" in rules and "No network" in rules
    assert "not a verdict" in rules and "say what they imply" not in rules


def test_write_adapter_cuts_long_errors_before_they_reach_the_llm(monkeypatch, note):
    huge = 'def adapt(path):\n    raise ValueError("bad header: " + "x" * 3_000_000)\n'
    fake = FakeLLM(codes=[huge, huge, huge])
    monkeypatch.setattr(llm, "ask_json", fake)
    with pytest.raises(AdapterError) as caught:
        synth.write_adapter(design_of(colour_design()), note, attempts=3)
    sizes = [len(json.dumps(data)) for data in fake.of(synth.CODE_SCHEMA)]
    assert len(sizes) == 3 and max(sizes) < 40_000, sizes
    assert len(json.dumps(caught.value.to_dict())) < 20_000
    assert all(len(a["reason"]) <= 1000 for a in caught.value.attempts)


def test_write_adapter_gives_up_with_every_attempts_error(monkeypatch, note):
    codes = [RAISES, "def adapt(path):\n    return {'text': (\n", "import socket\n\ndef adapt(path):\n    return {'text': 'x'}\n"]
    monkeypatch.setattr(llm, "ask_json", FakeLLM(codes=codes))
    with pytest.raises(AdapterError) as caught:
        synth.write_adapter(design_of(colour_design()), note, attempts=3)
    err = caught.value
    assert "No working adapter after 3 attempts" in err.reason
    assert "first problem" in str(err) and "does not parse" in str(err) and "socket" in str(err)
    assert [a["attempt"] for a in err.attempts] == [1, 2, 3]
    assert [a["cause"] for a in err.attempts] == ["exception", "code", "code"]


def test_write_adapter_checks_the_route_and_the_answer(monkeypatch, note):
    images = design_of(colour_design(route="images"), models=[{"name": "m", "native": ["text", "image"]}])
    monkeypatch.setattr(llm, "ask_json", FakeLLM(codes=[READS_CONTENT, "def adapt(path):\n    return {'text': '  '}\n"]))
    with pytest.raises(AdapterError) as caught:
        synth.write_adapter(images, note, attempts=1)
    assert "route is images" in caught.value.reason
    with pytest.raises(AdapterError, match="empty text"):
        synth.write_adapter(design_of(colour_design()), note, attempts=1)

    monkeypatch.setattr(llm, "ask_json", FakeLLM())  # no code comes back at all
    with pytest.raises(AdapterError) as caught:
        synth.write_adapter(design_of(colour_design()), note, attempts=2)
    assert caught.value.cause == "llm" and "gave no adapter code" in caught.value.reason


def test_code_checks_before_running():
    box = adapters.toolbox()
    fine = (
        "from __future__ import annotations\nimport os, pathlib, subprocess, struct\nimport numpy as np\n"
        "from PIL import Image\nimport xml.etree.ElementTree as ET\n\ndef adapt(path):\n    return {'text': 'x'}\n"
    )
    assert synth._static_problem(fine, box, "adapt") is None
    missing = synth._static_problem("import no_such_module_here\ndef adapt(p):\n    pass\n", box, "adapt")
    assert "no_such_module_here, which is neither in the standard library nor in the toolbox" in missing
    assert "network" in synth._static_problem("import urllib.request\ndef adapt(p):\n    pass\n", box, "adapt")
    assert "foreign calls" in synth._static_problem("import ctypes\ndef adapt(p):\n    pass\n", box, "adapt")
    assert "no top-level function adapt" in synth._static_problem("def adapt2(p):\n    pass\n", box, "adapt")
    assert "no top-level function generate" in synth._static_problem(READS_CONTENT, box, "generate")
    assert synth._code_from({"code": "```py\ndef adapt(p):\n    return 1\n```"}) == "def adapt(p):\n    return 1\n"
    assert synth._code_from({"code": "  "}) is None and synth._code_from(None) is None


def test_code_is_written_with_a_longer_llm_timeout_then_restored(monkeypatch, note):
    seen = []

    def ask(instruction, data, schema, *, max_tokens=400):
        seen.append((llm.TIMEOUT_S, max_tokens))
        return {"code": READS_CONTENT}

    before = llm.TIMEOUT_S
    monkeypatch.setattr(llm, "ask_json", ask)
    synth.write_adapter(design_of(colour_design()), note)
    assert seen == [(max(before, synth.LLM_TIMEOUT_S), synth.CODE_TOKENS)]
    assert llm.TIMEOUT_S == before


def test_write_adapter_takes_code_in_markdown_fences(monkeypatch, note):
    monkeypatch.setattr(llm, "ask_json", FakeLLM(codes=["```python\n" + READS_CONTENT.strip() + "\n```"]))
    code, output, used = synth.write_adapter(design_of(colour_design()), note)
    assert not code.startswith("```") and used == 1 and output["text"].startswith("The file says")


# ----------------------------------------------------------------------------- generated probes
def test_label_leaks_finds_labels_written_into_their_own_files(tmp_path):
    groups = {}
    for label in ["sphere", "cube"]:
        files = []
        for k in range(4):
            p = tmp_path / f"{label}{k}.obj"
            p.write_text(f"o {label}\nv {k} 0 0\n" if label == "sphere" else f"v {k} 1 1\n")
            files.append(p)
        groups[label] = files
    leaks = synth.label_leaks(groups)
    assert len(leaks) == 1 and leaks[0].startswith("'sphere' is written in 4 of its 4 files")
    assert synth.label_leaks({"a b": groups["cube"], "red": groups["cube"]}) == []


@pytest.mark.parametrize(
    "how",
    ["plural", "numbered", "camel", "gzipped", "zip archive", "zip member name", "two words"],
)
def test_label_leaks_sees_ordinary_ways_of_writing_a_label(tmp_path, how):
    import gzip
    import zipfile

    def content(label: str, k: int) -> bytes:
        name = {"plural": f"o {label}s", "numbered": f"o {label.title()}00{k}", "camel": f"o {label.title()}Mesh",
                "two words": f"o {label.title().replace(' ', '')}"}.get(how, f"o {label}")
        return f"{name}\nv {k} 0 0\n".encode()

    groups = {}
    for label in (["sci fi", "cube"] if how == "two words" else ["sphere", "cube"]):
        files = []
        for k in range(4):
            p = tmp_path / f"{label.replace(' ', '')}{k}.bin"
            if how == "gzipped":
                p.write_bytes(gzip.compress(content(label, k)))
            elif how.startswith("zip"):
                with zipfile.ZipFile(p, "w", zipfile.ZIP_DEFLATED) as z:
                    if how == "zip archive":
                        z.writestr("3D/3dmodel.model", f'<object name="{label}"><mesh/></object> {k}')
                    else:
                        z.writestr(f"{label.title()}.model", f"<mesh/> {k}")
            else:
                p.write_bytes(content(label, k))
            files.append(p)
        groups[label] = files
    leaks = synth.label_leaks(groups)
    assert len(leaks) == 2, leaks  # both labels are written into their own files


def test_generate_probes_repairs_a_leaky_generator(monkeypatch):
    fake = FakeLLM(generators=[LEAKY_GENERATOR, NUMBER_GENERATOR])
    monkeypatch.setattr(llm, "ask_json", fake)
    seen = []
    probes, code, used = synth.generate_probes(design_of(size_design()), n_per_label=8, on_attempt=lambda k, e: seen.append((k, e)))
    assert used == 2 and code.strip() == NUMBER_GENERATOR.strip()
    assert "name their own label" in seen[0][1].reason and seen[1][1] is None
    assert probes.provenance == "generated" and sorted(probes.labels) == sorted(SIZES)
    assert len(probes.items) == 24 and probes.meta["generator_attempts"] == 2
    assert all(re.fullmatch(r"[0-9a-f]{16}\.txt", p.name) for p in probes.files)
    assert all(p.is_file() and Path(config.home()) in p.parents for p in probes.files)
    second = fake.of(synth.GENERATOR_SCHEMA)[1]
    assert "name their own label" in second["earlier_attempts"][0]["reason"]
    assert second["labels"] == SIZES and second["n"] == 8

    with pytest.raises(ValueError, match="not \"generate\""):
        synth.write_generator(design_of(colour_design()), COLOURS, 8)


def test_generate_probes_shows_the_llm_its_rejected_code_and_short_counts(monkeypatch):
    networked = "import requests\n\ndef generate(out_dir, labels, n):\n    pass\n"
    short = NUMBER_GENERATOR.replace("for k in range(n):", "for k in range(n if label != 'large' else 2):")
    fake = FakeLLM(generators=[networked, short, NUMBER_GENERATOR])
    monkeypatch.setattr(llm, "ask_json", fake)
    probes, code, used = synth.generate_probes(design_of(size_design()), n_per_label=8, suffix=".txt")
    assert used == 3 and code.strip() == NUMBER_GENERATOR.strip()
    second, third = fake.of(synth.GENERATOR_SCHEMA)[1:]
    assert "network module" in second["earlier_attempts"][0]["reason"]
    assert second["last_code"].strip() == networked.strip()  # the rejected code goes back for repair
    assert "wrote too few .txt files ('large': 2)" in third["earlier_attempts"][1]["reason"]
    assert third["extension"] == ".txt"
    assert "consistency check" in probes.source and "same LLM" in probes.source


def test_run_generator_writes_in_the_sandbox(tmp_path):
    out = synth.scratch_dir() / "out"
    report = synth.run_generator(NUMBER_GENERATOR, out, SIZES, 5)
    assert report["per_label"] == {"small": 5, "medium": 5, "large": 5}
    with pytest.raises(AdapterError, match="boom"):
        synth.run_generator("def generate(out_dir, labels, n):\n    raise RuntimeError('boom')\n", out, SIZES, 5)
    if adapters.sandboxed():  # it may write into out_dir, and nowhere else outside its own folder
        beside = out.parent / "beside.txt"
        code = f"def generate(out_dir, labels, n):\n    open({str(beside)!r}, 'w').write('x')\n"
        with pytest.raises(AdapterError, match="Operation not permitted"):
            synth.run_generator(code, out, SIZES, 5)
        assert not beside.exists()


# ----------------------------------------------------------------------------- assess
def test_assess_holds_an_image_card_to_the_text_bars():
    request = FitRequest(model="openai-chat", dataset="p", modality="video")
    card = ReportCard("openai-chat", "video", "p", "closed", "adapter:video-mp4", "image", 20, 0.25, 0.6,
                      {"swapped": 0.2, "no_input": 0.25})
    ok, sentence = synth.assess(request, card)
    assert ok and "image bridge" in sentence
    card.accuracy = 0.35
    assert not synth.assess(request, card)[0]
    card.accuracy, card.controls["swapped"] = 0.6, 0.55
    assert not synth.assess(request, card)[0]
    assert card.kind == "image"  # the caller's card is untouched


@pytest.mark.parametrize("kind", ["text", "image"])
def test_assess_applies_the_native_and_unseen_categories_rules(kind):
    request = FitRequest(model="openai-chat", dataset="p", modality="molecule")
    card = ReportCard("openai-chat", "molecule", "p", "closed", "adapter:m", kind, 40, 0.25, 0.9,
                      {"swapped": 0.2, "no_input": 0.25, "native": 0.85})
    ok, sentence = synth.assess(request, card)  # it must not bypass the planner's native rule
    assert not ok and "the model already reads this file as it is: native 85.0%, adapter 90.0%" in sentence
    card.controls.update(native=0.5, unseen_categories=0.3)
    card.control_chance = {"unseen_categories": 0.5}
    ok, sentence = synth.assess(request, card)
    assert not ok and "works for the tested categories but not for new ones" in sentence
    card.controls["unseen_categories"] = 0.8
    assert synth.assess(request, card)[0]


def test_assess_wants_enough_files_and_an_interval_clear_of_chance():
    from sense.verify import _wilson

    request = FitRequest(model="fake", dataset="p", modality="note")

    def card(n, accuracy, chance=0.5, swapped=0.0):
        return ReportCard("fake", "note", "p", "closed", "adapter:note-txt", "text", n, chance, accuracy,
                          {"swapped": swapped, "no_input": chance}, interval=_wilson(accuracy, n))

    # 2 of 2 right on two labels: the point bars pass, but two files tell nothing
    ok, sentence = synth.assess(request, card(2, 1.0))
    assert not ok and "only 2 probe files" in sentence
    # 9 of 12 right on two labels: 75% clears the 65% bar, but the interval reaches down to chance
    ok, sentence = synth.assess(request, card(12, 0.75))
    assert not ok and "reaches down to the 50.0% chance level" in sentence
    assert synth.assess(request, card(40, 0.75))[0]  # the same score on 40 files is clear of chance
    assert synth.assess(request, card(10, 1.0))[0]
    no_interval = card(40, 0.75)
    no_interval.interval = None
    assert synth.assess(request, no_interval)[0]  # computed when the card has none


# ----------------------------------------------------------------------------- run_synth
def test_run_synth_installs_a_working_adapter_from_user_examples(monkeypatch, tmp_path, note):
    examples = colour_tree(tmp_path / "notes")
    for colour in COLOURS:  # a sidecar file of another type beside the examples is not an example
        (examples / colour / "README.md").write_text(f"notes about {colour}\n")
    monkeypatch.setattr(llm, "ask_json", FakeLLM(design=colour_design(), codes=[READS_CONTENT]))
    # These notes name their colour in plain words, so the model reads them as they are and the native control
    # would refuse the adapter (test_run_synth_refuses_an_adapter_the_model_does_not_need); this test is about
    # the rest of the run, so it leaves that control out.
    cards, installed, events = run("fake", note, "know which colour a note is about", examples=examples, native=False)

    assert installed == "adapter:colour-note-txt"
    assert steps(events) == ["design", "build", "data", "verify", "decide", "install", "done"]
    assert [(e.step, e.status) for e in events] == [
        ("design", "start"), ("design", "ok"), ("build", "start"), ("build", "ok"), ("data", "start"),
        ("data", "ok"), ("verify", "start"), ("verify", "ok"), ("decide", "start"), ("decide", "ok"),
        ("install", "start"), ("install", "ok"), ("done", "ok"),
    ]
    json.dumps([e.to_dict() for e in events], allow_nan=False)
    assert [e.t for e in events] == sorted(e.t for e in events)

    design_ok = last(events, "design", "ok").data
    assert design_ok["modality"] == "colour-note" and design_ok["route"] == "text"
    assert design_ok["detection"]["format"] == "text"
    build = last(events, "build", "ok").data
    assert build["attempts"] == 1 and len(build["code_sha256"]) == 64 and build["output_kind"] == "text"
    assert build["preview"] == "The file says: This note is about the colour blue. Serial 1"
    data = last(events, "data", "ok").data
    assert data["provenance"] == "user" and data["licence"] == "user supplied" and data["source"] == str(examples)
    assert data["labels"] == sorted(COLOURS) and data["counts"] == dict.fromkeys(sorted(COLOURS), 10)
    assert data["meta"]["skipped"]["other_types"] == 4 and data["meta"]["suffixes"] == [".txt"]
    assert re.fullmatch(r"colour-note-examples-[0-9a-f]{10}", data["name"])  # nothing points back at the folder
    # these notes say their colour in words, so the probes name their labels: a warning that travels with the card
    assert len(data["label_leaks"]) == 4 and "Warning" in last(events, "data", "ok").message
    card = last(events, "verify", "ok").data["card"]
    assert card["dataset"] == data["name"] and card["kind"] == "text" and card["accuracy"] >= 0.9
    assert card["n"] == 40  # every probe file is scored: the adapter was written before any existed
    decide = last(events, "decide", "ok").data
    assert decide["accepted"] is True and decide["install"] == installed
    done = last(events, "done").data
    assert done["installed"] == installed and done["needs_user"] is False

    (only,) = cards
    assert only.verdict == "installed" and "passed" in only.reason
    assert "Warning about the probe files" in only.reason and "'red' is written" in only.reason
    (row,) = registry.installed("fake")
    assert row["bridge"] == installed and row["modality"] == "colour-note" and row["card"]["verdict"] == "installed"
    assert "Warning about the probe files" in row["card"]["reason"]
    directory = registry.sense_dir("fake", "colour-note")
    assert (directory / "adapter.py").read_text().strip() == READS_CONTENT.strip()
    bridge = load_adapter_bridge(directory)
    manifest = bridge.manifest
    assert manifest["route"] == "text" and manifest["attempts"] == 1 and manifest["probes"]["provenance"] == "user"
    assert len(manifest["probes"]["label_leaks"]) == 4
    assert manifest["question"] == "Which colour does the note describe?" and manifest["written_by"]
    assert bridge.prepare(note).text.endswith("colour blue. Serial 1")
    assert [b.name for b in registry.get_bridges("fake")] == [installed]  # chat can load it


def test_run_synth_installs_nothing_the_registry_cannot_load(monkeypatch, tmp_path, note):
    # Installing a sense the registry cannot load back would make registry.get_bridges fail for the whole model,
    # and with it every chat with an attached file; so a loader that breaks blocks the install.
    def broken(directory):
        raise ValueError("unknown bridge kind")

    monkeypatch.setattr(registry, "load_bridge", broken)
    examples = colour_tree(tmp_path / "notes")
    monkeypatch.setattr(llm, "ask_json", FakeLLM(design=colour_design(), codes=[READS_CONTENT]))
    cards, installed, events = run("fake", note, "know which colour a note is about", examples=examples, native=False)
    assert installed is None and cards[0].verdict == "candidate" and "Not installed" in cards[0].reason
    assert last(events, "decide").data["accepted"] is True
    blocked = last(events, "install")
    assert blocked.status == "info" and "cannot load the adapter bridge back" in blocked.data["blocked"]
    assert registry.installed("fake") == [] and registry.get_bridges("fake") == []
    assert last(events, "done").data["installed"] is None
    json.dumps([e.to_dict() for e in events], allow_nan=False)


def test_run_synth_refuses_an_adapter_the_model_does_not_need(monkeypatch, tmp_path, note):
    # The notes say their colour in words: the model given a note as it is answers as well as with the adapter.
    examples = colour_tree(tmp_path / "notes")
    monkeypatch.setattr(llm, "ask_json", FakeLLM(design=colour_design(), codes=[READS_CONTENT]))
    cards, installed, events = run("fake", note, "know which colour a note is about", examples=examples)
    (card,) = cards
    assert installed is None and card.verdict == "rejected"
    assert "the model already reads this file as it is: native 100.0%, adapter 100.0%" in card.reason
    verify = last(events, "verify", "ok").data
    assert verify["controls"]["native"] == 1.0 and verify["control_chance"] == {} and "the raw file 100.0%" in \
        last(events, "verify", "ok").message
    assert last(events, "verify", "start").data["native"] is True
    assert "10 points above the model reading the raw file" in last(events, "decide", "start").message


KNOWS_FOUR = '''
KNOWN = ["red", "green", "blue", "yellow"]

def adapt(path):
    text = open(path, encoding="utf-8").read()
    for colour in KNOWN:  # built around the tested colours only
        if colour in text:
            return {"text": f"The note is about the colour {colour}."}
    return {"text": "The note is about some colour."}
'''


def test_run_synth_scores_categories_the_adapter_was_not_built_around(monkeypatch, tmp_path, note):
    examples = colour_tree(tmp_path / "notes")
    later = tmp_path / "later"
    for colour in ["purple", "orange", "pink", "Red"]:  # Red is already probed: dropped
        (later / colour).mkdir(parents=True)
        for k in range(4):
            (later / colour / f"n{k}.txt").write_text(f"This note is about the colour {colour.lower()}. {k}\n")

    monkeypatch.setattr(llm, "ask_json", FakeLLM(design=colour_design(), codes=[KNOWS_FOUR]))
    cards, installed, events = run("fake", note, "colour", examples=examples, native=False, unseen=later)
    (card,) = cards
    assert installed is None and card.verdict == "rejected"
    assert card.accuracy >= 0.9 and card.control_chance == {"unseen_categories": 0.3333}
    assert "works for the tested categories but not for new ones" in card.reason
    assert "Unseen categories:" in card.reason and f"source: {later}" in card.reason
    told = [e for e in events if e.step == "data" and e.status == "info"]
    assert len(told) == 1 and told[0].data["unseen"]["labels"] == ["Red", "orange", "pink", "purple"]
    verify = last(events, "verify", "ok").data
    assert verify["controls"]["unseen_categories"] == card.controls["unseen_categories"]
    assert verify["control_chance"] == {"unseen_categories": 0.3333}

    monkeypatch.setattr(llm, "ask_json", FakeLLM(design=colour_design(), codes=[READS_CONTENT]))
    cards, installed, events = run("fake", note, "colour", examples=examples, native=False, unseen=later)
    assert installed == "adapter:colour-note-txt" and cards[0].controls["unseen_categories"] >= 0.9
    manifest = load_adapter_bridge(registry.sense_dir("fake", "colour-note")).manifest
    assert manifest["unseen"]["labels"] == ["Red", "orange", "pink", "purple"]

    lonely = tmp_path / "lonely"
    (lonely / "purple").mkdir(parents=True)
    (lonely / "purple" / "a.txt").write_text("purple\n")
    monkeypatch.setattr(llm, "ask_json", FakeLLM(design=colour_design(), codes=[READS_CONTENT]))
    cards, _, events = run("fake", note, "colour", examples=examples, native=False, unseen=lonely)
    assert "unseen_categories" not in cards[0].controls and "Unseen categories not measured:" in cards[0].reason


def test_the_adapter_writer_never_sees_labels_probe_plans_or_probe_data(monkeypatch, tmp_path):
    examples = colour_tree(tmp_path / "notes")
    later = tmp_path / "later"
    for colour in ["purple", "orange"]:
        (later / colour).mkdir(parents=True)
        for k in range(4):
            (later / colour / f"{colour}_{k}.txt").write_text(f"This note is about the colour {colour}. {k}\n")
    plain = tmp_path / "plain.txt"
    plain.write_text("A note with serial number 7, in plain words.\n")  # it names no colour
    fake = FakeLLM(design=colour_design(install=[]), codes=[READS_CONTENT])
    monkeypatch.setattr(llm, "ask_json", fake)
    run("fake", plain, "know which colour a note is about", examples=examples, unseen=later)
    (request,) = fake.of(synth.CODE_SCHEMA)
    sent = json.dumps(request)
    assert set(request["design"]) == set(synth._ADAPTER_DESIGN_FIELDS)
    for word in [*COLOURS, "purple", "orange", "probe_plan", "probe_reason", "labels", "install", "probes",
                 str(examples), str(later), str(config.home())]:
        assert not re.search(rf"\b{re.escape(word)}\b", sent), word
    copies = sorted((config.home() / "probes").rglob("*.txt"))
    assert copies and not any(p.stem in sent for p in copies)  # no probe file's name either


def _plan(verdict="ok"):
    from agent import installer

    return installer.InstallPlan(requested=[{"name": "meshkit", "version": None, "why": "reads meshes"}], packages=[],
                                 verdict=verdict, requirements="", created="2026-11-15T00:00:00+00:00",
                                 problems=["no wheel for this platform"] if verdict == "refuse" else [], target={})


USES_MESHKIT = '''
import meshkit

def adapt(path):
    with open(path, encoding="utf-8") as f:
        return {"text": f"meshkit {meshkit.VALUE} says: " + f.read().strip()}
'''


@pytest.mark.parametrize("answer", [True, False])
def test_run_synth_asks_to_install_what_the_design_requests(monkeypatch, tmp_path, note, answer):
    from agent import installer, jobs

    asked, installs = [], []
    monkeypatch.setattr(installer, "plan", lambda requests: _plan())
    monkeypatch.setattr(installer, "ask_llm_review", lambda plan: "A small, well-known mesh library.")
    monkeypatch.setattr(jobs, "request_approval", lambda kind, payload, timeout=600: asked.append((kind, payload)) or answer)

    def install(plan, *, approved):
        assert approved is True
        site_dir = installer.toolbox_dir()
        (site_dir / "meshkit").mkdir(parents=True)
        (site_dir / "meshkit" / "__init__.py").write_text("VALUE = 42\n")
        info = site_dir / "meshkit-1.0.dist-info"
        info.mkdir()
        (info / "METADATA").write_text("Name: meshkit\nVersion: 1.0\n\n")
        (info / "top_level.txt").write_text("meshkit\n")
        (site_dir.parent / "installed.json").write_text(json.dumps([{"name": "meshkit", "version": "1.0"}]))
        installs.append(plan)
        return {"installed": [{"name": "meshkit", "version": "1.0", "sha256": "0" * 64}], "target": str(site_dir),
                "seconds": 0.1, "log_tail": ""}

    monkeypatch.setattr(installer, "install", install)
    wants = [{"name": "meshkit", "why": "reads meshes"}, {"name": "numpy", "why": "already there"}]
    codes = [USES_MESHKIT] if answer else [READS_CONTENT]
    fake = FakeLLM(design=colour_design(install=wants), codes=codes)
    monkeypatch.setattr(llm, "ask_json", fake)
    examples = colour_tree(tmp_path / "notes")
    cards, installed, events = run("fake", note, "colour", examples=examples, native=False)

    assert steps(events)[:3] == ["design", "install_request", "build"]
    assert last(events, "design", "ok").data["install"] == [{"name": "meshkit", "why": "reads meshes"}]
    ((kind, payload),) = asked
    assert kind == "install" and payload["summary"].startswith("OK to install meshkit")
    assert payload["review"] == "A small, well-known mesh library." and payload["plan"]["verdict"] == "ok"
    outcome = last(events, "install_request")
    request = fake.of(synth.CODE_SCHEMA)[0]
    if answer:
        assert outcome.status == "ok" and len(installs) == 1 and "Installed meshkit 1.0" in outcome.message
        assert "meshkit" in request["toolbox"]  # the adapter may import it, and it runs on the adapter's path
        assert last(events, "build", "ok").data["preview"].startswith("meshkit 42 says:")
    else:
        assert outcome.status == "info" and installs == [] and "declined" in outcome.message
        assert "meshkit" not in request["toolbox"]
    assert installed == "adapter:colour-note-txt"
    record = load_adapter_bridge(registry.sense_dir("fake", "colour-note")).manifest["install"]
    assert record["requested"] == [{"name": "meshkit", "why": "reads meshes"}] and record["verdict"] == "ok"
    assert record["approved"] is answer and [p["name"] for p in record["installed"]] == (["meshkit"] if answer else [])
    json.dumps([e.to_dict() for e in events], allow_nan=False)


def test_run_synth_does_not_ask_about_a_refused_install(monkeypatch, note, tmp_path):
    from agent import jobs

    asked = []
    monkeypatch.setattr(jobs, "request_approval", lambda *a, **k: asked.append(a) or True)
    # fake mode: agent.installer reaches neither PyPI nor uv, so its plan is refused
    fake = FakeLLM(design=colour_design(install=[{"name": "meshkit", "why": "reads meshes"}]), codes=[READS_CONTENT])
    monkeypatch.setattr(llm, "ask_json", fake)
    cards, _, events = run("fake", note, "colour", examples=colour_tree(tmp_path / "notes"), native=False)
    assert asked == []
    refused = last(events, "install_request")
    assert refused.status == "fail" and refused.data["verdict"] == "refuse" and refused.data["approved"] is None
    assert "Writing the adapter without it" in refused.message
    assert steps(events)[:3] == ["design", "install_request", "build"] and cards


def test_run_synth_refuses_an_adapter_that_ignores_the_file(monkeypatch, tmp_path, note):
    examples = colour_tree(tmp_path / "notes")
    monkeypatch.setattr(llm, "ask_json", FakeLLM(design=colour_design(), codes=[CONSTANT]))
    cards, installed, events = run("fake-api", note, "know which colour a note is about", examples=examples)
    assert installed is None
    assert steps(events) == ["design", "build", "data", "verify", "decide", "install", "done"]
    (card,) = cards
    assert card.verdict == "rejected" and "failed" in card.reason
    assert card.accuracy == card.controls["swapped"]  # the same text for every file: input makes no difference
    assert last(events, "decide").data["accepted"] is False
    assert last(events, "install").status == "info"
    assert last(events, "done").data == {"installed": None, "needs_user": False, "bridge": "adapter:colour-note-txt"}
    assert registry.installed("fake-api") == []
    json.dumps([e.to_dict() for e in events], allow_nan=False)


def test_run_synth_refuses_a_constant_label_on_a_skewed_folder(monkeypatch, tmp_path, note):
    # Saying "red" for every file scores 40 of 52 on a folder that is mostly red, well over the chance bar; only
    # the swapped-input control, which scores the same, shows that the file made no difference.
    examples = tmp_path / "skewed"
    for colour, n in {"red": 40, "green": 4, "blue": 4, "yellow": 4}.items():
        (examples / colour).mkdir(parents=True)
        for k in range(n):
            (examples / colour / f"note_{k:02d}.txt").write_text(f"Serial {colour} {k}\n")
    says_red = 'def adapt(path):\n    return {"text": "This note is about the colour red."}\n'
    monkeypatch.setattr(llm, "ask_json", FakeLLM(design=colour_design(), codes=[says_red]))
    monkeypatch.setattr(synth, "MIN_PER_LABEL", 4)
    cards, installed, events = run("fake", note, "know which colour a note is about", examples=examples)
    (card,) = cards
    assert card.n == 52 and card.accuracy >= card.chance + 0.15
    assert card.controls["swapped"] == card.accuracy and card.controls["no_input"] <= 0.5
    assert installed is None and card.verdict == "rejected" and "swapped input" in card.reason


def test_labels_never_leak_through_file_names(monkeypatch, tmp_path, note):
    examples = colour_tree(tmp_path / "notes", name_after_label=True)
    monkeypatch.setattr(llm, "ask_json", FakeLLM(design=colour_design(), codes=[READS_NAME]))
    cards, installed, events = run("fake", note, "know which colour a note is about", examples=examples)
    assert installed is None and cards[0].verdict == "rejected"
    copies = sorted((config.home() / "probes").rglob("*.txt"))
    assert len(copies) == 40
    assert all(re.fullmatch(r"[0-9a-f]{16}\.txt", p.name) for p in copies)
    assert not any(colour in str(p.relative_to(config.home())) for p in copies for colour in COLOURS)


def test_run_synth_verifies_on_generated_probe_files(monkeypatch, tmp_path):
    monkeypatch.setattr(synth, "GENERATED_PER_LABEL", 10)
    number = tmp_path / "reading.txt"
    number.write_text("512.25\n")
    fake = FakeLLM(design=size_design(), codes=[SIZE_ADAPTER], generators=[LEAKY_GENERATOR, NUMBER_GENERATOR])
    monkeypatch.setattr(llm, "ask_json", fake)
    cards, installed, events = run("fake", number, "know how large the number is")

    assert installed == "adapter:number-file-txt"
    assert steps(events) == ["design", "build", "data", "verify", "decide", "install", "done"]
    assert last(events, "build", "ok").data["preview"] == "The file holds the number 512.25, which is medium."
    retry = [e for e in events if e.step == "data" and e.status == "info"]
    assert len(retry) == 1 and "name their own label" in retry[0].message
    data = last(events, "data", "ok").data
    assert data["provenance"] == "generated" and data["labels"] == sorted(SIZES)
    assert data["counts"] == dict.fromkeys(sorted(SIZES), 10) and data["label_leaks"] == []
    assert data["generator_code"].strip() == NUMBER_GENERATOR.strip() and len(data["generator_sha256"]) == 64
    assert cards[0].verdict == "installed" and cards[0].accuracy >= 0.9 and cards[0].n == 30
    # what generated labels are worth is said in the trace, on the card and in the manifest
    assert "consistency check" in data["source"] and "consistency check" in data["caveat"]
    assert "consistency check" in cards[0].reason
    directory = registry.sense_dir("fake", "number-file")
    assert (directory / "generator.py").read_text().strip() == NUMBER_GENERATOR.strip()
    probes_manifest = load_adapter_bridge(directory).manifest["probes"]
    assert probes_manifest["provenance"] == "generated" and "consistency check" in probes_manifest["caveat"]
    # the adapter was written before any probe file existed, from the user's file alone
    (adapter_ask,) = fake.of(synth.CODE_SCHEMA)
    assert "512.25" in adapter_ask["sample"]["head"] and "generator" not in json.dumps(adapter_ask).lower()
    json.dumps([e.to_dict() for e in events], allow_nan=False)


def test_run_synth_asks_the_user_for_examples(monkeypatch, note):
    monkeypatch.setattr(llm, "ask_json", FakeLLM(design=colour_design(probe_plan="user"), codes=[READS_CONTENT]))
    cards, installed, events = run("fake", note, "know which colour a note is about")
    assert (cards, installed) == ([], None)
    assert steps(events) == ["design", "build", "data", "done"]
    done = last(events, "done")
    assert done.data["needs_user"] is True and done.data["installed"] is None
    assert done.data["wanted"]["labels"] == COLOURS and done.data["wanted"]["min_per_label"] == synth.MIN_PER_LABEL
    assert "one sub-folder per label" in done.message and "red, green, blue, yellow" in done.message
    json.dumps([e.to_dict() for e in events], allow_nan=False)

    monkeypatch.setattr(llm, "ask_json", FakeLLM(design=colour_design(probe_plan="dataset"), codes=[READS_CONTENT]))
    _, _, events = run("fake", note, "know which colour a note is about")
    assert last(events, "done").data["needs_user"] is True and "not wired" in last(events, "data").message


def test_run_synth_reports_unusable_examples(monkeypatch, tmp_path, note):
    lonely = tmp_path / "lonely"
    (lonely / "red").mkdir(parents=True)
    (lonely / "red" / "a.txt").write_text("red\n")
    monkeypatch.setattr(llm, "ask_json", FakeLLM(design=colour_design(), codes=[READS_CONTENT]))
    _, installed, events = run("fake", note, "colour", examples=lonely)
    assert installed is None and last(events, "data").status == "fail"
    assert last(events, "done").data["needs_user"] is True


def test_run_synth_stops_cleanly_without_an_llm(note):
    cards, installed, events = run("fake", note, "know the colour")  # fake mode: ask_json answers None
    assert (cards, installed) == ([], None)
    assert [(e.step, e.status) for e in events] == [("design", "start"), ("design", "fail"), ("done", "ok")]


def test_run_synth_writes_no_code_where_the_sandbox_cannot_be_applied(monkeypatch, note):
    fake = FakeLLM(design=colour_design(), codes=[READS_CONTENT])
    monkeypatch.setattr(llm, "ask_json", fake)
    monkeypatch.setattr(adapters, "sandbox_problem", lambda: "The macOS sandbox cannot be applied here.")
    cards, installed, events = run("fake", note, "colour")
    assert (cards, installed) == ([], None) and steps(events) == ["design", "build", "done"]
    assert last(events, "build").status == "fail" and "cannot be applied" in last(events, "done").message
    assert fake.of(synth.CODE_SCHEMA) == []  # no code was asked for, so none ran
    with pytest.raises(AdapterError) as caught:
        synth.write_adapter(design_of(colour_design()), note)
    assert caught.value.cause == "sandbox"


def test_run_synth_warns_when_there_is_no_sandbox(monkeypatch, note):
    monkeypatch.setattr(llm, "ask_json", FakeLLM(design=colour_design(probe_plan="user"), codes=[READS_CONTENT]))
    monkeypatch.setattr(adapters, "sandboxed", lambda: False)
    monkeypatch.setattr(adapters, "sandbox_problem", lambda: None)
    _, _, events = run("fake", note, "colour")
    warnings = [e for e in events if e.step == "build" and e.data.get("warning") == "unsandboxed"]
    assert len(warnings) == 1 and "no sandbox" in warnings[0].message


def test_run_synth_stops_when_no_adapter_works(monkeypatch, note):
    monkeypatch.setattr(llm, "ask_json", FakeLLM(design=colour_design(), codes=[RAISES, RAISES, RAISES]))
    _, installed, events = run("fake", note, "colour")
    assert installed is None and steps(events) == ["design", "build", "done"]
    assert [e.data["attempt"] for e in events if e.step == "build" and e.status == "info"] == [1, 2, 3]
    assert last(events, "build").status == "fail" and "after 3 attempts" in last(events, "build").message


def test_run_synth_raises_with_an_error_event_for_an_unknown_model(note):
    events = []
    with pytest.raises(ValueError, match="Unknown model"):
        synth.run_synth("no-such-model", note, "goal", events.append)
    assert [(e.step, e.status) for e in events] == [("design", "fail"), ("error", "fail")]
    assert events[-1].data == {"step": "design", "error": "ValueError"}
