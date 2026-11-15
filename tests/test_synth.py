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
    assert '{"text": str}' in data["contract"]
    rules = " ".join(data["rules"])
    assert "file name" in rules and "Never guess or state a label" in rules and "No network" in rules


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


def test_run_generator_writes_in_the_sandbox(tmp_path):
    out = synth.scratch_dir() / "out"
    report = synth.run_generator(NUMBER_GENERATOR, out, SIZES, 5)
    assert report["per_label"] == {"small": 5, "medium": 5, "large": 5}
    with pytest.raises(AdapterError, match="boom"):
        synth.run_generator("def generate(out_dir, labels, n):\n    raise RuntimeError('boom')\n", out, SIZES, 5)


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


# ----------------------------------------------------------------------------- run_synth
def test_run_synth_installs_a_working_adapter_from_user_examples(monkeypatch, tmp_path, note):
    examples = colour_tree(tmp_path / "notes")
    monkeypatch.setattr(llm, "ask_json", FakeLLM(design=colour_design(), codes=[READS_CONTENT]))
    cards, installed, events = run("fake", note, "know which colour a note is about", examples=examples)

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
    card = last(events, "verify", "ok").data["card"]
    assert card["dataset"] == data["name"] and card["kind"] == "text" and card["accuracy"] >= 0.9
    decide = last(events, "decide", "ok").data
    assert decide["accepted"] is True and decide["install"] == installed
    done = last(events, "done").data
    assert done["installed"] == installed and done["needs_user"] is False

    (only,) = cards
    assert only.verdict == "installed" and "passed" in only.reason
    (row,) = registry.installed("fake")
    assert row["bridge"] == installed and row["modality"] == "colour-note" and row["card"]["verdict"] == "installed"
    directory = registry.sense_dir("fake", "colour-note")
    assert (directory / "adapter.py").read_text().strip() == READS_CONTENT.strip()
    bridge = load_adapter_bridge(directory)
    manifest = bridge.manifest
    assert manifest["route"] == "text" and manifest["attempts"] == 1 and manifest["probes"]["provenance"] == "user"
    assert manifest["question"] == "Which colour does the note describe?" and manifest["written_by"]
    assert bridge.prepare(note).text.endswith("colour blue. Serial 1")


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
    assert cards[0].verdict == "installed" and cards[0].accuracy >= 0.9
    directory = registry.sense_dir("fake", "number-file")
    assert (directory / "generator.py").read_text().strip() == NUMBER_GENERATOR.strip()
    assert load_adapter_bridge(directory).manifest["probes"]["provenance"] == "generated"
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
