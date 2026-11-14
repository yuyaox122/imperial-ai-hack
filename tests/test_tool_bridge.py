"""The tool bridge in fake mode: the listen tool, building bridges by kind, availability, verification with
the tool-calling fake model and the OpenAI adapter (mocked client), and chat over HTTP.

No weights, no downloads and no network.
Run: SENSE_FAKE=1 .venv/bin/python -m pytest -q tests/test_tool_bridge.py
"""

from __future__ import annotations

import hashlib
import json
import re
import string
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from agent import jobs
from sense import bridges, config, datasets, models, registry, specialists
from sense.contracts import Bridge, ChoiceItem, FitRequest, ModelInfo, ReportCard, SenseInput
from sense.specialists import FAKE_CONCEPTS, FakeSpecialist
from sense.verify import verify
from server import app as server_app


@pytest.fixture(autouse=True)
def fake_home(monkeypatch, tmp_path):
    monkeypatch.setenv("SENSE_FAKE", "1")
    monkeypatch.setenv("SENSE_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("SENSE_SETS_DIR", raising=False)
    monkeypatch.delenv("SENSE_API_VERIFY_ITEMS", raising=False)
    monkeypatch.delenv("SENSE_API_CACHE", raising=False)
    assert config.fake()
    return tmp_path / "home"


@pytest.fixture
def api_model():
    return models.load_model("fake-api")


@pytest.fixture
def gtzan():
    return datasets.load_labelled_set("gtzan")


@pytest.fixture
def esc50():
    return datasets.load_labelled_set("esc50")


@pytest.fixture
def tool_bridge(api_model, gtzan):
    return bridges.make_tool_bridge(api_model, gtzan)


class RecordingSpecialist:
    """The fake specialist, recording every text it is asked to embed."""

    def __init__(self):
        self.inner = FakeSpecialist()
        self.texts: list[list[str]] = []
        self.modality, self.dim, self.label = self.inner.modality, self.inner.dim, self.inner.label

    def embed_texts(self, texts):
        self.texts.append(list(texts))
        return self.inner.embed_texts(texts)

    def embed_files(self, paths):
        return self.inner.embed_files(paths)


@pytest.fixture
def recorder(monkeypatch):
    spec = RecordingSpecialist()
    monkeypatch.setattr(specialists, "get_specialist", lambda modality: spec)
    return spec


def _round_trips(value) -> bool:
    return json.loads(json.dumps(value)) == value


# ----------------------------------------------------------------------------- the listen tool
def test_from_embedding_offers_the_listen_tool(tool_bridge, gtzan):
    assert isinstance(tool_bridge, Bridge)
    assert (tool_bridge.name, tool_bridge.kind, tool_bridge.model_name) == ("tool:gtzan", "tool", "fake-api")
    inp = tool_bridge.from_embedding(gtzan.emb[gtzan.test_idx[0]], source="clip.wav")
    assert inp.kind == "tool" and inp.modality == "audio" and inp.source == "clip.wav"
    assert inp.detail == "the model can call the listen tool to hear this clip"
    assert inp.text is None and inp.vector is None
    tool = inp.tool
    assert tool.name == "listen" and tool.calls == []
    assert "Fake CLAP" in tool.description and "gtzan labels" in tool.description
    assert tool.parameters["properties"]["candidates"]["items"] == {"type": "string"}
    assert tool.parameters["properties"]["candidates"]["maxItems"] == 10
    assert _round_trips(tool.parameters)


@pytest.mark.parametrize("name", ["gtzan", "esc50"])
def test_candidates_name_the_true_label_as_best_match(api_model, name):
    dataset = datasets.load_labelled_set(name)
    bridge = bridges.make_tool_bridge(api_model, dataset)
    rng = np.random.default_rng(0)
    for i in dataset.test_idx[:20]:
        truth = dataset.labels[dataset.y[i]]
        others = [dataset.labels[c] for c in rng.choice(len(dataset.labels), 4, replace=False) if dataset.labels[c] != truth]
        result = bridge.from_embedding(dataset.emb[i]).tool.call({"candidates": others + [truth]})
        assert result.startswith("Similarity of the clip to each candidate (Fake CLAP): ")
        assert result.endswith(f"Best match: {truth}.")
        scores = [float(s) for s in re.findall(r" (-?\d\.\d\d)", result)]
        assert len(scores) == len(others) + 1 and scores == sorted(scores, reverse=True)  # sorted, two decimals
        assert "-0.00" not in result


def test_candidates_are_worded_by_the_question(api_model, gtzan, esc50, recorder):
    """'<c> music' when the dataset's question is about music, else 'the sound of <c>'; always the latter for the
    caption bank. Each wording is embedded once and then remembered."""
    music = bridges.make_tool_bridge(api_model, gtzan)
    assert music.question_style == "music"
    clip = music.from_embedding(gtzan.emb[gtzan.test_idx[0]]).tool
    clip.call({"candidates": ["rock", " jazz "]})
    assert recorder.texts == [["rock music", "jazz music"]]
    clip.call({"candidates": ["jazz", "rock"]})
    assert recorder.texts == [["rock music", "jazz music"]]  # served from the bridge's own cache

    sound = bridges.make_tool_bridge(api_model, esc50)
    assert sound.question_style == "sound"
    sound.from_embedding(esc50.emb[esc50.test_idx[0]]).tool.call({"candidates": ["dog"]})
    captions = bridges.make_tool_bridge(api_model)
    assert captions.question_style == "sound"
    captions.from_embedding(gtzan.emb[gtzan.test_idx[0]]).tool.call({"candidates": ["rock"]})
    assert recorder.texts[1:] == [["the sound of dog"], ["the sound of rock"]]


def test_without_candidates_the_tool_names_the_nearest_phrases(api_model, gtzan, esc50):
    i = gtzan.test_idx[0]
    truth = gtzan.labels[gtzan.y[i]]
    labels = bridges.make_tool_bridge(api_model, gtzan).from_embedding(gtzan.emb[i]).tool
    for arguments in ({}, {"candidates": []}, {"candidates": None}):
        result = labels.call(arguments)
        assert result.startswith(f"Nearest labels (Fake CLAP): {truth} ")
        assert len(re.findall(r"\d\.\d\d", result)) == 3

    captions = bridges.make_tool_bridge(api_model)
    assert captions.name == "tool:captions" and captions.source == "captions"
    j = esc50.test_idx[0]
    result = captions.from_embedding(esc50.emb[j]).tool.call({})
    assert result.startswith("Nearest captions (Fake CLAP): ")
    quoted = re.findall(r'"([^"]+)"', result)
    assert len(quoted) == 3 and all(q in captions.phrases for q in quoted)
    assert esc50.labels[esc50.y[j]] in quoted[0]  # the fake caption bank covers every esc50 sound


@pytest.mark.parametrize("arguments, says", [
    ({"candidates": "rock"}, "must be a list"),
    ({"candidates": {"a": "rock"}}, "must be a list"),
    ({"candidates": ["rock", ""]}, "is empty"),
    ({"candidates": ["rock", "   "]}, "is empty"),
    ({"candidates": ["rock", 3]}, "must be a string"),
    ({"candidates": ["rock", None]}, "must be a string"),
    ({"candidates": [f"genre {k}" for k in range(11)]}, "at most 10"),
    ({"candidates": ["x" * 121]}, "at most 120 characters"),
    (["rock"], "must be a JSON object"),
    ("rock", "must be a JSON object"),
    (None, "must be a JSON object"),
])
def test_bad_arguments_get_an_error_text_instead_of_an_exception(tool_bridge, gtzan, arguments, says):
    tool = tool_bridge.from_embedding(gtzan.emb[gtzan.test_idx[0]]).tool
    result = tool.call(arguments)
    assert result.startswith("listen error: ") and says in result
    assert "Best match" not in result
    assert tool.calls == [{"arguments": arguments, "result": result}]  # bad calls are logged too


def test_ten_candidates_are_accepted(tool_bridge, gtzan):
    tool = tool_bridge.from_embedding(gtzan.emb[gtzan.test_idx[0]]).tool
    assert "Best match: " in tool.call({"candidates": list(gtzan.labels)})  # gtzan has exactly 10 labels


def test_every_call_is_logged_with_a_copy_of_its_arguments(tool_bridge, gtzan):
    tool = tool_bridge.from_embedding(gtzan.emb[gtzan.test_idx[0]]).tool
    candidates = ["rock", "jazz"]
    first = tool.call({"candidates": candidates})
    second = tool.call({})
    candidates.append("metal")  # the caller changing its list afterwards does not rewrite the log
    assert tool.calls == [
        {"arguments": {"candidates": ["rock", "jazz"]}, "result": first},
        {"arguments": {}, "result": second},
    ]
    other = tool_bridge.from_embedding(gtzan.emb[gtzan.test_idx[1]]).tool
    assert other.calls == []  # each clip has its own log


def test_key_identifies_the_clip_and_the_bridge(api_model, tool_bridge, gtzan):
    a, b = gtzan.emb[gtzan.test_idx[0]], gtzan.emb[gtzan.test_idx[1]]
    key = tool_bridge.from_embedding(a).tool.key
    assert re.fullmatch(r"[0-9a-f]{64}", key)
    assert key == tool_bridge.from_embedding(a.astype(np.float64), source="other name.wav").tool.key
    assert key != tool_bridge.from_embedding(b).tool.key
    assert key != bridges.make_tool_bridge(api_model).from_embedding(a).tool.key  # same clip, another bridge
    expected = hashlib.sha256(np.asarray(a, dtype=np.float32).tobytes() + b"tool:gtzan").hexdigest()
    assert key == expected


def test_an_embedding_of_the_wrong_size_is_refused(tool_bridge):
    with pytest.raises(ValueError, match="Build the bridge again"):
        tool_bridge.from_embedding(np.ones(7))


def test_text_embeddings_of_another_size_are_refused(tool_bridge, gtzan, monkeypatch):
    """A specialist whose text embeddings do not match the clips (another specialist, or fake mode against a real
    bridge) is a build problem, not a model mistake: it raises instead of answering the model."""
    class NarrowSpecialist(RecordingSpecialist):
        def embed_texts(self, texts):
            return np.ones((len(texts), 7), dtype=np.float32)

    tool = tool_bridge.from_embedding(gtzan.emb[gtzan.test_idx[0]]).tool
    monkeypatch.setattr(specialists, "get_specialist", lambda modality: NarrowSpecialist())
    with pytest.raises(ValueError, match="text embeddings have 7 values .* Build it again"):
        tool.call({"candidates": ["rock"]})


def test_the_candidate_cache_is_cleared_when_full(tool_bridge, gtzan, recorder, monkeypatch):
    monkeypatch.setattr(bridges, "_TEXT_CACHE_SIZE", 2)
    tool = tool_bridge.from_embedding(gtzan.emb[gtzan.test_idx[0]]).tool
    tool.call({"candidates": ["rock", "jazz"]})
    assert set(tool_bridge._text_cache) == {"rock music", "jazz music"}
    assert "Best match: " in tool.call({"candidates": ["pop"]})  # a third phrase does not fit: start again
    assert set(tool_bridge._text_cache) == {"pop music"}
    tool.call({"candidates": ["rock"]})
    assert recorder.texts == [["rock music", "jazz music"], ["pop music"], ["rock music"]]  # embedded again
    assert len(tool_bridge._text_cache) <= 2


def test_prepare_embeds_a_file(tool_bridge, tmp_path):
    rock = FAKE_CONCEPTS.index("rock")
    path = tmp_path / "song.wav"
    FakeSpecialist.write_sample(rock, path, seed=3)
    inp = tool_bridge.prepare(path)
    assert inp.kind == "tool" and inp.source == "song.wav"
    assert inp.tool.call({"candidates": ["jazz", "rock", "blues"]}).endswith("Best match: rock.")


@pytest.mark.parametrize("closed", [True, False], ids=["labels", "captions"])
def test_save_and_load_round_trip(api_model, gtzan, tmp_path, closed):
    bridge = bridges.make_tool_bridge(api_model, gtzan if closed else None)
    bridge.save(tmp_path / "saved")
    meta = json.loads((tmp_path / "saved" / "bridge.json").read_text())
    assert meta["kind"] == "tool" and meta["name"] == bridge.name and meta["model_name"] == "fake-api"
    assert meta["question_style"] == ("music" if closed else "sound")
    assert (tmp_path / "saved" / "bridge.npz").is_file() == closed  # the caption bank is reloaded, not saved
    again = bridges.load_bridge(tmp_path / "saved")
    assert isinstance(again, bridges.ToolBridge)
    assert (again.name, again.kind, again.source, again.question_style) == (
        bridge.name, "tool", bridge.source, bridge.question_style,
    )
    assert again.phrases == bridge.phrases
    for i in gtzan.test_idx[:5]:
        old, new = bridge.from_embedding(gtzan.emb[i]).tool, again.from_embedding(gtzan.emb[i]).tool
        assert new.key == old.key
        assert new.call({}) == old.call({})
        assert new.call({"candidates": ["rock", "jazz", "pop"]}) == old.call({"candidates": ["rock", "jazz", "pop"]})


@pytest.mark.parametrize("closed", [True, False], ids=["labels", "captions"])
def test_describe_a_tool_bridge(api_model, gtzan, closed):
    bridge = bridges.make_tool_bridge(api_model, gtzan if closed else None)
    facts = bridges.describe(bridge)
    assert _round_trips(facts)
    assert facts["name"] == bridge.name and facts["kind"] == "tool" and facts["model"] == "fake-api"
    assert facts["source"] == ("gtzan" if closed else "captions")
    assert facts["n_phrases"] == len(bridge.phrases) > 0
    assert facts["specialist"]["name"] == "fake-clap"
    for absent in ("fit_seconds", "fit_seconds_by_layer", "layer", "dev_accuracy", "n_fit"):
        assert facts[absent] is None


# ----------------------------------------------------------------------------- build
def test_build_each_kind(api_model, gtzan):
    fake = models.load_model("fake")
    built = {
        ("vector", "closed"): bridges.build("vector", fake, gtzan, "closed"),
        ("text", "closed"): bridges.build("text", fake, gtzan, "closed"),
        ("text", "open"): bridges.build("text", fake, gtzan, "open"),
        ("tool", "closed"): bridges.build("tool", api_model, gtzan, "closed"),
        ("tool", "open"): bridges.build("tool", api_model, gtzan, "open"),
    }
    assert {key: (b.name, b.kind) for key, b in built.items()} == {
        ("vector", "closed"): ("vector:gtzan", "vector"),
        ("text", "closed"): ("text:gtzan", "text"),
        ("text", "open"): ("text:captions", "text"),
        ("tool", "closed"): ("tool:gtzan", "tool"),
        ("tool", "open"): ("tool:captions", "tool"),
    }
    assert built[("vector", "closed")].n_fit == len(gtzan.fit_idx)
    assert built[("tool", "closed")].model_name == "fake-api"


def test_build_refuses_a_vector_bridge_for_a_model_that_is_not_open_weight(api_model, gtzan):
    """fake-api stands in for an API model: FakeModel could read vector inputs, but build() must refuse, as it
    would for openai-chat, so a test with fake-api catches a run that tries the vector bridge on an API model."""
    with pytest.raises(ValueError, match="Fake API model .* is not an open-weight model.*Use the text bridge"):
        bridges.build("vector", api_model, gtzan, "closed")


@pytest.mark.parametrize("kind", ["image", "native", "audio"])
def test_build_refuses_kinds_it_cannot_build(api_model, gtzan, kind):
    with pytest.raises(ValueError, match=f"Cannot build a '{kind}' bridge: the kinds that can be built are vector, text, tool"):
        bridges.build(kind, api_model, gtzan, "closed")


# ----------------------------------------------------------------------------- availability
def _tool_entry(info: ModelInfo, modality: str = "audio") -> dict:
    entries = {e["kind"]: e for e in bridges.availability(info, modality)}
    assert list(entries) == ["native", "text", "tool", "image", "vector"]
    assert _round_trips(list(entries.values()))
    return entries["tool"]


def test_tool_is_available_for_a_model_that_can_call_tools(api_model):
    info = ModelInfo(name="api", label="API model", native=["text", "image"], open_weights=False, tools=True)
    assert _tool_entry(info) == {
        "kind": "tool", "available": True, "why": "API model can call Fake CLAP (tones, fake mode) as a tool",
    }
    entry = _tool_entry(api_model.info)
    assert entry["available"] is True and entry["why"].startswith("Fake API model (no weights, tools) can call")
    vector = next(e for e in bridges.availability(api_model.info, "audio") if e["kind"] == "vector")
    assert vector["available"] is False


@pytest.mark.parametrize("info", [
    models.FakeModel(models._fake_info()).info,
    ModelInfo(name="qwen3.5-0.8b", label="Qwen3.5 0.8B (local)", native=["text", "image"], open_weights=True),
    ModelInfo(name="api", label="API model", native=["text", "image"], open_weights=False),
], ids=["fake", "qwen-like", "api-without-tools"])
def test_tool_is_unavailable_for_a_model_that_cannot_call_tools(info):
    assert _tool_entry(info) == {
        "kind": "tool", "available": False,
        "why": f"{info.label} cannot call tools; tool calls are wired for API models only",
    }


def test_local_qwen_models_cannot_call_tools():
    for name in ("qwen3.5-0.8b", "qwen3.5-2b"):
        assert models._local_info(name).tools is False  # reads the settings and the disk only; loads nothing


def test_tool_is_unavailable_without_a_specialist():
    info = ModelInfo(name="api", label="API model", native=["text"], open_weights=False, tools=True)
    entry = _tool_entry(info, "video")
    assert entry["available"] is False and "no video specialist" in entry["why"]


# ----------------------------------------------------------------------------- the fake models
def test_fake_api_model_answers_through_the_tool(api_model, tool_bridge, gtzan):
    i = gtzan.test_idx[0]
    truth = gtzan.labels[gtzan.y[i]]
    options = [truth] + [label for label in gtzan.labels if label != truth][:4]
    inp = tool_bridge.from_embedding(gtzan.emb[i])
    for order in (options, options[::-1]):
        [pick] = api_model.choose([ChoiceItem(gtzan.question, order, [inp])])
        assert order[pick] == truth
    assert [c["arguments"] for c in inp.tool.calls] == [{"candidates": options}, {"candidates": options[::-1]}]

    reply = api_model.chat("What is this?", [inp])
    assert reply.startswith(f"The audio clip sounds like {truth}. I called listen, which returned: Nearest labels")
    assert inp.tool.calls[-1]["arguments"] == {}


def test_fake_open_weight_model_still_answers_tool_inputs(tool_bridge, gtzan):
    """FakeModel is a test double: it takes a tool input whatever its ModelInfo says."""
    fake = models.load_model("fake")
    assert fake.info.tools is False
    i = gtzan.test_idx[0]
    truth = gtzan.labels[gtzan.y[i]]
    options = [label for label in gtzan.labels if label != truth][:4] + [truth]
    assert fake.choose([ChoiceItem(gtzan.question, options, [tool_bridge.from_embedding(gtzan.emb[i])])]) == [4]


def test_qwen_refuses_tool_inputs_with_a_clear_message(tool_bridge, gtzan):
    """No weights are loaded: the prompt renderer refuses the tool input before it touches the tokenizer."""
    qwen = object.__new__(models.QwenModel)
    qwen.info = models._local_info("qwen3.5-0.8b")
    inp = tool_bridge.from_embedding(gtzan.emb[gtzan.test_idx[0]], source="song.wav")
    with pytest.raises(ValueError, match="cannot call tools: tool calling is wired for API models only") as err:
        qwen._render([inp], "What is this?")
    assert "song.wav" in str(err.value) and "text or vector bridge" in str(err.value)


# ----------------------------------------------------------------------------- verification
def test_verify_a_tool_bridge_with_the_fake_api_model(api_model, tool_bridge, gtzan):
    card = verify(api_model, tool_bridge, gtzan, FitRequest(model="fake-api", dataset="gtzan"))
    assert (card.kind, card.bridge, card.model, card.verdict) == ("tool", "tool:gtzan", "fake-api", "candidate")
    assert card.n == min(config.api_verify_items(), len(gtzan.test_idx)) == 50  # not open-weight: the API cap
    assert card.accuracy >= 0.8
    assert card.controls["swapped"] <= 0.4 and card.controls["no_input"] <= 0.4
    assert "unseen_classes" not in card.controls
    assert _round_trips(card.to_dict())


def test_verify_caps_a_tool_bridge_like_any_api_bridge(api_model, esc50, monkeypatch):
    monkeypatch.setenv("SENSE_API_VERIFY_ITEMS", "12")
    captions = bridges.make_tool_bridge(api_model)
    made: list[SenseInput] = []
    from_embedding = captions.from_embedding

    def recording(emb, source=""):
        made.append(from_embedding(emb, source))
        return made[-1]

    monkeypatch.setattr(captions, "from_embedding", recording)
    card = verify(api_model, captions, esc50, FitRequest(model="fake-api", dataset="esc50", scope="open"))
    assert card.n == 12 and card.kind == "tool"
    assert card.accuracy >= 0.8 and card.controls["swapped"] <= 0.4 and card.controls["no_input"] <= 0.4
    assert "unseen_classes" not in card.controls  # that control is for vector bridges only
    assert len(made) == 24  # the bridge and swapped-input conditions; no input asks no tool
    assert all(len(inp.tool.calls) == 1 and len(inp.tool.calls[0]["arguments"]["candidates"]) == 5 for inp in made)


def test_verify_a_tool_bridge_through_the_openai_adapter(esc50, monkeypatch):
    """OpenAIChatModel with a mocked client that calls listen with the options, then answers with the letter of
    the best match. A repeat run is answered from the disk cache and calls no tool."""
    import sense.models_api as models_api

    monkeypatch.setenv("SENSE_API_VERIFY_ITEMS", "12")
    monkeypatch.delenv("SENSE_API_MODEL", raising=False)
    monkeypatch.setattr(config, "openai_key", lambda: "test-key")

    def answer(**kwargs):
        messages = kwargs["messages"]
        question = messages[1]["content"].split("\n\n")[0]
        options = [line.split(". ", 1)[1] for line in question.splitlines()[1:]]
        usage = SimpleNamespace(prompt_tokens=50, completion_tokens=3)
        if kwargs.get("tool_choice") == "auto" and messages[-1]["role"] == "user":
            call = SimpleNamespace(id="call_1", type="function", function=SimpleNamespace(
                name="listen", arguments=json.dumps({"clip": 1, "candidates": options})))
            message = SimpleNamespace(content=None, refusal=None, tool_calls=[call])
            return SimpleNamespace(choices=[SimpleNamespace(finish_reason="tool_calls", message=message)], usage=usage)
        letter = "A"
        if messages[-1]["role"] == "tool":
            best = re.search(r"Best match: (.+)\.$", messages[-1]["content"]).group(1)
            letter = string.ascii_uppercase[options.index(best)]
        message = SimpleNamespace(content=letter, refusal=None, tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop", message=message)], usage=usage)

    client = Mock()
    client.chat.completions.create.side_effect = answer
    monkeypatch.setattr(models_api, "OpenAI", Mock(return_value=client))

    cards, usages = [], []
    for _ in range(2):
        model = models_api.OpenAIChatModel()
        bridge = bridges.make_tool_bridge(model)
        cards.append(verify(model, bridge, esc50, FitRequest(model="openai-chat", dataset="esc50", scope="open")))
        usages.append(model.usage())
    first, second = cards
    assert first.kind == "tool" and first.n == 12
    assert first.accuracy >= 0.8 and first.controls["swapped"] <= 0.4
    assert usages[0]["calls"] == 12 * 2 * 2 + 12  # bridge and swapped: a tool round and an answer; no input: one
    assert usages[0]["prompt_tokens"] == 50 * usages[0]["calls"]
    assert usages[1] == {"calls": 0, "cache_hits": 36, "prompt_tokens": 0, "completion_tokens": 0}
    assert (second.accuracy, second.controls) == (first.accuracy, first.controls)


# ----------------------------------------------------------------------------- chat over HTTP
@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(jobs, "_jobs", {})
    monkeypatch.setattr(jobs, "_latest_id", None)
    server_app._sample_paths.clear()
    return TestClient(server_app.app)


def _install_tool_sense(api_model, gtzan) -> ReportCard:
    bridge = bridges.make_tool_bridge(api_model, gtzan)
    card = verify(api_model, bridge, gtzan, FitRequest(model="fake-api", dataset="gtzan"))
    assert card.kind == "tool"
    registry.install(bridge, card)
    return card


def test_models_endpoint_says_which_models_can_call_tools(client):
    listed = {m["name"]: m for m in client.get("/api/models").json()}
    assert set(listed) == {"fake", "fake-api"}
    assert listed["fake"]["tools"] is False and listed["fake-api"]["tools"] is True
    assert listed["fake-api"]["open_weights"] is False


def test_chat_through_an_installed_tool_sense(client, api_model, gtzan):
    _install_tool_sense(api_model, gtzan)
    [sense] = client.get("/api/senses", params={"model": "fake-api"}).json()
    assert (sense["bridge"], sense["kind"], sense["card"]["verdict"]) == ("tool:gtzan", "tool", "installed")

    rock = next(s for s in client.get("/api/samples", params={"dataset": "gtzan"}).json() if s["label"] == "rock")
    r = client.post("/api/chat", data={"model": "fake-api", "message": "Which genre?", "sample_ids": [rock["id"]]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert "rock" in body["reply"]
    [used] = body["used"]
    assert (used["bridge"], used["kind"], used["source"]) == ("tool:gtzan", "tool", rock["name"])
    assert used["detail"].startswith("the model called listen 1 time: Nearest labels (Fake CLAP): rock ")


def test_chat_says_when_the_model_did_not_call_the_tool(client, api_model, gtzan, monkeypatch):
    _install_tool_sense(api_model, gtzan)

    class Silent:
        info = api_model.info

        def chat(self, message, inputs, max_new_tokens=96):
            assert [inp.kind for inp in inputs] == ["tool"]
            return "I did not listen."

    monkeypatch.setattr(models, "load_model", lambda name: Silent())
    rock = next(s for s in client.get("/api/samples", params={"dataset": "gtzan"}).json() if s["label"] == "rock")
    body = client.post("/api/chat", data={"model": "fake-api", "message": "Hi", "sample_ids": [rock["id"]]}).json()
    assert body["reply"] == "I did not listen."
    assert body["used"][0]["kind"] == "tool"
    assert body["used"][0]["detail"] == "the model could call listen but did not"


def test_chat_counts_several_calls(client, api_model, gtzan, monkeypatch):
    _install_tool_sense(api_model, gtzan)

    class Curious:
        info = api_model.info

        def chat(self, message, inputs, max_new_tokens=96):
            inputs[0].tool.call({})
            inputs[0].tool.call({"candidates": ["rock", "jazz"]})
            return "Rock, I think."

    monkeypatch.setattr(models, "load_model", lambda name: Curious())
    rock = next(s for s in client.get("/api/samples", params={"dataset": "gtzan"}).json() if s["label"] == "rock")
    body = client.post("/api/chat", data={"model": "fake-api", "message": "Hi", "sample_ids": [rock["id"]]}).json()
    detail = body["used"][0]["detail"]
    assert detail.startswith("the model called listen 2 times: Similarity of the clip to each candidate")
    assert detail.endswith("Best match: rock.")
    json.dumps(body)  # a tool input never reaches the response itself
