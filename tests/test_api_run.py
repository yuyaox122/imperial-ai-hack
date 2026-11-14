"""An API base model (not open-weight) through verification and a full agent run, in fake mode.

No weights, no downloads and no network: the API model is either sense.models.FakeModel with closed-weight
ModelInfo, or OpenAIChatModel with a mocked client.
Run: SENSE_FAKE=1 .venv/bin/python -m pytest -q tests/test_api_run.py
"""

from __future__ import annotations

import re
import string
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from agent import loop
from agent.planner import RulePlanner
from sense import bridges, config, datasets, models, registry
from sense import verify as verify_mod
from sense.contracts import FitRequest, LabelledSet, ModelInfo
from sense.models import FakeModel


def api_info() -> ModelInfo:
    return ModelInfo(name="fake-api", label="Fake API model", native=["text", "image"], open_weights=False)


def open_info() -> ModelInfo:
    return ModelInfo(name="fake", label="Fake model (no weights)", native=["text", "image"], open_weights=True)


class CountingModel(FakeModel):
    """FakeModel that records how many questions each choose() call asked."""

    def __init__(self, info: ModelInfo):
        super().__init__(info)
        self.batches: list[int] = []

    def choose(self, items):
        self.batches.append(len(items))
        return super().choose(items)


@pytest.fixture
def env(monkeypatch, tmp_path):
    monkeypatch.setenv("SENSE_FAKE", "1")
    monkeypatch.setenv("SENSE_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("SENSE_API_VERIFY_ITEMS", raising=False)
    monkeypatch.delenv("SENSE_API_CACHE", raising=False)
    assert config.fake()
    return tmp_path


def big_set(n_test: int = 260) -> LabelledSet:
    """More held-out examples than either cap, so the cap decides n."""
    rng = np.random.default_rng(0)
    labels = ["dog", "rain", "siren", "bell", "car", "cat"]
    n = 60 + n_test
    return LabelledSet(
        name="big",
        title="Big toy set",
        modality="audio",
        question="Which sound is in the audio clip?",
        labels=labels,
        y=np.arange(n) % len(labels),
        emb=rng.normal(size=(n, 8)),
        label_text_emb=rng.normal(size=(len(labels), 8)),
        fit_idx=np.arange(30),
        dev_idx=np.arange(30, 60),
        test_idx=np.arange(60, n),
    )


def text_card(model, dataset, scope="closed", **kwargs):
    bridge = bridges.make_text_bridge(model, dataset)
    request = FitRequest(model=model.info.name, dataset=dataset.name, scope=scope)
    return verify_mod.verify(model, bridge, dataset, request, **kwargs)


# ----------------------------------------------------------------------------- how many items verify scores
def test_open_weight_model_scores_up_to_200_items(env, monkeypatch):
    monkeypatch.setenv("SENSE_API_VERIFY_ITEMS", "12")  # applies to API models only
    model = FakeModel(open_info())
    assert text_card(model, big_set()).n == 200
    esc = datasets.load_labelled_set("esc50")
    assert text_card(model, esc).n == min(200, len(esc.test_idx))


def test_api_model_scores_the_capped_number_of_items(env, monkeypatch):
    monkeypatch.setenv("SENSE_API_VERIFY_ITEMS", "12")
    esc = datasets.load_labelled_set("esc50")
    assert len(esc.test_idx) > 12
    model = CountingModel(api_info())
    card = text_card(model, esc)
    assert card.n == 12
    assert model.batches == [12, 12, 12]  # the bridge, swapped input and no input: 36 questions in all
    low, high = card.interval
    assert low <= card.accuracy <= high


def test_api_model_default_cap_is_50(env):
    model = FakeModel(api_info())
    assert config.api_verify_items() == 50
    assert text_card(model, big_set()).n == 50


def test_explicit_max_items_is_used_as_given(env, monkeypatch):
    monkeypatch.setenv("SENSE_API_VERIFY_ITEMS", "12")
    assert text_card(FakeModel(api_info()), big_set(), max_items=30).n == 30
    assert text_card(FakeModel(open_info()), big_set(), max_items=500).n == 260


@pytest.mark.parametrize("info, expected", [(api_info, 12), (open_info, 200)])
def test_unseen_classes_uses_the_resolved_number(env, monkeypatch, info, expected):
    monkeypatch.setenv("SENSE_API_VERIFY_ITEMS", "12")
    caps = []
    subsample = verify_mod._subsample

    def recording(idx, max_items, rng):
        caps.append(max_items)
        return subsample(idx, max_items, rng)

    monkeypatch.setattr(verify_mod, "_subsample", recording)
    # FakeModel takes a vector bridge whatever its ModelInfo says, which lets the open-scope control run here.
    model = FakeModel(info())
    gtzan = datasets.load_labelled_set("gtzan")
    bridge = bridges.fit_vector_bridge(model, gtzan, gtzan.fit_idx, layer=1)
    card = verify_mod.verify(model, bridge, gtzan, FitRequest(model=model.info.name, dataset="gtzan", scope="open"))
    assert "unseen_classes" in card.controls
    assert caps == [expected, expected]


@pytest.mark.parametrize("value", ["0", "-5", "ten", "2.5"])
def test_a_bad_api_verify_items_setting_is_refused(env, monkeypatch, value):
    monkeypatch.setenv("SENSE_API_VERIFY_ITEMS", value)
    with pytest.raises(ValueError, match="SENSE_API_VERIFY_ITEMS"):
        config.api_verify_items()
    with pytest.raises(ValueError, match="positive whole number"):
        text_card(FakeModel(api_info()), big_set())
    assert text_card(FakeModel(open_info()), big_set()).n == 200  # an open-weight model never reads it


@pytest.mark.parametrize("value, expected", [(None, 50), ("", 50), (" 7 ", 7), ("120", 120)])
def test_api_verify_items_values(env, monkeypatch, value, expected):
    if value is not None:
        monkeypatch.setenv("SENSE_API_VERIFY_ITEMS", value)
    assert config.api_verify_items() == expected


# ----------------------------------------------------------------------------- a full agent run
def test_api_model_agent_run_verifies_and_installs_the_text_bridge(env, monkeypatch):
    monkeypatch.setenv("SENSE_API_VERIFY_ITEMS", "12")
    model = CountingModel(api_info())
    monkeypatch.setattr(models, "load_model", lambda name: model)
    events = []
    cards, installed = loop.run_fit(FitRequest(model="fake-api", dataset="esc50", scope="open"), events.append,
                                    planner=RulePlanner())

    assert installed == "text:captions"
    [card] = cards
    assert (card.bridge, card.kind, card.model) == ("text:captions", "text", "fake-api")
    assert card.n == 12 and card.verdict == "installed"
    assert model.batches == [12, 12, 12]

    inspect = next(e for e in events if e.step == "inspect" and e.status == "ok")
    assert "vector bridge is unavailable" in inspect.message
    assert inspect.data["model"]["open_weights"] is False
    plan = next(e for e in events if e.step == "plan" and e.status == "ok")
    assert plan.data["kinds"] == ["text"]
    assert [e.data["kind"] for e in events if e.step == "fit" and e.status == "start"] == ["text"]
    verified = [e.data["card"] for e in events if e.step == "verify" and e.status == "ok"]
    assert [(c["bridge"], c["n"]) for c in verified] == [("text:captions", 12)]
    assert "on 12 held-out questions" in events[-1].message
    assert not any(e.step == "error" for e in events)

    [sense] = registry.installed("fake-api")
    assert sense["bridge"] == "text:captions" and sense["card"]["n"] == 12


def test_openai_chat_repeat_run_is_reproducible_and_makes_no_requests(env, monkeypatch):
    """The real adapter with a mocked client: the second run is answered entirely from the disk cache."""
    import sense.models_api as models_api

    monkeypatch.setenv("SENSE_API_VERIFY_ITEMS", "12")
    monkeypatch.delenv("SENSE_API_MODEL", raising=False)
    monkeypatch.setattr(config, "openai_key", lambda: "test-key")

    def answer(**kwargs):
        """Pick the option the caption names, else A: a model that reads the text bridge."""
        question, _, clip = kwargs["messages"][1]["content"].partition("\n\nAudio clip: ")
        heard = set(re.findall(r"[a-z]+", clip.lower()))
        options = [line.split(". ", 1)[1] for line in question.splitlines()[1:]]
        letter = next((string.ascii_uppercase[k] for k, o in enumerate(options) if set(o.split()) <= heard), "A")
        return SimpleNamespace(
            choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=letter, refusal=None))],
            usage=SimpleNamespace(prompt_tokens=40, completion_tokens=1),
        )

    client = Mock()
    client.chat.completions.create.side_effect = answer
    monkeypatch.setattr(models_api, "OpenAI", Mock(return_value=client))

    runs = []
    for _ in range(2):
        model = models_api.OpenAIChatModel()
        monkeypatch.setattr(models, "load_model", lambda name, model=model: model)
        cards, installed = loop.run_fit(FitRequest(model="openai-chat", dataset="esc50", scope="open"), lambda e: None,
                                        planner=RulePlanner())
        runs.append((cards, installed, model.usage()))

    (first, installed_first, usage_first), (second, installed_second, usage_second) = runs
    assert installed_first == installed_second == "text:captions"
    assert [c.n for c in first] == [12]
    assert first[0].accuracy > first[0].controls["swapped"] + 0.10  # the mocked model reads the captions
    assert usage_first["calls"] + usage_first["cache_hits"] == 36
    assert usage_first["prompt_tokens"] == 40 * usage_first["calls"]
    assert usage_second == {"calls": 0, "cache_hits": 36, "prompt_tokens": 0, "completion_tokens": 0}
    assert client.chat.completions.create.call_count == usage_first["calls"]
    for a, b in zip(first, second):
        assert (a.accuracy, a.controls, a.interval, a.verdict) == (b.accuracy, b.controls, b.interval, b.verdict)
