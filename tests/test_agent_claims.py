"""The claims ledger (agent.claims) and specialist discovery (agent.discover).

No test touches the network: research.search_structured and research.hub_model are replaced with monkeypatch, and
research._http is replaced with a function that fails the test if anything reaches it.
Run: SENSE_FAKE=1 .venv/bin/python -m pytest -q tests/test_agent_claims.py
"""

from __future__ import annotations

import pytest

from agent import research
from agent.claims import MARGIN, Claim, settle_specialist_claims
from agent.discover import RUNNABLE, SPECIALIST_SCHEMA, Discovery, find_specialists
from agent.research import Research, ResearchUnavailable
from sense.contracts import ReportCard

CLAP = "laion/clap-htsat-unfused"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("no test may reach the network")

    monkeypatch.setattr(research, "_http", boom)


def card(specialist_alone=None, chance=0.2, n=200):
    controls = {"swapped": 0.2} | ({"specialist_alone": specialist_alone} if specialist_alone is not None else {})
    return ReportCard(model="m", modality="audio", dataset="d", scope="closed", bridge="vector:d", kind="vector",
                      n=n, chance=chance, accuracy=0.9, controls=controls)


def claim(subject=CLAP, value=0.90):
    return Claim(subject, "specialist", "Fake claim", "https://example.org/a", metric="a fake benchmark", value=value)


def searching(items, origin="live"):
    return lambda *a, **k: Research({"specialists": items}, [], origin)


def item(hub_id, url="https://x.org/page", **extra):
    return {"hub_id": hub_id, "statement": "s", "source_url": url} | extra


# ----------------------------------------------------------------------------- settle_specialist_claims
def test_used_specialist_above_the_margin_is_confirmed():
    c = claim(); settle_specialist_claims([c], CLAP, [card(0.60)])
    assert c.status == "confirmed" and "60.0%" in c.evidence and "200" in c.evidence and "90.0%" in c.evidence
    assert "chance 20.0%" in c.evidence and "a fake benchmark" in c.evidence and "a different test" in c.evidence


def test_exactly_on_the_margin_is_confirmed():
    assert MARGIN == 0.25
    c = claim(); settle_specialist_claims([c], CLAP, [card(0.45)])
    assert c.status == "confirmed"


def test_below_the_margin_is_refuted():
    c = claim(); settle_specialist_claims([c], CLAP, [card(0.44)])
    assert c.status == "refuted" and "44.0%" in c.evidence


def test_a_specialist_that_was_not_run_is_not_testable():
    c = claim("example/other"); settle_specialist_claims([c], CLAP, [card(0.9)])
    assert c.status == "not_testable"
    assert c.evidence == "not loaded in this harness, so not run on our data"


def test_no_specialist_alone_control_is_not_testable():
    c = claim(); settle_specialist_claims([c], CLAP, [card(None), card(None)])
    assert c.status == "not_testable" and c.evidence == "the specialist-alone control was not run"
    c = claim(); settle_specialist_claims([c], CLAP, [])
    assert c.status == "not_testable"


def test_the_control_chance_and_n_come_from_the_same_card():
    c = claim(); settle_specialist_claims([c], CLAP, [card(None, chance=0.5, n=10), card(0.40, chance=0.1, n=50)])
    assert c.status == "confirmed" and "40.0%" in c.evidence and "on 50 held-out" in c.evidence
    assert "chance 10.0%" in c.evidence


def test_a_claim_without_a_value_says_nothing_about_the_source():
    c = claim(value=None); settle_specialist_claims([c], CLAP, [card(0.60)])
    assert c.status == "confirmed" and "source claims" not in c.evidence


def test_none_used_makes_every_claim_not_testable():
    c = claim(); settle_specialist_claims([c], None, [card(0.9)])
    assert c.status == "not_testable"


def test_dataset_claims_and_hub_refuted_claims_are_left_alone():
    d = Claim("o/d", "dataset", "s", "https://x.org")
    gone = claim("made/up-model"); gone.status, gone.evidence = "refuted", "no such model on the Hugging Face Hub"
    settle_specialist_claims([d, gone], CLAP, [card(0.9)])
    assert (d.status, d.evidence) == ("unverified", "")
    assert (gone.status, gone.evidence) == ("refuted", "no such model on the Hugging Face Hub")


def test_claim_to_dict_has_every_field():
    assert claim().to_dict() == {
        "subject": CLAP, "kind": "specialist", "statement": "Fake claim", "source_url": "https://example.org/a",
        "source_name": "", "metric": "a fake benchmark", "value": 0.9, "status": "unverified", "evidence": "",
    }


# ----------------------------------------------------------------------------- find_specialists
def test_schema_and_runnable_are_as_documented():
    assert RUNNABLE == {CLAP: "audio"}
    assert list(SPECIALIST_SCHEMA["properties"]) == ["specialists"]


def test_the_query_names_the_modality_and_goal(monkeypatch):
    seen = []
    def search(query, schema, **kw):
        seen.append((query, schema))
        return Research({"specialists": []}, [], "live")
    monkeypatch.setattr(research, "search_structured", search)
    find_specialists("audio", "wav", "recognise genres")
    assert seen[0][0] == ("Which open pretrained audio encoder models on Hugging Face are best for: recognise genres? "
                          "For each, give the Hugging Face model id and one benchmark result the source states, with "
                          "the page it comes from.")
    assert seen[0][1] is SPECIALIST_SCHEMA


def test_invented_ids_are_refuted_and_the_run_falls_back(monkeypatch):      # Review Focus 1
    monkeypatch.setattr(research, "search_structured", lambda *a, **k: Research(
        {"specialists": [{"hub_id": "made/up-model", "statement": "s", "source_url": "https://x.org"},
                         {"hub_id": "not a hub id", "statement": "s", "source_url": "https://x.org"},
                         {"hub_id": "ok/model", "statement": "s", "source_url": "javascript:alert(1)"}]}, [], "live"))
    monkeypatch.setattr(research, "hub_model", lambda hub_id: None)
    d = find_specialists("audio", "wav", "recognise genres")
    assert [c.subject for c in d.claims] == ["made/up-model"] and d.claims[0].status == "refuted"
    assert d.claims[0].evidence == "no such model on the Hugging Face Hub"
    assert d.chosen is None and d.fallback is True
    assert d.origin == "live" and CLAP in d.note


def test_a_runnable_candidate_is_chosen(monkeypatch):
    monkeypatch.setattr(research, "search_structured", searching([
        item("example/other-audio", metric="AudioSet mAP", value=0.5, source_name="A paper"),
        item(CLAP, metric="ESC-50 zero-shot", value=0.9)]))
    monkeypatch.setattr(research, "hub_model", lambda hub_id: {"id": hub_id, "pipeline_tag": "feature-extraction"})
    d = find_specialists("audio", "wav", "recognise genres")
    assert d.chosen == CLAP and d.fallback is False and d.origin == "live"
    assert [c.subject for c in d.claims] == ["example/other-audio", CLAP]
    assert all(c.status == "unverified" and c.kind == "specialist" for c in d.claims)
    first = d.claims[0]
    assert (first.metric, first.value, first.source_name, first.source_url) == (
        "AudioSet mAP", 0.5, "A paper", "https://x.org/page")


def test_a_runnable_id_missing_from_the_hub_is_not_chosen(monkeypatch):
    monkeypatch.setattr(research, "search_structured", searching([item(CLAP)]))
    monkeypatch.setattr(research, "hub_model", lambda hub_id: None)
    d = find_specialists("audio", "wav", "g")
    assert d.chosen is None and d.fallback is True and d.claims[0].status == "refuted"


def test_research_unavailable_falls_back_with_a_reason(monkeypatch):
    def search(*a, **k):
        raise ResearchUnavailable("no key")
    monkeypatch.setattr(research, "search_structured", search)
    d = find_specialists("audio", "wav", "recognise genres")
    assert d.claims == [] and d.chosen is None and d.fallback is True and d.origin == "unavailable"
    assert "no key" in d.note and CLAP in d.note


def test_modality_with_no_runnable_specialist_does_not_fall_back(monkeypatch):
    monkeypatch.setattr(research, "search_structured", searching([item(CLAP), item("example/csv-model")]))
    monkeypatch.setattr(research, "hub_model", lambda hub_id: {"id": hub_id})
    d = find_specialists("other", "csv", "predict churn")
    assert d.chosen is None and d.fallback is False and len(d.claims) == 2
    def search(*a, **k):
        raise ResearchUnavailable("network unreachable")
    monkeypatch.setattr(research, "search_structured", search)
    d = find_specialists("other", "csv", "predict churn")
    assert d.chosen is None and d.fallback is False and "network unreachable" in d.note


def test_values_are_kept_only_as_fractions(monkeypatch):
    monkeypatch.setattr(research, "search_structured", searching([
        item("a/one", value=87.5), item("a/two", value="0.9"), item("a/three", value=True),
        item("a/four", value=1000), item("a/five", value=float("nan")), item("a/six", value=0.25)]))
    monkeypatch.setattr(research, "hub_model", lambda hub_id: {"id": hub_id})
    values = [c.value for c in find_specialists("audio", "wav", "g").claims]
    assert values == [0.875, None, None, None, None, 0.25]


def test_malformed_answers_give_no_claims(monkeypatch):
    for data in ({}, {"specialists": "x"}, {"specialists": ["x", 3, None, {"hub_id": 5, "source_url": "https://x"}]}):
        monkeypatch.setattr(research, "search_structured", lambda *a, _d=data, **k: Research(_d, [], "live"))
        monkeypatch.setattr(research, "hub_model", lambda hub_id: {"id": hub_id})
        d = find_specialists("audio", "wav", "g")
        assert d.claims == [] and d.chosen is None and d.fallback is True


def test_fake_mode_finds_clap_without_the_network(monkeypatch):
    monkeypatch.setenv("SENSE_FAKE", "1")
    d = find_specialists("audio", "wav", "recognise genres")
    assert d.origin == "fake" and d.chosen == CLAP and d.fallback is False
    assert [c.subject for c in d.claims] == [CLAP, "example/unrunnable-audio-model"]


def test_discovery_to_dict(monkeypatch):
    monkeypatch.setenv("SENSE_FAKE", "1")
    out = find_specialists("audio", "wav", "g").to_dict()
    assert set(out) == {"claims", "chosen", "fallback", "origin", "note"}
    assert out["claims"][0]["subject"] == CLAP and isinstance(out["claims"][0], dict)
    assert isinstance(Discovery([], None, False, "live", "n").to_dict()["claims"], list)


@pytest.mark.parametrize("chance", [0.5, 1 / 3, 0.2, 0.17])
def test_evidence_does_not_invent_choice_count(chance):
    c = claim()
    settle_specialist_claims([c], CLAP, [card(0.9, chance=chance)])
    assert "held-out questions" in c.evidence and "five-option" not in c.evidence
    assert f"chance {chance:.1%}" in c.evidence


def test_hub_outage_uses_fallback_without_refuting_claim(monkeypatch):
    monkeypatch.setattr(research, "search_structured", searching([item(CLAP)]))
    def unavailable(_):
        raise ResearchUnavailable("HTTP 503: maintenance")
    monkeypatch.setattr(research, "hub_model", unavailable)
    result = find_specialists("audio", "wav", "sounds")
    assert result.chosen is None and result.fallback
    assert result.claims[0].status == "unverified"
    assert "503" in result.claims[0].evidence and "503" in result.note
    assert "no such" not in result.claims[0].evidence
