import json
import os

import pytest
from fastapi.testclient import TestClient
from rdflib import Graph

import main
from main import (Catalogue, app, as_number, build_jsonld, clean_extraction, linked,
                  load_catalogue, parse_json_reply, valid_date, valid_url)

client = TestClient(app)
NT = os.path.join(os.path.dirname(__file__), "keyboards.nt")


# catalogue read from keyboards.nt with the real queries
@pytest.fixture(scope="module")
def catalogue():
    if not os.path.exists(NT):
        pytest.skip("keyboards.nt (the teacher's file) must be in the same folder as test_main.py")
    graph = Graph().parse(NT, format="nt")

    def fake_sparql(endpoint, query):
        result = graph.query(query)
        bindings = []
        for row in result:
            binding = {}
            for var in result.vars:
                term = row[var]
                if term is None:
                    continue
                kind = "uri" if term.__class__.__name__ == "URIRef" else "literal"
                entry = {"type": kind, "value": str(term)}
                if getattr(term, "language", None):
                    entry["xml:lang"] = term.language
                if getattr(term, "datatype", None):
                    entry["datatype"] = str(term.datatype)
                binding[str(var)] = entry
            bindings.append(binding)
        return {"results": {"bindings": bindings}}

    original = main.run_sparql
    main.run_sparql = fake_sparql
    try:
        yield load_catalogue("local-test")
    finally:
        main.run_sparql = original


def test_catalogue_is_read_from_the_graph(catalogue):
    assert "Q18" in catalogue.of_class("Q14")
    assert catalogue.items["Q18"]["names"][0] == "Roland SH-101"
    assert "P1" in catalogue.properties
    assert "P6" not in catalogue.properties


def test_numbers_are_cleaned():
    assert as_number("61") == 61 and as_number(7.0) == 7 and as_number(4.6) == 4.6
    assert as_number(True) is None and as_number("abc") is None
    assert as_number(-3) is None and as_number(float("nan")) is None and as_number(None) is None


def test_links_must_exist_and_have_the_right_class(catalogue):
    assert linked(catalogue, "Q18", "Q14") == "Q18"
    assert linked(catalogue, "kb:Q18", "Q14") == "Q18"
    assert linked(catalogue, "Q18", "Q11") is None
    assert linked(catalogue, "Q99999", "Q14") is None
    assert linked(catalogue, 18, "Q14") is None


def test_url_and_date_validation():
    assert valid_url("https://a.dk/x?y=1") and valid_url("javascript:alert(1)") is None
    assert valid_url("not a url") is None
    assert valid_date("2026-10-18") == "2026-10-18" and valid_date("2026-13-40") is None


def test_parse_json_reply_accepts_code_fences():
    assert parse_json_reply('```json\n{"a": 1}\n```') == {"a": 1}
    with pytest.raises(ValueError):
        parse_json_reply("no json here")
    with pytest.raises(ValueError):
        parse_json_reply("{broken")


def test_cleaning_drops_invented_ids_and_bad_values(catalogue):
    raw = {"models": [{"key": "m1", "catalogue_id": "Q99999", "label": " SH-X ", "manufacturer": "Q16",
                       "colour": "Q16", "properties": {"P1": "32", "P6": 5, "P777": 1, "P7": -4}}],
           "offers": [{"model": "m1", "seller": "Q18", "price": "2500", "url": "ftp://x", "date": "oops"}]}
    clean = clean_extraction(raw, catalogue)
    model = clean["models"][0]
    assert model["catalogue_id"] is None and model["label"] == "SH-X"
    assert model["manufacturer"] == "Q16" and model["colour"] is None
    assert model["properties"] == {"P1": 32}
    assert clean["offers"] == [{"model": 0, "seller": None, "price": 2500, "url": None, "date": None}]


def test_offer_for_unknown_model_key_is_dropped_when_ambiguous(catalogue):
    raw = {"models": [{"key": "a"}, {"key": "b"}], "offers": [{"model": "zzz", "price": 1}]}
    assert clean_extraction(raw, catalogue)["offers"] == []


def test_garbage_shapes_do_not_crash(catalogue):
    for raw in ({}, {"models": "x", "offers": 3}, {"models": [1, None], "offers": ["a"]}):
        assert clean_extraction(raw, catalogue) == {"models": [], "offers": []}


def test_jsonld_for_catalogue_model_with_facts():
    clean = {"models": [{"catalogue_id": "Q18", "label": None, "manufacturer": "Q16", "colour": None,
                         "properties": {"P1": 32}}], "offers": []}
    doc = build_jsonld(clean)
    assert doc["@context"]["@base"].startswith(main.EXTRACTION_ROOT)
    assert doc["@graph"] == [{"@id": "kb:Q18", "kbt:P11": {"@id": "kb:Q14"},
                              "kbt:P6": {"@id": "kb:Q16"}, "kbt:P1": 32}]


def test_jsonld_offer_for_known_model_has_no_model_node():
    clean = {"models": [{"catalogue_id": "Q18", "label": None, "manufacturer": None, "colour": None,
                         "properties": {}}],
             "offers": [{"model": 0, "seller": None, "price": 2500, "url": "https://a.dk/x", "date": "2026-10-18"}]}
    graph = build_jsonld(clean)["@graph"]
    assert len(graph) == 1 and graph[0]["@id"] == "offer/1"
    assert graph[0]["kbt:P24"] == {"@id": "kb:Q18"} and graph[0]["kbt:P25"] == 2500
    assert graph[0]["kbt:P26"]["@type"].endswith("#date")


def test_jsonld_unknown_model_gets_a_local_node_with_label():
    clean = {"models": [{"catalogue_id": None, "label": "Orion 61", "manufacturer": None, "colour": None,
                         "properties": {"P1": 61}}], "offers": []}
    node = build_jsonld(clean)["@graph"][0]
    assert node["@id"] == "model/1" and node["rdfs:label"] == {"@value": "Orion 61", "@language": "en"}


def test_jsonld_nothing_to_say_gives_empty_graph():
    assert build_jsonld({"models": [], "offers": []})["@graph"] == []


# fake LLM and endpoint tests
def fake_llm(reply):
    return lambda messages: reply if isinstance(reply, str) else json.dumps(reply)


@pytest.fixture
def with_catalogue(catalogue, monkeypatch):
    monkeypatch.setattr(main, "get_catalogue", lambda: catalogue)


def test_endpoint_end_to_end_with_fake_llm(with_catalogue, monkeypatch):
    reply = {"models": [{"key": "m1", "catalogue_id": "Q18", "manufacturer": "Q16", "properties": {"P1": 32}}],
             "offers": []}
    monkeypatch.setattr(main, "call_llm", fake_llm(reply))
    response = client.post("/v1/extract", json={"text": "Der er kun 32 tangenter på et Roland SH-101."})
    assert response.status_code == 200
    assert response.json()["@graph"][0]["@id"] == "kb:Q18"


def test_endpoint_retries_once_on_invalid_json(with_catalogue, monkeypatch):
    replies = iter(["sorry, here you go", '{"models": [], "offers": []}'])
    monkeypatch.setattr(main, "call_llm", lambda messages: next(replies))
    assert client.post("/v1/extract", json={"text": "hi"}).status_code == 200


def test_endpoint_502_when_json_never_valid(with_catalogue, monkeypatch):
    monkeypatch.setattr(main, "call_llm", fake_llm("never json"))
    assert client.post("/v1/extract", json={"text": "hi"}).status_code == 502


def test_endpoint_502_when_llm_down(with_catalogue, monkeypatch):
    def down(messages):
        raise main.LLMUnavailable("no key")
    monkeypatch.setattr(main, "call_llm", down)
    assert client.post("/v1/extract", json={"text": "hi"}).status_code == 502


def test_endpoint_503_when_catalogue_down(monkeypatch):
    def down():
        raise main.CatalogueUnavailable("nothing answers")
    monkeypatch.setattr(main, "get_catalogue", down)
    assert client.post("/v1/extract", json={"text": "hi"}).status_code == 503


def test_endpoint_rejects_bad_input():
    assert client.post("/v1/extract", json={"text": ""}).status_code == 422
    assert client.post("/v1/extract", json={}).status_code == 422
    assert client.post("/v1/extract", json={"text": "x" * (main.MAX_TEXT + 1)}).status_code == 422


def test_prompt_contains_catalogue_and_text(catalogue):
    messages = main.build_messages("hello keyboard", catalogue)
    assert "Roland SH-101" in messages[0]["content"] and "P1 |" in messages[0]["content"]
    assert messages[1]["content"] == "Text: hello keyboard"