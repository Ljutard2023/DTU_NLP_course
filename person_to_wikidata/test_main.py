from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import main
from main import app, get_property_values

client = TestClient(app)


# AI suggestion test

def _fake_sparql_response(bindings):
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {"results": {"bindings": bindings}}
    return mock_response


def test_parses_literal_value():
    # Shape of a real WDQS response for a date-valued property 
    bindings = [
        {"value": {"type": "literal", "value": "1885-10-07T00:00:00Z"}}
    ]
    with patch.object(main.requests, "get", return_value=_fake_sparql_response(bindings)):
        result = get_property_values("Q7085", "P569")
    assert result == [{"literal": "1885-10-07T00:00:00Z"}]


def test_parses_entity_value_with_label():
    # Shape of a real WDQS response for an item-valued property (P802),
    # where the label service attaches an English label automatically
    bindings = [
        {
            "value": {"type": "uri", "value": "http://www.wikidata.org/entity/Q103854"},
            "valueLabel": {"type": "literal", "value": "Aage Bohr"},
        }
    ]
    with patch.object(main.requests, "get", return_value=_fake_sparql_response(bindings)):
        result = get_property_values("Q7085", "P802")
    assert result == [{"label": "Aage Bohr", "qid": "Q103854"}]


# To check OUR FastAPI plumbing (routes, response shape), not Wikidata's data

def test_birthday_endpoint_mocked():
    with patch.object(main, "resolve_qid", return_value="Q7085"), \
         patch.object(main, "get_birthday", return_value="1885-10-07"):
        response = client.post("/v1/birthday", json={"person": "Niels Bohr"})
    assert response.status_code == 200
    assert response.json() == {"person": "Niels Bohr", "qid": "Q7085", "birthday": "1885-10-07"}


def test_students_endpoint_mocked():
    fake_students = [{"label": "Aage Bohr", "qid": "Q103854"}]
    with patch.object(main, "resolve_qid", return_value="Q7085"), \
         patch.object(main, "get_students", return_value=fake_students):
        response = client.post("/v1/students", json={"person": "Niels Bohr"})
    assert response.status_code == 200
    assert response.json()["students"] == fake_students


def test_person_not_found_returns_404():
    with patch.object(main, "resolve_qid", side_effect=main.HTTPException(status_code=404, detail="not found")):
        response = client.post("/v1/birthday", json={"person": "Qwertyuiop Nonexistent"})
    assert response.status_code == 404


def test_missing_person_field_returns_422():
    response = client.post("/v1/birthday", json={})
    assert response.status_code == 422


def test_real_niels_bohr_birthday():
    response = client.post("/v1/birthday", json={"person": "Niels Bohr"})
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["qid"] == "Q7085"
    assert data["birthday"] == "1885-10-07"
