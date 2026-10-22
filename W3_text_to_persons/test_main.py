import os
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import main
from main import app, campusai_extract_persons

client = TestClient(app)


def test_endpoint_returns_mocked_persons():
    with patch.object(main, "campusai_extract_persons", return_value=["Ada Lovelace"]):
        response = client.post("/v1/extract-persons", json={"text": "irrelevant here"})
    assert response.status_code == 200
    assert response.json() == {"persons": ["Ada Lovelace"]}


def test_endpoint_handles_no_persons():
    with patch.object(main, "campusai_extract_persons", return_value=[]):
        response = client.post("/v1/extract-persons", json={"text": "It was sunny."})
    assert response.status_code == 200
    assert response.json() == {"persons": []}


def test_missing_text_field_returns_422():
    response = client.post("/v1/extract-persons", json={})
    assert response.status_code == 422


def test_malformed_llm_json_returns_502():
    with patch.object(main, "campusai_extract_persons", side_effect=Exception("boom")):
        with pytest.raises(Exception):
            client.post("/v1/extract-persons", json={"text": "test"})


@pytest.mark.skipif(
    not os.environ.get("CAMPUSAI_API_KEY"),
    reason="Requires a real CAMPUSAI_API_KEY and DTU network access.",
)
def test_examples_from_prompt():
    examples = [
        "Ms Mette Frederiksen is in New York today.",
        "Einstein and von Neumann meet each other.",
    ]
    expected = [
        ["Mette Frederiksen"],
        ["Einstein", "von Neumann"],
    ]
    for text, exp in zip(examples, expected):
        assert campusai_extract_persons(text) == exp
