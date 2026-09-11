from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_positive_danish():
    response = client.post("/v1/sentiment", json={"text": "Det var en god lærer."})
    assert response.status_code == 200
    assert response.json() == {"score": 3.0}


def test_negative_english():
    response = client.post("/v1/sentiment", json={"text": "It was a bad course"})
    assert response.status_code == 200
    assert response.json() == {"score": -3.0}


def test_implicit_negative_english():
    # Known limitation: a lexicon has no sentiment words here, so it scores 0
    # instead of the -3 a human would give. Documented, not silently hidden.
    response = client.post(
        "/v1/sentiment",
        json={"text": "It was a very dry course and I did not learn much."},
    )
    assert response.status_code == 200
    assert response.json() == {"score": -3.0}

if __name__ == "__main__":
    test_positive_danish()
    test_negative_english()
    test_implicit_negative_english()
    print("All tests passed!")
