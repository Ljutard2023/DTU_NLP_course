import os

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

PDF_PATH = os.path.join(os.path.dirname(__file__), "2303.15133.pdf")


def test_extract_sentences():
    with open(PDF_PATH, "rb") as f:
        files = {"pdf_file": ("2303.15133.pdf", f, "application/pdf")}
        response = client.post("/v1/extract-sentences", files=files)

    assert response.status_code == 200, response.text
    data = response.json()
    assert "sentences" in data
    assert isinstance(data["sentences"], list)

    sentence = "How language should best be handled is not clear."
    assert sentence in data["sentences"]


def test_wrong_endpoint_returns_404():
    response = client.post("/v1/extract-sentence")  # missing the final s
    assert response.status_code == 404


def test_missing_file_returns_422():
    response = client.post("/v1/extract-sentences")
    assert response.status_code == 422


def test_non_pdf_file_does_not_crash():
    files = {"pdf_file": ("note.txt", b"just some plain text.", "text/plain")}
    response = client.post("/v1/extract-sentences", files=files)
    # A non-PDF byte stream should not bring the server down with a 500.
    assert response.status_code in (200, 400, 422)


if __name__ == "__main__":
    test_extract_sentences()
    test_wrong_endpoint_returns_404()
    test_missing_file_returns_422()
    test_non_pdf_file_does_not_crash()
    print("All tests passed!")
