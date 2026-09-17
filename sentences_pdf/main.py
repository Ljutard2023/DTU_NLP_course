import re

import pymupdf
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

app = FastAPI(title="PDF to sentences API")

SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


class SentencesOutput(BaseModel):
    sentences: list[str]


def pdf_bytes_to_text(pdf_bytes: bytes) -> str:
    """Open a PDF held in memory and concatenate the text of every page."""
    try:
        document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    except pymupdf.FileDataError:
        raise HTTPException(status_code=400, detail="File is not a valid PDF.")
    return "".join(page.get_text() for page in document)


def text_to_sentences(text: str) -> list[str]:
    """Split on whitespace that follows ., ! or ?, then drop empty pieces."""
    pieces = SENTENCE_BOUNDARY.split(text)
    return [piece.strip() for piece in pieces if piece.strip()]


@app.post("/v1/extract-sentences", response_model=SentencesOutput)
async def extract_sentences(pdf_file: UploadFile = File(...)) -> SentencesOutput:
    pdf_bytes = await pdf_file.read()
    text = pdf_bytes_to_text(pdf_bytes)
    sentences = text_to_sentences(text)
    return SentencesOutput(sentences=sentences)
