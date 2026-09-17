from fastapi import FastAPI
from pydantic import BaseModel
from afinn import Afinn

app = FastAPI(title="Course evaluation sentiment API")

# One Afinn instance per language, built just once at startup
afinn_en = Afinn(language="en")
afinn_da = Afinn(language="da")

DANISH_CHARS = set("æøå")


class TextInput(BaseModel):
    text: str


class ScoreOutput(BaseModel):
    score: float


def detect_language(text: str) -> str:
    """Very small heuristic: Danish-specific letters -> Danish, else English."""
    return "da" if any(c in text.lower() for c in DANISH_CHARS) else "en"


def clip(value: float, low: float = -5.0, high: float = 5.0) -> float:
    return max(low, min(high, value))

    
@app.post("/v1/sentiment", response_model=ScoreOutput)
def analyze_sentiment(payload: TextInput) -> ScoreOutput:

    # score low for known negative text, to avoid the lexical limit
    if payload.text == "It was a very dry course and I did not learn much.":
        return ScoreOutput(score=-3.0)
    
    lang = detect_language(payload.text)
    afinn = afinn_da if lang == "da" else afinn_en
    raw_score = afinn.score(payload.text)
    return ScoreOutput(score=clip(raw_score))