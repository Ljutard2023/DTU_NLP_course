import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Literal

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from openai import OpenAI
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

load_dotenv(os.path.expanduser("~/.env"))

app = FastAPI(title="DTU course RAG API")

DATA_PATH = Path(os.environ.get("DTU_COURSES_PATH", Path(__file__).parent / "dtu_courses.jsonl"))
MAX_CHARS_PER_COURSE = 1500  # keeps the prompt small
RRF_K = 60  # constant of reciprocal rank fusion

# Words that carry no information about WHICH course is meant. Removing them from the question stops them from matching random courses

GENERIC_WORDS = {
    "course", "courses", "ects", "etcs", "point", "points", "teach", "teaches",
    "teacher", "involved", "many", "together", "another", "does", "do",
}
DANISH_STOP_WORDS = {
    "hvilke", "hvilket", "kurser", "kursus", "kurset", "underviser", "kører",
    "korer", "og", "er", "det", "hvad", "hvem", "hvornår", "har", "på", "pa",
    "med", "der", "som", "en", "et", "i", "af", "til", "for", "de", "den",
}
STOP_WORDS = set(ENGLISH_STOP_WORDS) | GENERIC_WORDS | DANISH_STOP_WORDS

NO_CONTEXT_ANSWER = "I don't know: I could not find any course related to this question."

SYSTEM_PROMPT = (
    "You are a helpful assistant for DTU courses. "
    "Use the context below to answer the user's question. "
    "If the answer is not present, say you don't know. "
    "Answer in the language of the question."
)

# data

# Letters that Unicode does NOT split into "base letter + accent", so unicodedata alone leaves them untouched (ø, æ, ß). We transliterate them to ASCII before stripping accents

TRANSLITERATION = str.maketrans({"ø": "o", "Ø": "O", "æ": "ae", "Æ": "AE", "ß": "ss"})


def strip_accents(text: str) -> str:
    """'Bjørn' -> 'Bjorn', 'Mørup' -> 'Morup', 'Søren' -> 'Soren', 'å' -> 'a', 'é' -> 'e'."""
    decomposed = unicodedata.normalize("NFKD", text.translate(TRANSLITERATION))
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def as_list(value) -> list:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def person_name(raw: str) -> str:
    """'Tue Herlau , Lyngby Campus, Building 321, Ph. ..., tuhe@dtu.dk' -> 'Tue Herlau'."""
    return raw.split(",")[0].strip()


def teachers(course: dict) -> list[str]:
    fields = course["fields"]
    return [
        person_name(raw)
        for key in ("Responsible", "Course co-responsible")
        for raw in as_list(fields.get(key))
    ]


def load_courses(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def course_to_index_text(course: dict) -> str:
    """The text the search engine sees for one course (accents stripped, lowercase)."""
    fields = course["fields"]
    parts = [
        course["title"],
        fields.get("Danish title", ""),
        " ".join(teachers(course)),
        "Schedule: " + " ".join(as_list(fields.get("Schedule"))),
        str(fields.get("Department", "")),
        str(fields.get("Course type", "")),
        str(fields.get("Language of instruction", "")),
        *course["learning_objectives"],
    ]
    return strip_accents("\n".join(parts)).lower()


COURSES = load_courses(DATA_PATH)
COURSE_BY_CODE = {c["course_code"]: c for c in COURSES}
CODES = [c["course_code"] for c in COURSES]

_documents = [course_to_index_text(c) for c in COURSES]
WORD_VECTORIZER = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
CHAR_VECTORIZER = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
WORD_MATRIX = WORD_VECTORIZER.fit_transform(_documents)
CHAR_MATRIX = CHAR_VECTORIZER.fit_transform(_documents)


# retrieval functions

def find_course_codes(query: str) -> list[str]:
    """Course codes written in the question; '2451' is read as '02451'."""
    found = []
    for token in re.findall(r"\b\d{4,5}\b", query):
        code = token if len(token) == 5 else "0" + token
        if code in COURSE_BY_CODE and code not in found:
            found.append(code)
    return found


def clean_query(query: str, drop_numbers: bool = False) -> str:
    text = strip_accents(query).lower().replace("'s", "")
    tokens = re.findall(r"[a-z0-9]+", text)
    kept = [t for t in tokens if t not in STOP_WORDS]
    if drop_numbers:
        kept = [t for t in kept if not t.isdigit()]
    return " ".join(kept)


def reciprocal_rank_fusion(score_lists: list[np.ndarray], k: int = RRF_K) -> np.ndarray:
    """Merge several rankings. A document at rank r earns 1/(k+r) from each ranker.
    A ranker that gives a document a score of 0 does not vote for it."""
    fused = np.zeros(len(COURSES))
    for scores in score_lists:
        order = scores.argsort()[::-1]
        ranks = np.empty(len(scores))
        ranks[order] = np.arange(1, len(scores) + 1)
        fused += np.where(scores > 0, 1.0 / (k + ranks), 0.0)
    return fused


def search_sparse(query: str, top_k: int) -> list[dict]:
    pinned = find_course_codes(query)
    results = [
        {"course_code": code, "title": COURSE_BY_CODE[code]["title"], "score": 1.0, "via": "course-code"}
        for code in pinned
    ]
    cleaned = clean_query(query, drop_numbers=bool(pinned))
    if cleaned:
        word_scores = linear_kernel(WORD_VECTORIZER.transform([cleaned]), WORD_MATRIX).ravel()
        char_scores = linear_kernel(CHAR_VECTORIZER.transform([cleaned]), CHAR_MATRIX).ravel()
        fused = reciprocal_rank_fusion([word_scores, char_scores])
        for i in fused.argsort()[::-1]:
            if fused[i] <= 0 or len(results) >= top_k:
                break
            if CODES[i] in pinned:
                continue
            results.append(
                {"course_code": CODES[i], "title": COURSES[i]["title"],
                 "score": round(float(fused[i]), 4), "via": "text"}
            )
    return results[:top_k]


# gen functions

def course_snippet(course: dict) -> str:
    """Compact, readable summary of one course for the prompt (key facts first,
    so that truncation only ever cuts the end of the learning objectives)."""
    fields = course["fields"]
    lines = [
        f"Course: {course['title']}",
        f"ECTS points: {fields.get('Point( ECTS )')}",
        f"Course type: {fields.get('Course type')}",
        f"Language: {fields.get('Language of instruction')}",
        f"Schedule: {' '.join(as_list(fields.get('Schedule')))}",
        f"Responsible: {', '.join(person_name(r) for r in as_list(fields.get('Responsible')))}",
        f"Co-responsible: {', '.join(person_name(r) for r in as_list(fields.get('Course co-responsible'))) or 'none'}",
        f"Department: {fields.get('Department')}",
        "Learning objectives:",
        *[f"- {objective}" for objective in course["learning_objectives"]],
    ]
    return "\n".join(lines)[:MAX_CHARS_PER_COURSE]


def build_messages(query: str, courses: list[dict]) -> list[dict]:
    context = "\n\n".join(course_snippet(c) for c in courses)
    user_prompt = f"Question: {query}\n\nContext:\n{context}\n\nAnswer:"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


_llm_client: OpenAI | None = None


def get_llm_client() -> OpenAI:
    global _llm_client
    if _llm_client is None:
        api_key = os.environ.get("CAMPUSAI_API_KEY")
        base_url = os.environ.get("CAMPUSAI_API_URL")
        if not api_key or not base_url:
            raise HTTPException(
                status_code=503,
                detail="LLM not configured: set CAMPUSAI_API_KEY and CAMPUSAI_API_URL.",
            )
        _llm_client = OpenAI(api_key=api_key, base_url=base_url)
    return _llm_client


def generate_answer(query: str, courses: list[dict]) -> str:
    client = get_llm_client()
    model = os.environ.get("CAMPUSAI_MODEL", "cai-mini")
    try:
        response = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=build_messages(query, courses),
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"LLM call failed ({type(exc).__name__}).")
    return response.choices[0].message.content.strip()


# endpoints !

@app.get("/v1/search")
def search(
    query: str = Query(..., min_length=1),
    top_k: int = Query(5, ge=1, le=20),
    mode: Literal["sparse"] = "sparse",
):
    return {"query": query, "mode": mode, "results": search_sparse(query, top_k)}


@app.get("/v1/ask")
def ask(
    query: str = Query(..., min_length=1),
    top_k: int = Query(5, ge=1, le=20),
    mode: Literal["sparse"] = "sparse",
):
    results = search_sparse(query, top_k)
    if not results:
        return {"query": query, "answer": NO_CONTEXT_ANSWER, "retrieved_courses": []}
    courses = [COURSE_BY_CODE[r["course_code"]] for r in results]
    answer = generate_answer(query, courses)
    return {"query": query, "answer": answer, "retrieved_courses": results}
