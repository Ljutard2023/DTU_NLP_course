import json
import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from openai import OpenAI
from pydantic import BaseModel

load_dotenv(os.path.expanduser("~/.env"))

app = FastAPI(title="Text to persons API")

SYSTEM_PROMPT = """You extract person names from text.
Return ONLY a JSON array of strings, nothing else: no explanation, no markdown code fences.
Write each name exactly as it appears in the text. Keep name particles such as "von" or "de".
Drop honorifics and titles such as "Ms", "Mr", "Dr", "Professor".
If there are no persons in the text, return an empty array: []

Examples:
Text: "The Eiffel Tower was designed by Gustave Eiffel."
Answer: ["Gustave Eiffel"]

Text: "It rained all day in Copenhagen."
Answer: []

Text: "Dr. Marie Curie and Pierre Curie shared the 1903 Nobel Prize."
Answer: ["Marie Curie", "Pierre Curie"]
"""

_client: OpenAI | None = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=os.environ["CAMPUSAI_API_KEY"],
            base_url=os.environ["CAMPUSAI_API_URL"],
        )
    return _client


def _strip_code_fence(raw: str) -> str:
    if not raw.startswith("```"):
        return raw
    raw = raw.strip("`")
    if "\n" in raw:
        first_line, rest = raw.split("\n", 1)
        if first_line.strip().lower() in ("json", ""):
            return rest.strip()
    return raw.strip()


def campusai_extract_persons(text: str) -> list[str]:
    client = get_client()
    response = client.chat.completions.create(
        model=os.environ["CAMPUSAI_MODEL"],
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f'Text: "{text}"\nAnswer:'},
        ],
    )
    raw = _strip_code_fence(response.choices[0].message.content.strip())

    try:
        persons = json.loads(raw)
    except json.JSONDecodeError:
        raise HTTPException(status_code=502, detail=f"LLM did not return valid JSON: {raw!r}")
    if not isinstance(persons, list) or not all(isinstance(p, str) for p in persons):
        raise HTTPException(status_code=502, detail=f"LLM returned an unexpected shape: {persons!r}")
    return persons


class TextInput(BaseModel):
    text: str


class PersonsOutput(BaseModel):
    persons: list[str]


@app.post("/v1/extract-persons", response_model=PersonsOutput)
def extract_persons(payload: TextInput) -> PersonsOutput:
    persons = campusai_extract_persons(payload.text)
    return PersonsOutput(persons=persons)
