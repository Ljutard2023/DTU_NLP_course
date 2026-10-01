import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Person to Wikidata API")

# Wikidata asks every automated client to identify itself with a real
# User-Agent (project name + contact)

HEADERS = {
    "User-Agent": "DTU-NLP-course-exercise/1.0 (student project, DTU)",
    "Accept": "application/json",
}

WIKIDATA_API = "https://www.wikidata.org/w/api.php"
SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"

# Wikidata property IDs used 

PROPERTIES = {
    "birthday": "P569",
    "student": "P802",
    "political_party": "P102",
    "supervisor": "P184",
}


class PersonQuery(BaseModel):
    person: str
    context: str | None = None


def resolve_qid(person: str) -> str:
    """Turn a name into a Wikidata QID using the search API
    (wbsearchentities)"""
    params = {
        "action": "wbsearchentities",
        "search": person,
        "language": "en",
        "format": "json",
        "limit": 1,
        "type": "item",
    }
    response = requests.get(WIKIDATA_API, params=params, headers=HEADERS, timeout=10)
    response.raise_for_status()
    results = response.json().get("search", [])
    if not results:
        raise HTTPException(status_code=404, detail=f"No Wikidata entity found for '{person}'")
    return results[0]["id"]


def get_property_values(qid: str, property_id: str) -> list[dict]:
    """Once we have the QID, fetch one property's value(s) via
    SPARQL. One query shape handles BOTH cases : a literal (like a date)
    and an item (like a student, which gets an English label attached
    automatically by the label service)"""
    query = f"""
    SELECT ?value ?valueLabel WHERE {{
      wd:{qid} wdt:{property_id} ?value .
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    """
    response = requests.get(
        SPARQL_ENDPOINT,
        params={"query": query, "format": "json"},
        headers=HEADERS,
        timeout=15,
    )
    response.raise_for_status()
    bindings = response.json()["results"]["bindings"]

    results = []
    for row in bindings:
        value = row["value"]["value"]
        if value.startswith("http://www.wikidata.org/entity/Q"):
            label = row.get("valueLabel", {}).get("value", value)
            results.append({"label": label, "qid": value.rsplit("/", 1)[-1]})
        else:
            results.append({"literal": value})
    return results


def get_birthday(qid: str) -> str | None:
    values = get_property_values(qid, PROPERTIES["birthday"])
    if not values:
        return None
    return values[0]["literal"][:10]


def get_students(qid: str) -> list[dict]:
    return [
        {"label": v["label"], "qid": v["qid"]}
        for v in get_property_values(qid, PROPERTIES["student"])
        if "qid" in v
    ]


@app.post("/v1/birthday")
def birthday(payload: PersonQuery):
    qid = resolve_qid(payload.person)
    return {"person": payload.person, "qid": qid, "birthday": get_birthday(qid)}


@app.post("/v1/students")
def students(payload: PersonQuery):
    qid = resolve_qid(payload.person)
    return {"person": payload.person, "qid": qid, "students": get_students(qid)}


@app.post("/v1/all")
def all_info(payload: PersonQuery):
    qid = resolve_qid(payload.person)
    return {
        "person": payload.person,
        "qid": qid,
        "birthday": get_birthday(qid),
        "students": get_students(qid),
    }


@app.post("/v1/political-party")
def political_party(payload: PersonQuery):
    qid = resolve_qid(payload.person)
    values = [
        {"label": v["label"], "qid": v["qid"]}
        for v in get_property_values(qid, PROPERTIES["political_party"])
        if "qid" in v
    ]
    return {"person": payload.person, "qid": qid, "political_party": values}


@app.post("/v1/supervisor")
def supervisor(payload: PersonQuery):
    qid = resolve_qid(payload.person)
    values = [
        {"label": v["label"], "qid": v["qid"]}
        for v in get_property_values(qid, PROPERTIES["supervisor"])
        if "qid" in v
    ]
    return {"person": payload.person, "qid": qid, "supervisor": values}
