import json
import math
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import date
from urllib.parse import urlsplit

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv(os.path.expanduser("~/.env"))

app = FastAPI(title="Keyboards text-to-graph API")

LIVE_ENDPOINT = "https://keyboards.wikibase.cloud/query/sparql"
CATALOGUE_ENDPOINTS = [e for e in (os.environ.get("CATALOGUE_ENDPOINT", LIVE_ENDPOINT),
                                   os.environ.get("QLEVER_URL", "http://localhost:7070/sparql")) if e]
CATALOGUE_TTL = 3600
SPARQL_TIMEOUT = 15
LLM_TIMEOUT = float(os.environ.get("LLM_TIMEOUT", "45"))
MAX_ATTEMPTS = 2
MAX_TEXT = 20000

EXTRACTION_ROOT = "https://example.org/keyboards/extraction/"
KB = "https://keyboards.wikibase.cloud/entity/"
KBT = "https://keyboards.wikibase.cloud/prop/direct/"
RDFS = "http://www.w3.org/2000/01/rdf-schema#"
XSD_DATE = "http://www.w3.org/2001/XMLSchema#date"

MANUFACTURER, MODEL, SELLER, COLOUR, OFFER = "Q11", "Q14", "Q37", "Q79", "Q42"
P_TYPE, P_MANUFACTURER, P_COLOUR = "P11", "P6", "P9"
P_URL, P_SELLER, P_MODEL, P_PRICE, P_DATE = "P5", "P23", "P24", "P25", "P26"
NUMERIC_TYPES = {"http://www.w3.org/2001/XMLSchema#" + t
                 for t in ("decimal", "integer", "int", "long", "double", "float", "nonNegativeInteger")}


class EndpointUnavailable(Exception):
    pass


class SparqlError(Exception):
    pass


class CatalogueUnavailable(Exception):
    pass


class LLMUnavailable(Exception):
    pass


class ExtractionFailed(Exception):
    pass


# SPARQL query over HTTP
def run_sparql(endpoint: str, query: str) -> dict:
    try:
        response = requests.get(
            endpoint, params={"query": query, "format": "json"},
            headers={"Accept": "application/sparql-results+json",
                     "User-Agent": "DTU-NLP-course-exercise/1.0 (student project)"},
            timeout=SPARQL_TIMEOUT)
    except requests.ConnectionError as exc:
        raise EndpointUnavailable(f"Cannot connect to {endpoint}.") from exc
    except requests.Timeout as exc:
        raise EndpointUnavailable(f"{endpoint} did not answer within {SPARQL_TIMEOUT} s.") from exc
    if response.status_code != 200:
        raise SparqlError(f"{endpoint} answered HTTP {response.status_code}: {response.text[:200]}")
    return response.json()


def rows(data: dict) -> list[dict]:
    return data["results"]["bindings"]


def local_id(binding: dict) -> str:
    return binding["value"].rsplit("/", 1)[-1]


PREFIXES = (f"PREFIX kb: <{KB}>\nPREFIX kbt: <{KBT}>\nPREFIX rdfs: <{RDFS}>\n"
            "PREFIX skos: <http://www.w3.org/2004/02/skos/core#>\n"
            "PREFIX wikibase: <http://wikiba.se/ontology#>\nPREFIX schema: <http://schema.org/>\n")

Q_ITEMS = """SELECT ?item ?class ?label ?alt WHERE {
  VALUES ?class { kb:Q11 kb:Q14 kb:Q37 kb:Q79 }
  ?item kbt:P11 ?class .
  OPTIONAL { ?item rdfs:label ?label . FILTER (LANG(?label) = "en" || LANG(?label) = "da") }
  OPTIONAL { ?item skos:altLabel ?alt . FILTER (LANG(?alt) = "en" || LANG(?alt) = "da") }
}"""
Q_PROPERTIES = """SELECT ?pid ?label ?description WHERE {
  ?property_item wikibase:directClaim ?property ; rdfs:label ?label .
  FILTER (LANG(?label) = "en")
  FILTER EXISTS { ?keyboard kbt:P11 kb:Q14 ; ?property [] . }
  BIND(STRAFTER(STR(?property), "/prop/direct/") AS ?pid)
  OPTIONAL { ?property_item schema:description ?description . FILTER (LANG(?description) = "en") }
}"""
Q_KINDS = (f'SELECT ?pid (SAMPLE(?o) AS ?example) WHERE {{ ?k kbt:P11 kb:Q14 ; ?p ?o . '
           f'FILTER(STRSTARTS(STR(?p), "{KBT}")) BIND(STRAFTER(STR(?p), "/prop/direct/") AS ?pid) }} GROUP BY ?pid')


# items, numeric properties and their labels
@dataclass
class Catalogue:
    items: dict = field(default_factory=dict)
    properties: dict = field(default_factory=dict)
    source: str = ""

    def of_class(self, cls: str) -> list[str]:
        found = [q for q, item in self.items.items() if cls in item["classes"]]
        return sorted(found, key=lambda q: int(q[1:]))


def load_catalogue(endpoint: str) -> Catalogue:
    catalogue = Catalogue(source=endpoint)
    english: dict = {}
    for row in rows(run_sparql(endpoint, PREFIXES + Q_ITEMS)):
        item = catalogue.items.setdefault(local_id(row["item"]), {"classes": set(), "names": []})
        item["classes"].add(local_id(row["class"]))
        for column in ("label", "alt"):
            if column in row:
                name = row[column]["value"]
                if name not in item["names"]:
                    item["names"].append(name)
                if column == "label" and row[column].get("xml:lang") == "en":
                    english.setdefault(local_id(row["item"]), name)
    for qid, name in english.items():
        names = catalogue.items[qid]["names"]
        names.remove(name)
        names.insert(0, name)
    if not catalogue.of_class(MODEL):
        raise SparqlError(f"{endpoint} returned no keyboard models.")
    kinds = {r["pid"]["value"]: r["example"] for r in rows(run_sparql(endpoint, PREFIXES + Q_KINDS))}
    for row in rows(run_sparql(endpoint, PREFIXES + Q_PROPERTIES)):
        pid = row["pid"]["value"]
        example = kinds.get(pid)
        if example and example.get("datatype") in NUMERIC_TYPES:
            catalogue.properties[pid] = {"label": row["label"]["value"],
                                         "description": row.get("description", {}).get("value", "")}
    return catalogue


# catalogue cache, refreshed every hour
_cache = {"catalogue": None, "time": 0.0}
_lock = threading.Lock()


def fetch_catalogue() -> Catalogue:
    problems = []
    for endpoint in CATALOGUE_ENDPOINTS:
        try:
            return load_catalogue(endpoint)
        except (EndpointUnavailable, SparqlError, KeyError, ValueError) as exc:
            problems.append(f"{endpoint}: {exc}")
    raise CatalogueUnavailable("No catalogue source answered. " + " | ".join(problems))


def get_catalogue() -> Catalogue:
    with _lock:
        age = time.monotonic() - _cache["time"]
        if _cache["catalogue"] is None or age > CATALOGUE_TTL:
            try:
                _cache["catalogue"], _cache["time"] = fetch_catalogue(), time.monotonic()
            except CatalogueUnavailable:
                if _cache["catalogue"] is None:
                    raise
                _cache["time"] = time.monotonic() - CATALOGUE_TTL + 60
        return _cache["catalogue"]


# prompt
RULES = """You extract facts about musical keyboards from a text (English or Danish) and return ONE JSON object, nothing else.

What to extract
- Only facts stated in the text. Leave out every property that is not mentioned. Missing information is not zero and not false: write 0 only when the text explicitly says that something is absent (for example "has no speakers").
- Only musical keyboards (pianos, synthesizers, organs, workstations, controllers). If the text is about anything else, or contains no keyboard facts, return {"models": [], "offers": []}.
- If the text speaks about several variants of one model family (for example "all three Xyz models"), give one entry for each matching catalogue item.
- A qualified value (for example a weight without batteries) is extracted as a plain number; drop the qualifier.
- Ignore facts that none of the listed properties can hold.

Units are implicit and never written: width, depth and height in mm; weight in kg; power consumption in W; price in DKK ("kr." means DKK). Convert other units (cm to mm). Danish numbers use "." for thousands and "," for decimals: "26.500 kr." is 26500 and "4,6 kg" is 4.6.
Count physical sockets, not audio channels: a stereo input is ONE socket; separate line-output sockets count once each and are not headphone sockets.

Models and the catalogue
- Link manufacturer, colour, seller and keyboard model to the catalogue below, by id (Q...), ONLY when you are sufficiently sure. Never invent an id. When unsure, use null.
- A model that is not in the catalogue has catalogue_id null; give its name in "label" without the manufacturer's name ("PSR-E483", not "Yamaha PSR-E483").
- Give the manufacturer and the colour of a model when the text implies them, even only through the model name ("Nord Stage" implies the manufacturer Nord).
- List every model the text mentions, also when it is only for sale. Put in "properties" only ids from the property list.

Offers
- An advertisement is an offer: it points to its model by key and may have a seller (catalogue only), a price, a url and a date (ISO yyyy-mm-dd; use the update date when one is stated). Use null for what the text does not give.

Output format
{"models": [{"key": "m1", "catalogue_id": "Q..." or null, "label": "..." or null, "manufacturer": "Q..." or null, "colour": "Q..." or null, "properties": {"P1": 61}}],
 "offers": [{"model": "m1", "seller": "Q..." or null, "price": 1000 or null, "url": "https://..." or null, "date": "2026-01-31" or null}]}"""

EXAMPLES = """Examples
Text: The Kurzweil K2700 has 88 keys and weighs 21 kg.
{"models": [{"key": "m1", "catalogue_id": "Q61", "label": null, "manufacturer": "Q59", "colour": null, "properties": {"P1": 88, "P7": 21}}], "offers": []}

Text: Sælger min Zenotron Orion 61 for 8.500 kr. 61 tangenter. https://example.com/orion-annonce
{"models": [{"key": "m1", "catalogue_id": null, "label": "Orion 61", "manufacturer": null, "colour": null, "properties": {"P1": 61}}], "offers": [{"model": "m1", "seller": null, "price": 8500, "url": "https://example.com/orion-annonce", "date": null}]}"""


def catalogue_text(catalogue: Catalogue) -> str:
    lines = ["Properties allowed in \"properties\" (all numbers):"]
    for pid in sorted(catalogue.properties, key=lambda p: int(p[1:])):
        info = catalogue.properties[pid]
        lines.append(f"{pid} | {info['label']} | {info['description']}".rstrip(" |"))
    for title, cls in (("Manufacturers", MANUFACTURER), ("Colours", COLOUR),
                       ("Sellers", SELLER), ("Keyboard models", MODEL)):
        lines.append(f"\n{title} (id | names):")
        for qid in catalogue.of_class(cls):
            names = catalogue.items[qid]["names"]
            also = f" (also: {', '.join(names[1:])})" if len(names) > 1 else ""
            lines.append(f"{qid} | {names[0] if names else qid}{also}")
    return "\n".join(lines)


def build_messages(text: str, catalogue: Catalogue) -> list[dict]:
    system = "\n\n".join([RULES, catalogue_text(catalogue), EXAMPLES])
    return [{"role": "system", "content": system},
            {"role": "user", "content": f"Text: {text}"}]


_llm_clients: dict = {}


# LLM call
def call_llm(messages: list[dict]) -> str:
    api_key, base_url = os.environ.get("CAMPUSAI_API_KEY"), os.environ.get("CAMPUSAI_API_URL")
    if not api_key or not base_url:
        raise LLMUnavailable("CAMPUSAI_API_KEY / CAMPUSAI_API_URL are not set.")
    try:
        client = _llm_clients.setdefault(
            (api_key, base_url), OpenAI(api_key=api_key, base_url=base_url, timeout=LLM_TIMEOUT, max_retries=0))
        response = client.chat.completions.create(
            model=os.environ.get("CAMPUSAI_MODEL", "cai"), temperature=0, messages=messages)
    except Exception as exc:
        raise LLMUnavailable(f"LLM call failed ({type(exc).__name__}).") from exc
    return response.choices[0].message.content or ""


def parse_json_reply(reply: str) -> dict:
    text = re.sub(r"```(?:json)?", "", reply)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end < start:
        raise ValueError("the reply contains no JSON object")
    data = json.loads(text[start:end + 1])
    if not isinstance(data, dict):
        raise ValueError("the JSON is not an object")
    return data


def ask_llm(text: str, catalogue: Catalogue) -> dict:
    messages = build_messages(text, catalogue)
    problem = ""
    for _ in range(MAX_ATTEMPTS):
        reply = call_llm(messages)
        try:
            return parse_json_reply(reply)
        except ValueError as exc:
            problem = str(exc)
            messages += [{"role": "assistant", "content": reply},
                         {"role": "user", "content": f"That is not valid JSON ({problem}). Return only the JSON object."}]
    raise ExtractionFailed(f"The language model did not return valid JSON after {MAX_ATTEMPTS} attempts ({problem}).")


# cleaning of the LLM output
def as_number(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, str):
        try:
            value = float(value.strip())
        except ValueError:
            return None
    if isinstance(value, (int, float)) and math.isfinite(value) and value >= 0:
        return int(value) if float(value).is_integer() else float(value)
    return None


def linked(catalogue: Catalogue, ident, cls: str):
    if not isinstance(ident, str):
        return None
    ident = ident.strip().removeprefix("kb:").upper()
    return ident if cls in catalogue.items.get(ident, {}).get("classes", set()) else None


def valid_url(value):
    if not isinstance(value, str):
        return None
    parts = urlsplit(value.strip())
    return value.strip() if parts.scheme in ("http", "https") and parts.netloc else None


def valid_date(value):
    try:
        return date.fromisoformat(value).isoformat() if isinstance(value, str) else None
    except ValueError:
        return None


def clean_extraction(raw: dict, catalogue: Catalogue) -> dict:
    models, by_key = [], {}
    for entry in raw.get("models") if isinstance(raw.get("models"), list) else []:
        if not isinstance(entry, dict):
            continue
        label = entry.get("label")
        properties = {}
        for pid, value in (entry.get("properties") if isinstance(entry.get("properties"), dict) else {}).items():
            number = as_number(value)
            if pid in catalogue.properties and number is not None:
                properties[pid] = number
        model = {"catalogue_id": linked(catalogue, entry.get("catalogue_id"), MODEL),
                 "label": label.strip()[:100] if isinstance(label, str) and label.strip() else None,
                 "manufacturer": linked(catalogue, entry.get("manufacturer"), MANUFACTURER),
                 "colour": linked(catalogue, entry.get("colour"), COLOUR),
                 "properties": properties}
        by_key[str(entry.get("key", len(models) + 1))] = len(models)
        models.append(model)
    offers = []
    for entry in raw.get("offers") if isinstance(raw.get("offers"), list) else []:
        if not isinstance(entry, dict):
            continue
        key = str(entry.get("model"))
        index = by_key.get(key, 0 if len(models) == 1 else None)
        if index is None:
            continue
        offers.append({"model": index, "seller": linked(catalogue, entry.get("seller"), SELLER),
                       "price": as_number(entry.get("price")), "url": valid_url(entry.get("url")),
                       "date": valid_date(entry.get("date"))})
    return {"models": models, "offers": offers}


# JSON-LD output
def build_jsonld(clean: dict) -> dict:
    base = f"{EXTRACTION_ROOT}{uuid.uuid4()}/"
    context = {"@base": base, "kb": KB, "kbt": KBT, "rdfs": RDFS}
    models, offers = clean["models"], clean["offers"]
    referenced = {offer["model"] for offer in offers}

    nodes: dict = {}
    node_of: dict = {}
    local_count = 0
    for index, model in enumerate(models):
        has_facts = bool(model["properties"] or model["colour"])
        if model["catalogue_id"]:
            node_id = f"kb:{model['catalogue_id']}"
            emit = has_facts
        else:
            emit = has_facts or index in referenced
            if emit:
                local_count += 1
                node_id = f"model/{local_count}"
        node_of[index] = node_id if (emit or model["catalogue_id"]) else None
        if not emit:
            continue
        statements = nodes.setdefault(node_id, {})
        if not model["catalogue_id"] and model["label"]:
            statements.setdefault("rdfs:label", {"@value": model["label"], "@language": "en"})
        statements.setdefault(f"kbt:{P_TYPE}", {"@id": f"kb:{MODEL}"})
        if model["manufacturer"]:
            statements.setdefault(f"kbt:{P_MANUFACTURER}", {"@id": f"kb:{model['manufacturer']}"})
        if model["colour"]:
            statements.setdefault(f"kbt:{P_COLOUR}", {"@id": f"kb:{model['colour']}"})
        for pid, number in model["properties"].items():
            statements.setdefault(f"kbt:{pid}", number)

    graph = [{"@id": node_id, **statements} for node_id, statements in nodes.items()]
    for number, offer in enumerate(offers, start=1):
        node = {"@id": f"offer/{number}", f"kbt:{P_TYPE}": {"@id": f"kb:{OFFER}"},
                f"kbt:{P_MODEL}": {"@id": node_of[offer["model"]]}}
        if offer["seller"]:
            node[f"kbt:{P_SELLER}"] = {"@id": f"kb:{offer['seller']}"}
        if offer["price"] is not None:
            node[f"kbt:{P_PRICE}"] = offer["price"]
        if offer["url"]:
            node[f"kbt:{P_URL}"] = {"@id": offer["url"]}
        if offer["date"]:
            node[f"kbt:{P_DATE}"] = {"@value": offer["date"], "@type": XSD_DATE}
        graph.append(node)
    return {"@context": context, "@graph": graph}


def extract(text: str) -> dict:
    catalogue = get_catalogue()
    return build_jsonld(clean_extraction(ask_llm(text, catalogue), catalogue))


class ExtractInput(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_TEXT)


@app.post("/v1/extract")
def extract_endpoint(payload: ExtractInput):
    try:
        return extract(payload.text)
    except CatalogueUnavailable as exc:
        raise HTTPException(status_code=503, detail=(
            f"{exc} Check your internet connection, or start the local QLever (`qlever start`)."))
    except (LLMUnavailable, ExtractionFailed) as exc:
        raise HTTPException(status_code=502, detail=str(exc))