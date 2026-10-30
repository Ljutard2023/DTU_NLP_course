"""Text-to-SPARQL for the Keyboards Wikibase.

question -> link known entities -> LLM writes SPARQL -> check it -> QLever runs it -> JSON
"""
import json
import os
import re
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv(os.path.expanduser("~/.env"))

app = FastAPI(title="Keyboards text-to-SPARQL API")

QLEVER_URL = os.environ.get("QLEVER_URL", "http://localhost:7070/sparql")
QLEVER_TIMEOUT = 30  # seconds
LLM_TIMEOUT = float(os.environ.get("LLM_TIMEOUT", "30"))  # seconds, per call
MAX_ATTEMPTS = 3  # first try + 2 repairs
MAX_ROWS = 1000

# teacher's test questions are the Wikibase items ABOVE Q86. We never put them in the prompt: using the test as a study sheet would prove nothing
HELD_OUT_FROM = 87
HIDDEN_PROPERTIES = {"P27", "P31"}

KB_ENTITY = "https://keyboards.wikibase.cloud/entity/"
KB_DIRECT = "https://keyboards.wikibase.cloud/prop/direct/"
XSD = "http://www.w3.org/2001/XMLSchema#"
PREFIXES = {
    "kb": KB_ENTITY,
    "kbt": KB_DIRECT,
    "xsd": XSD,
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "skos": "http://www.w3.org/2004/02/skos/core#",
}
NUMERIC_TYPES = {XSD + t for t in ("decimal", "integer", "int", "long", "double", "float", "nonNegativeInteger")}


class QLeverUnavailable(Exception):
    """QLever does not answer at all (not started, wrong port...)."""


class SparqlError(Exception):
    """QLever answered, but refused or failed the query."""


class LLMUnavailable(Exception):
    """The language model could not be used (no key, no VPN, timeout...)."""


class QueryGenerationError(Exception):
    """No usable SPARQL query could be produced for this question."""

#QLever

def run_sparql(query: str) -> dict:
    """Send one query to QLever and return its answer as SPARQL-results JSON."""
    try:
        response = requests.get(
            QLEVER_URL,
            params={"query": query, "format": "json"},
            headers={"Accept": "application/sparql-results+json"},
            timeout=QLEVER_TIMEOUT,
        )
    except requests.ConnectionError as exc:
        raise QLeverUnavailable(f"Cannot connect to QLever at {QLEVER_URL}.") from exc
    except requests.Timeout as exc:
        raise SparqlError(f"QLever did not answer within {QLEVER_TIMEOUT} s.") from exc
    if response.status_code != 200:
        raise SparqlError(response.text[:500])
    return response.json()


def bindings(data: dict) -> list[dict]:
    return data["results"]["bindings"]


def local_id(binding: dict) -> str:
    """'https://keyboards.wikibase.cloud/entity/Q1' -> 'Q1'."""
    return binding["value"].rsplit("/", 1)[-1]


def value_kind(binding: dict) -> str:
    if binding["type"] == "uri":
        return "item" if binding["value"].startswith(KB_ENTITY) else "link"
    datatype = binding.get("datatype", "")
    if datatype in NUMERIC_TYPES:
        return "number"
    return "date" if datatype.endswith("dateTime") else "text"


#text helpers

TRANSLITERATION = str.maketrans({"ø": "o", "Ø": "O", "æ": "ae", "Æ": "AE", "ß": "ss"})


def normalize(text: str) -> str:
    """'Hvad er bredden på Yamaha P-150?' -> 'hvad er bredden pa yamaha p 150'."""
    text = unicodedata.normalize("NFKD", text.translate(TRANSLITERATION))
    text = "".join(c for c in text if not unicodedata.combining(c)).lower()
    return " ".join(re.findall(r"[a-z0-9]+", text))


def danish_definite(label: str) -> list[str]:
    """'bredde' -> ['bredden', 'breddet'], 'vægt' -> ['vægten', 'vægtet']: Danish 'the ...'."""
    ends_in_e = label.endswith("e")
    return [label + ("n" if ends_in_e else "en"), label + ("t" if ends_in_e else "et")]

#catalogue

CATALOG_PREFIXES = (
    "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n"
    "PREFIX skos: <http://www.w3.org/2004/02/skos/core#>\n"
    "PREFIX wikibase: <http://wikiba.se/ontology#>\n"
    f"PREFIX kb: <{KB_ENTITY}>\nPREFIX kbt: <{KB_DIRECT}>\n"
)
Q_PROPERTY_LABELS = "SELECT ?p ?label WHERE { ?p wikibase:directClaim ?direct ; rdfs:label ?label . }"
Q_PROPERTY_KINDS = (
    "SELECT ?direct (SAMPLE(?o) AS ?example) WHERE { ?s ?direct ?o . "
    f'FILTER(STRSTARTS(STR(?direct), "{KB_DIRECT}")) }} GROUP BY ?direct'
)
Q_ITEM_LABELS = (
    "SELECT ?i ?label WHERE { ?i rdfs:label ?label . "
    f'FILTER(STRSTARTS(STR(?i), "{KB_ENTITY}Q")) }}'
)
Q_ITEM_ALTS = (
    "SELECT ?i ?label WHERE { ?i skos:altLabel ?label . "
    f'FILTER(STRSTARTS(STR(?i), "{KB_ENTITY}Q")) }}'
)
Q_ENUMS = (
    "SELECT DISTINCT ?role ?v WHERE { VALUES (?role ?p) "
    '{ ("class" kbt:P11) ("country" kbt:P19) ("manufacturer" kbt:P6) } ?s ?p ?v . }'
)
Q_EXAMPLES = "SELECT ?item ?question ?sparql WHERE { ?item kbt:P11 kb:Q43 ; kbt:P27 ?question ; kbt:P31 ?sparql . }"


@dataclass
class Catalog:
    """Everything the service knows about the graph, read once from QLever."""
    labels: dict = field(default_factory=dict)      # "Q1" -> [(text, lang, is_alt)]
    kinds: dict = field(default_factory=dict)       # "P2" -> "number"
    enums: dict = field(default_factory=dict)       # "country" -> ["Q26", ...]
    examples: list = field(default_factory=list)    # [{"id", "en", "da", "sparql"}]
    item_index: dict = field(default_factory=dict)      # normalised label -> {"Q1"}
    property_index: dict = field(default_factory=dict)  # normalised label -> {"P2"}

    def display_label(self, ident: str) -> str:
        names = self.labels.get(ident, [])
        for text, lang, is_alt in names:
            if lang == "en" and not is_alt:
                return text
        return names[0][0] if names else ident

    def make_index(self, prefix: str) -> dict:
        index: dict = {}
        for ident, names in self.labels.items():
            if not ident.startswith(prefix):
                continue
            for text, lang, is_alt in names:
                forms = [text]
                if prefix == "P" and lang == "da":
                    forms += danish_definite(text)
                for form in forms:
                    if normalize(form):
                        index.setdefault(normalize(form), set()).add(ident)
        return index


def query_catalog(sparql: str) -> list[dict]:
    return bindings(run_sparql(CATALOG_PREFIXES + sparql))


@lru_cache(maxsize=1)
def get_catalog() -> Catalog:
    """Load the catalogue on first use, then keep it in memory."""
    catalog = Catalog()

    def add(ident, binding, is_alt):
        catalog.labels.setdefault(ident, []).append(
            (binding["value"], binding.get("xml:lang", ""), is_alt))

    for row in query_catalog(Q_PROPERTY_LABELS):
        add(local_id(row["p"]), row["label"], False)
    for row in query_catalog(Q_ITEM_LABELS):
        add(local_id(row["i"]), row["label"], False)
    for row in query_catalog(Q_ITEM_ALTS):
        add(local_id(row["i"]), row["label"], True)
    for row in query_catalog(Q_PROPERTY_KINDS):
        catalog.kinds[local_id(row["direct"])] = value_kind(row["example"])
    for row in query_catalog(Q_ENUMS):
        catalog.enums.setdefault(row["role"]["value"], []).append(local_id(row["v"]))

    by_item: dict = {}
    for row in query_catalog(Q_EXAMPLES):
        entry = by_item.setdefault(local_id(row["item"]), {"en": [], "da": [], "sparql": row["sparql"]["value"]})
        lang = row["question"].get("xml:lang")
        if lang in ("en", "da"):
            entry[lang].append(row["question"]["value"])
    for ident, entry in by_item.items():
        number = int(ident[1:])
        if number < HELD_OUT_FROM and entry["en"]:
            catalog.examples.append({
                "id": ident, "en": entry["en"][0],
                "da": entry["da"][0] if entry["da"] else "", "sparql": entry["sparql"],
            })
    catalog.examples.sort(key=lambda e: int(e["id"][1:]))

    catalog.item_index = catalog.make_index("Q")
    catalog.property_index = catalog.make_index("P")
    return catalog


#entity linking

def link(text: str, index: dict) -> list[tuple[str, set]]:
    """Find known labels in the question: longest match first, left to right.
    'Yamaha P-150' wins over 'Yamaha' alone. Returns [(phrase, {ids})]."""
    tokens = normalize(text).split()
    longest = max((len(key.split()) for key in index), default=1)
    found, i = [], 0
    while i < len(tokens):
        for n in range(min(longest, len(tokens) - i), 0, -1):
            phrase = " ".join(tokens[i:i + n])
            if phrase in index:
                found.append((phrase, index[phrase]))
                i += n
                break
        else:
            i += 1
    return found


#prompt

RULES = """You translate questions about musical keyboards (English or Danish) into SPARQL for the Keyboards Wikibase.

Rules:
- Reply with ONE read-only SPARQL query (SELECT or ASK) and nothing else.
- Use only the identifiers listed below: kb:Q... for items, kbt:P... for properties.
- Select only the variables the question asks for. Do not add label columns.
- Width, depth and height are in millimetres.
- Numbers are stored as xsd:decimal: write "88"^^xsd:decimal, or compare with xsd:integer(?x).
- To find items that LACK a property, use FILTER NOT EXISTS."""


def schema_text(catalog: Catalog) -> str:
    lines = ["Properties (direct claims):"]
    pids = sorted((p for p in catalog.kinds if p not in HIDDEN_PROPERTIES), key=lambda p: int(p[1:]))
    for pid in pids:
        names = ", ".join(dict.fromkeys(text for text, _, _ in catalog.labels.get(pid, [])))
        lines.append(f"kbt:{pid} | {names} | {catalog.kinds[pid]}")
    titles = {"class": "Classes (values of kbt:P11)", "country": "Countries (values of kbt:P19)",
              "manufacturer": "Manufacturers (values of kbt:P6)"}
    for role, title in titles.items():
        members = sorted(catalog.enums.get(role, []), key=lambda q: int(q[1:]))
        lines.append(f"{title}: " + "; ".join(f"kb:{q} {catalog.display_label(q)}" for q in members))
    return "\n".join(lines)


def examples_text(catalog: Catalog) -> str:
    blocks = []
    for ex in catalog.examples:
        danish = f"   (Danish: {ex['da']})" if ex["da"] else ""
        blocks.append(f"Question: {ex['en']}{danish}\nSPARQL: {' '.join(ex['sparql'].split())}")
    return "Examples:\n" + "\n\n".join(blocks)


def build_messages(question: str, catalog: Catalog, items: list, props: list) -> list[dict]:
    system = "\n\n".join([RULES, schema_text(catalog), examples_text(catalog)])
    hints = []
    for phrase, ids in items:
        hints.append(f'- "{phrase}" may be: ' + ", ".join(f"kb:{i} ({catalog.display_label(i)})" for i in sorted(ids)))
    for phrase, ids in props:
        hints.append(f'- "{phrase}" may be: ' + ", ".join(f"kbt:{i} ({catalog.display_label(i)})" for i in sorted(ids)))
    hint_text = "Possible matches found in the question (verify them):\n" + "\n".join(hints) + "\n\n" if hints else ""
    return [{"role": "system", "content": system},
            {"role": "user", "content": f"{hint_text}Question: {question}\nSPARQL:"}]


# LLM

@lru_cache(maxsize=1)
def get_llm_client(api_key: str, base_url: str) -> OpenAI:
    return OpenAI(api_key=api_key, base_url=base_url, timeout=LLM_TIMEOUT, max_retries=0)


def call_llm(messages: list[dict]) -> str:
    api_key = os.environ.get("CAMPUSAI_API_KEY")
    base_url = os.environ.get("CAMPUSAI_API_URL")
    if not api_key or not base_url:
        raise LLMUnavailable("CAMPUSAI_API_KEY / CAMPUSAI_API_URL are not set.")
    try:
        response = get_llm_client(api_key, base_url).chat.completions.create(
            model=os.environ.get("CAMPUSAI_MODEL", "cai-mini"), temperature=0, messages=messages)
    except Exception as exc:
        raise LLMUnavailable(f"LLM call failed ({type(exc).__name__}).") from exc
    return response.choices[0].message.content or ""


# checking the LLM's SPARQL

ITEM_REF = re.compile(r"(?:\bkb:|entity/)(Q\d+)\b")
PROP_REF = re.compile(r"(?:\bkbt:|direct/)(P\d+)\b")
TOKENS_TO_IGNORE = re.compile(r"""<[^>\s]*>|"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'|#[^\n]*""")
FORBIDDEN = re.compile(r"\b(INSERT|DELETE|LOAD|CLEAR|DROP|CREATE|COPY|MOVE|ADD|SERVICE)\b", re.I)


def extract_sparql(reply: str) -> str:
    match = re.search(r"```(?:sparql)?\s*(.*?)```", reply, re.S | re.I)
    return (match.group(1) if match else reply).strip()


def ensure_prefixes(query: str) -> str:
    missing = [f"PREFIX {name}: <{uri}>" for name, uri in PREFIXES.items()
               if re.search(rf"\b{name}:", query) and not re.search(rf"\bPREFIX\s+{name}\s*:", query, re.I)]
    return "\n".join(missing + [query]) if missing else query


def check_query(query: str, catalog: Catalog) -> str | None:
    """Return a description of the problem, or None if the query may be run."""
    skeleton = TOKENS_TO_IGNORE.sub(" ", query)  # no IRIs, strings or comments left
    skeleton = re.sub(r"\bPREFIX\s+[\w-]*:", " ", skeleton, flags=re.I)
    skeleton = re.sub(r"[?$]\w+", " ", skeleton)  # no variable names left
    if not re.match(r"\s*(SELECT|ASK)\b", skeleton, re.I):
        return "Only SELECT or ASK queries are allowed."
    if FORBIDDEN.search(skeleton):
        return "Update operations and SERVICE are not allowed."
    unknown = [f"kb:{i}" for i in dict.fromkeys(ITEM_REF.findall(query)) if i not in catalog.labels]
    unknown += [f"kbt:{p}" for p in dict.fromkeys(PROP_REF.findall(query)) if p not in catalog.kinds]
    if unknown:
        return f"Unknown identifiers: {', '.join(unknown)}. Use only the identifiers listed in the schema."
    return None


def generate_and_run(question: str, catalog: Catalog, items: list, props: list) -> tuple[str, dict]:
    """Ask the LLM for a query; on a problem, show it the error and ask again."""
    messages = build_messages(question, catalog, items, props)
    problem = "no attempt made"
    for _ in range(MAX_ATTEMPTS):
        reply = call_llm(messages)
        query = ensure_prefixes(extract_sparql(reply))
        problem = check_query(query, catalog)
        if problem is None:
            try:
                return query, run_sparql(query)
            except SparqlError as exc:
                problem = f"QLever rejected the query: {exc}"
        messages += [{"role": "assistant", "content": reply},
                     {"role": "user", "content": f"That query cannot be used: {problem}\nReturn a corrected SPARQL query only."}]
    raise QueryGenerationError(f"No valid SPARQL query after {MAX_ATTEMPTS} attempts. Last problem: {problem}")


# fallback without the LLM

def template_query(items: list, props: list, catalog: Catalog) -> str | None:
    """The 'baseline' of the assignment: only for 'what is the <property> of <item>'."""
    item_ids = {i for _, ids in items for i in ids}
    prop_ids = {p for _, ids in props for p in ids}
    if len(item_ids) != 1 or len(prop_ids) != 1:
        return None
    item, prop = next(iter(item_ids)), next(iter(prop_ids))
    variable = re.sub(r"\W+", "_", catalog.display_label(prop).lower())
    return ensure_prefixes(f"SELECT ?{variable} WHERE {{ kb:{item} kbt:{prop} ?{variable} . }}")


# ---------------------------------------------------------------- the answer

def cell(binding: dict | None):
    if binding is None:
        return None
    if binding["type"] == "uri":
        return binding["value"]
    datatype = binding.get("datatype", "")
    if datatype in NUMERIC_TYPES:
        number = float(binding["value"])
        return int(number) if number.is_integer() else number
    if datatype.endswith("#boolean"):
        return binding["value"] == "true"
    return binding["value"]


def rows_from(data: dict) -> list[dict]:
    if "boolean" in data:  # ASK query
        return [{"boolean": data["boolean"]}]
    columns = data["head"]["vars"]
    return [{c: cell(row.get(c)) for c in columns} for row in bindings(data)[:MAX_ROWS]]


def text_to_query(text: str) -> dict:
    catalog = get_catalog()
    items = link(text, catalog.item_index)
    props = link(text, catalog.property_index)
    method = "llm"
    try:
        query, data = generate_and_run(text, catalog, items, props)
    except LLMUnavailable as exc:
        query = template_query(items, props, catalog)
        if query is None:
            raise QueryGenerationError(
                f"{exc} The question is too complex for the rule-based fallback. "
                "Check the VPN, the CAMPUSAI_* variables, or try again.") from exc
        try:
            data = run_sparql(query)
        except SparqlError as sparql_error:
            raise QueryGenerationError(f"Fallback query failed: {sparql_error}") from sparql_error
        method = "template-fallback"
    item_labels = [catalog.display_label(i) for i in dict.fromkeys(ITEM_REF.findall(query))]
    return {
        "query": text,
        "entities": item_labels,
        "items_as_strings": item_labels,
        "properties_as_strings": [catalog.display_label(p) for p in dict.fromkeys(PROP_REF.findall(query))],
        "sparql": query,
        "results": rows_from(data),
        "method": method,
    }


class QueryInput(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000)


@app.post("/v1/query")
def query_endpoint(payload: QueryInput):
    # A plain `def` (not `async def`): FastAPI runs it in a worker thread, so
    # the blocking calls below do not freeze other simultaneous requests.
    try:
        return text_to_query(payload.text)
    except QLeverUnavailable as exc:
        raise HTTPException(status_code=503, detail=(
            f"{exc} Start it with `qlever start` in the folder that holds the Qleverfile, "
            "then check that the address answers."))
    except QueryGenerationError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
