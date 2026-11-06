Text-to-graph
=============

In this exercise, you will implement a web service for extracting a knowledge graph from text about musical keyboards.

The [Keyboards Wikibase](https://keyboards.wikibase.cloud/) contains information about keyboard models, manufacturers, colours and offers. You will use its vocabulary and existing entities when constructing your graph.

Task
----

Implement a FastAPI service that accepts text describing one or more keyboard models or offers and returns the extracted facts as JSON-LD.

The service must not modify Wikibase.

A frontend will be provided for evaluating the service.

Extraction rules
----------------

- Extract only facts supported by the supplied text, apart from entity linking against the supplied catalogue.
- Omit unmentioned properties. Do not interpret missing information as zero or false.
- Express dimensions in **mm**, mass in **kg**, power consumption in **W**, and prices in **DKK**. These units are implicit in the output.
- Assume that `kr.` denotes DKK in the supplied Danish advertisements. Currency conversion is outside the exercise.
- Count physical sockets rather than audio channels. For these examples, P20 counts dedicated line-output sockets separately from headphone sockets recorded with P13.
- Extract qualified numerical values, such as weight excluding batteries, but omit their qualifiers.
- Ignore facts that cannot be represented using the supplied vocabulary.

### Entity linking

Reuse existing identifiers for manufacturers, colours, sellers and keyboard models when a sufficiently certain match exists. Do not invent Wikibase identifiers.

- If a manufacturer, colour or seller cannot be linked confidently to the catalogue, omit that relation.
- If a keyboard model cannot be matched, create a local model node with an `rdfs:label`.
- For advertisements, extract the offer and link it to the keyboard model. If the model cannot be matched, create a local model node and link the offer to it.
- Extracting additional model specifications from advertisements is optional.

### Local identifiers

Application code must generate a UUID for each extraction and use it in the JSON-LD `@base`, for example:

```text
https://example.org/keyboards/extraction/57c94ee6-73bf-4a18-b8c1-4b29e62a307d/
```

Local identifiers such as `model/1`, `model/2` and `offer/1` must be unique within that extraction.

The UUIDs in the examples are illustrative; your output need not reproduce them.

`https://example.org/keyboards/extraction/` must be the prefix.

API
---

Implement the following endpoint:

```http
POST /v1/extract
```

The request body must be a JSON object containing the input text:

```json
{
  "text": "Text describing keyboard models or offers."
}
```

Return a JSON object in JSON-LD format, rather than a string containing JSON. Use a shared `@context` and an `@graph` array. If no relevant facts are found, return an empty `@graph` array.

Vocabulary and catalogue
------------------------

The following prefixes are used:

```sparql
PREFIX kb: <https://keyboards.wikibase.cloud/entity/>
PREFIX kbt: <https://keyboards.wikibase.cloud/prop/direct/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX schema: <http://schema.org/>
```

### Offers

The following example illustrates the properties of an offer:

| Property | Example value | Meaning |
|---|---|---|
| P11 | Q42 | Instance of offer |
| P23 | Q38 | Seller: Digitalpiano.com |
| P24 | Q40 | Keyboard model on sale: Korg Pa700 |
| P25 | 9620 | Price, implicitly in DKK |
| P5 | `https://www.digitalpiano.com/dk/korg-pa-700.html` | Offer URL |
| P26 | `2025-09-19` | Advertisement date; use the update date when stated |

Omit optional information, such as the seller or URL, when it is not available in the text.

### Candidate entities

Some existing items and their classes can be obtained using the following SPARQL query. The selected classes are manufacturer, keyboard model, country and colour.

```sparql
PREFIX kb: <https://keyboards.wikibase.cloud/entity/>
PREFIX kbt: <https://keyboards.wikibase.cloud/prop/direct/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?item ?itemLabel ?class ?classLabel
WHERE {
  VALUES ?class { kb:Q11 kb:Q14 kb:Q28 kb:Q79 }

  ?item kbt:P11 ?class .

  OPTIONAL {
    ?item rdfs:label ?itemLabel .
    FILTER (LANG(?itemLabel) = "en")
  }

  OPTIONAL {
    ?class rdfs:label ?classLabel .
    FILTER (LANG(?classLabel) = "en")
  }
}
```

This query is a starting point; it does not provide a separate catalogue of sellers.

### Keyboard properties

The following SPARQL query returns descriptions of properties currently used on keyboard models:

```sparql
PREFIX kb: <https://keyboards.wikibase.cloud/entity/>
PREFIX kbt: <https://keyboards.wikibase.cloud/prop/direct/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX schema: <http://schema.org/>

SELECT
  (CONCAT(
    ?pid,
    " - ",
    ?propertyLabel,
    " - ",
    COALESCE(?propertyDescription, "")
  ) AS ?p)
WHERE {
  ?property_item wikibase:directClaim ?property ;
                 rdfs:label ?propertyLabel .

  FILTER (LANG(?propertyLabel) = "en")

  FILTER EXISTS {
    ?keyboard kbt:P11 kb:Q14 ;
              ?property [] .
  }

  BIND(STRAFTER(STR(?property), "/prop/direct/") AS ?pid)

  OPTIONAL {
    ?property_item schema:description ?propertyDescription .
    FILTER (LANG(?propertyDescription) = "en")
  }
}
ORDER BY ?pid
```

A similar query can be used for offers by replacing `kb:Q14` with `kb:Q42` in the `FILTER EXISTS` block.

Example 1: An offer
-------------------

### Input text

```text
Jeg sælger mit røde Nord Stage 4 med 88 tangenter, da jeg ikke får det brugt nok. Det står i rigtig pæn stand med enkelte små brugsspor og har kun været brugt hjemme.

Der er separate sektioner til piano, orgel og synthesizer samt masser af muligheder for at justere lyde og effekter. Et godt instrument, men mere end jeg har brug for.

Kan afhentes og prøves i Odense efter aftale. Strømkabel medfølger. Prisen er uden stativ.

Sælges for 26.500 kr.

Annonce opdateret: 18. oktober 2026.
```

### Example output

The keyboard model is linked to the existing item `kb:Q23`, Nord Stage 4 88. Additional model specifications are omitted in this example.

```json
{
  "@context": {
    "@base": "https://example.org/keyboards/extraction/57c94ee6-73bf-4a18-b8c1-4b29e62a307d/",
    "kb": "https://keyboards.wikibase.cloud/entity/",
    "kbt": "https://keyboards.wikibase.cloud/prop/direct/"
  },
  "@graph": [
    {
      "@id": "offer/1",
      "kbt:P11": {"@id": "kb:Q42"},
      "kbt:P24": {"@id": "kb:Q23"},
      "kbt:P25": 26500,
      "kbt:P26": {
        "@value": "2026-10-18",
        "@type": "http://www.w3.org/2001/XMLSchema#date"
      }
    }
  ]
}
```

Example 2: Two keyboard models
------------------------------

### Input text

```text
The Yamaha PSR-E483 and PSR-E583 look like close relatives on paper, and choosing between these two keyboard models involves more than counting keys. Both have 61 touch-sensitive keys and can sound up to 64 notes simultaneously. The E483 offers 860 voices, while the E583 increases that selection to 890.

Neither model saves you any desk space over the other. Each measures 99.2 cm across, 40.4 cm from front to back and 13.6 cm high. The PSR-E483 is slightly lighter at 7.0 kg; the PSR-E583 weighs 7.3 kg. Those weights exclude batteries. If you are arranging delivery, avoid confusing the instrument dimensions with the packaging: both boxes measure 1068 × 500 × 216 mm in width, depth and height.

The sound systems are a more noticeable difference. Each keyboard contains two speakers with a diameter of 12 cm, but the E483 has two 6 W amplifier channels, compared with two 12 W channels on the E583. Despite that difference, Yamaha lists power consumption as 13 W for each model, using the PA-150C adaptor for the E483 and the PA-300C for the E583.

The physical audio connections are the same on both instruments: one headphone socket, two separate line-output sockets labelled L/L+R and R, one stereo mini-jack AUX input, and one microphone input. The stereo AUX connection is a single socket, not two inputs.

Both models also provide a USB Type-A connection for a flash drive and a USB Type-C connection to a computer. The computer connection supports a stereo USB audio interface at 44.1 kHz and 16 bits. The E583 adds a modulation wheel and support for an expression pedal, whereas the E483 has neither. Both can run on six AA batteries.
```

### Example output

This example assumes that neither model has been matched to an existing catalogue item. Yamaha is linked to the existing manufacturer item `kb:Q2`.

```json
{
  "@context": {
    "@base": "https://example.org/keyboards/extraction/8a3f6c21-9d47-4b82-a5e0-63c719d2f408/",
    "kb": "https://keyboards.wikibase.cloud/entity/",
    "kbt": "https://keyboards.wikibase.cloud/prop/direct/",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#"
  },
  "@graph": [
    {
      "@id": "model/1",
      "rdfs:label": {
        "@value": "PSR-E483",
        "@language": "en"
      },
      "kbt:P1": 61,
      "kbt:P2": 992,
      "kbt:P3": 404,
      "kbt:P4": 136,
      "kbt:P6": {"@id": "kb:Q2"},
      "kbt:P7": 7.0,
      "kbt:P8": 64,
      "kbt:P10": 2,
      "kbt:P11": {"@id": "kb:Q14"},
      "kbt:P13": 1,
      "kbt:P16": 13,
      "kbt:P20": 2,
      "kbt:P21": 1,
      "kbt:P22": 1
    },
    {
      "@id": "model/2",
      "rdfs:label": {
        "@value": "PSR-E583",
        "@language": "en"
      },
      "kbt:P1": 61,
      "kbt:P2": 992,
      "kbt:P3": 404,
      "kbt:P4": 136,
      "kbt:P6": {"@id": "kb:Q2"},
      "kbt:P7": 7.3,
      "kbt:P8": 64,
      "kbt:P10": 2,
      "kbt:P11": {"@id": "kb:Q14"},
      "kbt:P13": 1,
      "kbt:P16": 13,
      "kbt:P20": 2,
      "kbt:P21": 1,
      "kbt:P22": 1
    }
  ]
}
```

## LLM API configuration

An OpenAI-compatible LLM API is available through CampusAI. Load its configuration from environment variables:

```python
import os

from dotenv import load_dotenv

load_dotenv()

CAMPUSAI_API_KEY = os.getenv("CAMPUSAI_API_KEY")
CAMPUSAI_MODEL = os.getenv("CAMPUSAI_MODEL")
CAMPUSAI_EMBED_MODEL = os.getenv("CAMPUSAI_EMBED_MODEL")
CAMPUSAI_API_URL = os.getenv("CAMPUSAI_API_URL")
```

Use the configured CampusAI endpoint for LLM requests. The embedding model is available if needed for entity linking.


Optional extension: Storage and querying
----------------------------------------

Extend the service with endpoints for storing JSON-LD graphs and querying the stored data using SPARQL. Demonstrate a query that combines keyboard specifications from one document with an offer from another. The service must continue to leave the public Keyboards Wikibase unchanged.

Example question:

"Which keyboard models weigh less than 8 kg and have an offer below DKK 3,000?"


Requirements & Resources
------------------------
- Python
- FastAPI
- Requests
- QLever endpoint
- CampusAI LLM API
- Text-to-graph
  - Knowledge graphs - Chapter "Text-to-graph"
- Entity linking
  - Natural language processing - Chapter "Entity linking"
- Prompt engineering
  - Natural language processing - Chapter "Prompt engineering".
- ReAct (for optional advanced)
  - Natural language processing - Section 8.5 ReAct prompting.
  - Scientific literature 
    - SPINACH: SPARQL-Based Information Navigation for Challenging Real-World Questions
    - GRASP: Generic Reasoning And SPARQL Generation Across Knowledge Graphs


Deliverables
------------
A zipped repository in root (git archive -o latest.zip HEAD)
