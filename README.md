# DTU2ESCO: which DTU courses give which ESCO skills?

Project description for the final four-week project (NLP, LLMs and knowledge graphs).
Individual project. Based on the proposal "Match DTU courses with ESCO ontology", with a reduced and quantified scope.

## 1. Idea

ESCO is the European classification of occupations and skills. Each occupation has essential and optional skills. DTU course descriptions list learning objectives in free text. The project builds a Web service that links the two:

- which ESCO skills does a DTU course (partly) provide?
- which ESCO occupations is a course useful for?
- which courses should a student take to move towards a given occupation, given what is already completed and an ECTS budget?

The mapping is stored as a small knowledge graph (RDF) and queried with SPARQL. A language model is used to judge matches and to understand questions in natural language (English or Danish).

## 2. Scope (what is in, what is out)

In:
- DTU course database 2025/2026: 1,565 courses, 19 departments, 1,177 taught in English and 388 in Danish. Each course has a title, about 9.5 learning objectives on average and structured fields (ECTS, department, responsible, prerequisites). This file has no free "content" text, so matching relies on title, learning objectives and a few fields.
- ESCO v1.2.1 (latest on the ESCO download page), downloaded as CSV or RDF.
- A subset of occupations (engineering related) and three departments (01, 34 and 22, about 440 courses) for the full LLM-checked mapping.

Out (possibly later, if time allows): the remaining departments, ESCO qualifications, other years of courses, a frontend, authentication.

Languages: English first. Danish courses and Danish questions are an extension, tried at the end if the English version works. ESCO itself is multilingual, so the labels are available.

## 3. Method

Comparing every course to every skill with an LLM would need millions of calls. The pipeline narrows the candidates in three steps:

1. **Retrieval.** Embed course text (title plus learning objectives) and ESCO skill labels with the CampusAI embedding model (`cai-embedding`). Keep the top 15 to 20 skills per course. A TF-IDF baseline is built on the same data for comparison.
2. **Reranking.** Rerank the candidates with `cai-reranker`, keep the top 10.
3. **LLM judgement.** One call per course: the model receives the course text and its 10 candidate skills and returns, for each skill, a verdict (`covered`, `partly`, `not covered`) with a short quote from the course as evidence. The score is mapped to a confidence in [0, 1]. Output is validated in code (skill ids must be in the candidate list, scores bounded), as in the course exercises.

Budget (estimate): for about 440 courses this is about 440 LLM calls, instead of millions. The estimate will be checked on a first run and reported.

Occupation coverage of a course: the share of an occupation's essential skills that the course covers, weighted by confidence.

Planner: choose courses that cover as many essential skills of a target occupation as possible, weighted by confidence, within a total ECTS limit, ignoring skills already covered by completed courses. A greedy algorithm comes first. An exact solution as a 0-1 integer program with `scipy.optimize.milp` is added only if time allows, to compare with the greedy result.

## 4. Knowledge graph

- ESCO is loaded as RDF (SKOS). Occupations, skills and the occupation-skill relations are queried with SPARQL.
- DTU courses are converted to RDF (course, title, ECTS, department, learning objectives).
- Each match is a small resource linking a course to a skill, with a confidence, the evidence quote and the method that produced it. Example:

```turtle
@prefix ex:   <https://example.org/dtu2esco/> .
@prefix esco: <http://data.europa.eu/esco/model#> .

ex:match/42578-recsys a ex:CourseSkillMatch ;
    ex:course   ex:course/42578 ;
    ex:skill    <http://data.europa.eu/esco/skill/505e4ef3-7ce4-437d-b7b4-5c608f71c258> ;
    ex:confidence 0.87 ;
    ex:evidence "recommender systems" ;
    ex:method   "embedding+rerank+llm" .
```

- Store: rdflib in memory. QLever (as in the text-to-query exercise) is a possible later step if the graph becomes too slow to query.

## 5. API (base path `/api/v1`)

| Method and path | Purpose | Status |
|---|---|---|
| `GET /courses/{course_number}` | course data and matched skills | core |
| `GET /courses/{course_number}/skills` | skills matched to the course | core |
| `GET /occupations/search?q=...` | link a text to an ESCO occupation (entity linking) | core |
| `GET /occupations/{occupation_id}/courses` | courses useful for an occupation | core |
| `POST /planner/occupation` | course plan for a target occupation | core |
| `GET /skills/{skill_id}/courses` | courses that provide a skill | if time allows |
| `GET /courses/{course_number}/occupations` | occupations a course is relevant for | if time allows |
| `POST /query` | question in natural language, routed to the endpoints above | if time allows |

Example: `GET /courses/42578`

```json
{
  "course_number": "42578",
  "title": "Advanced Business Analytics",
  "ects": 5,
  "matched_skills": [
    {
      "skill_uri": "http://data.europa.eu/esco/skill/505e4ef3-7ce4-437d-b7b4-5c608f71c258",
      "label": "build recommender systems",
      "confidence": 0.87,
      "evidence": "recommender systems"
    }
  ]
}
```

Example: `POST /planner/occupation`

```json
{
  "occupation_uri": "http://data.europa.eu/esco/occupation/258e46f9-0075-4a2e-adae-1ff0477e0f30",
  "completed_courses": ["02450"],
  "max_ects": 30,
  "strategy": "maximize_skill_coverage"
}
```

```json
{
  "target_occupation": "industrial mobile devices software developer",
  "skill_gap": ["..."],
  "recommended_courses": [{"course_number": "...", "ects": 5, "covered_skills": ["..."]}],
  "coverage_after_plan": 0.78
}
```

Example: `POST /query`

```json
{
  "question": "What courses should I take to become a data scientist?",
  "completed_courses": ["02451"],
  "constraints": {"max_ects": 25}
}
```

The service finds the intent (here occupation planning), links "data scientist" to an ESCO occupation, calls the planner and returns the plan with a short explanation. Unknown or ambiguous occupations return a clear error with the closest candidates instead of a guess. This endpoint is the first one dropped if the project falls behind.

## 6. Tools

- Python, FastAPI, Pydantic, `requests`/`openai` client for CampusAI (chat model `cai`, `cai-embedding`, `cai-reranker`).
- scikit-learn (TF-IDF baseline), NumPy, SciPy (`milp`).
- rdflib and QLever (SPARQL).
- Docker (one container for the service, the graph can be a second one).
- pytest for tests.

## 7. Tests

Unit tests that run without network or API key (LLM and embeddings replaced by fakes):
- validation of the LLM output (invented skill ids, out-of-range scores, broken JSON);
- conversion of courses and ESCO data to RDF;
- planner (ECTS limit respected, completed courses ignored, greedy never beats the exact solution);
- API tests with `TestClient` (status codes, response shape, 404 and 422 cases);
- optional tests with the real LLM, skipped when no key is set.

## 8. Evaluation

There is no ground truth, so one is built:
- a gold set of about 30 course-skill pairs annotated by hand (positive and negative);
- metrics: precision and recall at k for retrieval, precision and recall of the final matches against the gold set;
- comparison of three variants: TF-IDF only, embeddings with reranking, embeddings with reranking and LLM judgement;
- calibration of the confidence threshold on the gold set;
- if Danish is tried, results for English against Danish courses.

Known limits, stated in the final report: a learning objective is not proof that a skill is acquired, ESCO skill labels are short and ambiguous, and the gold set is small and annotated by one person.

## 9. Plan (four weeks)

1. Week 1: load ESCO and the courses, build the graph, TF-IDF baseline, first endpoints, Docker.
2. Week 2: embeddings, reranking, LLM judgement with validation, gold set, first evaluation.
3. Week 3: occupation search and endpoints, greedy planner.
4. Week 4: tests, final evaluation, README and report. Extras only if everything else works: exact planner, `/query`, Danish.

## 10. Data and repository

The course data comes from the DTU course database provided in the course, so the repository is private. ESCO is downloaded from the ESCO website (free of charge, download link sent by e-mail) and is not committed to the repository. A script downloads or loads it from a local path.
