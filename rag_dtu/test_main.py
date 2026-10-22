import os
from unittest.mock import patch

import numpy as np
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import main
from main import (
    MAX_CHARS_PER_COURSE,
    app,
    build_messages,
    clean_query,
    course_snippet,
    find_course_codes,
    reciprocal_rank_fusion,
    search_sparse,
)

client = TestClient(app)


def codes_of(query: str, top_k: int = 5) -> list[str]:
    return [r["course_code"] for r in search_sparse(query, top_k)]


# --- Small building blocks ------------------------------------------------

def test_course_code_without_leading_zero_is_recognised():
    assert find_course_codes("What is 2451?") == ["02451"]
    assert find_course_codes("difference between 02451 and 2452") == ["02451", "02452"]


def test_number_that_is_not_a_course_code_is_ignored():
    assert find_course_codes("room 99999") == []


def test_clean_query_drops_question_words():
    assert clean_query("How many ECTS points is Tue Herlau's course?") == "tue herlau"


def test_clean_query_strips_danish_letters():
    # Regression test: o-with-stroke is not decomposed by unicodedata and
    # used to turn "Bjørn" into the two garbage tokens "bj" and "rn".
    assert clean_query("Bjørn Sand Jensen") == "bjorn sand jensen"
    assert clean_query("Morten Mørup og Søren Hauberg") == "morten morup soren hauberg"
    assert clean_query("Ærø Åbenrå") == "aero abenra"


def test_ranker_with_zero_scores_votes_for_nobody():
    # Regression test: a ranker that finds nothing must not inject noise.
    fused = reciprocal_rank_fusion([np.zeros(len(main.COURSES))])
    assert fused.max() == 0


# --- Retrieval quality on the real data (no network needed) ---------------

def test_finds_course_of_named_teacher():
    assert codes_of("How many ECTS points is Tue Herlau's course?")[0] == "02465"
    assert codes_of("Which course is Hiba Nassar involved in?")[0] == "02462"


def test_finds_all_mri_courses():
    expected = {"22476", "22481", "22506", "22507"}
    assert expected <= set(codes_of("Are there any courses about MRI?"))


def test_survives_misspelled_teacher_name():
    # The assignment text itself spells it "Konvalenka"; the data says "Konvalinka".
    assert "02464" in codes_of("Does Ivana Konvalenka teach a course together with another teacher?")


def test_survives_typos_and_danish():
    assert "02462" in codes_of("Underviser Hibar Nassa i signler ogdta?")


def test_course_code_question_puts_that_course_first():
    assert codes_of("How many ECTS is the 02476?")[0] == "02476"
    assert codes_of("What is 2451?")[0] == "02451"


def test_unknown_words_give_no_results():
    assert search_sparse("the of and", 5) == []


# --- Prompt construction ---------------------------------------------------

def test_prompt_contains_question_and_key_facts():
    course = main.COURSE_BY_CODE["02465"]
    messages = build_messages("How many ECTS?", [course])
    user = messages[1]["content"]
    assert "How many ECTS?" in user
    assert "02465 Introduction to reinforcement learning and control" in user
    assert "Tue Herlau" in user
    assert "ECTS points: 5" in user


def test_snippet_never_exceeds_size_limit():
    assert all(len(course_snippet(c)) <= MAX_CHARS_PER_COURSE for c in main.COURSES)


# --- Endpoints ---------------------------------------------------------------

def test_search_endpoint_shape():
    response = client.get("/v1/search", params={"query": "MRI", "top_k": 5})
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "MRI" and data["mode"] == "sparse"
    assert len(data["results"]) <= 5
    assert {"course_code", "title", "score"} <= set(data["results"][0])


def test_search_rejects_bad_parameters():
    assert client.get("/v1/search").status_code == 422
    assert client.get("/v1/search", params={"query": "MRI", "top_k": 0}).status_code == 422
    assert client.get("/v1/search", params={"query": "MRI", "mode": "dense"}).status_code == 422


def test_ask_passes_retrieved_courses_to_the_llm():
    with patch.object(main, "generate_answer", return_value="mocked answer") as fake:
        response = client.get("/v1/ask", params={"query": "Which course is Hiba Nassar involved in?"})
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "mocked answer"
    assert data["retrieved_courses"][0]["course_code"] == "02462"
    courses_sent_to_llm = fake.call_args.args[1]
    assert courses_sent_to_llm[0]["course_code"] == "02462"


def test_ask_does_not_call_llm_when_nothing_was_retrieved():
    with patch.object(main, "generate_answer") as fake:
        response = client.get("/v1/ask", params={"query": "the of and"})
    assert response.status_code == 200
    assert response.json()["retrieved_courses"] == []
    assert "don't know" in response.json()["answer"]
    fake.assert_not_called()


def test_ask_reports_llm_failure_as_502():
    failure = HTTPException(status_code=502, detail="LLM call failed (TimeoutError).")
    with patch.object(main, "generate_answer", side_effect=failure):
        response = client.get("/v1/ask", params={"query": "Hiba Nassar"})
    assert response.status_code == 502


# --- Real LLM calls: need CAMPUSAI_API_KEY and the DTU network (VPN) ---------

needs_llm = pytest.mark.skipif(
    not os.environ.get("CAMPUSAI_API_KEY"),
    reason="Requires a real CAMPUSAI_API_KEY and DTU network access.",
)


@needs_llm
def test_real_answer_ects_of_herlau_course():
    response = client.get("/v1/ask", params={"query": "How many ECTS points is Tue Herlau's course?"})
    assert response.status_code == 200, response.text
    assert "5" in response.json()["answer"]


@needs_llm
def test_real_answer_hiba_nassar_course():
    response = client.get("/v1/ask", params={"query": "Which course is Hiba Nassar involved in?"})
    assert response.status_code == 200, response.text
    assert "02462" in response.json()["answer"]
