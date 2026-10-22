import json
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

course_ids, course_titles, course_texts = [], [], []
obj_data, obj_texts = [], []

tfidf_courses = TfidfVectorizer(lowercase=True, ngram_range=(1,2))
tfidf_objs = TfidfVectorizer(lowercase=True, ngram_range=(1,2))
tfidf_course_matrix, tfidf_obj_matrix = None, None

dense_model = None
dense_course_matrix, dense_obj_matrix = None, None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global tfidf_course_matrix, tfidf_obj_matrix
    global dense_model, dense_course_matrix, dense_obj_matrix

    with open("dtu_courses.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            item = json.loads(line)
            cid = item["course_code"]
            title = item["title"]
            objs = item.get("learning_objectives", [])
            
            course_text = title + "\n" + "\n".join(objs)
            course_ids.append(cid)
            course_titles.append(title)
            course_texts.append(course_text)

            for obj in objs:
                obj_data.append({"course_id": cid, "title": title, "objective": obj})
                obj_texts.append(obj)

    tfidf_course_matrix = tfidf_courses.fit_transform(course_texts)
    if obj_texts:
        tfidf_obj_matrix = tfidf_objs.fit_transform(obj_texts)

    dense_model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')
    dense_course_matrix = dense_model.encode(course_texts, convert_to_numpy=True)
    if obj_texts:
        dense_obj_matrix = dense_model.encode(obj_texts, convert_to_numpy=True)

    yield

app = FastAPI(lifespan=lifespan)

def compute_similarity(query: str, mode: str, alpha: float, sparse_vec, sparse_mat, dense_mat):
    if mode in ["sparse", "hybrid"]:
        q_sparse = sparse_vec.transform([query])
        score_sparse = cosine_similarity(q_sparse, sparse_mat)[0]
    else:
        score_sparse = np.zeros(sparse_mat.shape[0])

    if mode in ["dense", "hybrid"]:
        q_dense = dense_model.encode([query], convert_to_numpy=True)
        score_dense = cosine_similarity(q_dense, dense_mat)[0]
    else:
        score_dense = np.zeros(dense_mat.shape[0])

    if mode == "hybrid":
        return alpha * score_dense + (1 - alpha) * score_sparse
    return score_dense if mode == "dense" else score_sparse

@app.get("/v1/search")
def search_courses(query: str, top_k: int = 10, mode: str = "hybrid", alpha: float = 0.5):
    scores = compute_similarity(query, mode, alpha, tfidf_courses, tfidf_course_matrix, dense_course_matrix)
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = [{"course_id": course_ids[i], "title": course_titles[i], "score": float(scores[i])} for i in top_indices]
    return {"query": query, "results": results, "mode": mode}

@app.get("/v1/courses/{course_id}/similar")
def similar_courses(course_id: str, top_k: int = 10, mode: str = "hybrid", alpha: float = 0.5):
    if course_id not in course_ids:
        raise HTTPException(status_code=404, detail="Course not found")

    idx = course_ids.index(course_id)
    scores = compute_similarity(course_texts[idx], mode, alpha, tfidf_courses, tfidf_course_matrix, dense_course_matrix)
    
    scores[idx] = -1.0
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = [{"course_id": course_ids[i], "title": course_titles[i], "score": float(scores[i])} for i in top_indices]
    return {"query_course_id": course_id, "results": results, "mode": mode, "top_k": top_k}

@app.get("/v1/objectives/search")
def search_objectives(query: str, top_k: int = 10, mode: str = "hybrid", alpha: float = 0.5):
    if not obj_texts:
        return {"query": query, "results": []}

    scores = compute_similarity(query, mode, alpha, tfidf_objs, tfidf_obj_matrix, dense_obj_matrix)
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = [{"course_id": obj_data[i]["course_id"], "title": obj_data[i]["title"], "objective": obj_data[i]["objective"], "score": float(scores[i])} for i in top_indices]
    return {"query": query, "results": results}

@app.get("/v1/health")
def health():
    return {"status": "ok", "index_sizes": {"courses": len(course_ids), "objectives": len(obj_data)}}