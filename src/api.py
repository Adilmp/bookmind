"""
api.py — Week 4: serve BookMind as a small HTTP service.

Wraps the pieces built in Weeks 1–3 behind a FastAPI app:
  GET  /health            -> is the index loaded? how many chunks / chapters?
  POST /search  {query,k} -> ranked passages with chapter citations (hybrid: BM25 + embeddings)
  POST /ask     {query,k} -> grounded, cited answer (Claude or Ollama via llm.py, extractive fallback),
                             with every sentence checked against the passages (verify.py)
  POST /timeline {idea}  -> where an idea appears, chapter by chapter (no model, instant)
  GET  /quiz/chapters     -> chapters, whether questions exist, how many are due
  POST /quiz/generate {chapter, n} -> make study questions for a chapter (slow on a CPU)
  GET  /quiz?chapter=...  -> a chapter's questions and which are due for review
  POST /quiz/review {id, correct} -> record an answer (spaced review)
  POST /quiz/grade {id, answer}   -> optional: the LLM grades a typed answer

Design note: the search index (BM25, plus cached embeddings for hybrid search) is built ONCE at startup and shared across requests
(the CLI in answer.py rebuilds it per call — fine for a script, wasteful for a server).
If chunks.jsonl is missing, the service still boots and reports the problem via
/health and a clear 503, rather than crashing on import.

Run:  uvicorn api:app --host 0.0.0.0 --port 8000   (from the src/ directory)
"""
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import answer as answer_mod
import llm
import quiz as quiz_mod
import timeline as timeline_mod
import verify
from search import BookSearch

app = FastAPI(
    title="BookMind",
    version="0.4.0",
    description="Ask questions across a book and get answers grounded in cited passages.",
)

# ---- shared, lazily-built index ---------------------------------------------

_state = {"search": None, "error": None}


def _get_search() -> BookSearch:
    """Return the shared search index, building it on first use. The first run ever
    embeds the whole book (about 90 s on a CPU); later runs read the cache."""
    if _state["search"] is None:
        _state["search"] = BookSearch()  # raises if chunks.jsonl is missing
    return _state["search"]


@app.on_event("startup")
def _warm_index():
    """Build the index at boot so the first request isn't slow. Never crash the
    server if the corpus is missing — surface it through /health instead."""
    try:
        _get_search()
    except Exception as e:  # missing chunks.jsonl, corrupt data, etc.
        _state["error"] = f"{type(e).__name__}: {e}"


# ---- request models ----------------------------------------------------------

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    k: int = Field(5, ge=1, le=20)


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1)
    k: int = Field(5, ge=1, le=20)


class QuizGenerateRequest(BaseModel):
    chapter: str = Field(..., min_length=1)
    n: int = Field(6, ge=1, le=12)


class QuizReviewRequest(BaseModel):
    id: str = Field(..., min_length=1)
    correct: bool


class QuizGradeRequest(BaseModel):
    id: str = Field(..., min_length=1)
    answer: str = Field(..., min_length=1, max_length=2000)


class TimelineRequest(BaseModel):
    idea: str = Field(..., min_length=1, max_length=100)
    snippets_per_chapter: int = Field(2, ge=1, le=5)


# ---- endpoints ---------------------------------------------------------------

@app.get("/health")
def health():
    if _state["error"]:
        return {"status": "degraded", "detail": _state["error"], "chunks": 0}
    bs = _get_search()
    chapters = sorted({c["chapter"] for c in bs.chunks})
    retrieval = bs.mode if bs.mode == "bm25" or bs.dense is not None else "bm25 (fallback)"
    return {"status": "ok", "chunks": len(bs.chunks), "chapters": len(chapters),
            "retrieval": retrieval, "retrieval_error": bs.dense_error}


@app.post("/search")
def search(req: SearchRequest):
    bs = _require_index()
    return {"query": req.query, "results": bs.search(req.query, k=req.k)}


@app.post("/ask")
def ask(req: AskRequest):
    bs = _require_index()
    hits = bs.search(req.query, k=req.k)
    try:
        text = answer_mod._llm_answer(req.query, hits)
        mode = f"LLM ({llm.provider()}: {llm.model_name()})"
        support = verify.verify(text, hits)  # "show me where it says that"
    except Exception as e:  # no key / Ollama down / network / timeout -> extractive fallback
        text = answer_mod._extractive_answer(req.query, hits)
        mode = f"extractive (LLM unavailable: {type(e).__name__})"
        support = None  # an extractive answer is a quote, so there's nothing to check
    return {"query": req.query, "answer": text, "mode": mode, "sources": hits, "support": support}


@app.post("/timeline")
def idea_timeline(req: TimelineRequest):
    bs = _require_index()
    try:
        return timeline_mod.timeline(req.idea, bs.chunks, req.snippets_per_chapter)
    except ValueError as e:  # e.g. an idea with no letters or digits
        raise HTTPException(status_code=422, detail=str(e))


@app.get("/quiz/chapters")
def quiz_chapters():
    bs = _require_index()
    progress = quiz_mod.load_progress()
    out = []
    for chapter in quiz_mod.chapters(bs.chunks):
        saved = quiz_mod.load_quiz(chapter)
        questions = saved["questions"] if saved else []
        out.append({"chapter": chapter, "generated": saved is not None,
                    "questions": len(questions), "due": len(quiz_mod.due(questions, progress))})
    return out


@app.post("/quiz/generate")
def quiz_generate(req: QuizGenerateRequest):
    bs = _require_index()
    if req.chapter not in quiz_mod.chapters(bs.chunks):
        raise HTTPException(status_code=404, detail=f"unknown chapter: {req.chapter}")
    try:
        return quiz_mod.generate(req.chapter, bs.chunks, n=req.n)
    except Exception as e:  # no model reachable, timeout, ...
        raise HTTPException(status_code=503, detail=f"could not generate questions ({type(e).__name__}: {e})")


@app.get("/quiz")
def quiz_get(chapter: str):
    saved = quiz_mod.load_quiz(chapter)
    if saved is None:
        raise HTTPException(status_code=404, detail="no questions yet for this chapter: generate them first")
    progress = quiz_mod.load_progress()
    return {**saved,
            "due": [q["id"] for q in quiz_mod.due(saved["questions"], progress)],
            "progress": {q["id"]: progress.get(q["id"]) for q in saved["questions"]}}


@app.post("/quiz/review")
def quiz_review(req: QuizReviewRequest):
    _find_question(req.id)  # 404 for an unknown id
    progress = quiz_mod.load_progress()
    entry = quiz_mod.review(progress, req.id, req.correct)
    quiz_mod.save_progress(progress)
    return entry


@app.post("/quiz/grade")
def quiz_grade(req: QuizGradeRequest):
    question = _find_question(req.id)
    try:
        return quiz_mod.grade(question, req.answer)
    except Exception as e:  # no model, bad JSON, unexpected verdict
        raise HTTPException(status_code=503, detail=f"could not grade ({type(e).__name__}: {e})")


def _find_question(question_id):
    """Look a question up across the saved quizzes; 404 if it doesn't exist."""
    for chapter in quiz_mod.chapters(_require_index().chunks):
        saved = quiz_mod.load_quiz(chapter)
        for q in (saved or {}).get("questions", []):
            if q["id"] == question_id:
                return q
    raise HTTPException(status_code=404, detail=f"unknown question: {question_id}")


def _require_index() -> BookSearch:
    """Return the index or fail with a clear 503 if the corpus never loaded."""
    if _state["error"]:
        raise HTTPException(
            status_code=503,
            detail=f"Corpus not loaded ({_state['error']}). "
                   f"Run `python src/ingest.py <book.epub>` to build data/chunks.jsonl.",
        )
    try:
        return _get_search()
    except Exception as e:
        _state["error"] = f"{type(e).__name__}: {e}"
        raise HTTPException(status_code=503, detail=_state["error"])
