"""
api.py — Week 4: serve BookMind as a small HTTP service.

Wraps the pieces built in Weeks 1–3 behind a FastAPI app:
  GET  /health            -> is the index loaded? how many chunks / chapters?
  POST /search  {query,k} -> ranked passages with chapter citations (BM25)
  POST /ask     {query,k} -> grounded, cited answer (Claude or Ollama via llm.py, extractive fallback)
  POST /timeline {idea}  -> where an idea appears, chapter by chapter (no model, instant)

Design note: the BM25 index is built ONCE at startup and shared across requests
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
import timeline as timeline_mod
from search import BookSearch

app = FastAPI(
    title="BookMind",
    version="0.4.0",
    description="Ask questions across a book and get answers grounded in cited passages.",
)

# ---- shared, lazily-built index ---------------------------------------------

_state = {"search": None, "error": None}


def _get_search() -> BookSearch:
    """Return the shared BM25 index, building it on first use."""
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
    return {"status": "ok", "chunks": len(bs.chunks), "chapters": len(chapters)}


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
    except Exception as e:  # no key / Ollama down / network / timeout -> extractive fallback
        text = answer_mod._extractive_answer(req.query, hits)
        mode = f"extractive (LLM unavailable: {type(e).__name__})"
    return {"query": req.query, "answer": text, "mode": mode, "sources": hits}


@app.post("/timeline")
def idea_timeline(req: TimelineRequest):
    bs = _require_index()
    try:
        return timeline_mod.timeline(req.idea, bs.chunks, req.snippets_per_chapter)
    except ValueError as e:  # e.g. an idea with no letters or digits
        raise HTTPException(status_code=422, detail=str(e))


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
