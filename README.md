# BookMind

Ask questions across a book and get answers grounded in **cited passages** — plus (soon)
an auto-generated visual **concept map** of its ideas, and a **faithfulness evaluation**
harness that measures how much the system hallucinates.

> Built as a from-scratch RAG project: the retriever (BM25) is implemented by hand, not imported,
> so every part is understood, not magic.

## Status

- [x] **Week 1 — Retrieval.** EPUB → clean citeable chunks → BM25 search → cited passages. ✅ *working*
- [x] **Week 2 — Answer generation.** Grounded, cited answers via Claude + refusal guardrail; extractive fallback when no API key. ✅ *working*
- [x] **Week 2b — Concept maps.** Extract concepts + relations from a chapter → JSON / Mermaid / SVG; LLM path for labeled edges, offline fallback. ✅ *working*
- [x] **Week 3 — Evaluation harness.** Retrieval metrics (Recall@k, MRR) + citation-accuracy checker, refusal correctness, and RAG-vs-closed-book hallucination comparison. ✅ *working*
- [ ] Week 2c — dense/hybrid retrieval (improve against the eval numbers)
- [ ] Week 4 — Deploy (FastAPI + Docker + demo)

## Evaluation results

Run `python src/evaluate.py` (16-question gold set: 12 answerable + 4 adversarial).

| Metric | Result | Needs key? |
|---|---|---|
| Retrieval Recall@5 (BM25) | **83%** | no |
| Retrieval MRR | **0.688** | no |
| Citation accuracy | *(run with key)* | yes |
| Refusal correctness (adversarial) | *(run with key)* | yes |
| Hallucination rate: RAG vs closed-book | *(run with key)* | yes |

The citation checker is deterministic (verifies each `[Chapter]` against real chapters) and runs even without a key.

## Quickstart

```bash
pip install -r requirements.txt
python src/ingest.py data/raw/your-book.epub       # -> data/chunks.jsonl
python src/search.py "how do I stop overthinking"  # -> top passages with chapter citations
python src/answer.py "how do I stop overthinking"  # -> grounded, cited answer (needs ANTHROPIC_API_KEY)
```

## How it works (Week 1)

| Step | File | What it does |
|---|---|---|
| Ingest | `src/ingest.py` | Parses the EPUB in reading order, extracts clean paragraphs, splits into ~180-word overlapping chunks, tags each with its chapter (from the TOC). |
| Rank | `src/bm25.py` | BM25 implemented from scratch (TF saturation + length normalisation). |
| Search | `src/search.py` | Builds the index and returns the top passages for a query, each with a citation. |

## Data & copyright

The demo is developed against a personally-owned copy of *The Inner Game of Tennis*
(© W. Timothy Gallwey, 1974). **The book text and derived chunks are git-ignored** and never
committed. The public demo will use public-domain texts or a user-upload flow.
