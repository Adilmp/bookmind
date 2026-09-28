# BookMind

Ask questions across a book and get answers grounded in **cited passages**, with every
sentence checked against its evidence. Trace an idea through the book chapter by chapter,
and quiz yourself on a chapter with questions built from its own sentences. A
**faithfulness evaluation** harness measures how much the system hallucinates, and it all
runs on a free local model (Ollama) or Claude.

> Built as a from-scratch RAG project: the keyword retriever (BM25) is implemented by hand, not
> imported, and it is combined with embedding search by rank fusion that fits in ten lines, so every
> part is understood, not magic.

## Status

- [x] **Week 1 — Retrieval.** EPUB → clean citeable chunks → BM25 search → cited passages. ✅ *working*
- [x] **Week 2 — Answer generation.** Grounded, cited answers via Claude + refusal guardrail; extractive fallback when no API key. ✅ *working*
- [x] **Week 2b — Concept maps.** Built, then **removed**: the graphs (word co-occurrence, or LLM triples from a truncated prompt) gave readers nothing they could act on.
- [x] **Week 3 — Evaluation harness.** Retrieval metrics (Recall@k, MRR) + citation-accuracy checker, refusal correctness, and RAG-vs-closed-book hallucination comparison. ✅ *working*
- [x] **Week 4 — Deploy.** FastAPI service (`/search`, `/ask`), Streamlit demo, Dockerfile, and one-command run. ✅ *working*
- [x] **Week 2c — Hybrid retrieval.** BM25 + embeddings merged with Reciprocal Rank Fusion; measured against BM25 and embeddings alone on a larger test set (below). ✅ *working*
- [x] **Checked answers.** Each answer sentence is matched to its evidence and labelled ✅ / ⚠️ / ❌; thresholds calibrated on real answers (below). ✅ *working*
- [x] **Quiz me.** Study questions per chapter, grounded in exact quotes, with spaced review and optional AI grading. ✅ *working*
- [x] **Idea timeline.** Where an idea appears, chapter by chapter; counts checked against the original EPUB text for 8 ideas. ✅ *working*

## Evaluation results

Run `python src/evaluate.py`. The test set has 42 questions: 38 answerable (each labelled with
the chapter that answers it) and 4 off-topic ones the system must refuse.

**Retrieval** (does the right chapter appear in the top 5 passages?). BM25 needs nothing; the
other two need Ollama for embeddings:

| Questions | BM25 | Embeddings | **Hybrid (default)** |
|---|---|---|---|
| All 38: Recall@5 | 92% | 89% | **95%** |
| All 38: MRR | 0.703 | 0.763 | **0.796** |
| Everyday wording (19): MRR | 0.629 | 0.719 | **0.763** |
| The book's own terms (7): MRR | 0.929 | 0.857 | **1.000** |

| Answer quality | Result | Needs a model? |
|---|---|---|
| Citation accuracy | *(run with a model)* | yes |
| Refusal correctness (adversarial) | *(run with a model)* | yes |
| Hallucination rate: RAG vs closed-book | *(run with a model)* | yes |

The citation checker is deterministic (verifies each `[Chapter]` against real chapters) and runs even without a key.

## Quickstart

```bash
pip install -r requirements.txt
python src/ingest.py data/raw/your-book.epub       # -> data/chunks.jsonl
python src/search.py "how do I stop overthinking"  # -> top passages with chapter citations
python src/answer.py "how do I stop overthinking"  # -> grounded, cited answer (Claude key or local Ollama)
```

## Use a local model (Ollama) instead of Claude

Answers and the answer-level evaluation can run on a free local model. `src/llm.py`
is the only file that talks to a model: it uses Claude when `ANTHROPIC_API_KEY` is set and Ollama
otherwise (force one with `BOOKMIND_PROVIDER=anthropic|ollama`).

```bash
ollama pull qwen2.5:7b          # once
ollama serve                    # leave running
make api                        # answers now come from qwen2.5:7b
```

| Setting | Default | Meaning |
|---|---|---|
| `BOOKMIND_PROVIDER` | `auto` | `anthropic`, `ollama`, or `auto` (Claude if a key is set) |
| `BOOKMIND_OLLAMA_MODEL` | `qwen2.5:7b` | any model you've pulled |
| `BOOKMIND_OLLAMA_URL` | `http://127.0.0.1:11434` | where Ollama listens |
| `BOOKMIND_OLLAMA_TIMEOUT` | `600` | seconds to wait for a local answer |
| `BOOKMIND_RETRIEVAL` | `hybrid` | `hybrid`, `bm25` or `dense` (see "Retrieval") |
| `BOOKMIND_EMBED_MODEL` | `nomic-embed-text` | Ollama embedding model for search and answer checks (`ollama pull nomic-embed-text`) |

Measured on a laptop CPU with `qwen2.5:7b`: a cited answer took 27 s to 2 min and an off-topic
question was correctly refused ("I couldn't find this in the book."). `qwen2.5:0.5b` answered in 17 s but copied the passage instead of answering.
If Ollama isn't running, BookMind falls back to its offline modes as before.

## Retrieval: keyword, meaning, or both

BM25 matches words, so it misses a passage that says the same thing differently: "calling my
shots good or bad" versus the book's "letting go of judgment". Embeddings (`nomic-embed-text`
through Ollama) match meaning, but they drift towards passages that are merely on the same topic,
and they are worse at exact terms like "bounce-hit".

**Hybrid** runs both and merges the two rankings with Reciprocal Rank Fusion: each passage scores
`1/(60 + rank)` in each list, summed. Only ranks are used, so BM25's unbounded scores and cosine
similarity never have to be put on one scale. Each result reports which method ranked it and its
rank in each list.

What the numbers above do and don't show:

- **The test set was grown first.** The original 12 questions couldn't separate two retrievers,
  because one question moved Recall@5 by 8 points. The 26 new questions were written and
  committed before dense or hybrid retrieval was measured, so they weren't tuned to the new code.
  The RRF constant (60) is the standard value, not tuned here.
- **Hybrid never lost a question BM25 found.** It found one more, ranked the right chapter higher
  on 13 questions and lower on 7.
- **The gain is not yet proven.** A paired bootstrap puts the MRR difference over BM25 at +0.093,
  with a 95% interval from -0.03 to +0.22. That includes zero, so 38 questions can't rule out
  luck. Hybrid is the default because it was best or joint best on every slice and never worse
  on recall.
- **The labels are chapter-level:** a hit means a passage from the right chapter, not the exact
  passage.

Embedding all 326 chunks takes about 90 s on a laptop CPU. The vectors are cached in
`data/embeddings.json` (git-ignored) and rebuilt automatically when the chunks or the embedding
model change. If Ollama isn't running, search falls back to BM25, and both `/health` and every
result say so.

## Checking answers ("show me where it says that")

Every LLM answer from `/ask` comes back with a sentence-by-sentence check (`src/verify.py`).
For each sentence it finds the best-matching sentence (or pair of sentences) in the retrieved
passages and labels it ✅ supported, ⚠️ weak or ❌ unsupported. It also checks citations:
`[3]` must be a real passage number, and a chapter name must match one of the sources, so
`[9]` with five passages, or a mangled `[FOURING Trusting Self 2]`, is caught.

Similarity uses `nomic-embed-text` embeddings through Ollama when it's running (word overlap
otherwise). ✅ needs cosine ≥ 0.80, or ≥ 0.70 with at least half the words matching; ⚠️ is
≥ 0.70. These thresholds were set on 8 real `qwen2.5:7b` answers (21 sentences, labelled by
hand), the same sentences checked against another question's passages, and 8 invented claims:

| Group | ✅ | ⚠️ | ❌ |
|---|---|---|---|
| Real sentences judged supported (17) | 13 | 4 | 0 |
| Real sentences judged weak (4) | 0 | 4 | 0 |
| Checked against the wrong passages (21) | 0 | 9 | 12 |
| Invented claims (8) | 0 | 2 | 6 |

No unsupported sentence got ✅, so ✅ can be trusted; ⚠️ means "check it yourself". Limits:
similarity is not meaning (a sentence that reverses a passage can still look close), and the
calibration set is small and from one model.

## Quiz me

Pick a chapter; the model writes one question per passage (six passages spread across the
chapter) with a short answer and **the exact sentence that answers it**. `src/quiz.py` keeps a
question only if that quote is a complete sentence that really appears in the passage, and
gives a rejected passage one retry. Answer, reveal, and mark yourself; missed questions come
back the same day, remembered ones after 1, 3, 7 and 14 days. "Check my answer with AI" is
optional (about a minute on a CPU).

With `qwen2.5:7b` on a laptop CPU, a chapter took about 4 minutes, once. The first version of
the check accepted half-sentence "quotes", which produced questions that didn't match their
answers; requiring a complete sentence fixed that (5 of 6 passages gave a good question).
Questions are saved in `data/quiz/` and progress in `data/quiz_progress.json`, both
git-ignored: they're derived from the book's text.

## Run the service (Week 4)

BookMind ships as a small FastAPI service with a Streamlit demo on top. The index is
built once at startup and shared across requests.

```bash
make api          # FastAPI on :8000  (docs at http://localhost:8000/docs)
make ui           # Streamlit demo on :8501  (needs the API running)
make run          # both together
```

Endpoints:

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/health` | — | index status, chunk/chapter counts, retrieval mode (and why, if it fell back to BM25) |
| POST | `/search` | `{query, k}` | ranked passages with chapter citations, the method used and each passage's BM25 and embedding ranks |
| POST | `/ask` | `{query, k}` | grounded, cited answer (extractive fallback w/o key), plus `support`: every sentence checked against the passages |
| POST | `/timeline` | `{idea, snippets_per_chapter}` | where an idea appears, chapter by chapter, with highlighted sentences |
| GET | `/quiz/chapters` | — | chapters, whether questions exist, how many are due |
| POST | `/quiz/generate` | `{chapter, n}` | write study questions for a chapter (slow on a CPU; saved) |
| GET | `/quiz?chapter=…` | — | a chapter's questions and which are due |
| POST | `/quiz/review` | `{id, correct}` | record an answer (spaced review) |
| POST | `/quiz/grade` | `{id, answer}` | optional: the LLM grades a typed answer |

```bash
curl -s localhost:8000/ask -H 'content-type: application/json' \
  -d '{"query":"how do I stop overthinking?","k":5}' | jq .
```

### Docker

The image contains **only code** — the copyrighted corpus stays git-ignored and is
mounted at runtime.

```bash
make docker       # build
make docker-run   # run, mounting ./data and passing $ANTHROPIC_API_KEY
```

## How it works — component map

| Component | File | What it does |
|---|---|---|
| Ingest | `src/ingest.py` | Parses the EPUB in reading order, extracts clean paragraphs, splits into ~180-word overlapping chunks, tags each with its chapter (from the TOC). |
| Rank | `src/bm25.py` | BM25 implemented from scratch (TF saturation + length normalisation) — not imported. |
| Embed | `src/dense.py` | Embedding ("dense") search: cosine similarity over cached chunk vectors from Ollama. |
| Search | `src/search.py` | Hybrid search: BM25 and embedding rankings merged by Reciprocal Rank Fusion, falling back to BM25 without Ollama; each passage comes with a citation. |
| Answer | `src/answer.py` | Grounded, cited answer generation with a refusal guardrail; extractive fallback when no model is reachable. |
| Model | `src/llm.py` | The only file that calls a model: Claude with an API key, otherwise a local Ollama model. |
| Verify | `src/verify.py` | "Show me where it says that": labels each answer sentence supported / weak / unsupported with its best evidence, and checks that citations point at real sources. |
| Quiz | `src/quiz.py` | Writes study questions per chapter, keeping only those whose quote is a complete sentence of the passage; spaced review (Leitner boxes); optional LLM grading. |
| Timeline | `src/timeline.py` | Follows an idea (a word or exact phrase) through the book: mentions per chapter in reading order, skipping front/back matter, without double-counting the chunk overlap. No model: instant. |
| Evaluate | `src/evaluate.py` | Retrieval metrics (Recall@k, MRR) for BM25, embeddings and hybrid side by side, with a paired bootstrap interval; deterministic citation checker, refusal correctness, and RAG-vs-closed-book hallucination. |
| Serve | `src/api.py` | FastAPI service; builds the index once at startup and shares it across requests. |
| Demo | `src/ui.py` | Streamlit UI (thin HTTP client over the API): "Ask", "Idea timeline" and "Quiz me" tabs. |

## Data & copyright

The demo is developed against a personally-owned copy of *The Inner Game of Tennis*
(© W. Timothy Gallwey, 1974). **The book text and derived chunks are git-ignored** and never
committed. The public demo will use public-domain texts or a user-upload flow.
