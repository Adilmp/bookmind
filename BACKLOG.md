# BACKLOG — park "cool but not now" ideas here

The rule that gets this project finished: if an idea isn't in *this week's* scope,
it goes here instead of derailing the build. Ship first, expand later.

## Next up (Week 2)
- [ ] LLM answer generation over retrieved chunks (grounded, with inline citations)
- [ ] Refusal guardrail: answer "not in this book" instead of hallucinating
- [x] Dense embeddings + **hybrid** BM25 + dense retrieval (RRF); compared on a 38-question eval set (nomic-embed-text, not BGE-M3: it already runs in Ollama for answer checks)
- [ ] Grow the eval set past ~100 questions, with passage-level labels, so the hybrid gain can be confirmed or ruled out
- [x] ~~Concept-map generation~~: built, then dropped (not useful to readers)

## Week 3
- [ ] Gold Q&A set (~40–60 questions with reference passages)
- [ ] Eval metrics: Recall@k, MRR, faithfulness (RAGAS), citation accuracy, refusal correctness
- [ ] Baseline comparison: RAG vs. closed-book LLM (show hallucination drop)

## Week 4
- [ ] FastAPI service + Dockerfile + one-command run
- [ ] Streamlit / HF Space demo (Q&A)
- [ ] README results table + demo GIF + LinkedIn write-up

## Later / maybe
- [ ] Multi-book library
- [ ] Chapter/section-level summaries
- [ ] CI/CD (GitHub Actions runs eval on every push), monitoring dashboard
- [ ] Latency/cost measurement + managed-API-vs-self-hosted write-up
