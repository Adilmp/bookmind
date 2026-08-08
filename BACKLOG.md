# BACKLOG — park "cool but not now" ideas here

The rule that gets this project finished: if an idea isn't in *this week's* scope,
it goes here instead of derailing the build. Ship first, expand later.

## Next up (Week 2)
- [ ] LLM answer generation over retrieved chunks (grounded, with inline citations)
- [ ] Refusal guardrail: answer "not in this book" instead of hallucinating
- [ ] Dense embeddings (BGE-M3) + **hybrid** BM25 + dense retrieval; compare on eval set
- [ ] Concept-map generation: extract concept→relation→concept triples → interactive graph

## Week 3
- [ ] Gold Q&A set (~40–60 questions with reference passages)
- [ ] Eval metrics: Recall@k, MRR, faithfulness (RAGAS), citation accuracy, refusal correctness
- [ ] Baseline comparison: RAG vs. closed-book LLM (show hallucination drop)

## Week 4
- [ ] FastAPI service + Dockerfile + one-command run
- [ ] Streamlit / HF Space demo (Q&A + concept map)
- [ ] README results table + demo GIF + LinkedIn write-up

## Later / maybe
- [ ] Multi-book library
- [ ] Chapter/section-level summaries
- [ ] CI/CD (GitHub Actions runs eval on every push), monitoring dashboard
- [ ] Latency/cost measurement + managed-API-vs-self-hosted write-up
