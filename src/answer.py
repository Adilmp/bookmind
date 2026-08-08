"""
answer.py — Week 2: turn retrieved passages into a grounded, cited ANSWER.

Pipeline:  question -> BM25 retrieval -> grounded prompt -> Claude -> cited answer
Guardrail: the model is instructed to answer ONLY from the retrieved passages and
to say it couldn't find the answer rather than inventing one (the hallucination
guardrail that Week 3's evaluation will measure).

Credentials: uses a zero-arg Anthropic() client, which resolves ANTHROPIC_API_KEY
(or an `ant auth login` profile) from the environment. If no credentials are
available, it falls back to an EXTRACTIVE answer (stitched from the top passages)
so the pipeline still runs end-to-end.

Run:  python src/answer.py "how do I stop overthinking on the court?"
"""
import os
import sys
from search import BookSearch

MODEL = os.environ.get("BOOKMIND_MODEL", "claude-opus-5")

SYSTEM_PROMPT = """You are a careful study assistant for a single book. Follow these rules exactly:
- Answer ONLY using the numbered passages provided. Do not use outside knowledge.
- If the passages do not contain the answer, reply exactly: "I couldn't find this in the book."
- Cite the chapter for each claim in square brackets, e.g. [Quieting Self 1].
- Be concise: two or three sentences unless more is clearly needed.
- Do not invent quotations, chapter names, or page numbers."""


def build_context(hits):
    """Format retrieved chunks as a numbered, citeable context block."""
    lines = []
    for n, h in enumerate(hits, 1):
        lines.append(f"[{n}] (chapter: {h['chapter']})\n{h['text']}")
    return "\n\n".join(lines)


def _extractive_answer(query, hits):
    """No-LLM fallback: surface the most relevant passage with its citation."""
    if not hits:
        return "I couldn't find this in the book."
    top = hits[0]
    snippet = top["text"]
    if len(snippet) > 400:
        snippet = snippet[:400].rsplit(" ", 1)[0] + " …"
    return (
        f"(extractive fallback — no LLM credentials found)\n"
        f"Most relevant passage [{top['chapter']}]:\n{snippet}"
    )


def _llm_answer(query, hits):
    """Grounded generation with Claude. Raises if the SDK/credentials are unavailable."""
    import anthropic

    client = anthropic.Anthropic()  # resolves ANTHROPIC_API_KEY or an `ant` profile
    context = build_context(hits)
    user_msg = (
        f"Passages from the book:\n\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Answer using only the passages above, citing chapters in [brackets]."
    )
    resp = client.messages.create(
        model=MODEL,
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )
    return "".join(b.text for b in resp.content if b.type == "text").strip()


def answer(query, k=5):
    """Retrieve, then answer with citations (LLM if available, else extractive)."""
    bs = BookSearch()
    hits = bs.search(query, k=k)
    try:
        text = _llm_answer(query, hits)
        mode = f"LLM ({MODEL})"
    except Exception as e:  # missing SDK, no credentials, network, etc.
        text = _extractive_answer(query, hits)
        mode = f"extractive (LLM unavailable: {type(e).__name__})"
    return {"answer": text, "mode": mode, "sources": hits}


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) or "how do I stop overthinking on the court?"
    result = answer(query)
    print(f'\n  Q: "{query}"   [{result["mode"]}]\n')
    print(f"  A: {result['answer']}\n")
    print("  Sources:")
    for n, s in enumerate(result["sources"], 1):
        print(f"    [{n}] « {s['chapter']} »  (chunk #{s['chunk_id']}, score {s['score']})")
