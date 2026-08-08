"""
evaluate.py — Week 3: the evaluation harness (BookMind's differentiator).

Two tiers, so it always produces something useful:
  1. Retrieval metrics — Recall@k and MRR over the gold set. Deterministic, needs
     no credentials, produces REAL numbers today.
  2. Answer metrics (when an LLM key is present) — citation accuracy, refusal
     correctness on unanswerable questions, and a RAG-vs-closed-book hallucination
     comparison. These are the metrics that mirror LLM-evaluation work.

The citation checker is deterministic and is unit-demonstrated even with no key.

Run:  python src/evaluate.py
"""
import json
import os
import re
from search import BookSearch, load_chunks

GOLD = os.path.join(os.path.dirname(__file__), "..", "evaluation", "gold.jsonl")
REFUSAL = "couldn't find this in the book"


def load_gold(path=GOLD):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------- deterministic helpers ----------

def _normalize_chapter(name):
    """Drop a leading ordinal word (THREE, FOUR, ...) and lowercase."""
    return re.sub(r"^(one|two|three|four|five|six|seven|eight|nine|ten)\s+",
                  "", name.strip().lower())


def chapter_labels(chunks):
    return sorted({_normalize_chapter(c["chapter"]) for c in chunks})


def extract_citations(answer):
    """Pull [Chapter] citations out of an answer string."""
    return [c.strip() for c in re.findall(r"\[([^\]]+)\]", answer)]


def citation_is_valid(citation, labels):
    """A citation is valid if it matches a real chapter (either direction)."""
    c = citation.lower()
    return any(c in lab or lab in c for lab in labels)


def refused(answer):
    return REFUSAL in answer.lower()


# ---------- tier 1: retrieval metrics (no credentials) ----------

def retrieval_metrics(gold, k=5):
    bs = BookSearch()
    answerable = [g for g in gold if g["answerable"]]
    hits_at_k, reciprocal_ranks = 0, []
    for g in answerable:
        results = bs.search(g["q"], k=k)
        rank = None
        for i, r in enumerate(results):
            if g["chapter"] in _normalize_chapter(r["chapter"]):
                rank = i + 1
                break
        if rank:
            hits_at_k += 1
            reciprocal_ranks.append(1 / rank)
        else:
            reciprocal_ranks.append(0.0)
    n = len(answerable)
    return {
        "n": n,
        "recall_at_k": hits_at_k / n,
        "mrr": sum(reciprocal_ranks) / n,
        "k": k,
    }


# ---------- tier 2: answer metrics (needs an LLM key) ----------

def llm_available():
    """True if credentials are present. (This SDK defers the auth error to
    request time, so construction alone can't tell us — check the environment.)"""
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


def _closed_book_answer(query):
    """LLM with NO retrieval — the baseline RAG should beat on hallucination."""
    import anthropic
    client = anthropic.Anthropic()
    model = os.environ.get("BOOKMIND_MODEL", "claude-opus-5")
    resp = client.messages.create(
        model=model, max_tokens=512,
        system=('Answer only from "The Inner Game of Tennis" by W. Timothy Gallwey. '
                f'If it is not covered in that book, reply exactly: "{REFUSAL}"'),
        messages=[{"role": "user", "content": query}],
    )
    return "".join(b.text for b in resp.content if b.type == "text").strip()


def answer_metrics(gold):
    from answer import answer as rag_answer
    labels = chapter_labels(load_chunks())
    answerable = [g for g in gold if g["answerable"]]
    unanswerable = [g for g in gold if not g["answerable"]]

    total_cites, valid_cites, false_refusals = 0, 0, 0
    for g in answerable:
        text = rag_answer(g["q"])["answer"]
        if refused(text):
            false_refusals += 1
        for cit in extract_citations(text):
            total_cites += 1
            valid_cites += citation_is_valid(cit, labels)

    rag_correct_refusals, closed_correct_refusals = 0, 0
    for g in unanswerable:
        if refused(rag_answer(g["q"])["answer"]):
            rag_correct_refusals += 1
        if refused(_closed_book_answer(g["q"])):
            closed_correct_refusals += 1

    nu = len(unanswerable)
    return {
        "citation_accuracy": (valid_cites / total_cites) if total_cites else None,
        "citations_checked": total_cites,
        "false_refusal_rate": false_refusals / len(answerable),
        "rag_refusal_correctness": rag_correct_refusals / nu,
        "rag_hallucination_rate": 1 - rag_correct_refusals / nu,
        "closed_book_hallucination_rate": 1 - closed_correct_refusals / nu,
    }


def _demo_citation_checker():
    """Prove the deterministic checker even with no LLM key."""
    labels = chapter_labels(load_chunks())
    sample = "Trust the body to play [Trusting Self 2] and quiet the mind [Chapter 99 Made Up]."
    print("  Citation checker demo (deterministic):")
    for cit in extract_citations(sample):
        ok = citation_is_valid(cit, labels)
        print(f"    [{cit}] -> {'valid' if ok else 'FABRICATED'}")


if __name__ == "__main__":
    gold = load_gold()
    print(f"\n  Gold set: {len(gold)} questions "
          f"({sum(g['answerable'] for g in gold)} answerable, "
          f"{sum(not g['answerable'] for g in gold)} adversarial)\n")

    rm = retrieval_metrics(gold)
    print("  RETRIEVAL (BM25, deterministic):")
    print(f"    Recall@{rm['k']}: {rm['recall_at_k']:.0%}   MRR: {rm['mrr']:.3f}   (n={rm['n']})\n")

    if llm_available():
        am = answer_metrics(gold)
        print("  ANSWER (LLM):")
        ca = am["citation_accuracy"]
        print(f"    Citation accuracy: {ca:.0%} ({am['citations_checked']} checked)"
              if ca is not None else "    Citation accuracy: n/a")
        print(f"    False-refusal rate (answerable): {am['false_refusal_rate']:.0%}")
        print(f"    Refusal correctness (adversarial): {am['rag_refusal_correctness']:.0%}")
        print(f"    Hallucination rate — RAG: {am['rag_hallucination_rate']:.0%}"
              f"   vs closed-book: {am['closed_book_hallucination_rate']:.0%}\n")
    else:
        print("  ANSWER (LLM): skipped — set ANTHROPIC_API_KEY to run these metrics.\n")
        _demo_citation_checker()
