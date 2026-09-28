"""
evaluate.py — Week 3: the evaluation harness (BookMind's differentiator).

Two tiers, so it always produces something useful:
  1. Retrieval metrics — Recall@k and MRR over the gold set, for BM25, dense and
     hybrid side by side. BM25 needs nothing; dense and hybrid need Ollama for the
     embeddings (they fall back to BM25 without it, and the report says so).
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

def first_hit_rank(results, chapter):
    """1-based rank of the first result from the gold chapter, or None."""
    for i, r in enumerate(results, 1):
        if chapter in _normalize_chapter(r["chapter"]):
            return i
    return None


def retrieval_metrics(gold, k=5, mode="bm25", bs=None):
    """Recall@k and MRR for one retrieval mode. Also returns each question's rank, so two
    modes can be compared question by question, and the method that really ran (hybrid
    silently becomes bm25 if Ollama is down; the report must not claim otherwise)."""
    bs = bs or BookSearch(mode=mode)
    answerable = [g for g in gold if g["answerable"]]
    ranks, methods = [], set()
    for g in answerable:
        results = bs.search(g["q"], k=k, mode=mode)
        methods.update(r["method"] for r in results)
        ranks.append(first_hit_rank(results, g["chapter"]))
    n = len(answerable)
    return {
        "n": n,
        "recall_at_k": sum(r is not None for r in ranks) / n,
        "mrr": sum(1 / r for r in ranks if r) / n,
        "k": k,
        "ranks": ranks,
        "methods": sorted(methods),
    }


def paired_bootstrap(a_ranks, b_ranks, n=10000, seed=0):
    """95% interval for mean(RR of b) - mean(RR of a), resampling the same questions for
    both (paired). If the interval contains 0, the MRR difference could be luck."""
    import random
    rng = random.Random(seed)
    rr = lambda r: 1 / r if r else 0.0
    diffs = [rr(b) - rr(a) for a, b in zip(a_ranks, b_ranks)]
    means = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(n))
    return sum(diffs) / len(diffs), means[int(0.025 * n)], means[int(0.975 * n) - 1]


def compare_retrievers(gold, k=5, modes=("bm25", "dense", "hybrid")):
    """Run every mode on the same index; print overall and per-slice numbers, and how
    many questions each mode finds that BM25 misses (and the reverse)."""
    bs = BookSearch(mode="hybrid")
    answerable = [g for g in gold if g["answerable"]]
    runs = {m: retrieval_metrics(gold, k, m, bs) for m in modes}
    slices = {
        "all": lambda g: True,
        "original 12 (v1)": lambda g: g.get("set", "v1") == "v1",
        "paraphrase (v2)": lambda g: g.get("style") == "paraphrase",
        "keyword (v2)": lambda g: g.get("style") == "keyword",
    }
    print(f"  RETRIEVAL (Recall@{k} / MRR, chapter-level gold labels):")
    print("    " + f"{'slice':24}" + "".join(f"{m:>16}" for m in modes))
    for name, keep in slices.items():
        idx = [i for i, g in enumerate(answerable) if keep(g)]
        cells = []
        for m in modes:
            rs = [runs[m]["ranks"][i] for i in idx]
            recall = sum(r is not None for r in rs) / len(rs)
            mrr = sum(1 / r for r in rs if r) / len(rs)
            cells.append(f"{recall:5.0%} / {mrr:.3f}")
        print("    " + f"{name + f' n={len(idx)}':24}" + "".join(f"{c:>16}" for c in cells))
    for m in modes:
        if m != "bm25":
            base, other = runs["bm25"]["ranks"], runs[m]["ranks"]
            gains = sum(b is None and o is not None for b, o in zip(base, other))
            losses = sum(b is not None and o is None for b, o in zip(base, other))
            diff, lo, hi = paired_bootstrap(base, other)
            print(f"    {m} vs bm25: finds {gains} question(s) bm25 misses, misses {losses} it finds; "
                  f"MRR {diff:+.3f} (95% bootstrap interval {lo:+.3f} to {hi:+.3f})")
    ran = {m: runs[m]["methods"] for m in modes}
    if any(ran[m] != [m] for m in modes):
        print(f"    WARNING: some modes fell back to another method: {ran}")
    return runs


# ---------- tier 2: answer metrics (needs an LLM key) ----------

def llm_available():
    """True if the configured model (Claude with a key, or a running Ollama) is usable."""
    import llm
    return llm.available()


def _closed_book_answer(query):
    """LLM with NO retrieval — the baseline RAG should beat on hallucination."""
    import llm
    return llm.chat(
        'Answer only from "The Inner Game of Tennis" by W. Timothy Gallwey. '
        f'If it is not covered in that book, reply exactly: "{REFUSAL}"',
        query, max_tokens=512,
    )


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

    compare_retrievers(gold)
    print()

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
        print("  ANSWER (LLM): skipped — set ANTHROPIC_API_KEY or start Ollama to run these metrics.\n")
        _demo_citation_checker()
