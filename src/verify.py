"""
verify.py — "show me where it says that": check a generated answer sentence by sentence.

For every sentence of an answer, find the stretch of the retrieved passages (one or two
consecutive sentences) that supports it best, and label the answer sentence:

  supported    the passages say the same thing
  weak         related text, but not clearly the same claim
  unsupported  nothing in the passages backs it: a possible hallucination

It also checks the citations the model wrote: "[3]" must point at a passage that exists,
and a chapter name such as "[Quieting Self 1]" must match a chapter among the sources.

Similarity comes from sentence embeddings (nomic-embed-text through Ollama) when available,
otherwise from word overlap. Both are cheap next to asking an LLM to judge, and neither can
hallucinate. The thresholds below were calibrated on real answers from this book (README).

Limits: similarity is not the same as meaning. A sentence that reuses a passage's words but
reverses it ("Self 2 cannot be trusted") can still look supported. This flags likely
problems; it does not prove an answer correct.
"""
import math
import re

import llm
from bm25 import tokenize

# Calibrated on 8 real qwen2.5:7b answers from this book (see README, "Checking answers"):
# every fabricated or mismatched sentence scored below 0.79 on embeddings and below 0.40 on
# words, while genuinely supported sentences scored 0.71-0.95 and usually shared most words.
EMBED_SUPPORTED, EMBED_WEAK = 0.80, 0.70   # cosine similarity of sentence embeddings
EMBED_WORDS_BOOST = 0.50                   # 0.70-0.80 counts as supported if half the words match
WORDS_SUPPORTED, WORDS_WEAK = 0.60, 0.40   # word-overlap fallback when embeddings are unavailable

REFUSAL = "couldn't find this in the book"
CITATION = re.compile(r"\[([^\]]+)\]")
_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+|\n+")
STOPWORDS = set("""a an the and or but if of to in on at by for with from into is are was were be
been being do does did have has had it its this that these those he she they we you i him her
them his their our your my me us as so not no can will would should could just than too very
about what which who how why when where there here all any each more most some such only also
then""".split())


# ---------- small helpers ----------

def split_sentences(text):
    """Split text into sentences (on . ! ? … followed by whitespace, and on line breaks)."""
    return [s.strip() for s in _SENTENCE_END.split(text) if s.strip()]


def content_words(text):
    """The tokens that carry meaning: no stopwords, no citation markers."""
    return {t for t in tokenize(CITATION.sub(" ", text)) if t not in STOPWORDS}


def word_support(sentence, evidence):
    """Share of the sentence's content words that appear in the evidence (0 to 1)."""
    words = content_words(sentence)
    return len(words & content_words(evidence)) / len(words) if words else 0.0


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


def evidence_units(hits):
    """Candidate evidence: every sentence of every passage, and every pair of consecutive
    sentences (a claim can span two). Each unit remembers its passage number (1-based)."""
    units = []
    for n, hit in enumerate(hits, 1):
        sentences = split_sentences(hit["text"])
        for i, s in enumerate(sentences):
            units.append({"passage": n, "chapter": hit["chapter"], "text": s})
            if i + 1 < len(sentences):
                units.append({"passage": n, "chapter": hit["chapter"],
                              "text": s + " " + sentences[i + 1]})
    return units


# ---------- citations ----------

def resolve_citation(raw, hits):
    """Map what's inside one [...] to a source. Returns {"raw", "chapter", "valid"}.

    Numbers ("3", "1, 2") refer to passages. Anything else is read as a chapter name and must
    share most of its words with a chapter among the retrieved passages.
    """
    numbers = re.findall(r"\d+", raw)
    if numbers and not re.search(r"[A-Za-z]", raw):
        idx = [int(n) for n in numbers]
        valid = all(1 <= i <= len(hits) for i in idx)
        chapters = [hits[i - 1]["chapter"] for i in idx if 1 <= i <= len(hits)]
        return {"raw": raw, "chapter": "; ".join(dict.fromkeys(chapters)) or None, "valid": valid}
    ref = set(tokenize(raw))
    best, best_score = None, 0.0
    for chapter in dict.fromkeys(h["chapter"] for h in hits):
        score = len(ref & set(tokenize(chapter))) / len(ref) if ref else 0.0
        if score > best_score:
            best, best_score = chapter, score
    valid = best_score >= 0.6
    return {"raw": raw, "chapter": best if valid else None, "valid": valid}


# ---------- the check ----------

def verify(answer, hits, use_embeddings=True):
    """Check every sentence of `answer` against the retrieved passages `hits`.

    Returns {"method", "sentences": [...], "summary": {...}}. Each sentence entry has its
    status, score, best evidence (passage number, chapter, text) and resolved citations.
    """
    sentences = split_sentences(answer)
    units = evidence_units(hits)
    claims = [s for s in sentences if REFUSAL not in s.lower() and content_words(s)]

    vectors = None
    if use_embeddings and claims and units:
        try:
            # nomic-embed-text expects these task prefixes; without them rankings get worse
            vectors = llm.embed([f"search_query: {CITATION.sub('', s).strip()}" for s in claims]
                                + [f"search_document: {u['text']}" for u in units])
        except Exception:  # Ollama not running: fall back to word overlap
            vectors = None
    method = "embeddings" if vectors else "words"

    results, claim_i = [], 0
    for sentence in sentences:
        citations = [resolve_citation(c, hits) for c in CITATION.findall(sentence)]
        entry = {"sentence": sentence, "citations": citations}
        if REFUSAL in sentence.lower():
            entry.update(status="refusal", score=None, evidence=None)
        elif not content_words(sentence) or not units:
            entry.update(status="unsupported", score=0.0, evidence=None)
        else:
            if vectors:
                q = vectors[claim_i]
                scores = [cosine(q, vectors[len(claims) + j]) for j in range(len(units))]
                j = max(range(len(units)), key=scores.__getitem__)
                score, words = scores[j], word_support(sentence, units[j]["text"])
                supported = score >= EMBED_SUPPORTED or (score >= EMBED_WEAK and words >= EMBED_WORDS_BOOST)
                weak = score >= EMBED_WEAK
            else:
                scores = [word_support(sentence, u["text"]) for u in units]
                j = max(range(len(units)), key=scores.__getitem__)
                score = scores[j]
                supported, weak = score >= WORDS_SUPPORTED, score >= WORDS_WEAK
            claim_i += 1
            status = "supported" if supported else "weak" if weak else "unsupported"
            entry.update(status=status, score=round(score, 3), evidence=units[j])
        results.append(entry)

    checked = [r for r in results if r["status"] != "refusal"]
    supported = sum(r["status"] == "supported" for r in checked)
    return {
        "method": method,
        "sentences": results,
        "summary": {
            "sentences": len(checked),
            "supported": supported,
            "weak": sum(r["status"] == "weak" for r in checked),
            "unsupported": sum(r["status"] == "unsupported" for r in checked),
            "supported_share": round(supported / len(checked), 3) if checked else None,
            "invalid_citations": sum(not c["valid"] for r in results for c in r["citations"]),
        },
    }
