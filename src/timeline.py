"""
timeline.py — follow one idea through the book, chapter by chapter.

Give it an idea ("Self 1", "concentration", "trust") and it finds every passage that
mentions it, then groups the mentions by chapter in reading order, with a count per
chapter and the sentences that mention it. No model is involved, so it is instant and
always gives the same answer.

Matching rules (deliberately simple, so the result is easy to explain):
  - Words are compared the way the search index sees them: lowercase letters and digits
    (bm25.tokenize). "Self" matches "self" but not "yourself".
  - A multi-word idea must appear as the exact phrase ("self 1"), not as scattered words.
  - Chunks overlap by CHUNK_OVERLAP words (see ingest.py). A mention lying entirely inside
    that shared start was already counted in the previous chunk, so it is skipped; one that
    starts there but ends after it is counted (the previous chunk was cut mid-phrase).
  - Non-content pages (contents, copyright, ...) are skipped: the contents page lists
    every chapter title and would otherwise "mention" most ideas.

Run:  python src/timeline.py "Self 1"
"""
import re
import sys

from bm25 import tokenize
from ingest import CHUNK_OVERLAP
from search import load_chunks

NON_CONTENT = ("contents", "copyright", "dedication", "epigraph", "also by",
               "about the author", "acknowledg", "index")

_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")


def is_content(chapter):
    """False for front/back-matter pages such as the table of contents."""
    title = chapter.lower()
    return not any(word in title for word in NON_CONTENT)


def count_phrase(tokens, phrase):
    """How many times the token list `phrase` appears as a contiguous run in `tokens`."""
    n = len(phrase)
    return sum(1 for i in range(len(tokens) - n + 1) if tokens[i:i + n] == phrase)


def phrase_pattern(phrase):
    """Regex that finds the phrase in raw text: whole words, any punctuation/space between."""
    return re.compile(r"\b" + r"[\W_]+".join(map(re.escape, phrase)) + r"\b", re.IGNORECASE)


def new_mentions(chunk, previous, phrase):
    """Mentions of `phrase` in this chunk that the previous chunk did not already count."""
    continues = previous is not None and previous["source_file"] == chunk["source_file"]
    skip = CHUNK_OVERLAP if continues else 0
    tokens, word_of = [], []  # word_of[i]: which word of the chunk token i came from
    for w, word in enumerate(chunk["text"].split()):
        for tok in tokenize(word):
            tokens.append(tok)
            word_of.append(w)
    n = len(phrase)
    return sum(1 for i in range(len(tokens) - n + 1)
               if tokens[i:i + n] == phrase and word_of[i + n - 1] >= skip)


def matching_sentences(text, pattern):
    """Sentences of `text` that contain the phrase, each with the character spans to highlight."""
    out = []
    for sentence in _SENTENCE_END.split(text):
        spans = [[m.start(), m.end()] for m in pattern.finditer(sentence)]
        if spans:
            out.append({"text": sentence.strip(), "spans": spans})
    return out


def timeline(idea, chunks, snippets_per_chapter=2):
    """Mentions of `idea` per chapter, in reading order.

    Returns {"idea", "total_mentions", "chapters": [{"chapter", "mentions", "passages",
    "snippets": [{"chunk_id", "text", "spans"}]}]}. Every content chapter is listed, including
    those with zero mentions, so the reader can see where an idea is absent.
    """
    phrase = tokenize(idea)
    if not phrase:
        raise ValueError("the idea needs at least one letter or digit")
    pattern = phrase_pattern(phrase)

    chapters = {}  # dict keeps insertion order = reading order, because chunks are in order
    previous = None
    for chunk in chunks:
        title = chunk["chapter"]
        if is_content(title):
            entry = chapters.setdefault(title, {"chapter": title, "mentions": 0,
                                                "passages": 0, "snippets": []})
            mentions = new_mentions(chunk, previous, phrase)
            if mentions:
                entry["mentions"] += mentions
                entry["passages"] += 1
                if len(entry["snippets"]) < snippets_per_chapter:
                    seen = {s["text"] for s in entry["snippets"]}
                    fresh = [s for s in matching_sentences(chunk["text"], pattern) if s["text"] not in seen]
                    if fresh:  # the overlap can repeat a sentence already shown
                        entry["snippets"].append({"chunk_id": chunk["id"], **fresh[0]})
        previous = chunk

    result = list(chapters.values())
    return {
        "idea": idea,
        "total_mentions": sum(c["mentions"] for c in result),
        "chapters": result,
    }


if __name__ == "__main__":
    idea = " ".join(sys.argv[1:]) or "Self 1"
    t = timeline(idea, load_chunks())
    print(f'\n  "{t["idea"]}": {t["total_mentions"]} mentions\n')
    for c in t["chapters"]:
        bar = "#" * c["mentions"]
        print(f"  {c['mentions']:>4}  {c['chapter'][:48]:<48} {bar}")
