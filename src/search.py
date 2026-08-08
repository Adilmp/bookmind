"""
search.py — Load chunks, build the BM25 index, answer queries with citations.

This is the Week-1 deliverable: ask a question -> get the most relevant passages
from the book, each with a chapter citation. (LLM-generated answers come next week;
retrieval is the foundation everything else stands on.)
"""
import json
import os
from bm25 import BM25, tokenize

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "chunks.jsonl")


def load_chunks(path=DATA):
    chunks = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


class BookSearch:
    def __init__(self, path=DATA):
        self.chunks = load_chunks(path)
        corpus = [tokenize(c["text"]) for c in self.chunks]
        self.index = BM25(corpus)

    def search(self, query, k=5):
        hits = self.index.search(query, k=k)
        results = []
        for i, score in hits:
            c = self.chunks[i]
            results.append({
                "score": round(score, 3),
                "chapter": c["chapter"],
                "chunk_id": c["id"],
                "text": c["text"],
            })
        return results


def _snippet(text, n=300):
    return text if len(text) <= n else text[:n].rsplit(" ", 1)[0] + " …"


if __name__ == "__main__":
    import sys
    query = " ".join(sys.argv[1:]) or "how do I stop overthinking and trust myself"
    bs = BookSearch()
    print(f'\n  Q: "{query}"')
    print(f"  ({len(bs.chunks)} chunks indexed)\n")
    for rank, r in enumerate(bs.search(query, k=5), 1):
        print(f"  [{rank}]  score={r['score']}   « {r['chapter']} »  (chunk #{r['chunk_id']})")
        print(f"       {_snippet(r['text'])}\n")
