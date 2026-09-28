"""
search.py — Load chunks and find the passages that answer a question, with citations.

Three ways to rank (BOOKMIND_RETRIEVAL, default "hybrid"):

  bm25    keyword match (bm25.py, written by hand). Fast, exact, misses rewordings.
  dense   meaning match with embeddings (dense.py). Catches rewordings, can drift to
          passages that are merely "about the same topic".
  hybrid  both, merged with Reciprocal Rank Fusion (below). The default, because it
          scored best on the evaluation set (README, "Retrieval").

If Ollama isn't running, dense and hybrid fall back to BM25 instead of failing, and every
result says which method actually produced it.
"""
import json
import os

from bm25 import BM25, tokenize

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "chunks.jsonl")
MODES = ("hybrid", "bm25", "dense")
RRF_K = 60        # the constant from the original RRF paper (Cormack et al., 2009)
CANDIDATES = 50   # how deep each ranking goes before fusing


def load_chunks(path=DATA):
    chunks = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def rrf(rankings, k=RRF_K):
    """Reciprocal Rank Fusion: merge ranked lists of ids into one ranking.

    Each list gives an id 1 / (k + rank). Ids near the top of either list score well,
    ids near the top of both score best. Only ranks are used, never raw scores, so BM25's
    unbounded scores and cosine's 0-1 range never need to be put on the same scale.
    Ties break by id, so the order is deterministic. Returns [(id, score)], best first.
    """
    scores = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, 1):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: (-x[1], x[0]))


class BookSearch:
    def __init__(self, path=DATA, mode=None, dense_index=None):
        self.chunks = load_chunks(path)
        self.index = BM25([tokenize(c["text"]) for c in self.chunks])
        self.mode = mode or os.environ.get("BOOKMIND_RETRIEVAL", "hybrid")
        if self.mode not in MODES:
            raise ValueError(f"retrieval mode must be one of {MODES}, not {self.mode!r}")
        self.dense, self.dense_error = dense_index, None
        if self.mode != "bm25" and self.dense is None:
            try:
                from dense import DenseIndex
                self.dense = DenseIndex([c["text"] for c in self.chunks])
            except Exception as e:  # Ollama not running: keyword search still works
                self.dense_error = f"{type(e).__name__}: {e}"

    def search(self, query, k=5, mode=None):
        """Top-k passages. Each result has its chapter citation, the method that actually
        ranked it, and its rank in each underlying list (None if it wasn't in the top 50)."""
        mode = mode or self.mode
        bm25_hits = self.index.search(query, k=CANDIDATES)
        dense_hits = None
        if mode != "bm25" and self.dense is not None:
            try:
                dense_hits = self.dense.search(query, k=CANDIDATES)
            except Exception as e:  # Ollama stopped after startup
                self.dense_error = f"{type(e).__name__}: {e}"
        bm25_ids = [i for i, _ in bm25_hits]
        dense_ids = [i for i, _ in dense_hits] if dense_hits is not None else None

        if dense_hits is None:
            method, ranked = "bm25", bm25_hits[:k]
        elif mode == "dense":
            method, ranked = "dense", dense_hits[:k]
        else:
            method, ranked = "hybrid", rrf([bm25_ids, dense_ids])[:k]

        def rank_in(ids, i):
            return ids.index(i) + 1 if ids and i in ids else None

        results = []
        for i, score in ranked:
            c = self.chunks[i]
            results.append({
                "score": round(score, 4),
                "chapter": c["chapter"],
                "chunk_id": c["id"],
                "text": c["text"],
                "method": method,
                "ranks": {"bm25": rank_in(bm25_ids, i), "dense": rank_in(dense_ids, i)},
            })
        return results


def _snippet(text, n=300):
    return text if len(text) <= n else text[:n].rsplit(" ", 1)[0] + " …"


if __name__ == "__main__":
    import sys
    query = " ".join(sys.argv[1:]) or "how do I stop overthinking and trust myself"
    bs = BookSearch()
    hits = bs.search(query, k=5)
    print(f'\n  Q: "{query}"')
    print(f"  ({len(bs.chunks)} chunks indexed, ranked by {hits[0]['method'] if hits else bs.mode})\n")
    for rank, r in enumerate(hits, 1):
        print(f"  [{rank}]  score={r['score']}  ranks={r['ranks']}  « {r['chapter']} »  (chunk #{r['chunk_id']})")
        print(f"       {_snippet(r['text'])}\n")
