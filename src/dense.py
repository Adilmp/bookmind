"""
dense.py — meaning-based ("dense") retrieval with sentence embeddings.

BM25 matches words. It misses a passage that says the same thing in different words:
"calling my shots good or bad" vs the book's "letting go of judgment". Dense retrieval
turns every chunk and the question into a vector (nomic-embed-text through Ollama) and
ranks chunks by cosine similarity, so related meanings land close together.

Embedding the whole book takes about 90 seconds on a CPU, so the vectors are cached in
data/embeddings.json (git-ignored: derived from the book). The cache stores the model
name and a fingerprint of the chunk texts, and is rebuilt when either changes, so a
re-chunked book or a different model can never be scored against stale vectors.
"""
import hashlib
import json
import math
import os

import llm

CACHE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "embeddings.json")
BATCH = 32   # chunks per embedding request: bigger batches risk the request timeout


def fingerprint(texts):
    """A short hash of all chunk texts, in order: changes if any chunk changes."""
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()[:16]


def normalise(v):
    """Scale a vector to length 1, so cosine similarity becomes a plain dot product."""
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v] if n else v


class DenseIndex:
    def __init__(self, texts, cache_path=CACHE_PATH, embed=None):
        self.embed = embed or llm.embed          # injectable, so tests need no Ollama
        self.model = llm.EMBED_MODEL
        self.vectors = self._load_or_build(texts, cache_path)

    def _load_or_build(self, texts, cache_path):
        key = {"model": self.model, "fingerprint": fingerprint(texts), "count": len(texts)}
        if os.path.exists(cache_path):
            with open(cache_path, encoding="utf-8") as f:
                cached = json.load(f)
            if all(cached.get(k) == v for k, v in key.items()):
                return cached["vectors"]
        # nomic-embed-text expects task prefixes: "search_document:" for what is searched,
        # "search_query:" for the question. Leaving them out makes its rankings worse.
        vectors = []
        for start in range(0, len(texts), BATCH):
            batch = [f"search_document: {t}" for t in texts[start:start + BATCH]]
            vectors.extend(normalise(v) for v in self.embed(batch, timeout=300))
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(dict(key, vectors=vectors), f)
        return vectors

    def search(self, query, k=5):
        """Top-k (index, cosine similarity), best first."""
        q = normalise(self.embed([f"search_query: {query}"])[0])
        scored = [(i, sum(a * b for a, b in zip(q, v))) for i, v in enumerate(self.vectors)]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]
