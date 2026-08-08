"""
bm25.py — BM25 ranking implemented from scratch (no rank_bm25, no sklearn).

Why from scratch: so you can explain in an interview *why* it works, not just
that you imported it. BM25 scores how well a document matches a query using:

  score(D, Q) = sum over query terms q of:
        IDF(q) * ( f(q,D) * (k1 + 1) )
                 -----------------------------------------
                 ( f(q,D) + k1 * (1 - b + b * |D| / avgdl) )

  - f(q, D)  : how many times term q appears in document D  (term frequency)
  - |D|      : length of D in words;  avgdl : average doc length
  - IDF(q)   : rare terms are worth more than common ones
  - k1 (~1.5): controls term-frequency saturation (more hits help, with diminishing returns)
  - b  (~0.75): how much to penalise long documents

Compared to plain TF-IDF, BM25 adds (a) TF saturation and (b) length normalisation,
which is why it's still a strong baseline in 2026.
"""
import math
import re
from collections import Counter

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text):
    """Lowercase + split into word tokens. (Deliberately simple + explainable.)"""
    return _TOKEN_RE.findall(text.lower())


class BM25:
    def __init__(self, corpus_tokens, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.docs = corpus_tokens
        self.N = len(corpus_tokens)
        self.doc_len = [len(d) for d in corpus_tokens]
        self.avgdl = (sum(self.doc_len) / self.N) if self.N else 0.0
        self.tf = [Counter(d) for d in corpus_tokens]          # term freq per doc
        self.idf = self._compute_idf()

    def _compute_idf(self):
        """Smoothed IDF: log(1 + (N - df + 0.5) / (df + 0.5))."""
        df = Counter()
        for doc in self.docs:
            for term in set(doc):
                df[term] += 1
        idf = {}
        for term, freq in df.items():
            idf[term] = math.log(1 + (self.N - freq + 0.5) / (freq + 0.5))
        return idf

    def score(self, query_tokens, index):
        """BM25 score of one document (by index) against the query."""
        score = 0.0
        tf = self.tf[index]
        dl = self.doc_len[index]
        for term in query_tokens:
            if term not in tf:
                continue
            idf = self.idf.get(term, 0.0)
            freq = tf[term]
            denom = freq + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            score += idf * (freq * (self.k1 + 1)) / denom
        return score

    def search(self, query, k=5):
        """Return top-k (index, score), best first."""
        q = tokenize(query)
        scored = [(i, self.score(q, i)) for i in range(self.N)]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [(i, s) for i, s in scored[:k] if s > 0]
