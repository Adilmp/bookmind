"""Tests for src/search.py and src/dense.py. Run from the repo root:  python3 -m unittest discover -s tests"""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import dense  # noqa: E402
from search import BookSearch, rrf  # noqa: E402

CHUNKS = [
    {"id": 0, "chapter": "ONE A", "text": "judgment of every shot makes the muscles tight"},
    {"id": 1, "chapter": "TWO B", "text": "watch the seams of the ball and say bounce then hit"},
    {"id": 2, "chapter": "THREE C", "text": "letting go of good and bad labels quiets the mind"},
]


def fake_embed(texts, timeout=120):
    """2-d 'meanings': anything about labels or judging points one way, the rest the other."""
    return [[1.0, 0.1] if any(w in t for w in ("label", "judg", "calling")) else [0.1, 1.0]
            for t in texts]


class FakeDense:
    def __init__(self, ranking=None, broken=False):
        self.ranking, self.broken = ranking or [2, 0, 1], broken

    def search(self, query, k=5):
        if self.broken:
            raise OSError("Ollama stopped")
        return [(i, 1.0 - n / 10) for n, i in enumerate(self.ranking)][:k]


class RRFTests(unittest.TestCase):
    def test_agreement_beats_one_first_place(self):
        # "b" is 2nd in both lists; "a" is 1st in one list and absent from the other
        fused = [doc for doc, _ in rrf([["a", "b"], ["c", "b"]])]
        self.assertEqual(fused[0], "b")

    def test_score_is_sum_of_reciprocal_ranks(self):
        scores = dict(rrf([["x", "y"], ["y"]], k=60))
        self.assertAlmostEqual(scores["y"], 1 / 62 + 1 / 61)
        self.assertAlmostEqual(scores["x"], 1 / 61)

    def test_ties_break_by_id_so_order_is_stable(self):
        self.assertEqual([d for d, _ in rrf([[3], [1]])], [1, 3])


class BookSearchTests(unittest.TestCase):
    def setUp(self):
        fd, self.path = tempfile.mkstemp(suffix=".jsonl")
        with os.fdopen(fd, "w") as f:
            f.write("\n".join(json.dumps(c) for c in CHUNKS))
        self.addCleanup(os.remove, self.path)

    def test_hybrid_reports_method_and_both_ranks(self):
        bs = BookSearch(self.path, mode="hybrid", dense_index=FakeDense())
        hits = bs.search("bounce hit", k=3)
        self.assertEqual({h["method"] for h in hits}, {"hybrid"})
        ball = next(h for h in hits if h["chunk_id"] == 1)
        self.assertEqual(ball["ranks"], {"bm25": 1, "dense": 3})

    def test_bm25_mode_never_calls_the_dense_index(self):
        bs = BookSearch(self.path, mode="bm25", dense_index=FakeDense(broken=True))
        hits = bs.search("seams of the ball")
        self.assertEqual((hits[0]["chunk_id"], hits[0]["method"]), (1, "bm25"))
        self.assertIsNone(bs.dense_error)

    def test_falls_back_to_bm25_when_embeddings_fail(self):
        bs = BookSearch(self.path, mode="hybrid", dense_index=FakeDense(broken=True))
        hits = bs.search("seams of the ball")
        self.assertEqual(hits[0]["method"], "bm25")
        self.assertIn("Ollama stopped", bs.dense_error)
        self.assertIsNone(hits[0]["ranks"]["dense"])

    def test_dense_mode_uses_the_dense_ranking(self):
        bs = BookSearch(self.path, mode="dense", dense_index=FakeDense(ranking=[2, 0, 1]))
        self.assertEqual([h["chunk_id"] for h in bs.search("anything", k=2)], [2, 0])

    def test_unknown_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            BookSearch(self.path, mode="vector")


class DenseIndexTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.cache = os.path.join(self.dir.name, "emb.json")

    def embed(self, texts, timeout=120):
        self.calls.append(list(texts))
        return fake_embed(texts)

    def test_finds_meaning_and_uses_nomic_prefixes(self):
        idx = dense.DenseIndex([c["text"] for c in CHUNKS], self.cache, embed=self.embed)
        top = idx.search("calling shots bad", k=1)[0][0]
        self.assertIn(top, (0, 2))      # both chunks are about judging, neither shares "calling"
        self.assertTrue(all(t.startswith("search_document: ") for t in self.calls[0]))
        self.assertEqual(self.calls[-1], ["search_query: calling shots bad"])

    def test_cache_is_reused_and_rebuilt_when_the_book_changes(self):
        texts = [c["text"] for c in CHUNKS]
        dense.DenseIndex(texts, self.cache, embed=self.embed)
        dense.DenseIndex(texts, self.cache, embed=self.embed)
        self.assertEqual(len(self.calls), 1)                  # second load: from the cache
        dense.DenseIndex(texts[:2] + ["a changed chunk"], self.cache, embed=self.embed)
        self.assertEqual(len(self.calls), 2)                  # fingerprint changed: re-embedded

    def test_vectors_are_unit_length(self):
        v = dense.normalise([3.0, 4.0])
        self.assertAlmostEqual(v[0] ** 2 + v[1] ** 2, 1.0)


if __name__ == "__main__":
    unittest.main()
