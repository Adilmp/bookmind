"""Tests for src/verify.py. Run from the repo root:  python3 -m unittest discover -s tests"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import verify  # noqa: E402

HITS = [
    {"chapter": "THREE Quieting Self 1",
     "text": "Self 1 is the teller. It constantly gives instructions. Self 2 is the doer."},
    {"chapter": "FOUR Trusting Self 2",
     "text": "Trust your body to learn. Letting it happen is better than making it happen."},
]


class CitationTests(unittest.TestCase):
    def test_passage_numbers(self):
        self.assertTrue(verify.resolve_citation("2", HITS)["valid"])
        self.assertEqual(verify.resolve_citation("2", HITS)["chapter"], "FOUR Trusting Self 2")
        self.assertFalse(verify.resolve_citation("3", HITS)["valid"])       # only 2 passages
        self.assertTrue(verify.resolve_citation("1, 2", HITS)["valid"])

    def test_chapter_names_including_a_mangled_one(self):
        # qwen2.5:7b really wrote "[Quiet Self 1, THREE]" for "THREE Quieting Self 1"
        r = verify.resolve_citation("Quiet Self 1, THREE", HITS)
        self.assertTrue(r["valid"])
        self.assertEqual(r["chapter"], "THREE Quieting Self 1")
        self.assertTrue(verify.resolve_citation("Trusting Self 2", HITS)["valid"])

    def test_chapter_not_among_the_sources_is_invalid(self):
        self.assertFalse(verify.resolve_citation("The Meaning of Competition", HITS)["valid"])


class WordModeTests(unittest.TestCase):
    def test_sentence_splitting(self):
        self.assertEqual(verify.split_sentences("One. Two! Three?\nFour"), ["One.", "Two!", "Three?", "Four"])

    def test_supported_and_unsupported_sentences(self):
        answer = ("Self 1 constantly gives instructions [1]. "
                  "Gallwey recommends practising for four hours every morning [2].")
        r = verify.verify(answer, HITS, use_embeddings=False)
        self.assertEqual(r["method"], "words")
        statuses = [s["status"] for s in r["sentences"]]
        self.assertEqual(statuses, ["supported", "unsupported"])
        self.assertEqual(r["sentences"][0]["evidence"]["passage"], 1)
        self.assertEqual(r["summary"]["supported_share"], 0.5)

    def test_refusal_is_not_scored(self):
        r = verify.verify("I couldn't find this in the book.", HITS, use_embeddings=False)
        self.assertEqual(r["sentences"][0]["status"], "refusal")
        self.assertIsNone(r["summary"]["supported_share"])

    def test_invalid_citations_are_counted(self):
        r = verify.verify("Self 2 is the doer [7].", HITS, use_embeddings=False)
        self.assertEqual(r["summary"]["invalid_citations"], 1)


class EmbeddingModeTests(unittest.TestCase):
    def test_uses_embeddings_when_available(self):
        def fake_embed(texts, timeout=120):
            # answer text about "doer" points one way; only evidence mentioning "doer" matches it
            return [[1.0, 0.0] if "doer" in t else [0.0, 1.0] for t in texts]
        original = verify.llm.embed
        verify.llm.embed = fake_embed
        try:
            r = verify.verify("Self 2 is the doer.", HITS)
        finally:
            verify.llm.embed = original
        self.assertEqual(r["method"], "embeddings")
        self.assertEqual(r["sentences"][0]["status"], "supported")
        self.assertIn("doer", r["sentences"][0]["evidence"]["text"])

    def test_falls_back_to_words_when_embeddings_fail(self):
        def broken_embed(texts, timeout=120):
            raise OSError("Ollama not running")
        original = verify.llm.embed
        verify.llm.embed = broken_embed
        try:
            r = verify.verify("Self 2 is the doer.", HITS)
        finally:
            verify.llm.embed = original
        self.assertEqual(r["method"], "words")


if __name__ == "__main__":
    unittest.main()
