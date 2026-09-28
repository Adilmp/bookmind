"""Tests for src/timeline.py. Run from the repo root:  python3 -m unittest discover -s tests"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import ingest  # noqa: E402
from ingest import CHUNK_OVERLAP  # noqa: E402
from timeline import timeline  # noqa: E402


def make_chunks(words, chapter="ONE Test Chapter", source="ch1.xhtml", size=60, start_id=0):
    """Chunk a word list exactly the way ingest.py does (overlapping windows)."""
    pieces = ingest._chunk(words, size, CHUNK_OVERLAP)
    return [{"id": start_id + i, "chapter": chapter, "source_file": source, "text": p}
            for i, p in enumerate(pieces)]


class TimelineTests(unittest.TestCase):
    def count(self, idea, chunks):
        return timeline(idea, chunks)["total_mentions"]

    def test_whole_words_only(self):
        chunks = make_chunks("yourself itself self selfish".split())
        self.assertEqual(self.count("self", chunks), 1)

    def test_phrase_must_be_contiguous(self):
        chunks = make_chunks("self 1 is here but self then 1 is not".split())
        self.assertEqual(self.count("Self 1", chunks), 1)

    def test_mention_inside_the_overlap_is_counted_once(self):
        words = ["w%d" % i for i in range(90)]
        words[45] = "trust"                      # inside chunk 0 and in the overlap of chunk 1
        chunks = make_chunks(words)
        self.assertGreater(len(chunks), 1)
        self.assertEqual(self.count("trust", chunks), 1)

    def test_phrase_crossing_the_end_of_the_overlap_is_counted_once(self):
        # chunk 1 starts at word 30 and shares words 30-59 with chunk 0, so chunk 0 ends
        # after word 59; put "self" at 59 and "1" at 60: the phrase straddles the cut
        words = ["w%d" % i for i in range(120)]
        words[59], words[60] = "self", "1"
        self.assertEqual(self.count("self 1", make_chunks(words)), 1)

    def test_different_files_do_not_share_an_overlap(self):
        a = make_chunks(["trust"] + ["x"] * 30, source="a.xhtml")
        b = make_chunks(["trust"] + ["x"] * 30, source="b.xhtml", start_id=len(a))
        self.assertEqual(self.count("trust", a + b), 2)

    def test_non_content_pages_are_skipped_and_chapters_keep_reading_order(self):
        chunks = (make_chunks("contents trust trust".split(), chapter="Contents", source="toc")
                  + make_chunks("trust".split(), chapter="TWO Later", source="c2", start_id=10)
                  + make_chunks("nothing here".split(), chapter="THREE Last", source="c3", start_id=20))
        t = timeline("trust", chunks)
        self.assertEqual([c["chapter"] for c in t["chapters"]], ["TWO Later", "THREE Last"])
        self.assertEqual([c["mentions"] for c in t["chapters"]], [1, 0])

    def test_snippet_spans_point_at_the_phrase(self):
        chunks = make_chunks("The first sentence. Then Self 1, the teller, speaks. End.".split())
        snippet = timeline("self 1", chunks)["chapters"][0]["snippets"][0]
        start, end = snippet["spans"][0]
        self.assertEqual(snippet["text"][start:end], "Self 1")
        self.assertEqual(snippet["text"], "Then Self 1, the teller, speaks.")

    def test_empty_idea_is_rejected(self):
        with self.assertRaises(ValueError):
            timeline("  ?! ", make_chunks("a b c".split()))


if __name__ == "__main__":
    unittest.main()
