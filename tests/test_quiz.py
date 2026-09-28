"""Tests for src/quiz.py. Run from the repo root:  python3 -m unittest discover -s tests"""
import datetime as dt
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import quiz  # noqa: E402

# Invented text in the book's style (the real book text never goes into the repo)
PASSAGE = ("The first habit is to notice your footwork without comment. It should be observed calmly. "
           "This works best when you stop keeping score in your head. " + "filler " * 100)
CHUNK = {"id": 7, "chapter": "THREE Quieting Self 1", "text": PASSAGE}


class FakeChat:
    """Stands in for llm.chat: returns canned replies in order."""
    def __init__(self, replies):
        self.replies = list(replies)

    def __call__(self, system, user, max_tokens=1024, temperature=0.0, json_mode=False):
        return self.replies.pop(0)


class WithFakeModel(unittest.TestCase):
    def use_replies(self, *replies):
        original = quiz.llm.chat
        quiz.llm.chat = FakeChat(replies)
        self.addCleanup(setattr, quiz.llm, "chat", original)


class QuoteAndParsingTests(unittest.TestCase):
    def test_quote_must_really_be_in_the_passage(self):
        self.assertTrue(quiz.quote_in_passage("It should be observed calmly.", PASSAGE))
        self.assertTrue(quiz.quote_in_passage('"this works best when you stop keeping score in your head."', PASSAGE))
        self.assertFalse(quiz.quote_in_passage("You must never judge your strokes at all.", PASSAGE))
        self.assertFalse(quiz.quote_in_passage("notice your footwork.", PASSAGE))  # under 5 words

    def test_fragments_are_rejected(self):
        # the kinds of fragment qwen2.5:7b really returned, which passed the first version of the check
        self.assertFalse(quiz.quote_in_passage("The first habit is to notice your footwork",
                                               PASSAGE))  # no sentence ending
        self.assertFalse(quiz.quote_in_passage("This works best when you stop keeping score,", PASSAGE))
        self.assertTrue(quiz.quote_in_passage("The first habit is to notice your footwork without comment.", PASSAGE))

    def test_curly_quotes_and_spacing_are_normalised(self):
        text = "He said: “Let  it   happen” and it’s fine to wait."
        self.assertTrue(quiz.quote_in_passage('He said: "Let it happen" and it\'s fine to wait.', text))

    def test_parse_json_handles_fences_and_lists(self):
        self.assertEqual(quiz.parse_json('```json\n{"a": 1}\n```'), {"a": 1})
        self.assertEqual(quiz.parse_json('[{"a": 2}, {"a": 3}]'), {"a": 2})
        with self.assertRaises(ValueError):
            quiz.parse_json('"just a string"')


class PassageTests(unittest.TestCase):
    def test_passages_are_spread_and_short_ones_skipped(self):
        chunks = [{"id": i, "chapter": "C", "text": "w " * 150} for i in range(10)]
        chunks.append({"id": 10, "chapter": "C", "text": "short tail"})
        picked = [c["id"] for c in quiz.pick_passages(chunks, "C", 4)]
        self.assertEqual(picked, [0, 3, 6, 9])


class ReviewTests(unittest.TestCase):
    def test_right_answers_move_up_wrong_answers_restart(self):
        day = dt.date(2026, 9, 28)
        progress = {}
        self.assertEqual(quiz.review(progress, "q1", True, today=day)["due"], "2026-09-29")   # box 2: 1 day
        self.assertEqual(quiz.review(progress, "q1", True, today=day)["due"], "2026-10-01")   # box 3: 3 days
        p = quiz.review(progress, "q1", False, today=day)
        self.assertEqual((p["box"], p["due"], p["seen"], p["right"]), (1, "2026-09-28", 3, 2))

    def test_due_lists_new_and_overdue_questions_lowest_box_first(self):
        qs = [{"id": "new"}, {"id": "later"}, {"id": "overdue"}]
        progress = {"later": {"box": 3, "due": "2026-10-05"}, "overdue": {"box": 2, "due": "2026-09-27"}}
        ids = [q["id"] for q in quiz.due(qs, progress, today=dt.date(2026, 9, 28))]
        self.assertEqual(ids, ["new", "overdue"])

    def test_progress_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "p.json")
            self.assertEqual(quiz.load_progress(path), {})
            quiz.save_progress({"q": {"box": 2}}, path)
            self.assertEqual(quiz.load_progress(path), {"q": {"box": 2}})


class GenerateAndGradeTests(WithFakeModel):
    def test_generate_keeps_grounded_questions_and_rejects_invented_quotes(self):
        good = json.dumps({"question": "How should footwork be observed?", "answer": "Calmly, without comment.",
                           "quote": "It should be observed calmly."})
        invented = json.dumps({"question": "How long to practise?", "answer": "Two hours.",
                               "quote": "Practise for two hours every single morning."})
        # passage 1: good; passage 2: invented quote, then again on retry; passage 3: bad JSON twice
        self.use_replies(good, invented, invented, "not json at all", "still not json")
        chunks = [dict(CHUNK, id=i) for i in range(3)]
        with tempfile.TemporaryDirectory() as d:
            q = quiz.generate("THREE Quieting Self 1", chunks, n=3, quiz_dir=d)
            self.assertEqual((len(q["questions"]), q["rejected"]), (1, 2))
            self.assertEqual(quiz.load_quiz("THREE Quieting Self 1", quiz_dir=d)["questions"][0]["id"],
                             "three-quieting-self-1-0")
            self.assertIsNone(quiz.load_quiz("Some Other Chapter", quiz_dir=d))

    def test_grade_accepts_known_verdicts_only(self):
        question = {"question": "Q?", "answer": "A.", "quote": "It should be observed calmly."}
        self.use_replies('{"verdict": "Partly", "feedback": "Missing why."}', '{"verdict": "maybe"}')
        self.assertEqual(quiz.grade(question, "my answer"), {"verdict": "partly", "feedback": "Missing why."})
        with self.assertRaises(ValueError):
            quiz.grade(question, "my answer")


if __name__ == "__main__":
    unittest.main()
