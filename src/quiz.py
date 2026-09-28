"""
quiz.py — "Quiz me": study questions from the book, with spaced review.

1. Generate. For a chapter, pick passages spread across it and ask the LLM for ONE question
   per passage, a short answer, and the exact quote from the passage that answers it.
   A question is kept only if its quote is a complete sentence that really appears in the
   passage (checked in code, not by the model), so a question can't rest on something the
   book doesn't say or on a half-sentence fragment. A rejected passage gets one retry.
   Questions are saved in data/quiz/ (git-ignored: they're derived from the book's text).
2. Review. Questions you miss come back sooner (Leitner boxes: 0, 1, 3, 7, 14 days).
   Progress is saved in data/quiz_progress.json.
3. Grade (optional). The LLM compares your answer with the reference answer and quote.
   Self-grading ("I got it" / "I missed it") is instant and needs no model.

Run:  python src/quiz.py "Quieting Self 1"        # generate (slow with a local model)
"""
import datetime as dt
import json
import os
import re
import sys

import llm
from search import load_chunks
from timeline import is_content

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
QUIZ_DIR = os.path.join(DATA_DIR, "quiz")
PROGRESS_PATH = os.path.join(DATA_DIR, "quiz_progress.json")
INTERVALS_DAYS = [0, 1, 3, 7, 14]   # Leitner box 1..5: when a question comes back
MIN_WORDS = 100                      # skip short tail chunks: too little to ask about
MIN_QUOTE_WORDS = 5
RETRY_TEMPERATURES = (0.0, 0.5)   # first try, then one retry with a little variety


# ---------- chapters and passages ----------

def chapters(chunks):
    """Content chapters in reading order (front and back matter skipped)."""
    return list(dict.fromkeys(c["chapter"] for c in chunks if is_content(c["chapter"])))


def slug(chapter):
    return re.sub(r"[^a-z0-9]+", "-", chapter.lower()).strip("-")


def pick_passages(chunks, chapter, n):
    """Up to n passages from the chapter, spread evenly from start to end."""
    pool = [c for c in chunks if c["chapter"] == chapter and len(c["text"].split()) >= MIN_WORDS]
    if len(pool) <= n:
        return pool
    return [pool[round(i * (len(pool) - 1) / (n - 1))] for i in range(n)] if n > 1 else pool[:1]


# ---------- checking what the model wrote ----------

def normalise(text):
    """Lowercase, unify curly quotes and dashes, collapse whitespace: for quote matching."""
    text = text.lower().translate(str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "—": "-", "–": "-"}))
    return re.sub(r"\s+", " ", text).strip()


def quote_in_passage(quote, passage):
    """True if the quote is a complete sentence (5+ words, ending in . ! ? or …) that appears
    in the passage, ignoring case, curly quotes and spacing."""
    q = normalise(quote).strip(" \"'")
    if not q.endswith((".", "!", "?", "…")):
        return False  # a fragment like "As soon as we reflect," makes a poor question
    q = q.rstrip(".!?…")
    return len(q.split()) >= MIN_QUOTE_WORDS and q in normalise(passage)


def parse_json(raw):
    """Parse a model's JSON reply: strips code fences; takes the first item of a list."""
    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    data = json.loads(raw)
    if isinstance(data, list):
        data = data[0] if data else {}
    if not isinstance(data, dict):
        raise ValueError("expected a JSON object")
    return data


# ---------- 1. generate ----------

QUESTION_SYSTEM = "You write study questions about a book. Reply with JSON only."


def question_prompt(chapter, passage):
    return (
        f'Passage from the chapter "{chapter}":\n"""\n{passage}\n"""\n\n'
        "Write ONE question that a reader should be able to answer after reading this passage. "
        "Test an idea or a piece of advice, not a trivial detail.\n"
        'Reply with JSON: {"question": "...", "answer": "one or two sentences", '
        '"quote": "one complete sentence from the passage that answers it, copied word for word"}'
    )


def make_question(chapter, chunk):
    """Ask the model for one question about one passage, retrying once. Returns a question
    dict, or None if every reply was unusable (bad JSON, missing fields, or a quote that
    isn't a complete sentence of the passage)."""
    for temperature in RETRY_TEMPERATURES:
        try:
            data = parse_json(llm.chat(QUESTION_SYSTEM, question_prompt(chapter, chunk["text"]),
                                       max_tokens=300, temperature=temperature, json_mode=True))
        except (ValueError, json.JSONDecodeError):
            continue
        question, answer, quote = (str(data.get(k, "")).strip() for k in ("question", "answer", "quote"))
        if question and answer and quote_in_passage(quote, chunk["text"]):
            return {"id": f"{slug(chapter)}-{chunk['id']}", "chapter": chapter, "chunk_id": chunk["id"],
                    "question": question, "answer": answer, "quote": quote}
    return None


def generate(chapter, chunks, n=6, quiz_dir=QUIZ_DIR):
    """Generate and save questions for a chapter. Returns the saved quiz, with a count of the
    replies that were rejected by the quote check."""
    passages = pick_passages(chunks, chapter, n)
    if not passages:
        raise ValueError(f"no passages found for chapter {chapter!r}")
    questions, rejected = [], 0
    for chunk in passages:
        q = make_question(chapter, chunk)
        if q:
            questions.append(q)
        else:
            rejected += 1
    quiz = {"chapter": chapter, "model": f"{llm.provider()}: {llm.model_name()}",
            "created": dt.date.today().isoformat(), "rejected": rejected, "questions": questions}
    os.makedirs(quiz_dir, exist_ok=True)
    with open(os.path.join(quiz_dir, slug(chapter) + ".json"), "w", encoding="utf-8") as f:
        json.dump(quiz, f, ensure_ascii=False, indent=1)
    return quiz


def load_quiz(chapter, quiz_dir=QUIZ_DIR):
    """The saved quiz for a chapter, or None if it hasn't been generated yet."""
    path = os.path.join(quiz_dir, slug(chapter) + ".json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------- 2. review (Leitner boxes) ----------

def load_progress(path=PROGRESS_PATH):
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_progress(progress, path=PROGRESS_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=1)


def review(progress, question_id, correct, today=None):
    """Record one answer. Right: move up a box (come back later). Wrong: back to box 1 (today)."""
    today = today or dt.date.today()
    p = progress.get(question_id, {"box": 1, "seen": 0, "right": 0})
    p["box"] = min(p["box"] + 1, len(INTERVALS_DAYS)) if correct else 1
    p["seen"] += 1
    p["right"] += int(correct)
    p["due"] = (today + dt.timedelta(days=INTERVALS_DAYS[p["box"] - 1])).isoformat()
    progress[question_id] = p
    return p


def due(questions, progress, today=None):
    """Questions to ask now: never seen, or due today or earlier. Lowest box first."""
    today = (today or dt.date.today()).isoformat()
    pending = [q for q in questions
               if q["id"] not in progress or progress[q["id"]]["due"] <= today]
    return sorted(pending, key=lambda q: progress.get(q["id"], {}).get("box", 0))


# ---------- 3. grade (optional) ----------

GRADE_SYSTEM = "You grade a student's answer to a study question about a book. Reply with JSON only."
VERDICTS = ("correct", "partly", "incorrect")


def grade(question, user_answer):
    """Compare the student's answer with the reference answer and the book's quote.
    Returns {"verdict": correct|partly|incorrect, "feedback": "..."}; raises if unusable."""
    prompt = (
        f"Question: {question['question']}\n"
        f"Reference answer: {question['answer']}\n"
        f'The book says: "{question["quote"]}"\n'
        f"Student's answer: {user_answer}\n\n"
        "Judge only whether the student's answer captures the idea in the reference answer; "
        "wording doesn't matter.\n"
        'Reply with JSON: {"verdict": "correct" or "partly" or "incorrect", '
        '"feedback": "one sentence on what was right or missing"}'
    )
    data = parse_json(llm.chat(GRADE_SYSTEM, prompt, max_tokens=200, json_mode=True))
    verdict = str(data.get("verdict", "")).strip().lower()
    if verdict not in VERDICTS:
        raise ValueError(f"unexpected verdict: {verdict!r}")
    return {"verdict": verdict, "feedback": str(data.get("feedback", "")).strip()}


if __name__ == "__main__":
    wanted = " ".join(sys.argv[1:]) or "Quieting Self 1"
    all_chunks = load_chunks()
    chapter = next((c for c in chapters(all_chunks) if wanted.lower() in c.lower()), None)
    if not chapter:
        sys.exit(f"no chapter matches {wanted!r}")
    quiz = generate(chapter, all_chunks)
    print(f"{chapter}: {len(quiz['questions'])} questions kept, {quiz['rejected']} rejected")
    for q in quiz["questions"]:
        print(f"\n  Q: {q['question']}\n  A: {q['answer']}\n  quote: {q['quote']}")
