"""
ui.py — Week 4: a small Streamlit demo over the BookMind API.

Tabs: "Ask" (grounded, cited answers, each sentence checked against the passages)
"Idea timeline" (where an idea appears,
chapter by chapter) and "Quiz me" (study questions with spaced review). The UI is a thin client — it talks to the FastAPI service over
HTTP, so the same backend powers the demo, curl, and any future frontend.

Run:  streamlit run src/ui.py
      BOOKMIND_API=http://localhost:8000 streamlit run src/ui.py   # custom backend
"""
import os
import re

import requests
import streamlit as st

API = os.environ.get("BOOKMIND_API", "http://localhost:8000").rstrip("/")

st.set_page_config(page_title="BookMind", page_icon="📖", layout="wide")
st.title("📖 BookMind")
st.caption("Ask questions across a book — answers grounded in **cited passages**.")


@st.cache_data(ttl=5)
def _health():
    try:
        return requests.get(f"{API}/health", timeout=5).json()
    except Exception as e:
        return {"status": "unreachable", "detail": str(e)}


h = _health()
if h.get("status") == "ok":
    st.sidebar.success(f"API online · {h['chunks']} chunks · {h['chapters']} chapters")
else:
    st.sidebar.error(f"API {h.get('status', '?')}: {h.get('detail', '')}")
    st.sidebar.caption(f"Backend: {API}")



def _md_escape(text):
    """Escape characters that Markdown would treat as formatting."""
    return re.sub(r"([\\`*_{}\[\]<>#+|~])", r"\\\1", text)


def _highlight(text, spans):
    """Bold the [start, end) character spans of `text`, escaping everything else."""
    out, pos = [], 0
    for start, end in spans:
        out += [_md_escape(text[pos:start]), "**", _md_escape(text[start:end]), "**"]
        pos = end
    out.append(_md_escape(text[pos:]))
    return "".join(out)


MARK = {"supported": "✅", "weak": "⚠️", "unsupported": "❌", "refusal": "💬"}


def _show_support(support):
    """Show the answer sentence by sentence, each with its evidence from the passages."""
    s = support["summary"]
    if s["sentences"]:
        line = (f"**{s['supported']} of {s['sentences']}** sentences supported by the passages"
                f" · {s['weak']} weak · {s['unsupported']} unsupported")
        if s["invalid_citations"]:
            line += f" · **{s['invalid_citations']} citation(s) don't match the sources**"
        st.markdown(line + f"  \n_checked with {support['method']}_")
    for item in support["sentences"]:
        st.markdown(f"{MARK[item['status']]} {_md_escape(item['sentence'])}")
        bad = [c["raw"] for c in item["citations"] if not c["valid"]]
        if bad:
            st.caption("Citation not found among the sources: " + ", ".join(f"[{b}]" for b in bad))
        if item["evidence"]:
            ev = item["evidence"]
            with st.expander(f"Evidence · passage [{ev['passage']}] « {ev['chapter']} » · "
                             f"similarity {item['score']}"):
                st.markdown("> " + _md_escape(ev["text"]))


ask_tab, timeline_tab, quiz_tab = st.tabs(["Ask", "Idea timeline", "Quiz me"])

with ask_tab:
    q = st.text_input("Question", "How do I stop overthinking on the court?")
    k = st.slider("Passages to retrieve (k)", 1, 10, 5)
    if st.button("Ask", type="primary"):
        with st.spinner("Retrieving passages and composing a grounded answer (a local model can take ~2 minutes)…"):
            try:
                r = requests.post(f"{API}/ask", json={"query": q, "k": k}, timeout=300).json()
            except Exception as e:
                st.error(f"Request failed: {e}")
                r = None
        if r:
            st.markdown("### Answer")
            support = r.get("support")
            if support:
                _show_support(support)
            else:
                st.markdown(r["answer"])
            st.caption(f"mode: {r['mode']}")
            with st.expander(f"Sources ({len(r['sources'])})", expanded=True):
                for i, s in enumerate(r["sources"], 1):
                    st.markdown(f"**[{i}] « {s['chapter']} »** · chunk #{s['chunk_id']} "
                                f"· score {s['score']}")
                    st.write(s["text"])

with timeline_tab:
    st.caption("Follow one idea through the book: how often each chapter mentions it, "
               "and where. A word, or an exact phrase.")
    idea = st.text_input("Idea", "Self 1")
    if st.button("Trace idea", type="primary"):
        try:
            resp = requests.post(f"{API}/timeline", json={"idea": idea}, timeout=30)
        except Exception as e:
            st.error(f"Request failed: {e}")
            resp = None
        if resp is not None and resp.status_code != 200:
            st.error(resp.json().get("detail", resp.text))
        elif resp is not None:
            t = resp.json()
            found = [c for c in t["chapters"] if c["mentions"]]
            st.markdown(f"**{t['total_mentions']}** mentions of **{_md_escape(t['idea'])}** "
                        f"in **{len(found)}** of {len(t['chapters'])} chapters.")
            top = max((c["mentions"] for c in t["chapters"]), default=0) or 1
            for c in t["chapters"]:
                name_col, bar_col, n_col = st.columns([5, 6, 1])
                name_col.markdown(_md_escape(c["chapter"]))
                bar_col.progress(c["mentions"] / top)
                n_col.markdown(f"**{c['mentions']}**")
                if c["snippets"]:
                    with st.expander(f"Where it appears ({c['passages']} passages)"):
                        for snip in c["snippets"]:
                            st.markdown("> " + _highlight(snip["text"], snip["spans"]))

VERDICT = {"correct": "✅ Correct", "partly": "⚠️ Partly right", "incorrect": "❌ Not quite"}


def _get(path, **params):
    return requests.get(f"{API}{path}", params=params, timeout=30)


with quiz_tab:
    st.caption("Study a chapter with questions made from its own passages. Answer, reveal, and "
               "mark yourself: missed questions come back sooner.")
    try:
        chapter_list = _get("/quiz/chapters").json()
    except Exception as e:
        st.error(f"Request failed: {e}")
        chapter_list = []
    if chapter_list:
        labels = {c["chapter"]: (f"{c['chapter']}  ·  {c['due']} due of {c['questions']}"
                                 if c["generated"] else f"{c['chapter']}  ·  no questions yet")
                  for c in chapter_list}
        # a fixed key keeps the selection when the labels' due counts change after an answer
        chapter = st.selectbox("Chapter", list(labels), format_func=labels.get, key="quiz_chapter")
        info = next(c for c in chapter_list if c["chapter"] == chapter)

        if not info["generated"]:
            st.info("No questions for this chapter yet. A local model takes a few minutes to write them.")
            if st.button("Generate questions", type="primary"):
                with st.spinner("Writing questions from the chapter's passages…"):
                    r = requests.post(f"{API}/quiz/generate", json={"chapter": chapter}, timeout=1200)
                if r.status_code == 200:
                    q = r.json()
                    st.success(f"{len(q['questions'])} questions kept"
                               f" ({q['rejected']} rejected: their quote wasn't in the passage).")
                    st.rerun()
                else:
                    st.error(r.json().get("detail", r.text))
        else:
            quiz = _get("/quiz", chapter=chapter).json()
            due_ids = quiz["due"]
            by_id = {q["id"]: q for q in quiz["questions"]}
            if not due_ids:
                st.success(f"All {len(by_id)} questions reviewed. They'll come back when they're due.")
            else:
                q = by_id[due_ids[0]]
                st.markdown(f"**{len(due_ids)} due** · question from « {_md_escape(q['chapter'])} »")
                st.markdown(f"#### {_md_escape(q['question'])}")
                answer = st.text_area("Your answer (optional for self-marking)", key=f"answer-{q['id']}")
                reveal_col, ai_col = st.columns(2)
                if reveal_col.button("Show answer", key=f"show-{q['id']}"):
                    st.session_state["revealed"] = q["id"]
                if ai_col.button("Check my answer with AI", key=f"ai-{q['id']}", disabled=not answer.strip()):
                    with st.spinner("Grading (a local model can take a minute)…"):
                        r = requests.post(f"{API}/quiz/grade", json={"id": q["id"], "answer": answer}, timeout=660)
                    st.session_state["grade"] = (q["id"], r.json() if r.status_code == 200
                                                 else {"error": r.json().get("detail", r.text)})
                    st.session_state["revealed"] = q["id"]

                if st.session_state.get("revealed") == q["id"]:
                    graded = st.session_state.get("grade")
                    if graded and graded[0] == q["id"]:
                        g = graded[1]
                        if "error" in g:
                            st.warning(g["error"])
                        else:
                            st.markdown(f"**{VERDICT[g['verdict']]}**: {_md_escape(g['feedback'])}")
                    st.markdown(f"**Answer:** {_md_escape(q['answer'])}")
                    st.markdown(f"> {_md_escape(q['quote'])}")
                    got_col, missed_col = st.columns(2)
                    for col, correct, label in ((got_col, True, "✅ I got it"), (missed_col, False, "❌ I missed it")):
                        if col.button(label, key=f"mark-{correct}-{q['id']}"):
                            requests.post(f"{API}/quiz/review", json={"id": q["id"], "correct": correct}, timeout=30)
                            st.session_state.pop("revealed", None)
                            st.session_state.pop("grade", None)
                            st.rerun()
