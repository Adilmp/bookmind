"""
ui.py — Week 4: a small Streamlit demo over the BookMind API.

Tabs: "Ask" (grounded, cited answers). The UI is a thin client — it talks to the FastAPI service over
HTTP, so the same backend powers the demo, curl, and any future frontend.

Run:  streamlit run src/ui.py
      BOOKMIND_API=http://localhost:8000 streamlit run src/ui.py   # custom backend
"""
import os

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

(ask_tab,) = st.tabs(["Ask"])

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
            st.markdown(f"### Answer\n{r['answer']}")
            st.caption(f"mode: {r['mode']}")
            with st.expander(f"Sources ({len(r['sources'])})", expanded=True):
                for i, s in enumerate(r["sources"], 1):
                    st.markdown(f"**[{i}] « {s['chapter']} »** · chunk #{s['chunk_id']} "
                                f"· score {s['score']}")
                    st.write(s["text"])
