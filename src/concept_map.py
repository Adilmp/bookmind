"""
concept_map.py — Week 2b: turn a chapter (or the whole book) into a concept map.

Two extraction paths (same pattern as answer.py):
  - LLM path: Claude extracts concept -> relation -> concept triples (labeled edges).
  - Offline path: frequency + co-occurrence over the chunks (no credentials needed),
    so the feature runs and is testable end-to-end today.

Outputs three artifacts for one chapter:
  data/concept_map.json   (nodes + edges — the data your future UI renders)
  data/concept_map.mmd    (a Mermaid diagram — paste into any Markdown viewer)
  data/concept_map.svg    (a self-contained circular-layout image — no deps, no JS)

Run:  python src/concept_map.py "Quieting Self 1"
      python src/concept_map.py            # whole book
"""
import json
import math
import os
import re
import sys
from collections import Counter
from itertools import combinations
from search import load_chunks
from bm25 import tokenize

MODEL = os.environ.get("BOOKMIND_MODEL", "claude-opus-5")

STOPWORDS = set("""
a an the and or but if then else when while of to in on at by for with from into
is are was were be been being do does did have has had it its this that these those
he she they we you i him her them his hers their our your my me us as so not no yes
one two out up down over under again more most some such can will just than too very
about after before between during without within your you're don't didn't can't what
which who whom whose how why where here there all any each few other own same only own
""".split())


# ---------- offline extraction (no credentials) ----------

def _candidate_concepts(chunks, top_n):
    """Score unigrams + bigrams by frequency; return the top_n concept phrases."""
    uni, bi = Counter(), Counter()
    for c in chunks:
        toks = [t for t in tokenize(c["text"]) if len(t) >= 4 and t not in STOPWORDS]
        uni.update(toks)
        for a, b in zip(toks, toks[1:]):
            bi[f"{a} {b}"] += 1
    scored = Counter()
    for term, n in uni.items():
        scored[term] = n
    for term, n in bi.items():
        if n >= 2:                    # bigrams must recur; weight them above unigrams
            scored[term] = n * 2
    return [term for term, _ in scored.most_common(top_n)]


def _cooccurrence_edges(concepts, chunks, min_count=2, max_edges=25):
    """Edge between two concepts that appear in the same chunk >= min_count times."""
    texts = [c["text"].lower() for c in chunks]
    present = {c: [c in t for t in texts] for c in concepts}
    pair_counts = Counter()
    for a, b in combinations(concepts, 2):
        co = sum(1 for pa, pb in zip(present[a], present[b]) if pa and pb)
        if co >= min_count:
            pair_counts[(a, b)] = co
    edges = [{"source": a, "target": b, "weight": w, "relation": "related to"}
             for (a, b), w in pair_counts.most_common(max_edges)]
    return edges


def extract_offline(chunks, top_n=12):
    concepts = _candidate_concepts(chunks, top_n)
    edges = _cooccurrence_edges(concepts, chunks)
    return {"nodes": [{"id": c} for c in concepts], "edges": edges}


# ---------- LLM extraction (labeled relations) ----------

def extract_llm(chunks, top_n=12):
    import anthropic

    client = anthropic.Anthropic()
    text = "\n\n".join(c["text"] for c in chunks)[:12000]  # keep the prompt bounded
    prompt = (
        "Extract the key ideas of the following book text as a concept map.\n"
        f"Return STRICT JSON: a list of at most {top_n} objects, each "
        '{"source": "concept", "relation": "short verb phrase", "target": "concept"}.\n'
        "Use short concept labels (1-3 words). No prose, JSON only.\n\n"
        f"TEXT:\n{text}"
    )
    resp = client.messages.create(
        model=MODEL, max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = "".join(b.text for b in resp.content if b.type == "text").strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
    triples = json.loads(raw)
    nodes = {}
    edges = []
    for t in triples:
        s, r, o = t["source"].strip(), t.get("relation", "related to").strip(), t["target"].strip()
        nodes[s] = nodes.get(s, {"id": s})
        nodes[o] = nodes.get(o, {"id": o})
        edges.append({"source": s, "target": o, "weight": 1, "relation": r})
    return {"nodes": list(nodes.values()), "edges": edges}


# ---------- renderers ----------

def to_mermaid(graph):
    ids = {n["id"]: f"n{i}" for i, n in enumerate(graph["nodes"])}
    lines = ["graph LR"]
    for n in graph["nodes"]:
        lines.append(f'  {ids[n["id"]]}["{n["id"]}"]')
    for e in graph["edges"]:
        if e["source"] in ids and e["target"] in ids:
            lines.append(f'  {ids[e["source"]]} -->|{e["relation"]}| {ids[e["target"]]}')
    return "\n".join(lines)


def to_svg(graph, size=760):
    """Self-contained circular-layout SVG — no JS, no external assets."""
    nodes = graph["nodes"]
    n = len(nodes)
    cx = cy = size / 2
    r = size / 2 - 120
    pos = {}
    for i, node in enumerate(nodes):
        ang = 2 * math.pi * i / max(1, n) - math.pi / 2
        pos[node["id"]] = (cx + r * math.cos(ang), cy + r * math.sin(ang))
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {size} {size}" '
             f'font-family="system-ui,sans-serif">',
             f'<rect width="{size}" height="{size}" fill="#0f1117"/>']
    for e in graph["edges"]:
        if e["source"] in pos and e["target"] in pos:
            x1, y1 = pos[e["source"]]
            x2, y2 = pos[e["target"]]
            parts.append(f'<line x1="{x1:.0f}" y1="{y1:.0f}" x2="{x2:.0f}" y2="{y2:.0f}" '
                         f'stroke="#3b82f6" stroke-opacity="0.35" stroke-width="1.5"/>')
    for node in nodes:
        x, y = pos[node["id"]]
        label = node["id"]
        w = 8 * len(label) + 16
        parts.append(f'<rect x="{x-w/2:.0f}" y="{y-13:.0f}" width="{w:.0f}" height="26" rx="13" '
                     f'fill="#1e293b" stroke="#3b82f6"/>')
        parts.append(f'<text x="{x:.0f}" y="{y+4:.0f}" fill="#e2e8f0" font-size="13" '
                     f'text-anchor="middle">{label}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


# ---------- driver ----------

def build(chapter=None, top_n=12):
    chunks = load_chunks()
    if chapter:
        sel = [c for c in chunks if chapter.lower() in c["chapter"].lower()]
        chunks = sel or chunks
    try:
        graph = extract_llm(chunks, top_n)
        mode = f"LLM ({MODEL})"
    except Exception as e:
        graph = extract_offline(chunks, top_n)
        mode = f"offline (LLM unavailable: {type(e).__name__})"
    return graph, mode


if __name__ == "__main__":
    chapter = " ".join(sys.argv[1:]) or None
    graph, mode = build(chapter)
    here = os.path.join(os.path.dirname(__file__), "..", "data")
    with open(os.path.join(here, "concept_map.json"), "w") as f:
        json.dump(graph, f, indent=2)
    with open(os.path.join(here, "concept_map.mmd"), "w") as f:
        f.write(to_mermaid(graph))
    with open(os.path.join(here, "concept_map.svg"), "w") as f:
        f.write(to_svg(graph))
    print(f'  Concept map for: {chapter or "whole book"}   [{mode}]')
    print(f'  {len(graph["nodes"])} concepts, {len(graph["edges"])} links '
          f'-> data/concept_map.{{json,mmd,svg}}\n')
    print(to_mermaid(graph))
