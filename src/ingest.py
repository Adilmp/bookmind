"""
ingest.py — Turn an EPUB into clean, citeable text chunks.

Pipeline:  EPUB (zip of XHTML)  ->  reading-order text  ->  ~180-word chunks
Each chunk carries metadata (chapter title, source file, chunk id) so every
answer we generate later can cite exactly where it came from.

No heavy deps: standard library + BeautifulSoup only.
Run:  python src/ingest.py data/raw/inner-game-of-tennis.epub
"""
import sys
import json
import re
import zipfile
import os
from urllib.parse import unquote
from xml.etree import ElementTree as ET
from bs4 import BeautifulSoup

CHUNK_WORDS = 180        # target words per chunk
CHUNK_OVERLAP = 30       # words of overlap so ideas aren't split mid-thought


def _local(tag):
    """Strip the XML namespace: '{...}rootfile' -> 'rootfile'."""
    return tag.rsplit("}", 1)[-1]


def _iter(root, name):
    """Yield all descendants whose local tag name matches (namespace-agnostic)."""
    for el in root.iter():
        if _local(el.tag) == name:
            yield el


def _opf_path(z):
    """Find the OPF (package) file via META-INF/container.xml."""
    root = ET.fromstring(z.read("META-INF/container.xml"))
    for rf in _iter(root, "rootfile"):
        return rf.attrib["full-path"]
    raise RuntimeError("No rootfile found in container.xml")


def _spine_files(z, opf_path):
    """Return content documents in READING ORDER (per the OPF spine)."""
    base = os.path.dirname(opf_path)
    root = ET.fromstring(z.read(opf_path))
    manifest = {
        item.attrib["id"]: item.attrib["href"]
        for item in _iter(root, "item")
        if "id" in item.attrib and "href" in item.attrib
    }
    ordered = []
    for ref in _iter(root, "itemref"):
        href = manifest.get(ref.attrib.get("idref"))
        if href:
            full = os.path.normpath(os.path.join(base, href)).replace("\\", "/")
            ordered.append(full)
    return ordered


def _toc_titles(z, opf_path):
    """Map source-file -> chapter title using toc.ncx (best-effort)."""
    base = os.path.dirname(opf_path)
    titles = {}
    ncx = next((n for n in z.namelist() if n.endswith(".ncx")), None)
    if not ncx:
        return titles
    root = ET.fromstring(z.read(ncx))
    for nav in _iter(root, "navPoint"):
        label = next((t for t in _iter(nav, "text")), None)
        content = next((c for c in _iter(nav, "content")), None)
        if label is None or content is None:
            continue
        src = unquote(content.attrib.get("src", "")).split("#")[0]
        full = os.path.normpath(os.path.join(base, src)).replace("\\", "/")
        titles.setdefault(full, (label.text or "").strip())
    return titles


def _clean_paragraphs(html):
    """Extract readable paragraphs from one XHTML document."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    blocks = []
    for p in soup.find_all(["p", "h1", "h2", "h3", "blockquote", "li"]):
        text = re.sub(r"\s+", " ", p.get_text(" ", strip=True)).strip()
        if len(text) >= 2:
            blocks.append(text)
    if not blocks:  # fallback: whole-document text
        text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True)).strip()
        if text:
            blocks = [text]
    return blocks


def _chunk(words, size, overlap):
    """Yield overlapping word windows."""
    step = max(1, size - overlap)
    for start in range(0, len(words), step):
        window = words[start:start + size]
        if window:
            yield " ".join(window)
        if start + size >= len(words):
            break


def ingest(epub_path, out_path):
    z = zipfile.ZipFile(epub_path)
    opf = _opf_path(z)
    files = _spine_files(z, opf)
    titles = _toc_titles(z, opf)

    chunks = []
    current_chapter = "Front Matter"
    cid = 0
    for f in files:
        try:
            html = z.read(f).decode("utf-8", "ignore")
        except KeyError:
            continue
        # update chapter title if the TOC names this file
        if titles.get(f):
            current_chapter = titles[f]

        paragraphs = _clean_paragraphs(html)
        if not paragraphs:
            continue
        text = "\n".join(paragraphs)
        words = text.split()
        if len(words) < 20:      # skip tiny nav/cover pages
            continue
        for piece in _chunk(words, CHUNK_WORDS, CHUNK_OVERLAP):
            chunks.append({
                "id": cid,
                "chapter": current_chapter,
                "source_file": os.path.basename(f),
                "text": piece,
                "word_count": len(piece.split()),
            })
            cid += 1

    with open(out_path, "w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")

    chapters = sorted({c["chapter"] for c in chunks})
    print(f"OK  {len(chunks)} chunks from {len(files)} sections -> {out_path}")
    print(f"    {len(chapters)} chapters detected:")
    for ch in chapters:
        n = sum(1 for c in chunks if c["chapter"] == ch)
        print(f"      - {ch}  ({n} chunks)")
    return chunks


if __name__ == "__main__":
    epub = sys.argv[1] if len(sys.argv) > 1 else "data/raw/inner-game-of-tennis.epub"
    out = sys.argv[2] if len(sys.argv) > 2 else "data/chunks.jsonl"
    ingest(epub, out)
