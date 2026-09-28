"""
llm.py — the one place that talks to a language model.

Two backends behind one function, chat(system, user):
  - "anthropic": Claude via the Anthropic SDK (needs ANTHROPIC_API_KEY)
  - "ollama":    a local model through Ollama's HTTP API (free, private; needs `ollama serve`)

Choose with BOOKMIND_PROVIDER=anthropic|ollama. The default, "auto", uses Claude when an API
key is set and Ollama otherwise. Any failure raises an exception, so every caller can fall
back to its offline mode (e.g. extractive answers).

Settings (environment variables):
  BOOKMIND_PROVIDER        auto | anthropic | ollama          (default auto)
  BOOKMIND_MODEL           Claude model                        (default claude-opus-5)
  BOOKMIND_OLLAMA_MODEL    Ollama model                        (default qwen2.5:7b)
  BOOKMIND_OLLAMA_URL      where Ollama listens                (default http://127.0.0.1:11434)
  BOOKMIND_OLLAMA_TIMEOUT  seconds to wait for a local answer  (default 600; a 7B model
                           on a laptop CPU took up to ~2 minutes per answer)
  BOOKMIND_EMBED_MODEL     Ollama embedding model for verify.py (default nomic-embed-text)
"""
import json
import os
import urllib.request

PROVIDER = os.environ.get("BOOKMIND_PROVIDER", "auto")
CLAUDE_MODEL = os.environ.get("BOOKMIND_MODEL", "claude-opus-5")
OLLAMA_MODEL = os.environ.get("BOOKMIND_OLLAMA_MODEL", "qwen2.5:7b")
OLLAMA_URL = os.environ.get("BOOKMIND_OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_TIMEOUT_S = float(os.environ.get("BOOKMIND_OLLAMA_TIMEOUT", "600"))
EMBED_MODEL = os.environ.get("BOOKMIND_EMBED_MODEL", "nomic-embed-text")


def provider():
    """Which backend chat() will use."""
    if PROVIDER != "auto":
        return PROVIDER
    has_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
    return "anthropic" if has_key else "ollama"


def model_name():
    """The model chat() will call, for labels like "LLM (ollama: qwen2.5:7b)"."""
    return CLAUDE_MODEL if provider() == "anthropic" else OLLAMA_MODEL


def available():
    """True if the chosen backend looks usable, without spending a model call."""
    if provider() == "anthropic":
        return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/version", timeout=3):
            return True
    except OSError:
        return False


def chat(system, user, max_tokens=1024, temperature=0.0, json_mode=False):
    """Send one system prompt + one user message; return the model's text.

    json_mode=True asks Ollama to constrain its output to valid JSON (Claude follows the
    prompt's JSON instructions without it). Callers must still parse and validate.
    """
    backend = provider()
    if backend == "anthropic":
        return _claude(system, user, max_tokens, temperature)
    if backend == "ollama":
        return _ollama(system, user, max_tokens, temperature, json_mode)
    raise ValueError(f"unknown BOOKMIND_PROVIDER: {backend!r} (use anthropic or ollama)")


def _claude(system, user, max_tokens, temperature):
    import anthropic

    client = anthropic.Anthropic()  # resolves ANTHROPIC_API_KEY or an `ant` profile
    resp = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(b.text for b in resp.content if b.type == "text").strip()


def _ollama(system, user, max_tokens, temperature, json_mode=False):
    body = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        # num_ctx: Ollama's default context window can be smaller than five passages
        # plus instructions; 8192 tokens keeps the prompt from being silently cut.
        "options": {"temperature": temperature, "num_predict": max_tokens, "num_ctx": 8192},
    }
    if json_mode:
        body["format"] = "json"
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT_S) as resp:
        payload = json.load(resp)
    return payload["message"]["content"].strip()


def embed(texts, timeout=120):
    """Embedding vectors for `texts` from a local Ollama embedding model (one request).

    Embeddings always come from Ollama, even when answers come from Claude, because the
    Anthropic API has no embedding endpoint. Raises if Ollama is unreachable.
    """
    body = {"model": EMBED_MODEL, "input": list(texts)}
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/embed",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)["embeddings"]
