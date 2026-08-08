# BookMind API — Week 4 deployable service.
#
# The image contains only code, never the book: data/ is git-ignored (copyright)
# and mounted at runtime. search.py resolves ../data relative to /app/src, so mount
# the corpus at /app/data.
#
#   docker build -t bookmind .
#   docker run --rm -p 8000:8000 -v "$(pwd)/data:/app/data" \
#     -e ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY" bookmind
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY src/ ./src/

# Run from src/ so the modules' bare imports (from search import ...) resolve.
WORKDIR /app/src
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
