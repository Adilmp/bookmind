# BookMind — one-command runs (Week 4).
.PHONY: help install api ui run docker docker-run eval

help:
	@echo "make install     install Python deps"
	@echo "make api         run the FastAPI service on :8000"
	@echo "make ui          run the Streamlit demo on :8501 (needs the API)"
	@echo "make run         run API + UI together"
	@echo "make docker      build the API image"
	@echo "make docker-run  run the API image (mounts ./data, passes ANTHROPIC_API_KEY)"
	@echo "make eval        run the evaluation harness"

install:
	pip install -r requirements.txt

api:
	uvicorn api:app --host 0.0.0.0 --port 8000 --app-dir src

ui:
	streamlit run src/ui.py

# Start the API in the background, then the UI; Ctrl-C stops both.
run:
	uvicorn api:app --host 0.0.0.0 --port 8000 --app-dir src & echo $$! > .api.pid; \
	trap 'kill `cat .api.pid` 2>/dev/null; rm -f .api.pid' EXIT; \
	sleep 1; streamlit run src/ui.py

docker:
	docker build -t bookmind .

docker-run:
	docker run --rm -p 8000:8000 -v "$(PWD)/data:/app/data" \
	  -e ANTHROPIC_API_KEY="$(ANTHROPIC_API_KEY)" bookmind

eval:
	python src/evaluate.py
