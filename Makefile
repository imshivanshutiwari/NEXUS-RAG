.PHONY: install db fetch ingest run api evaluate test lint notebook clean

install:
	pip install -r requirements.txt

db:
	docker-compose up -d postgres

fetch:
	python data/processors/document_store.py --fetch

ingest:
	python ingestion/pipeline.py

run:
	python dashboard/app.py

api:
	uvicorn api.server:app --reload --port 8000

evaluate:
	python evaluation/benchmark_runner.py

test:
	pytest tests/ -v --color=yes

lint:
	black . --line-length 100 && flake8 . --max-line-length=100

notebook:
	jupyter notebook notebooks/

clean:
	find . -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
