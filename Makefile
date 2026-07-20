.PHONY: setup validate split train evaluate rank bundle ablate test lint serve reproduce

setup:
	uv sync --locked --all-groups

validate:
	uv run --locked retention validate

split:
	uv run --locked retention split

train:
	uv run --locked retention train

evaluate:
	uv run --locked retention evaluate

rank:
	uv run --locked retention rank

bundle:
	uv run --locked retention bundle

ablate:
	uv run --locked retention ablate

test:
	uv run --locked pytest -v

lint:
	uv run --locked ruff check src/ tests/ && uv run --locked mypy src/

serve:
	uv run --locked uvicorn retention.serving.app:app --host 0.0.0.0 --port 8000

reproduce:
	dvc repro && uv run --locked pytest -v
