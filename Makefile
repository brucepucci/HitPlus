.PHONY: lint format test test-regression

lint:
	uv run black --check src/ tests/
	uv run ruff check src/ tests/

format:
	uv run black src/ tests/
	uv run ruff check --fix src/ tests/

test:
	uv run pytest tests/unit/ --no-header -rN || test $$? -eq 5

test-regression:
	uv run pytest tests/regression/ -m regression --no-header -rN || test $$? -eq 5
