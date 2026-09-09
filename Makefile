# Common development tasks. Requires `pip install -e ".[dev]"`.

PYTHON ?= python3

.PHONY: help install test lint format typecheck check traces plots clean

help:
	@echo "install    install the package with development extras"
	@echo "test       run the test suite"
	@echo "lint       run ruff (lint + format check)"
	@echo "typecheck  run mypy"
	@echo "check      lint + typecheck + test"
	@echo "format     rewrite files with ruff format"
	@echo "traces     generate the sample traces into traces/"
	@echo "clean      remove caches and build artefacts"

install:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

format:
	$(PYTHON) -m ruff format .
	$(PYTHON) -m ruff check --fix .

typecheck:
	$(PYTHON) -m mypy

check: lint typecheck test

traces:
	$(PYTHON) -m cachesim gen-traces --out-dir traces

clean:
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache .hypothesis
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
