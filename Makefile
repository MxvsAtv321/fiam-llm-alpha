.PHONY: setup init run-embeddings run-text run-train run-blend run-scores run-backtest lint test ci

CONDA_ENV=fiam-llm-alpha
PY=python
CONFIG=config/defaults.yaml

setup:
	@if ! conda env list | grep -q "$(CONDA_ENV)"; then conda env create -f environment.yml; else conda env update -f environment.yml; fi
	@pre-commit install

init:
	@git lfs install
	@dvc init -q || true

run-embeddings:
	$(PY) runbooks/00_build_embeddings.py --config $(CONFIG)

run-text:
	$(PY) runbooks/01_make_text_features.py --config $(CONFIG)

run-train:
	$(PY) runbooks/02_fit_text_models.py --config $(CONFIG)

run-blend:
	$(PY) runbooks/03_blend_with_quant.py --config $(CONFIG)

run-scores:
	$(PY) runbooks/04_generate_scores_csv.py --config $(CONFIG)

run-backtest:
	$(PY) runbooks/05_backtest_eval.py --config $(CONFIG)

lint:
	ruff check . --fix
	ruff format .
	black --line-length 100 .
	mypy src

test:
	pytest --maxfail=1 --durations=25 --cov=src --cov-report=term-missing

ci: lint test
