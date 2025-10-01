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
	PYTHONPATH=src $(PY) runbooks/00_build_embeddings.py --config $(CONFIG)

run-text:
	PYTHONPATH=src $(PY) runbooks/01_make_text_features.py --config $(CONFIG)

run-train:
	PYTHONPATH=src $(PY) runbooks/02_fit_text_models.py --config $(CONFIG)

run-blend:
	PYTHONPATH=src $(PY) runbooks/03_blend_with_quant.py --config $(CONFIG)

run-scores:
	PYTHONPATH=src $(PY) runbooks/04_generate_scores_csv.py --config $(CONFIG)

run-backtest:
	PYTHONPATH=src $(PY) runbooks/05_backtest_eval.py --config $(CONFIG)

# Phase 3
.PHONY: run-numeric-blend run-risk-outputs run-backtest-final
run-numeric-blend:
	PYTHONPATH=src $(PY) runbooks/06_numeric_and_blend.py --config $(CONFIG)

run-risk-outputs:
	PYTHONPATH=src $(PY) runbooks/07_apply_risk_and_save_period_csvs.py --config $(CONFIG)

run-backtest-final:
	PYTHONPATH=src $(PY) runbooks/08_backtest_from_final_weights.py

lint:
	ruff check . --fix
	ruff format .
	black --line-length 100 .
	mypy src

test:
	PYTHONPATH=src pytest --maxfail=1 --durations=25 --cov=src --cov-report=term-missing

ci: lint test
