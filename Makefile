.PHONY: install test test-fast lint format typecheck train backtest docker-build docker-run clean

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v --cov=src --cov-report=html

test-fast:
	pytest tests/ -v -x --ignore=tests/test_models.py

lint:
	ruff check src/ tests/

format:
	black src/ tests/

typecheck:
	mypy src/

train:
	python -m src.models.model_registry --config configs/model_config.yaml

backtest:
	python -m src.backtesting.backtest_engine --config configs/backtest_config.yaml

docker-build:
	docker build -t ml-alpha-lab:latest .

docker-run:
	docker-compose up

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	rm -rf htmlcov dist build
