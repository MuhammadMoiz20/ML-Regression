.PHONY: help install install-dev test test-cov lint format type-check clean build docker-build docker-run docs

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install the package
	pip install -e .

install-dev: ## Install development dependencies
	pip install -e ".[dev]"
	pre-commit install

test: ## Run tests
	pytest

test-cov: ## Run tests with coverage
	pytest --cov=src/ml_regression --cov-report=html --cov-report=term-missing

lint: ## Run linting
	flake8 src/ tests/ examples/
	black --check src/ tests/ examples/

format: ## Format code
	black src/ tests/ examples/
	isort src/ tests/ examples/

type-check: ## Run type checking
	mypy src/ml_regression

clean: ## Clean up build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: ## Build the package
	python -m build

docker-build: ## Build Docker image
	docker build -t ml-regression .

docker-run: ## Run Docker container
	docker run -it --rm -v $(PWD):/app ml-regression

docker-test: ## Run tests in Docker
	docker-compose run test

docker-dev: ## Start development environment
	docker-compose up -d app

docs: ## Build documentation
	cd docs && make html

examples: ## Run all examples
	python examples/basic_linear_regression.py
	python examples/robust_regression.py
	python examples/regularized_regression.py

check: lint type-check test ## Run all checks

ci: clean install-dev check build ## Run CI pipeline locally
