# QueryLane Makefile
# Common developer workflows: test, lint, format, eval, build, and docker.

.PHONY: help test test-cov lint format check-secrets eval build check-pkg clean models-onnx models-all docker-cloud docker-multiarch explain web

help:
	@echo "QueryLane Development Commands:"
	@echo "  make test              Run test suite"
	@echo "  make test-cov          Run test suite with coverage report"
	@echo "  make lint              Run ruff linter"
	@echo "  make format            Run ruff code formatter"
	@echo "  make check-secrets     Scan repository for accidental secrets"
	@echo "  make eval              Run deterministic dev-set evaluation"
	@echo "  make build             Build sdist and wheel packages"
	@echo "  make check-pkg         Verify distribution packages with twine"
	@echo "  make models-onnx       Download & verify ONNX specialist models"
	@echo "  make models-all        Download & verify all ONNX and GGUF models"
	@echo "  make docker-cloud      Build cloud-only Docker image"
	@echo "  make docker-multiarch  Build multi-arch Docker image (amd64, arm64)"
	@echo "  make explain           Run routing explainer for a sample query"
	@echo "  make web               Start the interactive local web demo"
	@echo "  make clean             Remove build artifacts, caches, and dist/"

test:
	pytest

test-cov:
	pytest --cov=agent --cov-report=term-missing

lint:
	ruff check .

format:
	ruff format .

check-secrets:
	python scripts/check_secrets.py --all

eval:
	python -m eval.deterministic

build:
	python -m build

check-pkg: build
	twine check dist/*

models-onnx:
	python scripts/download_models.py onnx

models-all:
	python scripts/download_models.py all

docker-cloud:
	docker build -t querylane:latest .

docker-multiarch:
	docker buildx build --platform linux/amd64,linux/arm64 -t querylane:latest .

explain:
	python -m agent.explain "Extract all named entities: Steve Jobs visited Paris on June 12th."

web:
	python -m agent.web_demo --port 8000

clean:
	python -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').glob('*.egg-info')]; [shutil.rmtree(p, ignore_errors=True) for p in [pathlib.Path('build'), pathlib.Path('dist'), pathlib.Path('.pytest_cache'), pathlib.Path('.ruff_cache')]]"
