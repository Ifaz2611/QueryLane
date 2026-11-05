# QueryLane justfile
# Modern command runner recipes (test, lint, format, eval, build, and docker)

# Default: list recipes
default:
    @just --list

# Run pytest suite
test:
    pytest

# Run pytest with code coverage
test-cov:
    pytest --cov=agent --cov-report=term-missing

# Run ruff linter
lint:
    ruff check .

# Format code with ruff
format:
    ruff format .

# Check tracked files for secrets
check-secrets:
    python scripts/check_secrets.py --all

# Run deterministic dev set evaluation
eval:
    python -m eval.deterministic

# Build sdist and wheel package distributions
build:
    python -m build

# Check built packages with twine
check-pkg: build
    twine check dist/*

# Download and verify ONNX specialist models (NER and SST-2)
models-onnx:
    python scripts/download_models.py onnx

# Download and verify all ONNX and GGUF models
models-all:
    python scripts/download_models.py all

# Build cloud Docker image
docker-cloud:
    docker build -t querylane:latest .

# Build multi-architecture Docker image (amd64 and arm64)
docker-multiarch:
    docker buildx build --platform linux/amd64,linux/arm64 -t querylane:latest .

# Run routing explainer for a sample prompt
explain prompt="Extract all named entities: Steve Jobs visited Paris on June 12th.":
    python -m agent.explain "{{prompt}}"

# Start the interactive local web demo
web port="8000":
    python -m agent.web_demo --port {{port}}

# Remove temporary build artifacts, egg-info, and caches
clean:
    python -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').glob('*.egg-info')]; [shutil.rmtree(p, ignore_errors=True) for p in [pathlib.Path('build'), pathlib.Path('dist'), pathlib.Path('.pytest_cache'), pathlib.Path('.ruff_cache')]]"
