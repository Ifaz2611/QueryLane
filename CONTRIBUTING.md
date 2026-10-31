# Contributing to QueryLane

Thank you for your interest in contributing to QueryLane! This document provides guidelines and information for contributors.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Workflow](#development-workflow)
4. [Coding Standards](#coding-standards)
5. [Testing](#testing)
6. [Documentation](#documentation)
7. [Submitting Changes](#submitting-changes)
8. [Areas for Contribution](#areas-for-contribution)

## Code of Conduct

Be respectful, constructive, and inclusive. We aim to maintain a welcoming environment for all contributors.

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Git
- A Fireworks API key (for testing cloud functionality)

### Development Setup

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/your-username/QueryLane.git
   cd QueryLane
   ```

3. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # macOS/Linux
   .venv\Scripts\activate  # Windows
   ```

4. Install development dependencies:
   ```bash
   pip install -e ".[dev,specialists]"
   ```

5. Copy `.env.example` to `.env` and add your API credentials:
   ```bash
   cp .env.example .env
   # Edit .env with your API key
   ```

6. Run tests to verify your setup:
   ```bash
   pytest
   ```

## Development Workflow

### Branch Strategy

- `main` - Stable production code
- Create feature branches from `main`: `feature/your-feature-name`
- Create bugfix branches: `fix/your-bug-fix`
- Create documentation branches: `docs/your-doc-change`

### Making Changes

1. Create a new branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes following the coding standards below

3. Test your changes:
   ```bash
   pytest
   ruff check .
   ```

4. Commit your changes with clear messages:
   ```bash
   git add .
   git commit -m "feat: add new category for XYZ tasks"
   ```

5. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

6. Create a pull request on GitHub

## Coding Standards

### Style Guide

QueryLane uses `ruff` for linting with the following configuration:

- Line length: 100 characters
- Target Python version: 3.10
- Enabled rules: E (errors), F (pyflakes), W (warnings)

Run the linter:
```bash
ruff check .
```

Auto-fix issues:
```bash
ruff check --fix .
```

### Code Organization

- **Router logic** goes in `agent/router.py`
- **Specialists** go in `agent/specialists.py`
- **Cloud client** goes in `agent/cloud.py`
- **Local model** goes in `agent/local.py`
- **Prompts** go in `agent/prompts.py`
- **Tests** go in `tests/` directory

### Naming Conventions

- Functions and variables: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`
- Private members: `_leading_underscore`

### Documentation

- Add docstrings to all public functions and classes
- Use Google-style docstrings
- Include type hints where appropriate

Example:
```python
def classify_prompt(prompt: str) -> str:
    """Classify a prompt into a task category.

    Args:
        prompt: The input prompt text.

    Returns:
        The category name (e.g., 'factual', 'math', 'sentiment').
    """
    pass
```

## Testing

### Test Structure

Tests are located in the `tests/` directory and use `pytest`.

### Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_router.py

# Run specific test
pytest tests/test_router.py::test_classify_prompt

# Run with coverage
pytest --cov=agent
```

### Writing Tests

- Test files should be named `test_*.py`
- Test functions should be named `test_*`
- Use descriptive test names
- Include both positive and negative test cases
- Mock external API calls in tests

Example:
```python
def test_classify_sentiment_prompt():
    """Test that sentiment prompts are correctly classified."""
    prompt = "Classify the sentiment: I love this product!"
    category = classify_prompt(prompt)
    assert category == "sentiment"
```

### Test Coverage

Aim for high test coverage on new code. Current test suite covers:
- Router classification logic
- PAL safe evaluation
- Instruction stripping
- Specialist gates
- Data validation
- Main loop

Areas needing coverage (from TODO.md):
- Cloud client
- Local calibration
- Deadline handling

## Documentation

### Updating Documentation

When making changes that affect user-facing behavior:

1. Update relevant sections in `README.md`
2. Add examples to `GETTING_STARTED.md` if needed
3. Update configuration documentation if you add new environment variables
4. Document new features in appropriate sections

### API Documentation

- Add docstrings to all public functions
- Include parameter descriptions and return types
- Note any exceptions that may be raised

### Comments

- **Do** add comments for complex logic
- **Do** explain "why" not "what"
- **Don't** add comments that restate the code
- Keep comments up to date with code changes

## Submitting Changes

### Pull Request Process

1. Ensure your code passes all tests:
   ```bash
   pytest
   ruff check .
   ```

2. Update documentation if needed

3. Write a clear pull request description:
   - Summary of changes
   - Motivation for the change
   - Testing performed
   - Related issues (if any)

4. Link to any relevant issues in the PR description

### Pull Request Template

```markdown
## Summary
Brief description of changes.

## Motivation
Why this change is needed.

## Testing
How you tested this change.

## Checklist
- [ ] Tests pass
- [ ] Linting passes
- [ ] Documentation updated
- [ ] No breaking changes (or documented)
```

### Review Process

Maintainers will review your PR and may request changes. Be responsive to feedback and willing to iterate on your changes.

## Areas for Contribution

Based on [`TODO.md`](TODO.md), here are key areas where contributions are welcome:

### High Priority (P0)

- Extend test coverage to cloud client, local calibration, and deadline handling
- Add fuzz testing for `pal.safe_eval` with adversarial model output
- Add a license file
- Add secret-scanning pre-commit hook
- Run container as non-root user
- Add dependency pinning/lockfile

### Medium Priority (P1)

#### Routing and Accuracy
- Improve ONNX NER precision above current ~22% exact-match
- Enhance sentiment specialist coverage (negation scopes, quoted praise, mixed reviews)
- Add router confidence score
- Expand paraphrase test set
- Add regression tests for multipart-math detector
- Measure if a small classifier beats lexical router
- Add category-level fallback budgets

#### Data and Fine-Tuning
- Add data validation script for training data
- Deduplicate training datasets
- Add held-out validation split
- Make dataset generation reproducible with seeds

#### Reliability and Observability
- Add structured JSON logging with task id, category, lane, elapsed time, token usage
- Export per-run metrics
- Make time budget counters thread-safe
- Add backoff/jitter for connection errors
- Add per-task cloud time budget
- Handle edge cases in cloud responses
- Add `--dry-run` mode

#### Evaluation
- Add deterministic checks where possible to supplement LLM judge
- Run judge multiple times and report variance
- Add token-cost regression test
- Add end-to-end tests with mocked cloud client

### Lower Priority (P2)

#### Packaging
- Publish package metadata
- Add logo/wordmark
- Add docs page with lane diagram
- Add Makefile or justfile for common tasks
- Add CI with ruff/pytest
- Add CHANGELOG.md and semantic versioning
- Provide model download script with checksums
- Add multi-arch Docker builds
- Add web demo or CLI for lane decision explanation

#### Performance
- Benchmark ONNX sessions with tuned thread counts
- Cache tokenizer/session loads
- Evaluate dynamic quantization or smaller DistilBERT
- Add context-length checks before local generation
- Re-evaluate speculative category batching

#### Documentation
- Add architecture diagrams for lane order and escalation paths
- Document exact judge prompt and failure modes
- Add troubleshooting section
- Document how to add a new category end-to-end
- Add security note about prompt injection

## Getting Help

If you need help contributing:

1. Check existing issues and pull requests
2. Read the documentation thoroughly
3. Ask questions in your pull request
4. Be patient - maintainers are volunteers

## Recognition

Contributors will be acknowledged in the project. Major contributions may be highlighted in release notes.

## License

By contributing, you agree that your contributions will be licensed under the project's license (to be determined).

Thank you for contributing to QueryLane! 
