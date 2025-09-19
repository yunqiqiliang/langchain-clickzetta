# Makefile for LangChain ClickZetta

.PHONY: install test lint format clean build

# Install dependencies
install:
	pip install -e ".[dev]"

# Run tests
test:
	pytest tests/ -v

# Run unit tests only
test-unit:
	pytest tests/unit_tests/ -v

# Run integration tests only
test-integration:
	pytest tests/integration_tests/ -v

# Run linting
lint:
	ruff check .
	mypy langchain_clickzetta

# Format code
format:
	black langchain_clickzetta tests
	ruff check . --fix

# Type check
typecheck:
	mypy langchain_clickzetta

# Run all checks
check: format lint typecheck test

# Clean build artifacts
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

# Build package
build: clean
	python -m build

# Install in development mode
dev-install:
	pip install -e ".[dev]"

# Run debug scripts
debug-vectorstore:
	python scripts/debug/debug_vectorstore.py

debug-table:
	python scripts/debug/debug_table_creation.py

# Run integration tests
integration:
	python scripts/integration/test_real_integration.py

integration-dashscope:
	python scripts/integration/test_real_dashscope_integration.py

# Help
help:
	@echo "Available targets:"
	@echo "  install          - Install dependencies"
	@echo "  test             - Run all tests"
	@echo "  test-unit        - Run unit tests"
	@echo "  test-integration - Run integration tests"
	@echo "  lint             - Run linting"
	@echo "  format           - Format code"
	@echo "  typecheck        - Run type checking"
	@echo "  check            - Run all checks"
	@echo "  clean            - Clean build artifacts"
	@echo "  build            - Build package"
	@echo "  dev-install      - Install in development mode"
	@echo "  debug-*          - Run debug scripts"
	@echo "  integration*     - Run integration scripts"