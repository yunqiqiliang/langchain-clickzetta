# Contributing to LangChain-ClickZetta

Thank you for your interest in contributing to LangChain-ClickZetta! This document provides guidelines for contributing to this project.

## Development Setup

### Prerequisites
- Python 3.9+
- ClickZetta database access
- DashScope API key (for embeddings and LLM)

### Setting up the Development Environment

1. **Clone the repository**
   ```bash
   git clone https://github.com/yunqiqiliang/langchain-clickzetta.git
   cd langchain-clickzetta
   ```

2. **Create a virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Configure ClickZetta connection**
   Create `~/.clickzetta/connections.json`:
   ```json
   {
     "system_config": {
       "embedding": {
         "dashscope": {
           "api_key": "your-dashscope-api-key",
           "model": "text-embedding-v4",
           "dimensions": 1024
         }
       }
     },
     "connections": [
       {
         "name": "uat",
         "service": "your-service",
         "username": "your-username",
         "password": "your-password",
         "instance": "your-instance",
         "workspace": "your-workspace",
         "schema": "your-schema",
         "vcluster": "your-vcluster"
       }
     ]
   }
   ```

## Code Quality

### Linting and Formatting
```bash
# Format code
black langchain_clickzetta tests

# Check code style
ruff check .

# Type checking
mypy langchain_clickzetta
```

### Testing
```bash
# Run unit tests
pytest tests/unit/ -v

# Run integration tests (requires ClickZetta connection)
pytest tests/integration/ -v

# Run all tests
pytest -v

# Run with coverage
pytest --cov=langchain_clickzetta --cov-report=html
```

## Contributing Guidelines

### Code Style
- Follow PEP 8 conventions
- Use type hints for all public functions
- Write descriptive docstrings in Google style
- Keep line length under 88 characters

### Commit Messages
- Use clear, descriptive commit messages
- Start with a verb in imperative mood
- Reference issues when applicable

Example:
```
Add support for hybrid search in single table

- Implement ClickZettaHybridStore class
- Add unified retriever for vector and full-text search
- Update tests for new functionality

Fixes #123
```

### Pull Request Process

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write code following our style guidelines
   - Add or update tests as needed
   - Update documentation if necessary

3. **Test your changes**
   ```bash
   # Run all tests
   pytest

   # Check code quality
   black langchain_clickzetta tests
   ruff check .
   mypy langchain_clickzetta
   ```

4. **Commit and push**
   ```bash
   git add .
   git commit -m "Your descriptive commit message"
   git push origin feature/your-feature-name
   ```

5. **Create a Pull Request**
   - Provide a clear description of changes
   - Reference any related issues
   - Ensure all CI checks pass

### What to Contribute

#### High Priority
- Bug fixes
- Performance improvements
- Documentation improvements
- Test coverage expansion

#### Medium Priority
- New ClickZetta features integration
- Example code and tutorials
- Additional vector distance metrics
- Enhanced error handling

#### Ideas for New Features
- Async support for all operations
- Connection pooling
- Query optimization
- Additional data types support

## Testing Guidelines

### Unit Tests
- Test individual components in isolation
- Use mocks for external dependencies
- Focus on edge cases and error conditions

### Integration Tests
- Test real ClickZetta integration
- Use actual DashScope services when possible
- Clean up test data after each test

### Test Organization
```
tests/
├── unit_tests/           # Unit tests
│   ├── test_engine.py
│   ├── test_vectorstores.py
│   └── ...
├── integration_tests/    # Integration tests
│   ├── test_real_connection.py
│   └── test_hybrid_features.py
└── utils.py             # Test utilities
```

## Debugging

Use the debug scripts in `scripts/debug/` for development:
```bash
# Debug vector store
python scripts/debug/debug_vectorstore.py

# Debug table creation
python scripts/debug/debug_table_creation.py
```

## Documentation

### Code Documentation
- All public classes and methods must have docstrings
- Use Google-style docstrings
- Include examples in docstrings when helpful

### README Updates
- Update README.md for new features
- Add usage examples
- Update feature list if applicable

## Release Process

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Run full test suite
4. Create git tag
5. Build and upload to PyPI

## Getting Help

- **Issues**: Create an issue on GitHub
- **Questions**: Start a discussion in GitHub Discussions
- **Documentation**: Check `docs/` directory and README.md

## Code of Conduct

Please be respectful and constructive in all interactions. We are committed to providing a welcoming and inclusive environment for all contributors.

Thank you for contributing to LangChain-ClickZetta!