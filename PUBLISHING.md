# Publishing Guide for langchain-clickzetta

This guide explains how to publish the `langchain-clickzetta` package to PyPI.

## Prerequisites

1. **PyPI Account**: Create an account at [PyPI](https://pypi.org/account/register/)
2. **Test PyPI Account**: Create an account at [TestPyPI](https://test.pypi.org/account/register/) for testing
3. **API Tokens**: Generate API tokens for both PyPI and TestPyPI

## Pre-publication Checklist

Before publishing, ensure:

- [ ] All tests pass (`make test`)
- [ ] Code quality checks pass (`make lint`)
- [ ] Documentation is up to date
- [ ] Version number is updated in `pyproject.toml`
- [ ] CHANGELOG.md is updated with new version
- [ ] README.md installation instructions are correct

## Current Status

✅ **Package Configuration**: Complete
- Package name: `langchain-clickzetta`
- Current version: `0.1.0`
- Build system: `hatchling`
- All dependencies properly specified

✅ **Build Tools**: Installed
- `build` package for creating distributions
- `twine` package for uploading to PyPI

✅ **Package Build**: Successfully built
- `dist/langchain_clickzetta-0.1.0-py3-none-any.whl` (wheel)
- `dist/langchain_clickzetta-0.1.0.tar.gz` (source distribution)

## Publishing Steps

### Step 1: Build the Package

```bash
cd libs/clickzetta
source .venv/bin/activate
python -m build
```

This creates:
- `dist/langchain_clickzetta-0.1.0-py3-none-any.whl`
- `dist/langchain_clickzetta-0.1.0.tar.gz`

### Step 2: Test on TestPyPI (Recommended)

First, test the upload to TestPyPI:

```bash
# Upload to TestPyPI
twine upload --repository testpypi dist/*

# Test installation from TestPyPI
pip install --index-url https://test.pypi.org/simple/ langchain-clickzetta
```

### Step 3: Upload to PyPI

Once testing is successful:

```bash
# Upload to PyPI
twine upload dist/*
```

You'll be prompted for your PyPI credentials or API token.

### Step 4: Verify Installation

After publishing:

```bash
pip install langchain-clickzetta
```

## Authentication Options

### Option 1: Interactive Authentication
`twine` will prompt for username and password.

### Option 2: API Tokens (Recommended)
Create API tokens in your PyPI account settings and use:

```bash
twine upload --username __token__ --password <your-token> dist/*
```

### Option 3: Configuration File
Create `~/.pypirc`:

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = <your-pypi-token>

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = <your-testpypi-token>
```

## Version Management

### For New Releases

1. Update version in `pyproject.toml`:
   ```toml
   version = "0.1.1"  # or next version
   ```

2. Clean previous builds:
   ```bash
   rm -rf dist/ build/
   ```

3. Rebuild and publish:
   ```bash
   python -m build
   twine upload dist/*
   ```

## Post-Publication

1. **Update README**: Change installation instructions to:
   ```bash
   pip install langchain-clickzetta
   ```

2. **Create Git Tag**:
   ```bash
   git tag v0.1.0
   git push origin v0.1.0
   ```

3. **GitHub Release**: Create a release on GitHub with the same tag

## Common Issues and Solutions

### Issue: Package already exists
**Solution**: Update the version number in `pyproject.toml`

### Issue: Authentication failed
**Solution**: Check your API token or credentials

### Issue: Invalid package structure
**Solution**: Ensure `pyproject.toml` has correct `[tool.hatch.build.targets.wheel]` settings

### Issue: Missing dependencies
**Solution**: All dependencies should be listed in `pyproject.toml`

## Security Notes

- Never commit API tokens to version control
- Use API tokens instead of passwords when possible
- Consider using GitHub Actions for automated publishing

## Next Steps

Once published, users can install with:

```bash
pip install langchain-clickzetta
```

And the README.md installation section can be updated to reflect this.