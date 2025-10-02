# Contributing to NeuralMI

Thank you for your interest in contributing!

## Development Setup

```bash
git clone https://github.com/yourusername/neural_mi.git
cd neural_mi
pip install -e ".[dev]"
```

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=neural_mi --cov-report=html

# Run specific test file
pytest tests/test_estimators.py

# Run with verbose output
pytest -v
```

## Code Style

We follow PEP 8 with a few modifications:
- Line length: 100 characters
- Use type hints for all function signatures
- Use NumPy-style docstrings

## Adding New Features

1.  **Write tests first** (test-driven development)
2.  **Update documentation** (docstrings and README if needed)
3.  **Add example usage** in docstring
4.  **Run full test suite** before submitting PR

## Pull Request Process

1.  Create a feature branch: `git checkout -b feature/my-feature`
2.  Make changes and commit: `git commit -am "Add feature"`
3.  Push to your fork: `git push origin feature/my-feature`
4.  Open a pull request with clear description of changes
5.  Ensure all tests pass and coverage doesn't decrease

## Reporting Bugs

Open an issue with:
- Clear description of the problem
- Minimal code to reproduce
- Expected vs actual behavior
- System info (OS, Python version, PyTorch version)