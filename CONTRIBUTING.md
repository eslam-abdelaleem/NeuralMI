### **File: `CONTRIBUTING.md`**
This file provides guidelines for developers.

```markdown
# Contributing to NeuralMI

Thank you for your interest in contributing! We welcome bug reports, feature requests, and pull requests.

## Development Setup

1.  **Fork and Clone:** Fork the repository on GitHub and then clone it to your local machine.

2.  **Create a Virtual Environment:** It's highly recommended to use a virtual environment to manage dependencies.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install in Editable Mode:** Install the library in "editable" mode, which means your changes to the source code will be reflected immediately when you run Python.
    ```bash
    pip install -e .
    ```

4.  **Install Development Dependencies:** Install the packages required for running tests and formatting code.
    ```bash
    pip install -r requirements-dev.txt
    ```

## Running Tests

We use `pytest` for our test suite.

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run a specific test file
pytest tests/test_data_processors.py
```

Please ensure all tests pass before submitting a pull request. If you are adding a new feature, please include corresponding tests.

## Code Style
We follow the **PEP 8** style guide. Before committing, please format your code using ```black``` and ```isort```.

## Pull Request Process
1. Create a new branch for your feature or bug fix: ```git checkout -b feature/my-new-feature```
2. Make your changes and commit them with a clear, descriptive message.
3. Push your branch to your fork: ```git push origin feature/my-new-feature```
4. Open a pull request from your fork to the main ```neuralmi``` repository.
5. In the pull request description, please explain the changes you made and link to any relevant issues.

Thank you for contributing!


