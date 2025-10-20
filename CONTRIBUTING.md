# Contributing to NeuralMI

Thank you for your interest in contributing! We welcome bug reports, feature requests, and pull requests. By following these guidelines, you can help us maintain the quality and integrity of the library.

## Development Setup

1.  **Fork and Clone:** Fork the repository on GitHub and then clone it to your local machine.

2.  **Create a Virtual Environment:** It's highly recommended to use a virtual environment to manage dependencies.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install in Editable Mode:** Install the library in "editable" mode, which means your changes to the source code will be reflected immediately.
    ```bash
    pip install -e .
    ```

4.  **Install Development Dependencies:** Install the packages required for running tests.
    ```bash
    pip install -r requirements-dev.txt
    ```

## Running Tests & Ensuring Quality

`NeuralMI` is a scientific library, and maintaining its correctness is our highest priority. We have a comprehensive test suite to ensure every part of the library works as expected.

### Running the Full Test Suite

Before submitting any changes, you must run the full test suite to ensure your changes have not introduced any regressions.

```bash
# Run all tests
pytest

```
### Checking Test Coverage
All new contributions must be accompanied by tests. We aim for a high level of test coverage to ensure the library is robust. After running the tests, you should generate a coverage report to identify any parts of your new code that are not tested.

```bash
# Run tests and generate a coverage report in the terminal
pytest --cov=neural_mi
```

Look for the `TOTAL` coverage percentage at the bottom of the report. Any new files you add should have near 100% coverage, and any changes you make should not decrease the overall coverage of the library.


## Pull Request Process
1. Create a new branch for your feature or bug fix: ```git checkout -b feature/my-new-feature```
2. Make your changes and commit them with a clear, descriptive message.
3. Ensure all tests pass `pytest`.
4. Ensure your changes are well-tested by checking the coverage report (`pytest --cov=neural_mi`).
5. Push your branch to your fork: ```git push origin feature/my-new-feature```
6. Open a pull request from your fork to the main ```neuralmi``` repository.
7. In the pull request description, please explain the changes you made and link to any relevant issues.

Thank you for contributing!


