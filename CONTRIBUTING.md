# Contributing to NeuralMI

Bug reports, feature requests and pull requests are all welcome.

## Reporting a problem

Open an issue on the [issue tracker](https://github.com/eslam-abdelaleem/NeuralMI/issues). The bug form asks for the
mode, the call, the data shapes, and `result.summary()` with `result.runs`.

## Development setup

Fork and clone the repository, then work on a branch in a virtual environment:

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -e ".[dev]"
```

The dev extra installs every optional part: the test tools, the plotting,
vision and tutorial packages, and the docs toolchain. Building the docs also
needs pandoc from your system's package manager. Source changes take effect
without reinstalling.

## Before a pull request

Run the whole suite with `pytest`. [`TESTING.md`](https://github.com/eslam-abdelaleem/NeuralMI/blob/main/reference/TESTING.md)
lists the faster and serial variants, maps every test file to what it covers,
and says where to look when something breaks.

A change to a setting, a default or a warning also changes the reference pages.
[`PARAMETERS.md`](https://github.com/eslam-abdelaleem/NeuralMI/blob/main/reference/PARAMETERS.md)
lists every setting and
[`MESSAGES.md`](https://github.com/eslam-abdelaleem/NeuralMI/blob/main/reference/MESSAGES.md)
every warning, and `tests/test_docs_drift.py` fails until both match the code.
[`INTERNALS.md`](https://github.com/eslam-abdelaleem/NeuralMI/blob/main/reference/INTERNALS.md)
has the codebase map and the steps for adding an estimator, a processor, an
encoder or an analysis mode.

The docs site keeps no text of its own for the reference pages or the README's
install steps and quickstart. Stubs in `docs/source/` include them, so edit the
originals. The figures in the README and on the site come from
`docs/figures/readme_figures.ipynb`, which also runs the README quickstart and
the code in `reference/ANATOMY.md` and fails when either has drifted. Rerun it
after a change that could move a figure or a printed number.

## Code style

Follow PEP 8, use descriptive names, comment the parts that are genuinely
subtle, and keep functions small enough to test one at a time. Public functions
carry type hints and NumPy-format docstrings. A warning passes
`stacklevel=user_stacklevel()` from `neural_mi.logger`, so that it names the
line in the caller's code that led to it, and `tests/test_run.py` fails on a
hardcoded level.

## Pull request process

1. Branch: `git checkout -b feature/my-new-feature`
2. Make the change, with tests, and commit it with a clear message.
3. Check that `pytest` passes and that `pytest --cov=neural_mi` shows no lower total.
4. Push to your fork: `git push origin feature/my-new-feature`
5. Open a pull request against `dev`, describing what changed and linking any
   related issue.
