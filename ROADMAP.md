# Roadmap

Known issues and planned improvements, kept on the development branch. Each
entry says what was observed and where to start.

## Known issues

**The hybrid critic learns slowly with a variational layer.** On correlated
Gaussians at 2.00 bits, `critic_type='hybrid'` with `use_variational=True`
reaches 0.66 bits after 40 epochs and 1.56 after 150, still rising, while the
separable critic with the same layer reaches 1.94. The cause has not been
traced. Start from `neural_mi/models/critics.py` and the variational wrapper in
`neural_mi/models/embeddings.py`.

## Improvements

**Split the largest modules.** `run.py` (about 1,600 lines), `training/trainer.py`,
`data/temporal.py`, `models/embeddings.py` and `analysis/rigorous.py` (each about
1,200 to 1,350 lines) mix several concerns. Splitting them along those concerns
would make each easier to test and review.

**Collapse the two layers under `run()`.** `run()` lowers the config objects into
keyword arguments for `_run_flat()`, which validates them and builds the
parameter dictionary. One layer that works on the config objects directly would
remove the translation between them.

**Merge test files with overlapping scope.** `test_visualize.py` and
`test_visualize_extended.py`, and `test_run.py` and `test_run_config_api.py`,
cover neighbouring ground. Merging each pair would make it clearer where a new
test belongs.

**Check the notebooks and the README in CI.** `docs/figures/readme_figures.ipynb`
runs the README quickstart and the code in `reference/ANATOMY.md`, but only when
someone runs it. A CI job that executes its two checking sections, and a slow job
that executes the tutorials, would catch drift at the pull request.

**Adopt a lockfile for development.** `pyproject.toml` declares every dependency
as a range. A lockfile (for example from `uv lock`) would record the exact
versions the tutorial outputs were produced with and give every contributor the
same environment. It needs one tool across development and CI, and a relock
whenever a dependency changes.

**Give the internal docstrings a language pass.** The docstrings rendered in the
API reference have had one. The rest are developer notes and have not.

**Archive releases on Zenodo.** Connecting the repository to Zenodo gives every
release a DOI. When the JOSS paper is accepted, `CITATION.cff` should point to it.
