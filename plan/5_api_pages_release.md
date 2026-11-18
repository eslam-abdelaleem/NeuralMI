# Phases A and P: the public API, the pages and the release

Phase A changes what a user types. It comes after phase E, because each change is then a small
edit of `run()` instead of a change in several engines. Phase P covers the pages that the
earlier phases make stale, and everything outside the package that the first release needs.

## Phase A: the public API

At the end of phase E, `run()` already takes a quantity, every analysis already accepts the
modifiers, and the old spellings survive as names that set them. This phase removes the old
spellings, so that each choice of a call has one argument. Each step changes one part of the
API together with every test, tutorial cell and reference page that uses the old form. The
counts come from a search of the repository at commit `39f42d1`. Each step also updates
`tests/public_surface.txt` (S-04). The tutorials are edited here and run again once, in P-13.

Steps A-02, A-03 and A-10 follow from points that the owner has not confirmed yet (see
"Decisions" in `plan/README.md`).

### A-01 `rigorous` is an argument that every analysis takes

**Why.** The correction is requested in four ways today: `mode='rigorous'` with a `Rigorous`
config, a field `rigorous=True` on `Transfer`, `Conditional` and `Interaction` with seven fit
fields copied into each (config.py:369-376, 434-441, 467-474), an argument `rigorous=` of every
named function, and a table that translates between them (quantities.py:62-94). After E-08 the
engine has one modifier.

**Change.**
1. `run(..., rigorous=None)` takes `True`, a `Rigorous` or a dict, with every analysis that the
   table of E-08 allows.
2. Remove `mode='rigorous'` from `MODES`.
3. Remove the field `rigorous` and the seven fit fields from `Transfer`, `Conditional` and
   `Interaction`.

**Tests.** `mode='rigorous'` appears 17 times in 11 test files. Rewrite those calls.

**Pages.** One tutorial cell, 15 lines in five reference pages, and one line each in `README.md`
and `paper.md`.

**Done when.** `grep -rn "mode='rigorous'" --include=*.py --include=*.md --include=*.ipynb .`
shows nothing outside `plan/` and `CHANGELOG.md`.

### A-02 `quantity` and `w_data` are arguments of `run()`, and three modes and three configs go

**Why.** `mode='conditional'`, `mode='interaction'` and `mode='transfer'` name what is measured,
while the other modes name what is done with the estimator. The third signal is passed inside a
config (`Conditional(w_data=w)`), although it is data like X and Y. After A-01 that config has
no other field.

**Change.**
1. Change the signature to `run(x_data, y_data=None, w_data=None, *, quantity=None, ...)`.
2. A call with `w_data` and no `quantity` measures `I(X; Y | W)`. `quantity='interaction'`
   measures the three-term form. A `quantity` that needs a third stream raises when `w_data` is
   missing, and a `w_data` that the quantity does not read warns (run.py:1196-1201).
3. Remove `'conditional'`, `'interaction'` and `'transfer'` from `MODES`. The modes that remain
   are `estimate`, `lag`, `pairwise`, `dimensionality` and `precision`.
4. Delete the classes `Conditional`, `Interaction` and `Transfer` and the arguments
   `conditional=`, `interaction=` and `transfer=`.
5. Export `Quantity` from the package.

**Tests.** `Conditional(` appears 75 times in 20 test files, `Interaction(` 42 times in 10 and
`Transfer(` 31 times in 8. The three modes appear 162 times. Rewrite those calls.

**Pages.** Three tutorial cells and about 17 lines of the reference pages.

**Done when.** `nmi.run(x, y, w)` returns a conditional MI, `nmi.Conditional` does not exist,
and `MODES` has five keys.

### A-03 The named functions are wrappers of `run()`

**Why.** After E-09 each named function builds a `Quantity` and calls `run()`. This step fixes
their public form.

**Change.**
1. Each wrapper takes its data, its windows, `stride` and, for transfer entropy,
   `bidirectional`. It forwards every other argument to `run()`, so `mode`, `sweep_grid`,
   `rigorous` and `permutation` work with every named function.
2. Keep all twelve functions. `interaction_information(x, y, w)` is
   `run(x, y, w, quantity='interaction')`, and `offset_quantity(data, spec)` is
   `run(data, quantity=Quantity('offsets', ...))`.
3. State in each docstring the call of `run()` that the wrapper makes.

**Tests.** Add a test that a named function with `mode='lag'` runs, since the wrapper forwards
the analysis.

**Pages.** `reference/USING.md` presents the named functions as shorthands of `run()`.

**Done when.** Every function of `quantities.py` has a body of at most about ten lines.

### A-04 `permutation` is one argument

**Why.** The permutation test is set by five flat arguments of `run()` (run.py:664-668), while
every other group of settings is a config. Its settings are validated in two places (run.py:1047
and 1182-1190), and B-03 added the missing check of `n_permutations`.

**Change.**
1. Add the config `Permutation` with the fields `n_permutations`, `shuffle`, `min_shift` and
   `min_permutations`, and one schema `PERMUTATION_SETTINGS` in `defaults.py` with their types,
   bounds and defaults.
2. `run(..., permutation=None)` and every named function take `True`, a `Permutation` or a
   dict. Remove the five flat arguments.
3. Validate the settings against the schema in one place.

**Tests.** `permutation_test=` appears 31 times in 6 test files. Rewrite those calls.

**Pages.** Six lines of the reference pages, and a new table in `reference/PARAMETERS.md`.

**Done when.** `inspect.signature(nmi.run)` has no parameter that starts with `permutation_`.

### A-05 `history_window` and `future_window` name every window in time

**Why.** The length of the past is called `k` in five functions, `history_window` in two and
`h` in one. The length of the future is `future_k` in one function and `prediction_horizon` in
`te_spec`, where it is the number of future steps (analysis/transfer.py:40).
`transfer_entropy()` cannot set it at all (quantities.py:723-726), and `predictive_information`
forces the future to the length of the past.

**Change.** Rename the arguments as in the table. Every window counts whole steps.

| Function | Today | At the end |
|---|---|---|
| `active_information_storage` | `k`, `future_k=1` | `history_window`, `future_window=1` |
| `predictive_information` | `k` | `history_window`, `future_window=None` |
| `cross_predictive_information` | `k` | `history_window`, `future_window=None` |
| `transfer_entropy`, `conditional_transfer_entropy` | `history_window` | `history_window`, `future_window=1` |
| `instantaneous_exchange` | `k` | `history_window` |
| `directed_information_rate` | `k` | `history_window` |
| `mi_rate` | `h`, `half_width` | `history_window`, `half_width` |
| `block_mi` | `window_size` | unchanged, because a processor cuts this window |

1. A `future_window` of `None` takes the value of `history_window`.
2. Rename the argument `prediction_horizon` of `te_spec` to `future_window`, and the arguments
   of the `_spec_*` functions (quantities.py:400-440).
3. A window may be a list, and the list is a key of the grid (E-10). Several lists give every
   combination of their values. `mi_rate` refuses two lists today (quantities.py:854-860), and
   E-08 lifts that refusal. The name of the argument is the column of the result.

**Tests.** The five functions that take `k` are called 63 times in 9 test files, and `mi_rate`
20 times in 5. `prediction_horizon` appears 9 times in 2 test files and `future_k` twice.

**Pages.** Eight tutorial cells and about 18 lines of the reference pages. `reference/THEORY.md`
can keep `k` as a symbol in its formulas if it says which argument sets it.

**Done when.** `grep -rn "future_k\|prediction_horizon" neural_mi/` shows nothing, and no named
function has a parameter called `k` or `h`.

### A-06 `sweep_grid` works with every analysis, and `mode='sweep'` goes

**Why.** `estimate` refuses a `sweep_grid` and `sweep` requires one (analysis/modes.py:775-781).
Every other mode takes a grid without changing its name, and the named functions already choose
between the two by whether a grid is present (quantities.py:118 and 664). As a result
`active_information_storage(x, k=5)` reports the mode `'estimate'` and `k=[5]` reports `'sweep'`
for the same number. After E-08 and E-10 a grid is a modifier of every analysis.

**Change.**
1. `MODES['estimate']` accepts a grid, and `'sweep'` is removed.
2. A result with one configuration and one repeat is shown as a training curve. A result with
   more is shown as the table of a sweep.

**Tests.** `mode='sweep'` appears 44 times in 19 test files.

**Pages.** Eight cells in three notebooks and seven lines of the reference pages. State in
`reference/USING.md` the three kinds of key that a grid accepts: a setting of the network or of
the training, a setting that changes the rows, and a parameter of the quantity.

**Done when.** `nmi.run(x, y, sweep_grid={'run_id': range(3)})` runs with the default mode.

### A-07 A result names its quantity, lists its keys and shows what separates its best configurations

**Why.** `Results.get(key)` searches four tables (results.py:515-569), and nothing lists the
keys that exist. The representation of a result shows its mode and not its quantity.
`Results.best()` already ranks the configurations of a sweep by their held-out MI and returns
the ones that the held-out rows cannot tell apart from the best (results.py:570-603). The
held-out value is the one that the bound guarantees, so this ranking has a basis in theory. No
theory chooses among the configurations that the ranking cannot tell apart, and the table gives
the user nothing to choose by.

**Change.**
1. Add `Results.keys()`. It returns the keys that `get` can read, grouped by the table that
   contains them.
2. Show the quantity's name and symbol from `view` (E-11) in `__repr__` and in `summary()`.
3. Add to the table of `best()` the gap between the training value and the held-out value of
   each configuration, and its number of parameters where the runs record it. Keep the order by
   held-out MI, and apply no rule to the configurations that cannot be told apart. Say in the
   docstring that the smallest gap and the smallest network are two reasonable ways to choose
   among them, and that neither has a basis in theory.
4. Let `best()` work on any result with more than one configuration, whatever its analysis.
5. Say in the docstring of `to_json` that it exports a result, and that `save` and `load` are
   the pair that restores one.

**Done when.** `set(result.keys())` equals the set of keys for which `result.get` succeeds, in
a test over every case of the snapshot, and the table of `best()` has a column for the gap.

### A-08 The public names of the package

**Why.** `__all__` exports `utils`, `validation` and `results` as public modules
(neural_mi/__init__.py:45-74), and no page documents a use of them. `Results.plot` calls
`plot_sweep_heatmap` and `plot_sweep_bar` (results.py:764), but `visualize/__init__.py` does not
export them. The pages reach `create_dataset` through `neural_mi.data.handler`.

**Change.**
1. Remove `validation` and `results` from `__all__`. L-05 already removed `utils`.
2. Export `create_dataset` and `build_offset_arrays` from `neural_mi.data`.
3. Export the two sweep plots from `neural_mi.visualize`.
4. List every public name in `docs/source/api_reference.rst`. `Preprocessing` is missing today.

**Done when.** Every name in `tests/public_surface.txt` appears in `api_reference.rst`.

### A-09 The generators return one form (optional)

**Why.** The ten generators return three shapes of result. Two return `(x, y)` without the
exact MI, six return `(x, y, mi)`, and `generate_shared_latent_gaussian` returns the data with
an oracle. Four default to torch tensors and four to numpy arrays. The size argument is called
`n_samples`, `n_windows` or `T`.

**Change.**
1. Return `(x, y, mi)` from every generator of a pair, with the exact value in bits. Keep the
   oracle form of `generate_shared_latent_gaussian`.
2. Return numpy arrays by default.
3. Delete the unreachable fallback at `generators/oracle.py:1002-1006` and the assignment that
   does nothing at line 1081, and correct the annotation at line 514.

**Pages.** Every tutorial builds its data with a generator, so each needs an edit.

**Done when.** A test calls every generator and checks the form of what it returns.

### A-10 A Gaussian estimator gives the cheap linear answer

**Why.** Every estimate that the library makes today trains a network or searches for nearest
neighbours. A user has no cheap answer to compare it with, and `plot_cross_correlation` (deleted
in X-08) was the only trace of one. The cheap counterpart of `I(A; B | C)` on the same rows is
the information between Gaussian variables that have the covariance of those rows. It has a
closed form in the determinants of three or four covariance matrices and takes milliseconds.
For one channel on each side it is `-0.5 * log(1 - r**2)`, where `r` is the correlation
coefficient. The estimators that train nothing already have a place in the registry (R-03), so
this estimator needs no path of its own.

**Change.**
1. Add `estimators/gaussian.py` with `gaussian_mi(x, y)` and `gaussian_cmi(x, y, z)`, computed
   from the log-determinants of the covariance of the flattened rows.
2. Register it as `estimator='gaussian'`, with `trains_network=False` and `gaussian_cmi` as its
   conditional form. It then runs wherever `ksg` runs: with every quantity, with `mode='lag'`
   (the cross-correlation function, in bits), with `mode='pairwise'` (the correlation map, in
   bits), with a grid and with the rigorous correction.
3. Refuse a call whose rows are not more numerous than the dimensions of a row, because the
   covariance is then singular. Warn when the rows are fewer than a multiple of the dimensions
   that is a setting, because the value is then biased upward.
4. Warn when a stream is given as spike times, because the covariance of padded spike times
   has no meaning.
5. Establish first whether a grid over `estimator_name` runs today with a trained and an
   untrained estimator side by side. If it does, a grid over the two names gives the cheap
   answer next to the trained one in one table.
6. Consider an option that first maps each channel through its ranks to a normal variable, as
   `ksg` does with `gaussianize` (estimators/ksg.py:175).

**Tests.** Compare the estimator with the exact value of `LatentGaussian` on Gaussian data, for
plain MI, for a conditional quantity and for transfer entropy. Add one snapshot case.

**Pages.** Add the estimator to `reference/PARAMETERS.md`. Add a short section to
`reference/USING.md` that advises running it first.

**Done when.** `nmi.run(x, y, estimator='gaussian')` returns the exact value on Gaussian data
within sampling error, and the same call with `mode='lag'` returns a curve.

## Phase P: pages, packaging and release

Each step says under **When** whether it can be done at any time or must wait for the code.

### P-01 Write the changelog of the first release

**When.** At any time. Set the date at the tag.

**Why.** `CHANGELOG.md` has about 4,890 lines for a library that was never published. It contains
a section `[1.0.0]`, a development record under `[Unreleased]` of about 3,650 lines, and below
them the sections `[2.1.0]` and `[2.0.0]` from March 2026. Those two sections name files that
no longer exist. The `[1.0.0]` section describes changes to things that no outside user saw.

**Change.** Replace the file by one entry for the first release. Describe what the library does,
from `README.md` and `reference/USING.md`: the modes, the named quantities, the estimators, the
processors, the rigorous correction, the permutation test, the transforms and the generators
with exact answers. Add the supported versions of Python and torch and the known limits. Git
keeps the development record.

**Done when.** The file has one release section and an empty `[Unreleased]` above it.

### P-02 Keep the version in one place

**When.** At any time.

**Why.** The version is written by hand in `pyproject.toml`, `neural_mi/__init__.py:2` and
`CITATION.cff`. The changelog and `paper.md` give 30 September 2026 as the date, and the commits
continue into November.

**Change.**
1. Declare the version as dynamic in `pyproject.toml` and read it from
   `neural_mi.__version__`.
2. Add a test that the version in `CITATION.cff` equals it.
3. Set the release date in the changelog, the paper and `CITATION.cff` when the tag is made.

**Done when.** The version string appears once in the repository outside `CITATION.cff`.

### P-03 Set the classifiers and test Python 3.13

**When.** At any time.

**Why.** `pyproject.toml` declares "Development Status :: 5 - Production/Stable" for a first
release. CI tests Python 3.11 and 3.12. This plan was prepared under Python 3.13 with torch
2.15, where the package imported and its small runs worked.

**Change.** Choose the status with the owner. Add Python 3.13 to the classifiers and to the
matrix of `.github/workflows/tests.yml`.

**Done when.** CI passes on the three versions.

### P-04 Check what the wheel and the source distribution contain

**When.** After phase A.

**Why.** The audit built the wheel and found only the package, `py.typed` and the licence in it.
The source distribution was never built, and the repository has no `MANIFEST.in`.

**Change.**
1. Add `MANIFEST.in`. Include the changelog, the citation file, the licence and the README, and
   leave out `plan/`, the tutorial data and the built documentation.
2. Add a CI job that runs `python -m build` and `twine check dist/*`, lists the archive, installs
   the wheel in a clean environment and imports it.

**Done when.** The job passes and the listing contains no file that should not ship.

### P-05 Publish from a tag

**When.** After P-04.

**Why.** The repository has no publish workflow and no tag.

**Change.** Add `.github/workflows/release.yml`. On a tag `v*` it builds, checks and publishes
to PyPI through trusted publishing, after a first job that publishes to TestPyPI. The owner
registers the publisher on both sites, since only the account holder can.

**Done when.** A release candidate tag reaches TestPyPI and installs from it.

### P-06 Make the README work on PyPI

**When.** After P-05 has reserved the name.

**Why.** The README uses relative paths for its images, and PyPI cannot resolve them. Its
install line is `pip install .`.

**Change.** Use absolute URLs for the images and `pip install neural_mi` for the install line.

**Done when.** `twine check` reports no problem in the long description, and the page on
TestPyPI shows the images.

### P-07 Move the roadmap out of the release

**When.** Before phase L.

**Why.** `ROADMAP.md` lives on `dev` and `tests/test_docs_drift.py:287` checks every name in
backticks in it against the code. Each rename of this plan would then fail that test until the
roadmap is edited too.

**Change.** Turn the open items of the roadmap into issues and delete the file, or keep it and
remove it from the check. Remove the branch of the test in either case.

**Done when.** The docs test does not read `ROADMAP.md`.

### P-08 Store the tutorial data as arrays, with their source

**When.** At any time.

**Why.** Tutorial 04 loads two pickles of 7.9 MB with `pickle.load`. A pickle can run code when
it is loaded. These two also refer to `numpy.core`, a path that numpy 2 deprecates. The data are
a subset of the CRCNS data set hc-11 (https://crcns.org/data-sets/hc/hc-11), recorded by
Grosmark and Buzsáki (2016). Nothing in the repository says so beside the citation in the
notebook.

**Change.**
1. Convert both files to `.npz` and load them with `allow_pickle=False`.
2. Add `tutorials/data/README.md` with the source, the citation that CRCNS asks for, and how
   the subset was made (the sessions, the time span and the processing).
3. Read the terms of use on the page of hc-11 before the release. If they do not allow a subset
   to be redistributed in a public repository, replace the files by a script that downloads
   and prepares them.
4. Update the loading cell of tutorial 04 and the two lines of `.gitignore`.

**Done when.** `git ls-files tutorials/data` shows no `.pkl` file, and the README of the folder
names the data set.

### P-09 Build the documentation site without warnings

**When.** After phase A.

**Why.** `docs/source/getting_started.md` includes the whole of `USING.md`. `USING.md` is also a page
of its own, so the site shows that text twice. The build was not run during the audit.

**Change.** Remove the second inclusion, and build with `sphinx-build -W` in
`.github/workflows/docs.yml`.

**Done when.** The build passes with warnings as errors.

### P-10 Limit the CI runs

**When.** At any time.

**Why.** `tests.yml` runs the full suite with the slow tests on every push and sets no time
limit. `draft-pdf.yml` builds the paper on every push from an action pinned to `master`.

**Change.** Set `timeout-minutes` on each job. Build the paper only when `paper.md` or
`paper.bib` changes or on request, from a pinned version of the action.

**Done when.** A push that does not touch the paper does not start the paper job.

### P-11 Cite the software

**When.** At the release.

**Why.** `CITATION.cff` points to the papers and has no entry for the software, and the release
has no DOI. The repository has no code of conduct and no security policy.

**Change.** Archive the release on Zenodo and put its DOI in `CITATION.cff` and `paper.md`. Add
`CODE_OF_CONDUCT.md` and `SECURITY.md`.

**Done when.** GitHub's "Cite this repository" gives the software with its DOI.

### P-12 Rewrite INTERNALS.md and TESTING.md for the new layout

**When.** After phases E, W and T.

**Why.** `reference/INTERNALS.md` describes the modules as they are today, with `_run_flat`,
`utils` and the two ladders. `reference/TESTING.md` must list every test file, and this plan
adds several.

**Change.**
1. Rewrite `INTERNALS.md` from the section "What the code looks like at the end" of
   `plan/README.md`: the path of a call, the layers, the tree and the home of each concept.
2. Rewrite its sections on adding a mode, an encoder, an estimator and a processor for the
   registries of phase R.
3. Describe the snapshot, the layer test and the surface test in `TESTING.md`.

**Done when.** `tests/test_docs_drift.py` passes and `INTERNALS.md` names no deleted module.

### P-13 Rewrite USING.md around the three choices of a call, and run the tutorials again

**When.** After phase A.

**Why.** `reference/USING.md` presents the modes and the named functions as two lists that
overlap. After phase A a call makes three choices: the quantity, the analysis and the modifiers. The five tutorials and the
notebook of the README figures store their outputs, and the API and some numbers have changed.

**Change.**
1. Organise `USING.md` as the quantity (chosen signals, then chosen times with the named
   functions), the analysis and the modifiers.
2. Check its table of what each mode accepts against `MODES`.
3. Run the five tutorials and `docs/figures/readme_figures.ipynb` again. Compare every number
   that a reference page quotes with the new outputs.
4. Update the examples of `README.md` and `paper.md`.

**Done when.** Every notebook runs from a clean kernel, and `tests/test_docs_drift.py` passes.

### P-14 Remove this folder

**When.** Before the release branch is cut.

**Why.** The plan describes work that is finished by then.

**Change.** Delete `plan/` from the branch that is released.

**Done when.** `git ls-files plan` shows nothing on that branch.
