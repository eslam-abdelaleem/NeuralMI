# Phases A and P: the public API, the pages and the release

Phase A changes what a user types. It comes after phase E, because each change is then a small
edit of one front door instead of a change in several engines. Phase P covers the pages that the
earlier phases make stale, and everything outside the package that the first release needs.

## Phase A: the public API

Each step changes one part of the API, together with every test, tutorial cell and reference
page that uses the old form. The counts come from a search of the repository at commit
`39f42d1`. Each step also updates `tests/public_surface.txt` (S-04). The tutorials are edited
here and run again once, in P-13.

Steps A-03 and A-06 and the name in A-05 depend on answers that the owner has not given yet
(see "Decisions" in `plan/README.md`).

### A-01 `rigorous` is an argument of `run()` and of every named function

**Why.** The correction is requested in four ways today: `mode='rigorous'` with a `Rigorous`
config, a field `rigorous=True` on `Transfer`, `Conditional` and `Interaction` with seven fit
fields copied into each (config.py:369-376, 434-441, 467-474), an argument `rigorous=` of every
named function, and a table that translates between them (quantities.py:62-94). After E-08 the
engine has one modifier.

**Change.**
1. `run(..., rigorous=None)` takes `True`, a `Rigorous` or a dict, with any mode that
   `Mode.rigorous_refused` allows. A refused mode raises with the reason.
2. Remove `mode='rigorous'` from `MODES`.
3. Remove the field `rigorous` and the seven fit fields from `Transfer`, `Conditional` and
   `Interaction`.
4. Accept `Rigorous(temporal_chunking=...)` for every quantity. The named functions refuse it
   for a difference quantity today (quantities.py:87-92), and `_Chunker` already refuses the one
   case that cannot be honoured (rigorous.py:628-633).

**Tests.** `mode='rigorous'` appears 17 times in 11 test files. Rewrite those calls.

**Pages.** One tutorial cell, 15 lines in five reference pages, and one line each in `README.md`
and `paper.md`. The table of modes in `reference/USING.md` loses its row for `rigorous` and
gains a column that says which modes accept the argument.

**Done when.** `grep -rn "mode='rigorous'" --include=*.py --include=*.md --include=*.ipynb .`
shows nothing outside `plan/` and `CHANGELOG.md`.

### A-02 `run()` takes `w_data`, and the configs `Conditional` and `Interaction` go

**Why.** After A-01 the only field left on `Conditional` and `Interaction` is `w_data`. The
third signal is data, like X and Y. Its processor is already set in `Processing`.
`interaction_information()` is the same call as `mode='interaction'` (quantities.py:777-799).

**Change.**
1. Change the signature to `run(x_data, y_data=None, w_data=None, *, ...)`.
2. Delete the classes `Conditional` and `Interaction` and the arguments `conditional=` and
   `interaction=`.
3. Delete `interaction_information()`.
4. Keep the warning for a `w_data` that the mode does not read (run.py:1196-1201).

**Tests.** `Conditional(` appears 75 times in 20 test files, `Interaction(` 42 times in 10, and
`interaction_information(` 6 times in 2. Rewrite those calls.

**Pages.** Two tutorial cells and five lines of the reference pages.

**Done when.** `nmi.run(x, y, w, mode='conditional')` runs, and `nmi.Conditional` does not
exist.

### A-03 Transfer entropy is a named function only

**Why.** Transfer entropy belongs to the family of chosen times. After E-09 `mode='transfer'`
is a thin call of `transfer_entropy()`, so the library has two doors to one quantity.

**Change.** Remove `mode='transfer'` from `MODES`, delete the class `Transfer` and the argument
`transfer=`. The functions `transfer_entropy` and `conditional_transfer_entropy` take the
windows, `stride` and `bidirectional` themselves.

**Tests.** `mode='transfer'` appears 49 times in 9 test files and `Transfer(` 31 times in 8.
Rewrite those calls to the functions.

**Pages.** Five lines of the reference pages. No tutorial uses the mode.

**Done when.** `nmi.Transfer` does not exist and `MODES` has no key `'transfer'`.

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
3. At most one window of a call may be a list. `mi_rate` has this rule today
   (quantities.py:854-860). The list is the key that changes the rows in E-10. The name of the
   argument is the column of the result.

**Tests.** The five functions that take `k` are called 63 times in 9 test files, and `mi_rate`
20 times in 5. `prediction_horizon` appears 9 times in 2 test files and `future_k` twice.

**Pages.** Eight tutorial cells and about 18 lines of the reference pages. `reference/THEORY.md`
can keep `k` as a symbol in its formulas if it says which argument sets it.

**Done when.** `grep -rn "future_k\|prediction_horizon" neural_mi/` shows nothing, and no named
function has a parameter called `k` or `h`.

### A-06 `mode='estimate'` takes a grid

**Why.** `estimate` refuses a `sweep_grid` and `sweep` requires one (analysis/modes.py:775-781).
Every other mode takes a grid without changing its name, and the named functions already choose
between the two by whether a grid is present (quantities.py:118 and 664). As a result
`active_information_storage(x, k=5)` reports the mode `'estimate'` and `k=[5]` reports `'sweep'`
for the same number.

**Change.**
1. `MODES['estimate']` accepts a grid, and `'sweep'` is removed.
2. A result with one configuration and one repeat is shown as a training curve. A result with
   more is shown as the table of a sweep, and `Results.best` ranks its configurations.

**Tests.** `mode='sweep'` appears 44 times in 19 test files.

**Pages.** Eight cells in three notebooks and seven lines of the reference pages.

**Done when.** `nmi.run(x, y, sweep_grid={'run_id': range(3)})` runs with the default mode.

### A-07 A result names its quantity and lists its keys

**Why.** `Results.get(key)` searches four tables (results.py:515-569), and nothing lists the
keys that exist. The representation of a result shows its mode and not its quantity.

**Change.**
1. Add `Results.keys()`. It returns the keys that `get` can read, grouped by the table that
   contains them.
2. Show the quantity's name and symbol from `view` (E-11) in `__repr__` and in `summary()`.
3. Say in the docstring of `to_json` that it exports a result, and that `save` and `load` are
   the pair that restores one.

**Done when.** `set(result.keys())` equals the set of keys for which `result.get` succeeds, in
a test over every case of the snapshot.

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

**When.** At any time. The owner supplies the facts.

**Why.** Tutorial 04 loads two pickles of 7.9 MB with `pickle.load`. A pickle can run code when
it is loaded. These two also refer to `numpy.core`, a path that numpy 2 deprecates. The notebook
cites Grosmark and Buzsáki (2016), but nothing states the licence of the data or how the subset
was made.

**Change.**
1. Convert both files to `.npz` and load them with `allow_pickle=False`.
2. Add `tutorials/data/README.md` with the source, the licence and the derivation.
3. Update the loading cell of tutorial 04 and the two lines of `.gitignore`.

**Done when.** `git ls-files tutorials/data` shows no `.pkl` file.

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

### P-13 Rewrite USING.md around the two front doors, and run the tutorials again

**When.** After phase A.

**Why.** `reference/USING.md` presents the modes and the named functions as two lists that
overlap. After phase A they are the two families of the library. The five tutorials and the
notebook of the README figures store their outputs, and the API and some numbers have changed.

**Change.**
1. Organise `USING.md` as chosen signals with `run()`, chosen times with the named functions,
   then the analyses and the two modifiers.
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
