# NeuralMI Pre-Release Audit Report

Date: 2026-03-10
Branch audited: `jules-docs-polish-3439027828623265045` (commits applied on `paper` branch)

---

## Mechanical fixes applied

| Fix | File |
|-----|------|
| Added `__version__ = "1.1.0"` | `neural_mi/__init__.py` |
| Changed `version` from `"0.1.0"` to `"1.1.0"` | `pyproject.toml` |
| Changed `description` from placeholder to descriptive string | `pyproject.toml` |
| Created CHANGELOG.md with [1.1.0] entry | `CHANGELOG.md` (new file) |

All required exports (`run`, `generators`, `Results`, `set_verbose`) were already
present in `__init__.py`. No exports needed to be added.

**Note:** `paper.md` and `paper.bib` could not be copied to the repo root because
the macOS Downloads folder is not accessible from the current environment (EPERM).
Please copy these two files manually with:
```bash
cp ~/Downloads/paper.md ~/Downloads/paper.bib \
  "<path-to-repo>"
git -C "<path-to-repo>" add paper.md paper.bib
git -C "<path-to-repo>" commit -m "Add JOSS paper source files"
```

---

## Summary counts

| Category | Count |
|----------|-------|
| Inconsistencies found | 8 |
| Missing documentation | 3 |
| Stale content | 4 |
| Minor issues | 10 |
| Tests passing | 211 / 212 (1 skipped) |
| Overall coverage | 71% |

---

## Issues by file

### README.md

**[STALE]** Learning Path section lists only seven tutorials (01–07) and omits
`08_Models_Estimators_and_Validation.ipynb`, which exists in `tutorials/` and covers
permutation testing, custom models, and estimator selection. A reviewer who reads
the README will not know Tutorial 8 exists.

**[MINOR]** The Quickstart example contains a dead variable:
```python
x_raw_transposed = x_raw.T   # created but never used
y_raw_transposed = y_raw.T   # created but never used
```
The `run()` call two lines below correctly uses `x_raw.T` directly. The two
`_transposed` assignments are confusing clutter.

**[MINOR]** The installation code block contains prose mid-block:
```
# 1. Clone the repository from GitHub (if in Jupyter or Colab, remember to add "!" before running terminal commands like the following
```
This looks like a shell comment when rendered. Move the parenthetical note to the
prose above the block.

---

### NEURALMI_REFERENCE.md

**[INCONSISTENCY]** §8 Base Parameters table lists `verbose` default as `True`:
> `| verbose | bool | True | |`

The actual default in both `run()` (line 84: `verbose: bool = False`) and
`defaults.py` (`'verbose': {..., 'default': True}`) is `False` at the `run()`
call site. (Note: `defaults.py` shows `True` as the trainer-internal default,
but `run()` explicitly defaults to `False`, overriding this. The reference
documents the user-facing default, which is `False`.)

**[INCONSISTENCY]** §5 `mode` parameter entry lists only five modes:
> `mode='estimate'`, `sweep`, `dimensionality`, `rigorous`, `lag`

The code supports nine: the four above plus `precision`, `conditional`,
`transfer`, `pairwise`. These four are described in §6 but are absent from the
`mode` parameter row in §5's function signature reference table, which is the
first place a user looks.

**[INCONSISTENCY]** §7 `result.plot()` table claims `estimate` mode produces
a "Training curve (train/test MI vs epoch)":
> `| estimate | Training curve (train/test MI vs epoch) |`

The actual `Results.plot()` method (results.py line 193) raises
`NotImplementedError` for `mode='estimate'`. There is no training-curve plot
implemented.

**[MISSING]** §7 documents `result.summary()` as a method:
> "Prints a human-readable summary to stdout."

The `Results` class (`results.py`) does **not** implement a `summary()` method.
Only `plot()` and `compare()` exist. Calling `result.summary()` on any result
would raise `AttributeError`.

**[MINOR]** §2 Quick Start gives `pip install neural-mi` (hyphenated).
`pyproject.toml` names the package `neuralmi` (no separator); `setup.py` names
it `neural_mi` (underscore). The package is not yet on PyPI, so this is
currently moot, but the three names should be reconciled before release. The
canonical name for pip installs should match `pyproject.toml`.

---

### THEORY.md

**[MINOR]** Section 6 opens with informal language unsuitable for a reference
document:
> "Historically -like last year-, finding the dimensionality of the shared
> information between two datasets …"

"Like last year" is a casual aside. Suggest: "Until recently, …"

**[MINOR]** Section 6 embeds a raw URL using LaTeX `\texttt`:
> `($\texttt{https://arxiv.org/abs/2602.08105}$)`

In Markdown/MkDocs this renders as literal text, not a link. Use:
> `([arxiv:2602.08105](https://arxiv.org/abs/2602.08105))`

---

### CONCEPTS.md

**[MINOR]** Opening sentence:
> "The document is based heavily on [this paper](https://arxiv.org/abs/2506.00330)."

The arXiv ID `2506.00330` has a `25` prefix (2025-June) and a submission number
of `330`, which is implausibly low for that month. This may be a pre-submission
placeholder; verify and update before JOSS submission so the link is live.

---

### DEVELOPERS_GUIDE.md

**[INCONSISTENCY]** The `analysis/` section states:
> "**`workflow.py`**: Implements the `mode='rigorous'` analysis, including
> subsampling and extrapolation logic."

Since refactoring, `rigorous.py` is the dispatcher for `mode='rigorous'`.
`workflow.py` now contains the low-level `AnalysisWorkflow` class, which is an
internal detail. The guide should say:
- `rigorous.py` — public dispatcher for `mode='rigorous'`.
- `workflow.py` — internal `AnalysisWorkflow` class; prefer `run_rigorous_analysis`.

---

### neural_mi/run.py docstring

**[INCONSISTENCY]** The `mode` parameter type stub is incomplete:
> `mode : {'estimate', 'sweep', 'dimensionality', 'rigorous', 'lag'}, default='estimate'`

Four supported modes are missing: `'precision'`, `'conditional'`, `'transfer'`,
`'pairwise'`. The error message on line 800 correctly lists all nine, but the
docstring does not.

**[MINOR]** The `tau_grid` docstring describes only the `noise` corruption method:
> "a list of additive uniform noise centered around 0 (interval [-tau/2, tau/2])"

The default corruption method is `'rounding'` (deterministic quantization), which
does not add noise. The docstring conflates the two methods and is misleading for
the common case.

**[MINOR]** Docstring syntax error on the `tau_grid` line:
> ``For ``mode=``precision``` — mismatched backticks (double-open, triple-close).
Should be: ``For ``mode='precision'``,``

---

### neural_mi/results.py docstring

**[STALE]** `dataframe` attribute docstring says:
> "Populated in 'sweep', 'dimensionality', and 'rigorous' modes."

`dataframe` is also populated by `'lag'`, `'precision'`, and `'pairwise'` modes.
The docstring should list all six.

---

### neural_mi/analysis/dimensionality.py

**[MISSING]** `run_dimensionality_analysis` has no `Returns` section in its
docstring. The function returns a `pd.DataFrame` but the docstring does not
describe the return type or the columns users can expect.

---

### neural_mi/analysis/precision.py

**[MISSING]** `run_precision_analysis` has **no docstring at all**. The function
body begins immediately with `logger.info(...)`. Given this is a public analysis
mode with several non-trivial parameters, it needs at least a summary, a
Parameters section, and a Returns section.

---

### Tutorials

**Tutorial 01 — 01_A_First_Estimate.ipynb**
- Data: **synthetic** (`generate_correlated_gaussians`, ground truth MI = 2.0 bits).
- [MINOR] Cell 13 markdown says "at small sample sizes (n = 200)" but the
  sample-size loop in Cell 12 uses `[100, 300, 500, 1000, 3000, 5000]`. The
  value 200 never appears. Should say "n = 100" or "very small n".

**Tutorial 02 — 02_Neural_Data_Formats.ipynb**
- Data: **synthetic** (sine waves for continuous; generated spike trains;
  generated categorical states).
- No issues found.

**Tutorial 03 — 03_Temporal_Correlations_and_Splits.ipynb**
- Data: **synthetic**.
- **[STALE] CRITICAL:** Cell 3 calls `nmi.generators.generate_windowed_dependency_data(...)`.
  This function **does not exist** in `neural_mi/generators/synthetic.py`. Running
  this cell raises `AttributeError`. Tutorial 03 is completely broken. The
  nearest equivalent is `generate_temporally_convolved_data` or a custom generator;
  clarify which was intended.

**Tutorial 04 — 04_Sweeps.ipynb**
- Data: **synthetic** (same as Tutorial 03).
- **[STALE] CRITICAL:** Uses the same non-existent
  `generate_windowed_dependency_data` (Cell 2, line 74). Tutorial 04 is broken
  for the same reason as Tutorial 03.

**Tutorial 05 — 05_Rigorous_Estimation.ipynb**
- Data: **synthetic** (`generate_nonlinear_from_latent`, ground truth MI = 3.0 bits).
- [MINOR] Cell 1 markdown sentence is grammatically incomplete:
  > "…making it a harder estimation problem because of the"
  The sentence ends abruptly; the rest was presumably truncated.

**Tutorial 06 — 06_Temporal_Questions.ipynb**
- Data: **synthetic** (`generate_correlated_spike_trains`, known 20 ms delay,
  5 ms jitter).
- No API issues found.

**Tutorial 07 — 07_Population_Questions.ipynb**
- Data: **real** — hippocampal spikes from `data/hippocampus_achilles.npz` and
  Allen Brain Observatory LFP from six `.npy` files in `data/`. These files are
  not included in the repository and must be downloaded separately. No instructions
  for obtaining them are provided in the notebook or README.
- [MINOR] Cell 3 contains typo: "For this tutoiral" → "For this tutorial".

**Tutorial 08 — 08_Models_Estimators_and_Validation.ipynb**
- Data: **synthetic** (`generate_correlated_gaussians`).
- Uses permutation testing and custom PyTorch models; all API calls match the
  current code.
- No issues found.

---

### Tests

- **211 passed, 1 skipped, 14 warnings** (27–30 s on Apple Silicon).
- One Matplotlib deprecation warning: `plt.cm.get_cmap` is deprecated in
  Matplotlib 3.7 and removed in 3.11. The call is in `visualize/plot.py:269`.
  This will become an error when Matplotlib 3.11 is released.

**Files below 70% coverage:**

| File | Coverage | Notes |
|------|----------|-------|
| `neural_mi/data/temporal.py` | 65% | Largest file; many edge-case paths untested |
| `neural_mi/results.py` | 41% | `plot()` branches for precision/rigorous untested |
| `neural_mi/visualize/plot.py` | 47% | Most plot paths not exercised by tests |
| `neural_mi/smoke_test.py` | 0% | Not imported by tests; intended for manual use |
| `neural_mi/logger.py` | 68% | Fine-grained verbosity paths untested |
| `neural_mi/utils.py` | 68% | Several utility helpers not covered |

**Test files with fewer than 3 tests:** None — all test files are substantive.

---

### Docs Deployment

- `docs.yml` deploys from `branches: ["main"]` on every push to `main`.
- `tests.yml` runs on push/PR to `main`, testing Python 3.9 and 3.10.
- **[INCONSISTENCY]** `pyproject.toml` declares `requires-python = ">=3.11"` but
  `tests.yml` tests on Python 3.9 and 3.10. These are below the declared minimum.
  Either lower the `requires-python` bound to `>=3.9` or update the CI matrix to
  test only 3.11+. If the library genuinely requires 3.11 features, the CI will
  silently pass tests on unsupported Python versions.
- Docs deployment from `main` is correct practice; no change needed. The
  `workflow_dispatch` trigger allows manually building docs from any branch via
  the Actions tab.

---

## Top 5 recommended fixes before JOSS submission

**1. Fix Tutorials 03 and 04 (CRITICAL — code will not run)**
`generate_windowed_dependency_data` does not exist. These two tutorials raise
`AttributeError` at the first code cell. Either add the missing generator to
`neural_mi/generators/synthetic.py` or replace the call with the correct
function name (`generate_temporally_convolved_data` or similar). A JOSS reviewer
who runs the tutorials will encounter this immediately.

**2. Add `Results.summary()` or remove it from the reference**
The reference promises `result.summary()` as a core method of the Results object.
It does not exist. Either implement it (a few lines to print mode, MI, and key
details) or remove it from the reference and all tutorial prose that references it.
Missing documented methods are a red flag for reviewers evaluating API robustness.

**3. Fix `Results.plot()` for `estimate` mode or update the reference**
The reference documents a training-curve plot for `mode='estimate'`. The code
raises `NotImplementedError`. Either implement the training-curve plot (the data
is available in `result.details['loss_history']`) or update §7 to state that
`estimate` mode does not support `.plot()`. This is the most visible inconsistency
for any user who tries the Quickstart.

**4. Complete the `mode` parameter docstring in `run.py` and `NEURALMI_REFERENCE.md §5`**
The `mode` parameter lists only 5 of the 9 supported modes in both the run()
docstring and the reference §5. JOSS reviewers will check the API completeness.
This is a one-line fix in both files.

**5. Fix the Python version inconsistency in CI**
`pyproject.toml` declares `requires-python = ">=3.11"` but CI tests Python 3.9
and 3.10. This creates a false sense of cross-version compatibility. Align the CI
matrix with the declared requirement before publication so the badge truthfully
reflects tested versions.

---

## Things that are fine and do not need attention

- **All 211 tests pass** on the current branch. The single skip is intentional.
- **`__init__.py` exports** are complete: `run`, `Results`, `generators`,
  `set_verbose`, `set_verbosity` are all exported correctly.
- **THEORY.md core content** is mathematically accurate and well-organized.
  The InfoNCE ceiling derivation, SMILE clipping discussion, and bias-correction
  extrapolation procedure are all correct.
- **NEURALMI_REFERENCE.md §3–6** (estimator table, embedding model table, critic
  table, all mode descriptions) accurately reflects the code. The split-mode
  guidance (blocked vs. random) is correct.
- **`analysis/` modules** for `conditional`, `transfer`, `pairwise`, `rigorous`,
  and `lag` all have correct docstrings and their return values match what the
  reference documents.
- **`Results` dataclass structure** (`mode`, `params`, `mi_estimate`, `dataframe`,
  `details`) matches the reference exactly.
- **`Results.compare()` static method** is fully implemented and matches the
  documented API.
- **CONTRIBUTING.md** is accurate and complete for the current development workflow.
- **Tutorial 01, 02, 05, 06, 08** all run correctly against the current API.
  Tutorial 07 is valid code but requires external data files.
- **Unit conversion** (nats → bits via `_convert_mi_units`) is applied
  consistently across all modes including sub-keys in `details`.
- **Blocked split gap logic** is correctly described in CONCEPTS.md and in the
  `split_gap_fraction` docstring.
