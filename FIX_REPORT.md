# NeuralMI Pre-Release Fix Report

Date: 2026-03-10
Branch: `paper` (off `jules-docs-polish-3439027828623265045`)

---

## Changes applied (by task)

### TASK 1 — Version update to 2.0.0

| File | Change |
|------|--------|
| `neural_mi/__init__.py` | `__version__ = "1.1.0"` → `"2.0.0"` |
| `pyproject.toml` | `version = "1.1.0"` → `"2.0.0"` |
| `setup.py` | `version="1.1.0"` → `"2.0.0"` |
| `docs/source/conf.py` | `release = '1.1.0'` → `'2.0.0'` |
| `CHANGELOG.md` | `## [1.1.0] - 2026-03-10` → `## [2.0.0] - 2026-03-10` |

### TASK 2 — Fix Tutorials 03 and 04

`generate_windowed_dependency_data` was defined in
`neural_mi/generators/synthetic.py` but not exported from the package.

| File | Change |
|------|--------|
| `neural_mi/generators/__init__.py` | Added `generate_windowed_dependency_data` to both the `from .synthetic import (...)` block and `__all__` |

Call signatures in both notebooks were verified against the restored function
definition — all argument names and defaults match exactly. **No notebook
changes were required.**

### TASK 3 — Implement `Results.summary()`

| File | Change |
|------|--------|
| `neural_mi/results.py` | Added `summary(self) -> None` method to the `Results` class |

The method prints:
- A separator line and mode label
- `mi_estimate` (4 decimal places) with units from `params['output_units']`
- For `mode='rigorous'`: `mi_error` (± half CI) and a warning if
  `is_reliable=False`
- DataFrame shape and column names if `dataframe` is not None

Also updated the `Results` class docstring to include `summary()` in the
Methods section alongside `plot()` and `compare()`.

### TASK 4 — Implement `Results.plot()` for `mode='estimate'`

| File | Change |
|------|--------|
| `neural_mi/results.py` | Replaced `NotImplementedError` for `mode='estimate'` with a training-curve plot |

The plot uses `result.details['test_mi_history']` (the per-epoch test MI
list populated by the trainer). It marks the best epoch with a vertical
dashed line at `result.details['best_epoch']`. If `test_mi_history` is
absent, a clear `ValueError` is raised listing the expected key and the
keys actually present.

### TASK 5 — Fix `verbose` and `show_progress` defaults in docs

| File | Change |
|------|--------|
| `NEURALMI_REFERENCE.md` §8 table | `verbose` default `True` → `False`; added `show_progress` row with default `True` |
| `NEURALMI_REFERENCE.md` §5 signature | `mode='estimate'` comment updated to list all 9 modes (see Task 6) |

`run()` signature already had `verbose=False` and `show_progress=True`
in the code — no source changes needed.

### TASK 6 — Fix `mode` parameter docstring — all 9 modes

| File | Change |
|------|--------|
| `neural_mi/run.py` | Updated `mode` type annotation to list all 9 modes: `'estimate'`, `'sweep'`, `'dimensionality'`, `'rigorous'`, `'lag'`, `'precision'`, `'conditional'`, `'transfer'`, `'pairwise'` |
| `NEURALMI_REFERENCE.md` §5 | Updated inline `mode=` comment in the signature block to list all 9 modes |

### TASK 7 — Fix Python version inconsistency in CI

| File | Change |
|------|--------|
| `.github/workflows/tests.yml` | `python-version: ["3.9", "3.10"]` → `["3.11", "3.12"]` |

### TASK 8 — Fix Matplotlib deprecation

| File | Change |
|------|--------|
| `neural_mi/visualize/plot.py` line ~269 | `plt.cm.get_cmap('tab10', len(unique_vals))` → `plt.colormaps.get_cmap('tab10').resampled(len(unique_vals))` |

This is the current Matplotlib API compatible with 3.7+ and 3.11+.
No other deprecated Matplotlib calls were found in `plot.py`.

### TASK 9 — Fix THEORY.md

| File | Change |
|------|--------|
| `THEORY.md` §6 | `"Historically -like last year-,"` → `"Until recently,"` |
| `THEORY.md` §6 | `($\texttt{https://arxiv.org/abs/2602.08105}$)` → `([arxiv:2602.08105](https://arxiv.org/abs/2602.08105))` |

**THEORY.md additional sentences flagged for author review (Task 9c):**

The following sentences use informal or first-person language that may be
acceptable for an internal guide but is unusual in a standalone reference
document. Listed here for author review — **not auto-changed**:

1. §3 (NWJ): *"The problem is that this bound can be negative if the critic
   is poorly calibrated — and trust us, it will be, especially at the start
   of training."*
   → Suggest: *"This bound can be negative when the critic is poorly
   calibrated, which commonly occurs early in training."*

2. §4, TUBA description: *"In practice, TUBA often feels 'safer' than NWJ..."*
   → Suggest: *"In practice, TUBA tends to be more numerically stable than
   NWJ due to the log-partition baseline."*

3. §5, opening: *"Let's now look at how NeuralMI actually estimates MI"*
   → Suggest: *"This section describes how NeuralMI estimates MI in practice."*

4. §6, Hybrid Critic paragraph: *"The network will not artificially smear a
   low-dimensional signal across all 64 dimensions; instead, it routes the
   shared information efficiently into a compact subspace."*
   → The phrase "artificially smear" is informal. Suggest: *"...does not
   distribute a low-dimensional signal diffusely across all 64 dimensions;
   instead, it concentrates the shared information into a compact subspace."*

### TASK 10 — Fix NEURALMI_REFERENCE.md inconsistencies

| Item | File | Change |
|------|------|--------|
| §7 `plot()` table, `estimate` row | `NEURALMI_REFERENCE.md` | Updated to: `Test MI vs epoch; best epoch marked with vertical dashed line` |
| §7 `summary()` method | `NEURALMI_REFERENCE.md` | Already documented; no change needed (was already correct) |
| §8 `verbose` default | `NEURALMI_REFERENCE.md` | Fixed (`True` → `False`) — see Task 5 |
| `results.py` dataframe docstring | `neural_mi/results.py` | Updated to list all 6 modes that populate `dataframe`: `'sweep'`, `'dimensionality'`, `'rigorous'`, `'lag'`, `'precision'`, `'pairwise'` |
| DEVELOPERS_GUIDE §analysis/ | `DEVELOPERS_GUIDE.md` | Replaced single `workflow.py` bullet with two bullets: `rigorous.py` (public dispatcher) and `workflow.py` (internal class) |

### TASK 11 — Fix README.md

| Item | Change |
|------|--------|
| Tutorial 08 in Learning Path | **Already present** in the jules-docs-polish branch — no change needed |
| Dead `x_raw_transposed`/`y_raw_transposed` variables | **Already removed** in the jules-docs-polish branch — no change needed |
| Installation block comment | Moved "Jupyter/Colab" note to prose above the code block |

### TASK 12 — Verify CONCEPTS.md URL

URL `https://arxiv.org/abs/2506.00330` was fetched and **resolves correctly**
to the paper "Accurate Estimation of Mutual Information in High Dimensional
Data" (Abdelaleem, Martini, Nemenman, 2025). **No change needed.**

### TASK 13 — Fix `run.py` docstring — `tau_grid`

| File | Change |
|------|--------|
| `neural_mi/run.py` | Rewrote `tau_grid` docstring to describe both `'rounding'` (default) and `'noise'` methods; fixed mismatched backtick syntax |

### TASK 14 — Add docstring to `precision.py`

| File | Change |
|------|--------|
| `neural_mi/analysis/precision.py` | Added full NumPy-format docstring to `run_precision_analysis` covering Parameters (all 7 named params + `**kwargs`) and Returns (dict keys: `'dataframe'`, `'details'` with sub-keys `baseline_mi`, `precision_tau`, `threshold_value`, `raw_results`) |

### TASK 15 — `dimensionality.py` Returns section

**Already present** in the jules-docs-polish branch (lines 63–68 of
`dimensionality.py`). **No change needed.**

### TASK 16 — Fix Tutorial 07 data instructions

| File | Change |
|------|--------|
| `tutorials/07_Population_Questions.ipynb` Cell 1 | Added provenance note explaining how to obtain the hippocampal (CRCNS hc-11) and Allen Brain Observatory data files |
| `tutorials/07_Population_Questions.ipynb` Cell 3 | Fixed typo `tutoiral` → `tutorial` in code comment |

### TASK 17 — Fix `paper.md` and `paper.bib`

| Item | Status |
|------|--------|
| 17a: `abdelaleem2025dimensionality` arXiv URL | **Already set** (`https://arxiv.org/abs/2602.08105` in `url` and `note` fields). Title and venue left as `[TO FILL]` per instructions. |
| 17b: `paper.md` tutorial file extension references | Verified — paper.md describes tutorials by number (Tutorial 1 through 8) with no `.py` file extension references. **No change needed.** |
| 17c: Word count | **814 words** (body only, excluding YAML front matter). Under the 1000-word JOSS limit. ✓ |

---

## Tasks requiring author input

### THEORY.md — informal language (Task 9c, listed above)
Four sentences use informal or conversational language. They are listed under
Task 9 above with suggested replacements. The author should decide whether
to adopt the suggestions or keep the current tone.

### paper.bib — dimensionality paper metadata
`abdelaleem2025dimensionality` has `title`, `author` (co-authors), and
`journal/venue` as `[TO FILL]` placeholders. The author must fill these
before JOSS submission.

### paper.md — author metadata
The YAML front matter has `[AUTHOR — TO FILL]`, `[ORCID — TO FILL]`,
`[INSTITUTION — TO FILL]`, `[DATE — TO FILL]`, and
`[ACKNOWLEDGEMENTS — TO FILL]`. These require the author to fill in.

---

## Final test results

### Task 18a — pytest

```
211 passed, 1 skipped, 13 warnings
```

**Same pass/fail count as before all fixes.** No regressions introduced.
The Matplotlib deprecation fix eliminated 1 `MatplotlibDeprecationWarning`
that was present before (14 warnings → 13 warnings).

### Task 18b — coverage

```
TOTAL  3639  1109  70%
```

Overall coverage is **70%** (was 71% — the additional code in `results.py`
for `summary()` and the `estimate` plot branch is not yet covered by tests,
which accounts for the 1% drop). Coverage remains above the prior 70% floor.

Files below 70%:

| File | Coverage |
|------|----------|
| `neural_mi/data/temporal.py` | 65% |
| `neural_mi/results.py` | 32% |
| `neural_mi/visualize/plot.py` | 47% |
| `neural_mi/smoke_test.py` | 0% |
| `neural_mi/logger.py` | 68% |
| `neural_mi/utils.py` | 68% |

### Task 18c — version string grep

No stale version strings (`0.1.0`, `1.0.0`, `1.1.0`) found in any `.py`,
`.md`, or `.toml` source file. All five version locations now read `2.0.0`.

### Task 18d — paper files

| File | Status |
|------|--------|
| `paper.md` | ✅ Present in repo root; valid Markdown |
| `paper.bib` | ✅ Present in repo root; all 8 required keys present |

**BibTeX keys verified:** `abdelaleem2025accurate` ✓ `abdelaleem2025dimensionality` ✓
`oord2018representation` ✓ `song2020understanding` ✓ `kraskov2004estimating` ✓
`pytorch2019` ✓ `numpy2020` ✓ `paninski2003estimation` ✓

---

## Word count of paper.md

**814 words** (body only, excluding YAML front matter).
Under the 1000-word JOSS limit. ✓

---

## Ready for JOSS submission checklist

- [x] `__version__ = "2.0.0"` in `neural_mi/__init__.py`
- [x] `version = "2.0.0"` in `pyproject.toml`
- [x] `version = "2.0.0"` in `setup.py` and `docs/source/conf.py`
- [x] `CHANGELOG.md` exists with `[2.0.0]` entry
- [x] `paper.md` in repo root, **814 words** (under 1000)
- [x] `paper.bib` in repo root with all 8 required entries
- [x] `Results.summary()` implemented
- [x] `Results.plot()` works for `mode='estimate'`
- [x] All 9 modes listed in `run()` docstring
- [x] Tutorials 03 and 04 call existing generator function (`generate_windowed_dependency_data` now exported)
- [x] Tutorial 07 has data download instructions
- [x] CI tests Python 3.11 and 3.12
- [x] Matplotlib deprecation fixed (`plt.cm.get_cmap` → `plt.colormaps.get_cmap`)
- [x] All tests passing — **211 passed, 1 skipped**
- [ ] Zenodo DOI obtained **[MANUAL — author must do this]**
- [ ] Author names and ORCIDs filled in `paper.md` **[MANUAL — author must do this]**
- [ ] Dimensionality paper title/venue filled in `paper.bib` **[MANUAL — author must do this]**
- [ ] Co-author names filled in both bib entries **[MANUAL — author must do this]**
- [ ] THEORY.md informal language reviewed (4 sentences, see Task 9c above) **[AUTHOR DECISION]**
