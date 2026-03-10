# NeuralMI Pre-Release Fix Script
# Branch: paper (off jules-docs-polish-3439027828623265045)
# Run all tasks in order. Write progress to FIX_REPORT.md as you go.

================================================================
PRELIMINARY: LOCATE KEY FILES
================================================================
Before starting, confirm the following files exist and note their paths:
- neural_mi/__init__.py
- neural_mi/run.py
- neural_mi/results.py
- neural_mi/visualize/plot.py
- neural_mi/generators/synthetic.py
- pyproject.toml
- .github/workflows/tests.yml
- THEORY.md
- NEURALMI_REFERENCE.md
- paper.md  (repo root)
- paper.bib (repo root)
- tutorials/01_A_First_Estimate.ipynb through tutorials/08_Models_Estimators_and_Validation.ipynb

Report any that are missing before proceeding.

================================================================
TASK 1: VERSION NUMBER
================================================================
Update the version to "2.0.0" everywhere it appears:
- neural_mi/__init__.py  →  __version__ = "2.0.0"
- pyproject.toml         →  version = "2.0.0"
- CHANGELOG.md           →  rename [1.1.0] heading to [2.0.0]

================================================================
TASK 2: FIX TUTORIALS 03 AND 04
================================================================
The function `generate_windowed_dependency_data` has been restored to
`neural_mi/generators/synthetic.py`. Verify it is present and callable.

Then open the following two notebook files and check every cell that
references `generate_windowed_dependency_data`:
- tutorials/03_Temporal_Correlations_and_Splits.ipynb
- tutorials/04_Sweeps.ipynb

For each occurrence, confirm the call signature matches the restored
function's actual signature in synthetic.py. If any argument names or
default values differ between the notebook calls and the function
definition, update the notebook calls to match the function.

Do NOT change the function itself — only fix the call sites if needed.

================================================================
TASK 3: IMPLEMENT Results.summary()
================================================================
Open neural_mi/results.py.

Add a `summary()` method to the Results class. It should print a
human-readable summary to stdout. Requirements:

- Print the analysis mode
- Print mi_estimate if it is not None (with 4 decimal places, and units
  from params dict if available, defaulting to 'bits')
- If mode is 'rigorous', also print details['mi_error'] as the confidence
  interval half-width if present
- If mode is 'rigorous', print details['is_reliable'] as a warning if False
- If dataframe is not None, print its shape (rows × columns) and column names
- Print a short separator line above and below the output for readability

The method signature should be: def summary(self) -> None

Also update the docstring of the Results class to include summary() in the
list of methods alongside plot() and compare().

================================================================
TASK 4: IMPLEMENT Results.plot() FOR mode='estimate'
================================================================
Open neural_mi/results.py.

Find the plot() method. Locate the branch that currently raises
NotImplementedError for mode='estimate'.

Replace it with a training curve plot using data already available in
result.details. The plot should:
- Have two lines: training MI and test MI vs epoch number
- Use result.details['loss_history'] for training MI if present
  (it may be keyed as 'train_mi_history' or 'test_mi_history' —
  inspect what keys are actually present in details for estimate mode
  and use whatever is there)
- Mark the best epoch with a vertical dashed line if
  result.details['best_epoch'] is present
- Label axes: x = "Epoch", y = "MI (bits)" (or nats if applicable)
- Title: "Training curve"
- Return the ax object as the method already does for other modes

If the required keys are genuinely absent from details for estimate mode,
do not raise NotImplementedError — instead raise a clear ValueError with
a message explaining which keys were expected but not found.

Also update NEURALMI_REFERENCE.md §7 Results.plot() table to match
whatever you implement.

================================================================
TASK 5: FIX verbose AND show_progress DEFAULTS
================================================================
5a. In neural_mi/run.py:
- Confirm verbose defaults to False in the function signature
- Confirm show_progress defaults to True in the function signature
- If either is wrong, fix the signature

5b. In NEURALMI_REFERENCE.md §8 Base Parameters table:
- Find the verbose row and set its Default column to False
- Find the show_progress row (add it if missing) and set its Default to True

5c. In NEURALMI_REFERENCE.md §5 run() function signature comment block:
- Find the verbose= line and ensure it shows False
- Find the show_progress= line and ensure it shows True

================================================================
TASK 6: FIX mode PARAMETER DOCSTRING — list all 9 modes
================================================================
6a. In neural_mi/run.py, find the docstring for the `mode` parameter.
Currently it lists only 5 modes. Update it to list all 9:
'estimate', 'sweep', 'dimensionality', 'rigorous', 'lag',
'precision', 'conditional', 'transfer', 'pairwise'

6b. In NEURALMI_REFERENCE.md §5, find the `mode` parameter row in the
run() signature reference. Update it to list all 9 modes.

================================================================
TASK 7: FIX PYTHON VERSION INCONSISTENCY
================================================================
Open .github/workflows/tests.yml.

Find the python-version matrix. It currently tests 3.9 and 3.10.
pyproject.toml declares requires-python = ">=3.11".

Update the CI matrix to test Python 3.11 and 3.12 only.
Do not change pyproject.toml — the >=3.11 requirement is correct.

================================================================
TASK 8: FIX MATPLOTLIB DEPRECATION
================================================================
Open neural_mi/visualize/plot.py, line ~269.

Find the call to `plt.cm.get_cmap(...)` which is deprecated in
Matplotlib 3.7 and removed in 3.11.

Replace it with `matplotlib.colormaps[name]` or `matplotlib.colormaps.get_cmap(name)`
which is the current API. Use whichever form is consistent with the
existing imports in that file.

Check the rest of plot.py for any other deprecated Matplotlib calls
(e.g. plt.show() called inside library functions, use of rcParams in
ways that have changed) and fix any you find.

================================================================
TASK 9: FIX THEORY.md
================================================================
9a. Section 6, opening sentence currently reads:
  "Historically -like last year-, finding the dimensionality..."
Replace with:
  "Until recently, finding the dimensionality..."

9b. Section 6 contains this raw LaTeX URL:
  ($\texttt{https://arxiv.org/abs/2602.08105}$)
Replace with proper Markdown hyperlink:
  ([arxiv:2602.08105](https://arxiv.org/abs/2602.08105))

9c. Read all of THEORY.md and flag any other sentences that use
casual, informal, or colloquial language unsuitable for a technical
reference document. For each one, suggest a replacement in FIX_REPORT.md
but DO NOT automatically change them — list them for author review.

================================================================
TASK 10: FIX NEURALMI_REFERENCE.md INCONSISTENCIES
================================================================
10a. §7 Results.plot() table — estimate mode row:
Currently says: "Training curve (train/test MI vs epoch)"
Update to match whatever was implemented in Task 4.

10b. §7 — add summary() to the documented methods if not already there
after Task 3.

10c. §8 Base Parameters — results.py dataframe docstring says it is
only populated in 'sweep', 'dimensionality', and 'rigorous' modes.
Find this text (it may be in results.py, the reference, or both) and
update it to list all six modes that populate dataframe:
'sweep', 'dimensionality', 'rigorous', 'lag', 'precision', 'pairwise'

10d. DEVELOPERS_GUIDE.md — analysis/ section:
Currently says workflow.py implements mode='rigorous'.
Update to:
- rigorous.py — public dispatcher for mode='rigorous'
- workflow.py — internal AnalysisWorkflow class; prefer run_rigorous_analysis

================================================================
TASK 11: FIX README.md
================================================================
11a. Learning Path section lists only 7 tutorials. Add Tutorial 08:
  "**08_Models_Estimators_and_Validation**: Understand the trade-offs
  between critic architectures, estimators, and custom models, including
  permutation testing for significance."

11b. Quickstart example: remove the two unused variable assignments:
  x_raw_transposed = x_raw.T   ← delete this line
  y_raw_transposed = y_raw.T   ← delete this line

11c. Installation code block: the comment
  "# 1. Clone the repository from GitHub (if in Jupyter or Colab,
  remember to add "!" before running terminal commands like the following"
  is mid-block prose. Move it to a sentence in the prose paragraph
  immediately above the code block.

================================================================
TASK 12: FIX CONCEPTS.md
================================================================
The opening paragraph links to arXiv paper 2506.00330. Verify this URL
is correct and live: https://arxiv.org/abs/2506.00330
If it resolves, no change needed. If it does not resolve, update the
link to the correct arXiv URL for the MI estimation paper.

================================================================
TASK 13: FIX run.py DOCSTRING — tau_grid
================================================================
Find the tau_grid parameter docstring in neural_mi/run.py.

Currently it describes only the noise corruption method. Update it to
correctly describe both methods:
- 'rounding' (default): deterministic quantization — spike times are
  rounded to the nearest multiple of tau
- 'noise': additive uniform noise drawn from U(-tau/2, tau/2)

Also fix the backtick syntax error on the same line:
  ``For ``mode=``precision```
Should be:
  For ``mode='precision'``,

================================================================
TASK 14: ADD DOCSTRING TO precision.py
================================================================
Open neural_mi/analysis/precision.py.

The function run_precision_analysis has no docstring. Add one following
NumPy docstring format. It should include:
- A one-sentence summary
- Parameters section covering: x_data, y_data, tau_grid, corrupt_target,
  corruption_method, n_noise_samples, threshold_ratio, and any other
  top-level parameters the function actually accepts
- Returns section describing the Results object and which details keys
  are populated

Base the docstring content on how the function actually behaves —
read the function body to confirm parameter descriptions are accurate.

================================================================
TASK 15: ADD Returns SECTION TO dimensionality.py DOCSTRING
================================================================
Open neural_mi/analysis/dimensionality.py.

Find run_dimensionality_analysis. Its docstring is missing a Returns
section. Add one that describes:
- The Results object returned
- The columns present in result.dataframe
- Which details keys are populated

================================================================
TASK 16: FIX TUTORIAL 07 DATA INSTRUCTIONS
================================================================
Open tutorials/07_Population_Questions.ipynb.

Find the Data section (should be near the top, before the first code cell).

Add a note explaining how to obtain the data files. The note should say:

"The data files for this tutorial are included in the `data/` subfolder
of the tutorials directory in this repository. They can also be downloaded
separately: the hippocampal dataset (`hippocampus_achilles.npz`) is the
Achilles dataset from Grosmark & Buzsáki (2016), available from the
CRCNS data sharing repository (crcns.org, dataset hc-11). The Allen Brain
Observatory spike data (six `.npy` files) were extracted from the Allen
SDK; see the Allen Brain Observatory documentation for access instructions."

Also add a one-line note to README.md under the Tutorial 07 entry in the
Learning Path, noting that this tutorial requires data files that are
included in the repo's tutorials/data/ folder.

================================================================
TASK 17: FIX paper.md AND paper.bib
================================================================
paper.md and paper.bib are now in the repo root on the paper branch.

17a. In paper.bib:
- Find the entry for abdelaleem2025dimensionality
- The journal/venue field currently says [JOURNAL/VENUE — TO FILL]
  and the title says [FULL TITLE — TO FILL]
- Read THEORY.md Section 6 — it references arxiv:2602.08105 and
  describes the dimensionality method. Use this to fill in what you can.
- Set the arXiv URL to https://arxiv.org/abs/2602.08105
- Leave author names, title, and venue as [TO FILL] placeholders —
  do not guess these

17b. In paper.md:
- Find the Tutorials paragraph and update it to reflect the correct
  notebook names (01_A_First_Estimate.ipynb through
  08_Models_Estimators_and_Validation.ipynb) rather than .py filenames
- The current text already describes tutorials correctly; just verify
  the file extension references if any exist

17c. Count the words in paper.md (excluding YAML front matter).
Report the word count in FIX_REPORT.md. JOSS requires under 1000 words.
If over 1000 words, identify which section is longest and flag it for
the author to trim — do NOT automatically shorten it.

================================================================
TASK 18: FINAL VERIFICATION
================================================================
After all tasks above are complete:

18a. Run: pytest --tb=short 2>&1 | tail -40
     Report pass/fail counts. If any tests that previously passed now
     fail, identify which task likely caused the regression.

18b. Run: pytest --cov=neural_mi --cov-report=term-missing 2>&1 | tail -30
     Report overall coverage. Flag if it dropped below 71%.

18c. Do a final grep across the entire codebase for any remaining
     references to the old version numbers (0.1.0, 1.0.0, 1.1.0):
     grep -r "0\.1\.0\|1\.0\.0\|1\.1\.0" --include="*.py" --include="*.md" --include="*.toml" .
     Report any hits that were not intentionally left (e.g. a CHANGELOG
     entry for an old version is fine; a version string in __init__.py is not).

18d. Verify paper.md exists in repo root and is valid Markdown.
     Verify paper.bib exists in repo root and contains all expected keys:
     abdelaleem2025accurate, abdelaleem2025dimensionality, oord2018representation,
     song2020understanding, kraskov2004estimating, pytorch2019, numpy2020,
     paninski2003estimation

================================================================
TASK 19: WRITE FIX_REPORT.md
================================================================
Write FIX_REPORT.md to the repo root with the following structure:

# NeuralMI Pre-Release Fix Report
Date: [today]
Branch: paper

## Changes applied (by task)
[For each task: what was changed, which files, what the change was]

## Tasks requiring author input
[List any tasks where something was flagged for review rather than
auto-fixed — especially THEORY.md casual language from Task 9c]

## Final test results
[Output from Task 18a and 18b]

## Word count of paper.md
[From Task 17c — flag if over 1000]

## Ready for JOSS submission checklist
[ ] __version__ = "2.0.0" in __init__.py
[ ] Version = "2.0.0" in pyproject.toml
[ ] CHANGELOG.md exists with [2.0.0] entry
[ ] paper.md in repo root, under 1000 words
[ ] paper.bib in repo root with all 8 required entries
[ ] Results.summary() implemented
[ ] Results.plot() works for mode='estimate'
[ ] All 9 modes listed in run() docstring
[ ] Tutorials 03 and 04 call existing generator function
[ ] Tutorial 07 has data download instructions
[ ] CI tests Python 3.11 and 3.12
[ ] Matplotlib deprecation fixed
[ ] All tests passing (report count)
[ ] Zenodo DOI obtained [MANUAL — author must do this]
[ ] Author names and ORCIDs filled in paper.md [MANUAL — author must do this]
[ ] Dimensionality paper title/venue filled in paper.bib [MANUAL — author must do this]
