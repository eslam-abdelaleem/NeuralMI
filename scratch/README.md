# scratch

Working material for the correctness fixes in `ROADMAP.md` (C3 to C23). It is
not part of the library, nothing imports it, and the folder is deleted once the
correctness fixes are done.

## investigation/

The scripts and results of the `mi_rate` investigation behind the roadmap's
numbers. Every script takes this directory as its first argument, puts it on
`sys.path`, and imports `neural_mi` from the environment, for example:

```bash
python scratch/investigation/experiments.py scratch/investigation scale out.json
```

What is reusable for the remaining items:

- `reallike.py`: the smooth two-process model (a shared damped oscillation
  under larger private ones, finely sampled, as in field potentials), with exact
  Gaussian values of every term. C15's accuracy test on a smooth,
  self-predictable process can be built on it.
- `exact_terms.py`: exact joint, conditioning and rate terms of `mi_rate` for
  the easy oracle (AR(1) plus noise) and the smooth model, at any `h` and
  half-width.
- `harness.py`: runs `nmi.mi_rate` and pulls out each term's per-network
  diagnostics.
- `whiten_helpers.py`, `whiten_check.py`, `chain_whitened_sweep.py`,
  `block_whitened.py`: autoregressive pre-whitening and its checks, the
  reference for C7's time-domain option.
- `sn_study.py`, `sn_summary.py`: the spectral-normalisation study behind C5
  and the Estimation entry on it.
- `check_*.py`: the small checks behind individual findings, among them
  `check_k0_equivalence.py` (C3), `check_spike_scale*.py` (C8),
  `check_custom_critic.py` (C9) and `check_c1_case.py` (the 2.41 case of C1).
- `c18_budget.py`, `c18_analyse.py`: the epoch-budget measurement behind C18
  and C19. The first trains five oracles with and without shifted windows for
  300 epochs with stopping off and records every network's curves
  (`results/c18_budget.json`). The second replays shorter budgets and patience
  rules on those curves with the library's own smoothing and epoch rule. Both
  take the repository as their first argument.
- `c20_ksg_ties.py`: KSG's conditional estimate on spike windows where the
  truth is 0, behind C20. It takes the repository as its first argument.
- `c21_c22_tutorial_settings.py`: the two tutorial results the full run on the
  defaults lost, under the old settings and variants of the defaults, behind
  C21 and C22 (`results/c21_c22_tutorial_settings.txt`). It takes the
  repository as its first argument.
- `c21_rung_rules.py`: rigorous ladders on tutorial 02's data at several sample
  sizes and on independent data, refitted under four rules for which rungs the
  fit uses, behind C21 (`results/c21_rung_rules.txt`). It takes the repository
  as its first argument.
- `c22_split_bisect.py`: tutorial 04 section 7's October call on the library of
  any commit, run by `git bisect` behind C22 (`results/c22_split_bisect.txt`).
  It takes the checkout to test, which `PYTHONPATH` must also name.
- `c22_leak_settings.py`: where a random split of overlapping windows still
  reads high on the current library: less data, smooth data, and encoders that
  read a pattern wherever it sits (`results/c22_leak_settings.txt`). It takes
  the repository as its first argument.
- `c23_tutorial_warnings.py`: every warning the tutorials' saved outputs show,
  by cell and section, with the library's count of each, behind C23. It takes
  the repository and a git revision (`results/c23_tutorial_warnings.txt` holds
  the full run on the defaults and the October run before the fixes).

`results/` holds the JSON outputs those scripts wrote.
