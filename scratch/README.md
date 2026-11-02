# scratch

Working material for the correctness fixes in `ROADMAP.md` (C3 to C15). It is
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

`results/` holds the JSON outputs those scripts wrote.
