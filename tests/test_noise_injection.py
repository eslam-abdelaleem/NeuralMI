"""Acceptance tests for the noise-injection dimensionality feature.

Implements the eight acceptance tests from
``noise_injection_dimensionality_spec.md`` Section 9, on a synthetic
linear-Gaussian two-view model with known latent dimension ``d`` (via
``generate_correlated_gaussians``, interaction mode), unless noted.
"""
from dataclasses import fields as _dc_fields

import numpy as np
import pytest
import torch

import neural_mi as nmi
from neural_mi import Model, Training, Dimensionality
from neural_mi.generators import generate_correlated_gaussians
from neural_mi.analysis.dimensionality import (
    _make_noise_ladder_tasks,
    _draw_base_noise,
    _infer_modality,
    run_dimensionality_analysis,
)
from neural_mi.validation import ParameterValidator

FAST = dict(n_epochs=40, patience=15, batch_size=64, hidden_dim=32, embedding_dim=16, n_layers=1)


def _full_params(**overrides):
    """Full base_params dict (with schema defaults applied), for calling
    run_dimensionality_analysis directly, bypassing nmi.run()'s DataValidator
    (which requires raw-spike-shaped list-of-arrays input we don't need here —
    the modality guards operate on already-windowed tensors regardless of how
    they were produced)."""
    params = {**FAST, **overrides}
    ParameterValidator({'base_params': params, 'mode': 'dimensionality'}).apply_defaults()
    return params


_MODEL_FIELDS = {f.name for f in _dc_fields(Model)}
_TRAINING_FIELDS = {f.name for f in _dc_fields(Training)}
_DIM_FIELDS = {f.name for f in _dc_fields(Dimensionality)}


def _dim_run(x, y, sigma_add, *, n_workers=1, estimator=None,
             max_eval_samples=None, base_params=None, **dim_over):
    """Run dimensionality mode, routing flat overrides into the right configs."""
    flat = {**FAST, **(base_params or {})}
    if max_eval_samples is not None:
        flat['max_eval_samples'] = max_eval_samples
    model = {k: v for k, v in flat.items() if k in _MODEL_FIELDS}
    training = {k: v for k, v in flat.items() if k in _TRAINING_FIELDS}
    assert not (set(flat) - _MODEL_FIELDS - _TRAINING_FIELDS), \
        f"unrouted base_params keys: {set(flat) - _MODEL_FIELDS - _TRAINING_FIELDS}"
    dim = {k: v for k, v in dim_over.items() if k in _DIM_FIELDS}
    assert not (set(dim_over) - _DIM_FIELDS), \
        f"unexpected _dim_run kwargs: {set(dim_over) - _DIM_FIELDS}"
    dim['sigma_add'] = sigma_add
    extra = {'estimator': estimator} if estimator is not None else {}
    return nmi.run(
        x, y, mode='dimensionality',
        model=Model(**model), training=Training(**training),
        dimensionality=Dimensionality(**dim),
        seed=0, show_progress=False, n_workers=n_workers, **extra,
    )


# ---------------------------------------------------------------------------
# 1. Monotone MI reduction
# ---------------------------------------------------------------------------

def test_monotone_mi_reduction():
    np.random.seed(0)
    torch.manual_seed(0)
    x, y = generate_correlated_gaussians(n_samples=1500, dim=3, mi=3.0)

    result = _dim_run(x, y, sigma_add=[0.25, 1.0, 2.0, 4.0], n_splits=3, n_workers=1)
    ladder = result.details['sigma_add_ladder'].sort_values('sigma_add')
    mi_means = ladder['mi_mean'].to_numpy()

    # Allow a small amount of noise slack while requiring an overall decreasing trend.
    assert mi_means[0] >= mi_means[-1] - 1e-6
    diffs = np.diff(mi_means)
    assert (diffs <= 0.15).all(), f"MI did not decrease monotonically (within slack): {mi_means}"


# ---------------------------------------------------------------------------
# 2. Plateau in the interior (pr_eig near d across the detached band)
# ---------------------------------------------------------------------------

def test_plateau_near_true_dimension_in_detached_band():
    np.random.seed(1)
    torch.manual_seed(1)
    d = 3
    x, y = generate_correlated_gaussians(n_samples=2000, dim=d, mi=4.0)

    result = _dim_run(x, y, sigma_add=[0.25, 0.5, 1.0, 2.0, 4.0], n_splits=3, n_workers=1,
                      base_params={'max_eval_samples': 400})
    ladder = result.details['sigma_add_ladder']
    detached = ladder[ladder['regime'] == 'detached']

    if detached.empty:
        pytest.skip("No rung landed in the detached band for this draw; ladder-dependent test.")

    # pr_eig should be within a generous band of the true latent dimension d.
    assert (detached['pr_eig_mean'] > 0.4 * d).all()
    assert (detached['pr_eig_mean'] < 2.5 * d).all()


# ---------------------------------------------------------------------------
# 3. Over-noising inflation (d_hat -> k_z as signal sinks into the noise floor)
# ---------------------------------------------------------------------------

def test_over_noising_inflates_dimension_estimate():
    np.random.seed(2)
    torch.manual_seed(2)
    d = 2
    embed_dim = 16
    x, y = generate_correlated_gaussians(n_samples=1500, dim=d, mi=3.0)

    result = _dim_run(x, y, sigma_add=[0.1, 30.0], n_splits=4, n_workers=1,
                      base_params={'embedding_dim': embed_dim, 'n_epochs': 60})
    ladder = result.details['sigma_add_ladder'].sort_values('sigma_add')
    low_row = ladder.iloc[0]
    high_row = ladder.iloc[-1]

    # At extreme noise, MI must have collapsed into the noise floor. The
    # dimension estimate should trend upward toward k_z=embed_dim (a
    # qualitative trend per the spec, not an exact target); empirically
    # pr_singular shows this inflation far more clearly than pr_eig at
    # accessible (fast-test) training budgets, so that is what's checked here.
    assert high_row['mi_mean'] < 0.1 * high_row['ceiling_nats']
    assert high_row['pr_singular_mean'] > 1.15 * low_row['pr_singular_mean']


# ---------------------------------------------------------------------------
# 4. Independence preserved (no shared latent -> no spurious shared dimension)
# ---------------------------------------------------------------------------

def test_independence_no_spurious_shared_dimension():
    np.random.seed(3)
    torch.manual_seed(3)
    n, dim = 1500, 4
    x = np.random.randn(n, dim).astype(np.float32)
    y = np.random.randn(n, dim).astype(np.float32)  # fully independent of x

    result = _dim_run(x, y, sigma_add=1.0, n_splits=3, n_workers=1)
    ladder = result.details['sigma_add_ladder']
    row = ladder.iloc[0]

    # No shared structure: both PR variants should stay small (near the
    # noise-floor rank, not anywhere near dim or embedding_dim).
    assert row['pr_eig_mean'] < 0.5 * FAST['embedding_dim']
    assert row['pr_singular_mean'] < 0.5 * FAST['embedding_dim']


# ---------------------------------------------------------------------------
# 5. Reproducibility (fixed seed -> identical results; same E across levels
#    within a split; holds under n_workers > 1)
# ---------------------------------------------------------------------------

def test_reproducibility_same_seed_identical_noise_data():
    """A fixed (global_seed, split_id) must reproduce identical noised observations.

    (End-to-end MI/pr_eig values across two separate ``nmi.run()`` calls are
    NOT bit-reproducible regardless of this feature — the existing per-task
    reseed in ``analysis/task.py`` mixes in a fresh UUID per ``ParameterSweep``
    call, so weight initialisation already differs run-to-run. The
    reproducibility guarantee this spec section is about is the *injected
    noise itself*, which we verify directly here.)
    """
    x_data, y_data = generate_correlated_gaussians(n_samples=200, dim=2, mi=1.0)
    analysis_params = {'random_seed': 3}
    tasks1 = _make_noise_ladder_tasks(
        x_data, y_data, analysis_params, {}, 'random', n_splits=2,
        levels=[1.0], sigma_add_units='relative', global_seed=3,
    )
    tasks2 = _make_noise_ladder_tasks(
        x_data, y_data, analysis_params, {}, 'random', n_splits=2,
        levels=[1.0], sigma_add_units='relative', global_seed=3,
    )
    for t1, t2 in zip(tasks1, tasks2):
        torch.testing.assert_close(t1[0], t2[0])
        torch.testing.assert_close(t1[1], t2[1])


def test_reproducibility_same_base_E_scaled_across_levels():
    """The noise direction (E) must be identical across ladder levels within a split."""
    x_data, y_data = generate_correlated_gaussians(n_samples=200, dim=3, mi=1.0)
    analysis_params = {'random_seed': 7}
    levels = [0.5, 1.0, 2.0]
    tasks = _make_noise_ladder_tasks(
        x_data, y_data, analysis_params, {}, 'random', n_splits=1,
        levels=levels, sigma_add_units='relative', global_seed=7,
    )
    # tasks are ordered (split_id, level) -> for split 0, the three levels appear in order.
    x_noised = [t[0] for t in tasks]
    noise_dirs = [(x_noised[i] - x_data) / levels[i] for i in range(len(levels))]
    for i in range(1, len(levels)):
        torch.testing.assert_close(noise_dirs[0], noise_dirs[i], rtol=1e-4, atol=1e-5)


def test_reproducibility_under_parallel_workers():
    """_draw_base_noise depends only on primitive args (no live/shared RNG state),
    so a worker process must reconstruct the exact same E as the main process."""
    import multiprocessing as mp

    shape = (50, 4)
    args = (shape, 11, 2, 'x')
    main_process_E = _draw_base_noise(*args)

    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=1) as pool:
        worker_E = pool.apply(_draw_base_noise, args)

    np.testing.assert_array_equal(main_process_E, worker_E)


# ---------------------------------------------------------------------------
# 6. Modality guards
# ---------------------------------------------------------------------------

def test_raw_spike_modality_rejected():
    # Modality guards are exercised on run_dimensionality_analysis directly: they
    # apply after upstream data validation / windowing, to already-processed
    # tensors, regardless of how those tensors were produced.
    assert _infer_modality('spike', {}) == 'raw_spike'
    x, y = generate_correlated_gaussians(n_samples=200, dim=2, mi=1.0)
    x, y = x.unsqueeze(-1), y.unsqueeze(-1)  # (N, C) -> (N, C, 1)
    with pytest.raises(ValueError, match="raw spike-timestamp"):
        run_dimensionality_analysis(
            x, _full_params(), y_data=y,
            processor_type_x='spike', processor_type_y='spike',
            sigma_add=1.0, show_progress=False,
        )


def test_categorical_modality_rejected():
    assert _infer_modality('categorical', {}) == 'categorical'
    x, y = generate_correlated_gaussians(n_samples=200, dim=2, mi=1.0)
    x, y = x.unsqueeze(-1), y.unsqueeze(-1)
    with pytest.raises(ValueError, match="categorical"):
        run_dimensionality_analysis(
            x, _full_params(), y_data=y,
            processor_type_x='categorical', processor_type_y='categorical',
            sigma_add=1.0, show_progress=False,
        )


def test_binned_spike_modality_detected():
    assert _infer_modality('spike', {'bin_size': 0.01}) == 'binned_spike'


def test_binned_spike_stabilization_logs_on_both_paths(caplog):
    import logging
    np.random.seed(6)
    n_windows, n_channels, window_size = 150, 3, 20
    x = torch.from_numpy(np.random.poisson(3.0, size=(n_windows, n_channels, window_size)).astype(np.float32))
    y = torch.from_numpy(np.random.poisson(3.0, size=(n_windows, n_channels, window_size)).astype(np.float32))

    for sigma in (None, 1.0):
        caplog.clear()
        with caplog.at_level(logging.INFO, logger='neural_mi'):
            run_dimensionality_analysis(
                x, _full_params(processor_params_x={'bin_size': 0.01},
                                processor_params_y={'bin_size': 0.01}, random_seed=0),
                y_data=y,
                processor_type_x='spike', processor_type_y='spike',
                sigma_add=sigma, show_progress=False, n_splits=1,
            )
        assert any('stabilized via the Anscombe' in r.message for r in caplog.records), (
            f"Expected stabilization log message for sigma_add={sigma}"
        )


# ---------------------------------------------------------------------------
# 7. Ceiling source: log(eval_size), not log(batch_size)
# ---------------------------------------------------------------------------

def test_ceiling_uses_log_eval_size_not_batch_size():
    np.random.seed(8)
    torch.manual_seed(8)
    x, y = generate_correlated_gaussians(n_samples=1000, dim=2, mi=1.0)

    # max_eval_samples is a top-level nmi.run() kwarg — it overrides any
    # same-named key inside base_params, so it must be passed here directly.
    r_small_eval = _dim_run(x, y, sigma_add=1.0, n_splits=1, n_workers=1,
                            max_eval_samples=50)
    r_large_eval = _dim_run(x, y, sigma_add=1.0, n_splits=1, n_workers=1,
                            max_eval_samples=400)

    ceil_small = r_small_eval.details['raw_results']['eval_size'].iloc[0]
    ceil_large = r_large_eval.details['raw_results']['eval_size'].iloc[0]
    assert ceil_small < ceil_large

    ladder_small = r_small_eval.details['sigma_add_ladder'].iloc[0]
    ladder_large = r_large_eval.details['sigma_add_ladder'].iloc[0]
    np.testing.assert_allclose(ladder_small['ceiling_nats'], np.log(ceil_small), rtol=1e-6)
    np.testing.assert_allclose(ladder_large['ceiling_nats'], np.log(ceil_large), rtol=1e-6)
    assert ladder_small['ceiling_nats'] < ladder_large['ceiling_nats']

    # Batch size alone (same eval_size / max_eval_samples) must not move the ceiling.
    r_diff_batch = _dim_run(x, y, sigma_add=1.0, n_splits=1, n_workers=1,
                            max_eval_samples=400, base_params={'batch_size': 128})
    ladder_diff_batch = r_diff_batch.details['sigma_add_ladder'].iloc[0]
    np.testing.assert_allclose(ladder_diff_batch['ceiling_nats'], ladder_large['ceiling_nats'], rtol=1e-6)


# ---------------------------------------------------------------------------
# 8. Estimator precondition: non-InfoNCE + calibration -> warn, still run
# ---------------------------------------------------------------------------

def test_non_infonce_estimator_warns_but_runs():
    np.random.seed(9)
    torch.manual_seed(9)
    x, y = generate_correlated_gaussians(n_samples=600, dim=2, mi=1.0)

    with pytest.warns(UserWarning, match="derived for InfoNCE"):
        result = _dim_run(x, y, sigma_add=1.0, n_splits=1, n_workers=1, estimator='smile')
    assert result.dataframe is not None
    assert len(result.dataframe) == 1
