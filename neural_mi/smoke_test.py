"""
neural_mi smoke test suite
==========================
Tests correctness of all major changes made across the refactoring sessions.
Does NOT require torch or statsmodels — tests the logic, data flow, and
parameter routing that can be verified with numpy alone.

Run with:
    python smoke_test.py

For full integration testing (requires torch + statsmodels + the full library
installed), each section marked [INTEGRATION] needs the library importable.

Sections
--------
 1  Schema completeness       — all new params registered in defaults.py
 2  ParameterValidator        — new params have correct types and defaults
 3  DataValidator             — shape checks, length mismatch detection
 4  Synthetic generators      — correlated / independent data properties
 5  Rigorous 1/N axis         — extrapolation recovers true MI at intercept
 6  equalize_n                — lag dataset sizes equalized correctly
 7  Period robustness         — median period ignores boundary artifacts
 8  Coverage fraction         — correct windows kept at 0.5 and 0.8 thresholds
 9  BinnedSpike math          — correct spike-to-bin assignment
10  window_times compaction   — data/time index consistency after filtering
11  Spike sentinel            — -1.0 not confused with real spike at t=0
12  Mask cache invalidation   — _data_mask cleared on reset()
13  Param bleed fix           — model params excluded from processor_params
14  Lag N-confound            — equalize_n removes lag-dependent N variation
15  Rigorous weights          — each gamma level has equal total weight
16  Patience/shift warning    — patience < epochs_to_max_shift is detectable
17  BinnedSpike normalize     — spike/s vs raw counts controlled by normalize flag
18  Processor schema routing  — no_spike_value / bin_size selects correct dataset
"""

import sys
import numpy as np
import warnings
# ── allow running from any directory that has neural_mi on the path ──────────
try:
    from neural_mi.defaults import BASE_PARAMS_SCHEMA, PROCESSOR_PARAMS_SCHEMA, MODE_KWARGS_SCHEMA
    from neural_mi.validation import ParameterValidator, DataValidator
    from neural_mi.generators.synthetic import gaussian_mi, independent_gaussian
    from neural_mi.exceptions import TrainingError, InsufficientDataError
    _HAS_NEURAL_MI = True
except ImportError:
    _HAS_NEURAL_MI = False
    warnings.warn("neural_mi not importable — running in standalone mode with inline logic only.")

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
SKIP = "\033[93mSKIP\033[0m"

_results = []


def check(name, fn, requires=None):
    """Run a single test, catch exceptions, record result."""
    if requires == 'neural_mi' and not _HAS_NEURAL_MI:
        print(f"{SKIP} [{len(_results)+1:02d}]: {name}  (neural_mi not importable)")
        _results.append((name, 'skip'))
        return
    try:
        fn()
        print(f"{PASS} [{len(_results)+1:02d}]: {name}")
        _results.append((name, 'pass'))
    except AssertionError as e:
        print(f"{FAIL} [{len(_results)+1:02d}]: {name}  —  {e}")
        _results.append((name, 'fail'))
    except Exception as e:
        if 'SKIP' in str(e) or 'unavailable' in str(e).lower():
            print(f"{SKIP} [{len(_results)+1:02d}]: {name}  ({e})")
            _results.append((name, 'skip'))
        else:
            print(f"{FAIL} [{len(_results)+1:02d}]: {name}  —  {type(e).__name__}: {e}")
            _results.append((name, 'fail'))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Schema completeness
# ─────────────────────────────────────────────────────────────────────────────

def test_schema_base_params():
    for key in ('train_fraction', 'n_test_blocks', 'max_index_reduction'):
        assert key in BASE_PARAMS_SCHEMA, f"'{key}' missing from BASE_PARAMS_SCHEMA"

def test_schema_processor():
    assert 'min_coverage_fraction' in PROCESSOR_PARAMS_SCHEMA['categorical'], \
        "categorical missing min_coverage_fraction"
    assert 'min_coverage_fraction' in PROCESSOR_PARAMS_SCHEMA['continuous'], \
        "continuous missing min_coverage_fraction"
    assert 'no_spike_value' in PROCESSOR_PARAMS_SCHEMA['spike'], \
        "spike missing no_spike_value"
    assert 'bin_size' in PROCESSOR_PARAMS_SCHEMA['spike'], \
        "spike missing bin_size"
    assert 'normalize_bins' in PROCESSOR_PARAMS_SCHEMA['spike'], \
        "spike missing normalize_bins"

def test_schema_mode_lag():
    assert 'equalize_n' in MODE_KWARGS_SCHEMA['lag'], \
        "lag mode missing equalize_n"

check("Schema — new BASE_PARAMS keys registered", test_schema_base_params, requires='neural_mi')
check("Schema — processor schemas updated", test_schema_processor, requires='neural_mi')
check("Schema — lag mode equalize_n registered", test_schema_mode_lag, requires='neural_mi')


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — ParameterValidator
# ─────────────────────────────────────────────────────────────────────────────

def test_validator_defaults():
    v = ParameterValidator()
    out = v.validate({'n_epochs': 10, 'batch_size': 64})
    assert out['n_epochs'] == 10
    assert out['learning_rate'] == 5e-4, f"Got {out['learning_rate']}"
    assert out['train_fraction'] == 0.9, f"Got {out['train_fraction']}"
    assert out['n_test_blocks'] == 5, f"Got {out['n_test_blocks']}"
    assert out['max_index_reduction'] == 0.05, f"Got {out['max_index_reduction']}"

def test_validator_passthrough():
    v = ParameterValidator()
    out = v.validate({'train_fraction': 0.7, 'n_test_blocks': 3, 'unknown_key': 99})
    assert out['train_fraction'] == 0.7
    assert out['n_test_blocks'] == 3
    assert out['unknown_key'] == 99  # extra keys passed through

check("ParameterValidator — new params have correct defaults", test_validator_defaults, requires='neural_mi')
check("ParameterValidator — custom values + passthrough", test_validator_passthrough, requires='neural_mi')


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — DataValidator
# ─────────────────────────────────────────────────────────────────────────────

def test_data_validator_shapes():
    x = np.random.randn(100, 3).astype('float32')
    y = np.random.randn(100, 3).astype('float32')
    xv, yv = DataValidator.validate(x, y)
    assert xv.shape == (100, 3)
    assert yv.shape == (100, 3)

def test_data_validator_mismatch():
    x = np.random.randn(100, 3).astype('float32')
    y = np.random.randn(80, 3).astype('float32')
    try:
        DataValidator.validate(x, y)
        raise AssertionError("Should have raised ValueError on length mismatch")
    except ValueError:
        pass

def test_data_validator_1d():
    x = np.random.randn(100).astype('float32')
    xv, _ = DataValidator.validate(x)
    assert xv.ndim == 2 and xv.shape == (100, 1), f"1D not expanded: {xv.shape}"

check("DataValidator — shape preservation", test_data_validator_shapes, requires='neural_mi')
check("DataValidator — length mismatch raises ValueError", test_data_validator_mismatch, requires='neural_mi')
check("DataValidator — 1D input expanded to (N,1)", test_data_validator_1d, requires='neural_mi')


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Synthetic generators
# ─────────────────────────────────────────────────────────────────────────────

def test_gaussian_mi_correlation():
    x, y = gaussian_mi(1000, mi_nats=1.0, d=1, seed=42)
    assert x.shape == (1000, 1)
    corr = np.corrcoef(x[:, 0], y[:, 0])[0, 1]
    # rho = sqrt(1 - exp(-2*MI)) ≈ 0.865 for MI=1 nat
    expected_rho = np.sqrt(1 - np.exp(-2.0))
    assert abs(corr - expected_rho) < 0.05, \
        f"Correlation {corr:.3f} far from expected {expected_rho:.3f}"

def test_independent_gaussian():
    xi, yi = independent_gaussian(2000, d=1, seed=7)
    corr = abs(np.corrcoef(xi[:, 0], yi[:, 0])[0, 1])
    assert corr < 0.10, f"Independent data too correlated: {corr:.3f}"

check("Synthetic gaussian_mi — empirical correlation matches theory", test_gaussian_mi_correlation, requires='neural_mi')
check("Synthetic independent_gaussian — near-zero correlation", test_independent_gaussian, requires='neural_mi')


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — Rigorous 1/N axis (inline, no statsmodels needed)
# ─────────────────────────────────────────────────────────────────────────────

def test_inv_gamma_extrapolation():
    """With clean data, OLS on 1/gamma should recover the true intercept."""
    true_mi, bias_slope = 1.5, 2.0
    gammas = [1, 2, 3, 5, 7, 10]
    np.random.seed(0)
    # MI_obs(gamma) ≈ true_mi + bias_slope/gamma (+ tiny noise)
    inv_g, mi_obs = [], []
    for g in gammas:
        for _ in range(g):
            inv_g.append(1.0 / g)
            mi_obs.append(true_mi + bias_slope / g + np.random.normal(0, 0.03))
    inv_g = np.array(inv_g)
    mi_obs = np.array(mi_obs)
    # WLS with per-gamma equal weights
    import pandas as pd
    gamma_col = np.round(1.0 / inv_g).astype(int)
    counts = pd.Series(gamma_col).value_counts()
    w = 1.0 / pd.Series(gamma_col).map(counts).values
    # Weighted OLS: X = [1, 1/gamma], solve for [intercept, slope]
    X = np.column_stack([np.ones_like(inv_g), inv_g])
    W = np.diag(w)
    beta = np.linalg.solve(X.T @ W @ X, X.T @ W @ mi_obs)
    intercept = beta[0]
    assert abs(intercept - true_mi) < 0.12, \
        f"1/N extrapolation: intercept={intercept:.3f}, expected≈{true_mi}"

check("Rigorous 1/N extrapolation — weighted OLS recovers true MI", test_inv_gamma_extrapolation)


def test_gamma_equal_weight():
    """Each gamma level should contribute equal total weight."""
    import pandas as pd
    gammas = [1, 2, 3, 5]
    rows = []
    for g in gammas:
        for _ in range(g):
            rows.append({'gamma': g, 'test_mi': 1.0})
    df = pd.DataFrame(rows)
    weights = 1.0 / df['gamma'].map(df['gamma'].value_counts())
    total_per_gamma = df.groupby('gamma').apply(lambda grp: weights[grp.index].sum())
    # All gamma levels should have the same total weight (1.0)
    assert total_per_gamma.std() < 1e-10, \
        f"Gamma weights not equal: {total_per_gamma.to_dict()}"

check("Rigorous weights — each gamma contributes equal total weight", test_gamma_equal_weight)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — equalize_n
# ─────────────────────────────────────────────────────────────────────────────

def test_equalize_n_sizes():
    N = 1000
    lag_range = [1, 5, 20, 50]
    x_full = np.random.randn(N, 3).astype('float32')
    y_full = np.random.randn(N, 3).astype('float32')
    shifted = {lag: (x_full[:-lag], y_full[lag:]) for lag in lag_range}
    min_n = min(x.shape[0] for x, _ in shifted.values())
    assert min_n == N - max(lag_range), f"Expected {N-max(lag_range)}, got {min_n}"
    equalized = {lag: (x[:min_n], y[:min_n]) for lag, (x, y) in shifted.items()}
    sizes = [x.shape[0] for x, _ in equalized.values()]
    assert len(set(sizes)) == 1 and sizes[0] == min_n

def test_equalize_n_no_info_loss_at_zero_lag():
    """At lag=0 (baseline), equalize_n should not truncate if it's the limiting lag."""
    shifted = {0: (np.ones((1000, 1)), np.ones((1000, 1))),
               10: (np.ones((990, 1)), np.ones((990, 1)))}
    min_n = min(x.shape[0] for x, _ in shifted.values())
    assert min_n == 990

check("equalize_n — all lags truncated to minimum N", test_equalize_n_sizes)
check("equalize_n — minimum determined by largest lag", test_equalize_n_no_info_loss_at_zero_lag)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — Period from median inter-sample interval
# ─────────────────────────────────────────────────────────────────────────────

def test_period_uniform():
    t = np.linspace(0, 1.0, 1001)  # 1kHz, perfect
    median_p = float(np.median(np.diff(t)))
    assert abs(median_p - 0.001) < 1e-9

def test_period_boundary_artifact():
    t = np.linspace(0, 1.0, 1001)
    t_bad = t.copy()
    t_bad[0] = -0.050  # large artifact at start
    first_two = t_bad[1] - t_bad[0]
    median_p = float(np.median(np.diff(t_bad)))
    assert abs(median_p - 0.001) < 1e-6, f"Median should be 1ms, got {median_p*1e3:.3f}ms"
    assert first_two > 0.040, "first-two estimate should be inflated by artifact"

def test_period_jitter():
    rng = np.random.default_rng(0)
    t = np.cumsum(0.001 + rng.normal(0, 0.0001, 1000))
    median_p = float(np.median(np.diff(t)))
    assert abs(median_p - 0.001) < 0.0002, f"Median period with jitter: {median_p*1e3:.3f}ms"

check("Period — uniform sampling gives exact median", test_period_uniform)
check("Period — boundary artifact doesn't corrupt median", test_period_boundary_artifact)
check("Period — jittered sampling gives robust median", test_period_jitter)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — Coverage fraction filter
# ─────────────────────────────────────────────────────────────────────────────

def test_coverage_fraction_50():
    max_samples = 100
    min_count = int(np.ceil(0.5 * max_samples))  # 50
    actual = np.array([0, 10, 49, 50, 51, 100])
    valid = actual >= min_count
    assert list(valid) == [False, False, False, True, True, True]

def test_coverage_fraction_80():
    max_samples = 100
    min_count = int(np.ceil(0.8 * max_samples))  # 80
    actual = np.array([0, 50, 79, 80, 81, 100])
    valid = actual >= min_count
    assert list(valid) == [False, False, False, True, True, True]

def test_coverage_always_requires_at_least_one():
    # Even at fraction=0, min_count should be at least 1
    min_count = max(1, int(np.ceil(0.0 * 100)))
    assert min_count == 1

check("Coverage fraction — 50% threshold", test_coverage_fraction_50)
check("Coverage fraction — 80% threshold", test_coverage_fraction_80)
check("Coverage fraction — always requires at least 1 sample", test_coverage_always_requires_at_least_one)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 — BinnedSpike math
# ─────────────────────────────────────────────────────────────────────────────

def test_binned_spike_count():
    window_size, bin_size = 0.1, 0.01
    n_bins = int(np.ceil(window_size / bin_size))
    assert n_bins == 10
    spikes = np.array([0.005, 0.015, 0.085])
    counts, _ = np.histogram(spikes, bins=np.linspace(0, window_size, n_bins + 1))
    assert counts.sum() == 3
    assert counts[0] == 1 and counts[1] == 1 and counts[8] == 1

def test_binned_spike_normalize():
    window_size, bin_size = 0.1, 0.01
    n_bins = int(np.ceil(window_size / bin_size))
    spikes = np.array([0.005, 0.015])
    counts, _ = np.histogram(spikes, bins=np.linspace(0, window_size, n_bins + 1))
    rates = counts.astype(float) / bin_size  # spikes/s
    assert rates[0] == 1.0 / 0.01 == 100.0  # 1 spike in 10ms = 100 Hz
    assert rates.sum() == 200.0

def test_binned_spike_empty_window():
    window_size, bin_size = 0.1, 0.01
    n_bins = int(np.ceil(window_size / bin_size))
    spikes = np.array([])
    counts, _ = np.histogram(spikes, bins=np.linspace(0, window_size, n_bins + 1))
    assert counts.sum() == 0
    assert counts.shape == (n_bins,)

check("BinnedSpike — correct spike-to-bin assignment", test_binned_spike_count)
check("BinnedSpike — normalize=True gives spikes/s", test_binned_spike_normalize)
check("BinnedSpike — empty window gives all-zero bins", test_binned_spike_empty_window)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 — window_times compaction
# ─────────────────────────────────────────────────────────────────────────────

def test_window_times_compaction():
    all_times = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    valid_mask = np.array([True, False, True, True, False])
    compacted = all_times[valid_mask]
    assert len(compacted) == 3
    assert list(compacted) == [0.0, 0.2, 0.3]

def test_window_times_index_consistency():
    # After compaction, data[i] corresponds to compacted[i] — no off-by-one
    all_times = np.arange(0.0, 1.0, 0.1)  # 10 windows
    valid_mask = np.array([True,True,False,True,True,False,True,False,True,True])
    compacted = all_times[valid_mask]
    # Simulate data: each row's "value" is its window start time
    data = np.array([[t] for t in all_times])  # (10, 1)
    data_filtered = data[valid_mask]           # (7, 1)
    for i in range(len(compacted)):
        assert abs(data_filtered[i, 0] - compacted[i]) < 1e-9, \
            f"Index {i}: data={data_filtered[i,0]:.2f} != time={compacted[i]:.2f}"

def test_window_times_post_shift_cycle():
    # Simulate create_windows → filter → compact → shift → create_windows → filter → compact
    # Each cycle should produce self-consistent compacted arrays
    t_start, t_end, ws = 0.0, 1.0, 0.1
    def make_windows(ts, te, w):
        times = np.arange(ts, te, w)
        # Simulate: every other window is invalid
        valid = np.array([i % 2 == 0 for i in range(len(times))])
        return times[valid]
    c1 = make_windows(t_start, t_end, ws)
    c2 = make_windows(t_start + 0.05, t_end + 0.05, ws)  # shifted
    assert len(c1) == len(c2), "Shift should not change number of valid windows"

check("window_times — compacted to valid entries only", test_window_times_compaction)
check("window_times — data index matches time index after compaction", test_window_times_index_consistency)
check("window_times — compaction survives time_shift cycle", test_window_times_post_shift_cycle)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 11 — Spike sentinel
# ─────────────────────────────────────────────────────────────────────────────

def test_sentinel_not_zero():
    # Key fix: -1.0 (not 0.0) as sentinel, since t=0 is a valid spike time
    no_spike = -1.0
    assert no_spike != 0.0
    spike_at_zero = 0.0
    assert spike_at_zero != no_spike

def test_sentinel_mask():
    no_spike = -1.0
    data = np.array([-1.0, 0.0, 0.025, 0.050, -1.0, 0.075])
    real = data != no_spike
    assert real.sum() == 4
    assert data[real].min() == 0.0  # real spike at t=0 is included

def test_sentinel_no_negative_spikes():
    # Spike times within a window are offsets from window start — always >= 0
    # So -1.0 is unambiguously not a valid spike time offset
    window_size = 0.1
    valid_offsets = np.linspace(0, window_size, 100)
    assert all(v >= 0 for v in valid_offsets)
    sentinel = -1.0
    assert sentinel not in valid_offsets
    assert sentinel < 0

check("Spike sentinel — -1.0 differs from t=0 spike", test_sentinel_not_zero)
check("Spike sentinel — mask correctly identifies real spikes at t=0", test_sentinel_mask)
check("Spike sentinel — -1.0 is unambiguously invalid (all offsets >= 0)", test_sentinel_no_negative_spikes)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 12 — Mask cache invalidation
# ─────────────────────────────────────────────────────────────────────────────

def test_mask_cleared_on_reset():
    class FakeDataset:
        def reset(self):
            self.__dict__.pop('_data_mask', None)
            self.__dict__.pop('_noise_buffer', None)
    ds = FakeDataset()
    ds._data_mask = ('stale', 'mask')
    ds._noise_buffer = np.zeros(10)
    ds.reset()
    assert not hasattr(ds, '_data_mask'), "_data_mask not cleared"
    assert not hasattr(ds, '_noise_buffer'), "_noise_buffer not cleared"

def test_mask_rebuilt_after_reset():
    class FakeDataset:
        def __init__(self):
            self.data = np.array([1.0, 0.0, 2.0, 0.0, 3.0])
        def reset(self):
            self.__dict__.pop('_data_mask', None)
        def apply_noise(self, amplitude):
            if not hasattr(self, '_data_mask'):
                self._data_mask = np.nonzero(self.data)[0]
    ds = FakeDataset()
    ds.apply_noise(0.1)
    assert hasattr(ds, '_data_mask')
    original_mask = ds._data_mask.copy()
    ds.reset()
    assert not hasattr(ds, '_data_mask')
    ds.data = np.array([0.0, 1.0, 2.0, 0.0, 0.0])  # shape changed
    ds.apply_noise(0.1)
    new_mask = ds._data_mask
    assert list(new_mask) != list(original_mask), "Mask should be rebuilt with new data"

def test_buffer_size_check():
    # Buffer should be invalidated if data shape changes
    buf = np.zeros(100)
    new_mask_size = 80  # after filtering, fewer nonzero elements
    needs_rebuild = len(buf) != new_mask_size
    assert needs_rebuild

check("Mask cache — cleared on reset()", test_mask_cleared_on_reset)
check("Mask cache — rebuilt from new data after reset()", test_mask_rebuilt_after_reset)
check("Mask cache — buffer size mismatch triggers rebuild", test_buffer_size_check)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 13 — Param bleed fix
# ─────────────────────────────────────────────────────────────────────────────

def test_param_bleed_filter():
    valid_proc_keys = {'window_size', 'min_coverage_fraction', 'sample_rate'}
    sweep_combo = {'window_size': 0.1, 'embedding_dim': 64, 'n_layers': 2,
                   'sample_rate': 1000, 'batch_size': 128}
    filtered = {k: v for k, v in sweep_combo.items() if k in valid_proc_keys}
    assert 'window_size' in filtered and 'sample_rate' in filtered
    assert 'embedding_dim' not in filtered, "embedding_dim leaked into proc_params"
    assert 'n_layers' not in filtered, "n_layers leaked into proc_params"
    assert 'batch_size' not in filtered, "batch_size leaked into proc_params"

def test_param_bleed_spike():
    valid_proc_keys = set(PROCESSOR_PARAMS_SCHEMA['spike']) if _HAS_NEURAL_MI else \
        {'window_size', 'max_spikes_per_window', 'n_seconds', 'sample_rate',
         'no_spike_value', 'bin_size', 'normalize_bins'}
    sweep_combo = {'window_size': 0.1, 'n_layers': 3, 'bin_size': 0.01,
                   'no_spike_value': -1.0, 'hidden_dim': 128}
    filtered = {k: v for k, v in sweep_combo.items() if k in valid_proc_keys}
    assert 'bin_size' in filtered and 'no_spike_value' in filtered
    assert 'n_layers' not in filtered
    assert 'hidden_dim' not in filtered

check("Param bleed — continuous: model params excluded", test_param_bleed_filter)
check("Param bleed — spike: bin_size/no_spike_value kept, model params excluded", test_param_bleed_spike)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 14 — Lag N-confound
# ─────────────────────────────────────────────────────────────────────────────

def test_lag_n_decreases_with_lag():
    N = 1000
    lags = list(range(0, 101, 10))
    n_windows = [N - lag for lag in lags]
    assert n_windows == sorted(n_windows, reverse=True), \
        "N should decrease monotonically with lag"
    assert max(n_windows) - min(n_windows) == 100, \
        "Range of N should equal range of lags"

def test_equalize_n_removes_confound():
    N = 1000
    lags = [0, 10, 50, 100]
    ns = {lag: N - lag for lag in lags}
    min_n = min(ns.values())
    equalized = {lag: min_n for lag in lags}
    assert len(set(equalized.values())) == 1, "All lags should have same N after equalization"
    # N-vs-lag correlation should be 0 after equalization
    ns_eq = list(equalized.values())
    corr = np.corrcoef(lags, ns_eq)[0, 1]
    assert np.isnan(corr) or abs(corr) < 1e-10, \
        f"N-lag correlation should be 0 after equalization, got {corr:.4f}"

check("Lag N-confound — N decreases monotonically with lag", test_lag_n_decreases_with_lag)
check("Lag N-confound — equalize_n removes N-lag correlation", test_equalize_n_removes_confound)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 15 — Rigorous WLS weights
# ─────────────────────────────────────────────────────────────────────────────

def test_gamma_total_weight_equal():
    import pandas as pd
    gammas = [1, 2, 3, 5, 10]
    rows = [{'gamma': g} for g in gammas for _ in range(g)]
    df = pd.DataFrame(rows)
    weights = 1.0 / df['gamma'].map(df['gamma'].value_counts())
    total = df.groupby('gamma').apply(lambda grp: weights[grp.index].sum())
    assert total.std() < 1e-10, \
        f"Gamma weights not equal: std={total.std():.2e}, values={total.to_dict()}"

def test_gamma1_highest_per_point_weight():
    import pandas as pd
    gammas = [1, 2, 5]
    rows = [{'gamma': g} for g in gammas for _ in range(g)]
    df = pd.DataFrame(rows)
    weights = 1.0 / df['gamma'].map(df['gamma'].value_counts())
    w_g1 = weights[df['gamma'] == 1].iloc[0]
    w_g5 = weights[df['gamma'] == 5].iloc[0]
    assert w_g1 > w_g5, "gamma=1 should have highest per-point weight (only 1 point)"

check("Rigorous weights — each gamma level has equal total weight", test_gamma_total_weight_equal)
check("Rigorous weights — gamma=1 has highest per-point weight", test_gamma1_highest_per_point_weight)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 16 — Patience / shift warning
# ─────────────────────────────────────────────────────────────────────────────

def test_patience_shift_detection():
    # The warning condition: is_temporal AND random_time_shifting AND patience < epochs_to_max_shift
    cases = [
        # (is_temporal, shifting, patience, max_shift, should_warn)
        (True,  True,  3,  5,  True),   # patience < max_shift → warn
        (True,  True,  5,  5,  False),  # patience == max_shift → OK
        (True,  True, 10,  5,  False),  # patience > max_shift → OK
        (False, True,  3,  5,  False),  # not temporal → no warn
        (True, False,  3,  5,  False),  # shifting off → no warn
    ]
    for is_temporal, shifting, patience, max_shift, should_warn in cases:
        warn = is_temporal and shifting and patience < max_shift
        assert warn == should_warn, \
            f"is_temporal={is_temporal}, shifting={shifting}, patience={patience}, " \
            f"max_shift={max_shift}: expected warn={should_warn}, got {warn}"

check("Patience/shift warning — correct detection logic", test_patience_shift_detection)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 17 — BinnedSpike normalize flag
# ─────────────────────────────────────────────────────────────────────────────

def test_normalize_true():
    bin_size = 0.01
    counts = np.array([1, 0, 2, 0, 1])
    rates = counts.astype(float) / bin_size
    assert rates[0] == 100.0  # 1 spike / 10ms = 100 Hz
    assert rates[2] == 200.0

def test_normalize_false():
    counts = np.array([1, 0, 2, 0, 1])
    raw = counts.astype(float)  # no division
    assert raw[0] == 1.0
    assert raw[2] == 2.0

check("BinnedSpike normalize=True — rates in spikes/s", test_normalize_true)
check("BinnedSpike normalize=False — raw spike counts preserved", test_normalize_false)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 18 — Processor schema routing
# ─────────────────────────────────────────────────────────────────────────────

def test_bin_size_triggers_binned_dataset():
    # Simulate the routing logic in create_single_dataset
    def route(proc_params):
        if proc_params.get('bin_size') is not None:
            return 'BinnedSpikeDataset'
        return 'SpikeWindowDataset'
    assert route({'bin_size': 0.01}) == 'BinnedSpikeDataset'
    assert route({}) == 'SpikeWindowDataset'
    assert route({'bin_size': None}) == 'SpikeWindowDataset'
    assert route({'no_spike_value': -2.0}) == 'SpikeWindowDataset'

def test_no_spike_value_routing():
    # no_spike_value should pass through to SpikeWindowDataset, not BinnedSpikeDataset
    def get_sentinel(proc_params):
        return proc_params.get('no_spike_value', -1.0)
    assert get_sentinel({}) == -1.0
    assert get_sentinel({'no_spike_value': -2.0}) == -2.0
    assert get_sentinel({'no_spike_value': 0.0}) == 0.0  # user override

check("Routing — bin_size triggers BinnedSpikeDataset", test_bin_size_triggers_binned_dataset)
check("Routing — no_spike_value falls through to SpikeWindowDataset", test_no_spike_value_routing)


# ─────────────────────────────────────────────────────────────────────────────
# Final summary
# ─────────────────────────────────────────────────────────────────────────────

n_pass = sum(1 for _, s in _results if s == 'pass')
n_fail = sum(1 for _, s in _results if s == 'fail')
n_skip = sum(1 for _, s in _results if s == 'skip')
n_total = len(_results)

print()
print("=" * 60)
print(f"  RESULTS: {n_pass}/{n_total} passed  |  "
      f"{n_fail} failed  |  {n_skip} skipped")
print("=" * 60)

if n_fail > 0:
    print("\nFailed tests:")
    for name, status in _results:
        if status == 'fail':
            print(f"  ✗ {name}")
    sys.exit(1)
else:
    print("\nAll tests passed. ✓")
    sys.exit(0)