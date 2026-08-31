# tests/test_generators.py
"""Every generator reports the quantity an estimate should be checked against.

These tests assert the reported value is right, not merely that data comes out
with the expected shape: a generator whose stated truth is wrong is worse than
no generator, since it silently invalidates whatever is validated against it.
"""
import numpy as np
import pytest
import torch

from neural_mi.generators import (
    SharedLatentGaussian,
    generate_categorical_pair,
    generate_correlated_gaussians,
    generate_lagged_pair,
    generate_nonlinear_from_latent,
    generate_spike_pair,
    generate_windowed_multichannel,
    generate_windowed_oscillatory,
    generate_xor_pair,
    mi_to_rho,
    pmf_mi_bits,
    symmetric_joint_pmf,
)
from neural_mi.generators import oracle


class TestPmfHelpers:
    def test_symmetric_pmf_is_a_distribution_with_uniform_marginals(self):
        p = symmetric_joint_pmf(4, 0.85)
        assert p.sum() == pytest.approx(1.0)
        assert p.sum(0) == pytest.approx(np.full(4, 0.25))
        assert p.sum(1) == pytest.approx(np.full(4, 0.25))

    def test_independence_gives_zero_mi(self):
        """rho = 1/n_levels makes the two independent, so the MI is exactly 0."""
        for k in (2, 3, 5):
            assert pmf_mi_bits(symmetric_joint_pmf(k, 1.0 / k)) == pytest.approx(0.0, abs=1e-12)

    def test_mi_rises_monotonically_towards_log2_levels(self):
        vals = [pmf_mi_bits(symmetric_joint_pmf(4, r)) for r in (0.3, 0.5, 0.7, 0.9, 0.99)]
        assert all(b > a for a, b in zip(vals, vals[1:]))
        assert vals[-1] < np.log2(4)


class TestSpikePair:
    @pytest.mark.parametrize('coding', ['count', 'timing'])
    def test_shapes_and_reported_mi(self, coding):
        x, y, mi = generate_spike_pair(n_windows=200, n_neurons=5, coding=coding, seed=0)
        assert len(x) == len(y) == 5
        for spikes in x + y:
            assert isinstance(spikes, np.ndarray)
            assert np.all(np.diff(spikes) >= 0), "spike times must be sorted"
        assert mi == pytest.approx(pmf_mi_bits(symmetric_joint_pmf(4, 0.85)))

    def test_reported_mi_matches_the_latent_pmf(self):
        for n_levels, rho in ((2, 0.9), (4, 0.85), (6, 0.7)):
            _, _, mi = generate_spike_pair(n_windows=50, n_levels=n_levels, rho=rho, seed=0)
            assert mi == pytest.approx(pmf_mi_bits(symmetric_joint_pmf(n_levels, rho)))

    def test_spikes_stay_inside_their_windows(self):
        n_windows, w = 40, 0.5
        x, _, _ = generate_spike_pair(n_windows=n_windows, window_size=w, seed=0)
        for spikes in x:
            assert spikes.min() >= 0.0
            assert spikes.max() <= n_windows * w

    def test_count_coding_puts_the_information_in_the_rate(self):
        """Under 'count' the per-window spike count is the latent, so counts vary."""
        x, _, _ = generate_spike_pair(n_windows=300, n_neurons=1, coding='count',
                                      window_size=1.0, seed=0)
        counts = np.bincount(x[0].astype(int), minlength=300)[:300]
        assert len(set(counts.tolist())) > 1

    def test_timing_coding_leaves_the_count_uninformative(self):
        """Under 'timing' the count is drawn independently for each population."""
        x, y, _ = generate_spike_pair(n_windows=400, n_neurons=1, coding='timing',
                                      window_size=1.0, seed=0)
        cx = np.bincount(x[0].astype(int), minlength=400)[:400]
        cy = np.bincount(y[0].astype(int), minlength=400)[:400]
        # Independent draws: the two count series should be essentially uncorrelated.
        assert abs(np.corrcoef(cx, cy)[0, 1]) < 0.2

    def test_lag_delays_the_second_population(self):
        """Y's content starts `lag_windows` later than X's.

        Checked on a train other than the first: neuron 0 of each population
        carries an extra spike at t=0 that pins the window grid, so its minimum
        is 0 regardless of the lag.
        """
        x, y, _ = generate_spike_pair(n_windows=100, n_neurons=3, lag_windows=5,
                                      window_size=1.0, seed=0)
        assert y[1].min() >= 5.0, "Y's content should begin after the lag"
        assert x[1].min() < 1.0, "X's content should begin in its first window"

    def test_window_grid_is_pinned_to_zero(self):
        """Spike windowing takes its origin from the earliest spike, so the
        generator places one at t=0. Without it the analysis grid lands
        mid-window and every window straddles two independent latent draws."""
        from neural_mi.data.handler import create_dataset
        n_windows, w = 50, 1.0
        x, y, _ = generate_spike_pair(n_windows=n_windows, window_size=w,
                                      n_neurons=2, coding='timing', seed=0)
        assert min(s.min() for s in x) == 0.0
        d = create_dataset(x_data=x, y_data=y,
                           processor_type_x='spike', processor_type_y='spike',
                           processor_params_x={'window_size': w},
                           processor_params_y={'window_size': w})
        starts = np.asarray(d.x_dataset.window_manager.window_times)
        assert np.allclose(starts % w, 0.0, atol=1e-9), "grid must align to the window size"
        assert len(d.x_dataset) == n_windows, (
            f"expected {n_windows} windows, got {len(d.x_dataset)}; a misaligned "
            f"grid silently drops windows and changes the estimand")

    @pytest.mark.parametrize('kwargs,match', [
        (dict(coding='rate'), "coding must be"),
        (dict(lag_windows=-1), "non-negative"),
        (dict(lag_windows=100, n_windows=100), "smaller than n_windows"),
        (dict(n_levels=1), "at least 2"),
        (dict(rho=1.5), r"rho must lie"),
    ])
    def test_rejects_invalid_arguments(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            generate_spike_pair(**{'n_windows': 100, **kwargs})


class TestXorPair:
    def test_mi_approaches_one_bit_as_noise_vanishes(self):
        _, _, mi = generate_xor_pair(200, noise=0.01, use_torch=False, seed=0)
        assert mi == pytest.approx(1.0, abs=1e-3)

    def test_mi_falls_as_noise_grows(self):
        vals = [generate_xor_pair(200, noise=n, use_torch=False, seed=0)[2]
                for n in (0.1, 0.3, 0.6, 1.0)]
        assert all(b < a for a, b in zip(vals, vals[1:]))
        assert vals[-1] > 0.0

    def test_y_is_the_xor_of_the_two_bits(self):
        x, y, _ = generate_xor_pair(300, noise=1e-6, use_torch=False, seed=0)
        assert np.allclose(np.round(y[:, 0]), np.bitwise_xor(x[:, 0], x[:, 1]))

    def test_neither_bit_alone_predicts_y(self):
        """The synergy: each single bit is independent of Y."""
        x, y, _ = generate_xor_pair(20000, noise=1e-6, use_torch=False, seed=0)
        for col in (0, 1):
            means = [y[x[:, col] == b, 0].mean() for b in (0, 1)]
            assert means[0] == pytest.approx(means[1], abs=0.05)

    def test_zero_noise_is_rejected_with_the_limit_named(self):
        with pytest.raises(ValueError, match="limiting value is 1 bit"):
            generate_xor_pair(10, noise=0.0)

    def test_torch_and_numpy_variants_agree_in_shape(self):
        xt, yt, _ = generate_xor_pair(50, seed=0)
        xn, yn, _ = generate_xor_pair(50, use_torch=False, seed=0)
        assert isinstance(xt, torch.Tensor) and isinstance(xn, np.ndarray)
        assert tuple(xt.shape) == xn.shape == (50, 2)
        assert tuple(yt.shape) == yn.shape == (50, 1)


class TestCategoricalPair:
    def test_reported_mi_matches_the_channel_pmf(self):
        for k, agree in ((3, 0.9), (4, 0.75), (2, 0.95)):
            _, _, mi = generate_categorical_pair(200, n_categories=k, agreement=agree,
                                                 use_torch=False, seed=0)
            joint = np.full((k, k), (1.0 - agree) / (k * k))
            np.fill_diagonal(joint, joint[0, 0] + agree / k)
            assert mi == pytest.approx(pmf_mi_bits(joint))

    def test_perfect_agreement_gives_log2_categories(self):
        _, _, mi = generate_categorical_pair(200, n_categories=4, agreement=1.0,
                                             use_torch=False, seed=0)
        assert mi == pytest.approx(np.log2(4))

    def test_empirical_agreement_matches_the_parameter(self):
        x, y, _ = generate_categorical_pair(20000, n_categories=3, agreement=0.9,
                                            use_torch=False, seed=0)
        # y copies x with probability `agreement`, else lands uniformly, so it
        # matches with probability agreement + (1 - agreement)/K.
        expected = 0.9 + 0.1 / 3
        assert (x == y).mean() == pytest.approx(expected, abs=0.02)

    def test_shape_dtype_and_alphabet(self):
        x, y, _ = generate_categorical_pair(100, n_channels=2, n_categories=3,
                                            use_torch=False, seed=0)
        assert x.shape == y.shape == (100, 2)
        assert x.dtype == int and x.max() < 3

    def test_rejects_invalid_agreement(self):
        with pytest.raises(ValueError, match="agreement must lie"):
            generate_categorical_pair(10, agreement=0.0)


class TestLaggedPair:
    @pytest.mark.parametrize('lag', [0, 10, 25])
    def test_cross_correlation_peaks_at_the_requested_lag(self, lag):
        x, y, _ = generate_lagged_pair(n_samples=3000, lag=lag, seed=0)
        n = len(x)
        span = range(-40, 41)
        corr = [abs(np.corrcoef(x[max(0, -L):n - max(0, L), 0],
                                y[max(0, L):n - max(0, -L), 0])[0, 1]) for L in span]
        assert list(span)[int(np.argmax(corr))] == lag

    def test_reported_mi_is_the_value_at_the_peak(self):
        x, y, mi = generate_lagged_pair(n_samples=2000, lag=15, seed=0)
        ref = SharedLatentGaussian(dims={'x': 1, 'y': 1}, d=1, phi=0.95,
                                   noise=0.5, coupling=3.0, seed=0)
        assert mi == pytest.approx(ref.exact(A=[('x', 0)], B=[('y', 0)]))

    def test_shapes_and_dtype(self):
        x, y, _ = generate_lagged_pair(n_samples=500, lag=10, dim=3, seed=0)
        assert x.shape == y.shape == (500, 3)
        assert x.dtype == np.float32

    def test_rejects_negative_lag(self):
        with pytest.raises(ValueError, match="non-negative"):
            generate_lagged_pair(lag=-1)


class TestGaussianGenerators:
    def test_correlated_gaussians_shape_and_type(self):
        x, y = generate_correlated_gaussians(n_samples=100, dim=5, mi=2.0, use_torch=True)
        assert x.shape == (100, 5) and y.shape == (100, 5)
        assert isinstance(x, torch.Tensor)

    def test_correlated_gaussians_numpy(self):
        x, y = generate_correlated_gaussians(n_samples=100, dim=5, mi=2.0, use_torch=False)
        assert isinstance(x, np.ndarray) and x.shape == (100, 5)

    def test_mi_to_rho_round_trips(self):
        """rho is chosen so the pair carries exactly the requested MI."""
        for dim, mi in ((1, 1.0), (5, 2.0), (8, 0.5)):
            rho = mi_to_rho(dim, mi)
            recovered = -0.5 * dim * np.log2(1 - rho ** 2)
            assert recovered == pytest.approx(mi)

    def test_nonlinear_from_latent_shapes(self):
        x, y = generate_nonlinear_from_latent(100, 4, 50, 2.0)
        assert x.shape == (100, 50) and y.shape == (100, 50)
        xn, yn = generate_nonlinear_from_latent(100, 4, 50, 2.0, use_torch=False)
        assert isinstance(xn, np.ndarray)


class TestWindowedGenerators:
    """Windowed generators that report their own analytically known MI."""

    def test_windowed_oscillatory_shape_and_dtype(self):
        X, Y, true_mi = generate_windowed_oscillatory(
            n_windows=20, n_channels=3, window_size=64, latent_mi=1.0)
        assert X.shape == (20, 3, 64) and Y.shape == (20, 3, 64)
        assert X.dtype == np.float32 and Y.dtype == np.float32
        assert isinstance(true_mi, float) and true_mi > 0

    def test_windowed_oscillatory_true_mi_scales_linearly_with_n_channels(self):
        """true_mi is a deterministic function of the parameters, so doubling
        n_channels must exactly double it."""
        _, _, mi_1ch = generate_windowed_oscillatory(n_windows=5, n_channels=1,
                                                     latent_mi=1.0, snr=2.0)
        _, _, mi_2ch = generate_windowed_oscillatory(n_windows=5, n_channels=2,
                                                     latent_mi=1.0, snr=2.0)
        assert mi_2ch == pytest.approx(2 * mi_1ch)

    def test_windowed_multichannel_shape_and_dtype(self):
        X, Y, true_mi = generate_windowed_multichannel(
            n_windows=15, n_channels=4, window_size=50, latent_mi=0.5)
        assert X.shape == (15, 4, 50) and Y.shape == (15, 4, 50)
        assert isinstance(true_mi, float) and true_mi > 0

    def test_windowed_multichannel_matches_oscillatory_for_one_channel(self):
        common = dict(window_size=200, sample_rate=500.0, latent_mi=0.5, snr=3.0)
        _, _, multichannel_mi = generate_windowed_multichannel(
            n_windows=10, n_channels=1, f_min_hz=4.0, f_max_hz=4.0, **common)
        _, _, oscillatory_mi = generate_windowed_oscillatory(
            n_windows=10, n_channels=1, f_carrier_hz=4.0, **common)
        assert multichannel_mi == pytest.approx(oscillatory_mi, rel=1e-5)
