# tests/test_known_answers.py
"""Known answers that the correctness fixes were found against, kept so that each error would fail a test.

Every case has an exact value: a Gaussian model's exact conditional information, the rate of a smooth
process from its spectra, or a conditional MI that is zero because W is a copy of X on every row the
quantity reads. Each runs the default model and training. Each tolerance was set from repeated runs and
fails on the error its fix removed (ROADMAP C15 and C17).
"""
import warnings

import numpy as np
import pytest
from scipy.signal import lfilter

import neural_mi as nmi
from neural_mi.generators.oracle import LatentGaussian

LN2 = np.log(2.0)


def _quiet(call, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return call(*args, show_progress=False, **kwargs)


# --------------------------------------------------------------------------
# A smooth, self-predictable process: a shared damped oscillation under larger
# private ones, finely sampled, as in field potentials. Each part is an AR(2)
# oscillator, and the rate follows from the spectra.
# --------------------------------------------------------------------------
class _SmoothPair:
    SHARED, PRIVATE, SIGMA = (0.98, 0.02), (0.99, 0.005), 0.05      # (damping, cycles per sample)

    @staticmethod
    def _ar2(damping, frequency):
        return 2 * damping * np.cos(2 * np.pi * frequency), -damping ** 2

    def _spectrum(self, part, w):
        a1, a2 = self._ar2(*part)
        return 1.0 / np.abs(1 - a1 * np.exp(-1j * w) - a2 * np.exp(-2j * w)) ** 2

    def rate_bits(self):
        """-1/2 <log(1 - |coherence|^2)> over frequency, the information rate of the two series."""
        w = np.linspace(-np.pi, np.pi, 1 << 16, endpoint=False)
        shared, private = self._spectrum(self.SHARED, w), self._spectrum(self.PRIVATE, w)
        each = shared + private + self.SIGMA ** 2
        return float(-0.5 * np.mean(np.log1p(-(shared / each) ** 2)) / LN2)

    def sample(self, n, seed, burn=5000):
        rng = np.random.default_rng(seed)

        def oscillation(part):
            a1, a2 = self._ar2(*part)
            return lfilter([1.0], [1, -a1, -a2], rng.standard_normal(n + burn))[burn:]

        shared = oscillation(self.SHARED)
        x = shared + oscillation(self.PRIVATE) + self.SIGMA * rng.standard_normal(n)
        y = shared + oscillation(self.PRIVATE) + self.SIGMA * rng.standard_normal(n)
        return x[:, None].astype(np.float32), y[:, None].astype(np.float32)


@pytest.mark.slow
def test_the_rate_of_a_smooth_process_after_whitening():
    """The conditioning term of mi_rate on smooth data is several nats and the rate a small residual of it,
    which the networks read poorly: without whitening one run read high and others 0. After `time='ar'`
    the default call reads within 30% of the exact rate. The test this replaces ran an easy process with a
    GRU and accepted +/- 0.5 bits on a 0.85-bit target."""
    pair = _SmoothPair()
    x, y = pair.sample(20000, seed=0)
    exact = pair.rate_bits()
    r = _quiet(nmi.mi_rate, x, y, h=5, half_width=20, preprocessing=nmi.Preprocessing(time='ar'), seed=0)
    assert abs(r.mi_estimate - exact) < 0.3 * exact, f"{r.mi_estimate:.4f} bits against an exact {exact:.4f}"


def test_an_estimate_does_not_depend_on_the_units_with_channel_preprocessing():
    """Without a transform the same data read 2 bits at unit scale and about 0 at a thousand times it or a
    thousandth of it."""
    x, y = nmi.generators.generate_correlated_gaussians(n_samples=2000, dim=3, mi=2.0, seed=0)
    x, y = np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32)
    values = [_quiet(nmi.run, x * scale, y * scale, preprocessing=nmi.Preprocessing(channel='zscore'),
                     split=nmi.Split(mode='random'), seed=0).mi_estimate
              for scale in (1.0, 1e3, 1e-3)]
    assert max(values) - min(values) < 0.01 and abs(values[0] - 2.0) < 0.3


# --------------------------------------------------------------------------
# One row plan per quantity (C17)
# --------------------------------------------------------------------------
class TestAWThatStartsLate:
    """W is X's own record from 40 s on. On the rows every term reads, from 40 s, W is X, so
    I(X; Y | W) = 0 however much X says about Y. Each term used to build its rows from its own streams, and
    the joint term read rows from 0 s, where W was silent and X carried what W did not: the default call
    read 0.25 bits."""

    def test_spike_trains_read_no_conditional_information(self):
        x, y, _ = nmi.generators.generate_spike_pair(n_windows=400, window_size=1.0, coding='count', seed=0)
        w = [s[s >= 40.0] for s in x]
        p = {'window_size': 1.0}
        r = _quiet(nmi.run, x, y, mode='conditional', conditional=nmi.Conditional(w_data=w), seed=0,
                   processing=nmi.Processing(x='spike', x_params=p, y='spike', y_params=p,
                                             w='spike', w_params=p))
        components = r.dataframe.iloc[0]
        assert components['mi_w_y_mean'] > 0.5                         # X, and so W, says much about Y
        assert abs(r.mi_estimate) < 0.15

    def test_a_clock_that_starts_late_reads_none_with_ksg(self):
        """The same grid route on continuous streams, where KSG reads exactly: X and W are the same rows, so
        the joint term's conditional estimate given W has nothing to add. (KSG cannot read this case on
        spike windows, whose empty slots tie, and warns there.)"""
        g = LatentGaussian.shared(dims={'x': 2, 'y': 2}, d=1, phi=0.9, coupling=1.0, seed=0)
        data = g.sample(12000, seed=0)
        t = np.arange(12000) / 100.0
        late = t >= 40.0
        p = {'window_size': 0.05, 'sample_rate': 100.0}
        r = _quiet(nmi.run, data['x'], data['y'], mode='conditional', estimator='ksg', seed=0,
                   conditional=nmi.Conditional(w_data=data['x'][late]),
                   processing=nmi.Processing(x='continuous', x_params=p, y='continuous', y_params=p,
                                             w='continuous', w_params=p, x_time=t, y_time=t, w_time=t[late]))
        components = r.dataframe.iloc[0]
        assert components['mi_w_y_mean'] > 0.1
        assert abs(components['mi_xw_y_mean'] - components['mi_w_y_mean']) < 0.01


class TestAWWindowUnlikeX:
    """I(X[-4..0]; Y[0] | W[-9..-1]) on a Gaussian with an exact answer: W's window is not X's."""
    _model = LatentGaussian.shared(dims={'x': 2, 'y': 2, 'w': 1}, d=2, phi=0.7, coupling=2.0, seed=0)
    _spec = {'A': [('x', s) for s in range(-4, 1)], 'B': [('y', 0)], 'C': [('w', s) for s in range(-9, 0)]}

    def test_run_refuses_a_processor_window_unlike_xs(self):
        """One grid has one window and one step, so run() cannot place W's window of its own."""
        data = self._model.sample(2000, seed=0)
        with pytest.raises(ValueError, match="Stream 'w' asks for window_size=.* share one grid"):
            nmi.run(data['x'], data['y'], mode='conditional', conditional=nmi.Conditional(w_data=data['w']),
                    processing=nmi.Processing(x='continuous', x_params={'window_size': 5},
                                              y='continuous', y_params={'window_size': 5},
                                              w='continuous', w_params={'window_size': 9}),
                    show_progress=False)

    @pytest.mark.slow
    def test_offset_quantity_places_it_and_reads_the_exact_value(self):
        exact = self._model.exact(self._spec['A'], self._spec['B'], self._spec['C'])
        r = _quiet(nmi.offset_quantity, self._model.sample(8000, seed=0), self._spec, seed=0)
        assert abs(r.mi_estimate - exact) < 0.2 * exact, f"{r.mi_estimate:.4f} bits against an exact {exact:.4f}"


def test_a_step_longer_than_the_window_reads_the_exact_block_mi():
    """Windows of 5 samples every 8, shifted each epoch. X and Y used to move by different amounts on some
    epochs when the step was longer than the window, so the networks trained on pairs that did not belong
    together. The process is stationary, so the exact block MI holds at every placement."""
    g = LatentGaussian.shared(dims={'x': 2, 'y': 2}, d=1, phi=0.9, coupling=1.0, seed=0)
    data = g.sample(3000 * 8, seed=0)
    p = {'window_size': 5, 'step_size': 8}
    r = _quiet(nmi.run, data['x'], data['y'], seed=0,
               processing=nmi.Processing(x='continuous', x_params=p, y='continuous', y_params=p))
    exact = g.block_mi(5)
    assert abs(r.mi_estimate - exact) < 0.25 * exact, f"{r.mi_estimate:.4f} bits against an exact {exact:.4f}"


# --------------------------------------------------------------------------
# A rigorous ladder on a starved sample (C21)
# --------------------------------------------------------------------------
@pytest.mark.slow
def test_a_ladder_of_chunks_too_small_to_resolve_is_not_reliable():
    """Tutorial 02's data at 400 samples, exact 4 bits. From gamma=5 on a chunk holds 8 or fewer held-out
    rows, where no curve can resolve: its top, at most log(n) nats on n rows, stays within 3 standard errors,
    sqrt(2 log(n) / n), of zero. Those rungs read noise, the line through every rung extrapolated far below
    the exact value, and is_reliable was True. Without them too few values of gamma are left."""
    x, y = nmi.generators.generate_nonlinear_from_latent(n_samples=400, latent_dim=10, observed_dim=500,
                                                         mi=4.0, seed=0)
    r = _quiet(nmi.run, x, y, mode='rigorous', split=nmi.Split(mode='random'), seed=0)
    fit = r.runs.iloc[0]
    assert not fit['is_reliable'] and not fit['enough_gamma_points']
    assert fit['zero_rungs'] + fit['unresolved_rungs'] >= sum(range(5, 11))
