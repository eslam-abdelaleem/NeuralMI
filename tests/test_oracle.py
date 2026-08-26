# tests/test_oracle.py
"""Tests for the exact-value oracle and the offset-spec array builder.

Two things are locked in here. First, that ``SharedLatentGaussian`` reproduces
the identities the temporal taxonomy rests on, since a tutorial that claims
Massey's conservation law holds needs both sides to be exact. Second, that the
general :func:`build_offset_arrays` produces byte-identical arrays to every
hand-written builder in the library, so the offset spec printed beside a named
quantity really is the computation that quantity performs.
"""
import numpy as np
import pytest
import torch

from neural_mi.generators import SharedLatentGaussian, generate_shared_latent_gaussian
from neural_mi.utils import build_offset_arrays
from neural_mi.analysis.offsets import build_past_future, build_cross_offset
from neural_mi.quantities import (
    _build_mi_rate_arrays, _build_inst_exchange_arrays, _build_dir_info_rate_arrays,
)

K = 25


def _past(name, k=K):
    return [(name, s) for s in range(-k, 0)]


@pytest.fixture(scope='module')
def oracle():
    return SharedLatentGaussian(dims={'x': 4, 'y': 4, 'w': 4}, d=2, phi=0.9, seed=0)


class TestOracleIdentities:
    """The exact values must satisfy the taxonomy's identities."""

    def test_block_mi_is_extensive(self, oracle):
        """I_w = rate*w + b, so the slope must match the spectral rate.

        This is the fact the tutorial series is rebuilt around: block MI has no
        plateau in w, it grows linearly forever.
        """
        rate = oracle.mi_rate()
        slope, intercept = oracle.affine_fit(30, 60)
        assert abs(slope / rate - 1) < 1e-3, f"slope {slope} should match rate {rate}"
        assert intercept > 0, "the subextensive intercept should be positive here"
        # And it really does outgrow any realistic estimator ceiling.
        assert oracle.block_mi(30) > 20.0

    def test_directed_information_splits(self, oracle):
        """DI rate = TE(x->y) + instantaneous exchange."""
        te = oracle.exact(_past('x'), [('y', 0)], _past('y'))
        inst = oracle.exact([('x', 0)], [('y', 0)], _past('x') + _past('y'))
        di = oracle.exact([('x', s) for s in range(-K, 1)], [('y', 0)], _past('y'))
        assert abs(di - (te + inst)) < 1e-9

    def test_massey_conservation(self, oracle):
        """MI rate = DI rate (x->y) + TE(y->x), with a two-sided x window."""
        two_sided = oracle.exact([('x', s) for s in range(-K, K + 1)], [('y', 0)], _past('y'))
        di = oracle.exact([('x', s) for s in range(-K, 1)], [('y', 0)], _past('y'))
        te_reverse = oracle.exact(_past('y'), [('x', 0)], _past('x'))
        assert abs(two_sided - (di + te_reverse)) < 1e-9

    def test_two_sided_window_recovers_the_rate(self, oracle):
        """Only an acausal x window reaches the symmetric rate.

        A shared latent violates Massey's no-feedback condition, so the causal
        estimand converges strictly below the rate.
        """
        rate = oracle.mi_rate()
        two_sided = oracle.exact([('x', s) for s in range(-K, K + 1)], [('y', 0)], _past('y'))
        causal = oracle.exact(_past('x'), [('y', 0)], _past('y'))
        assert abs(two_sided / rate - 1) < 1e-3
        assert causal < rate

    def test_storage_bounded_by_excess_entropy(self, oracle):
        """AIS <= E_X: storage in use cannot exceed storage held."""
        ais = oracle.exact(_past('x'), [('x', 0)])
        excess = oracle.exact(_past('x'), [('x', s) for s in range(0, K)])
        assert ais <= excess

    def test_interaction_information_identity(self, oracle):
        """I(X,W;Y) - I(X;Y) - I(W;Y) equals I(X;Y|W) - I(X;Y)."""
        i_xw_y = oracle.exact([('x', 0), ('w', 0)], [('y', 0)])
        i_x_y = oracle.exact([('x', 0)], [('y', 0)])
        i_w_y = oracle.exact([('w', 0)], [('y', 0)])
        i_x_y_given_w = oracle.exact([('x', 0)], [('y', 0)], [('w', 0)])
        assert abs((i_xw_y - i_x_y - i_w_y) - (i_x_y_given_w - i_x_y)) < 1e-9

    def test_conditioning_on_nothing_is_plain_mi(self, oracle):
        assert oracle.exact([('x', 0)], [('y', 0)], []) == pytest.approx(
            oracle.exact([('x', 0)], [('y', 0)]))

    def test_tau_matches_phi(self):
        o = SharedLatentGaussian(phi=0.9)
        assert o.tau == pytest.approx(-1.0 / np.log(0.9))


class TestOracleSampling:

    def test_sample_shapes_and_orientation(self, oracle):
        s = oracle.sample(T=500, seed=1)
        assert set(s) == {'x', 'y', 'w'}
        for name, arr in s.items():
            assert arr.shape == (500, 4), "samples must be (n_timepoints, n_channels)"

    def test_sample_is_seeded(self, oracle):
        assert np.allclose(oracle.sample(T=200, seed=3)['x'],
                           oracle.sample(T=200, seed=3)['x'])
        assert not np.allclose(oracle.sample(T=200, seed=3)['x'],
                               oracle.sample(T=200, seed=4)['x'])

    def test_empirical_covariance_tracks_exact(self, oracle):
        """A long draw should reproduce the model's own lag-0 cross-covariance."""
        s = oracle.sample(T=200_000, seed=0)
        empirical = (s['x'] - s['x'].mean(0)).T @ (s['y'] - s['y'].mean(0)) / len(s['x'])
        exact = oracle._cross_cov('x', 'y', 0)
        assert np.abs(empirical - exact).max() < 0.05

    def test_convenience_returns_data_and_oracle(self):
        data, o = generate_shared_latent_gaussian(T=300, dims={'x': 3, 'y': 3}, seed=2)
        assert data['x'].shape == (300, 3)
        assert isinstance(o, SharedLatentGaussian)
        assert o.exact([('x', 0)], [('y', 0)]) > 0


class TestOracleErrors:

    def test_unknown_process_names_the_available_ones(self, oracle):
        with pytest.raises(KeyError, match="Unknown process"):
            oracle.exact([('nope', 0)], [('y', 0)])

    def test_empty_group_rejected(self, oracle):
        with pytest.raises(ValueError, match="at least one"):
            oracle.exact([], [('y', 0)])

    def test_nonstationary_phi_rejected(self):
        with pytest.raises(ValueError, match=r"\[0, 1\)"):
            SharedLatentGaussian(phi=1.0)

    def test_single_process_has_no_default_pair(self):
        o = SharedLatentGaussian(dims={'x': 4})
        with pytest.raises(ValueError, match="single process"):
            o.block_mi(5)

    def test_noise_dict_must_cover_every_process(self):
        with pytest.raises(ValueError, match="missing entries"):
            SharedLatentGaussian(dims={'x': 4, 'y': 4}, noise={'x': 1.0})


class TestBuilderMatchesNamedQuantities:
    """The offset spec printed in the tutorials must be what the code does.

    Each case builds the same quantity two ways, through the library's
    hand-written builder and through the general spec builder, and requires the
    arrays to be identical. A failure here means a tutorial would advertise an
    offset pattern that does not reproduce the named function.
    """

    T, C, KK = 400, 3, 6

    @pytest.fixture(scope='class')
    def data(self):
        rng = np.random.default_rng(0)
        return {'x': rng.normal(size=(self.T, self.C)).astype('float32'),
                'y': rng.normal(size=(self.T, self.C)).astype('float32')}

    def _assert_same(self, named, spec_built):
        for got, want in zip(spec_built, named):
            assert got.shape == want.shape
            assert torch.allclose(got, want)

    def test_active_information_storage(self, data):
        k = self.KK
        named = build_past_future(data['x'], past_len=k, future_len=1)
        a, b, c, _ = build_offset_arrays(
            data, {'A': [('x', s) for s in range(-k, 0)], 'B': [('x', 0)]})
        assert c is None
        self._assert_same(named, (a, b))

    def test_excess_entropy(self, data):
        k = self.KK
        named = build_past_future(data['x'], past_len=k, future_len=k)
        a, b, _, _ = build_offset_arrays(
            data, {'A': [('x', s) for s in range(-k, 0)],
                   'B': [('x', s) for s in range(0, k)]})
        self._assert_same(named, (a, b))

    def test_cross_predictive_information(self, data):
        k = self.KK
        named = build_cross_offset(data['x'], data['y'], past_len=k, future_len=k)
        a, b, _, _ = build_offset_arrays(
            data, {'A': [('x', s) for s in range(-k, 0)],
                   'B': [('y', s) for s in range(0, k)]})
        self._assert_same(named, (a, b))

    def test_mi_rate(self, data):
        w, h = 5, 4
        named = _build_mi_rate_arrays(data['x'], data['y'], h, w)
        built = build_offset_arrays(
            data, {'A': [('x', s) for s in range(-w, w + 1)],
                   'B': [('y', 0)],
                   'C': [('y', s) for s in range(-h, 0)]})[:3]
        self._assert_same(named, built)

    def test_instantaneous_exchange(self, data):
        k = self.KK
        named = _build_inst_exchange_arrays(data['x'], data['y'], k)
        built = build_offset_arrays(
            data, {'A': [('x', 0)], 'B': [('y', 0)],
                   'C': [('x', s) for s in range(-k, 0)] + [('y', s) for s in range(-k, 0)]})[:3]
        self._assert_same(named, built)

    def test_directed_information_rate(self, data):
        k = self.KK
        named = _build_dir_info_rate_arrays(data['x'], data['y'], k)
        built = build_offset_arrays(
            data, {'A': [('x', s) for s in range(-k, 1)], 'B': [('y', 0)],
                   'C': [('y', s) for s in range(-k, 0)]})[:3]
        self._assert_same(named, built)


class TestBuilderBehaviour:

    @pytest.fixture
    def data(self):
        rng = np.random.default_rng(1)
        return {'x': rng.normal(size=(100, 2)).astype('float32'),
                'y': rng.normal(size=(100, 3)).astype('float32')}

    def test_valid_count_is_the_offset_span(self, data):
        a, b, c, n = build_offset_arrays(
            data, {'A': [('x', s) for s in range(-10, 0)], 'B': [('x', 3)]})
        assert n == 100 - 10 - 3
        assert a.shape == (n, 2, 10) and b.shape == (n, 2, 1)

    def test_channels_concatenate_across_processes(self, data):
        a, _, _, _ = build_offset_arrays(
            data, {'A': [('x', -1), ('y', -1)], 'B': [('x', 0)]})
        assert a.shape[1] == 2 + 3

    def test_one_dimensional_input_is_promoted(self):
        a, b, _, n = build_offset_arrays(
            {'x': np.arange(50, dtype='float32')},
            {'A': [('x', -1)], 'B': [('x', 0)]})
        assert a.shape == (49, 1, 1)

    def test_ragged_group_raises_with_counts(self, data):
        with pytest.raises(ValueError, match="different offset counts"):
            build_offset_arrays(
                data, {'A': [('x', 0)], 'B': [('y', 0)],
                       'C': [('x', -1), ('y', -3), ('y', -2), ('y', -1)]})

    def test_unknown_process_is_named(self, data):
        with pytest.raises(ValueError, match="which data does not provide"):
            build_offset_arrays(data, {'A': [('z', 0)], 'B': [('y', 0)]})

    def test_mismatched_lengths_rejected(self):
        with pytest.raises(ValueError, match="share a timepoint count"):
            build_offset_arrays({'x': np.zeros((50, 1)), 'y': np.zeros((40, 1))},
                                {'A': [('x', 0)], 'B': [('y', 0)]})

    def test_offsets_wider_than_the_series_rejected(self, data):
        with pytest.raises(ValueError, match="no valid samples"):
            build_offset_arrays(data, {'A': [('x', -200)], 'B': [('x', 0)]})

    def test_missing_group_keys_rejected(self, data):
        with pytest.raises(ValueError, match="both 'A' and 'B'"):
            build_offset_arrays(data, {'A': [('x', 0)]})


class TestBuilderEndToEnd:
    """The spec path must actually estimate, not merely produce arrays."""

    def test_spec_built_arrays_estimate_a_positive_quantity(self):
        import neural_mi as nmi
        o = SharedLatentGaussian(dims={'x': 3, 'y': 3}, d=2, phi=0.9, seed=0)
        data = o.sample(T=4000, seed=0)
        k = 8
        spec = {'A': [('x', s) for s in range(-k, 0)], 'B': [('x', 0)]}
        a, b, c, n = build_offset_arrays(data, spec)
        exact = o.exact(spec['A'], spec['B'])

        result = nmi.run(a, b, mode='estimate',
                         model=nmi.Model(embedding_dim=8, hidden_dim=32, n_layers=2),
                         training=nmi.Training(n_epochs=20, patience=10, batch_size=128),
                         n_workers=1, seed=0, show_progress=False)
        assert np.isfinite(result.mi_estimate)
        # A lower-bound estimator on a short run should land below the exact
        # value while still finding a clearly positive dependence.
        assert 0.05 < result.mi_estimate < exact * 1.5
