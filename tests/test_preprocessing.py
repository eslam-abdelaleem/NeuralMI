"""Changes to the data before rows are built are opt-in, and the data check changes nothing.

``Preprocessing(channel=..., time=...)`` transforms each continuous stream, X, Y
and W apart. The channel transforms are invertible and leave every quantity
unchanged; autoregressive whitening leaves the rate quantities unchanged and
warns elsewhere. Spike-time and categorical streams take neither, and rows
passed in take the channel transform only.
"""
import logging
import warnings

import numpy as np
import pytest
import torch
from scipy.stats import norm, rankdata

import neural_mi as nmi
from neural_mi.preprocessing import (ARWhitening, Copula, PCA, Transform, ZCA, ZScore, apply_preprocessing,
                                     check_streams)


def _ar2(n=4000, seed=0, channels=2):
    """A smooth AR(2) series per channel, its own past predicting most of its variance."""
    rng = np.random.default_rng(seed)
    x = np.zeros((n + 200, channels))
    e = rng.standard_normal((n + 200, channels))
    for t in range(2, n + 200):
        x[t] = 1.6 * x[t - 1] - 0.8 * x[t - 2] + e[t]
    return x[200:]


def _caught(fn, *args, **kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        out = fn(*args, **kwargs)
    return out, [str(w.message) for w in caught]


class TestTransforms:

    def test_zscore_centres_and_scales_and_leaves_a_constant_channel_unscaled(self):
        data = np.c_[np.random.default_rng(0).normal(5, 3, 500), np.full(500, 7.0)]
        out = ZScore().fit(data).transform(data)
        assert out[:, 0].mean() == pytest.approx(0, abs=1e-12) and out[:, 0].std() == pytest.approx(1)
        assert np.all(out[:, 1] == 0)

    def test_zca_spheres(self):
        rng = np.random.default_rng(1)
        data = rng.standard_normal((2000, 3)) @ np.array([[2, 0, 0], [1, 1, 0], [0, 3, 0.5]])
        out = ZCA().fit(data).transform(data)
        assert np.allclose(np.cov(out, rowvar=False), np.eye(3), atol=1e-6)

    def test_pca_rotates_and_cuts_only_when_asked(self):
        rng = np.random.default_rng(2)
        data = rng.standard_normal((500, 4)) * [5, 3, 1, 0.5]
        full = PCA().fit(data).transform(data)
        centred = data - data.mean(axis=0)
        assert np.allclose(np.linalg.norm(full, axis=1), np.linalg.norm(centred, axis=1))
        cut = PCA(n_components=2).fit(data)
        assert cut.transform(data).shape == (500, 2) and not cut.invertible

    def test_copula_is_the_rank_transform_ksg_used(self):
        data = np.random.default_rng(3).standard_normal((300, 2)) ** 3
        expected = norm.ppf((rankdata(data, axis=0) - 0.5) / len(data))
        assert np.allclose(Copula().fit(data).transform(data), expected)

    def test_ar_whitening_removes_the_own_past_and_keeps_the_length(self):
        x = _ar2(channels=1)
        fitted = ARWhitening().fit(x)
        assert len(fitted.coefficients_[0]) >= 2 and fitted.predicted_[0] > 0.9
        out = fitted.transform(x)
        assert out.shape == x.shape
        lag1 = np.corrcoef(out[10:-1, 0], out[11:, 0])[0, 1]
        assert abs(lag1) < 0.05
        assert len(ARWhitening(order=3).fit(x).coefficients_[0]) == 3

    def test_the_spectrum_and_ksg_use_the_same_transforms(self):
        from neural_mi.estimators.ksg import NOISE, prepare
        from neural_mi.utils import _whiten
        z = np.random.default_rng(4).standard_normal((200, 3)) * [1, 2, 3]
        z = z - z.mean(axis=0)
        assert np.allclose(_whiten(z, 'std'), z / z.std(axis=0))
        a = np.random.default_rng(5).standard_normal((100, 2)) * 4 + 1
        expected = (a - a.mean(axis=0)) / a.std(axis=0)
        noise = NOISE * np.random.default_rng(6).standard_normal(a.shape)
        assert np.allclose(prepare(a, False, np.random.default_rng(6)), expected + noise)


class TestApplying:

    def test_channel_transforms_remove_the_units(self):
        x = np.random.default_rng(7).standard_normal((400, 3)).astype(np.float32)
        a = apply_preprocessing({'x': x}, {}, nmi.Preprocessing(channel='zscore'), quantity='q', rate=False)
        b = apply_preprocessing({'x': x * 1e3}, {}, nmi.Preprocessing(channel='zscore'), quantity='q', rate=False)
        assert a['x'].dtype == np.float32 and np.allclose(a['x'], b['x'], atol=1e-5)

    def test_rows_passed_in_take_the_channel_transform_only(self):
        rows = torch.randn(50, 2, 5) * 100
        (out,), messages = _caught(lambda: apply_preprocessing(
            {'x': rows}, {}, nmi.Preprocessing(channel='zscore', time='ar'), quantity='q', rate=True).values())
        assert torch.is_tensor(out) and out.shape == rows.shape
        assert np.allclose(out.permute(0, 2, 1).reshape(-1, 2).std(dim=0, unbiased=False).numpy(), 1, atol=1e-4)
        assert any('leaves x as it is in time' in m for m in messages)

    def test_spike_and_categorical_streams_take_nothing(self):
        spikes = [np.sort(np.random.default_rng(8).uniform(0, 10, 50))]
        out, messages = _caught(apply_preprocessing, {'x': spikes}, {'x': 'spike'},
                                nmi.Preprocessing(channel='zscore'), quantity='q', rate=False)
        assert out['x'] is spikes
        assert any("a spike stream, as it is" in m and "input_scaling='standard'" in m for m in messages)

    def test_a_time_transform_warns_outside_the_rate_quantities(self):
        x = _ar2()
        _, rate = _caught(apply_preprocessing, {'x': x}, {}, nmi.Preprocessing(time='ar'), quantity='q', rate=True)
        _, other = _caught(apply_preprocessing, {'x': x}, {}, nmi.Preprocessing(time='ar'), quantity='q', rate=False)
        assert not any('changes what' in m for m in rate)
        assert any('changes what q measures' in m for m in other)

    def test_a_cut_by_pca_warns(self):
        x = np.random.default_rng(9).standard_normal((300, 4))
        out, messages = _caught(apply_preprocessing, {'x': x}, {}, nmi.Preprocessing(channel=PCA(n_components=2)),
                                quantity='q', rate=False)
        assert out['x'].shape == (300, 2) and any('PCA keeps 2 of the 4 components of x' in m for m in messages)

    def test_a_transform_of_your_own(self):
        class Square(Transform):
            def transform(self, data):
                return data ** 2

        x = np.random.default_rng(10).standard_normal((100, 2))
        out, messages = _caught(apply_preprocessing, {'x': x}, {}, nmi.Preprocessing(channel=Square()),
                                quantity='q', rate=False)
        assert np.allclose(out['x'], x ** 2) and any('a transform of your own' in m for m in messages)
        with pytest.raises(ValueError, match="whose domain is 'channel'"):
            apply_preprocessing({'x': x}, {}, nmi.Preprocessing(time=Square()), quantity='q', rate=False)
        with pytest.raises(ValueError, match='not a channel transform'):
            apply_preprocessing({'x': x}, {}, nmi.Preprocessing(channel='whiten'), quantity='q', rate=False)


class TestThroughTheCalls:

    def test_run_applies_and_records_it(self, monkeypatch):
        from neural_mi.analysis import modes
        seen = {}

        def capture(mode, x, y, *args, **kwargs):
            seen['x'] = x
            raise RuntimeError('captured')

        monkeypatch.setattr(modes, 'produce', capture)
        x = np.random.default_rng(11).standard_normal((300, 2)).astype(np.float32) * 1e3
        with pytest.raises(RuntimeError, match='captured'):
            nmi.run(x, x, preprocessing={'channel': 'zscore'}, show_progress=False)
        assert np.allclose(np.asarray(seen['x']).std(axis=0), 1, atol=1e-4)

    def test_the_call_records_what_it_applied(self):
        x = np.random.default_rng(12).standard_normal((300, 2))
        result = nmi.run(x, x + np.random.default_rng(13).standard_normal((300, 2)), estimator='ksg',
                         preprocessing=nmi.Preprocessing(channel='copula'), show_progress=False)
        assert result.params['preprocessing'] == {'channel': 'copula'}

    def test_a_named_quantity_transforms_its_series_once(self):
        x, y = _ar2(seed=1), _ar2(seed=2)
        result, messages = _caught(nmi.mi_rate, x, y, h=2, half_width=1, estimator='ksg', show_progress=False,
                                   preprocessing=nmi.Preprocessing(time='ar'))
        assert result.params['preprocessing'] == {'time': 'ar'}
        assert not any('changes what' in m or 'as it is in time' in m for m in messages)
        _, messages = _caught(nmi.active_information_storage, x, k=2, estimator='ksg', show_progress=False,
                              preprocessing=nmi.Preprocessing(time='ar'))
        assert any('changes what active_information_storage() measures' in m for m in messages)

    def test_transfer_entropy_takes_it_as_a_rate_quantity(self):
        x, y = _ar2(seed=3), _ar2(seed=4)
        result, messages = _caught(nmi.transfer_entropy, x, y, history_window=2, estimator='ksg',
                                   show_progress=False, preprocessing=nmi.Preprocessing(time='ar'))
        assert result.params['preprocessing'] == {'time': 'ar'}
        assert not any('changes what' in m for m in messages)


class TestTheCheck:

    def test_it_warns_about_what_needs_attention(self):
        rng = np.random.default_rng(14)
        bad = np.c_[rng.standard_normal(200), np.full(200, 3.0)]
        _, messages = _caught(check_streams, {'x': bad}, {}, quantity='q', rate=False)
        assert any('x has constant channel(s) [1]' in m for m in messages)
        nan = rng.standard_normal((200, 2))
        nan[5, 1] = np.nan
        _, messages = _caught(check_streams, {'x': nan}, {}, quantity='q', rate=False)
        assert any('non-finite values (NaN or infinite) in channel(s) [1]' in m for m in messages)

    def test_it_warns_about_units_unless_a_channel_transform_is_set(self):
        x = np.random.default_rng(15).standard_normal((200, 2)) * 1e3
        _, messages = _caught(check_streams, {'x': x}, {}, quantity='q', rate=False)
        assert any('typical scale of' in m and "Preprocessing(channel='zscore')" in m for m in messages)
        _, messages = _caught(check_streams, {'x': x}, {}, quantity='q', rate=False,
                              preprocessing=nmi.Preprocessing(channel='zscore'))
        assert not any('typical scale' in m for m in messages)

    def test_its_report_names_the_own_past_for_a_rate_quantity(self, caplog):
        x = _ar2()
        with caplog.at_level(logging.INFO, logger='neural_mi'):
            check_streams({'x': x, 'y': x}, {}, quantity="mode='transfer'", rate=True)
        assert "Data check for mode='transfer'" in caplog.text
        assert 'own past predicts up to' in caplog.text and "Preprocessing(time='ar')" in caplog.text

    def test_it_changes_nothing(self):
        x = np.random.default_rng(16).standard_normal((100, 2)) * 1e3
        before = x.copy()
        _caught(check_streams, {'x': x}, {}, quantity='q', rate=True)
        assert np.array_equal(x, before)


class TestTheRowsPartOfTheCheck:

    @staticmethod
    def _spike_call(params):
        rng = np.random.default_rng(17)
        spikes = [np.sort(rng.uniform(0, 20, 400)) for _ in range(3)]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            nmi.run(spikes, spikes, processing=nmi.Processing(x='spike', x_params=params),
                    model=nmi.Model(embedding_dim=4, hidden_dim=8, n_layers=1),
                    training=nmi.Training(n_epochs=1, shift_windows=False), show_progress=False, seed=0)
        return [str(w.message) for w in caught if 'typical scale' in str(w.message)]

    def test_it_reports_the_rows_their_held_out_count_and_scale(self, caplog):
        x = np.random.default_rng(18).standard_normal((300, 2)).astype(np.float32)
        with caplog.at_level(logging.INFO, logger='neural_mi'):
            nmi.run(x, x, model=nmi.Model(embedding_dim=4, hidden_dim=8, n_layers=1),
                    training=nmi.Training(n_epochs=1), show_progress=False, seed=0)
        # 30 held-out rows are fewer than one default batch of 128, which the line says.
        assert ('Data check: 300 samples reach the estimator, 30 of them held out, fewer than one training '
                'batch (batch_size=128)') in caplog.text
        assert "ceiling of log(30) low; their scale is x 1" in caplog.text

    def test_spike_times_far_from_unit_scale_warn(self):
        messages = self._spike_call({'window_size': 0.01})
        assert messages and 'spike times within each window' in messages[0]

    def test_rates_from_fine_bins_warn_and_name_the_counts(self):
        messages = self._spike_call({'window_size': 0.1, 'bin_size': 0.001, 'normalize_bins': True})
        assert messages and 'normalize_bins=False keeps the counts' in messages[0]

    def test_rows_the_tasks_cut_are_checked_too(self, caplog):
        """With shift_windows on, the default, each task cuts and moves its own rows. The check reads the
        rows at offset 0, so the call still reports their scale and warns."""
        rng = np.random.default_rng(17)
        spikes = [np.sort(rng.uniform(0, 20, 400)) for _ in range(3)]
        with warnings.catch_warnings(record=True) as caught, caplog.at_level(logging.INFO, logger='neural_mi'):
            warnings.simplefilter('always')
            nmi.run(spikes, spikes, processing=nmi.Processing(x='spike', x_params={'window_size': 0.01}),
                    model=nmi.Model(embedding_dim=4, hidden_dim=8, n_layers=1),
                    training=nmi.Training(n_epochs=1), show_progress=False, seed=0)
        assert any('spike times within each window' in str(w.message) for w in caught)
        assert 'their scale is x ' in caplog.text


class TestTheCheckOnLargeStreams:

    def test_it_reads_every_row_one_stretch_at_a_time(self, monkeypatch):
        from neural_mi import preprocessing
        monkeypatch.setattr(preprocessing, 'CHECK_ROWS', 40)
        rng = np.random.default_rng(19)
        x = rng.standard_normal((100, 2))
        x[:40, 1] = 2.0                       # constant over the first stretch only
        finite, std = preprocessing._channel_moments(x)
        assert finite.all() and np.allclose(std, x.std(axis=0))
        _, messages = _caught(check_streams, {'x': x}, {}, quantity='q', rate=False)
        assert not any('constant channel' in m for m in messages)
        x[90, 0] = np.inf                     # in the last stretch
        _, messages = _caught(check_streams, {'x': x}, {}, quantity='q', rate=False)
        assert any('non-finite values (NaN or infinite) in channel(s) [0]' in m for m in messages)


class TestEncodersThatScaleTheirInputs:
    """With Model(input_scaling=...) the encoders scale what they receive, so the scale is no warning."""

    def test_the_stream_check_leaves_the_scale_to_them(self):
        x = np.random.default_rng(20).standard_normal((200, 2)) * 1e3
        _, messages = _caught(check_streams, {'x': x}, {}, quantity='q', rate=False, encoder_scales=True)
        assert not any('typical scale' in m for m in messages)

    def test_the_rows_check_leaves_the_scale_to_them(self):
        from neural_mi.run import _report_rows
        rows = {'x': torch.rand(100, 3, 5) * 1e-3}
        _, plain = _caught(_report_rows, 100, 'windows', 10, {}, rows, {'x': 'spike'})
        _, scaled = _caught(_report_rows, 100, 'windows', 10, {'input_scaling': 'standard'}, rows, {'x': 'spike'})
        assert any('typical scale' in m for m in plain) and not any('typical scale' in m for m in scaled)

    def test_a_named_quantity_passes_it_to_its_check(self):
        from neural_mi.quantities import _prepared_series
        x = torch.randn(300, 2) * 1e3
        _, messages = _caught(_prepared_series, {'x': x}, None, None, 'mi_rate', encoder_scales=True)
        assert not any('typical scale' in m for m in messages)
