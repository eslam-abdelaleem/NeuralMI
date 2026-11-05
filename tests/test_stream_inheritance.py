# tests/test_stream_inheritance.py
"""One rule for the settings of Y and W, on every path.

A stream without a processor of its own reads with X's, and one without
settings of its own reads with X's settings, kept to the keys its processor
takes (``defaults.stream_processing``). ``run()`` resolves Y and W once, so the
rows it cuts, the rows a task cuts for itself under window shifting, the parts
of a processor grid and the lag scan all read the same settings. Every test
here compares a call that leaves a stream to inherit with the same call that
passes X's settings explicitly: every network must read the same rows, and the
estimates must agree.
"""
import warnings

import numpy as np
import pytest

import neural_mi as nmi
from neural_mi.data.handler import create_dataset
from neural_mi.defaults import stream_processing

_MODEL = nmi.Model(embedding_dim=4, hidden_dim=8, n_layers=1)
_RATE = 100.0
_X_PARAMS = {'window_size': 0.05, 'sample_rate': _RATE}


def _streams(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.standard_normal((n, 2)), axis=0) * 0.1
    y = x + rng.standard_normal((n, 2))
    w = x[:, :1] + rng.standard_normal((n, 1))
    return x.astype(np.float32), y.astype(np.float32), w.astype(np.float32)


def _run(*args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return nmi.run(*args, model=_MODEL, show_progress=False, seed=0, **kwargs)


def _rows(result):
    """What every network of a result read: its held-out rows and the windows it kept. A stream read in
    the wrong unit keeps other windows, whatever the networks then learn. A difference keeps them per
    network in its details, a single network in `runs`."""
    tables = [d['trainings'] for d in result.details.values() if 'trainings' in d] or [result.runs]
    return [list(t[['eval_size', 'n_windows_retained']].itertuples(index=False, name=None)) for t in tables]


def _same(inherited, explicit):
    assert _rows(inherited) == _rows(explicit)
    assert np.array_equal(inherited.dataframe['mi_mean'].to_numpy(), explicit.dataframe['mi_mean'].to_numpy())


class TestTheRule:

    def test_a_stream_without_a_processor_or_settings_reads_with_xs(self):
        assert stream_processing(None, None, 'continuous', _X_PARAMS) == ('continuous', _X_PARAMS)

    def test_settings_are_kept_to_the_keys_the_streams_processor_takes(self):
        spike_x = {'window_size': 0.1, 'bin_size': 0.01, 'sample_rate': _RATE}
        assert stream_processing('continuous', None, 'spike', spike_x) == (
            'continuous', {'window_size': 0.1, 'sample_rate': _RATE})

    def test_settings_given_are_the_streams_own(self):
        assert stream_processing(None, {'window_size': 0.2}, 'continuous', _X_PARAMS) == (
            'continuous', {'window_size': 0.2})

    def test_without_a_processor_for_x_nothing_is_inherited(self):
        assert stream_processing(None, None, None, None) == (None, None)

    def test_a_stream_given_as_rows_built_already_inherits_nothing(self):
        """A 3-D stream has no raw series for a processor to read, whichever stream it is."""
        built = np.zeros((10, 2, 5))
        assert stream_processing(None, None, 'continuous', {'window_size': 5}, built) == (None, None)
        assert stream_processing(None, None, 'continuous', {'window_size': 5}, np.zeros((50, 2)))[0] == 'continuous'


class TestEveryPathReadsTheSameSettings:

    @pytest.mark.parametrize('shift', [True, False], ids=['shifting', 'not_shifting'])
    def test_y_left_to_inherit_reads_like_y_given_xs_settings(self, shift):
        """X on a clock of 100 Hz. Before the rule held everywhere, a Y left to inherit lost X's
        sample_rate on the default shifting path and was read in samples."""
        x, y, _ = _streams()
        training = nmi.Training(n_epochs=2, batch_size=32, shift_windows=shift)
        inherited = _run(x, y, processing=nmi.Processing(x='continuous', x_params=_X_PARAMS), training=training)
        explicit = _run(x, y, processing=nmi.Processing(x='continuous', x_params=_X_PARAMS, y='continuous',
                                                        y_params=_X_PARAMS), training=training)
        _same(inherited, explicit)
        assert inherited.params['processor_params_y']['sample_rate'] == _RATE

    def test_a_window_sweep_gives_every_task_xs_other_settings(self):
        """mode='sweep' windows inside each task: each task's Y reads the grid's window and keeps X's
        sample_rate."""
        x, y, _ = _streams()
        grid = {'window_size': [0.05, 0.1]}
        training = nmi.Training(n_epochs=2, batch_size=32)
        inherited = _run(x, y, mode='sweep', sweep_grid=grid, training=training,
                         processing=nmi.Processing(x='continuous', x_params=_X_PARAMS))
        explicit = _run(x, y, mode='sweep', sweep_grid=grid, training=training,
                        processing=nmi.Processing(x='continuous', x_params=_X_PARAMS, y='continuous',
                                                  y_params=_X_PARAMS))
        _same(inherited, explicit)

    def test_a_processor_grid_gives_every_part_xs_other_settings(self):
        """mode='conditional' splits a window grid into one call per window. W names its processor and
        no settings, and Y names neither: in every part both read the grid's window with X's
        sample_rate, not the grid's window alone."""
        x, y, w = _streams()
        common = dict(mode='conditional', sweep_grid={'window_size': [0.05, 0.1]},
                      training=nmi.Training(n_epochs=2, batch_size=32), conditional=nmi.Conditional(w_data=w))
        inherited = _run(x, y, processing=nmi.Processing(x='continuous', x_params=_X_PARAMS, w='continuous'),
                         **common)
        explicit = _run(x, y, processing=nmi.Processing(x='continuous', x_params=_X_PARAMS, y='continuous',
                                                        y_params=_X_PARAMS, w='continuous', w_params=_X_PARAMS),
                        **common)
        _same(inherited, explicit)

    def test_a_lag_scan_reads_y_in_xs_units(self):
        x, y, _ = _streams()
        training = nmi.Training(n_epochs=2, batch_size=32)
        lags = nmi.Lag(lag_range=[0.0, 0.05])
        inherited = _run(x, y, mode='lag', lag=lags, training=training,
                         processing=nmi.Processing(x='continuous', x_params=_X_PARAMS))
        explicit = _run(x, y, mode='lag', lag=lags, training=training,
                        processing=nmi.Processing(x='continuous', x_params=_X_PARAMS, y='continuous',
                                                  y_params=_X_PARAMS))
        _same(inherited, explicit)

    def test_create_dataset_keeps_xs_settings_to_the_keys_ys_processor_takes(self):
        """A categorical Y next to a continuous X takes X's window and clock and nothing it cannot read."""
        x, _, _ = _streams()
        labels = np.random.default_rng(1).integers(0, 3, (len(x), 1))
        dataset = create_dataset(x, labels, processor_type_x='continuous', processor_params_x=_X_PARAMS,
                                 processor_type_y='categorical')
        explicit = create_dataset(x, labels, processor_type_x='continuous', processor_params_x=_X_PARAMS,
                                  processor_type_y='categorical', processor_params_y=_X_PARAMS)
        assert len(dataset) == len(explicit)
