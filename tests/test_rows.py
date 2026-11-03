"""One record of how rows lie in time, written where rows are built and read wherever time order matters.

Every builder of rows records their geometry (``neural_mi/data/rows.py``): the
processors their windows, the named quantities the rows they slice by offset,
transfer entropy its histories. The split's leak checks, the rigorous chunks
and the block shuffle read that record and nothing else.
"""
import math
import warnings

import numpy as np
import pytest
import torch

import neural_mi as nmi
from neural_mi.data.rows import (RowGeometry, describe_rows, offset_rows, record_rows, row_geometry, row_identity,
                                 same_rows, window_rows)

_MODEL = nmi.Model(embedding_dim=4, hidden_dim=8, n_layers=1)
_TRAINING = nmi.Training(n_epochs=2, batch_size=32)


def _series(n=600, seed=0):
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.standard_normal((n, 1)), axis=0) * 0.1
    y = x + rng.standard_normal((n, 1))
    return x.astype(np.float32), y.astype(np.float32)


def _quiet(fn, *args, **kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = fn(*args, model=_MODEL, training=kwargs.pop('training', _TRAINING), show_progress=False,
                    seed=0, **kwargs)
    return result, [str(w.message) for w in caught]


def _recorded(result):
    """The (row_step, row_span) every network of a result recorded, as one set."""
    tables = [result.runs]
    for entry in result.details.values():
        if 'trainings' in entry:
            tables.append(entry['trainings'])
    pairs = set()
    for table in tables:
        if 'row_step' in table.columns:
            pairs |= {(float(a), float(b)) for a, b in zip(table['row_step'], table['row_span'])}
    return pairs


class TestGeometry:

    def test_offsets_on_a_raw_series_count_samples(self):
        assert offset_rows(5, 2) == RowGeometry(step=2.0, span=5.0)

    def test_offsets_on_a_grid_build_on_its_windows(self):
        # Windows of 4 every 2: five of them in a row cover 2 * 4 + 4.
        assert offset_rows(5, 1, window_rows(4, 2)) == RowGeometry(step=2.0, span=12.0)

    def test_rows_per_span_and_overlap(self):
        assert RowGeometry(1, 5).rows_per_span == 5 and RowGeometry(1, 5).overlapping
        assert RowGeometry(4, 4).rows_per_span == 1 and not RowGeometry(4, 4).overlapping
        assert RowGeometry(2, 5).rows_per_span == 3

    def test_no_record_means_no_known_time_order(self):
        assert row_geometry({}) is None
        assert row_geometry({'row_step': 1, 'row_span': 3}) == RowGeometry(1.0, 3.0)


class TestEveryBuilderRecordsItsRows:

    @pytest.mark.parametrize('call, expected', [
        (lambda x, y, **kw: nmi.active_information_storage(x, k=4, stride=2, **kw), (2.0, 5.0)),
        (lambda x, y, **kw: nmi.predictive_information(x, k=3, **kw), (1.0, 6.0)),
        (lambda x, y, **kw: nmi.cross_predictive_information(x, y, k=3, **kw), (1.0, 6.0)),
        (lambda x, y, **kw: nmi.instantaneous_mi(x, y, **kw), (1.0, 1.0)),
        (lambda x, y, **kw: nmi.block_mi(x, y, window_size=4, **kw), (4.0, 4.0)),
        (lambda x, y, **kw: nmi.transfer_entropy(x, y, history_window=3, stride=2, **kw), (2.0, 4.0)),
        (lambda x, y, **kw: nmi.conditional_transfer_entropy(x, y, y[::-1].copy(), history_window=3, **kw),
         (1.0, 4.0)),
        (lambda x, y, **kw: nmi.mi_rate(x, y, h=3, half_width=5, **kw), (1.0, 11.0)),
        (lambda x, y, **kw: nmi.mi_rate(x, y, h=7, half_width=2, **kw), (1.0, 10.0)),
        (lambda x, y, **kw: nmi.instantaneous_exchange(x, y, k=3, **kw), (1.0, 4.0)),
        (lambda x, y, **kw: nmi.directed_information_rate(x, y, k=2, **kw), (1.0, 3.0)),
    ], ids=['ais', 'predictive', 'cross_predictive', 'instantaneous', 'block', 'transfer',
            'conditional_transfer', 'mi_rate', 'mi_rate_long_history', 'exchange', 'directed_rate'])
    def test_the_rows_of_every_temporal_quantity_are_recorded(self, call, expected):
        x, y = _series()
        result, _ = _quiet(call, x, y)
        assert _recorded(result) == {expected}

    def test_a_grid_puts_the_rows_one_step_apart(self):
        x, y = _series()
        processing = nmi.Processing(x='continuous', y='continuous', x_params={'window_size': 1})
        result, _ = _quiet(nmi.mi_rate, x, y, h=1, half_width=1, processing=processing)
        assert _recorded(result) == {(1.0, 3.0)}

    @pytest.mark.parametrize('x_params, message', [
        ({'window_size': 4}, 'one time step wide'),
        ({'window_size': 1, 'step_size': 0.5}, 'tile time'),
    ], ids=['wide', 'overlapping'])
    def test_a_grid_whose_rows_are_not_single_steps_is_refused(self, x_params, message):
        x, y = _series()
        processing = nmi.Processing(x='continuous', y='continuous', x_params=x_params)
        with pytest.raises(ValueError, match=message):
            _quiet(nmi.mi_rate, x, y, h=1, half_width=1, processing=processing)

    def test_rows_passed_in_built_are_not_known_to_be_in_time_order(self):
        rng = np.random.default_rng(1)
        x = rng.standard_normal((300, 2, 5)).astype(np.float32)
        result, _ = _quiet(nmi.run, x, x + rng.standard_normal(x.shape).astype(np.float32))
        assert _recorded(result) == set()


class TestTheSplitReadsTheRecord:

    def test_a_gap_shorter_than_a_row_warns_for_rows_built_by_offset(self):
        # 370 rows of span 31, 37 held out in five blocks of 7: a gap of 4 rows
        # buys 4 samples, far short of a row.
        x, _ = _series(400)
        _, messages = _quiet(nmi.active_information_storage, x, k=30)
        assert any('Blocked split may leak raw samples' in m for m in messages)

    def test_a_random_split_of_overlapping_rows_warns(self):
        x, _ = _series()
        _, messages = _quiet(nmi.active_information_storage, x, k=4, split=nmi.Split(mode='random'))
        assert any('Random split of rows that share samples' in m for m in messages)

    def test_a_random_split_of_rows_that_share_nothing_is_quiet(self):
        x, y = _series()
        _, messages = _quiet(nmi.instantaneous_mi, x, y, split=nmi.Split(mode='random'))
        assert not any('Random split of rows' in m for m in messages)

    def test_the_split_dimensionality_shares_runs_the_same_checks(self):
        from neural_mi.analysis.dimensionality import _get_or_create_shared_split
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            _get_or_create_shared_split({'row_step': 1.0, 'row_span': 50.0}, 1000)
            _get_or_create_shared_split({'row_step': 1.0, 'row_span': 50.0, 'split_mode': 'random'}, 1000)
        messages = [str(w.message) for w in caught]
        assert any('Blocked split may leak' in m for m in messages)
        assert any('Random split of rows that share samples' in m for m in messages)


class TestRigorousChunks:

    @staticmethod
    def _chunker(x, params, **kwargs):
        from neural_mi.analysis.rigorous import _Chunker
        return _Chunker(x, x, params, **kwargs)

    def test_recorded_rows_are_cut_in_time_order(self):
        chunker = self._chunker(torch.zeros(10, 1, 3), {'row_step': 1, 'row_span': 3}, deferred=False)
        assert chunker.mode == 'contiguous'
        assert [c.tolist() for c in chunker.chunks(2)] == [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]

    def test_other_rows_are_cut_in_a_random_order_unless_told(self):
        np.random.seed(0)
        assert self._chunker(torch.zeros(10, 1), {}, deferred=False).mode == 'permuted'
        assert self._chunker(torch.zeros(10, 1), {}, deferred=False, temporal_chunking=True).mode == 'contiguous'

    def test_a_recording_each_chunk_windows_cannot_be_permuted(self):
        params = {'processor_type_x': 'continuous', 'processor_type_y': 'continuous', 'shift_windows': True,
                  'processor_params_x': {'window_size': 5}}
        x = torch.zeros(200, 1)
        assert self._chunker(x, params, deferred=True).mode == 'contiguous'
        with pytest.raises(ValueError, match=r"temporal_chunking=False\) cannot apply"):
            self._chunker(x, params, deferred=True, temporal_chunking=False)

    @pytest.mark.parametrize('call', [
        lambda x, y, **kw: nmi.active_information_storage(x, k=3, **kw),
        lambda x, y, **kw: nmi.mi_rate(x, y, h=2, half_width=2, **kw),
        lambda x, y, **kw: nmi.transfer_entropy(x, y, history_window=2, **kw),
    ], ids=['ais', 'mi_rate', 'transfer'])
    def test_every_temporal_quantity_is_cut_into_contiguous_chunks(self, call):
        x, y = _series(800)
        result, _ = _quiet(call, x, y, rigorous=nmi.Rigorous(gamma_range=[1, 2], min_gamma_points=2))
        assert set(result.runs['chunking_mode']) == {'contiguous'}

    def test_independent_rows_are_still_permuted(self):
        rng = np.random.default_rng(2)
        x = rng.standard_normal((800, 2)).astype(np.float32)
        result, _ = _quiet(nmi.run, x, x + rng.standard_normal(x.shape).astype(np.float32), mode='rigorous',
                           rigorous=nmi.Rigorous(gamma_range=[1, 2], min_gamma_points=2))
        assert set(result.runs['chunking_mode']) == {'permuted'}


class TestBlockShuffle:

    @pytest.mark.parametrize('x, params, rows', [
        (torch.zeros(40, 1, 5), {'row_step': 1.0, 'row_span': 5.0}, 5),
        (torch.zeros(40, 1, 5), {'row_step': 2.0, 'row_span': 5.0}, 3),
        (torch.zeros(40, 1, 5), {}, 1),
        (torch.zeros(400, 1), {'row_step': 1.0, 'row_span': 4.0}, 4),
        (torch.zeros(400, 1), {'processor_type_x': 'continuous', 'processor_params_x': {'window_size': 10}}, 10),
        (torch.zeros(400, 1), {'row_step': 1.0, 'row_span': 4.0,
                               'processor_params_x': {'window_size': 1, 'preprocessed': True}}, 4),
    ], ids=['offset_rows', 'strided_rows', 'unknown_rows', 'transfer_series', 'series_windowed_later',
            'transfer_series_off_a_grid'])
    def test_a_block_holds_the_rows_one_row_spans(self, x, params, rows):
        from neural_mi.analysis.permutation import _window_rows
        assert _window_rows(x, params, {}) == rows


class TestOffsetQuantity:
    """A spec with no named function runs as the named quantities do, rows' geometry included."""

    def test_a_conditional_spec_runs_with_its_geometry_and_records_the_spec(self):
        x, y = _series()
        spec = {'A': [('x', s) for s in range(-3, 0)], 'B': [('y', 0)], 'C': [('y', s) for s in range(-2, 0)]}
        result, _ = _quiet(nmi.offset_quantity, {'x': x, 'y': y}, spec, stride=2)
        assert result.params['spec'] == spec and result.mode == 'conditional'
        assert _recorded(result) == {(2.0, 4.0)}

    def test_processing_needs_the_stream_names_it_reads(self):
        x, y = _series()
        with pytest.raises(ValueError, match="Name the streams 'x', 'y' and 'w'"):
            nmi.offset_quantity({'a': x, 'b': y}, {'A': [('a', -1)], 'B': [('b', 0)]},
                                processing=nmi.Processing(x='continuous', x_params={'window_size': 1}))


class TestShiftingRowsAStrideThins:
    """A shift applies wherever rows thin a series and nowhere else, and moves them by less than one
    row step: offset rows at a stride above one change phase each epoch, and windows that start one
    sample apart never move."""

    def test_the_phase_shifter_keeps_one_row_in_every_stride_at_every_phase(self):
        from neural_mi.data.shift_windowing import RowPhaseShifter
        x = torch.arange(10, dtype=torch.float32).reshape(10, 1, 1)
        shifter = RowPhaseShifter((x, x + 100), x + 1000, 3)
        assert shifter.n_windows == 3
        for phase in range(3):
            (a, c), b = shifter.windows_at(phase)
            assert a[:, 0, 0].tolist() == [phase, phase + 3, phase + 6]
            assert c[:, 0, 0].tolist() == [v + 100 for v in a[:, 0, 0].tolist()]
            assert b[:, 0, 0].tolist() == [v + 1000 for v in a[:, 0, 0].tolist()]
        draws = {shifter.random_shift(torch.Generator().manual_seed(s)) for s in range(40)}
        assert draws == {0, 1, 2}

    def test_a_window_shift_stays_within_one_step(self):
        from neural_mi.data.shift_windowing import WindowShifter
        generator = torch.Generator().manual_seed(0)
        tiled = WindowShifter(torch.zeros(200, 1), window_size=10, step_size=4)
        assert {tiled.random_shift(generator) for _ in range(60)} == {0, 1, 2, 3}
        dense = WindowShifter(torch.zeros(200, 1), window_size=10, step_size=1)
        assert {dense.random_shift(generator) for _ in range(20)} == {0}

    @pytest.mark.parametrize('stride, shift, phases', [(3, True, 3), (3, False, None), (1, True, None)],
                             ids=['strided_and_shifting', 'not_shifting', 'every_step'])
    def test_offset_rows_shift_only_where_a_stride_thins_them(self, monkeypatch, stride, shift, phases):
        from neural_mi.data import shift_windowing
        seen = []
        real = shift_windowing.try_build_row_phase_dataset

        def spy(x_data, y_data, params, data_device='cpu'):
            seen.append(params.get('_row_phases'))
            return real(x_data, y_data, params, data_device)
        monkeypatch.setattr(shift_windowing, 'try_build_row_phase_dataset', spy)
        x, _ = _series(900)
        training = nmi.Training(n_epochs=2, batch_size=32, shift_windows=shift)
        result, _ = _quiet(nmi.active_information_storage, x, k=4, stride=stride, training=training)
        assert set(seen) == {phases}
        assert _recorded(result) == {(float(stride), 5.0)}

    def test_a_row_reaches_one_step_further_when_it_shifts(self):
        # 200 rows of span 5 and step 3, 20 held out in five blocks of 4: a gap of
        # 2 rows buys 6 samples, enough for a row of 5 that stays put and short of
        # the 8 a shifting row reaches.
        x, _ = _series(604)
        for shift, leaks in ((False, False), (True, True)):
            training = nmi.Training(n_epochs=2, batch_size=16, shift_windows=shift)
            _, messages = _quiet(nmi.active_information_storage, x, k=4, stride=3, training=training)
            assert any('Blocked split may leak' in m for m in messages) == leaks

    def test_a_block_of_rows_built_at_every_step_holds_one_span(self):
        from neural_mi.analysis.permutation import _window_rows
        params = {'row_step': 3.0, 'row_span': 5.0, '_row_phases': 3}
        assert _window_rows(torch.zeros(40, 1, 5), params, {}) == 5


class TestRowIdentity:
    """Each task reports which rows it read, so that the terms of a difference, trained in separate
    tasks, can be checked to have read the same ones (sweep.combine_terms)."""

    def test_a_grid_of_spike_times_starts_at_the_latest_first_spike(self):
        from neural_mi.data.handler import create_dataset
        rng = np.random.default_rng(0)
        early = [np.sort(rng.uniform(0.0, 20.0, 100)) for _ in range(2)]
        late = [np.sort(rng.uniform(5.0, 20.0, 80)) for _ in range(2)]
        spike = {'window_size': 0.5}
        dataset = create_dataset(early, late, processor_type_x='spike', processor_params_x=spike,
                                 processor_type_y='spike', processor_params_y=spike)
        identity = row_identity(dataset, {})
        assert identity['origin'] == pytest.approx(min(s[0] for s in late))
        assert identity['step'] == 0.5 and identity['spans'] == {'x': 0.5, 'y': 0.5}
        assert identity['count'] == len(dataset) and identity['kept']

    def test_rows_resliced_from_the_raw_arrays_start_at_their_first_sample(self):
        from neural_mi.data.shift_windowing import try_build_shift_windows_dataset
        x, y = _series()
        params = {'shift_windows': True, 'processor_type_x': 'continuous', 'processor_type_y': 'continuous',
                  'processor_params_x': {'window_size': 10, 'step_size': 0.5}}
        dataset = try_build_shift_windows_dataset(x, y, params)
        assert row_identity(dataset, params) == {'count': len(dataset), 'origin': 0.0, 'step': 5.0,
                                                 'spans': {'x': 10.0, 'y': 10.0}, 'kept': None}

    def test_rows_built_before_the_task_are_known_by_their_count_and_record(self):
        from neural_mi.data.handler import PairedDataset
        from neural_mi.data.static import StaticDataset
        dataset = PairedDataset(StaticDataset(torch.zeros(7, 1, 3)), StaticDataset(torch.zeros(7, 1, 3)))
        assert row_identity(dataset, {}) == {'count': 7, 'origin': None, 'step': None, 'spans': None,
                                             'kept': None}
        params = {}
        record_rows(params, window_rows(3, 3))
        assert row_identity(dataset, params)['step'] == 3.0

    def test_same_rows_compares_what_both_identities_know(self):
        rows = {'count': 400, 'origin': 0.05, 'step': 0.5, 'spans': {'x': 0.5, 'y': 0.5}, 'kept': 'a1'}
        assert same_rows(rows, dict(rows))
        for change in ({'count': 320}, {'origin': 40.0}, {'step': 1.0}, {'kept': 'b2'},
                       {'spans': {'x': 0.5, 'y': 1.0}}):
            assert not same_rows(rows, {**rows, **change}), change
        # X's window belongs to its term: the conditioning term's X is W.
        assert same_rows(rows, {**rows, 'spans': {'x': 1.0, 'y': 0.5}})
        assert same_rows(rows, {'count': 400, 'origin': None, 'step': None, 'spans': None, 'kept': None})

    def test_an_identity_reads_as_words(self):
        rows = {'count': 400, 'origin': 0.05, 'step': 0.5, 'spans': {'y': 0.5}, 'kept': None}
        assert describe_rows(rows) == '400 rows starting at 0.05, one every 0.5, Y over 0.5'
        assert describe_rows({'count': 7, 'origin': None, 'step': None, 'spans': None}) == '7 rows'
