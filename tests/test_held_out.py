"""The held-out rows are counted once, and a held-out set smaller than one training batch is named.

The held-out curve chooses the epoch the estimate is read at and says whether
it is resolved. Its rows are each other's negatives, so on fewer rows than a
training batch its ceiling sits below the objective's and its noise is large.
The data check says so, and every warning about the curve names the count. It
is not a warning of its own, which fired on every dataset smaller than ten
batches whatever its curve then showed.
"""
import logging
import warnings

import numpy as np
import pytest

import neural_mi as nmi
from neural_mi.training.trainer import Trainer, held_out_rows, too_few_held_out

_MODEL = nmi.Model(embedding_dim=4, hidden_dim=8, n_layers=1)


def _messages(n, *, batch_size=64, **kwargs):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((n, 2)).astype(np.float32)
    y = (x + rng.standard_normal((n, 2))).astype(np.float32)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        nmi.run(x, y, model=_MODEL, training=nmi.Training(n_epochs=1, batch_size=batch_size),
                show_progress=False, seed=0, **kwargs)
    return [str(w.message) for w in caught]


class TestTheCount:

    @pytest.mark.parametrize('n', [300, 301, 1000, 1999, 57])
    def test_both_splits_hold_out_the_count(self, n):
        expected = held_out_rows(n, 0.9)
        _, random_test = Trainer._create_random_split(None, n, 0.9)
        _, blocked_test = Trainer._create_blocked_split(None, n, 0.9, 5)
        assert len(random_test) == len(blocked_test) == expected

    def test_a_tenth_of_a_round_count_is_exact(self):
        # 300 * (1 - 0.9) is 29.999..., which a blocked split used to round down to 29.
        assert held_out_rows(300, 0.9) == 30

    def test_the_line_is_one_training_batch(self):
        assert too_few_held_out(63, 64) and not too_few_held_out(64, 64)
        assert not too_few_held_out(0, 64)


def _data_check(caplog, n, **kwargs):
    """The data check's line on the rows, which the call logs at the info level."""
    with caplog.at_level(logging.INFO, logger='neural_mi'):
        messages = _messages(n, verbose=True, **kwargs)
    [line] = [r.getMessage() for r in caplog.records if r.getMessage().startswith('Data check:')]
    return line, messages


class TestTheDataCheckSaysSo:

    def test_fewer_held_out_rows_than_a_batch_are_named_and_do_not_warn(self, caplog):
        line, messages = _data_check(caplog, 300)
        assert '30 of them held out, fewer than one training batch (batch_size=64)' in line
        assert not any('are held out' in m for m in messages)

    def test_enough_held_out_rows_say_nothing_of_a_batch(self, caplog):
        line, _ = _data_check(caplog, 1000)
        assert '100 of them held out' in line and 'training batch' not in line

    def test_the_callers_own_held_out_rows_are_counted(self, caplog):
        line, _ = _data_check(caplog, 1000, split=nmi.Split(train_indices=np.arange(980),
                                                             test_indices=np.arange(980, 1000)))
        assert '20 of them held out, fewer than one training batch' in line

    def test_a_nonparametric_estimate_holds_nothing_out(self, caplog):
        rng = np.random.default_rng(0)
        x = rng.standard_normal((300, 2)).astype(np.float32)
        with caplog.at_level(logging.INFO, logger='neural_mi'):
            nmi.run(x, x + rng.standard_normal((300, 2)).astype(np.float32), estimator='ksg',
                    show_progress=False, verbose=True)
        [line] = [r.getMessage() for r in caplog.records if r.getMessage().startswith('Data check:')]
        assert 'a nonparametric estimate holds none out' in line and 'training batch' not in line


class TestTheCurveWarningsNameIt:

    def test_a_curve_warning_ends_with_the_count(self):
        from neural_mi.training.epochs import warn_about_curve
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            warn_about_curve('still rising', 5, 5, 6, 10, held_out=29)
            warn_about_curve('still rising', 5, 5, 6, 10)
        noted, plain = (str(w.message) for w in caught)
        assert 'scores only 29 held-out rows' in noted and 'held-out rows' not in plain

    def test_the_trainer_passes_the_count_only_when_it_is_small(self, monkeypatch):
        from neural_mi.training import trainer as trainer_module
        seen = []
        real = trainer_module.warn_about_curve

        def spy(*args, **kwargs):
            seen.append(kwargs.get('held_out'))
            return real(*args, **kwargs)

        monkeypatch.setattr(trainer_module, 'warn_about_curve', spy)
        # A curve that never rises above zero gets the message about non-positive
        # values instead of these, so the small run trains until its curve does.
        for n, epochs in ((300, 30), (3000, 3)):
            rng = np.random.default_rng(0)
            x = rng.standard_normal((n, 2)).astype(np.float32)
            y = (x + 0.1 * rng.standard_normal((n, 2))).astype(np.float32)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                nmi.run(x, y, model=_MODEL, training=nmi.Training(n_epochs=epochs, batch_size=64),
                        show_progress=False, seed=0)
        assert seen == [30, None]
