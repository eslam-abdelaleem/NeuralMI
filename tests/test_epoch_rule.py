# tests/test_epoch_rule.py
"""Which epoch of the held-out curve the estimate is read at, and the diagnosis of the curve.

choose_epoch reads the earliest epoch at the top of the smoothed curve
('plateau_start') or its maximum ('peak'), measuring the curve's noise from the
curve itself. These tests give it 100 noisy draws of curves of each shape the
diagnosis names and check how often it reads and diagnoses them as it should.
"""
import warnings

import numpy as np
import pytest

from neural_mi.logger import _kind
from neural_mi.training.epochs import choose_epoch, curve_noise, warn_about_curve
from neural_mi.training.trainer import Trainer

E = np.arange(60, dtype=float)
REACH = Trainer._smoothing_reach(1.0, 5)


def _reads(curve, sigma=0.03, rule='plateau_start', ran_to_limit=False, draws=100):
    """(epoch read, peak, diagnosis) for each noisy draw of the curve."""
    out = []
    for seed in range(draws):
        history = curve + sigma * np.random.default_rng(seed).standard_normal(len(curve))
        out.append(choose_epoch(history, Trainer._smooth(None, list(history), 1.0, 5), rule, REACH, ran_to_limit))
    return np.array([r[0] for r in out]), np.array([r[1] for r in out]), np.array([r[2] for r in out])


def test_noise_is_the_spread_from_epoch_to_epoch_and_ignores_a_trend():
    rng = np.random.default_rng(0)
    assert curve_noise(0.1 * np.arange(2000) + 0.05 * rng.standard_normal(2000)) == pytest.approx(0.05, rel=0.1)
    assert curve_noise([0.1, 0.3]) == 0.0


def test_a_clean_peak_is_read_at_the_peak():
    curve = 0.2 + np.where(E <= 15, np.exp(-0.5 * ((E - 15) / 3) ** 2), np.exp(-0.5 * ((E - 15) / 6) ** 2))
    epochs, _, diagnoses = _reads(curve)
    assert np.all(np.abs(epochs - 15) <= 2) and np.all(diagnoses == 'ok')


def test_a_plateau_is_read_at_its_start_and_the_peak_rule_at_its_maximum():
    curve = 1 - np.exp(-(E + 1) / 2)                       # reaches 98% of its rise at epoch 7
    epochs, peaks, diagnoses = _reads(curve)
    assert np.median(np.abs(epochs - 7)) <= 2 and np.mean(np.abs(epochs - 7) <= 4) >= 0.85
    assert np.median(peaks - epochs) >= 10 and np.mean(diagnoses == 'ok') >= 0.85
    assert np.array_equal(_reads(curve, rule='peak')[0], peaks)


def test_a_curve_flat_from_the_start_is_read_at_epoch_zero():
    epochs, _, diagnoses = _reads(np.ones(60))
    assert np.median(epochs) == 0 and np.mean(epochs <= 5) >= 0.9 and np.mean(diagnoses == 'ok') >= 0.85


@pytest.mark.parametrize('curve, ran_to_limit, diagnosis', [
    (E / 60, True, 'still rising'),
    (1 - np.exp(-(E + 1) / 2), True, 'ok'),               # a plateau that ran to the limit is not rising
    (np.zeros(60), False, 'not resolved'),
    (np.where(E <= 2, 1 - 0.3 * (2 - E), 1 - 0.15 * np.abs(E - 2) ** 1.5), False, 'too fast'),
    (1 - np.exp(-(np.arange(120) + 1) / 12), False, 'fuzzy start'),
])
def test_each_diagnosis(curve, ran_to_limit, diagnosis):
    assert np.mean(_reads(curve, sigma=0.01, ran_to_limit=ran_to_limit)[2] == diagnosis) >= 0.85


@pytest.mark.parametrize('diagnosis', ['not resolved', 'still rising', 'too fast', 'fuzzy start'])
def test_each_warning_differs_between_networks_only_in_its_numbers(diagnosis):
    """The library shows a message once per call and counts repeats that differ only in numbers."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        warn_about_curve(diagnosis, 3, 4, 6, 100)
        warn_about_curve(diagnosis, 17, 21, 6, 200)
        warn_about_curve('ok', 3, 4, 6, 100)
    first, second = (str(w.message) for w in caught)
    assert _kind('UserWarning', first) == _kind('UserWarning', second)
