# neural_mi/training/epochs.py
"""Which epoch the reported estimate is read at, and what the held-out curve says about that choice.

The held-out curve is noisy, so its highest point is not always the epoch to read.
On a curve that rises and falls it is close. On a plateau it lands wherever the
noise happens to be highest, often many epochs into a stretch where the network
no longer improves on held-out data and only fits its training rows further,
which raises the training-side value that the estimate reads.

``'plateau_start'`` reads the earliest epoch at the top instead. The noise of the
curve is measured from the curve itself, near its top: on the raw curve up to its
smoothed top and the epochs the smoothing reads past it, a span the smoothing
settings set and no constant of its own. A curve keeps falling after its top
once the network fits its training rows, and grows noisier as it falls. Counted
in the noise, that fall would lower the threshold and move the epoch read
earlier, the more so the longer training ran. The level of the top is the median of the smoothed curve near its maximum,
and the epoch read is the first whose smoothed value comes within half the noise
of that level: the peak of a clean peak, the start of a plateau. ``'peak'``
reads the smoothed maximum.

The diagnosis says when the curve cannot support the choice. The thresholds were
set on simulated curves of known shape: a peak, a fast or slow plateau, a
plateau that declines or drifts up, an early collapse, a curve still rising when
training ends, and one that never rises, with and without early stopping.

Where a curve's approach to its top starts is measured, not judged from its
shape. The approach starts at the first epoch within three noise units of the
top. The trainer reads the training-side value there as well, on the same rows,
and calls the start uncertain ('fuzzy start') when the two readings differ by
more than FUZZY_SE of the estimate's standard errors.
"""
import warnings
from typing import Optional, Tuple

import numpy as np

from neural_mi.logger import user_stacklevel

EPOCH_RULES = ('plateau_start', 'peak')
RESOLVED_NOISE = 8      # the top must stand this many noise units above zero
FAST_EPOCHS = 8         # an early top that falls clearly within this many epochs came too fast
APPROACH_NOISE = 3      # the approach to the top starts this many noise units below it
FUZZY_SE = 3            # readings at the approach and at the epoch read this many standard errors apart
CLEAR_SE = 3            # a held-out value this many standard errors above zero shows the network learned something
RISING_EPOCHS = 20      # a curve that rose by RISING_NOISE noise units over this many final epochs
RISING_NOISE = 3        # when training ran to its limit was still rising


def curve_noise(history) -> float:
    """Epoch-to-epoch noise of the raw held-out curve.

    Second differences cancel a smooth trend, so a steep rise or fall does not
    count as noise, and the median ignores the few epochs where the curve turns
    sharply. For white noise of standard deviation sigma the second differences
    have standard deviation sqrt(6) sigma. A curve of fewer than three epochs has
    no measurable noise and returns 0.
    """
    d = np.diff(np.asarray(history, dtype=float), 2)
    d = d[np.isfinite(d)]
    return float(np.median(np.abs(d)) / (0.6745 * np.sqrt(6))) if len(d) else 0.0


def standard_error(block_values, n_rows: int) -> float:
    """Standard error of an estimate averaged over evaluation blocks: their spread over the root of their
    count, or sqrt(2 I / N) for a single block of N rows reading I nats."""
    v = np.asarray(block_values, dtype=float)
    return float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float(np.sqrt(2 * max(v[0], 0.0) / n_rows))


def stands_above_zero(test_mi: Optional[float], n_rows: Optional[int]) -> bool:
    """Whether a network's held-out value, in nats on ``n_rows`` rows, stands clearly above zero: by more than
    CLEAR_SE of its standard errors (:func:`standard_error`), that is above 2 CLEAR_SE^2 / n_rows nats. A
    network that learned nothing reads within that of zero. Without a row count, any value above zero counts."""
    if test_mi is None or not np.isfinite(test_mi) or test_mi <= 0:
        return False
    if not n_rows:
        return True
    return bool(test_mi > CLEAR_SE * standard_error([test_mi], int(n_rows)))


def choose_epoch(history, smoothed, rule: str, reach: Optional[int],
                 ran_to_limit: bool, n_rows: Optional[int] = None) -> Tuple[int, int, str, int]:
    """The epoch to read, the epoch of the smoothed maximum, the diagnosis of the curve, and the first epoch of
    its approach to the top.

    ``history`` is the raw held-out curve and ``smoothed`` the smoothed one, in
    nats. ``reach`` is how many epochs the smoothing looks ahead, None when
    unknown (a custom smoothing function). ``ran_to_limit`` is whether training
    used every epoch it was allowed. ``n_rows`` is the number of held-out rows
    every epoch scores, None when unknown. The diagnosis is one of:

    - ``'ok'``.
    - ``'not resolved'``: the top does not stand clearly above the curve's noise:
      RESOLVED_NOISE times its epoch-to-epoch noise, or zero by CLEAR_SE
      standard errors of its held-out rows (:func:`stands_above_zero`). Every
      epoch scores the same rows, so their sampling error moves the whole curve
      together and never shows in its epoch-to-epoch noise.
    - ``'still rising'``: training ran to its limit while the curve was still
      rising clearly over its last epochs.
    - ``'too fast'``: the curve reached its top within the smoothing's reach of
      the start and fell clearly soon after, so the start of the curve placed it.

    The trainer adds ``'fuzzy start'`` when the value read at the start of the
    approach moves the estimate (see the module docstring).
    """
    s = np.asarray(smoothed, dtype=float)
    peak = int(np.nanargmax(s))
    noise = curve_noise(np.asarray(history, dtype=float)[:peak + (reach or 0) + 1])
    level = float(np.median(s[s >= s[peak] - 2 * noise]))
    start = int(np.flatnonzero(s >= level - 0.5 * noise)[0])
    epoch = start if rule == 'plateau_start' else peak
    approach = int(np.flatnonzero(s >= level - APPROACH_NOISE * noise)[0])
    return epoch, peak, _diagnose(s, start, level, noise, reach, ran_to_limit, n_rows), approach


def _diagnose(s, start, level, noise, reach, ran_to_limit, n_rows=None) -> str:
    if n_rows and not stands_above_zero(level, n_rows):
        return 'not resolved'
    if not noise > 0:
        return 'ok'
    if level < RESOLVED_NOISE * noise:
        return 'not resolved'
    if ran_to_limit and s[-1] - s[max(0, len(s) - 1 - RISING_EPOCHS)] > RISING_NOISE * noise:
        return 'still rising'
    drop = np.flatnonzero(s[start:] < level - 3 * noise)
    if reach is not None and start < reach and len(drop) and drop[0] < FAST_EPOCHS:
        return 'too fast'
    return 'ok'


def warn_about_curve(diagnosis: str, epoch: int, peak: int, reach: Optional[int], n_epochs: int,
                     approach: Optional[int] = None, shift: Optional[float] = None, se: Optional[float] = None,
                     units: str = 'bits', held_out: Optional[int] = None) -> None:
    """Warn when the diagnosis is not 'ok'. The texts differ between networks only in their numbers, so the
    library shows each kind once per call and counts the rest. ``shift`` and ``se`` are in ``units``.
    ``held_out`` is the count of held-out rows the curve scores, given when it is smaller than one training
    batch (``trainer.too_few_held_out``), and each message then names it."""
    note = (f" The curve scores only {held_out} held-out rows, fewer than one training batch, so its noise is "
            f"large and its ceiling of log({held_out}) low. More data, or a lower Split(train_fraction=...), "
            f"gives it more." if held_out else "")
    if diagnosis == 'not resolved':
        warnings.warn(
            f"The held-out MI curve does not rise clearly above its own noise, either its epoch-to-epoch noise or "
            f"the sampling error of its held-out rows, so the data do not settle the epoch the estimate is read at, "
            f"nor the estimate. Check the curve with result.plot(). More data, more epochs or another network may "
            f"resolve it.{note}", UserWarning, stacklevel=user_stacklevel())
    elif diagnosis == 'still rising':
        warnings.warn(
            f"The held-out MI was still rising when training reached n_epochs={n_epochs}. The estimate may then "
            f"be an under-trained lower bound. Raise n_epochs, or learning_rate so that each epoch goes further."
            f"{note}", UserWarning, stacklevel=user_stacklevel())
    elif diagnosis == 'too fast':
        warnings.warn(
            f"The held-out MI reached its top at epoch {peak} and fell clearly soon after. A top this early lies "
            f"within the {reach} epochs over which the curve is smoothed, so the start of the curve placed it. Lower "
            f"learning_rate to spread the rise over more epochs.{note}", UserWarning, stacklevel=user_stacklevel())
    elif diagnosis == 'fuzzy start':
        warnings.warn(
            f"The held-out MI approached its top slowly, so where the top starts is uncertain. Read at the start "
            f"of the approach (epoch {approach}) instead of epoch {epoch}, the estimate moves by {shift:.4f} "
            f"{units}, more than {FUZZY_SE:g} times its standard error of {se:.4f} {units}. Check the curve with "
            f"result.plot().{note}", UserWarning, stacklevel=user_stacklevel())
