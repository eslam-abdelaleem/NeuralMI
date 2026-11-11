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

How closely the held-out rows pin the estimate is measured at the top of the
curve. An epoch whose smoothed held-out value lies within ``top_se`` standard
errors of the curve's maximum (:func:`standard_error` of the maximum on the
held-out rows) is one that those rows cannot tell apart from the top. These
epochs are the band (:func:`top_band`). The trainer reads the training-side
value at the first and the last epoch of the band, on the rows that the estimate
is read on. It diagnoses a flat top ('flat top') when the two readings differ by
more than ``band_share`` of the estimate.

Every threshold here is a setting of ``Training(...)``, with its default in
:data:`neural_mi.defaults.BASE_PARAMS_SCHEMA`.
"""
import warnings
from typing import Optional, Tuple

import numpy as np

from neural_mi.defaults import default_of
from neural_mi.logger import user_stacklevel

EPOCH_RULES = ('plateau_start', 'peak')


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


def stands_above_zero(test_mi: Optional[float], n_rows: Optional[int],
                      clear_se: float = default_of('clear_se')) -> bool:
    """Whether a network's held-out value, in nats on ``n_rows`` rows, stands clearly above zero: by more than
    ``clear_se`` of its standard errors (:func:`standard_error`), that is above 2 clear_se^2 / n_rows nats. A
    network that learned nothing reads within that of zero. Without a row count, any value above zero counts."""
    if test_mi is None or not np.isfinite(test_mi) or test_mi <= 0:
        return False
    if not n_rows:
        return True
    return bool(test_mi > clear_se * standard_error([test_mi], int(n_rows)))


def choose_epoch(history, smoothed, rule: str, reach: Optional[int],
                 ran_to_limit: bool, n_rows: Optional[int] = None, *,
                 level_noise: float = default_of('level_noise'),
                 start_noise: float = default_of('start_noise'),
                 resolved_noise: float = default_of('resolved_noise'),
                 fast_epochs: int = default_of('fast_epochs'),
                 fall_noise: float = default_of('fall_noise'),
                 clear_se: float = default_of('clear_se'),
                 rising_epochs: int = default_of('rising_epochs'),
                 rising_noise: float = default_of('rising_noise')) -> Tuple[int, int, str]:
    """The epoch to read, the epoch of the smoothed maximum, and the diagnosis of the curve.

    ``history`` is the raw held-out curve and ``smoothed`` the smoothed one, in
    nats. ``reach`` is how many epochs the smoothing looks ahead, None when
    unknown (a custom smoothing function). ``ran_to_limit`` is whether training
    used every epoch it was allowed. ``n_rows`` is the number of held-out rows
    every epoch scores, None when unknown. The diagnosis is one of:

    - ``'ok'``.
    - ``'not resolved'``: the top does not stand clearly above the curve's noise:
      ``resolved_noise`` times its epoch-to-epoch noise, or zero by ``clear_se``
      standard errors of its held-out rows (:func:`stands_above_zero`). Every
      epoch scores the same rows, so their sampling error moves the whole curve
      together and never shows in its epoch-to-epoch noise.
    - ``'still rising'``: training ran to its limit while the curve rose by more
      than ``rising_noise`` noise units over its last ``rising_epochs`` epochs.
    - ``'too fast'``: the curve reached its top within the smoothing's reach of
      the start and fell by more than ``fall_noise`` noise units within
      ``fast_epochs`` epochs after, so the start of the curve placed it.

    The level of the top is the median of the smoothed values within
    ``level_noise`` noise units of the maximum, and the plateau starts at the first
    epoch within ``start_noise`` noise units of that level.

    The trainer adds ``'flat top'`` when the estimate read at the two ends of the
    band differs by more than ``band_share`` of the estimate (see the module
    docstring).
    """
    s = np.asarray(smoothed, dtype=float)
    peak = int(np.nanargmax(s))
    noise = curve_noise(np.asarray(history, dtype=float)[:peak + (reach or 0) + 1])
    level = float(np.median(s[s >= s[peak] - level_noise * noise]))
    start = int(np.flatnonzero(s >= level - start_noise * noise)[0])
    epoch = start if rule == 'plateau_start' else peak
    diagnosis = _diagnose(s, start, level, noise, reach, ran_to_limit, n_rows, resolved_noise=resolved_noise,
                          fast_epochs=fast_epochs, fall_noise=fall_noise, clear_se=clear_se,
                          rising_epochs=rising_epochs, rising_noise=rising_noise)
    return epoch, peak, diagnosis


def band_floor(top: float, n_rows: Optional[int], top_se: float = default_of('top_se')) -> float:
    """The lowest smoothed held-out value, in nats, that lies within ``top_se`` standard errors of a top of
    ``top`` on ``n_rows`` held-out rows (:func:`standard_error`). Without a row count it is the top itself."""
    return top - top_se * (standard_error([top], int(n_rows)) if n_rows else 0.0)


def top_band(smoothed, n_rows: Optional[int], top_se: float = default_of('top_se')) -> Tuple[int, int]:
    """The first and the last epoch of the band: the epochs whose smoothed held-out value, in nats, lies within
    ``top_se`` standard errors of the curve's maximum. The standard error is that of the maximum on ``n_rows``
    held-out rows (:func:`band_floor`). Without a row count the band is the maximum alone.

    The first epoch of the band beats every epoch before it, and the last comes at or after the maximum, so
    the trainer keeps the networks at both (``Trainer.train``)."""
    s = np.asarray(smoothed, dtype=float)
    inside = np.flatnonzero(s >= band_floor(float(np.nanmax(s)), n_rows, top_se))
    return int(inside[0]), int(inside[-1])


def _diagnose(s, start, level, noise, reach, ran_to_limit, n_rows, *, resolved_noise, fast_epochs, fall_noise,
              clear_se, rising_epochs, rising_noise) -> str:
    if n_rows and not stands_above_zero(level, n_rows, clear_se):
        return 'not resolved'
    if not noise > 0:
        return 'ok'
    if level < resolved_noise * noise:
        return 'not resolved'
    if ran_to_limit and s[-1] - s[max(0, len(s) - 1 - rising_epochs)] > rising_noise * noise:
        return 'still rising'
    drop = np.flatnonzero(s[start:] < level - fall_noise * noise)
    if reach is not None and start < reach and len(drop) and drop[0] < fast_epochs:
        return 'too fast'
    return 'ok'


def _values_read(values: Optional[Tuple[float, float, float]], units: str) -> str:
    """The sentence that gives the estimate and the held-out value of the network read, from ``values``:
    the reported estimate, the measured one and the held-out value, in ``units``."""
    if values is None:
        return ""
    reported, measured, held = values
    estimate = (f"{reported:.4f} {units}" if reported == measured
                else f"{reported:.4f} {units} (measured as {measured:.4f} {units})")
    return (f" At the epoch read, the estimate is {estimate} and the held-out MI of the same network is "
            f"{held:.4f} {units}. Only the held-out value is measured on rows that the network did not train on.")


def warn_about_curve(diagnosis: str, epoch: int, peak: int, reach: Optional[int], n_epochs: int,
                     units: str = 'bits', held_out: Optional[int] = None,
                     values: Optional[Tuple[float, float, float]] = None,
                     band: Optional[Tuple[int, int, float, float]] = None, *,
                     resolved_noise: float = default_of('resolved_noise'), clear_se: float = default_of('clear_se'),
                     rising_noise: float = default_of('rising_noise'), rising_epochs: int = default_of('rising_epochs'),
                     fall_noise: float = default_of('fall_noise'), fast_epochs: int = default_of('fast_epochs'),
                     top_se: float = default_of('top_se'), band_share: float = default_of('band_share')) -> None:
    """Warn when the diagnosis is not 'ok'. The texts differ between networks only in their numbers, so the
    library shows each kind once per call and counts the rest.
    ``held_out`` is the count of held-out rows the curve scores, given when it is smaller than one training
    batch (``trainer.too_few_held_out``), and each message then names it. ``values`` are the reported
    estimate, the measured one and the held-out value of the network read, in ``units``, which the 'not
    resolved' and 'flat top' messages state. ``band`` is the band's first and last epoch and the estimate read
    at each, in ``units`` (:func:`top_band`). Each message names the settings that decided its diagnosis."""
    note = (f" The curve scores only {held_out} held-out rows, fewer than one training batch, so its noise is "
            f"large and its ceiling of log({held_out}) low. More data, or a lower Split(train_fraction=...), "
            f"gives it more." if held_out else "")
    if diagnosis == 'not resolved':
        warnings.warn(
            f"The held-out MI curve does not rise clearly above its own noise. Its top stands less than "
            f"Training(resolved_noise={resolved_noise:g}) times its epoch-to-epoch noise above zero, or less than "
            f"Training(clear_se={clear_se:g}) standard errors of its held-out rows, so the data settle neither the "
            f"epoch the estimate is read at nor the estimate. Check the curve with result.plot(). More data, more "
            f"epochs or another network may resolve it.{_values_read(values, units)}{note}",
            UserWarning, stacklevel=user_stacklevel())
    elif diagnosis == 'still rising':
        warnings.warn(
            f"The held-out MI was still rising when training reached n_epochs={n_epochs}. It rose by more than "
            f"Training(rising_noise={rising_noise:g}) noise units over its last Training(rising_epochs="
            f"{rising_epochs}) epochs. The estimate may then be an under-trained lower bound. Raise n_epochs, or "
            f"learning_rate so that each epoch goes further.{note}", UserWarning, stacklevel=user_stacklevel())
    elif diagnosis == 'too fast':
        warnings.warn(
            f"The held-out MI reached its top at epoch {peak} and fell clearly soon after, by more than "
            f"Training(fall_noise={fall_noise:g}) noise units within Training(fast_epochs={fast_epochs}) epochs. "
            f"A top this early lies "
            f"within the {reach} epochs over which the curve is smoothed, so the start of the curve placed it. Lower "
            f"learning_rate to spread the rise over more epochs.{note}", UserWarning, stacklevel=user_stacklevel())
    elif diagnosis == 'flat top':
        first, last, at_first, at_last = band
        warnings.warn(
            f"The held-out MI stays within Training(top_se={top_se:g}) standard errors of its top from epoch "
            f"{first} to epoch {last}, so the held-out rows cannot tell these epochs apart. The estimate is "
            f"{at_first:.4f} {units} when read at the first of these epochs and {at_last:.4f} {units} when read at "
            f"the last. The difference is more than Training(band_share={band_share:g}) of the estimate of "
            f"{values[1]:.4f} {units}, so the held-out rows cannot pin the estimate down more closely. Another "
            f"configuration may do better, or the data may not resolve the estimate more finely. Check the curve "
            f"with result.plot().{note}", UserWarning, stacklevel=user_stacklevel())


def falls_short(estimate: float, held: Optional[float], estimate_se: float, n_rows: Optional[int],
                gap_share: float = default_of('gap_share'), clear_se: float = default_of('clear_se')) -> bool:
    """Whether the held-out value of the network read, ``held``, falls short of its estimate, both in nats:
    below ``gap_share`` of a positive estimate, by more than ``clear_se`` standard errors of their difference.
    The estimate's standard error is ``estimate_se``, and the held-out value's is that of one block of
    ``n_rows`` rows (:func:`standard_error`). The ratio of the two values is free of scale, and the standard
    errors keep a small estimate on few rows from counting a gap that noise alone makes."""
    if held is None or not np.isfinite(held) or not estimate > 0 or held >= gap_share * estimate:
        return False
    held_se = standard_error([max(held, 0.0)], int(n_rows)) if n_rows else 0.0
    return bool(estimate - held > clear_se * float(np.hypot(estimate_se, held_se)))


def warn_about_gap(estimate: float, held: float, units: str = 'bits', gap_share: float = default_of('gap_share'),
                   clear_se: float = default_of('clear_se')) -> None:
    """The warning for a held-out value that falls short of the estimate (:func:`falls_short`), both in
    ``units``."""
    warnings.warn(
        f"At the epoch read, the held-out MI of the network ({held:.4f} {units}) is less than "
        f"Training(gap_share={gap_share:g}) of the estimate ({estimate:.4f} {units}). The difference is more than "
        f"Training(clear_se={clear_se:g}) of its standard errors. The estimate is read on the rows that the "
        f"network trained on, so it exceeds the information when the network fits those rows beyond what holds "
        f"for new ones. The held-out value is a lower bound in expectation. The rest of the estimate may be "
        f"information that the network has not yet learned to generalise, or detail of its training rows that "
        f"it memorised. This run cannot show which. More rows narrow the difference. Check the curves with "
        f"result.plot(). Training(eval_train=True) adds the training-side curve.",
        UserWarning, stacklevel=user_stacklevel())


def warn_about_ceiling(estimate: float, held: Optional[float], train_ceiling: Optional[float],
                       test_ceiling: Optional[float], train_rows: Optional[int], test_rows: Optional[int],
                       held_out_rows: int, units: str = 'bits',
                       ceiling_fraction: float = default_of('ceiling_fraction')) -> Optional[str]:
    """Warn when a value of the network read lies above ``ceiling_fraction`` of the InfoNCE ceiling of its
    side, the log of the rows that each evaluation of that side scores (``train_rows``, ``test_rows``). The
    values and the ceilings are in ``units``, and ``held_out_rows`` counts every held-out row of the split.
    A held-out value near its ceiling means that the true MI may be higher. An estimate near its ceiling while
    the held-out value is not may lie far above the true MI, because a network reads its training rows that
    close to the ceiling by memorising each row as well as by learning the information. Returns the side that
    warned, 'held-out' or 'training', or None."""
    if test_ceiling is not None and held is not None and held > ceiling_fraction * test_ceiling:
        if held_out_rows > test_rows:
            advice = (f"Each evaluation scores {test_rows} of the {held_out_rows} held-out rows. Raising "
                      f"max_eval_samples (and test_subset_size, if set) scores more of them.")
        else:
            advice = "Only more held-out rows raise this ceiling, from more data or a lower Split(train_fraction=...)."
        warnings.warn(
            f"The held-out MI ({held:.4f} {units}) is near its ceiling of log({test_rows}) = {test_ceiling:.4f} "
            f"{units}. InfoNCE can read no more than this when each evaluation scores {test_rows} held-out rows. "
            f"The value is above Training(ceiling_fraction={ceiling_fraction:g}) of the ceiling, so the true MI may "
            f"be higher. {advice} estimator='smile' has no log(n) ceiling, but its clip bounds its estimate "
            f"instead.",
            UserWarning, stacklevel=user_stacklevel())
        return 'held-out'
    if train_ceiling is not None and estimate > ceiling_fraction * train_ceiling:
        held_text = f", while the held-out MI of the same network is {held:.4f} {units}" if held is not None else ""
        warnings.warn(
            f"The estimate ({estimate:.4f} {units}) is near its ceiling of log({train_rows}) = "
            f"{train_ceiling:.4f} {units}{held_text}. InfoNCE can read no more than this ceiling when each "
            f"evaluation scores {train_rows} training rows. The estimate is above Training(ceiling_fraction="
            f"{ceiling_fraction:g}) of it. A network can read its training rows this close to the ceiling by "
            f"memorising each row as well as by learning the information, so the estimate may lie far above the "
            f"true MI. The held-out value is a lower bound in expectation. Check the curves with result.plot(). "
            f"Training(eval_train=True) adds the training-side curve.",
            UserWarning, stacklevel=user_stacklevel())
        return 'training'
    return None
