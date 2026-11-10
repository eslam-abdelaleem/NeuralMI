# neural_mi/preprocessing.py
"""Changes to the streams before rows are built, made only when asked, and a check of the data that changes nothing.

``Preprocessing(channel=..., time=...)`` transforms each continuous stream (X,
Y and W apart) before the library builds rows from it. A channel transform acts
across a stream's channels at each time point, a time transform along each
channel's time axis. The built-in ones are invertible, so they leave the
information a stream carries in place:

- channel: :class:`ZScore`, :class:`ZCA`, :class:`PCA` and :class:`Copula`,
  which leave every quantity unchanged (``PCA(n_components=k)`` below the
  channel count drops components, and warns);
- time: :class:`ARWhitening`, which leaves the rate quantities unchanged in the
  limit of long windows and changes every other quantity, which warns.

A transform of your own subclasses :class:`Transform`: it declares its
``domain``, learns from a stream in ``fit`` and applies in ``transform``. The
library cannot check that it preserves information and says so.

:func:`check_streams` reports what the streams look like before any network
sees them, and warns about what needs attention: non-finite or constant
channels, and channels far from unit scale. It suggests the matching option and
changes nothing.
"""
import copy
import math
import numbers
import warnings
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from scipy.signal import lfilter
from scipy.stats import norm

from neural_mi.defaults import BASE_PARAMS_SCHEMA, CHECK_SETTINGS, setting
from neural_mi.logger import logger, user_stacklevel

# The rate quantities: their value is unchanged when each process is filtered invertibly by its own past.
RATE_QUANTITIES = ('mi_rate', 'transfer_entropy', 'conditional_transfer_entropy',
                   'directed_information_rate', 'instantaneous_exchange')


def check_settings(preprocessing) -> Dict[str, Any]:
    """The settings of the data check (:data:`neural_mi.defaults.CHECK_SETTINGS`) that ``preprocessing``, a
    ``Preprocessing`` or a dict, sets. :func:`neural_mi.defaults.setting` reads each, or its default."""
    given = preprocessing.to_kwargs() if hasattr(preprocessing, 'to_kwargs') else dict(preprocessing or {})
    checks = {name: value for name, value in given.items() if name in CHECK_SETTINGS and value is not None}
    # Checked here, where every path first reads them, so a named quantity refuses a bad value before its
    # own data check runs, as run() does.
    for name, value in checks.items():
        schema = BASE_PARAMS_SCHEMA[name]
        kind = numbers.Integral if schema['type'] is int else numbers.Real
        if isinstance(value, bool) or not isinstance(value, kind) or value < schema['min']:
            raise ValueError(f"Preprocessing({name}={value!r}) must be {'a whole number' if kind is numbers.Integral else 'a number'} "
                             f"of at least {schema['min']:g}.")
    return checks


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

class Transform:
    """A change to one stream, learned from that stream and applied to it.

    ``domain`` is ``'channel'``, a transform across channels at each time
    point, or ``'time'``, a transform along each channel's time axis. ``fit``
    and ``transform`` take a 2-D array of shape ``(n, n_channels)``: a series,
    or for a channel transform every time point of every row of a stream passed
    in rows. The library fits a fresh copy on each stream. ``invertible`` says
    whether the transform keeps the information a stream carries; the library
    believes it only of its own transforms.
    """
    domain = 'channel'
    invertible = False

    def fit(self, data: np.ndarray) -> 'Transform':
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @property
    def label(self) -> str:
        """The name ``result.params`` records."""
        return type(self).__name__


# A channel counts as constant when its standard deviation is no more than this share of the largest of its
# stream, a numerical tolerance well below any variation that a recording in any unit carries.
CONSTANT_SHARE = 1e-8


def constant_channels(std: np.ndarray) -> np.ndarray:
    """Which channels are constant: those whose spread is no more than :data:`CONSTANT_SHARE` of the largest,
    and every channel when none varies."""
    std = np.asarray(std, dtype=float)
    floor = CONSTANT_SHARE * float(std.max()) if std.size and std.max() > 0 else 0.0
    return std <= floor


def _spread(std: np.ndarray) -> np.ndarray:
    """The standard deviations to divide by: a constant channel (:func:`constant_channels`) keeps its scale
    instead of being divided by it."""
    return np.where(constant_channels(std), 1.0, std)


class ZScore(Transform):
    """Each channel centred and divided by its standard deviation. A constant channel is centred only."""
    invertible = True
    label = 'zscore'

    def fit(self, data):
        self.mean_ = data.mean(axis=0)
        self.scale_ = _spread(data.std(axis=0))
        return self

    def transform(self, data):
        return (data - self.mean_) / self.scale_


class ZCA(Transform):
    """The channels sphered: centred and multiplied by the inverse square root of their covariance, which
    keeps them as close to the originals as sphering can. Directions of no variance are left unscaled."""
    invertible = True
    label = 'zca'

    def fit(self, data):
        self.mean_ = data.mean(axis=0)
        centred = data - self.mean_
        cov = centred.T @ centred / max(len(data) - 1, 1)
        u, s, _ = np.linalg.svd(cov)
        s = np.maximum(s, 1e-8 * float(s.max()) if s.size and s.max() > 0 else 1e-8)
        self.matrix_ = u @ np.diag(1.0 / np.sqrt(s)) @ u.T
        return self

    def transform(self, data):
        return (data - self.mean_) @ self.matrix_.T


class PCA(Transform):
    """The channels rotated onto their principal axes, largest variance first, after centring.

    With ``n_components`` below the channel count only the first components are
    kept. A cut by variance can drop the information, which need not lie in the
    directions of largest variance, so the library warns.
    """
    invertible = True

    def __init__(self, n_components: Optional[int] = None):
        if n_components is not None and (isinstance(n_components, bool) or not isinstance(n_components, int)
                                         or n_components < 1):
            raise ValueError(f"PCA(n_components=...) takes a whole number of components from 1, "
                             f"got {n_components!r}.")
        self.n_components = n_components

    @property
    def label(self) -> str:
        return 'pca' if self.n_components is None else f'pca(n_components={self.n_components})'

    def fit(self, data):
        self.mean_ = data.mean(axis=0)
        centred = data - self.mean_
        cov = centred.T @ centred / max(len(data) - 1, 1)
        u, _, _ = np.linalg.svd(cov)
        k = u.shape[1] if self.n_components is None else min(self.n_components, u.shape[1])
        self.components_ = u[:, :k]
        self.invertible = k == u.shape[1]
        return self

    def transform(self, data):
        return (data - self.mean_) @ self.components_


class Copula(Transform):
    """Each channel mapped through its own empirical distribution to a standard normal (its ranks, mid-ranks
    for ties). The map is monotone in every channel, so it keeps the information and removes the scale and
    the shape of each channel's distribution."""
    invertible = True
    label = 'copula'

    def fit(self, data):
        self.sorted_ = np.sort(data, axis=0)
        return self

    def transform(self, data):
        n = len(self.sorted_)
        out = np.empty_like(data, dtype=float)
        for j in range(data.shape[1]):
            below = np.searchsorted(self.sorted_[:, j], data[:, j], side='left')
            through = np.searchsorted(self.sorted_[:, j], data[:, j], side='right')
            rank = below + (through - below + 1) / 2.0
            out[:, j] = norm.ppf(np.clip((rank - 0.5) / n, 0.5 / n, 1.0 - 0.5 / n))
        return out


def _autocovariance(x: np.ndarray, max_lag: int) -> np.ndarray:
    """The biased autocovariance of a centred series at lags 0 to ``max_lag``, by FFT."""
    n = len(x)
    size = 1 << (2 * n - 1).bit_length()
    spectrum = np.fft.rfft(x, size)
    return np.fft.irfft(spectrum * np.conj(spectrum), size)[:max_lag + 1] / n


def _levinson(r: np.ndarray) -> Tuple[list, np.ndarray]:
    """The autoregressive fits of every order to ``len(r) - 1`` from an autocovariance: the coefficients of
    each order and the variance each leaves unpredicted (Levinson-Durbin)."""
    max_order = len(r) - 1
    coefficients, errors = [np.zeros(0)], [r[0]]
    a = np.zeros(0)
    error = r[0]
    for p in range(1, max_order + 1):
        if error <= 0:
            break
        k = (r[p] - np.dot(a, r[p - 1:0:-1])) / error
        a = np.concatenate([a - k * a[::-1], [k]])
        error = error * (1.0 - k * k)
        coefficients.append(a)
        errors.append(error)
    return coefficients, np.asarray(errors)


def max_ar_order(n: int) -> int:
    """The largest order the AIC search considers for a series of ``n`` samples: ``10 log10(n)``, the default
    of R's ``ar``, and below ``n``."""
    return max(1, min(n - 1, int(10 * math.log10(max(n, 2)))))


class ARWhitening(Transform):
    """Each channel filtered by its own autoregressive fit, leaving what its past does not predict.

    The order is chosen per channel by AIC among the fits of Levinson-Durbin up
    to :func:`max_ar_order`, unless ``order`` fixes it. The filter starts from
    rest, so the series keeps its length and every stream stays aligned with
    the others and with its clock. Its first ``order`` samples are predicted
    from a partial past.

    Filtering each process invertibly by its own past leaves the rate
    quantities unchanged in the limit of long windows (``mi_rate``, transfer
    entropy, conditional transfer entropy, the directed information rate,
    instantaneous exchange). Every other quantity changes.
    """
    domain = 'time'
    invertible = True

    def __init__(self, order: Optional[int] = None):
        if order is not None and (isinstance(order, bool) or not isinstance(order, int) or order < 0):
            raise ValueError(f"ARWhitening(order=...) takes a whole number from 0, got {order!r}.")
        self.order = order

    @property
    def label(self) -> str:
        return 'ar' if self.order is None else f'ar(order={self.order})'

    def fit(self, data):
        self.mean_ = data.mean(axis=0)
        n = len(data)
        cap = max_ar_order(n) if self.order is None else min(self.order, n - 1)
        self.coefficients_, self.predicted_ = [], []
        for j in range(data.shape[1]):
            r = _autocovariance(data[:, j] - self.mean_[j], cap)
            fits, errors = _levinson(r)
            if r[0] <= 0:
                chosen = 0
            elif self.order is None:
                aic = n * np.log(np.maximum(errors, 1e-300)) + 2 * np.arange(len(errors))
                chosen = int(np.argmin(aic))
            else:
                chosen = len(fits) - 1
            self.coefficients_.append(fits[chosen])
            self.predicted_.append(float(1.0 - errors[chosen] / r[0]) if r[0] > 0 else 0.0)
        return self

    def transform(self, data):
        out = np.empty_like(data, dtype=float)
        for j, a in enumerate(self.coefficients_):
            out[:, j] = lfilter(np.r_[1.0, -a], [1.0], data[:, j] - self.mean_[j])
        return out


CHANNEL_TRANSFORMS = {'zscore': ZScore, 'zca': ZCA, 'pca': PCA, 'copula': Copula}
TIME_TRANSFORMS = {'ar': ARWhitening}


def _resolve(value, table: Dict[str, type], domain: str) -> Optional[Transform]:
    if value is None:
        return None
    if isinstance(value, str):
        if value not in table:
            raise ValueError(f"Preprocessing({domain}={value!r}) is not a {domain} transform. Use one of "
                             f"{sorted(table)}, or a Transform of your own.")
        return table[value]()
    if isinstance(value, type) and issubclass(value, Transform):
        value = value()
    if not isinstance(value, Transform):
        raise TypeError(f"Preprocessing({domain}=...) takes a name or a Transform, got {type(value).__name__}.")
    if value.domain != domain:
        raise ValueError(f"Preprocessing({domain}=...) got {value.label}, whose domain is {value.domain!r}. "
                         f"Pass it as Preprocessing({value.domain}=...).")
    return value


def resolve(preprocessing) -> Dict[str, Optional[Transform]]:
    """The transforms a ``Preprocessing`` asks for, by domain: ``{'time': ..., 'channel': ...}``."""
    settings = preprocessing.to_kwargs() if hasattr(preprocessing, 'to_kwargs') else dict(preprocessing or {})
    return {'time': _resolve(settings.get('time'), TIME_TRANSFORMS, 'time'),
            'channel': _resolve(settings.get('channel'), CHANNEL_TRANSFORMS, 'channel')}


def describe(preprocessing) -> Optional[Dict[str, str]]:
    """What ``result.params`` records of a ``Preprocessing``: the label of each transform it applies."""
    if preprocessing is None:
        return None
    transforms = resolve(preprocessing)
    return {domain: t.label for domain, t in transforms.items() if t is not None} or None


# ---------------------------------------------------------------------------
# Applying them to the streams of a call
# ---------------------------------------------------------------------------

def stream_kind(data, processor: Optional[str]) -> str:
    """``'spike'``, ``'categorical'``, ``'series'`` (a regularly sampled ``(T, C)`` stream) or ``'rows'``
    (an array of rows built already, three or more dimensions)."""
    if processor in ('spike', 'categorical'):
        return processor
    if isinstance(data, list):
        return 'spike'
    return 'series' if np.ndim(data) <= 2 else 'rows'


def _as_array(data) -> np.ndarray:
    if torch.is_tensor(data):
        return data.detach().cpu().numpy().astype(float)
    return np.asarray(data, dtype=float)


def _like(original, array: np.ndarray):
    """``array`` in the type and dtype of ``original``."""
    if torch.is_tensor(original):
        return torch.as_tensor(array, dtype=original.dtype)
    if isinstance(original, np.ndarray) and np.issubdtype(original.dtype, np.floating):
        return array.astype(original.dtype)
    return array


def _channels_last(array: np.ndarray) -> Tuple[np.ndarray, tuple]:
    """A stream's channels as columns: a series as it is, rows with every time point of every row a sample."""
    if array.ndim == 1:
        return array[:, None], (array.shape, None)
    if array.ndim == 2:
        return array, (array.shape, None)
    moved = np.moveaxis(array, 1, -1)
    return moved.reshape(-1, array.shape[1]), (array.shape, moved.shape)


def _restore(columns: np.ndarray, layout: tuple) -> np.ndarray:
    shape, moved = layout
    if moved is None:
        return columns.reshape(shape[0], -1) if len(shape) == 2 else columns[:, 0] if columns.shape[1] == 1 \
            else columns
    return np.moveaxis(columns.reshape(*moved[:-1], columns.shape[1]), -1, 1)


def apply_preprocessing(streams: Dict[str, Any], processors: Dict[str, Optional[str]], preprocessing, *,
                        quantity: str, rate: bool) -> 'OrderedDict[str, Any]':
    """The streams of a call, each transformed as ``preprocessing`` asks before rows are built from it.

    ``processors`` names each stream's processor (``None`` for a regularly
    sampled series or rows passed in). Each transform is fitted on one stream
    at a time, the time transform first and the channel transform on what it
    leaves. A spike-time or categorical stream takes neither, and rows passed in
    built take the channel transform only; both say so. ``quantity`` names the
    call in messages and ``rate`` says whether it is a rate quantity, which a
    time transform leaves unchanged; any other quantity warns.
    """
    out = OrderedDict(streams)
    if preprocessing is None:
        return out
    transforms = resolve(preprocessing)
    if transforms['time'] is None and transforms['channel'] is None:
        return out
    for t in transforms.values():
        if t is not None and type(t) not in (*CHANNEL_TRANSFORMS.values(), *TIME_TRANSFORMS.values()):
            warnings.warn(
                f"Preprocessing applies {t.label}, a transform of your own. The library cannot check that "
                f"it keeps the information the streams carry: an invertible transform does, and one that "
                f"drops, merges or clips values does not.",
                UserWarning, stacklevel=user_stacklevel())
    if transforms['time'] is not None and not rate:
        warnings.warn(
            f"Preprocessing(time={transforms['time'].label!r}) filters each channel by its own past, which "
            f"changes what {quantity} measures. Only the rate quantities ({', '.join(RATE_QUANTITIES)}) "
            f"are unchanged by it, in the limit of long windows.",
            UserWarning, stacklevel=user_stacklevel())
    for name, data in streams.items():
        if data is None:
            continue
        kind = stream_kind(data, processors.get(name))
        if kind in ('spike', 'categorical'):
            settings = ("its bin_size and normalize_bins" if kind == 'spike' else "its encoding")
            warnings.warn(
                f"Preprocessing leaves {name}, a {kind} stream, as it is: its rows come from its "
                f"processor, whose own settings ({settings}) set their values. Model(input_scaling="
                f"'standard') scales the rows an encoder receives.",
                UserWarning, stacklevel=user_stacklevel())
            continue
        array = _as_array(data)
        if kind == 'rows' and transforms['time'] is not None:
            warnings.warn(
                f"Preprocessing(time=...) leaves {name} as it is in time: it arrived as rows built "
                f"already, {tuple(array.shape)}, and has no single series to filter. Pass the series, "
                f"and let a processor or a named quantity build the rows.",
                UserWarning, stacklevel=user_stacklevel())
        columns, layout = _channels_last(array)
        if kind == 'series' and transforms['time'] is not None:
            columns = copy.deepcopy(transforms['time']).fit(columns).transform(columns)
        channel = transforms['channel']
        if channel is not None:
            fitted = copy.deepcopy(channel).fit(columns)
            if isinstance(fitted, PCA) and not fitted.invertible:
                warnings.warn(
                    f"PCA keeps {fitted.components_.shape[1]} of the {columns.shape[1]} components of "
                    f"{name}, those of largest variance. The information need not lie in them: a signal "
                    f"shared under larger private ones is cut first. n_components=None keeps every "
                    f"component and leaves every quantity unchanged.",
                    UserWarning, stacklevel=user_stacklevel())
            columns = fitted.transform(columns)
        out[name] = _like(data, _restore(columns, layout))
    return out


# ---------------------------------------------------------------------------
# The check
# ---------------------------------------------------------------------------

def _fmt(value: float) -> str:
    return f"{value:.3g}"


def _channel_view(chunk) -> np.ndarray:
    """A stretch of a stream with its channels as columns, without converting its type."""
    array = chunk.detach().cpu().numpy() if torch.is_tensor(chunk) else np.asarray(chunk)
    if array.ndim == 1:
        return array[:, None]
    if array.ndim == 2:
        return array
    return np.moveaxis(array, 1, -1).reshape(-1, array.shape[1])


def _channel_moments(data, rows: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Whether every value of each channel is finite, and each channel's mean and standard deviation over
    every row, read in stretches of ``rows`` rows whose moments combine exactly (Chan et al.), so no more
    than one stretch is copied at a time."""
    finite, n, mean, m2 = None, 0, None, None
    for start in range(0, len(data), rows):
        block = _channel_view(data[start:start + rows]).astype(float)
        ok = np.isfinite(block).all(axis=0)
        finite = ok if finite is None else finite & ok
        count, block_mean = len(block), block.mean(axis=0)
        block_m2 = ((block - block_mean) ** 2).sum(axis=0)
        if mean is None:
            n, mean, m2 = count, block_mean, block_m2
            continue
        delta, total = block_mean - mean, n + count
        mean = mean + delta * count / total
        m2 = m2 + block_m2 + delta ** 2 * n * count / total
        n = total
    return finite, mean, np.sqrt(m2 / max(n, 1))


def _shapes(columns: np.ndarray, std: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """For each column of ``columns``, the share of its values equal to its most common value, and its
    standard deviation ``std`` in interquartile ranges (infinite when the quartiles coincide)."""
    ordered = np.sort(columns, axis=0)
    tied = np.empty(ordered.shape[1])
    for j in range(ordered.shape[1]):
        starts = np.flatnonzero(np.r_[True, ordered[1:, j] != ordered[:-1, j], True])
        tied[j] = np.diff(starts).max() / len(ordered)
    q25, q75 = np.quantile(ordered, [0.25, 0.75], axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(q75 > q25, std / (q75 - q25), np.inf)
    return tied, ratio


def _channels(indices) -> str:
    return f"channel {indices[0]}" if len(indices) == 1 else f"channels {indices}"


def check_streams(streams: Dict[str, Any], processors: Dict[str, Optional[str]], *, quantity: str,
                  rate: bool, preprocessing=None, encoder_scales=False) -> None:
    """Report what the streams of a call look like before any network sees them, and change nothing.

    For each regularly sampled stream or array of rows: its channels, their
    scales, whether its encoder scales it and, for a rate quantity on a series,
    how much of each channel's variance its own past predicts linearly. The
    report is logged at the info level. It warns about non-finite and constant
    channels; about channels far from unit scale or zero when the encoder leaves
    them as they are; about channel scales that spread widely, means so far
    from zero that single precision resolves the variation coarsely, and heavy
    tails; and, when the encoder scales the stream, about channels that are
    mostly one repeated value, as padded or sparse rows are. Each warning names
    the option that addresses it.

    ``encoder_scales`` says whether the encoders scale their own inputs
    (``Model(input_scaling=...)``): one answer for every stream, or a stream's
    name to its answer. The check reads every row in stretches of
    ``check_rows`` rows, so it never copies a large stream, fits the own past on
    the first ``check_rows`` rows, and reads quartiles and repeated values from
    the first ``sample_rows``. Its thresholds come from ``preprocessing``
    (:func:`check_settings`).
    """
    asked = describe(preprocessing) or {}
    checks = check_settings(preprocessing)
    scale_band, predicted_share = setting(checks, 'scale_band'), setting(checks, 'predicted_share')
    rows, sample_rows = setting(checks, 'check_rows'), setting(checks, 'sample_rows')
    spread_band, offset_band = setting(checks, 'spread_band'), setting(checks, 'offset_band')
    tail_band, tie_share = setting(checks, 'tail_band'), setting(checks, 'tie_share')
    parts, any_scaled = [], False
    for name, data in streams.items():
        if data is None:
            continue
        kind = stream_kind(data, processors.get(name))
        if kind in ('spike', 'categorical'):
            parts.append(f"{name}: {kind}, rows built by its processor")
            continue
        scaled = bool(encoder_scales.get(name, False) if isinstance(encoder_scales, dict) else encoder_scales)
        finite, mean, std = _channel_moments(data, rows)
        if not finite.all():
            bad = np.flatnonzero(~finite).tolist()
            warnings.warn(
                f"{name} has non-finite values (NaN or infinite) in channel(s) {bad}. Every network that "
                f"reads them produces NaN. Remove or fill them before the call.",
                UserWarning, stacklevel=user_stacklevel())
            parts.append(f"{name}: {len(finite)} channels, {len(bad)} with non-finite values")
            continue
        constant = np.flatnonzero(constant_channels(std)).tolist()
        if constant:
            warnings.warn(
                f"{name} has constant channel(s) {constant}. They carry no information and, under "
                f"Preprocessing(channel='zca'), no direction to sphere. Drop them before the call.",
                UserWarning, stacklevel=user_stacklevel())
        live_index = np.setdiff1d(np.arange(len(std)), constant)
        live, live_mean = std[live_index], mean[live_index]
        line = f"{name}: {len(std)} channels"
        if live.size:
            typical, typical_mean = float(np.median(live)), float(np.median(np.abs(live_mean)))
            line += f", scale {_fmt(live.min())} to {_fmt(live.max())}"
            if scaled:
                any_scaled = True
                line += ", centred and scaled channel by channel by its encoder"
            elif 'channel' not in asked:
                _warn_unit_scale(name, typical, typical_mean, scale_band)
            spread = float(live.max() / live.min())
            if spread > spread_band:
                _warn_spread(name, spread, spread_band, scaled)
            far = live_index[np.abs(live_mean) > offset_band * live].tolist()
            if far:
                warnings.warn(
                    f"{_channels(far).capitalize()} of {name} sit more than Preprocessing(offset_band="
                    f"{offset_band:g}) of their standard deviations from zero. The networks read the rows in "
                    f"single precision, which resolves the variation of such a channel coarsely, and no scaling "
                    f"inside a network recovers what is lost. Subtract each channel's mean before the call, "
                    f"while the data are in double precision, or set Preprocessing(channel='zscore'), which "
                    f"centres each channel before the rows are built.",
                    UserWarning, stacklevel=user_stacklevel())
            tied, ratio = _shapes(_channel_view(data[:sample_rows]).astype(float)[:, live_index], live)
            repeated = live_index[tied > tie_share].tolist()
            if repeated and scaled:
                warnings.warn(
                    f"In {_channels(repeated)} of {name}, more than Preprocessing(tie_share={tie_share:g}) of the "
                    f"values are one repeated value, as in padded or sparse rows such as spike rows built by "
                    f"hand. The encoder scales each channel to unit scale by default, which moves the repeated "
                    f"value away from 0 and lifts a channel that is quiet because it carries little to the level "
                    f"of the others. Model(input_scaling='none') leaves the rows as they are. Spike times read by "
                    f"a spike processor, such as Processing(x='spike') for X, are left as they are under the "
                    f"default as well.",
                    UserWarning, stacklevel=user_stacklevel())
            heavy = live_index[(ratio > tail_band) & np.isfinite(ratio) & (tied <= tie_share)].tolist()
            if heavy:
                warnings.warn(
                    f"{_channels(heavy).capitalize()} of {name} have heavy tails. In each, the standard deviation "
                    f"exceeds Preprocessing(tail_band={tail_band:g}) interquartile ranges. Scaling divides by the "
                    f"standard deviation, which the extreme values set, so most values of such a channel end up "
                    f"close together. Preprocessing(channel='copula') maps each channel through its ranks to a "
                    f"normal distribution before the rows are built, which leaves every quantity unchanged.",
                    UserWarning, stacklevel=user_stacklevel())
        if rate and kind == 'series' and live.size and len(data) > 2:
            columns = _channel_view(data[:rows]).astype(float)
            ar = ARWhitening().fit(columns[:, [j for j in range(columns.shape[1]) if j not in constant]])
            share = max(ar.predicted_)
            line += f", own past predicts up to {share:.1%} of a channel's variance"
            if share > predicted_share and 'time' not in asked:
                line += (" (a large conditioning term, of which the rate is a small residual; "
                         "Preprocessing(time='ar') removes it and leaves the rate unchanged)")
        parts.append(line)
    if parts:
        logger.info(f"Data check for {quantity}: " + "; ".join(parts) + "."
                    + (" Model(input_scaling='none') turns the encoders' scaling off." if any_scaled else ""))


def _warn_unit_scale(name: str, typical: float, typical_mean: float, scale_band: float) -> None:
    """The warning for a stream that its encoder leaves as it is while its channels sit far from unit scale,
    or far from zero."""
    remedy = ("Model(input_scaling='auto'), the default, centres and scales each channel of a continuous stream "
              "inside the network, and Preprocessing(channel='zscore') does so before the rows are built. Both "
              "leave every quantity unchanged.")
    if not 1.0 / scale_band <= typical <= scale_band:
        warnings.warn(
            f"The channels of {name} have a typical scale of {_fmt(typical)}, more than a factor of "
            f"Preprocessing(scale_band={scale_band:g}) from unit scale. The networks start from weights sized for "
            f"inputs near unit scale, so the estimate depends on the units. The same dependence can read close to "
            f"its value at unit scale and close to 0 far from it. {remedy}",
            UserWarning, stacklevel=user_stacklevel())
    elif typical_mean > scale_band:
        warnings.warn(
            f"The channels of {name} have a typical mean of {_fmt(typical_mean)}, more than "
            f"Preprocessing(scale_band={scale_band:g}) from zero. The networks start from weights sized for inputs "
            f"near zero, so an offset this large can hide the variation that carries the information, and the "
            f"estimate can read far below it. {remedy}",
            UserWarning, stacklevel=user_stacklevel())


def _warn_spread(name: str, spread: float, spread_band: float, scaled: bool) -> None:
    """The warning for channel scales that spread widely within a stream, which reads differently whether
    the encoder scales each channel or leaves them as they are."""
    if scaled:
        body = ("Its encoder scales each channel to unit scale, which gives every channel the same weight. That "
                "is right when the scales differ only by units. When a channel is quiet because it carries "
                "little, scaling lifts its noise to the level of the signal, and the network needs more rows to "
                "learn to ignore it. Model(input_scaling='none') keeps the channels' relative scales, and "
                "dividing the stream by one number before the call, such as its overall standard deviation, "
                "then brings it near unit scale.")
    else:
        body = ("The encoders leave the channels as they are, so the loudest ones dominate what the networks "
                "receive, and the estimate can miss information that the quiet ones carry. "
                "Model(input_scaling='auto'), the default, scales each channel to unit scale inside the network.")
    warnings.warn(
        f"The standard deviations of the channels of {name} span a factor of {_fmt(spread)}, more than "
        f"Preprocessing(spread_band={spread_band:g}). {body}", UserWarning, stacklevel=user_stacklevel())


def prepare(streams: Dict[str, Any], processors: Dict[str, Optional[str]], preprocessing, *, quantity: str,
            rate: bool, check: bool = True, encoder_scales=False) -> 'OrderedDict[str, Any]':
    """The streams of a call transformed as ``preprocessing`` asks (:func:`apply_preprocessing`), and
    checked as the networks will receive them (:func:`check_streams`, which reads ``encoder_scales``): the
    one step every path of a call takes before rows are built. ``check=False`` leaves the check to a caller
    that ran it already."""
    out = apply_preprocessing(streams, processors, preprocessing, quantity=quantity, rate=rate)
    if check:
        check_streams(out, processors, quantity=quantity, rate=rate, preprocessing=preprocessing,
                      encoder_scales=encoder_scales)
    return out
