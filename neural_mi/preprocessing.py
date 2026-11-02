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
import warnings
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from scipy.signal import lfilter
from scipy.stats import norm

from neural_mi.logger import logger, user_stacklevel

# The rate quantities: their value is unchanged when each process is filtered invertibly by its own past.
RATE_QUANTITIES = ('mi_rate', 'transfer_entropy', 'conditional_transfer_entropy',
                   'directed_information_rate', 'instantaneous_exchange')
# Channels count as being far from unit scale beyond this factor either way.
SCALE_BAND = 10.0
# The share of a channel's variance its own past predicts linearly above which a whitening is suggested.
PREDICTED_SHARE = 0.9


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


def _spread(std: np.ndarray) -> np.ndarray:
    """The standard deviations to divide by: a channel is constant when its spread is no more than 1e-8 of
    the largest (exactly zero when every channel is), and keeps its scale instead of being divided by it."""
    floor = 1e-8 * float(std.max()) if std.size and std.max() > 0 else 0.0
    return np.where(std > floor, std, 1.0)


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


def check_streams(streams: Dict[str, Any], processors: Dict[str, Optional[str]], *, quantity: str,
                  rate: bool, preprocessing=None) -> None:
    """Report what the streams of a call look like before any network sees them, and change nothing.

    For each regularly sampled stream or array of rows: its channels, their
    scales and, for a rate quantity on a series, how much of each channel's
    variance its own past predicts linearly. It warns about non-finite and
    constant channels and about channels far from unit scale, and suggests the
    ``Preprocessing`` option that addresses each. The report itself is
    logged at the info level.
    """
    asked = describe(preprocessing) or {}
    parts = []
    for name, data in streams.items():
        if data is None:
            continue
        kind = stream_kind(data, processors.get(name))
        if kind in ('spike', 'categorical'):
            parts.append(f"{name}: {kind}, rows built by its processor")
            continue
        columns, _ = _channels_last(_as_array(data))
        finite = np.isfinite(columns).all(axis=0)
        if not finite.all():
            bad = np.flatnonzero(~finite).tolist()
            warnings.warn(
                f"{name} has non-finite values (NaN or infinite) in channel(s) {bad}. Every network that "
                f"reads them produces NaN. Remove or fill them before the call.",
                UserWarning, stacklevel=user_stacklevel())
            parts.append(f"{name}: {columns.shape[1]} channels, {len(bad)} with non-finite values")
            continue
        std = columns.std(axis=0)
        constant = np.flatnonzero(std <= (1e-8 * std.max() if std.max() > 0 else 0.0)).tolist()
        if constant:
            warnings.warn(
                f"{name} has constant channel(s) {constant}. They carry no information and, under "
                f"Preprocessing(channel='zca'), no direction to sphere. Drop them before the call.",
                UserWarning, stacklevel=user_stacklevel())
        live = np.delete(std, constant)
        line = f"{name}: {columns.shape[1]} channels"
        if live.size:
            typical = float(np.median(live))
            line += f", scale {_fmt(live.min())} to {_fmt(live.max())}"
            if not (1.0 / SCALE_BAND <= typical <= SCALE_BAND) and 'channel' not in asked:
                warnings.warn(
                    f"The channels of {name} have a typical scale of {_fmt(typical)}, more than a factor "
                    f"of {SCALE_BAND:g} from unit scale. The networks start from weights sized for inputs "
                    f"near unit scale, and the estimate depends on the units: a dependence of 2 nats reads "
                    f"near 0 at a thousandth or a thousand times its scale. Preprocessing(channel='zscore') "
                    f"brings every channel to unit scale and leaves every quantity unchanged.",
                    UserWarning, stacklevel=user_stacklevel())
        if rate and kind == 'series' and live.size and len(columns) > 2:
            ar = ARWhitening().fit(columns[:, [j for j in range(columns.shape[1]) if j not in constant]])
            share = max(ar.predicted_)
            line += f", own past predicts up to {share:.1%} of a channel's variance"
            if share > PREDICTED_SHARE and 'time' not in asked:
                line += (" (a large conditioning term, of which the rate is a small residual; "
                         "Preprocessing(time='ar') removes it and leaves the rate unchanged)")
        parts.append(line)
    if parts:
        logger.info(f"Data check for {quantity}: " + "; ".join(parts) + ".")


def prepare(streams: Dict[str, Any], processors: Dict[str, Optional[str]], preprocessing, *, quantity: str,
            rate: bool, check: bool = True) -> 'OrderedDict[str, Any]':
    """The streams of a call transformed as ``preprocessing`` asks (:func:`apply_preprocessing`), and
    checked as the networks will receive them (:func:`check_streams`): the one step every path of a call
    takes before rows are built. ``check=False`` leaves the check to a caller that ran it already."""
    out = apply_preprocessing(streams, processors, preprocessing, quantity=quantity, rate=rate)
    if check:
        check_streams(out, processors, quantity=quantity, rate=rate, preprocessing=preprocessing)
    return out
