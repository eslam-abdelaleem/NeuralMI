# neural_mi/utils.py

import inspect
import warnings

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as _lr_sched
import torch.nn as nn
from collections import OrderedDict
from typing import Dict, Any, Optional, Tuple
import numpy as np

import multiprocessing as mp
import os
import platform
import tempfile

from neural_mi.models.embeddings import (
    DeepSets,
    MLP, VariationalWrapper, ChannelScaling, InputScaling,
    CNN1D, CNN2D, GRU, LSTM, TCN, Transformer,
    PretrainedBackboneEmbedding, LRUEmbedding, BranchedEmbedding,
)
from neural_mi.models.critics import SeparableCritic, ConcatCritic, BaseCritic, HybridCritic
from neural_mi.defaults import setting
from neural_mi.logger import logger, user_stacklevel

# Every built-in embedding_model string -> class.
_EMBEDDING_CLASSES = {
    'mlp': MLP,
    'cnn': CNN1D,
    'cnn2d': CNN2D,
    'gru': GRU,
    'lstm': LSTM,
    'tcn': TCN,
    'transformer': Transformer,
    'pretrained_backbone': PretrainedBackboneEmbedding,
    'lru': LRUEmbedding,
    'deepsets': DeepSets,
}

def _ensure_cpu(data):
    """Move *data* to CPU if it is a tensor on a non-CPU device.

    Multiprocessing workers receive data via pickle (spawn context), so all tensors must reside on CPU. CUDA and MPS shared-memory mechanisms are not available across process boundaries.  Call this on
    every tensor before adding it to a Pool task tuple.

    Non-tensor inputs (numpy arrays, None, etc.) are returned unchanged.
    A tuple (BranchedEmbedding's compound "X-role" data) is mapped over
    element-wise, so a GPU/MPS-resident element inside it actually gets
    moved before crossing the spawn boundary too.
    """
    if isinstance(data, tuple):
        return tuple(_ensure_cpu(d) for d in data)
    if isinstance(data, torch.Tensor) and data.device.type != 'cpu':
        return data.cpu()
    return data


def get_device(device_str: Optional[str] = None) -> torch.device:
    """The device named, or the CPU when none is. Which device a call trains on when it
    names none is chosen before its tasks run (:mod:`neural_mi.device`)."""
    return torch.device(device_str or 'cpu')

_mp_configured = False

def _configure_multiprocessing() -> None:
    """Lazily configure multiprocessing for parallel pool creation.

    Called once, lazily, just before any Pool is created. Guarded by
    _mp_configured so it is idempotent even if workflow.py and sweep.py both
    call it in the same process.
    """
    global _mp_configured
    if _mp_configured:
        return
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        # Already set, e.g. user called set_start_method themselves; respect it.
        logger.debug("Multiprocessing start method already set; skipping.")
    if platform.system() == "Darwin":
        # macOS spawn workers inherit the parent's TMPDIR which may point to an
        # app sandbox directory not accessible to child processes.
        custom_temp = tempfile.mkdtemp()
        os.environ["TMPDIR"] = custom_temp
        logger.debug(f"macOS: set TMPDIR={custom_temp} for spawn workers.")
    _mp_configured = True


def _is_spike_input(data) -> bool:
    """Raw spike times arrive as a list of one ragged array per neuron."""
    return isinstance(data, (list, tuple)) and not isinstance(data, torch.Tensor) and (
        len(data) > 0 and all(np.ndim(np.asarray(d, dtype=object)) <= 1 for d in data)
        and not np.isscalar(data[0])
    )


def float_tensor(data) -> torch.Tensor:
    """``data`` as a float32 tensor, converting a tensor rather than copying it."""
    if torch.is_tensor(data):
        return data.float()
    return torch.from_numpy(np.asarray(data, dtype=np.float32))


def with_window_axis(t: torch.Tensor) -> torch.Tensor:
    """A 2-D ``(N, C)`` tensor with the trailing window axis of 1 every stream carries inside the library;
    any other tensor as it is."""
    return t.unsqueeze(-1) if t.ndim == 2 else t


def _as_tensor(data) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        return data
    if _is_spike_input(data):
        raise TypeError(
            "Raw spike times were passed to a quantity that indexes by integer "
            "time offset. That needs a regularly sampled (n_timepoints, "
            "n_channels) series. processing=Processing(x='spike', "
            "x_params={'window_size': b, 'bin_size': b}) puts the spikes on a grid "
            "of one bin per row first, keeping silent bins so that consecutive "
            "rows stay one bin apart. block_mi windows the data itself and takes "
            "spike times through processing=Processing(x='spike', y='spike')."
        )
    return torch.as_tensor(np.asarray(data), dtype=torch.float32)


def validate_stride(stride: Any, quantity: str = "this quantity") -> int:
    """Check a ``stride`` argument and return it as an int.

    ``stride`` is the offset family's analogue of a processor's ``step_size``:
    the distance in samples between the start of one row and the start of the
    next. It is deliberately a separate parameter with a separate name, because
    ``step_size`` reads any value below 1 as a fraction of ``window_size``, and
    these quantities have no single window length for a fraction to refer to
    (``cross_predictive_information`` has its own ``k``,
    ``mi_rate`` has ``h`` and ``W``). Whole samples only, therefore, with no
    fractional reading.
    """
    if isinstance(stride, bool) or not isinstance(stride, (int, np.integer)):
        raise ValueError(
            f"stride must be a whole number of samples, got {stride!r}. Unlike a "
            f"processor's step_size, stride has no fractional reading: {quantity} "
            f"has no single window length for a fraction to be taken of. For a "
            f"fractional step over a raw series, use block_mi's step_size."
        )
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}.")
    return int(stride)


def validate_window(value: Any, name: str, quantity: str = "this quantity", minimum: int = 1) -> int:
    """Check a window parameter (``k``, ``h``, ``half_width``, ``history_window``...) and return it as
    an int, as :func:`validate_stride` checks ``stride``.

    A window counts steps of the series, so it is a whole number, at least
    ``minimum``: 1 for a window that has to hold something, 0 for one that
    may be empty, such as a conditioning history.
    """
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(
            f"{name} must be a whole number of steps, got {value!r}. {quantity} counts its "
            f"windows in steps of the series."
        )
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum} for {quantity}, got {value}.")
    return int(value)


def build_offset_arrays(data: Dict[str, Any], spec: Dict[str, Any],
                        dtype: torch.dtype = torch.float32,
                        stride: int = 1) -> Tuple[Any, Any, Any, int]:
    """Turn an offset specification into aligned, trainable arrays.

    Every quantity in the temporal taxonomy is ``I(A; B | C)`` under a
    different choice of which processes and time offsets go into each group.
    This turns such a choice into the three arrays :func:`neural_mi.run`
    consumes. Every named quantity defined by time offsets builds its rows
    here, and :func:`neural_mi.offset_quantity` runs any other spec the same
    way, with the rows' geometry, which arrays passed to :func:`neural_mi.run`
    by hand do not carry.

    Offsets are in time bins relative to a common reference: negative is past,
    zero is present, positive is future. All three groups are cut to the same
    valid range, the widest window over which every requested offset
    exists.

    ``stride`` thins that range, keeping every ``stride``-th reference position
    instead of all of them. It applies to the shared reference, never to the
    offsets themselves, so every group stays aligned on the same time points.

    Parameters
    ----------
    data : dict of str to array-like
        One entry per process, each of shape ``(n_timepoints, n_channels)``,
        the timepoints-first convention used throughout the library. All
        processes must share the same number of timepoints.
    spec : dict
        Keys ``'A'`` and ``'B'`` are required, ``'C'`` is optional. Each maps
        to a sequence of ``(process_name, offset)`` pairs.
    dtype : torch.dtype, optional
        Element type of the returned tensors. Defaults to ``torch.float32``.

    Returns
    -------
    tuple of (Tensor, Tensor, Tensor or None, int)
        Arrays for A, B and C, each of shape
        ``(n_valid, n_channels, n_offsets)``, plus ``n_valid``. C is ``None``
        when the group is empty or absent.

    Raises
    ------
    ValueError
        If a group mixes processes whose offset counts differ, since the
        result would have no single window length. Give each process in that
        group the same number of offsets, or estimate it as a dual-branch
        conditional instead.

    Examples
    --------
    Active information storage, ``I(X_past; X_0)``:

    >>> from neural_mi.utils import build_offset_arrays
    >>> spec = {'A': [('x', s) for s in range(-10, 0)], 'B': [('x', 0)]}
    >>> a, b, c, n = build_offset_arrays({'x': signal}, spec)   # doctest: +SKIP

    Transfer entropy from x to y, ``I(X_past; Y_0 | Y_past)``:

    >>> spec = {'A': [('x', s) for s in range(-10, 0)],
    ...         'B': [('y', 0)],
    ...         'C': [('y', s) for s in range(-10, 0)]}

    With ``C`` empty the pair goes to ``run(mode='estimate')``. With ``C``
    populated it goes to ``run(mode='conditional', conditional=Conditional(
    w_data=c, align='dual_branch'))``, the dual branch being needed because A,
    B and C generally have different window lengths.
    """
    if 'A' not in spec or 'B' not in spec:
        raise ValueError(f"spec needs both 'A' and 'B' keys, got {sorted(spec)}.")

    groups = {g: list(spec.get(g) or []) for g in ('A', 'B', 'C')}
    if not groups['A'] or not groups['B']:
        raise ValueError("Groups 'A' and 'B' must each list at least one (process, offset) pair.")

    tensors = {}
    lengths = set()
    for name, arr in data.items():
        t = arr if torch.is_tensor(arr) else torch.as_tensor(np.asarray(arr))
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        if t.ndim != 2:
            raise ValueError(
                f"Process {name!r} has shape {tuple(t.shape)}. Expected "
                f"(n_timepoints, n_channels)."
            )
        tensors[name] = t.to(dtype)
        lengths.add(t.shape[0])
    if len(lengths) > 1:
        raise ValueError(
            f"All processes must share a timepoint count and have {sorted(lengths)}. "
            f"Truncate them to a common length before building offsets."
        )

    requested = {v for g in groups.values() for (v, _) in g}
    unknown = requested - set(tensors)
    if unknown:
        raise ValueError(
            f"spec references {sorted(unknown)}. data provides only "
            f"{sorted(tensors)}."
        )

    stride = validate_stride(stride, 'build_offset_arrays')
    offsets = [o for g in groups.values() for (_, o) in g]
    lo, hi = min(offsets), max(offsets)
    total = lengths.pop()
    start = -lo
    n_positions = total - hi - start
    if n_positions <= 0:
        raise ValueError(
            f"Offsets span {lo} to {hi} and leave no valid samples in a series "
            f"of {total} timepoints. Shorten the offset range or use a longer recording."
        )
    # Reference positions are start, start+stride, ... within n_positions, so
    # the count is the number of strided steps that still fit.
    n_valid = (n_positions - 1) // stride + 1

    out = {}
    for group_name, entries in groups.items():
        if not entries:
            out[group_name] = None
            continue
        by_process = OrderedDict()
        for (v, o) in entries:
            by_process.setdefault(v, []).append(o)
        blocks = []
        for v, offs in by_process.items():
            # Stride the shared reference, not each offset: every column is
            # the same n_valid reference positions, read at its own offset.
            cols = [tensors[v][start + o::stride][:n_valid] for o in offs]
            blocks.append(torch.stack(cols, dim=-1))
        widths = {b.shape[-1] for b in blocks}
        if len(widths) > 1:
            counts = {v: len(o) for v, o in by_process.items()}
            raise ValueError(
                f"Group {group_name!r} gives different offset counts per process "
                f"({counts}) and has no single window length. Give each process the "
                f"same number of offsets or split the group across the dual-branch "
                f"conditional path."
            )
        out[group_name] = torch.cat(blocks, dim=1)

    return out['A'], out['B'], out['C'], n_valid


def _shift_data(x_data: Any, y_data: Any, lag: int,
                x_processor_type: Optional[str], y_processor_type: str,
                sample_rate: Optional[float] = None) -> tuple:
    """Shifts data to apply a time lag between X and Y.

    Sign convention (shared by every branch below): for ``lag > 0``, Y is
    compared against its own future relative to X (X's early samples are
    paired with Y's later samples).

    Parameters
    ----------
    x_data : array-like
        Data for variable X.
    y_data : array-like
        Data for variable Y.
    lag : int or float
        The lag value. For 'spike' data, always seconds. For 'continuous'/
        'categorical' data, seconds if `sample_rate` is given, otherwise a raw
        sample-index offset. For pre-processed data (``y_processor_type`` is
        ``None``) always an index offset along axis 0, whose unit follows the
        array's shape: timepoints for 1-D/2-D input, **windows** for 3-D
        pre-windowed input. The 3-D case warns, since ``lag`` there means
        ``lag * step_size`` timepoints.

        Any lag assumes axis 0 is in time order. ``None`` means "already
        prepared, do not window", not "IID", so that assumption usually holds
        for a pre-processed recording and is the reason lagging static data is
        meaningful at all. It cannot be verified here; on genuinely unordered
        rows a shift simply breaks the pairing, giving full MI at lag 0 and
        roughly zero elsewhere.
    x_processor_type : str, optional
        X's processor type: one of 'continuous', 'categorical', 'spike', or
        None. Only consulted to detect the mixed-modality case (spike paired
        with non-spike); when both sides are non-spike, only
        `y_processor_type` determines how the shift is applied (both arrays
        get the same array-index treatment regardless of which specific
        non-spike type each one is).
    y_processor_type : str or None
        Y's processor type: one of 'continuous', 'categorical', 'spike', or
        ``None`` for pre-processed static data. Any other value raises instead of
        returning the data unshifted. A lag analysis that silently applied no
        lag would report a flat profile that looks like a real negative result.

    Raises
    ------
    ValueError
        If ``y_processor_type`` is not one of the recognised values.
    sample_rate : float, optional
        Samples per second for continuous/categorical data. If provided, lag is
        interpreted as seconds and converted to samples. If None, lag is treated
        as a raw sample-index offset and a warning is emitted, since the unit is
        then ambiguous relative to spike data (always seconds).
    """
    x_is_spike = x_processor_type == 'spike'
    y_is_spike = y_processor_type == 'spike'

    if x_is_spike and not y_is_spike:
        # Mixed modality: spike-X, non-spike-Y. Shift X's spike times forward
        # by `lag` (instead of shifting Y, which isn't spike data and has no
        # per-event timestamps to offset) -- this keeps the lag>0="Y is in
        # the future" convention: advancing X's clock by `lag` is equivalent
        # to comparing X's past against Y's present.
        logger.info(
            f"Mixed-modality lag with X 'spike' and Y '{y_processor_type}'. X's spike "
            f"times are shifted by +{lag}s and Y stays at its original times."
        )
        x_shifted = [spikes + lag for spikes in x_data]
        return x_shifted, y_data

    if y_is_spike:
        # Y is spike (X may be spike too, or non-spike -- either way only Y's
        # spike times need shifting; X is unaffected in both sub-cases).
        if not x_is_spike:
            logger.info(
                f"Mixed-modality lag with Y 'spike' and X '{x_processor_type}'. Y's spike "
                f"times are shifted by -{lag}s and X stays at its original times."
            )
        y_shifted = [spikes - lag for spikes in y_data]
        return x_data, y_shifted

    if y_processor_type in ('continuous', 'categorical') or y_processor_type is None:
        # `None` means pre-processed static data (create_single_dataset's
        # convention). It shifts along the same axis as the windowed types,
        # but the *unit* of that axis depends on what the caller supplied,
        # so it is resolved from ndim below instead of assumed.
        # Convert to numpy if needed
        if torch.is_tensor(x_data):
            x_data = x_data.detach().cpu().numpy()
        elif not isinstance(x_data, np.ndarray):
            x_data = np.array(x_data)

        if torch.is_tensor(y_data):
            y_data = y_data.detach().cpu().numpy()
        elif not isinstance(y_data, np.ndarray):
            y_data = np.array(y_data)

        if y_processor_type is None:
            # Static (pre-processed) data. There is no sample_rate to convert
            # with, so the lag is an index offset by construction. What that
            # index *means* depends on the array's shape, and the two cases
            # carry different units -- so name the unit instead of guess
            # silently. This mirrors how the sample_rate ambiguity above is
            # handled: pick the documented reading and say so.
            if x_data.ndim >= 3:
                warnings.warn(
                    f"Lag on pre-processed data with shape {x_data.shape}. Axis 0 counts "
                    f"windows. lag={lag} shifts by {lag} windows ({lag} x step_size "
                    f"timepoints). Pass unwindowed data with "
                    f"processing=Processing(x='continuous', ...) for a lag in timepoints.",
                    UserWarning, stacklevel=user_stacklevel()
                )
            else:
                logger.info(
                    f"Lag on pre-processed data with shape {x_data.shape}. Axis 0 counts "
                    f"timepoints and lag={lag} shifts by {lag} samples."
                )
            lag_samples = int(lag)
        elif sample_rate is not None:
            # Lag provided in seconds, convert to samples
            lag_samples = int(round(lag * sample_rate))
        else:
            # No sample_rate given: treat lag as a raw sample-index offset, but warn
            warnings.warn(
                f"Lag units for '{y_processor_type}' data are ambiguous without a sample_rate. "
                f"Treating lag={lag} as samples (index offset). "
                f"To specify lag in seconds, set 'sample_rate' in Processing(x_params=...). "
                f"Spike data always use seconds. Mixing processor types without "
                f"sample_rate gives inconsistent lag scales.",
                UserWarning, stacklevel=user_stacklevel()
            )
            lag_samples = int(lag)

        # Slice axis 0 only, so 1-D (which StaticDataset accepts), 2-D and 3-D
        # data are all handled.
        if lag_samples == 0:
            return x_data, y_data
        elif lag_samples > 0:
            # y is in the future of x
            return x_data[:-lag_samples], y_data[lag_samples:]
        else:
            # y is in the past of x
            return x_data[-lag_samples:], y_data[:lag_samples]

    raise ValueError(
        f"_shift_data cannot apply a lag to y_processor_type={y_processor_type!r}. "
        f"Expected 'spike', 'continuous', 'categorical' or None (pre-processed). "
        f"Returning the data unshifted would report a lag analysis that never "
        f"applied a lag."
    )

    
def _accepts_kwarg(cls: type, name: str) -> bool:
    """Whether ``cls.__init__`` can receive ``name``, directly or via **kwargs."""
    try:
        params = inspect.signature(cls.__init__).parameters
    except (TypeError, ValueError):
        return False
    if name in params:
        return True
    return any(prm.kind is inspect.Parameter.VAR_KEYWORD for prm in params.values())


def mi_report_units(base_params: Optional[Dict[str, Any]] = None,
                    output_units: Optional[str] = None) -> Tuple[float, str]:
    """Scale factor and label for reporting an internal MI value to the caller.

    Every MI in this library is computed in nats and converted to the caller's
    ``output_units`` at the boundary. Anything that prints a value before that
    boundary, a warning or a log line, has to apply the same conversion, or the
    number it shows will not match the number the caller reads back.

    Parameters
    ----------
    base_params : dict, optional
        The run's ``base_params``, which carries ``output_units``.
    output_units : str, optional
        Used directly when the caller has the value but not the dict.

    Returns
    -------
    scale : float
        Multiply a nats value by this before displaying it.
    label : str
        ``'bits'`` or ``'nats'``, for the message.

    Examples
    --------
    >>> scale, units = mi_report_units({'output_units': 'bits'})
    >>> f"{0.6931 * scale:.3f} {units}"
    '1.000 bits'
    """
    units = output_units or (base_params or {}).get('output_units', 'bits')
    to_bits = units == 'bits'
    return (1.0 / np.log(2) if to_bits else 1.0), ('bits' if to_bits else 'nats')


# The Model settings an encoder or a head can read beyond its sizes. Each
# built-in encoder declares the ones it reads (BaseEmbedding.settings).
MODEL_SETTINGS = ('use_spectral_norm', 'dropout', 'norm_layer', 'bias', 'kernel_size', 'cnn_pooling',
                  'bidirectional', 'nhead', 'pytorch_predefined', 'pretrained')


# Every setting build_critic reads, and so the list a saved network carries for
# extract_embeddings to rebuild it as it was trained (embeddings_io.save_network).
# build_critic sees nothing else, so the two cannot drift apart.
ARCHITECTURE_KEYS = (
    'critic_type', 'embedding_model', 'embedding_dim', 'hidden_dim', 'n_layers',
    'embedding_model_y', 'embedding_dim_y', 'hidden_dim_y', 'n_layers_y',
    'custom_critic', 'custom_embedding_cls', 'custom_embedding_cls_y', 'hidden_dim_head', 'n_layers_head',
    'use_variational', 'shared_encoder', 'max_n_batches',
    'input_dim_x', 'input_dim_y', 'n_channels_x', 'n_channels_y',
    '_difference_term', '_condition', 'processor_params_x', 'processor_params_y', 'input_scaling',
) + MODEL_SETTINGS


# What a custom critic recipe can take, by name: the input sizes the task
# computed, and the architecture settings.
CRITIC_SIZES = ('input_dim_x', 'input_dim_y', 'n_channels_x', 'n_channels_y', 'window_size_x', 'window_size_y')
CRITIC_SETTINGS = ('embedding_dim', 'hidden_dim', 'n_layers', 'input_scaling') + MODEL_SETTINGS


def check_custom_critic(recipe) -> None:
    """Refuse a ``custom_critic`` that is not a recipe: an object would be trained in place and shared by every
    repeat and every term of a difference."""
    if recipe is None:
        return
    if isinstance(recipe, nn.Module) or not callable(recipe):
        raise TypeError(
            f"Model(custom_critic=...) takes a class or a function that builds the critic, and got "
            f"{type(recipe).__name__!r} object. The library calls it in each task, under that task's seed, "
            f"so every repeat and every term of a difference trains a critic of its own. An object would be "
            f"trained in place and shared by all of them. Pass the class, Model(custom_critic=MyCritic), or a "
            f"function, Model(custom_critic=make_critic) with def make_critic(input_dim_x, input_dim_y): "
            f"return MyCritic(input_dim_x, input_dim_y). It receives the parameters it names among "
            f"{', '.join(CRITIC_SIZES + CRITIC_SETTINGS)}."
        )


def critic_recipe_takes(recipe) -> Optional[set]:
    """The names a critic recipe takes among ``CRITIC_SIZES`` and ``CRITIC_SETTINGS``, or None when it takes
    them all (``**kwargs``)."""
    try:
        parameters = inspect.signature(recipe).parameters
    except (TypeError, ValueError):
        return set()
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return None
    return set(parameters) & set(CRITIC_SIZES + CRITIC_SETTINGS)


def _custom_critic(recipe, params: Dict[str, Any]) -> nn.Module:
    """The critic a ``custom_critic`` recipe builds, called with the sizes and settings it names."""
    check_custom_critic(recipe)

    def window(side):
        flat, channels = params.get(f'input_dim_{side}'), params.get(f'n_channels_{side}')
        if flat is None or channels is None:
            return None
        return tuple(f // c for f, c in zip(flat, channels)) if isinstance(channels, tuple) else flat // channels

    norm = setting(params, 'norm_layer')
    available = {**{key: params.get(key) for key in CRITIC_SIZES[:4]},
                 'window_size_x': window('x'), 'window_size_y': window('y'),
                 **{name: setting(params, name) for name in CRITIC_SETTINGS}}
    available['norm_layer'] = None if norm in ('auto', 'none') else norm
    takes = critic_recipe_takes(recipe)
    kwargs = available if takes is None else {name: available[name] for name in takes}
    required = [name for name, p in inspect.signature(recipe).parameters.items()
                if p.default is inspect.Parameter.empty and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
                and name not in kwargs]
    if required:
        raise TypeError(
            f"custom_critic {getattr(recipe, '__name__', recipe)!r} requires {required}, which the library "
            f"does not pass. It can name any of {', '.join(CRITIC_SIZES + CRITIC_SETTINGS)}."
        )
    critic = recipe(**kwargs)
    if not isinstance(critic, nn.Module):
        raise TypeError(f"custom_critic must build a torch.nn.Module and built a {type(critic).__name__!r}.")
    return critic


def _names(cls, key: str) -> bool:
    """Whether ``cls.__init__`` names ``key`` as a parameter of its own."""
    try:
        return key in inspect.signature(cls.__init__).parameters
    except (TypeError, ValueError):
        return False


def build_critic(critic_type: str, embedding_params: Dict[str, Any],
                 custom_embedding_cls: Optional[type] = None) -> BaseCritic:
    """Builds and returns a critic model based on the provided parameters.

    This function expects `embedding_params` to be fully populated with
    defaults (e.g., via `ParameterValidator.apply_defaults`). It strictly
    accesses required parameters and will raise a KeyError if something is missing,
    preventing silent failures from missing defaults.

    For ``critic_type='hybrid'``, the decision head MLP that scores the
    concatenated embeddings can be configured independently of the embedding
    networks via two optional keys in ``embedding_params``:

    - ``hidden_dim_head`` (int or None): hidden width of the decision head.
      Defaults to ``None``, which resolves to ``min(64, hidden_dim)``.
    - ``n_layers_head`` (int or None): number of hidden layers in the decision
      head.  Defaults to ``None``, which resolves to ``max(1, n_layers - 1)``.

    These parameters have no effect for ``critic_type`` values other than
    ``'hybrid'``.
    """
    
    # Only the architecture keys reach the build, so a network saved with them
    # rebuilds as it was trained. A setting absent from them takes its schema
    # default.
    embedding_params = {k: embedding_params[k] for k in ARCHITECTURE_KEYS if k in embedding_params}
    if embedding_params.get('custom_critic') is not None:
        # The caller's recipe builds the whole critic, in each task anew.
        return _custom_critic(embedding_params['custom_critic'], embedding_params)
    use_variational = setting(embedding_params, 'use_variational')
    max_n_batches = setting(embedding_params, 'max_n_batches')
    # The class also travels in embedding_params, so a caller that only has the
    # dict (embeddings_io rebuilding a saved run) still gets it.
    custom_embedding_cls = custom_embedding_cls or embedding_params.get('custom_embedding_cls')

    # X sets the architecture and Y follows it unless given its own, the same
    # "unset means same as X" convention processor_type_y already uses. Naming
    # a built-in for Y drops X's custom class on that side, since the two
    # settings are alternatives and the explicit one wins.
    _y_model = embedding_params.get('embedding_model_y')
    _y_custom = embedding_params.get('custom_embedding_cls_y')
    spec_x = {
        'model_type': setting(embedding_params, 'embedding_model').lower(),
        'custom_cls': custom_embedding_cls,
        'hidden_dim': setting(embedding_params, 'hidden_dim'),
        'n_layers': setting(embedding_params, 'n_layers'),
        'embedding_dim': setting(embedding_params, 'embedding_dim'),
    }
    spec_y = {
        'model_type': _y_model.lower() if _y_model else spec_x['model_type'],
        'custom_cls': _y_custom or (None if _y_model else spec_x['custom_cls']),
        'hidden_dim': embedding_params.get('hidden_dim_y') or spec_x['hidden_dim'],
        'n_layers': embedding_params.get('n_layers_y') or spec_x['n_layers'],
        'embedding_dim': embedding_params.get('embedding_dim_y') or spec_x['embedding_dim'],
    }
    _Y_KEYS = ('embedding_model_y', 'custom_embedding_cls_y', 'hidden_dim_y',
               'n_layers_y', 'embedding_dim_y')
    _given_y = [k for k in _Y_KEYS if embedding_params.get(k) is not None]

    shared_encoder = setting(embedding_params, 'shared_encoder')
    if _given_y:
        if critic_type == 'concat':
            raise ValueError(
                f"{_given_y[0]} was set with critic_type='concat'. ConcatCritic "
                f"scores the raw concatenated pair and has no separate embedding "
                f"networks or Y-side encoder to configure. Switch to "
                f"critic_type='separable' or 'hybrid'."
            )
        if shared_encoder:
            raise ValueError(
                f"{_given_y[0]} was set with shared_encoder=True. One encoder "
                f"instance cannot also be a second, different one. Set "
                f"shared_encoder=False to give the two sides their own encoders."
            )
    if (embedding_params.get('embedding_dim_y') is not None
            and spec_y['embedding_dim'] != spec_x['embedding_dim']
            and critic_type == 'separable'):
        raise ValueError(
            f"embedding_dim_y={spec_y['embedding_dim']} differs from "
            f"embedding_dim={spec_x['embedding_dim']}. critic_type='separable' "
            f"scores a pair by the dot product of the two embeddings and needs "
            f"one width. critic_type='hybrid' concatenates them and accepts "
            f"different widths."
        )

    # The Model settings every network of the critic can read, resolved once.
    # norm_layer 'auto' is layer normalisation for the hybrid critic in
    # mode='dimensionality', which resolves it there, and none everywhere else.
    norm = setting(embedding_params, 'norm_layer')
    # An explicit bias of None means unset, not off.
    _bias = setting(embedding_params, 'bias')
    values = {name: setting(embedding_params, name) for name in MODEL_SETTINGS}
    values['norm_layer'] = None if norm in ('auto', 'none') else norm
    values['bias'] = True if _bias is None else bool(_bias)
    # Every MLP the critic builds around its encoders (the hybrid critic's
    # head, the concat critic's network, the head of a difference term's
    # branched encoder) reads the call's MLP settings.
    head_settings = {name: values[name] for name in MLP.settings}

    def _resolve_side(spec, side):
        """The class, the ``input_dim`` it expects, and its kwargs, for one side.

        A built-in class gets the Model settings it declares (``settings``), a
        custom class every one its ``__init__`` names, and ``bias`` also
        through ``**kwargs``.
        """
        if (spec['n_layers'] == 0 and not spec['custom_cls'] and spec['model_type'] != 'mlp'
                and not isinstance(spec['hidden_dim'], list)):
            raise ValueError(
                f"n_layers=0 builds a linear encoder, which only the MLP can be. "
                f"embedding_model={spec['model_type']!r} needs at least one layer."
            )
        if spec['custom_cls']:
            EmbeddingModel = spec['custom_cls']
        elif spec['model_type'] in _EMBEDDING_CLASSES:
            EmbeddingModel = _EMBEDDING_CLASSES[spec['model_type']]
        else:
            raise ValueError(
                f"Unknown embedding_model: {spec['model_type']!r}. Built-in options: "
                f"{sorted(_EMBEDDING_CLASSES)}. Pass a class via custom_embedding_cls "
                f"for anything else."
            )
        model_kwargs = {
            'hidden_dim': spec['hidden_dim'],
            'embedding_dim': spec['embedding_dim'],
            'n_layers': spec['n_layers'],
        }
        if spec['custom_cls']:
            model_kwargs.update({name: values[name] for name in MODEL_SETTINGS
                                 if _names(EmbeddingModel, name)})
            if _accepts_kwarg(EmbeddingModel, 'bias'):
                model_kwargs['bias'] = values['bias']
        else:
            model_kwargs.update({keyword: values[name]
                                 for name, keyword in EmbeddingModel.settings.items()})
        # A class's own `input_style` (see BaseEmbedding) decides the input_dim
        # convention it's built with -- 'channels' (raw channel count, the
        # window/sequence axis handled internally by the class) or 'flattened'
        # (n_channels * window_size, the default for any class that doesn't
        # declare otherwise, including a third-party custom class that never
        # sets it). Read per side, since the two sides can be different classes.
        if getattr(EmbeddingModel, 'input_style', 'flattened') == 'channels':
            input_dim = embedding_params[f'n_channels_{side}']
        else:
            input_dim = embedding_params[f'input_dim_{side}']
        return EmbeddingModel, input_dim, model_kwargs

    Model_x, input_dim_x, model_kwargs = _resolve_side(spec_x, 'x')
    Model_y, input_dim_y, model_kwargs_y = _resolve_side(spec_y, 'y')
    hidden_dim, n_layers = spec_x['hidden_dim'], spec_x['n_layers']
    embedding_dim = spec_x['embedding_dim']
    critic_kwargs = {
        'embedding_dim': embedding_dim,
        'max_n_batches': max_n_batches,
        'use_variational': use_variational,
    }

    if shared_encoder and critic_type == 'concat':
        raise ValueError(
            "shared_encoder=True is incompatible with critic_type='concat'. "
            "ConcatCritic operates on raw concatenated inputs and has no separate "
            "embedding networks to share. Switch to critic_type='separable' or 'hybrid'."
        )
    # A term of a difference quantity: its X encoder reads each part of X on a
    # branch of its own, under one head (BranchedEmbedding).
    difference_term = embedding_params.get('_difference_term', False)
    if shared_encoder and difference_term:
        raise ValueError(
            "shared_encoder=True cannot serve a term of a difference quantity. Its X encoder reads the "
            "conditioning set and the rest of X on branches of their own under one head (BranchedEmbedding), "
            "and cannot also be Y's encoder. Set shared_encoder=False."
        )

    bias_x = bias_y = values['bias']

    for Model_, unused in ((Model_x, None), (Model_y, None)):
        if not (bias_x and bias_y) and not getattr(Model_, 'zero_preserving', True):
            warnings.warn(
                f"bias=False was requested and {Model_.__name__} carries an "
                f"input-independent additive term (a positional encoding or biases "
                f"baked into pretrained weights). An all-zero input will not embed "
                f"to zero. The bias terms it does own are still removed.",
                UserWarning, stacklevel=user_stacklevel(),
            )
            break

    # A custom embedding class following the minimal BaseEmbedding contract
    # (input_dim, hidden_dim, embedding_dim, n_layers) must still build, so an
    # explicit bias request that such a class cannot receive is reported rather
    # than forced on it.
    for Model_, kwargs in ((Model_x, model_kwargs), (Model_y, model_kwargs_y)):
        if 'bias' not in kwargs and not (bias_x and bias_y):
            # Only reachable when bias was actually turned off AND the class cannot
            # receive it. Gating this on any other capability would report a bias
            # problem the class does not have.
            warnings.warn(
                f"bias=False applies to the embedding layers. "
                f"{Model_.__name__}.__init__ does not accept a `bias` "
                f"argument and its layers keep their bias terms. Add `bias=True` to "
                f"the signature and pass it to the layers to support this.",
                UserWarning, stacklevel=user_stacklevel(),
            )

    # A class that names window_size gets the window length, the flattened size
    # over the channel count (a pair of them for a compound input), so it need
    # not average over the window to stay independent of its length.
    for Model_, kwargs, side in ((Model_x, model_kwargs, 'x'), (Model_y, model_kwargs_y, 'y')):
        n_ch, flat = embedding_params.get(f'n_channels_{side}'), embedding_params.get(f'input_dim_{side}')
        if _names(Model_, 'window_size') and n_ch is not None and flat is not None:
            kwargs['window_size'] = (tuple(f // c for f, c in zip(flat, n_ch)) if isinstance(n_ch, tuple)
                                     else flat // n_ch)

    # DeepSets masks padded slots, so it needs the sentinel the processor used.
    # Taken per side, since X and Y can be processed differently.
    for Model_, kwargs, key in ((Model_x, model_kwargs, 'processor_params_x'),
                                (Model_y, model_kwargs_y, 'processor_params_y')):
        if _accepts_kwarg(Model_, 'no_spike_value'):
            params = embedding_params.get(key) or {}
            kwargs['no_spike_value'] = params.get('no_spike_value', 0.0)

    net_x_base = (_branched(Model_x, input_dim_x, model_kwargs, embedding_params['n_channels_x'],
                            embedding_params.get('_condition'), head_settings)
                  if difference_term else Model_x(input_dim_x, **model_kwargs))
    net_y_base = net_x_base if shared_encoder else Model_y(input_dim_y, **model_kwargs_y)

    # Model(input_scaling='standard'): every encoder reads its input centred and
    # scaled channel by channel, by statistics the trainer sets from the
    # training rows (models.critics.fit_input_scaling). They are buffers of the
    # network, saved and applied with it.
    input_scaling = setting(embedding_params, 'input_scaling')
    if input_scaling == 'standard':
        for Model_ in (Model_x, Model_y):
            if Model_ is DeepSets and critic_type != 'concat':
                raise ValueError(
                    "Model(input_scaling='standard') cannot serve embedding_model='deepsets'. DeepSets tells a "
                    "spike from padding by the padding value, which scaling moves. Use another encoder, or "
                    "leave input_scaling unset."
                )
        if critic_type != 'concat':
            net_x_base = InputScaling(net_x_base, embedding_params['n_channels_x'])
            net_y_base = net_x_base if shared_encoder else InputScaling(net_y_base, embedding_params['n_channels_y'])

    # Optionally wrap with VariationalWrapper, works for every encoder type
    if use_variational:
        net_x = VariationalWrapper(net_x_base, spec_x['embedding_dim'], bias=bias_x)
        net_y = net_x if shared_encoder else VariationalWrapper(
            net_y_base, spec_y['embedding_dim'], bias=bias_y)
    else:
        net_x, net_y = net_x_base, net_y_base

    # Warn when the first embedding layer is severely overparameterized.
    # Large first layers (input_dim * hidden_dim) are the most common cause of
    # overfitting in neuroscience datasets where windows are high-dimensional but
    # sample counts are modest. 500k is a practical threshold, not a hard limit.
    _first_hidden = hidden_dim[0] if isinstance(hidden_dim, list) else hidden_dim
    _last_hidden  = hidden_dim[-1] if isinstance(hidden_dim, list) else hidden_dim
    # A pair of inputs counts with both branches' sizes.
    _input_dim_x_total = sum(input_dim_x) if isinstance(input_dim_x, tuple) else input_dim_x
    first_layer_params = _input_dim_x_total * _first_hidden
    if first_layer_params > 500_000:
        logger.info(
            f"Large first embedding layer detected: input_dim_x={input_dim_x} x "
            f"hidden_dim={hidden_dim} = {first_layer_params:,} parameters. "
            f"It may overfit small datasets. A smaller window_size or hidden_dim or "
            f"another embedding model avoids that."
        )

    # --- Critic Assembly ---
    if critic_type == 'separable':
        return SeparableCritic(embedding_net_x=net_x, embedding_net_y=net_y, **critic_kwargs)
    elif critic_type == 'hybrid':
        # Hybrid concatenates the two embeddings before scoring, so the head's
        # input is their combined width and the two may differ.
        decision_head_input_dim = spec_x['embedding_dim'] + spec_y['embedding_dim']
        _head_hidden_dim = embedding_params.get('hidden_dim_head') or min(64, _last_hidden)
        _head_n_layers = embedding_params.get('n_layers_head') or max(1, n_layers - 1)
        decision_head = MLP(input_dim=decision_head_input_dim, hidden_dim=_head_hidden_dim, embedding_dim=1,
                            n_layers=_head_n_layers, **head_settings)
        return HybridCritic(embedding_net_x=net_x, embedding_net_y=net_y, decision_head=decision_head, **critic_kwargs)
    elif critic_type == 'concat':
        # The concat critic scores the flattened pair and builds no encoder, so
        # its input is the flattened sizes whatever embedding_model names.
        concat_input_dim = embedding_params['input_dim_x'] + embedding_params['input_dim_y']
        concat_net = MLP(input_dim=concat_input_dim, hidden_dim=hidden_dim, embedding_dim=1, n_layers=n_layers,
                         **head_settings)
        if use_variational:
            # The concat network's one output is the score of a pair, so the
            # variational layer makes each score a Gaussian draw and its KL term
            # regularises the scores. There is no per-variable embedding here
            # for it to compress.
            concat_net = VariationalWrapper(concat_net, 1, bias=bias_x)
        scaling = ((ChannelScaling(embedding_params['n_channels_x']), ChannelScaling(embedding_params['n_channels_y']))
                   if input_scaling == 'standard' else None)
        return ConcatCritic(embedding_net=concat_net, scaling=scaling, **critic_kwargs)
    else:
        raise ValueError(f"Unknown critic_type: {critic_type}")


def _branched(branch_cls, input_dim, kwargs, n_channels, condition, head) -> BranchedEmbedding:
    """The X encoder of a term of a difference, with ``branch_cls`` built as the call's X encoder for each part.

    A joint term's X arrives as a pair with its two sizes, or as one tensor
    split by ``condition = ('first' | 'last', n)``: its first or last ``n``
    channels are the conditioning set, in a size convention of ``input_dim``
    per ``n_channels`` channels, at one window length. Any other term's X is
    one part. ``head`` holds the head's MLP settings.
    """
    if condition is not None:
        per_channel, n = input_dim // n_channels, condition[1]
        input_dim = ((n_channels - n) * per_channel, n * per_channel)
        if 'window_size' in kwargs:
            kwargs = {**kwargs, 'window_size': (kwargs['window_size'],) * 2}
    return BranchedEmbedding(input_dim, branch_cls=branch_cls, condition=condition, head=head, **kwargs)


_OPTIMIZERS = {
    'adam': optim.Adam,
    'adamw': optim.AdamW,
    'sgd': optim.SGD,
    'rmsprop': optim.RMSprop,
    'adagrad': optim.Adagrad,
}
_SCHEDULER_NAMES = {'cosine', 'step', 'plateau', 'cosine_warmup'}


def build_optimizer_and_scheduler(
    params: Dict[str, Any],
    critic: nn.Module,
    decoder_x: Optional[nn.Module] = None,
    decoder_y: Optional[nn.Module] = None,
) -> Tuple[torch.optim.Optimizer, Optional[Any]]:
    """Build the optimizer and optional LR scheduler from base_params.

    Shared by task.py (per-sweep training, which may include reconstruction
    decoders and a per-head LR multiplier) and precision.py (a single
    train-once-evaluate-many baseline model, no decoders). This function
    expects ``params`` to be fully populated (e.g. via
    ``ParameterValidator.apply_defaults``): it accesses ``learning_rate`` and
    ``n_epochs`` strictly and will raise a ``KeyError`` if either is missing,
    instead of silently substituting a different default than
    ``BASE_PARAMS_SCHEMA``'s.

    Parameters
    ----------
    params : dict
        Must contain ``learning_rate``. May contain ``optimizer``,
        ``optimizer_params``, ``lr_head_multiplier``, ``scheduler``,
        ``scheduler_params`` (and ``n_epochs`` if a scheduler is requested).
    critic : nn.Module
        The critic whose parameters (plus any decoder parameters) are
        optimized.
    decoder_x, decoder_y : nn.Module, optional
        Reconstruction decoders whose parameters, if given, are added to the
        optimizer alongside the critic's. ``None`` (default) omits them.

    Returns
    -------
    tuple[torch.optim.Optimizer, scheduler or None]
    """
    _opt_val = setting(params, 'optimizer')
    if isinstance(_opt_val, type):
        OptCls = _opt_val
    else:
        OptCls = _OPTIMIZERS.get(str(_opt_val).lower())
        if OptCls is None:
            raise ValueError(
                f"Unknown optimizer '{_opt_val}'. "
                f"Supported names: {list(_OPTIMIZERS.keys())}. "
                f"You can also pass a torch.optim.Optimizer subclass directly."
            )

    # Collect all trainable parameters (critic + optional decoders).
    # When lr_head_multiplier is set and the critic has a decision_head (i.e. hybrid),
    # split into two param groups so the head can train at a different rate.
    _base_lr = params['learning_rate']
    _head_mult = params.get('lr_head_multiplier')
    _decoder_params = []
    if decoder_x is not None:
        _decoder_params.extend(decoder_x.parameters())
    if decoder_y is not None:
        _decoder_params.extend(decoder_y.parameters())

    if _head_mult is not None and _head_mult != 1.0 and hasattr(critic, 'decision_head'):
        _head_ids = {id(p) for p in critic.decision_head.parameters()}
        _encoder_params = [p for p in critic.parameters() if id(p) not in _head_ids]
        _encoder_params.extend(_decoder_params)
        _param_groups = [
            {'params': _encoder_params,                        'lr': _base_lr},
            {'params': list(critic.decision_head.parameters()), 'lr': _base_lr * _head_mult},
        ]
        optimizer = OptCls(_param_groups, **setting(params, 'optimizer_params'))
    else:
        _all_params = list(critic.parameters()) + _decoder_params
        optimizer = OptCls(_all_params, lr=_base_lr, **setting(params, 'optimizer_params'))

    _sched_val = setting(params, 'scheduler')
    scheduler = None
    if _sched_val is not None:
        _sched_params = setting(params, 'scheduler_params')
        n_epochs = params['n_epochs']
        if isinstance(_sched_val, type):
            scheduler = _sched_val(optimizer, **_sched_params)
        elif _sched_val == 'cosine':
            scheduler = _lr_sched.CosineAnnealingLR(optimizer, T_max=n_epochs, **_sched_params)
        elif _sched_val == 'step':
            scheduler = _lr_sched.StepLR(optimizer, step_size=max(1, n_epochs // 3), **_sched_params)
        elif _sched_val == 'plateau':
            scheduler = _lr_sched.ReduceLROnPlateau(optimizer, mode='max', **_sched_params)
        elif _sched_val == 'cosine_warmup':
            warmup = max(1, int(n_epochs * 0.1))
            warmup_sched = _lr_sched.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup)
            cosine_sched = _lr_sched.CosineAnnealingLR(optimizer, T_max=max(1, n_epochs - warmup))
            scheduler = _lr_sched.SequentialLR(
                optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup]
            )
        else:
            raise ValueError(
                f"Unknown scheduler '{_sched_val}'. "
                f"Supported names: {sorted(_SCHEDULER_NAMES)}. "
                f"You can also pass a torch.optim.lr_scheduler class directly."
            )
    return optimizer, scheduler


def _whiten(z: np.ndarray, whitening: Optional[str]) -> np.ndarray:
    """Normalize centred embeddings before their cross-covariance is taken."""
    if whitening == 'std':
        std = z.std(axis=0, keepdims=True)
        # Dead dimensions keep their (zero) scale instead of dividing by zero.
        return z / np.where(std > 1e-8, std, 1.0)
    if whitening == 'zca':
        cov = (z.T @ z) / (z.shape[0] - 1)
        u, s, _ = np.linalg.svd(cov)
        w = u @ np.diag(1.0 / np.sqrt(np.maximum(s, 1e-8))) @ u.T
        return z @ w.T
    if whitening is not None:
        raise ValueError(f"Unknown whitening {whitening!r}. Expected 'std', 'zca' or None.")
    return z


def cross_covariance_svd(zx, zy, whitening: Optional[str] = 'std') -> Dict[str, np.ndarray]:
    """The SVD of the cross-covariance of two embeddings.

    The one computation behind both the spectrum that participation ratios are
    read from and the rotation that orders embedding dimensions by shared
    variance. The embeddings are centred, normalized by `whitening` to compute
    the cross-covariance :math:`C = Z_X^\top Z_Y / (N - 1)`, and :math:`C` is
    decomposed.

    Parameters
    ----------
    zx, zy : np.ndarray or torch.Tensor
        Embeddings of shape ``(N, d_x)`` and ``(N, d_y)``.
    whitening : {'std', 'zca', None}, default='std'
        ``'std'`` divides each dimension by its standard deviation, so the
        spectrum counts dimensions with shared variance regardless of the
        embedding's output scale. ``'zca'`` spheres each embedding fully and
        needs ``N`` well above the embedding size. ``None`` leaves the scale as
        it is.

    Returns
    -------
    dict
        ``'zx_centered'``, ``'zy_centered'`` (original scale),
        ``'singular_values'`` (largest first), ``'rotation_x'`` (left singular
        vectors, ``(d_x, min(d_x, d_y))``) and ``'rotation_y'`` (right singular
        vectors, ``(d_y, min(d_x, d_y))``).
    """
    if hasattr(zx, 'detach'):
        zx = zx.detach().cpu().float().numpy()
    if hasattr(zy, 'detach'):
        zy = zy.detach().cpu().float().numpy()
    zx_c = np.asarray(zx, dtype=np.float64)
    zy_c = np.asarray(zy, dtype=np.float64)
    zx_c = zx_c - zx_c.mean(axis=0, keepdims=True)
    zy_c = zy_c - zy_c.mean(axis=0, keepdims=True)
    n = zx_c.shape[0]
    if n <= 1:
        return {'zx_centered': zx_c, 'zy_centered': zy_c, 'singular_values': np.array([]),
                'rotation_x': np.eye(zx_c.shape[1]), 'rotation_y': np.eye(zy_c.shape[1])}
    cov_xy = (_whiten(zx_c, whitening).T @ _whiten(zy_c, whitening)) / (n - 1)
    u, s, vt = np.linalg.svd(cov_xy, full_matrices=False)
    return {'zx_centered': zx_c, 'zy_centered': zy_c, 'singular_values': s,
            'rotation_x': u, 'rotation_y': vt.T}


def compute_cross_covariance_spectrum(zx, zy, whitening: Optional[str] = 'std') -> np.ndarray:
    """Singular values of the cross-covariance of two embeddings.

    See :func:`cross_covariance_svd` for `whitening`.
    """
    return cross_covariance_svd(zx, zy, whitening)['singular_values']


def compute_cross_covariance_rotation(zx, zy, whitening: Optional[str] = 'std') -> Dict[str, np.ndarray]:
    """Embeddings re-expressed so that dimension 0 carries the most shared variance.

    The rotation comes from :func:`cross_covariance_svd`. Whitening sets the
    rotation's axes only: the returned embeddings are the centred originals at
    their own scale, projected onto those axes, so they order dimensions the
    same way the participation ratio does.

    Returns
    -------
    dict
        ``'zx_rotated'`` and ``'zy_rotated'`` (``(N, min(d_x, d_y))``),
        ``'singular_values'``, ``'rotation_x'`` and ``'rotation_y'``. Apply a
        rotation to new data as ``zx_new @ rotation_x``.
    """
    svd = cross_covariance_svd(zx, zy, whitening)
    return {
        'zx_rotated': svd['zx_centered'] @ svd['rotation_x'],
        'zy_rotated': svd['zy_centered'] @ svd['rotation_y'],
        'singular_values': svd['singular_values'],
        'rotation_x': svd['rotation_x'],
        'rotation_y': svd['rotation_y'],
    }


def compute_spectral_metrics(spectrum: np.ndarray, eps: float = 1e-12) -> Dict[str, float]:
    """Computes dimensionality metrics from singular values."""
    s = np.array(spectrum)
    s = s[s > eps]
    
    metrics = {}
    
    # 1. Variance/Energy-based PR
    lam = s**2
    metrics["pr_eig"] = (lam.sum())**2 / (lam**2).sum() if lam.sum() > 0 else 0.0

    # 2. "Soft" PR (Based on Singular Values)
    metrics["pr_singular"] = (s.sum())**2 / (s**2).sum() if s.sum() > 0 else 0.0

    # 3. Effective Rank and Spectral Entropy
    if s.sum() > 0:
        p = s / s.sum()
        p = p[p > 0]  # guard against numerical zeros before log
        entropy = -np.sum(p * np.log(p))
        metrics["effective_rank"] = np.exp(entropy)
        metrics["spectral_entropy"] = float(entropy)
    else:
        metrics["effective_rank"] = 0.0
        metrics["spectral_entropy"] = 0.0

    return metrics


def warn_if_blocked_split_leaks(gap_size: int, block: int, step: float,
                                 span: float, gap_fraction: float,
                                 path_label: str = "") -> bool:
    """Warn if a blocked train/test split's gap doesn't fully separate rows in time.

    ``gap_size`` (a count of *rows* excluded on each side of a test block,
    already computed by the caller) buys a stretch of time ``gap_size * step``,
    where ``step`` is the distance between the starts of consecutive rows and
    ``span`` the stretch one row covers (:mod:`neural_mi.data.rows`). When rows
    overlap and the buffer is shorter than a row, a training row and a held-out
    row can share samples although their indices do not.

    Returns True if a leak was detected (and a warning raised), else False.
    """
    if step <= 0 or span <= 0:
        return False
    buffer = gap_size * step
    if buffer >= span:
        return False
    min_gap_fraction = (span / (block * step)) if (block > 0 and step > 0) else float('inf')
    _where = f" ({path_label})" if path_label else ""
    warnings.warn(
        f"Blocked split may leak raw samples between train and test{_where}. A gap of "
        f"{gap_size} rows, whose starts are {step:.3g} apart, buys {buffer:.3g} of separation "
        f"against the {span:.3g} one row covers, so a training row and a held-out row can share "
        f"up to {span - buffer:.3g} of it. Increase Split(gap_fraction=...) to at least "
        f"{min_gap_fraction:.4f} (currently {gap_fraction}) to eliminate this.",
        UserWarning, stacklevel=user_stacklevel()
    )
    return True


def warn_if_random_split_leaks(rows) -> None:
    """Warn that a random split of rows that share samples leaves each held-out row among training rows
    that nearly copy it (``rows`` is a :class:`neural_mi.data.rows.RowGeometry`)."""
    warnings.warn(
        f"Random split of rows that share samples: consecutive rows start {rows.step:.3g} apart and "
        f"each covers {rows.span:.3g}, so a held-out row shares samples with up to "
        f"{2 * (rows.rows_per_span - 1)} rows around it, most of them in training. The held-out "
        f"curve then follows the training rows and keeps rising as the network overfits, so the "
        f"epoch it chooses comes late and the value read there is high. Split(mode='blocked') "
        f"holds out contiguous blocks with a gap around each.",
        UserWarning, stacklevel=user_stacklevel()
    )
