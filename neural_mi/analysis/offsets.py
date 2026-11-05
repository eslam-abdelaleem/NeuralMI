# neural_mi/analysis/offsets.py
"""The one path from streams to the rows of every quantity defined by time offsets.

Every such quantity is :math:`I(A; B \\mid C)` for a choice of processes and
offsets in each group, a spec (:func:`neural_mi.utils.build_offset_arrays`).
Its streams reach the builder as ``(T, C)`` series in one of two ways: as
given, when they are regularly sampled already, or through ``processing=``,
which puts every stream on one grid of rows one step wide that tile time
(:func:`one_step_series`). The rows a spec builds carry their geometry
(:func:`spec_geometry`), which :func:`neural_mi.run` records for everything
that depends on the rows' time order (:mod:`neural_mi.data.rows`).
"""
import warnings
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple

from neural_mi.data.rows import RowGeometry, offset_rows
from neural_mi.logger import user_stacklevel
from neural_mi.utils import build_offset_arrays


def spec_geometry(spec: Dict[str, Any], stride: int = 1) -> RowGeometry:
    """The geometry of the rows a spec builds: each row covers the steps from the earliest offset of any
    group to the latest, and consecutive rows start ``stride`` steps apart."""
    offsets = [o for group in ('A', 'B', 'C') for (_, o) in (spec.get(group) or [])]
    return offset_rows(max(offsets) - min(offsets) + 1, stride)


def build_rows(series: Dict[str, Any], spec: Dict[str, Any],
               stride: int = 1) -> Tuple[Any, Any, Optional[Any], RowGeometry]:
    """The rows a spec defines on ``series``, and their geometry: the one builder of rows by offset.

    Returns ``(a, b, c, geometry)``, with ``c`` None when the spec has no C
    group (:func:`neural_mi.utils.build_offset_arrays`, :func:`spec_geometry`).
    """
    a, b, c, _ = build_offset_arrays(series, spec, stride=stride)
    return a, b, c, spec_geometry(spec, stride)


def resolve_stream_processing(processing) -> dict:
    """Each stream's processor, parameters and clock from a ``Processing``.

    A stream with no processor of its own reads with X's, and one with no
    parameters of its own reads with X's parameters, kept to the keys its
    processor takes. Returns ``{name: (processor_type, params, time)}`` for
    ``'x'``, ``'y'`` and ``'w'``.
    """
    from neural_mi.config import Processing
    from neural_mi.defaults import stream_processing
    if isinstance(processing, dict):
        processing = Processing(**processing)
    out = {'x': (processing.x, dict(processing.x_params or {}), processing.x_time)}
    for name in ('y', 'w'):
        proc, params = stream_processing(getattr(processing, name), getattr(processing, f'{name}_params'),
                                         processing.x, processing.x_params)
        out[name] = (proc, dict(params or {}), getattr(processing, f'{name}_time'))
    return out


def grid_rows(specs):
    """Put every stream on one grid and hand back its rows.

    Offsets index rows and read a fixed interval into each step, which holds
    only while the rows are uniformly spaced and mean the same instant in every
    stream. A raw ``(T, C)`` array satisfies that by being a grid already. Any
    other modality does not: spike times are a point process until they are
    binned, and two streams on different clocks share no row. The streams are
    built onto one shared grid by their processors and handed back as
    ``(N, C, w)`` rows.

    Validation is off on purpose. Dropping windows compacts the row axis and so
    destroys the uniform spacing offsets depend on: a 40 s gap in a 1 s grid
    turns the steps into ``{1.0, 41.0}``, and an offset of five rows then means
    five seconds in one place and forty-five in another. Keeping every row
    leaves the gap visible to the coverage warnings.

    Parameters
    ----------
    specs : mapping
        ``{name: dict(data=..., time=..., processor_type=..., processor_params=...)}``,
        in the order the streams should be built. Streams whose data is None
        are skipped.

    Returns
    -------
    tuple
        ``(rows, geometry)``: the rows by stream name, and the geometry of the
        grid's windows (:mod:`neural_mi.data.rows`) in the grid's own unit.
    """
    from neural_mi.data.handler import create_dataset
    from neural_mi.data.rows import window_rows
    specs = OrderedDict((n, s) for n, s in specs.items() if s.get('data') is not None)
    for name, spec in specs.items():
        if (spec.get('processor_type') == 'spike'
                and (spec.get('processor_params') or {}).get('drop_empty_windows') is True):
            warnings.warn(
                f"drop_empty_windows=True has no effect on stream {name!r} here. The rows a "
                f"quantity defined by time offsets builds keep every window, since dropping one "
                f"would put two distant windows next to each other and change what an offset means.",
                UserWarning, stacklevel=user_stacklevel(),
            )
    bundle = create_dataset(specs, validate_windows=False)
    rows = OrderedDict((name, bundle.stream(name).data) for name in bundle.stream_names)
    manager = getattr(bundle, 'window_manager', None)
    geometry = window_rows(manager.window_size, manager.resolve_step()) if manager is not None else None
    return rows, geometry


def one_step_series(specs, caller: str) -> 'OrderedDict[str, Any]':
    """Every stream on one grid of rows one step wide that tile time, as ``(T, C)`` series: the one way
    ``processing=`` reaches a quantity defined by time offsets.

    An offset counts rows, so a row has to be one time step, and the rows have
    to tile time. With rows wider than one step an offset of one row would skip
    the steps inside a row, and with rows that overlap it would reach back into
    the row before, so a present would share samples with its own past. A
    resolution-1 grid gives one bin per row for spikes and one sample per row
    for continuous data. A categorical stream spends its trailing axis on
    categories, which cannot be flattened without changing what a lag means, so
    it is refused.
    """
    rows, grid = grid_rows(specs)
    wide = {n: r.shape[-1] for n, r in rows.items() if r.shape[-1] != 1}
    if wide:
        raise ValueError(
            f"{caller} needs a grid whose rows are one time step wide. "
            f"{sorted(wide)} came back wider than that ({wide}). Set each stream's "
            f"window_size to one step (bin_size for spikes, 1/sample_rate for "
            f"continuous). A categorical stream cannot satisfy this because its "
            f"encoder uses that axis for categories."
        )
    if grid is not None and grid.overlapping:
        raise ValueError(
            f"{caller} needs a grid whose rows tile time. Its windows of {grid.span:g} start "
            f"every {grid.step:g}, so neighbouring rows share samples and an offset of one row "
            f"reaches back into the row before. Leave step_size unset, which steps by one window."
        )
    return OrderedDict((n, r.squeeze(-1)) for n, r in rows.items())
