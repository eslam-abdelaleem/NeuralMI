# neural_mi/data/rows.py
"""How the rows of a call lie in time.

Rows built from a time series follow each other in time, and neighbouring rows
can share samples. Whatever builds them records how, once: the distance between
the starts of consecutive rows (``row_step``) and the stretch one row covers,
from its first sample to its last (``row_span``), both in one unit. The
processors record their windows, the named quantities the offsets they slice
(through :func:`neural_mi.run`), and transfer entropy its histories. Rows
without the record are not known to be in time order: arrays passed in already
built, or samples in no order.

What depends on the time order reads the record and nothing else: the leak
checks of the split (``Trainer``), the chunks of a rigorous ladder
(:mod:`neural_mi.analysis.rigorous`) and the blocks of a block shuffle
(:mod:`neural_mi.analysis.permutation`).
"""
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

STEP_KEY = 'row_step'
SPAN_KEY = 'row_span'
# Rows built by offset at every step of their series, of which each task keeps
# one in every _row_phases, at a phase drawn each epoch when windows shift.
PHASES_KEY = '_row_phases'


@dataclass(frozen=True)
class RowGeometry:
    """Rows in time order: consecutive rows start ``step`` apart, and each covers ``span``, in one unit."""
    step: float
    span: float

    def __post_init__(self):
        if not (self.step > 0 and self.span > 0):
            raise ValueError(f"A row step and span must be positive, got step={self.step}, span={self.span}.")

    @property
    def rows_per_span(self) -> int:
        """The consecutive rows that one row's span takes in, itself included: the rows it can share
        samples with on one side, plus one."""
        return max(1, math.ceil(self.span / self.step - 1e-9))

    @property
    def overlapping(self) -> bool:
        """Whether neighbouring rows share samples."""
        return self.span > self.step + 1e-12


def window_rows(window_size: float, step: float) -> RowGeometry:
    """The rows a processor cuts: windows of ``window_size`` every ``step``, in the processor's unit."""
    return RowGeometry(step=float(step), span=float(window_size))


def offset_rows(n_steps: int, stride: int = 1, series: Optional[RowGeometry] = None) -> RowGeometry:
    """The rows sliced by offset from a series: each covers ``n_steps`` consecutive steps of the series,
    and consecutive rows start ``stride`` steps apart.

    ``series`` is the geometry of the series' own rows when they are the windows of a grid
    (:func:`neural_mi.analysis.offsets.grid_rows`). Without it each step is one sample, and the
    geometry is counted in samples.
    """
    series = series or RowGeometry(1.0, 1.0)
    return RowGeometry(step=stride * series.step, span=(n_steps - 1) * series.step + series.span)


def record_rows(params: Dict[str, Any], geometry: Optional[RowGeometry]) -> None:
    """Write ``geometry`` into ``params``, where every reader finds it. ``None`` writes nothing."""
    if geometry is not None:
        params[STEP_KEY] = geometry.step
        params[SPAN_KEY] = geometry.span


def row_geometry(params: Dict[str, Any]) -> Optional[RowGeometry]:
    """The geometry recorded in ``params``, or ``None`` when its rows are not known to be in time order."""
    step, span = params.get(STEP_KEY), params.get(SPAN_KEY)
    if step is None or span is None:
        return None
    return RowGeometry(float(step), float(span))


def rows_in_time(params: Dict[str, Any]) -> bool:
    """Whether the rows ``params`` describes are known to be in time order."""
    return row_geometry(params) is not None


def kept_rows(n_rows: int, params: Dict[str, Any]) -> int:
    """How many of ``n_rows`` passed rows a task keeps: all of them, or, for rows built at every step of
    their series so that they can shift (``_row_phases``), one in every ``_row_phases`` at any phase."""
    phases = int(params.get(PHASES_KEY) or 1)
    return n_rows if phases <= 1 else max(0, (n_rows - phases) // phases + 1)
