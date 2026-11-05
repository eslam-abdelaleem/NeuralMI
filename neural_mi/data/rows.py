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

Each training task also reports which rows it read (:func:`row_identity`), so
that the terms of a difference, trained in separate tasks, can be checked to
have read the same rows (:func:`neural_mi.analysis.sweep.combine_terms`).
"""
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

STEP_KEY = 'row_step'
SPAN_KEY = 'row_span'
# Rows built by offset at every step of their series, of which each task keeps
# one in every _row_phases, at a phase drawn each epoch when windows shift.
PHASES_KEY = '_row_phases'
# A task's result carries the identity of its rows under this key.
IDENTITY_KEY = '_row_identity'


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


def offset_rows(n_steps: int, stride: int = 1) -> RowGeometry:
    """The rows sliced by offset from a series: each covers ``n_steps`` consecutive samples of the series,
    and consecutive rows start ``stride`` samples apart. The geometry is counted in samples."""
    return RowGeometry(step=float(stride), span=float(n_steps))


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


def row_identity(dataset, params: Dict[str, Any]) -> Dict[str, Any]:
    """Which rows a task reads, before any shift: the identity a :class:`~neural_mi.data.plan.RowPlan` gave
    the dataset (``count``, ``origin``, ``step``, each stream's window in ``spans`` and a digest of the
    kept rows' start times in ``kept``), or else ``count`` and the recorded ``step``. Rows built by
    offset and kept from phase 0 also know their ``origin``. Rows built before the task, passed in or
    windowed once by :func:`neural_mi.run`, are known by their count and the recorded geometry, and
    every task of a call receives the same ones."""
    given = getattr(dataset, '_row_identity', None)
    if given is not None:
        return dict(given)
    geometry = row_geometry(params)
    identity = {'count': len(dataset), 'origin': None, 'step': geometry.step if geometry else None,
                'spans': None, 'kept': None}
    if getattr(dataset, '_window_shifter', None) is not None:
        # Rows built by offset, kept from phase 0.
        identity['origin'] = 0.0
    return identity


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b))


def same_rows(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    """Whether two tasks' :func:`row_identity` name the same rows: equal counts, and equal origins, steps,
    kept rows and windows of Y wherever both identities know them."""
    if a['count'] != b['count']:
        return False
    for key in ('origin', 'step'):
        if a.get(key) is not None and b.get(key) is not None and not _close(a[key], b[key]):
            return False
    if a.get('kept') and b.get('kept') and a['kept'] != b['kept']:
        return False
    span_a, span_b = (a.get('spans') or {}).get('y'), (b.get('spans') or {}).get('y')
    return span_a is None or span_b is None or _close(span_a, span_b)


def describe_rows(identity: Dict[str, Any]) -> str:
    """A :func:`row_identity` in words, for a message."""
    text = f"{identity['count']} rows"
    if identity.get('origin') is not None:
        text += f" starting at {identity['origin']:.6g}"
    if identity.get('step') is not None:
        text += f", one every {identity['step']:.6g}"
    span = (identity.get('spans') or {}).get('y')
    if span is not None:
        text += f", Y over {span:.6g}"
    return text
