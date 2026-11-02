"""How each named quantity's builder treats degenerate window parameters (array construction only)."""
import warnings
import numpy as np
from neural_mi.analysis.offsets import build_past_future, build_cross_offset
from neural_mi.analysis.transfer import _build_te_arrays
from neural_mi.quantities import (_build_mi_rate_arrays, _build_inst_exchange_arrays,
                                  _build_dir_info_rate_arrays)

warnings.simplefilter('ignore')
x = np.random.default_rng(0).standard_normal((200, 2)).astype(np.float32)
y = np.random.default_rng(1).standard_normal((200, 2)).astype(np.float32)

cases = {
    'AIS k=0 (build_past_future past_len=0)': lambda: build_past_future(x, past_len=0, future_len=1),
    'AIS k=-1': lambda: build_past_future(x, past_len=-1, future_len=1),
    'AIS k=2.5': lambda: build_past_future(x, past_len=2.5, future_len=1),
    'cross-predictive k=0': lambda: build_cross_offset(x, y, past_len=0, future_len=0),
    'TE history_window=0': lambda: _build_te_arrays(x, y, history_window=0),
    'TE history_window=-2': lambda: _build_te_arrays(x, y, history_window=-2),
    'mi_rate h=0 (allowed: plain MI)': lambda: _build_mi_rate_arrays(x, y, h=0, half_width=3),
    'mi_rate h=-1': lambda: _build_mi_rate_arrays(x, y, h=-1, half_width=3),
    'mi_rate half_width=0': lambda: _build_mi_rate_arrays(x, y, h=2, half_width=0),
    'mi_rate half_width=-1': lambda: _build_mi_rate_arrays(x, y, h=2, half_width=-1),
    'mi_rate h=2.5': lambda: _build_mi_rate_arrays(x, y, h=2.5, half_width=3),
    'inst_exchange k=0 (allowed: plain MI)': lambda: _build_inst_exchange_arrays(x, y, k=0),
    'inst_exchange k=-1': lambda: _build_inst_exchange_arrays(x, y, k=-1),
    'dir_info_rate k=-1': lambda: _build_dir_info_rate_arrays(x, y, k=-1),
}
for name, fn in cases.items():
    try:
        out = fn()
        shapes = [None if t is None else tuple(t.shape) for t in out]
        print(f"ACCEPTED  {name:42s} -> shapes {shapes}")
    except Exception as e:  # noqa
        print(f"refused   {name:42s} -> {type(e).__name__}: {str(e)[:95]}")
