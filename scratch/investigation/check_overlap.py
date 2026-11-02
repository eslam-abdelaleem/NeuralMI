"""Does mi_rate guard against grid rows that are wider than one step / overlap? (array construction only)"""
import warnings
import numpy as np
import torch
from collections import OrderedDict
from neural_mi.config import Processing
from neural_mi.quantities import _on_one_grid, _build_mi_rate_arrays
from neural_mi.analysis.offsets import one_step_rows

warnings.simplefilter('ignore')
T = 2000
t = np.arange(T, dtype=np.float32)
x = t.reshape(-1, 1)          # the sample index itself, so overlaps are visible
y = (t + 0.5).reshape(-1, 1)

proc = Processing(x='continuous', x_params={'window_size': 10, 'step_size': 2})
rows = _on_one_grid(OrderedDict((('x', x), ('y', y))), proc)
print("grid rows:", {k: tuple(v.shape) for k, v in rows.items()})
a, b, c = _build_mi_rate_arrays(rows['x'], rows['y'], h=2, half_width=1)
print("A (X_all) shape", tuple(a.shape), " B (Y_0) shape", tuple(b.shape), " C (Y_past) shape", tuple(c.shape))
i = 5
y0 = set(b[i].flatten().tolist())
ypast = set(c[i].flatten().tolist())
print(f"row {i}: Y_0 samples {sorted(y0)[:3]}..{sorted(y0)[-1]}; Y_past samples {sorted(ypast)[:3]}..{sorted(ypast)[-1]}")
print(f"samples of Y_0 that also appear in its conditioning set Y_past: {len(y0 & ypast)} of {len(y0)}")
try:
    one_step_rows(rows, "mode='transfer'")
except ValueError as e:
    print("transfer entropy would refuse these rows:", str(e)[:90], "...")
