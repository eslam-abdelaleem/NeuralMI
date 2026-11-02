"""Are instantaneous_exchange(k=0) / directed_information_rate(k=0) the same rows as instantaneous_mi?
Raw arrays, and spike times through processing= (array construction only, no training)."""
import warnings
from collections import OrderedDict
import numpy as np
import torch
from neural_mi.config import Processing
from neural_mi.data.handler import create_dataset
from neural_mi.quantities import (_on_one_grid, _build_inst_exchange_arrays, _build_dir_info_rate_arrays,
                                  _build_mi_rate_arrays)
from neural_mi.analysis.offsets import as_rows

warnings.simplefilter('ignore')
rng = np.random.default_rng(0)
x = rng.standard_normal((500, 2)).astype(np.float32)
y = rng.standard_normal((500, 2)).astype(np.float32)
a_ie, b_ie, _ = _build_inst_exchange_arrays(x, y, k=0)
a_di, b_di, _ = _build_dir_info_rate_arrays(x, y, k=0)
a_mi, b_mi = as_rows(x), as_rows(y)                # what instantaneous_mi trains on (run's fast path)
print("raw arrays: IE(k=0) == instantaneous_mi rows:", torch.equal(a_ie, a_mi) and torch.equal(b_ie, b_mi),
      "| DIR(k=0) == instantaneous_mi rows:", torch.equal(a_di, a_mi) and torch.equal(b_di, b_mi))
a_r, b_r, c_r = _build_mi_rate_arrays(x, y, h=0, half_width=0)
print("raw arrays: mi_rate(h=0, half_width=0) == instantaneous_mi rows:", torch.equal(a_r, a_mi) and torch.equal(b_r, b_mi))
a_r3, _, _ = _build_mi_rate_arrays(x, y, h=0, half_width=3)
print("mi_rate(h=0, half_width=3) X rows:", tuple(a_r3.shape), "-> I(X_{-3..3}; Y_0), no named quantity")

# spike times, sparse enough that many 10 ms windows are silent
T = 60.0
spikes_x = [np.sort(rng.uniform(0, T, 300)) for _ in range(2)]
spikes_y = [np.sort(rng.uniform(0, T, 300)) for _ in range(2)]
params = {'window_size': 0.01, 'bin_size': 0.01}
proc = Processing(x='spike', x_params=params)
rows = _on_one_grid(OrderedDict((('x', spikes_x), ('y', spikes_y))), proc)       # IE/DIR route
ds = create_dataset(x_data=spikes_x, y_data=spikes_y, processor_type_x='spike', processor_type_y='spike',
                    processor_params_x=params, processor_params_y=params)       # instantaneous_mi route (run)
print(f"spikes, 10 ms windows: IE/DIR(k=0) route builds {rows['x'].shape[0]} rows; "
      f"instantaneous_mi route builds {ds.x_data.shape[0]} rows")
