# Using NeuralMI

This guide shows how to call NeuralMI for each task and how to read what comes
back. [PARAMETERS.md](PARAMETERS.md) lists every setting with its default,
[THEORY.md](THEORY.md) explains what each reported quantity means and why the
estimators behave as they do, and [MESSAGES.md](MESSAGES.md) explains every
warning the library prints. [INTERNALS.md](INTERNALS.md) is for people changing
the library.

## Contents

- [Installing and a first estimate](#installing-and-a-first-estimate)
- [What comes back](#what-comes-back)
- [Choosing the analysis](#choosing-the-analysis)
- [Repeats, grids and tests](#repeats-grids-and-tests)
- [Preparing the data](#preparing-the-data)
- [The named quantities](#the-named-quantities)
- [Models, estimators and critics](#models-estimators-and-critics)
- [Plotting and animation](#plotting-and-animation)
- [Speed, memory and logging](#speed-memory-and-logging)

---

## Installing and a first estimate

NeuralMI needs Python 3.11 or newer and PyTorch 2.4 or newer. From the
repository root:

```bash
pip install .
```

The `[viz]` extra adds scikit-learn and umap-learn for the embedding plots.
`[vision]` adds torchvision for `embedding_model='pretrained_backbone'`.

```python
import numpy as np
import neural_mi as nmi

rng = np.random.default_rng(0)
x = rng.standard_normal((5000, 4))
y = 0.7 * x + 0.3 * rng.standard_normal((5000, 4))

result = nmi.run(x, y, mode='estimate', seed=0)
print(result.mi_estimate)
result.summary()
```

`run()` holds out 10% of the samples in contiguous blocks, trains a critic
network on the rest, finds the epoch at which the smoothed MI on the held-out
blocks peaks, and reports the MI on the training samples at that epoch as
`mi_estimate`. The value is in bits unless `Output(units='nats')` asks for
nats. [THEORY.md](THEORY.md#the-reported-number) explains why the training side
is the value reported.

Every setting belongs to a config object: `Processing` for how raw data are
read, `Model` for the networks, `Training` for the optimisation loop, `Split`
for the train and test sets, `Estimator` for the bound, `Output` for units and
extras, and one config per analysis mode. Every config is optional and can be
replaced by a plain `dict` with the same keys.

```python
from neural_mi import Model, Training, Split, Output

result = nmi.run(x, y, mode='estimate',
                 model=Model(embedding_dim=32, hidden_dim=128),
                 training=Training(n_epochs=100, batch_size=256),
                 split=Split(mode='random'),
                 output=Output(units='nats'),
                 seed=0)
```

---

## What comes back

Every call returns a `Results` object with the same fields in every mode.

| field | holds |
|---|---|
| `mode` | the analysis mode that produced it |
| `mi_estimate` | the headline value of a one-row result (`None` otherwise) |
| `dataframe` | one row per configuration and per value of the mode's axis |
| `runs` | one row per repeat of the procedure |
| `details` | structured diagnostics, keyed by `config_id` |
| `params` | the full configuration the call ran with, defaults included |

A *configuration* is one combination of the values in `sweep_grid`. A call
without a grid has the single configuration `config_id=0`. Some modes add an
*axis* inside each configuration: `lag` for `mode='lag'`, `tau` for
`mode='precision'`, `ch_x`, `ch_y` for `mode='pairwise'`, and `split_id`,
`embedding_dim` for `mode='dimensionality'`. A *repeat* is one independent run
of the whole procedure on the same data. Repeats are indexed by `run_id`.
`params['config_keys']` and
`params['axis_keys']` name the columns that index `dataframe`.

`dataframe` always has `config_id`, the grid and axis keys, `mi_mean`, `mi_std`,
`n_runs` and `n_zero`, followed by the columns of the mode (listed per mode in
[Choosing the analysis](#choosing-the-analysis)). `mi_mean` and `mi_std` use the
repeats that produced a value. A repeat reported as 0 produced nothing and is
counted in `n_zero` ([THEORY.md](THEORY.md#the-spread-over-repeats)). `mi_std`
is the spread over repeats of the procedure on the same data and is NaN below
two repeats. It says how much the estimate moves when the networks are retrained
and is never an interval on the population value
([THEORY.md](THEORY.md#the-spread-over-repeats)).

`runs` has one row per repeat holding the repeat's value `mi` and the
diagnostics of the network behind it: `test_mi`, `best_epoch`, `eval_size`,
`train_eval_size`, the ceilings `train_ceiling_mi` and `test_ceiling_mi`, how
close each side came to its ceiling (`train_saturation`, `test_saturation`),
`window_retention` and the window counts for windowed data, and the held-out MI
at every epoch in `test_mi_history`. A repeat that trains several networks (the
difference quantities and the rigorous ladder) keeps each network's diagnostics
in `details[config_id]['trainings']`.

`mi_estimate` is `None` whenever `dataframe` has more than one row. A lag scan,
a pairwise matrix, a precision curve or any grid over several configurations is
read from `dataframe`.

### Reading one value

`result.get(key)` looks in `dataframe`, then `details`, then the kept
embeddings, then `runs`, and returns the value when exactly one row,
configuration or repeat holds it. When several do, it raises and names the table
to read.

```python
result.get('test_mi_mean')            # one configuration: a number
result.get('amplification_factor')    # the difference quantities
result.get('eval_size')               # one repeat: a number
result.get('no_such_key', default=0)  # nothing holds it: the default
```

### Printing, saving and exporting

`result.summary()` prints the headline, its spread, the components of a
difference quantity, and the reliability of a rigorous fit with the checks that
decided it.

`result.save(path)` pickles the whole object and returns the absolute path. With
no path or a directory the file is named
`neuralmi_{mode}_{YYYYMMDD_HHMMSS}.pkl`. An existing file is never overwritten.
`Results.load(path)` reads it back.

`result.to_dict()` turns every field into JSON-ready values: arrays into nested
lists and tables into lists of records. `result.to_json(path)` writes it to a
file named the same way. Use `save()` and `load()` for an exact round trip.

### Embeddings

`Output(return_embeddings=True)` keeps each repeat's embedding of every window
in original sample order. The arrays line up with the windowed data and with any
labels indexed the same way. They are keyed in `details[config_id]['embeddings']`
by `run_id`, by `(lag, run_id)` in `mode='lag'`, by `(ch_x, ch_y, run_id)` in
`mode='pairwise'` and by `(split_id, embedding_dim, run_id)` in
`mode='dimensionality'`. Each entry holds `embeddings_x` and `embeddings_y`.
`mode='dimensionality'` also keeps the best restart's embeddings at each split's
reading in `details[config_id]['embeddings_at_bound']`.
`Output(return_rotated_embeddings=True)` adds versions rotated
so that dimension 0 carries the most variance shared between X and Y.
`Output(track_embeddings=True)` keeps them at every epoch for
`result.animate()`.

Embeddings are available where a repeat is one network: `estimate`, `sweep`,
`lag`, `pairwise` and `dimensionality`. The difference quantities, `rigorous`
and `precision` refuse the setting because they train several networks per
repeat or evaluate one network many times.

### Saving the trained networks

`Training(save_best_model_path=...)` saves the best epoch of every network a
call trains together with the settings needed to rebuild it.
`nmi.extract_embeddings(path, x, y)` later embeds new data with a saved network,
windowed as the training data were.

A call that trains one network writes to the path as given, adding `.pt` when it
has no extension. A call that trains several saves each under the path with its
identifying labels added to the name: its grid values and `run_id`, plus its
`gamma` and `chunk`, `component`, `lag`, channel pair or `split_id` where the
call has them.

```python
result = nmi.run(x, y, mode='sweep',
                 sweep_grid={'embedding_dim': [16, 32], 'run_id': range(2)},
                 training=Training(save_best_model_path='models/best.pt'))
# models/best_embedding_dim-16_run_id-0.pt, ..., models/best_embedding_dim-32_run_id-1.pt
result.runs[['embedding_dim', 'run_id', 'model_path']]
```

Each path is recorded as `model_path` beside its network's row in `result.runs`
(in `details[config_id]['trainings']` where a repeat trains several networks). A
single saved network's path is `result.get('model_path')`. A directory in place
of a file name saves under the generated name `neuralmi_<mode>_<timestamp>.pt`
with the same labels. Because a large grid or rigorous ladder of whole networks
can fill a disk, a call that saves several networks warns before training and
afterwards logs how many it saved and their total size. To keep one model, rerun
the configuration you want on its own. Permutation trials are never saved.

---

## Choosing the analysis

| mode | reports | headline |
|---|---|---|
| `'estimate'` | $I(X;Y)$ from one trained network | `mi_estimate` |
| `'sweep'` | $I(X;Y)$ for every configuration of `sweep_grid` | `dataframe` |
| `'rigorous'` | $I(X;Y)$ extrapolated to infinite data | `mi_estimate`, `runs['is_reliable']` |
| `'lag'` | $I(X;Y)$ with Y shifted by each lag | `dataframe` over `lag` |
| `'precision'` | how MI falls as X is coarsened in time or value | `get('precision_tau')` |
| `'conditional'` | $I(X;Y \mid W)$ | `mi_estimate` |
| `'interaction'` | $I(X,W;Y) - I(X;Y) - I(W;Y)$ | `mi_estimate` |
| `'transfer'` | transfer entropy from X to Y | `mi_estimate` |
| `'pairwise'` | MI between every pair of channels | `get('mi_matrix')` |
| `'dimensionality'` | the smallest embedding size that carries 95% of the MI | `get('dimension_at_most')` |

The functions of [the named quantities](#the-named-quantities) build their
arrays, call one of these modes and return its `Results`.

### `estimate`

```python
result = nmi.run(x, y, mode='estimate', training=Training(n_epochs=100))
result.mi_estimate
result.plot()          # held-out MI against epoch, with the chosen epoch marked
```

`dataframe` adds the held-out MI at the chosen epoch as `test_mi_mean`.
`mode='estimate'` runs one configuration once and ignores a `sweep_grid` with a
warning. `mode='sweep'` runs the grid.

### `sweep`

```python
result = nmi.run(x, y, mode='sweep',
                 sweep_grid={'embedding_dim': [16, 32, 64], 'run_id': range(3)},
                 n_workers=4)
result.dataframe[['embedding_dim', 'mi_mean', 'mi_std', 'n_runs']]
result.plot()          # MI against embedding_dim, shaded by the spread over repeats
```

Any setting of `Model`, `Training`, `Split`, `Estimator` or the processor
parameters can be a key of `sweep_grid`. `run_id` adds repeats. In a grid,
`Split(mode=...)` is `split_mode`, `Split(gap_fraction=...)` is
`split_gap_fraction`, and `Estimator(name=...)` and `Estimator(params=...)` are
`estimator_name` and `estimator_params`. Because they would run the same call
under different labels, the grid refuses a key that no setting reads and a key
the mode reads only once from its config (such as `history_window`). A sweep
over the capacity of the network (`embedding_dim`, `hidden_dim`, `n_layers`) and
over `n_epochs` is the way to find a configuration whose estimate has stopped
moving before committing to a rigorous run.

### `rigorous`

```python
from neural_mi import Rigorous

result = nmi.run(x, y, mode='rigorous',
                 rigorous=Rigorous(gamma_range=range(1, 11), confidence_level=0.95),
                 n_workers=4)
print(f"{result.mi_estimate:.3f} ± {result.get('mi_error'):.3f} bits,"
      f" reliable: {result.get('is_reliable')}")
result.plot()          # the ladder, the fit and the extrapolated point
```

At each $\gamma$ in `gamma_range` the data are cut into $\gamma$ chunks, one
network is trained per chunk, and the MI is fitted against $\gamma$ and
extrapolated to the infinite-data limit at $\gamma = 0$
([THEORY.md](THEORY.md#correcting-it)). One call trains $\sum \gamma$ networks,
55 for the default range.

`runs` holds each repeat's fit: the extrapolated value `mi`, the fit's
half-width `mi_error` and prediction half-width `mi_error_pred`, the `slope`,
`is_reliable`, the checks behind it (`enough_gamma_points`,
`linear_region_found`, `leverage_warning`, `saturated_gammas`), the diagnostics
reported beside it (`fit_quality_warning`, `r_squared`, `max_abs_residual`, and
`zero_rungs` for the rungs that produced nothing), and the curvature statistics
the linear region was chosen on. When the fit extrapolates below zero `mi` is 0
and `mi_raw` holds the fitted value. `details[config_id]['trainings']` holds
every network of the ladder.

`is_reliable` is `False` when the fit used fewer than `min_gamma_points` values
of $\gamma$, when no linear region was found, when leaving out $\gamma = 1$
shifts the intercept by more than `leverage_threshold`, and, in
`mode='rigorous'`, when a $\gamma$ in the fit sits at its estimator's ceiling.
An unreliable fit should not be reported even with a caveat. A reliable one
means the part of the bias that shrinks with more data has been removed and the
fit is well determined. Bias that is the same at every chunk size passes through
the extrapolation untouched ([THEORY.md](THEORY.md#correcting-it)).

At $\gamma = k$ each chunk holds about $N/k$ samples. When a chunk's held-out
part can no longer fill one evaluation batch, the run warns. The rungs beyond
that point are noise that a straight line can still pass through.
`Training(min_reliable_samples=...)` moves the point at which it warns.

With repeats, `mi_mean` and `mi_std` are the mean and spread of the repeats'
extrapolations. `dataframe['n_reliable']` counts the reliable fits.
`dataframe['mi_error']` is filled for a single repeat only because the intervals
of repeats that share data cannot be combined. Windowed time-series data are cut
into the same contiguous chunks for every repeat. Static data are reordered at
random for each repeat.

### `lag`

```python
from neural_mi import Lag

result = nmi.run(x, y, mode='lag', lag=Lag(lag_range=range(-20, 21)), n_workers=8)
df = result.dataframe
peak = df.loc[df['mi_mean'].idxmax(), 'lag']
result.plot()
```

A positive lag compares X with Y's future. Lags are in seconds for spike data
and for streams with a `sample_rate` and in samples for everything else ([units
of time](#units-of-time)). `dataframe` adds `test_mi_mean` and the number of
windows at each lag in `n_windows_built`. Longer lags leave fewer samples.
`Lag(equalize_n=True)` cuts every lag to the sample count of the largest one so
that all lags use the same amount of data.

### `precision`

```python
from neural_mi import Precision, Processing

result = nmi.run(spikes, behaviour, mode='precision',
                 processing=Processing(x='spike', x_params={'window_size': 0.05},
                                       y='continuous', y_time=behaviour_t),
                 precision=Precision(tau_grid=[0.001, 0.002, 0.005, 0.01, 0.02, 0.05],
                                     threshold_ratio=[0.9, 0.5]))
result.get('baseline_mi')
result.get('precision_tau')                  # None when MI never fell below the threshold
result.get('precision_thresholds')           # one entry per ratio
result.plot()
```

`mode='precision'` evaluates one network trained on clean data and then frozen
on data corrupted at each $\tau$ in `tau_grid`. The `'rounding'` method moves
every value to the centre of its bin of width $\tau$. `'noise'` adds uniform
jitter on $[-\tau/2, \tau/2]$ and averages over `n_noise_samples` draws. The
precision is the smallest $\tau$ at which the MI falls below `threshold_ratio`
times the baseline ([THEORY.md](THEORY.md#spike-timing-precision)).
`corrupt_target` chooses whether X, Y or both are corrupted.
`Split(train_indices=..., test_indices=...)` fixes the split the baseline is
trained on.

Both methods corrupt only the entries that hold a measurement. A spike window
holds its spike times in a fixed number of slots and fills the unused ones with
`no_spike_value`. Those slots, empty bins and the zeros that pad a gap in a
continuous recording stay as they are. Neither method creates a spike or moves
one onto the empty value. On spike data with `bin_size` the mode degrades the
counts in each bin and leaves the spike times alone. Timing precision is
measured on the spike-time representation without `bin_size`.

`dataframe` has one row per $\tau$ including $\tau = 0$. `mi_estimate` is
therefore `None` and the baseline is `get('baseline_mi')`. `runs` has one row
per evaluation (one per noise draw under `'noise'`).

Past the threshold the curve leaves the region where the estimator can be read.
Because the frozen critic saw only clean inputs, a lower bound such as InfoNCE
can fall arbitrarily far below zero on the corrupted ones. Points below
zero are reported as 0 with a warning and keep their measured values in
`mi_raw`. The reading is the $\tau$ at which the curve crosses the threshold.

`mode='precision'` runs one configuration once and ignores `sweep_grid` and
repeats with a warning. It refuses `rigorous=True`, `permutation_test=True` and
`return_embeddings`.

### `conditional`

```python
from neural_mi import Conditional

result = nmi.run(x, y, mode='conditional', conditional=Conditional(w_data=w),
                 sweep_grid={'run_id': range(5)}, n_workers=4)
result.mi_estimate, result.get('mi_std')
result.dataframe[['mi_xw_y_mean', 'mi_w_y_mean', 'amplification_factor']]
result.plot()          # the components beside the difference
```

$I(X;Y \mid W)$ is computed repeat by repeat as $I(X,W;Y) - I(W;Y)$ from two
networks. `runs` holds each repeat's value and its components `mi_xw_y` and
`mi_w_y`. A repeat that comes out negative is reported as 0 and keeps the
measured difference in `mi_raw`. `dataframe` adds the component means and the
`amplification_factor` by which the difference magnifies the components'
relative error. A factor of 10 or more warns because a difference that small
should be read with its components ([THEORY.md](THEORY.md#amplification)).

`Conditional(align='dual_branch')` embeds W in its own branch, for a W whose
window length differs from X's. It needs `Model(embedding_model='dual_branch')`.
The named quantities that use it set `align` themselves and ask for the model
([the named quantities](#the-named-quantities)).

### `interaction`

```python
from neural_mi import Interaction

result = nmi.run(x, y, mode='interaction', interaction=Interaction(w_data=w))
result.mi_estimate                      # negative for redundancy, positive for synergy
result.dataframe[['mi_xw_y_mean', 'mi_x_y_mean', 'mi_w_y_mean', 'amplification_factor']]
```

Interaction information $I(X,W;Y) - I(X;Y) - I(W;Y)$
([THEORY.md](THEORY.md#interaction-information)) combines three networks per
repeat and is reported as measured with its sign. Its amplification factor is
the sum of the three components' magnitudes over the result's magnitude. A
three-term combination reaches a large factor more readily than a two-term one.

### `transfer`

```python
from neural_mi import Transfer

result = nmi.run(x, y, mode='transfer',
                 transfer=Transfer(history_window=10, bidirectional=True),
                 n_workers=4)
result.mi_estimate                                   # TE from X to Y
result.dataframe[['te_yx_mean', 'directionality_index_mean']]
```

Transfer entropy from X to Y is $I(X_{past}, Y_{past}; Y_{fut}) - I(Y_{past};
Y_{fut})$ for histories of `history_window` rows and a future
`prediction_horizon` rows ahead. Its components are the columns
`i_xypast_yfuture` and `i_ypast_yfuture`. `Transfer(bidirectional=True)` also
estimates the direction from Y to X (`te_yx`, with components `i_yxpast_xfuture`
and `i_xpast_xfuture` and its own `amplification_factor_yx`) and the
directionality index $(\mathrm{TE}_{X \to Y} - \mathrm{TE}_{Y \to X}) /
(|\mathrm{TE}_{X \to Y}| + |\mathrm{TE}_{Y \to X}|)$. A repeat that comes out
negative is reported as 0 and keeps the measured value in `mi_raw` (`te_yx_raw`
for the reverse direction). Because transfer entropy measures predictive
information flow, a large value in one direction is no evidence of a causal
influence in that direction
([THEORY.md](THEORY.md#temporal-information-quantities)).

`Transfer(w_data=w)` conditions both components on W's history as well, for
conditional transfer entropy. `details[config_id]['n_samples']` gives the number
of history windows built.

The histories are built row by row from `(T, n_channels)` input series. With
`processing=` every stream is first put on one grid whose rows are one time step
wide ([one grid](#one-grid-for-every-stream)). `window_size` must then be one
step: `bin_size` for spikes and one sample for continuous data. Categorical
streams spend that axis on categories and are refused.

### `pairwise`

```python
result = nmi.run(x, mode='pairwise', n_workers=8)          # every pair of X's channels
result = nmi.run(x, y, mode='pairwise', n_workers=8)       # X's channels against Y's
matrix = result.get('mi_matrix')
result.plot()                                               # the matrix as a heatmap
```

`mode='pairwise'` trains one network per pair of channels. `dataframe` has one
row per pair with repeat-averaged values that `mi_matrix` arranges as a matrix.
Without `y_data` each pair is estimated once and written to both halves of a
matrix with a zero diagonal. Adding the matrix to its transpose would double
every entry. With `Pairwise(pairs=[(0, 1), (2, 5)])` only the listed pairs are
estimated and the other entries stay 0. `Output(channel_names_x=...)` labels the
heatmap.

### `dimensionality`

```python
from neural_mi import Dimensionality

result = nmi.run(x, mode='dimensionality',
                 dimensionality=Dimensionality(n_splits=5), n_workers=4)
result.get('dimension_at_most'), result.get('dimension_at_most_std')
result.plot()          # MI against embedding_dim, with the reading
```

`mode='dimensionality'` estimates the MI at a series of embedding sizes and
reports the smallest one whose curve reaches `saturation_ratio` (0.95) of its
plateau ([THEORY.md](THEORY.md#the-dimension-that-carries-the-shared-information)).
The reading bounds from above the number of dimensions needed to carry that
fraction of the shared information. Combinations of true factors that an encoder
with spare capacity builds look like factors in its spectrum and add nothing to
the MI. The participation ratio counts them and the curve does not.

A large reference fit runs first (`reference_dim`, 64 unless set). Its
participation ratio sizes the grid of `embedding_dim`. The grid stops once three
values in a row reach the threshold. `Dimensionality(embedding_dims=range(1, 13))`
fits exactly the values given. Each value gets `n_restarts` networks (4) and the
best one counts. The messages say when the grid grows, stops early or is
extended ([MESSAGES.md](MESSAGES.md#dimensionality)).

Without `y_data` the mode compares two halves of X's channels split by
`split_method`. `'random'` draws `n_splits` (5) channel assignments and reports
the median reading with the standard deviation over splits. The other split
methods give one split. With `y_data` the two views are X and Y. The image
splits (`'horizontal'`, `'vertical'`, `'row_interleaved'`, `'col_interleaved'`,
`'diagonal'`, `'antidiagonal'`) need `(N, C, H, W)` input. The two diagonal
splits cannot be used with a convolutional encoder.

`dataframe` has one row per split and `embedding_dim`. Its `mi_mean` averages
the restarts, `mi_best` is the best of them and `mi_curve` is the running
maximum that the reading is taken from. The participation ratios are
`pr_eig_mean` and `pr_singular_mean`. `runs` holds every network. `details`
holds `dimension_at_most`, `dimension_at_most_std`,
`dimension_at_most_per_split`, `plateau` (per split), `embedding_dims` (every
value fitted), `reference_dim`, the reference's `pr_singular` and `pr_eig`, and
`stopped_early`.

This mode uses the hybrid critic unless `critic_type` is set. It runs a
separable critic with a warning and refuses the concat critic because that
critic has no embedding of either side. Unless set, `n_epochs` is 500 with
`patience` 50 and `shared_encoder` is `True` without `y_data`. The hybrid
critic's encoders use layer normalisation here unless `norm_layer` is set. It
stops the fits at an embedding size equal to the dimension from stalling short
of the full value. It also divides out each sample's overall scale and lowers
the whole curve where that scale carries information. The reading is taken
against the curve's own plateau. `sweep_grid`
refuses `embedding_dim` and `run_id` because the mode sets both.

---

## Repeats, grids and tests

### Repeats and configurations

`run_id` in `sweep_grid` repeats every configuration on the same data:

```python
result = nmi.run(x, y, mode='sweep',
                 sweep_grid={'hidden_dim': [64, 256], 'run_id': range(5)},
                 n_workers=8, seed=0)
```

Each repeat retrains its networks from a different initialisation and draws a
different window shift at every epoch. `mi_std` measures the variability of the
procedure over those repeats and says nothing about how the estimate would move
on a new recording.

A configuration grid runs every combination of its keys. The keys can be
settings of any config or processor parameters such as `window_size`,
`step_size` or `bin_size`. Every value of a processor parameter is windowed
separately. A processor parameter that no stream's processor reads is
refused. Values given as lists come back as tuples in `runs` and `dataframe`.

### What each mode accepts

| mode | repeats | configuration grid | `rigorous` |
|---|---|---|---|
| `estimate` | warns and runs once | warns and runs the base configuration | through `mode='rigorous'` |
| `precision` | warns and runs once | warns and runs once | refused |
| `sweep` | yes | yes | through `mode='rigorous'` with the same grid |
| `rigorous` | one extrapolation per repeat | yes | built in |
| `lag` | yes | yes | refused |
| `conditional`, `interaction`, `transfer` | yes | yes | `rigorous=True` on the mode's config |
| `pairwise` | yes | yes | refused |
| `dimensionality` | through `n_restarts` | one curve per configuration | refused |
| every named quantity | yes | yes | `rigorous=True` |

`n_workers` above 1 warns when the call trains one network and there is nothing
to run in parallel.

### Bias correction for the difference quantities

`rigorous=True` on `Conditional`, `Interaction` or `Transfer` extrapolates the
quantity itself. At every chunk the components are trained on the same samples.
Part of the noise they share cancels when they are combined. The combined values
are then fitted against $\gamma$.

```python
result = nmi.run(x, y, mode='conditional',
                 conditional=Conditional(w_data=w, rigorous=True, gamma_range=range(1, 11)),
                 n_workers=8)
result.get('is_reliable'), result.get('mi_error')
```

The fit settings are the fields of `Rigorous` set on the mode's config. `runs`
holds the same fit columns as `mode='rigorous'`.
`details[config_id]['trainings']` holds one row per chunk with the combined
value.

### Permutation tests

`permutation_test=True` tests every row of the result against a null
distribution. Each null trial reruns the call with the source X moved in time
and Y and W in place. Moving X breaks its link with Y and leaves Y, W, the link
between them and Y's own history in transfer entropy as recorded. X keeps its
own temporal structure except at the seams where it wraps or where blocks meet.

```python
result = nmi.run(x, y, mode='estimate', permutation_test=True, n_permutations=200,
                 n_workers=8, seed=0)
result.get('p_value')
result.get('null_distribution')
```

The default `permutation_shuffle='circular'` shifts X by one random offset with
wrap-around, drawn away from zero by at least 10% of the recording's length so
that no trial leaves X nearly aligned. `permutation_shuffle='block'` cuts X into
blocks one window long and reorders them. Both apply to windowed rows, to raw
series and to spike times. A spike population moves as a whole and keeps the
structure among its neurons.

The null has one set of trials per row of `dataframe`. A lag scan gets a null at
every lag and a pairwise matrix a null for every pair.
`details[config_id]['null_distribution']` holds the trial values: a list for a
configuration without an axis and a dict keyed by the axis value otherwise. For
a quantity that cannot be negative the trials are reported like the result with
no value below zero ([THEORY.md](THEORY.md#the-reported-number)).
`null_distribution_raw` holds the same trials as measured.
`dataframe['p_value']` is $(1 + \#\{\text{null} \geq \text{observed}\}) / (1 +
n)$.

The smallest p-value $n$ trials can give is $1/(n+1)$. A test with fewer than
100 trials warns. Every trial reruns the whole call with every configuration of the grid. The test therefore costs $n$ times the call.

`rigorous`, `precision` and `rigorous=True` refuse a permutation test.
`dimensionality` warns and computes no null because its curve is no single
value for a null to sit under. `pairwise` without `y_data` warns and computes no
null because moving X would move both sides of each pair.

### Workers and seeds

`n_workers` runs the independent tasks of a call in parallel: repeats,
configurations, lags, pairs, chunks of the rigorous ladder, splits, and
permutation trials. With `seed` set, each task reseeds from the seed and a fixed
key of its own. A seeded call therefore returns the same numbers at any
`n_workers`.

A script that runs with `n_workers` above 1 needs a `__main__` guard. The
workers start by re-importing the module that launched them. A top-level
`nmi.run(...)` would run again in every worker.

```python
if __name__ == '__main__':
    result = nmi.run(x, y, mode='sweep', sweep_grid=grid, n_workers=4)
```

A notebook needs no guard because its kernel is never re-imported.

---

## Preparing the data

### Processors

`Processing` names the processor that reads each raw stream. Without one, the
data are taken as already windowed.

| data | processor | raw shape |
|---|---|---|
| continuous signals (LFP, EEG, calcium) | `'continuous'` | `(n_timepoints, n_channels)` |
| spike trains | `'spike'` | a list with one array of spike times per neuron |
| discrete states | `'categorical'` | `(n_timepoints, n_channels)` of labels |
| already windowed | none | `(n_samples, n_channels)` or `(n_samples, n_channels, window)` |
| image-like | none | `(N, C, H, W)`, passed through unchanged |

```python
from neural_mi import Processing

processing = Processing(x='spike', x_params={'window_size': 0.1, 'bin_size': 0.01},
                        y='continuous', y_params={'window_size': 0.1, 'sample_rate': 1000},
                        y_time=behaviour_t)
result = nmi.run(spike_times, behaviour, mode='estimate', processing=processing)
```

Every processor turns its stream into a tensor of windows of shape `(n_windows,
n_channels, width)`. Already-windowed 2-D data gain a trailing axis of 1. 4-D
data are passed on unchanged for `embedding_model='cnn2d'` and the image
augmentations.

Categorical labels of an integer type are used as they are and must be
non-negative. Labels of another numeric type are mapped to $0, \dots, n-1$ in
sorted order with a warning. Text labels are refused.

The width a window ends up with depends on the processor:

| processor | width of a window of `window_size=w` |
|---|---|
| continuous | $w$ samples, covering $[t, t+w)$ |
| categorical | the number of categories, under `encoding='majority_vote'` and `'probability'`; categories times samples under `'full_trajectory'` |
| spike, spike times | `max_spikes_per_window`, set by the densest window |
| spike, with `bin_size` | $w / \mathrm{bin\_size}$ bins |

`x_window_width` and `y_window_width` on a dataset built by `create_dataset`
give the widths directly.

### Units of time

A stream with a clock (`x_time`, `y_time`, `w_time`) is windowed in the clock's
units. A continuous or categorical stream with a `sample_rate` and no clock is
windowed in seconds on the clock $t_i = i / \mathrm{sample\_rate}$. A continuous
or categorical stream with neither is windowed in samples. Spike times are read
as seconds. `window_size` and `step_size` are in these units. The lags of
`mode='lag'` are in seconds for spike data and for streams with a `sample_rate`.
Every other stream counts its lags in samples even when it has a clock. On data
that are already windowed a lag counts windows and warns.

`step_size` sets how far each window advances and below 1 is read as a fraction
of `window_size`:

| `step_size` | step at `window_size=0.5` |
|---|---|
| `None` (default) | 0.5: windows touch |
| `0.25` | 0.125, a quarter of the window, 75% overlap |
| `2.0` | 2.0, an absolute step |

The fraction rule keeps `step_size=0.25` at 75% overlap for every window size.
In seconds, where windows and steps below 1 are ordinary, the rule makes
`window_size=0.5, step_size=0.5` a 0.25 s step. The library warns whenever
`window_size` is below 1 and `step_size` falls between 0 and 1, naming the step
applied. An absolute step `a` below 1 is requested as `step_size=a /
window_size`.

### One grid for every stream

All the streams of a call are windowed on one grid. The grid starts at the
latest start among the streams, a window is kept only where every stream has
data, and a window means the same interval in X, Y and W. Each stream keeps its
own processor, parameters and clock:

```python
processing = Processing(x='spike', x_params={'window_size': 1.0, 'bin_size': 0.1},
                        y='continuous', y_time=pos_t,
                        w='categorical', w_time=pos_t)
result = nmi.run(spikes, position, mode='conditional',
                 conditional=Conditional(w_data=direction), processing=processing)
```

Y and W read with X's processor when they name none. A stream that falls back to
X's parameters keeps only the keys its own processor takes. The call reports any
stream that asks for a `window_size` other than the one all streams share.

A continuous or categorical window survives only when at least
`min_coverage_fraction` of it holds samples. `details` and `runs` report the
fraction of windows kept (`window_retention`) and the counts behind it
(`n_windows_built`, `n_windows_retained`). Because validity combines across
streams, three streams at 62% each keep about 24% of windows jointly. Retention
below 50% warns and names the stream responsible.

### Silent spike windows

The default drops spike windows without spikes and estimates the MI in bits per
active window over the windows where the population fired.
`x_params={'drop_empty_windows': False}` keeps silent windows and estimates the
MI in bits per window. Because whether the population is active can itself carry
information, neither quantity is a rescaled version of the other
([THEORY.md](THEORY.md#silent-windows)).

Keeping silent windows is safe only when the whole extent was observed. In spike
times a stretch without spikes looks the same as a stretch without recording.
The extent is covered when unobserved stretches were cut out beforehand. A
continuous stream with its own clock in the pairing also covers it because its
coverage check masks unobserved stretches.

Dropping silent windows also breaks the time axis by leaving consecutive
windows more than one step apart. Any analysis that indexes windows by offset
needs them kept. A per-bin series for offsets sets `window_size` equal to `bin_size`
and keeps silent windows.

A window sweep on spike data changes the estimand as it goes because wider
windows are more likely to hold a spike. On a 3 Hz population, retention rose
from 0.155 at 20 ms to 1.0 at 1 s. `drop_empty_windows=False` holds retention at
1.0 across the sweep.

### Window shifting

By default `Training(shift_windows=True)` and `Training(shift_time=True)` make
training re-tile the windows from a fresh random offset every epoch to show the
network many sets of window boundaries. `shift_windows` re-slices a regular grid
and serves continuous and categorical streams. `shift_time` rebuilds windows
from the raw data and serves spike streams, including a spike stream paired with
a continuous or categorical one that has a `sample_rate`. Because evaluation
always reads the unshifted windows, the reported values do not depend on the
last shift drawn.

| call | continuous and categorical | spike |
|---|---|---|
| `estimate`, `sweep`, `pairwise`, `dimensionality`, `precision`, `lag` | shifted | shifted |
| `conditional`, `interaction` | shifted | shifted |
| `rigorous` | shifted | shifted for spike pairs only |
| `transfer` | not shifted | not shifted |
| `block_mi` | shifted | shifted |
| other named quantities | not shifted | not shifted |
| already-windowed data | not shifted | not shifted |

A spike stream paired with a continuous or categorical one is shifted only when
that stream has a `sample_rate` that makes a shift mean the same time on both
sides. Because transfer entropy and the named quantities built from offsets use
every row as a sample one step apart, a shift would only relabel their rows.
Setting a shift explicitly where it cannot apply warns. A windowed generator
whose MI is known for its own windows needs both switched off
([Generators](#generators-with-a-known-answer)).

### `create_dataset`

`neural_mi.data.handler.create_dataset` builds the windowed streams without
training anything, for checking widths and retention or for driving the
`Trainer` directly. It raises a named error when a processor and a data shape
disagree. Given a mapping, it builds any number of named streams on one grid:

```python
from collections import OrderedDict
from neural_mi.data.handler import create_dataset

built = create_dataset(OrderedDict((
    ('spikes', dict(data=spikes, processor_type='spike',
                    processor_params={'window_size': 1.0, 'bin_size': 0.1})),
    ('position', dict(data=pos, time=pos_t, processor_type='continuous',
                      processor_params={'window_size': 1.0})),
    ('direction', dict(data=direction, time=pos_t, processor_type='categorical',
                       processor_params={'window_size': 1.0})),
)))
built.stream('direction').data                 # (n_windows, 1, width)
built.n_windows_retained, built.window_retention
```

`create_dataset(x, y, ...)` builds two streams named `x` and `y`. Three or
more streams return a bundle that reaches the first two as `x_data` and
`y_data` and every stream as `stream(name)`.

### Generators with a known answer

`nmi.generators` makes synthetic data with an exactly known MI or lag. An
analysis can be checked on it before it is trusted on a recording.

```python
from neural_mi import generators

x, y = generators.generate_correlated_gaussians(n_samples=5000, dim=4, mi=1.5)     # mi in bits
x, y = generators.generate_nonlinear_from_latent(n_samples=5000, latent_dim=2,
                                                 observed_dim=8, mi=1.0)
x, y, exact = generators.generate_categorical_pair(n_samples=5000, n_categories=3,
                                                   agreement=0.9)
x, y, exact = generators.generate_xor_pair(n_samples=5000, noise=0.05)
x, y, exact = generators.generate_lagged_pair(n_samples=5000, lag=30)
x, y, exact = generators.generate_spike_pair(n_windows=4000, window_size=1.0,
                                             coding='timing')
x, y, exact = generators.generate_windowed_oscillatory(n_windows=4000)
x, y, exact = generators.generate_windowed_multichannel(n_windows=4000, n_channels=8)
rho = generators.mi_to_rho(dim=4, mi=1.5)
```

The information in `generate_xor_pair` is purely synergistic. Each input alone
carries none about Y while the two together determine it. `exact` approaches 1
bit as `noise` goes to zero. `generate_lagged_pair` puts the dependence at a
known lag. `mode='lag'` should peak at `lag` and approach `exact` there.
`generate_spike_pair` puts the information in the spike count under
`coding='count'` and in where a burst falls under `coding='timing'`. Under
timing coding the count is drawn independently.

A windowed generator's `exact` is the MI between the windows it built. The
analysis has to use the same `window_size` and no shifting:

```python
x, y, exact = generators.generate_spike_pair(coding='timing', window_size=1.0)
nmi.run(x, y, mode='estimate',
        processing=Processing(x='spike', y='spike', x_params={'window_size': 1.0}),
        training=Training(shift_time=False, shift_windows=False))
```

With shifting on, a window spans two independent draws and can carry more than
`exact`. An estimate above `exact` points to the setup.

`SharedLatentGaussian` gives the exact value of $I(A;B \mid C)$ for any choice
of processes and time offsets. Every temporal quantity then has a number to
check against:

```python
from neural_mi.generators import SharedLatentGaussian

oracle = SharedLatentGaussian(dims={'x': 8, 'y': 8, 'w': 8}, d=2, phi=0.9)
data = oracle.sample(T=20000, seed=0)                # {'x': (20000, 8), 'y': ..., 'w': ...}

past = lambda v, k=25: [(v, s) for s in range(-k, 0)]
oracle.exact(past('x'), [('x', 0)])                              # active information storage
oracle.exact(past('x'), [('y', 0)], past('y'))                   # transfer entropy, X to Y
oracle.exact([('x', 0)], [('y', 0)], past('x') + past('y'))      # instantaneous exchange
oracle.block_mi(30)                                              # grows without bound in w
oracle.mi_rate()                                                 # bits per step
oracle.affine_fit(30, 60)                                        # slope and intercept of I_w
```

The processes share one AR(1) latent with correlation time $\tau = -1/\log\phi$.
Because the shared latent violates Massey's no-feedback condition, the directed
quantities converge below the symmetric MI rate that only a two-sided window
over X recovers ([THEORY.md](THEORY.md#temporal-information-quantities)).
`oracle.sample(..., return_latents=True)` also returns the latent.
`oracle.snr(name)` gives each channel's ratio of latent to noise variance.
Because the projections are random, the channels of one process can differ in
SNR by two orders of magnitude. A channel to display is best chosen by its SNR.
`generate_shared_latent_gaussian(T, ...)` returns the data and the oracle
together.

---

## The named quantities

`nmi.transfer_entropy`, `nmi.mi_rate` and the other named functions each build
the arrays their quantity needs and call `run()`. They take every keyword
`run()` takes and return the same `Results`. [THEORY.md](THEORY.md#temporal-information-quantities)
defines each quantity and the identities between them.

| function | quantity | runs as |
|---|---|---|
| `active_information_storage(x, k, future_k=1)` | $I(X_{past}(k); X_0)$ | `estimate` |
| `predictive_information(x, k)` | $I(X_{past}(k); X_{fut}(k))$ | `estimate` |
| `instantaneous_mi(x, y)` | $I(X_0; Y_0)$ | `estimate` |
| `cross_predictive_information(x, y, k)` | $I(X_{past}(k); Y_{fut}(k))$ | `estimate` |
| `block_mi(x, y, window_size)` | $I(X_{1:w}; Y_{1:w})$ | `estimate`, windowed by the processors |
| `transfer_entropy(x, y, history_window)` | $I(Y_0; X_{past} \mid Y_{past})$ | `transfer` |
| `conditional_transfer_entropy(x, y, w, history_window)` | $I(Y_0; X_{past} \mid Y_{past}, W_{past})$ | `transfer` |
| `interaction_information(x, y, w)` | $I(X,W;Y) - I(X;Y) - I(W;Y)$ | `interaction` |
| `mi_rate(x, y, h, half_width=20)` | $I(X_{all}(L); Y_0 \mid Y_{past}(h))$ | `conditional` with `align='dual_branch'` |
| `instantaneous_exchange(x, y, k)` | $I(X_0; Y_0 \mid X_{past}(k), Y_{past}(k))$ | `conditional` with `align='dual_branch'` |
| `directed_information_rate(x, y, k)` | $I(X_{past}(k), X_0; Y_0 \mid Y_{past}(k))$ | `conditional` with `align='dual_branch'` |

$X_{all}(L)$ is a window of $2L+1$ steps centred on $Y_0$ whose $L$ is set by
`half_width`.

### Calling them

Inputs passed on their own are `(T, n_channels)` series sampled on a regular
grid whose rows the offsets count. With `processing=` the offsets count the
windows of one shared grid onto which each processor first puts its stream. This
is the route for spike times and for streams on different clocks:

```python
r = nmi.active_information_storage(
    spike_times, k=3,
    processing=Processing(x='spike', x_params={'window_size': 0.5, 'bin_size': 0.1,
                                                'drop_empty_windows': False}))
```

Window validation is off on this route and keeps every window because dropping
one would put two distant windows next to each other and change what an offset
means. Gaps stay visible through the coverage warnings. Keep
`drop_empty_windows=False` on spike streams for the same reason.

`block_mi` is the one quantity that windows the data itself. It sets
`window_size`, merges it into the `Processing` given, and reads continuous
data when given none.

`mi_rate`, `instantaneous_exchange` and `directed_information_rate` embed their
conditioning set in its own branch because it has a different window length from
the other group. They need `Model(embedding_model='dual_branch', ...)` and raise
at once without it. At `h=0` or `k=0` nothing is conditioned on and any model
works.

```python
from neural_mi import Model, Training

model = Model(embedding_model='dual_branch', branch_model='gru', embedding_dim=16)
r = nmi.mi_rate(x, y, h=[0, 5, 10, 20], half_width=20, model=model,
                training=Training(n_epochs=100), n_workers=4)
```

`h` and `half_width` bias the MI rate in opposite directions. Too little
history on Y leaves Y's own storage in and reads high. Too narrow a window on X
leaves signal out and reads low. Sweep one, fix it past its knee, then
sweep the other. An iterable for both raises.

### Sweeps, repeats and rigour

A quantity's own parameter (`k`, `history_window`, `window_size`, `h`,
`half_width`) takes a scalar or an iterable. An iterable runs every value in
parallel across `n_workers` and returns one `Results` with a configuration per
value. `sweep_grid` adds repeats and further settings at every value.

```python
r = nmi.transfer_entropy(x, y, history_window=[2, 4, 8, 16],
                         sweep_grid={'run_id': range(3)}, n_workers=8)
r.dataframe[['history_window', 'mi_mean', 'mi_std']]
```

`rigorous=True` (or a `Rigorous(...)` with fit settings) extrapolates every
repeat of every quantity. `transfer_entropy` and `conditional_transfer_entropy`
also take `bidirectional=True`.

For a quantity built as a difference, `mi_std` over repeats says whether the
difference is resolved at all. A value whose spread exceeds it has no size or
sign to read.

### `stride`

The offset-built quantities cut a row at every time step by default.
Neighbouring rows then overlap in all but one step. `stride=n` cuts one row
every `n` steps:

```python
r = nmi.active_information_storage(x, k=5, stride=4)
```

Neighbouring rows that nearly duplicate each other inflate a random split and
overstate the $\log K$ ceiling because $K$ counts rows without knowing how many
are near copies. `stride` counts whole rows and refuses a fraction. It is kept
apart from `step_size` because the fractional reading of `step_size` refers to a
single window length these quantities do not have.

### Quantities without a named function

Every quantity above is $I(A;B \mid C)$ for some choice of processes and
offsets in each group. `build_offset_arrays` builds the three groups from a
spec, for a quantity with no function of its own:

```python
from neural_mi.utils import build_offset_arrays

spec = {'A': [('x', s) for s in range(-10, 0)],     # X's past
        'B': [('y', 0)],                            # Y now
        'C': [('y', s) for s in range(-10, 0)]}     # given Y's past
a, b, c, n_valid = build_offset_arrays({'x': x, 'y': y}, spec, stride=1)

r = nmi.run(a, b, mode='conditional',
            conditional=Conditional(w_data=c, align='dual_branch'),
            model=Model(embedding_model='dual_branch'))
```

Offsets count rows with negative offsets in the past and zero at the present.
Every group is cut to the range over which all its offsets exist. Each array has
shape `(n_valid, n_channels, n_offsets)`. Without a `C` group, the pair goes to
`mode='estimate'`.

---

## Models, estimators and critics

### Estimators

The critic scores every pairing of a batch. The estimator turns the score matrix
into a bound on the MI
([THEORY.md](THEORY.md#estimating-mutual-information-with-a-critic)).

| `Estimator(name=...)` | bias and variance | ceiling |
|---|---|---|
| `'infonce'` (default) | low variance, biased low near its ceiling | $\log N$ nats, $N$ the number of samples evaluated |
| `'smile'` | lower bias, higher variance | none |

InfoNCE cannot report more than the log of the number of samples it is evaluated
on. That number is `eval_size` on the held-out side and `train_eval_size` on the
reported training side, both capped by `max_eval_samples`. `runs` reports each
side's ceiling and how close the estimate came to it (`train_saturation`,
`test_saturation`). `batch_size` sets how many negatives each training step sees
and places no cap on the reported value. An estimate near its ceiling calls for
a larger `max_eval_samples` or for SMILE.

SMILE is the less numerically robust of the two. Called through `nmi.run()` it
produced no NaN over eight data draws. A `Trainer` assembled by hand produced
NaN on six of six draws at one input geometry and four of four at another.
InfoNCE trained on every one of those draws. InfoNCE's read-out is bounded by
construction where SMILE's clipped density ratio is bounded in variance only.
`Estimator(name='smile', params={'clip': 5.0})` sets the clip. A lower clip
lowers the variance and raises the bias.

### Encoders

Each side passes through an encoder before the critic scores it. Every encoder
outputs `(batch, embedding_dim)`.

| `embedding_model` | input | for |
|---|---|---|
| `'mlp'` (default) | `(N, C, W)`, flattened | a general default |
| `'cnn'` | `(N, C, W)` | local patterns along the window |
| `'cnn2d'` | `(N, C, H, W)` | image-like input of any size |
| `'gru'`, `'lstm'` | `(N, C, W)` | sequences, optionally bidirectional |
| `'lru'` | `(N, C, W)` | long sequences (a linear recurrent unit) |
| `'tcn'` | `(N, C, W)` | long windows (dilated convolutions) |
| `'transformer'` | `(N, C, W)` | self-attention over the window with `nhead` heads |
| `'deepsets'` | `(N, C, W)` | raw spike times, invariant to the order of a window's spikes |
| `'pretrained_backbone'` | `(N, C, H, W)` | a frozen torchvision network with a trainable head |
| `'dual_branch'` | two inputs of different lengths | the conditioning set of `align='dual_branch'` |

The encoder classes in the table are importable from `neural_mi.models`.
`pretrained_backbone` takes the torchvision name in `pytorch_predefined` and
ImageNet weights with `pretrained=True`. Input of a size other than 224 by 224
pixels is resized by bilinear upsampling on the first forward pass and warns.
`dual_branch` builds each branch from `branch_model`, `'gru'` by default.

`embedding_model` sets the encoder of both sides. `embedding_model_y`,
`custom_embedding_cls_y`, `hidden_dim_y`, `n_layers_y` and `embedding_dim_y`
override it for Y:

```python
model = Model(embedding_model='deepsets', embedding_model_y='mlp',
              hidden_dim=128, hidden_dim_y=32, embedding_dim=32)
```

A separable critic scores a pair by the dot product of the two embeddings and
needs one `embedding_dim`. Only the hybrid critic accepts a different
`embedding_dim_y`. The concat critic has no separate encoders and refuses every
Y-side setting. `shared_encoder=True` uses one encoder for both sides and
refuses any asymmetry.

`bias=False` builds the encoders without bias terms so that an all-zero input
embeds to exactly zero. The MLP uses an RMS normalisation without learned
parameters because `LayerNorm` and `BatchNorm` subtract the mean and move a zero
away from zero. Spectral normalisation is unaffected. Because the transformer's
positional encoding and a pretrained backbone's frozen biases add terms that do
not depend on the input, these two encoders warn that the setting cannot fully
apply.

### Critics

| `critic_type` | scores a pair by |
|---|---|
| `'separable'` (default) | the dot product of separately computed embeddings |
| `'hybrid'` | a small network on the concatenated embeddings, sized by `hidden_dim_head` and `n_layers_head` |
| `'concat'` | one network on the concatenated raw inputs |

The separable critic scores a whole batch with one matrix product. The hybrid
and concat critics put every pairing through a network. Each side passes through
that network's first layer once and the pairs are scored in large blocks. The
hybrid critic then trains in about the time the separable critic takes and the
concat critic in a few times that. The concat critic has no `embedding_dim`.
`mode='dimensionality'` uses the hybrid critic.

`Model(norm_layer='layer')` helps the hybrid critic train on wide data. It also
divides out each sample's overall scale. Where that scale carries information
the estimate reads lower.

### Custom encoders and critics

A custom encoder subclasses `BaseEmbedding` and declares how its `input_dim`
is counted. `input_style = 'flattened'`, the default, gives
`n_channels * window` as one number. `input_style = 'channels'` gives the
channel count, for an encoder that handles the window axis itself:

```python
import torch.nn as nn
from neural_mi.models import BaseEmbedding

class MyEmbedding(BaseEmbedding):
    def __init__(self, input_dim, embedding_dim, hidden_dim=64, **kwargs):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, embedding_dim))

    def forward(self, x):                        # x: (batch, channels, window)
        return self.net(x.flatten(1))

class MySequenceEmbedding(BaseEmbedding):
    input_style = 'channels'

    def __init__(self, input_dim, embedding_dim, **kwargs):
        super().__init__()
        self.net = nn.GRU(input_dim, embedding_dim, batch_first=True)

    def forward(self, x):
        _, h = self.net(x.permute(0, 2, 1))
        return h.squeeze(0)

result = nmi.run(x, y, model=Model(custom_embedding_cls=MyEmbedding, embedding_dim=32))
```

A class receives `bias` only when its `__init__` accepts it directly or through
`**kwargs`. An explicit `bias=False` that cannot reach it warns.
`Model(custom_critic=...)` takes a whole critic module. Any architecture setting
passed beside it is ignored with a warning.

### Regularised objectives

`Model(use_variational=True)` makes any encoder variational and adds a KL term
weighted against the MI term by `beta`. Under the concat critic the variational
layer acts on each pair's score because there are no separate encoders.

`Model(use_decoder=True)` adds a decoder that reconstructs each input from its
embedding. Its reconstruction loss, weighted by `decoder_lambda` (or
`decoder_lambda_x` and `decoder_lambda_y`), keeps the embedding from collapsing.
The decoder mirrors the encoder: an MLP, CNN, CNN2D, GRU, LSTM, LRU, TCN or
transformer decoder for the matching encoder, and an MLP decoder with a warning
for the others. `decoder_output_activation_x` and `_y` choose `'linear'` or
`'sigmoid'` outputs with a squared-error loss or `'softmax'` with cross-entropy.
`decoder_recon_loss` in `runs` is the weighted reconstruction term at the best
epoch. [THEORY.md](THEORY.md#regularised-objectives) gives the objectives.

### Augmentations

`Training(augmentation_params=...)` perturbs every training batch and leaves the
evaluation batches behind the reported values clean. `augmentation_params_x` and
`augmentation_params_y` set each side separately and turn it off with `{}`.

```python
training = Training(augmentation_params_x={'gaussian_noise': {'std': 0.1}},
                    augmentation_params_y={})
```

| key | setting | effect |
|---|---|---|
| `gaussian_noise` | `{'std': 0.1}` | add Gaussian noise |
| `intensity_scale` | `{'lo': 0.8, 'hi': 1.2}` | multiply each sample by a random factor |
| `channel_dropout` | `{'p': 0.1}` | zero each channel with probability `p` |
| `random_flip_h`, `random_flip_v` | `True` or `{'prob': 0.5}` | flip along height or width |
| `random_rotation_90` | `True` | rotate by a random multiple of 90° |
| `random_crop` | `{'padding': int}` | pad by reflection and crop back at random |
| `random_erase` | `{'prob': float, 'scale': (lo, hi)}` | zero a random rectangle |
| `time_mask`, `freq_mask` | `{'max_width': int}`, `{'max_height': int}` | zero a band of columns or rows |
| `gaussian_blur` | `{'kernel_size': int, 'sigma': float}` | blur each channel |
| `custom` | a callable or a list of them | any function from a batch to a batch of the same shape |

The flips, rotation, crop, erase, masks and blur need `(N, C, H, W)` input and
are skipped with a warning on anything else. Within a batch the image
augmentations run first, then the others, then the custom ones.

---

## Plotting and animation

`result.plot()` draws the figure of the result's mode: the held-out MI against
epoch for `estimate`, the ladder and fit for `rigorous`, the components beside
the difference for the difference quantities, MI against $\tau$ for `precision`,
the matrix for `pairwise` and MI against embedding size for `dimensionality`. A result over
several configurations (a sweep or a lag scan) is drawn as MI against the swept
keys: a line for one key, a heatmap for two and bars for more. `kind='line'`,
`'heatmap'` or `'bar'` overrides the choice. `config_id=` picks the
configuration whose repeats the per-mode figure draws. `ax=` draws into existing
axes and `show=False` leaves the figure open.

`Results.compare([r1, r2], labels=[...])` overlays results of one mode: the
training curves of `estimate` results, the curves of results over one swept
key, and the extrapolations of `rigorous` results. The training-curve and
extrapolation overlays take one repeat per result.

`result.animate(output_path='training.gif')` animates the recorded training of
one network: the MI, the spectral metrics, the spectrum and the embeddings. The
embedding panel needs `Output(track_embeddings=True)`. `embedding_labels=`
colours its points by one or more labels. A `.mp4` path writes a video.
`config_id=` and `run_id=` pick the network, together with the axis values in
`mode='lag'` (`lag=`), `'pairwise'` (`ch_x=` and `ch_y=`) and
`'dimensionality'` (`split_id=` and `embedding_dim=`). A call that matches more
than one network is refused with the keys to pass.

`neural_mi.visualize` holds the functions behind these, each taking `ax=` and
`show=` and returning the axes:

| function | draws |
|---|---|
| `plot_sweep_curve(df, param_col)` | MI against one swept parameter |
| `plot_bias_correction_fit(trainings, fit)` | a rigorous ladder, its fit and the extrapolated point |
| `plot_dimensionality_curve(runs, details)` | MI against `embedding_dim` with the threshold and the reading |
| `plot_embeddings(z, color=...)` | embeddings in 2-D or 3-D, by PCA, t-SNE or UMAP |
| `plot_cross_correlation(x, y, true_lag)` | the linear cross-correlation against lag |
| `analyze_mi_heatmap(df)` | the region of a lag by window-size heatmap where MI rises |
| `animate_training(result, ...)` | the animation behind `result.animate()` |
| `set_publication_style()` | a matplotlib style for figures in papers |

---

## Speed, memory and logging

`device=None` picks CUDA, then Apple's MPS, then the CPU. Because MPS pays a
fixed cost for every GPU operation, the CPU can be several times faster on small
networks and batches. MPS starts to win on larger networks. Small synthetic checks run
faster with `device='cpu'`. `Training(use_amp='auto')` uses mixed precision on
CUDA.

`Training(max_eval_samples=...)` bounds memory through the largest number of
samples any one evaluation uses. `Model(max_n_batches=...)` bounds it through
the largest block of the score matrix computed at once.
`Training(dataset_device='cpu')`, the default, keeps the data in main memory and
moves each batch to the device. `'auto'` keeps the data on the device and is
faster when they fit in its memory.

NeuralMI logs warnings and errors by default. `nmi.set_verbosity(level)` sets the
logging level for the session and `nmi.set_verbose(True)` shows informational
messages beside the warnings. A
call keeps that level unless it passes `verbose=True` (informational messages
for this call) or `verbose=False` (warnings and errors only). [MESSAGES.md](MESSAGES.md) is keyed by the text of each message.
The library's own exceptions derive from `nmi.NeuralMIError`:
`nmi.DataShapeError` for a shape a processor cannot read,
`nmi.InsufficientDataError` for too little data for the request, and
`nmi.TrainingError` for a training run that failed. The warnings about a
quantity combined from several networks share the class `nmi.CombinationWarning`
so that `warnings.filterwarnings` can silence them as a group.

Within one call each kind of warning or log message is shown the first time it
comes up. Later ones that differ only in their numbers are counted. The call ends
with one line per kind giving how often it was raised. In IPython and Jupyter the
count runs over every NeuralMI call of a cell. The line then comes when the cell
ends. A loop of calls in one cell therefore shows each warning once. In a script,
`with nmi.grouped_warnings():` around the loop does the same. Messages raised
outside a NeuralMI call are left alone. A warning raised in a worker process
names the line of the call.
