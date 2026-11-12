# Using NeuralMI

This guide shows how to call NeuralMI for each task and how to read the results.
Three other pages go deeper: [PARAMETERS.md](PARAMETERS.md) lists every setting
with its default, [THEORY.md](THEORY.md) explains what each reported quantity
means and why the estimators behave as they do, and [MESSAGES.md](MESSAGES.md)
explains every warning that the library prints. [INTERNALS.md](INTERNALS.md) is
for people who change the library.

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

NeuralMI needs Python 3.11 or newer and PyTorch 2.4 or newer. Install it from the
root of the repository:

```bash
pip install .
```

The `[viz]` extra adds scikit-learn and umap-learn for the embedding plots. The
`[vision]` extra adds torchvision for `embedding_model='pretrained_backbone'`.

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

`run()` sets aside 10% of the samples in contiguous blocks as held-out data and
trains a critic network on the rest. After each epoch it evaluates the mutual
information on the held-out blocks. It then smooths this held-out curve, picks
the earliest epoch at which the curve reaches its top and reports the mutual
information on the training samples at that epoch as `mi_estimate`. The value is
in bits unless you ask for nats with `Output(units='nats')`.
[THEORY.md](THEORY.md#the-reported-number) explains why the library reports the
training value rather than the held-out one.

The held-out samples determine the epoch, so the library tells you when there are
fewer of them than one training batch. The data check that runs before training
says so, as does every message about the held-out curve.

Every setting belongs to a config object: `Processing` for how the raw data are
processed, `Model` for the networks, `Training` for the optimisation loop,
`Split` for the training and held-out sets, `Estimator` for the bound, `Output`
for units and extras, and one config for each analysis mode. Every config is
optional and can be replaced by a plain `dict` with the same keys. The values in
the example below only show where each setting goes. The library is tested and
documented with its defaults, so they are a reasonable starting point for any
data.

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
`mode='precision'`, `ch_x` and `ch_y` for `mode='pairwise'`, and `split_id` and
`embedding_dim` for `mode='dimensionality'`. A *repeat* is one independent run of
the whole procedure on the same data. Each repeat has its own `run_id`.
`params['config_keys']` and `params['axis_keys']` name the columns that index
`dataframe`.

`dataframe` always has the columns `config_id`, the grid and axis keys,
`mi_mean`, `mi_std`, `n_runs`, `n_zero` and `n_failed`. The columns of the mode
follow them (listed for each mode in [Choosing the analysis](#choosing-the-analysis)).
How `mi_mean` and `mi_std` are computed depends on the kind of quantity. For a
difference quantity or a rigorous fit, they use the measured value of every
repeat, negative values included. `mi_mean` is then reported as 0 when it is
negative, while `mi_raw_mean` keeps the measured mean. A repeat with no value
(NaN) is counted in `n_failed` and left out
([THEORY.md](THEORY.md#the-reported-number)). This happens, for example, when a
term of a difference failed. For a single network, `mi_mean` and `mi_std` use the
repeats that produced a value. A repeat reported as 0 produced nothing and is
counted in `n_zero` ([THEORY.md](THEORY.md#the-spread-over-repeats)).

`mi_std` is the spread over repeats of the procedure on the same data (NaN with
fewer than two repeats). It measures how much the estimate changes when the
networks are retrained. It is never an interval on the population value
([THEORY.md](THEORY.md#the-spread-over-repeats)).

`runs` has one row per repeat. Each row gives the repeat's value `mi` and the
diagnostics of the network behind it:

| column | meaning |
|---|---|
| `test_mi` | the held-out MI at the epoch that the library picked |
| `best_epoch` | the epoch that the library picked |
| `peak_epoch` | the epoch at which the smoothed held-out curve is highest |
| `curve_diagnosis` | the library's verdict on the held-out curve |
| `band_first_epoch`, `band_last_epoch` | the first and the last epoch of the top of the held-out curve (the *band* in the code) |
| `band_low`, `band_high` | the lower and the higher of the training values at those two epochs |
| `eval_size`, `train_eval_size` | the number of held-out and training samples that each evaluation scores |
| `test_eval_blocks`, `train_eval_blocks` | the number of blocks in which each of those groups is scored |
| `train_ceiling_mi`, `test_ceiling_mi` | the InfoNCE ceilings of the two groups |
| `train_saturation`, `test_saturation` | how close the training and the held-out values came to their ceilings |
| `window_retention` and the window counts | for windowed data |
| `test_mi_history` | the held-out MI at every epoch |

A repeat that trains several networks (the difference quantities and the
rigorous ladder) keeps the diagnostics of each network in
`details[config_id]['trainings']`.

`mi_estimate` is `None` whenever `dataframe` has more than one row. The results
of a lag scan, a pairwise matrix, a precision curve or any grid over several
configurations are in `dataframe`.

### Reading one value

`result.get(key)` looks for the key in `dataframe`, then in `details`, then in
the kept embeddings and then in `runs`. It returns the value when exactly one
row, configuration or repeat contains it. When several do, it raises an error
that names the table to look in.

```python
result.get('test_mi_mean')            # one configuration: a number
result.get('amplification_factor')    # the difference quantities
result.get('eval_size')               # one repeat: a number
result.get('no_such_key', default=0)  # nothing holds it: the default
```

### Printing, saving and exporting

`result.summary()` prints the headline value and its spread. For a difference
quantity it also prints the components, and for a rigorous fit it prints the
reliability together with the checks behind it.

`result.save(path)` pickles the whole object and returns the absolute path.
Without a path, or with a directory, the file is named
`neuralmi_{mode}_{YYYYMMDD_HHMMSS}.pkl`. The method never overwrites an existing
file. `Results.load(path)` loads the object back.

`result.to_dict()` converts every field into values that JSON can store, with
arrays as nested lists and tables as lists of records. `result.to_json(path)`
writes these values to a file named the same way. Use `save()` and `load()` for
an exact round trip.

### Embeddings

`Output(return_embeddings=True)` keeps the embedding of every window for each
repeat, in the original order of the samples. The arrays therefore line up with
the windowed data and with any labels indexed the same way. Their keys in
`details[config_id]['embeddings']` are `run_id`, `(lag, run_id)` in `mode='lag'`,
`(ch_x, ch_y, run_id)` in `mode='pairwise'` and `(split_id, embedding_dim, run_id)`
in `mode='dimensionality'`. Each entry contains `embeddings_x` and
`embeddings_y`. `mode='dimensionality'` also keeps, for each split, the
embeddings of the best restart at the dimension that the split reports, in
`details[config_id]['embeddings_at_bound']`.
`Output(return_rotated_embeddings=True)` adds versions rotated so that dimension
0 has the most variance shared between X and Y. `Output(track_embeddings=True)`
keeps the embeddings at every epoch for `result.animate()`.

Embeddings are available in the modes in which a repeat trains one network:
`estimate`, `sweep`, `lag`, `pairwise` and `dimensionality`. The difference
quantities, `rigorous` and `precision` refuse the setting, because they train
several networks per repeat or evaluate one network many times.

### Saving the trained networks

`Training(save_best_model_path=...)` saves every network that a call trains, at
its best epoch, together with the settings needed to rebuild it. Later,
`nmi.extract_embeddings(path, x, y)` embeds new data with a saved network and
windows them the way the training data were windowed.

A call that trains one network writes to the path as given and adds `.pt` when
the path has no extension. A call that trains several networks saves each one
under the path with its identifying labels added to the name. These labels are
its grid values and `run_id`, plus its `gamma` and `chunk`, `component`, `lag`,
channel pair or `split_id` where the call has them.

```python
result = nmi.run(x, y, mode='sweep',
                 sweep_grid={'embedding_dim': [16, 32], 'run_id': range(2)},
                 training=Training(save_best_model_path='models/best.pt'))
# models/best_embedding_dim-16_run_id-0.pt, ..., models/best_embedding_dim-32_run_id-1.pt
result.runs[['embedding_dim', 'run_id', 'model_path']]
```

Each path is recorded as `model_path` in the row of its network in
`result.runs`, or in `details[config_id]['trainings']` where a repeat trains
several networks. The path of a single saved network is
`result.get('model_path')`. If you give a directory instead of a file name, the
networks are saved under the generated name `neuralmi_<mode>_<timestamp>.pt` with
the same labels. A large grid or a rigorous ladder of whole networks can fill a
disk, so a call that saves several networks warns before training and afterwards
logs how many it saved and their total size. To keep one model, rerun the
configuration you want on its own. Permutation trials are never saved.

---

## Choosing the analysis

Each mode reports one kind of quantity:

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
| `'dimensionality'` | the smallest embedding size that carries `saturation_ratio` of the MI | `get('dimension_at_most')` |

The functions of [the named quantities](#the-named-quantities) build their
arrays, call one of these modes and return its `Results`.

### `estimate`

```python
result = nmi.run(x, y, mode='estimate')
result.mi_estimate
result.plot()          # held-out MI against epoch, with the chosen epoch marked
```

`dataframe` also gives the held-out MI at the chosen epoch as `test_mi_mean`.
`mode='estimate'` runs one configuration once and refuses a `sweep_grid`. Use
`mode='sweep'` to run a grid.

### `sweep`

```python
result = nmi.run(x, y, mode='sweep',
                 sweep_grid={'embedding_dim': [16, 32, 64], 'run_id': range(3)},
                 n_workers=4)
result.dataframe[['embedding_dim', 'mi_mean', 'mi_std', 'n_runs']]
result.plot()          # MI against embedding_dim, shaded by the spread over repeats
```

Any setting of `Model`, `Training`, `Split` or `Estimator` and any processor
parameter can be a key of `sweep_grid`. The key `run_id` adds repeats. In a grid,
`Split(mode=...)` becomes `split_mode` and `Split(gap_fraction=...)` becomes
`split_gap_fraction`. `Estimator(name=...)` and `Estimator(params=...)` become
`estimator_name` and `estimator_params`.

The grid refuses four kinds of key, because each would run the same call under
different labels:

- a key that no setting reads
- a key that the mode reads only once from its config, such as `history_window`
- a key that the mode sets itself, such as `lag` in `mode='lag'` or
  `embedding_dim` and `run_id` in `mode='dimensionality'`
- a setting that nothing in the call reads, such as `embedding_dim` under the
  concat critic

A fixed setting that nothing reads is ignored with a warning instead. Each
configuration of a grid computes what a separate call with its settings computes.
Processor parameters, `input_scaling` and `shift_windows` change how the call
prepares its data, so a grid over one of them prepares the data once for each of
its values. Before you commit to a rigorous run, sweep the capacity of the network
(`embedding_dim`, `hidden_dim`, `n_layers`) and `n_epochs` to find a
configuration whose estimate no longer changes.

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

For each $\gamma$ in `gamma_range`, the library cuts the data into $\gamma$
chunks and trains one network per chunk. It then fits the MI against $\gamma$
and extrapolates the fit to the limit of infinite data at $\gamma = 0$
([THEORY.md](THEORY.md#correcting-it)). One call trains $\sum \gamma$ networks
(55 for the default range).

Each row of `runs` describes the fit of one repeat:

| column | meaning |
|---|---|
| `mi` | the extrapolated value |
| `mi_error`, `mi_error_pred` | the half-widths of the fit's confidence interval and of its prediction interval |
| `slope` | the slope of the fit |
| `is_reliable` | whether the fit can be reported (see below) |
| `enough_gamma_points`, `linear_region_found`, `leverage_warning`, `saturated_gammas` | the checks behind `is_reliable` |
| `fit_quality_warning`, `r_squared`, `max_abs_residual` | diagnostics reported alongside `is_reliable` |
| `zero_rungs`, `unresolved_rungs` | the rungs that produced nothing and the rungs whose held-out curves did not resolve, both left out of the fit |

`runs` also gives the curvature statistics on which the linear region was
chosen. When the fit extrapolates below zero, `mi` is 0 and `mi_raw` keeps the
fitted value. `details[config_id]['trainings']` keeps every network of the
ladder.

`is_reliable` is `False` in four cases:

- the fit used fewer than `min_gamma_points` values of $\gamma$
- no linear region was found
- leaving out $\gamma = 1$ shifts the intercept by more than `leverage_threshold`
- in `mode='rigorous'`, a value of $\gamma$ in the fit lies at the ceiling of its
  estimator

The rungs that are left out are removed before these checks, so a ladder whose
deep chunks are too small for their networks ends up with too few values of
$\gamma$ instead of a fit through noise. Do not report an unreliable fit, even
with a caveat. A reliable fit means that the part of the bias that shrinks with
more data has been removed and that the fit is well determined. A bias that is
the same at every chunk size passes through the extrapolation unchanged
([THEORY.md](THEORY.md#correcting-it)).

At $\gamma = k$ each chunk has about $N/k$ samples. The run warns when the
held-out part of a chunk can no longer fill one evaluation batch, because the
rungs beyond that point are noise that a straight line can still pass through.
`Training(min_reliable_samples=...)` moves the point at which the run warns.

With repeats, `mi_mean` and `mi_std` are the mean and the spread of the
extrapolated values of the repeats. `dataframe['n_reliable']` counts the reliable
fits. `dataframe['mi_error']` is filled only for a single repeat, because the
intervals of repeats that share data cannot be combined. Rows in time order (the
windows that a processor cut, the rows that a named quantity builds and the
histories of transfer entropy) are cut into the same contiguous chunks for every
repeat and every quantity. Other rows are reordered at random for each repeat,
since the library cannot know whether an array that you pass in already built is
in time order. `Rigorous(temporal_chunking=True)` tells it that the array is.
`runs['chunking_mode']` records which chunking was used.

### `lag`

```python
from neural_mi import Lag

result = nmi.run(x, y, mode='lag', lag=Lag(lag_range=range(-20, 21)), n_workers=8)
df = result.dataframe
peak = df.loc[df['mi_mean'].idxmax(), 'lag']
result.plot()
```

A positive lag compares X with the future of Y. Lags are in seconds for spike
data and for streams with a `sample_rate`. For everything else they are in
samples ([units of time](#units-of-time)). `dataframe` adds `test_mi_mean` and
gives the number of windows at each lag in `n_windows_built`. Longer lags leave
fewer samples. `Lag(equalize_n=True)` cuts every lag to the number of samples of
the largest lag so that all lags use the same amount of data.

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

`mode='precision'` trains one network on the clean data, freezes it and
evaluates it on data corrupted at each $\tau$ in `tau_grid`. The `'rounding'`
method moves every value to the centre of its bin of width $\tau$. The `'noise'`
method adds uniform jitter on $[-\tau/2, \tau/2]$ and averages over
`n_noise_samples` draws. The precision is the smallest $\tau$ at which the MI
falls below `threshold_ratio` times the baseline
([THEORY.md](THEORY.md#spike-timing-precision)). `corrupt_target` sets whether X,
Y or both are corrupted. `Split(train_indices=..., test_indices=...)` fixes the
split on which the baseline is trained.

Both methods corrupt only the entries that contain a measurement. A spike window
stores its spike times in a fixed number of slots and fills the unused slots with
`no_spike_value`. These unused slots, empty bins and the zeros that pad a gap in a
continuous recording stay as they are. Neither method creates a spike or moves
one onto the empty value. On spike data with `bin_size`, the mode degrades the
counts in each bin and leaves the spike times alone. To measure timing
precision, use the spike-time representation without `bin_size`.

`dataframe` has one row for each $\tau$, including $\tau = 0$. `mi_estimate` is
therefore `None`, and the baseline is `get('baseline_mi')`. `runs` has one row
per evaluation (one per noise draw under `'noise'`).

Beyond the threshold, the curve leaves the region in which the estimator can be
trusted. The frozen critic saw only clean inputs, so a lower bound such as
InfoNCE can fall arbitrarily far below zero on the corrupted inputs. Points below
zero are reported as 0 with a warning and keep their measured values in
`mi_raw`. The precision is the $\tau$ at which the curve crosses the threshold.

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

The library computes $I(X;Y \mid W)$ for each repeat as $I(X,W;Y) - I(W;Y)$
from two networks. `runs` gives the value of each repeat and its components
`mi_xw_y` and `mi_w_y`. A repeat that comes out negative is reported as 0 and
keeps the measured difference in `mi_raw`. A term whose network learned nothing
enters as 0 with a warning that names the term.

Sometimes the network of a term gives a negative value on its training rows
although its held-out value lies clearly above zero. Such a repeat has no value
(NaN). `failed_term` names the term. The same rule applies to every quantity that
combines separately trained networks
([THEORY.md](THEORY.md#the-reported-number)).

`dataframe` adds the means of the components and the `amplification_factor`.
This factor says how much the difference magnifies the relative error of the
components. A factor of `Training(amplification_threshold=...)` or more triggers
a warning, because a difference that small should be interpreted together with
its components ([THEORY.md](THEORY.md#amplification)).

The joint term takes X and W as a pair and encodes each part on its own branch
([The encoder of a difference](#the-encoder-of-a-difference)). X and W therefore
need not share a channel count, a window length or a modality. A spike X with a
continuous W works, and so do rows of W that were built before the call with
their own window length. When a processor windows W, W shares the grid of X and
Y, including their window and step
([One grid for every stream](#one-grid-for-every-stream)).

### `interaction`

```python
from neural_mi import Interaction

result = nmi.run(x, y, mode='interaction', interaction=Interaction(w_data=w))
result.mi_estimate                      # negative for redundancy, positive for synergy
result.dataframe[['mi_xw_y_mean', 'mi_x_y_mean', 'mi_w_y_mean', 'amplification_factor']]
```

Interaction information $I(X,W;Y) - I(X;Y) - I(W;Y)$
([THEORY.md](THEORY.md#interaction-information)) combines three networks per
repeat. The library reports it as measured, including its sign. Its
amplification factor is the sum of the magnitudes of the three components divided
by the magnitude of the result. A combination of three terms reaches a large
factor more readily than a combination of two.

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
estimates the direction from Y to X and the directionality index
$(\mathrm{TE}_{X \to Y} - \mathrm{TE}_{Y \to X}) /
(|\mathrm{TE}_{X \to Y}| + |\mathrm{TE}_{Y \to X}|)$. The reverse direction is
`te_yx`, with components `i_yxpast_xfuture` and `i_xpast_xfuture` and its own
`amplification_factor_yx`. A repeat that comes out negative is reported as 0 and
keeps the measured value in `mi_raw` (`te_yx_raw` for the reverse direction).
Transfer entropy measures predictive information flow, so a large value in one
direction is no evidence of a causal influence in that direction
([THEORY.md](THEORY.md#temporal-information-quantities)).

For conditional transfer entropy, `Transfer(w_data=w)` also conditions both
components on the history of W. `details[config_id]['n_samples']` gives the
number of history windows that were built.

The library builds the histories row by row from input series of shape
`(T, n_channels)`. With `processing=`, every stream is first put on one grid whose
rows are one time step wide and cover time without gaps
([one grid](#one-grid-for-every-stream)). `window_size` must then be one step:
`bin_size` for spikes and one sample for continuous data. Leave `step_size`
unset. Categorical streams use the last axis for their categories, so they are
refused.

### `pairwise`

```python
result = nmi.run(x, mode='pairwise', n_workers=8)          # every pair of X's channels
result = nmi.run(x, y, mode='pairwise', n_workers=8)       # X's channels against Y's
matrix = result.get('mi_matrix')
result.plot()                                               # the matrix as a heatmap
```

`mode='pairwise'` trains one network per pair of channels. `dataframe` has one
row per pair with the values averaged over repeats. `mi_matrix` arranges these
values as a matrix. Without `y_data`, each pair is estimated once and written to
both halves of a matrix with a zero diagonal, so adding the matrix to its
transpose would double every entry. With `Pairwise(pairs=[(0, 1), (2, 5)])`,
only the listed pairs are estimated and the other entries stay 0.
`Output(channel_names_x=...)` labels the heatmap.

### `dimensionality`

```python
from neural_mi import Dimensionality

result = nmi.run(x, mode='dimensionality',
                 dimensionality=Dimensionality(n_splits=5), n_workers=4)
result.get('dimension_at_most'), result.get('dimension_at_most_std')
result.plot()          # MI against embedding_dim, with the reading
```

`mode='dimensionality'` estimates the MI at a series of embedding sizes and
reports the smallest size whose curve reaches `saturation_ratio` (0.95) of its
plateau ([THEORY.md](THEORY.md#the-dimension-that-carries-the-shared-information)).
This value is an upper bound on the number of dimensions needed to carry that
fraction of the shared information. An encoder with spare capacity can build
combinations of the true factors. These combinations look like factors in its
spectrum but add nothing to the MI, so the participation ratio counts them while
the curve does not.

A large reference fit runs first (`reference_dim`, 64 unless set). Its
participation ratio sets the size of the grid of `embedding_dim`. The grid of
each split stops once three values in a row reach the threshold.
`Dimensionality(embedding_dims=range(1, 13))` fits exactly the values given.
Each value gets `n_restarts` networks (4) and the best one counts. The messages
say when the grid grows, stops early or is extended
([MESSAGES.md](MESSAGES.md#dimensionality)).

Without `y_data`, the mode splits the channels of X into two halves with
`split_method` and compares the halves. `'random'` draws `n_splits` (5)
different channel assignments and reports the median value with the standard
deviation over splits. The other split methods give one split. With `y_data`, the two views are
X and Y. The image splits (`'horizontal'`, `'vertical'`, `'row_interleaved'`,
`'col_interleaved'`, `'diagonal'`, `'antidiagonal'`) need input of shape
`(N, C, H, W)`. The two diagonal splits cannot be used with a convolutional
encoder.

`dataframe` has one row per split and `embedding_dim`. Its `mi_mean` averages
the restarts, `mi_best` is the best of them and `mi_curve` is the running maximum
from which the reported dimension is taken. The participation ratios are
`pr_eig_mean` and `pr_singular_mean`. `runs` keeps every network. `details` gives
`dimension_at_most`, `dimension_at_most_std`, `dimension_at_most_per_split`,
`plateau` (per split), `embedding_dims` (every value fitted), `reference_dim`, the
participation ratios of the reference fit (`pr_singular` and `pr_eig`) and
`stopped_early`.

This mode uses the hybrid critic unless `critic_type` is set. It runs a
separable critic with a warning and refuses the concat critic, because that
critic has no embedding of either side. Unless you set them, `n_epochs` is 500
with `patience` 50. Without `y_data`, `shared_encoder` is `True` unless set. Here
the hybrid critic uses layer normalisation in its MLP layers unless `norm_layer`
is set. Layer normalisation keeps the fits at an embedding size equal to the
dimension from stalling below the full value. It also divides out the overall
scale of each sample's hidden activations, so it can lower the whole curve where
that scale carries information. The reported dimension is measured against the
curve's own plateau.
`sweep_grid` refuses `embedding_dim` and `run_id` because the mode sets both.

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
different window shift at every epoch. `mi_std` measures how much the procedure
varies over these repeats. It says nothing about how the estimate would change
on a new recording.

A configuration grid runs every combination of its keys. The keys can be
settings of any config or processor parameters such as `window_size`,
`step_size` or `bin_size`. The library windows the data separately for every
value of a processor parameter. It refuses a processor parameter that no
processor of the call reads. Values given as lists come back as tuples in `runs`
and `dataframe`.

### Choosing between configurations

A single estimate is the best that its configuration can do, but another
configuration may do better. For real work, sweep the configurations, take the
best by held-out value and keep it for later analyses.

A configuration grid returns one estimate per configuration. Each is the
training value at the epoch that the library picks from its held-out curve.
Compare configurations by their held-out values (`test_mi` in `runs`) rather
than by the reported ones. On average, the held-out value is a lower bound on
the information. The training value is not, because a network that overfits can
push it above the true value. A run warns when the held-out value of its network
falls below `Training(gap_share=...)` of the estimate by more than noise
([MESSAGES.md](MESSAGES.md#training)). For a difference quantity, compare the sum
of the held-out values of its terms (`details[config_id]['trainings']`, one row
per term).

`result.best` makes this comparison for `mode='sweep'`. Its first row is the
configuration with the highest held-out MI, averaged over the repeats of the
configuration (`test_mi_mean` in `dataframe`). The rows after it are the
configurations whose held-out MI lies within `Training(top_se=...)` standard
errors of that value, so the held-out rows cannot tell them apart from the best
one. `result.summary()` names them.

The best value is the highest of several values measured on the same held-out
rows, so it is optimistic. Report the value of the chosen configuration from a
run of its own. Choose its nulls the same way, since the choice is part of the
estimate. If configurations that tie on held-out data report different
estimates, the held-out data cannot determine the estimate. That spread then
belongs in its uncertainty.

### What each mode accepts

| mode | repeats | configuration grid | `rigorous` |
|---|---|---|---|
| `estimate` | refused | refused | through `mode='rigorous'` |
| `precision` | refused | refused | refused |
| `sweep` | yes | yes | through `mode='rigorous'` with the same grid |
| `rigorous` | one extrapolation per repeat | yes | built in |
| `lag` | yes | yes | refused |
| `conditional`, `interaction`, `transfer` | yes | yes | `rigorous=True` on the mode's config |
| `pairwise` | yes | yes | refused |
| `dimensionality` | through `n_restarts` | one curve per configuration | refused |
| every named quantity | yes | yes | `rigorous=True` |

A call with `n_workers` above 1 warns when it trains only one network, because
there is nothing to run in parallel.

### Bias correction for the difference quantities

`rigorous=True` on `Conditional`, `Interaction` or `Transfer` extrapolates the
quantity itself. At every chunk the components are trained on the same samples,
so part of their shared noise cancels when they are combined. The library then
fits the combined values against $\gamma$.

```python
result = nmi.run(x, y, mode='conditional',
                 conditional=Conditional(w_data=w, rigorous=True, gamma_range=range(1, 11)),
                 n_workers=8)
result.get('is_reliable'), result.get('mi_error')
```

You set the fit with the fields of `Rigorous` on the config of the mode. `runs`
has the same fit columns as in `mode='rigorous'`.
`details[config_id]['trainings']` has one row per chunk with the combined value.

### Permutation tests

`permutation_test=True` tests every row of the result against a null
distribution. Each null trial reruns the call with the source X shifted in time
while Y and W stay in place. Shifting X breaks its link with Y. Y, W, the link
between them and the history of Y in transfer entropy stay as recorded. X keeps
its own temporal structure except at the seams where it wraps around or where
blocks meet.

```python
result = nmi.run(x, y, mode='estimate', permutation_test=True, n_permutations=200,
                 n_workers=8, seed=0)
result.get('p_value')
result.get('null_distribution')
```

The default `permutation_shuffle='circular'` shifts X by one random offset with
wrap-around. The offset is at least 10% of the length of the recording away from
zero, so that no trial leaves X nearly aligned. `permutation_shuffle='block'` cuts
X into blocks as long as the span of one row and reorders them, so rows that
share samples move together. Both work on windowed rows, raw series and spike
times. A spike population moves as a whole and keeps the structure among its
neurons.

The null has one set of trials per row of `dataframe`. A lag scan gets a null at
every lag, and a pairwise matrix gets one for every pair.
`details[config_id]['null_distribution']` contains the trial values: a list for a
configuration without an axis and a dict keyed by the axis value otherwise. For a
quantity that cannot be negative, the trials are reported like the result, with
no value below zero ([THEORY.md](THEORY.md#the-reported-number)).
`null_distribution_raw` keeps the same trials as measured.
`dataframe['p_value']` is $(1 + \#\{\text{null} \geq \text{observed}\}) / (1 +
n)$.

The smallest p-value that $n$ trials can give is $1/(n+1)$, so a test with fewer
than 100 trials warns. Every trial reruns the whole call with every configuration
of the grid, so the test costs $n$ times as much as the call.

`rigorous`, `precision` and `rigorous=True` refuse a permutation test before
doing any work. So does `dimensionality`, because its result is a curve rather
than a single value that a null could be compared with. So does `pairwise`
without `y_data`, because moving X would move both sides of each pair.

### Workers and seeds

`n_workers` runs the independent tasks of a call in parallel: repeats,
configurations, lags, pairs, chunks of the rigorous ladder, splits, and
permutation trials. With `seed` set, each network seeds itself from the seed, its
repeat (`run_id`) and its role in the estimate (the gamma and chunk of a rung, the
split and width of a dimensionality fit). Its place in a grid plays no part. A
seeded call therefore returns the same numbers at any `n_workers`. A
configuration also gives the same number alone or in a grid, in any order, and
whether a setting is passed in the call or as a grid value. Every configuration
runs a given repeat from the same seeds, so a comparison between settings is
paired. Each repeat draws its own seeds, including those of its window shifts. A
call without a seed draws one from NumPy's global state and records it in
`result.params['random_seed']`, so passing that seed back reproduces the call.

Each worker gets an equal share of the CPU threads
(`os.cpu_count() // n_workers`, and at least one). Workers that each used every
core would compete for the cores. The small networks that are typical here would
then train more slowly in parallel than with one thread each. Setting
`OMP_NUM_THREADS` before the call keeps your own count.

The terms of a difference quantity (conditional MI, transfer entropy,
interaction information and the named quantities built on them) are separate
trainings on the same rows. They use one plan of every stream that the quantity
reads ([One grid for every stream](#one-grid-for-every-stream)). Every term of a
repeat draws its split, its evaluation rows and its shifts from generators that
are seeded alike, whether or not a seed is passed to the call. Each task reports
which rows it used. The library would refuse a repeat whose terms used different
rows.

A script that runs with `n_workers` above 1 needs a `__main__` guard. The
workers start by re-importing the module that launched them, so a top-level
`nmi.run(...)` would run again in every worker.

```python
if __name__ == '__main__':
    result = nmi.run(x, y, mode='sweep', sweep_grid=grid, n_workers=4)
```

A notebook needs no guard because its kernel is never re-imported.

---

## Preparing the data

### Processors

`Processing` names the processor for each raw stream. Without a processor, the
library treats the data as already windowed.

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
n_channels, width)`. Already windowed 2-D data get a last axis of size 1. 4-D
data are passed on unchanged for `embedding_model='cnn2d'` and the image
augmentations.

Categorical labels of an integer type are used as they are and must be
non-negative. Labels of another numeric type are mapped to $0, \dots, n-1$ in
sorted order with a warning. Text labels are refused.

The width of a window depends on the processor:

| processor | width of a window of `window_size=w` |
|---|---|
| continuous | $w$ samples, covering $[t, t+w)$ |
| categorical | the number of categories, under `encoding='majority_vote'` and `'probability'`; categories times samples under `'full_trajectory'` |
| spike, spike times | `max_spikes_per_window`, set by the densest window |
| spike, with `bin_size` | $w / \mathrm{bin\_size}$ bins |

`x_window_width` and `y_window_width` on a dataset built by `create_dataset`
give the widths directly.

### Units of time

A stream with a clock (`x_time`, `y_time`, `w_time`) is windowed in the units of
its clock. A continuous or categorical stream with a `sample_rate` but no clock
is windowed in seconds on the clock $t_i = i / \mathrm{sample\_rate}$. A
continuous or categorical stream with neither is windowed in samples. Spike
times are taken as seconds. `window_size` and `step_size` are in these units.
The lags of `mode='lag'` are in seconds for spike data and for streams with a
`sample_rate`. Every other stream counts its lags in samples even when it has a
clock. On data that are already windowed, a lag counts windows. The library warns
when this happens.

`step_size` sets how far each window moves on. Below 1, it is taken as a fraction
of `window_size`:

| `step_size` | step at `window_size=0.5` |
|---|---|
| `None` (default) | 0.5: windows touch |
| `0.25` | 0.125, a quarter of the window, 75% overlap |
| `2.0` | 2.0, an absolute step |

The fraction rule keeps `step_size=0.25` at 75% overlap for every window size.
In seconds, windows and steps below 1 are common, so the rule turns
`window_size=0.5, step_size=0.5` into a step of 0.25 s. The library warns
whenever `window_size` is below 1 and `step_size` falls between 0 and 1. The
warning names the step that it applied. To request an absolute step `a` below 1,
set `step_size=a /
window_size`.

### One grid for every stream

All the streams of a call are windowed on one grid (the *row plan* of the call).
The grid starts at the latest start among the streams and ends at the earliest
end. A window is kept only where every stream has data. A window covers the same
interval in X, Y and W. Each stream keeps its own processor, parameters and
clock:

```python
processing = Processing(x='spike', x_params={'window_size': 1.0, 'bin_size': 0.1},
                        y='continuous', y_time=pos_t,
                        w='categorical', w_time=pos_t)
result = nmi.run(spikes, position, mode='conditional',
                 conditional=Conditional(w_data=direction), processing=processing)
```

When Y and W name no processor, they use the processor of X. When they are given
no parameters, they use the parameters of X. A stream that falls back to the
parameters of X keeps only the keys that its own processor takes. Parameters
given to a stream are its own and are not merged with those of X. Every setting
is resolved once for the whole call, from these rules and the defaults
(`defaults.stream_processing`).

A stream passed as rows that are already built (a 3-D array) has no raw series
to process and inherits nothing. Rows built in advance cannot share a call with a
windowed stream, because they have no time axis to align windows against. Every
stream is therefore either processed by a processor or passed already built.
When some streams need building, `create_dataset` builds the rows of every stream
on one grid. Rows built in advance must also match every other stream row for
row. Streams whose counts differ are refused rather than cut to the shorter one.
The grid has one window and one step, those of X, so the library refuses a
stream that asks for another window or for a step of another length.
`nmi.offset_quantity` places a stream at its own offsets, before the window of X
or across it ([Quantities without a named function](#quantities-without-a-named-function)).

Every term of a quantity uses the rows of the plan and only selects its streams.
For example, the conditioning term of conditional MI uses W and Y on the same
rows on which the joint term uses X, W and Y. The library fits its choices to the
data stream by stream, so they come out the same in every term. These choices are
the categories of a categorical stream, the bursty neurons of a spike stream
(`exclude_bursty_neurons`) and its spike slots.

On every route, a continuous or categorical window lies within the samples of
its stream, from the first sample to the end of the sample period of the last
one. The library never cuts a window that would run past either end. Inside that
stretch, a window is kept only when at least `min_coverage_fraction` of it
contains samples. `details` and `runs` report the fraction of windows kept
(`window_retention`) and the counts behind it (`n_windows_built`,
`n_windows_retained`). A window must be valid in every stream, so three streams
that each keep 62% of their windows keep only about 24% jointly. When retention
falls below 50%, the library warns and names the responsible stream.

### Silent spike windows

`drop_empty_windows=True` is the one default that changes the data. It drops
spike windows that contain no spikes, so the MI is estimated in bits per active
window over the windows in which the population fired. It is the default
because, in spike times, silence looks the same as a stretch without recording.
A row is kept only where every spike stream of the call fired, in every term.
`x_params={'drop_empty_windows': False}` keeps silent windows and estimates the
MI in bits per window. Whether the population is active can itself carry
information, so neither quantity is a rescaled version of the other
([THEORY.md](THEORY.md#silent-windows)).

Keeping silent windows is safe only when the whole extent was observed, because
in spike times a stretch without spikes looks the same as a stretch without
recording. This is the case when unobserved stretches were cut out beforehand.
It is also the case when the pairing includes a continuous stream with its own
clock, because the coverage check of that stream masks unobserved stretches.

Dropping silent windows also breaks the time axis, because consecutive windows
can then lie more than one step apart. For that reason, the quantities defined
by time offsets keep every window. Setting `drop_empty_windows=True` for them
warns that it has no effect. To get a series with one row per bin for these
quantities, set `window_size` equal to `bin_size`.

A sweep over the window size on spike data changes the quantity being estimated,
because wider windows are more likely to contain a spike. On a sparsely firing
population, retention rises from a small fraction of the windows at short
windows to all of them at long ones. `drop_empty_windows=False` keeps every
window across the sweep.

### Preprocessing the streams

Nothing changes the data unless you ask for it. `Preprocessing` transforms each
continuous stream before rows are built from it. It fits each transform on X, Y
and W separately. `result.params['preprocessing']` records what it applied:

```python
from neural_mi import Preprocessing
from neural_mi.preprocessing import PCA

nmi.run(x, y, preprocessing=Preprocessing(channel='zscore'))
nmi.mi_rate(x, y, h=10, half_width=5, preprocessing=Preprocessing(time='ar', channel='zscore'))
nmi.run(x, y, preprocessing=Preprocessing(channel=PCA(n_components=8)))
```

The channel transforms (`'zscore'`, `'zca'`, `'pca'`, `'copula'`) are
invertible, so every quantity keeps its value. They also remove the dependence of
the estimate on units. As an example of this dependence, a pair gives an estimate
close to its true value at unit scale but close to 0 at a thousand times that
scale or a thousandth of it. `PCA(n_components=k)` with `k` below the channel
count is the exception. It keeps only the directions of largest variance and
warns, because the information need not lie in them.

`time='ar'` filters each channel with its own autoregressive fit and keeps the
length of the series. AIC chooses the order of each fit. The filter leaves the
rate quantities unchanged in the limit of long windows and shrinks their
conditioning term ([THEORY.md](THEORY.md#getting-a-rate)). On smooth, finely
sampled signals, that term is several nats and the rate is only a small residual
of it. The networks estimate such a residual poorly. In one test with a shared
oscillation under larger private ones, `mi_rate` came close to the exact rate
after whitening but gave 0 or failed without it. Every other quantity changes
under the filter, so a call for any of them warns.

Spike-time and categorical streams take neither transform, because their
processors set their rows. The encoders then scale what they receive
([The scale of the rows](#the-scale-of-the-rows)). Arrays passed in as rows take
only the channel transform. To write a transform of your own, subclass
`nmi.preprocessing.Transform`. The subclass declares its `domain` (`'channel'`
or `'time'`), learns from a stream in `fit` and applies itself in `transform`,
both on `(n, n_channels)` arrays. The library warns that it cannot check whether
your transform keeps the information.

Every call also checks its streams after any preprocessing and before rows are
built. This check changes nothing. It warns about non-finite or constant
channels and about channels more than ten times away from unit scale. Its report
is logged at the info level (`verbose=True`). The report gives the channel scales
of each stream and, for a rate quantity, how much of the variance of each channel
its own past predicts, together with the option that would help.

### Window shifting

By default, `Training(shift_windows=True)` moves the windows of the training rows
at every epoch, so that the network sees many sets of window boundaries. The rows
stay the same rows. At each epoch the library draws one offset within one step
and starts the windows of every stream that much later. X, Y and W therefore
move together. Every term of a difference also moves its rows by the same offset
at the same epoch. The grid leaves one step at its end for the shift. Rows cut by
sample number that start one sample apart never move, because a shift would only
relabel them. On a grid in time they move by a fraction of a sample, taking
values between samples. Evaluation always uses the unshifted rows, so the
reported values do not depend on the last shift drawn.

| call | rows a processor cuts | rows passed in built already |
|---|---|---|
| every mode but `transfer` | shifted | not shifted |
| `transfer`, the named quantities, `offset_quantity` | shifted at `stride` above 1 | shifted at `stride` above 1 |
| `block_mi` | shifted | not shifted |

On a grid in time (a spike stream, an irregular clock or clocks that start at
different times), a moved window can fail the rule of its stream. Examples are a
spike window that falls silent under `drop_empty_windows` and a continuous window
that moves onto a gap. The row then keeps its unmoved windows in every stream for
that epoch, so training never sees a row that the estimate leaves out.

Transfer entropy and the quantities built from offsets keep every step as a row
at the default `stride=1`. At a larger `stride`, they are built at every step,
and each epoch keeps one row in every `stride` from a newly drawn phase. This is
the counterpart of a window shift for offsets. A shifting row reaches up to one
step further. The leak check of the split takes this into account. If you set
`shift_windows=True` explicitly where it cannot apply, the library warns. A
windowed generator whose MI is known for its own windows needs shifting switched
off ([Generators](#generators-with-a-known-answer)).

### `create_dataset`

`neural_mi.data.handler.create_dataset` builds the windowed streams without
training anything. Use it to check widths and retention or to drive the
`Trainer` directly. It raises a named error when a processor and the shape of the
data do not match. Given a mapping, it builds any number of named streams on one
grid:

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

`create_dataset(x, y, ...)` builds two streams named `x` and `y`. With three or
more streams, it returns a bundle that gives the first two streams as `x_data`
and `y_data` and every stream as `stream(name)`.

### Generators with a known answer

`nmi.generators` makes synthetic data with an exactly known MI or lag, so that
you can check an analysis before you trust it on a recording.

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
carries no information about Y, but the two together determine it. `exact`
approaches 1 bit as `noise` goes to zero. `generate_lagged_pair` puts the
dependence at a known lag, so `mode='lag'` should peak at `lag` and approach
`exact` there. `generate_spike_pair` puts the information in the spike count
under `coding='count'` and in the position of a burst under `coding='timing'`.
Under timing coding, the count is drawn independently.

The `exact` of a windowed generator is the MI between the windows that it built,
so the analysis has to use the same `window_size` and no shifting:

```python
x, y, exact = generators.generate_spike_pair(coding='timing', window_size=1.0)
nmi.run(x, y, mode='estimate',
        processing=Processing(x='spike', y='spike', x_params={'window_size': 1.0}),
        training=Training(shift_windows=False))
```

With shifting on, a window spans two independent draws and can carry more
information than `exact`. An estimate above `exact` therefore points to a problem
in the setup.

`LatentGaussian` gives the exact value of $I(A;B \mid C)$ for any choice of
processes and time offsets, so every temporal quantity has a number to check
against:

```python
from neural_mi.generators import LatentGaussian

oracle = LatentGaussian.shared(dims={'x': 8, 'y': 8, 'w': 8}, d=2, phi=0.9)
data = oracle.sample(T=20000, seed=0)                # {'x': (20000, 8), 'y': ..., 'w': ...}

past = lambda v, k=25: [(v, s) for s in range(-k, 0)]
oracle.exact(past('x'), [('x', 0)])                              # active information storage
oracle.exact(past('x'), [('y', 0)], past('y'))                   # transfer entropy, X to Y
oracle.exact([('x', 0)], [('y', 0)], past('x') + past('y'))      # instantaneous exchange
oracle.block_mi(30)                                              # grows without bound in w
oracle.mi_rate()                                                 # bits per step
oracle.affine_fit(30, 60)                                        # slope and intercept of I_w
oracle.storage_transfer('y', 'x')                                # storage of Y, transfer X to Y, infinite past
```

`LatentGaussian.shared` builds processes that share one AR(1) latent with
correlation time $\tau = -1/\log\phi$. The shared latent violates Massey's
no-feedback condition, so the directed quantities converge below the symmetric
MI rate. Only a two-sided window over X recovers that rate
([THEORY.md](THEORY.md#temporal-information-quantities)).
`oracle.sample(..., return_latents=True)` also returns the latent.
`oracle.snr(name)` gives the ratio of latent to noise variance for each channel.
The projections are random, so the channels of one process can differ in SNR by
two orders of magnitude. Choose a channel to display by its SNR.
`generate_shared_latent_gaussian(T, ...)` returns the data and the oracle
together.

The general model takes the latent dynamics, a readout matrix per process and the
noise: `LatentGaussian(Phi, proj, noise, Q)`, with $Z_t = \Phi Z_{t-1} +
\eta_t$, $\eta_t \sim N(0, Q)$ and $V_t = M_V Z_t + \epsilon_V$. A $\Phi$ that
couples components lets one group of latents drive another, so transfer has a
direction. A latent that only one process reads gives that process a memory of
its own. To ask about single channels, a spec can name channel indices into the
stacked processes instead of a process name.

```python
import numpy as np

Phi = np.array([[0.9, 0.0],                          # X's latent, with memory
                [0.5, 0.6]])                         # Y's latent, driven by X's at lag 1
proj = {'x': np.c_[np.ones((4, 1)), np.zeros((4, 1))],
        'y': np.c_[np.zeros((4, 1)), np.ones((4, 1))]}
directed = LatentGaussian(Phi, proj, noise=1.0)
directed.storage_transfer('y', 'x', k=10)            # from 10 bins of history
directed.exact([(0, -1)], [(4, 0)])                  # one channel of X to one of Y
```

---

## The named quantities

`nmi.transfer_entropy`, `nmi.mi_rate` and the other named functions each build
the arrays that their quantity needs and then call `run()`. They take every
keyword that `run()` takes and return the same `Results`.
[THEORY.md](THEORY.md#temporal-information-quantities) defines each quantity and
the identities between them.

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
| `mi_rate(x, y, h, half_width)` | $I(X_{all}(L); Y_0 \mid Y_{past}(h))$ | `conditional` |
| `instantaneous_exchange(x, y, k)` | $I(X_0; Y_0 \mid X_{past}(k), Y_{past}(k))$ | `conditional` |
| `directed_information_rate(x, y, k)` | $I(X_{past}(k), X_0; Y_0 \mid Y_{past}(k))$ | `conditional` |

$X_{all}(L)$ is a window of $2L+1$ steps centred on $Y_0$. `half_width` sets its
$L$.

### Calling them

Inputs passed on their own are series of shape `(T, n_channels)` sampled on a
regular grid. The offsets count the rows of this grid. With `processing=`, each
processor first puts its stream onto one shared grid with one time step per row.
The offsets then count the rows of that grid. For this, `window_size` is one step
(`bin_size` for spikes, one sample for continuous data) and `step_size` stays
unset, so the rows cover time without gaps. This is the route for spike times
and for streams on different clocks. It is the same for every quantity defined by
time offsets, transfer entropy included:

```python
r = nmi.active_information_storage(
    spike_times, k=3,
    processing=Processing(x='spike', x_params={'window_size': 0.1, 'bin_size': 0.1,
                                                'drop_empty_windows': False}))
```

The library refuses a grid whose rows are wider than one step or overlap,
because an offset of one row would then skip the steps inside a row or reach back
into the row before. `instantaneous_mi` builds its rows the same way as the
`k=0` forms of `instantaneous_exchange` and `directed_information_rate`. Pass
rows built in any other way to `nmi.run()`.

Window validation is off on this route, so every window is kept. Dropping a
window would put two distant windows next to each other and change what an
offset means. Gaps stay visible through the coverage warnings. For the same
reason, keep `drop_empty_windows=False` on spike streams.

`block_mi` is the one quantity that windows the data itself. It sets
`window_size`, merges it into the given `Processing` and treats the data as
continuous when no `Processing` is given.

`mi_rate`, `instantaneous_exchange` and `directed_information_rate` condition on
a group with its own window length. Like every joint term, their joint term takes
the two groups as a pair ([the encoder of a difference](#the-encoder-of-a-difference)).

```python
r = nmi.mi_rate(x, y, h=[0, 5, 10, 20], half_width=20, n_workers=4)
```

A window on X that is too narrow (`half_width`) leaves signal out, so the
estimate comes out too low. Too little history on Y (`h`) can make the estimate
too high or too low. It is too high when the memory of Y comes from what Y shares
with X, and too low when Y has a private memory that its own past would cancel
([THEORY.md](THEORY.md#temporal-information-quantities)). Sweep one of the two,
fix it past its knee and then sweep the other. Passing an iterable for both
raises an error. `half_width` has no default, because the answer depends on it
as much as on `h`.

Every window parameter (`k`, `future_k`, `h`, `half_width`, `history_window`) is
a whole number of steps and is checked the way `stride` is. The conditioning
histories (`h` of `mi_rate`, and `k` of `instantaneous_exchange` and
`directed_information_rate`) may be 0. There is then nothing to condition on, so
the call computes `instantaneous_mi` with one network and warns. For `mi_rate`,
this is the MI between the present of Y and the whole window of X. In the sweep
above, the `h=0` point is this baseline.

### Sweeps, repeats and rigour

The parameter of a quantity (`k`, `history_window`, `window_size`, `h`,
`half_width`) takes a scalar or an iterable. With an iterable, the call runs every
value in parallel across `n_workers` and returns one `Results` with one
configuration per value. `sweep_grid` adds repeats and further settings at every
value.

```python
r = nmi.transfer_entropy(x, y, history_window=[2, 4, 8, 16],
                         sweep_grid={'run_id': range(3)}, n_workers=8)
r.dataframe[['history_window', 'mi_mean', 'mi_std']]
```

`rigorous=True` (or a `Rigorous(...)` with fit settings) extrapolates every
repeat of every quantity. `transfer_entropy` and `conditional_transfer_entropy`
also take `bidirectional=True`.

For a quantity built as a difference, `mi_std` over repeats tells you whether the
difference is resolved at all. A value smaller than its spread has no reliable
size or sign.

### `stride`

By default, the quantities built from offsets cut a row at every time step, so
neighbouring rows overlap in all but one step. `stride=n` cuts one row every `n`
steps:

```python
r = nmi.active_information_storage(x, k=5, stride=4)
```

Neighbouring rows that nearly duplicate each other can inflate the estimate under
a random split when the encoder treats them as copies (a recurrent encoder, or a
CNN that averages over positions). They also overstate the $\log K$ ceiling,
because $K$ counts rows without regard to how many of them are near copies.
`stride` counts whole rows and refuses a fraction. It is separate from
`step_size`, because the fractional meaning of `step_size` refers to a single
window length that these quantities do not have.

### Quantities without a named function

Every quantity above is $I(A;B \mid C)$ for some choice of processes and offsets
in each group. Each builds its rows from this *spec*. `offset_quantity` runs a
spec for a quantity that has no function of its own, in the same way as the named
functions:

```python
spec = {'A': [('x', s) for s in range(-10, 0)],     # X's past
        'B': [('y', 0)],                            # Y now
        'C': [('y', s) for s in range(-10, 0)]}     # given Y's past
r = nmi.offset_quantity({'x': x, 'y': y}, spec, stride=1)
```

Offsets count rows, with negative offsets in the past and zero at the present.
Every group is cut to the range over which all its offsets exist. A spec with a
`C` group runs `mode='conditional'`. Its joint term takes the pair (A, C). A
spec without a `C` group is a single MI estimate. `result.params['spec']` records
the spec. The rows record their geometry, so the leak checks of the split, the
rigorous chunks and the permutation blocks treat them as the rows of a named
quantity. `neural_mi.utils.build_offset_arrays(data, spec, stride=...)` returns
the arrays themselves, each of shape `(n_valid, n_channels, n_offsets)`. If you
pass these arrays to `nmi.run()` by hand, the library does not know that they are
in time order.

---

## Models, estimators and critics

### Estimators

The critic scores every pairing of samples in a batch. The estimator turns the
matrix of scores into a bound on the MI
([THEORY.md](THEORY.md#estimating-mutual-information-with-a-critic)).
`'ksg'` trains no critic and estimates from nearest neighbours instead.

| `Estimator(name=...)` | bias and variance | ceiling |
|---|---|---|
| `'infonce'` (default) | low variance, biased low near its ceiling | $\log N$ nats, $N$ the number of samples evaluated |
| `'smile'` | lower bias at large MI, higher variance | none |
| `'ksg'` | no training; comes out too low at large MI and in many dimensions | none |

InfoNCE cannot report more than the logarithm of the number of samples on which
it is evaluated. For the held-out set, the library draws `eval_size` rows once
and scores them at every epoch, in `test_eval_blocks` blocks of at most
`max_eval_samples` rows. For the reported training value, it evaluates
`train_eval_size` rows in `train_eval_blocks` blocks in the same way. On each
side, the rows of one block set the ceiling. `Training(train_subset_size='full')`
evaluates the reported estimate on every training row in this way, with the
precision of all the rows and the memory cost of one block.
`Training(test_subset_size='full')` does the same for the held-out curve at every
epoch. `runs` reports the ceiling of each side and how close the estimate came to
it (`train_saturation`, `test_saturation`). `batch_size` sets how many negatives
each training step sees and places no cap on the reported value.

A held-out value near its ceiling calls for more held-out rows in each
evaluation or for SMILE. A larger `max_eval_samples` gives more rows when the
split has more held-out rows than one evaluation scores. Otherwise only more data
helps. An estimate near the ceiling of the training values while its held-out
value is far below may come from a network that memorised its training rows
([MESSAGES.md](MESSAGES.md#ceiling-and-capacity)).

The output of InfoNCE is bounded by $\log N$. SMILE clips only the scores of the
unpaired samples in its normalising term, so its output has no such bound.
`Estimator(name='smile', params={'clip': 5.0})` sets the clip. A lower clip
lowers the variance but can raise the bias.

`estimator='ksg'` is the Kraskov-Stögbauer-Grassberger estimator
([THEORY.md](THEORY.md#estimating-mutual-information-from-neighbours-ksg)). It
trains no network and uses every row, so the `Model`, `Training` and `Split`
settings have no effect. A call that sets any of them warns. KSG runs in every
mode except `'dimensionality'` and `'precision'`, because these two modes use the
embeddings of a trained critic. Conditional MI, transfer entropy, interaction
information and the named quantities built on them use its conditional form on
the same rows. `Estimator(name='ksg', params=...)` takes three parameters:

- `k` (default 4) is the number of neighbours.
- `gaussianize` (default True) maps each column through its ranks to a standard
  normal first.
- `exclude` (default `'auto'`) sets the separation in rows below which two rows
  are never neighbours. `'auto'` uses the autocorrelation time of the rows.

`runs` reports the value as `train_mi`. Repeats on the same rows return the same
value. `mode='rigorous'` gives the spread and corrects the bias. The neighbour
search uses a k-d tree up to 20 dimensions and compares every pair of rows beyond
that. The cost then grows with the square of the number of rows. Thousands of
rows with a thousand dimensions take seconds on a laptop CPU.
`neural_mi.estimators.ksg_mi(x, y)` and `ksg_cmi(x, y, z)` take arrays directly
and return nats.

KSG assumes continuous values. Spike windows are not continuous, because their
padded slots and their counts produce ties. KSG breaks the ties of each variable
with its own noise, so a variable and an exact copy of it no longer look alike. A
conditional estimate given that copy then comes out away from zero where the
true value is 0. Use the trained estimators on spike data. KSG warns when most
values of a column are tied.

### Encoders

Each side passes through an encoder before the critic scores it. Every encoder
outputs an array of shape `(batch, embedding_dim)`.

| `embedding_model` | input | for |
|---|---|---|
| `'mlp'` (default) | `(N, C, W)`, flattened | a general default |
| `'cnn'` | `(N, C, W)` | local patterns along the window, kept at their positions (`cnn_pooling='mean'` averages them over the window) |
| `'cnn2d'` | `(N, C, H, W)` | image-like input of any size |
| `'gru'`, `'lstm'` | `(N, C, W)` | sequences, optionally bidirectional |
| `'lru'` | `(N, C, W)` | long sequences (a linear recurrent unit) |
| `'tcn'` | `(N, C, W)` | long windows (dilated convolutions) |
| `'transformer'` | `(N, C, W)` | self-attention over the window with `nhead` heads |
| `'deepsets'` | `(N, C, W)` | raw spike times, invariant to the order of a window's spikes |
| `'pretrained_backbone'` | `(N, C, H, W)` | a frozen torchvision network with a trainable head |

The encoder classes in the table are importable from `neural_mi.models`.
`pretrained_backbone` takes the name of the torchvision network in
`pytorch_predefined` and loads ImageNet weights with `pretrained=True`. Input of
a size other than 224 by 224 pixels is resized by bilinear upsampling on the
first forward pass, with a warning.

`embedding_model` sets the encoder of both sides. `embedding_model_y`,
`custom_embedding_cls_y`, `hidden_dim_y`, `n_layers_y` and `embedding_dim_y`
override it for Y:

```python
model = Model(embedding_model='deepsets', embedding_model_y='mlp',
              hidden_dim=128, hidden_dim_y=32, embedding_dim=32)
```

A separable critic scores a pair by the dot product of the two embeddings and
needs one `embedding_dim`. Only the hybrid critic accepts a different
`embedding_dim_y`. The concat critic builds no encoder, so it ignores every
encoder setting with a warning, including the settings for Y and
`shared_encoder`. `shared_encoder=True` uses one encoder for both sides and
refuses any asymmetry.

`bias=False` builds the encoders without bias terms, so that an all-zero input
has an embedding of exactly zero. The MLP then uses an RMS normalisation without
learned parameters, because `LayerNorm` and `BatchNorm` subtract the mean and
move a zero away from zero. Spectral normalisation is unaffected. The positional
encoding of the transformer and the frozen biases of a pretrained backbone add
terms that do not depend on the input, so these two encoders warn that the
setting cannot fully apply.

`Model(activation=...)` sets the nonlinearity of every rectified hidden layer
that the library builds. These are the hidden layers of the MLP, CNN, CNN2D, TCN
and DeepSets encoders, the feed-forward blocks of the transformer, the MLPs
around the encoders and the decoders. The default is a leaky ReLU. A ReLU gives
zero output and zero gradient for a negative input, so a unit whose input is
negative on every row stops learning. A leaky ReLU keeps a small slope for a
negative input, so such a unit can still learn. The other options are `'relu'`,
`'gelu'`, `'elu'`, `'silu'`, `'tanh'` and `'sigmoid'`. The GRU and LSTM cells and
the LRU block keep their own nonlinearities. A call whose networks have none of
these layers is told that the setting has no effect.

An MLP encoder also has a linear path from its input to its embedding
(`Model(linear_path=True)`). The path is a linear map with spectral
normalisation, multiplied by one trainable number that starts at zero. The path
therefore starts closed and opens as training finds a use for it. It gives the
encoder the linear part of the dependence directly. This helps when the
information is spread over many weak directions and the rows are few, because the
layers of the network would otherwise fit the training rows before they find
every direction. The path is defined only for the flattened input of the MLP, so
the other encoders have none. Neither do the MLP heads of the hybrid and concat
critics. An MLP without hidden layers is already linear and has no path either.
`linear_path=False` builds the MLP without it.

### The encoder of a difference

Conditional MI, transfer entropy, interaction information and the named
quantities built on them subtract one term from another. The subtracted term
takes the conditioning set alone, while the joint term takes it together with
the rest of X. Every term of such a difference encodes X with one branch per part
of X and one small MLP head on top (`BranchedEmbedding`). The joint term has two
branches and every other term has one. Each branch is the encoder that
`embedding_model` or `custom_embedding_cls` names. The conditioning set therefore
takes the same path, branch and then head, in every term that uses it. The terms
learn what they share in the same way, so their errors on it cancel in the
difference ([THEORY.md](THEORY.md#the-encoder-of-a-difference)). Each term keeps
its own weights, training and epoch.

In the joint term, X is the pair (rest of X, conditioning set), so its two parts
may differ in channels, window length and modality. Every stage that processes X
takes the pair: the decoder reconstructs each part, the augmentations and the
corruptions act on each part, a permutation null moves the part of the pair that
belongs to X, and the `'concat'` critic joins the flattened parts. KSG estimates
the joint term as the MI of the conditioning set with Y plus the conditional MI
of the rest given the conditioning set. `shared_encoder=True` is refused for these
terms, because the encoder of X is branched and the encoder of Y is not.

### Critics

| `critic_type` | scores a pair by |
|---|---|
| `'separable'` (default) | the dot product of separately computed embeddings |
| `'hybrid'` | a small network on the concatenated embeddings, sized by `hidden_dim_head` and `n_layers_head` |
| `'concat'` | one network on the concatenated raw inputs |

The separable critic scores a whole batch with one matrix product. The hybrid
and concat critics pass every pairing through a network. Each side passes
through the first layer of that network only once. The pairs are then scored in
large blocks. As a result, the hybrid critic trains in about the time that the
separable critic takes, while the concat critic takes a few times as long.

The concat critic has no `embedding_dim` and gives no embedding of either side.
With this critic, a request that needs the embeddings (`use_decoder`,
`return_embeddings`, `track_embeddings`, `track_spectral_history`) is refused
before training. Its results also have no participation ratios. Each critic
declares whether it embeds each side (`embeds_each_side`). Every check uses this
declaration. `mode='dimensionality'` uses the hybrid critic.

`Model(norm_layer='layer')` helps the hybrid critic train on wide data. It also
divides out the overall scale of each sample's hidden activations, so the
estimate can come out lower where that scale carries information.

Every score of a critic that the library builds starts at exactly zero
(`Model(zero_start=True)`). The separable critic starts half of the readout rows
of each side at zero: the first half for X and the second half for Y. The two
embeddings then start in complementary halves, so their dot product is zero for
every pair, while each zeroed row receives a gradient from the live half of the
other side. The hybrid and concat critics start the last layer of the network
that scores the pair at zero. Without the zero start, the spread of the first
scores depends on the dimension of the data and on the size of the network. A
shared encoder has one readout. It could start at zero only as a whole and would
then never move, so a shared encoder keeps its start. A custom critic starts as
its recipe builds it.

### Custom encoders and critics

A custom encoder subclasses `BaseEmbedding` and declares how its `input_dim` is
counted. The default `input_style = 'flattened'` gives `n_channels * window` as
one number. `input_style = 'channels'` gives the channel count and suits an
encoder that handles the window axis itself:

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

A class receives `bias` only when its `__init__` accepts it, directly or through
`**kwargs`. An explicit `bias=False` that cannot reach the class warns.

`Model(custom_critic=...)` takes a class or function that builds the whole
critic, in the same way as `custom_embedding_cls` takes an encoder class. Every
task calls it under its own seed and passes the parameters that it names. These
can be input sizes (`input_dim_x`, `input_dim_y`, `n_channels_x`, `n_channels_y`,
`window_size_x`, `window_size_y`) and architecture settings (`embedding_dim`,
`hidden_dim`, `n_layers` and the network settings). Every repeat and every term
of a difference therefore trains its own critic, sized for its X. In the joint
term of a difference, X is the pair (rest of X, conditioning set). The critic then
receives a tuple of two tensors. Each size of X that it names is a pair of the
sizes of the two parts. An architecture setting that the critic does not name is
ignored with a warning. `input_scaling` reaches the critic only when the critic
names it.

`use_variational` and `beta` apply as they do to any critic. The loss is then the
KL term that its `forward` returns minus `beta` times the MI term. A recipe gives
the embedding of each side by defining a method `get_training_embeddings(x, y)`
that returns `(z_x, z_y)`. A decoder rebuilds each side from this embedding. The
library uses these embeddings wherever it uses those of a critic (decoders,
returned embeddings, `extract_embeddings`). An object is refused, because one
object would be trained in place and shared by every repeat. With `n_workers`
above 1, the class or function must be defined at module level so that the
workers can import it.

```python
class MyCritic(nmi.models.BaseCritic):
    def __init__(self, input_dim_x, input_dim_y, embedding_dim):
        super().__init__()
        self.fx = torch.nn.Linear(input_dim_x, embedding_dim)
        self.fy = torch.nn.Linear(input_dim_y, embedding_dim)

    def forward(self, x, y):
        zx = self.fx(x.reshape(len(x), -1))
        zy = self.fy(y.reshape(len(y), -1))
        return zx @ zy.T, torch.tensor(0.0)

result = nmi.run(x, y, model=Model(custom_critic=MyCritic, embedding_dim=16))
```

### The scale of the rows

The networks start from weights sized for inputs near zero and of unit scale. By
default (`Model(input_scaling='auto')`), each encoder therefore removes the units
and the offset of its input before its first layer. It uses statistics of the
training rows, so that an estimate does not depend on the units of the data. How
it does so depends on the kind of stream behind the input:

- A continuous stream, or an array passed in as rows, can contain a different
  quantity in each channel. Each channel is centred and divided by its own
  standard deviation.
- Every neuron of a spike stream shares one unit, so the stream takes one scale
  and a quiet neuron stays quiet. Spike counts are centred neuron by neuron and
  divided by the spread of the whole stream. Spike times are divided by their
  root mean square over the whole stream and are not centred, so the padding of
  empty slots stays where it is.
- A categorical stream consists of one-hot vectors or probabilities and has no
  unit to remove, so it is left as it is.

An encoder built without biases (`Model(bias=False)`) cannot add an offset back,
so its input is scaled without centring. DeepSets tells a spike from padding by
the padding value, so its scaling leaves that value in place. X, Y and W each
have their own statistics, as does each side of a shared encoder. The statistics
are part of the network. They are saved with it and applied by
`extract_embeddings`. `Model(input_scaling='none')` leaves every input as it is.

Scaling channel by channel gives every channel of a continuous stream the same
weight. This is right when the channels differ only in units, but it amplifies
the noise of a channel that is quiet because it carries little signal. The data
check reports which streams the encoders scale. It warns in three cases: when the
channel scales of a stream spread widely, when a channel is mostly one repeated
value (as in padded or sparse rows) and when a channel has heavy tails. For heavy
tails, `Preprocessing(channel='copula')` is a better choice
([MESSAGES.md](MESSAGES.md#preprocessing-and-the-data-check)).

### Regularised objectives

`Model(use_variational=True)` makes any encoder variational and adds a KL term
weighted against the MI term by `beta`. Under the concat critic, the variational
layer acts on the score of each pair, because there are no separate encoders.

`Model(use_decoder=True)` adds a decoder that reconstructs each input from its
embedding. The decoder is built for the size of the embedding that the critic
gives. It needs a critic that gives an embedding of each side. The separable and
hybrid critics do, and a custom critic does through `get_training_embeddings`.
`critic_type='concat'` scores the pair without embedding either side, so it
refuses the decoder. The reconstruction loss keeps the embedding from
collapsing. `decoder_lambda` (or `decoder_lambda_x` and `decoder_lambda_y`) sets
its weight. The decoder mirrors the encoder: an MLP, CNN, CNN2D, GRU, LSTM, LRU,
TCN or transformer decoder for the matching encoder, and an MLP decoder with a
warning for the others. `decoder_output_activation_x` and `_y` choose `'linear'`
or `'sigmoid'` outputs with a squared-error loss, or `'softmax'` outputs with
cross-entropy. `decoder_recon_loss` in `runs` is the weighted reconstruction term
at the best epoch. [THEORY.md](THEORY.md#regularised-objectives) gives the
objectives.

### Augmentations

`Training(augmentation_params=...)` perturbs every training batch and leaves clean
the evaluation batches from which the reported values come.
`augmentation_params_x` and `augmentation_params_y` set each side separately.
`{}` turns augmentation off for that side.

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

The flips, the rotation, the crop, the erase, the masks and the blur need input
of shape `(N, C, H, W)`. On any other input they are skipped with a warning.
Within a batch, the image augmentations run first, then the others and then the
custom ones.

---

## Plotting and animation

`result.plot()` draws the figure of the result's mode: the held-out MI against
epoch for `estimate`, the ladder and fit for `rigorous`, the components beside the
difference for the difference quantities, MI against $\tau$ for `precision`, the
matrix for `pairwise` and MI against embedding size for `dimensionality`. A result
over several configurations (a sweep or a lag scan) is drawn as MI against the
swept keys: a line for one key, a heatmap for two and bars for more.
`kind='line'`, `'heatmap'` or `'bar'` overrides the choice. `config_id=` picks the
configuration whose repeats the figure of the mode draws. `ax=` draws into
existing axes, and `show=False` leaves the figure open. For transfer entropy
estimated in both directions, the title names the direction that dominates when
the directionality index lies more than `direction_band=` from zero (0.1 by
default).

`Results.compare([r1, r2], labels=[...])` overlays results of one mode: the
training curves of `estimate` results, the curves of results over one swept key,
and the extrapolations of `rigorous` results. The overlays of training curves and
of extrapolations take one repeat per result.

`result.animate(output_path='training.gif')` animates the recorded training of
one network: the MI, the spectral metrics, the spectrum and the embeddings. The
embedding panel needs `Output(track_embeddings=True)`. `embedding_labels=`
colours its points by one or more labels. A `.mp4` path writes a video.
`config_id=` and `run_id=` pick the network, together with the axis values in
`mode='lag'` (`lag=`), `'pairwise'` (`ch_x=` and `ch_y=`) and `'dimensionality'`
(`split_id=` and `embedding_dim=`). A call that matches more than one network is
refused with an error that names the keys to pass.

`neural_mi.visualize` contains the functions behind these figures. Each takes
`ax=` and `show=` and returns the axes:

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

A call that names no device trains on the CPU unless a GPU is clearly faster for
it. A GPU pays a fixed cost for every operation, so on the small networks and
batches that are typical of neural data the CPU can be several times faster.
Parallel workers that share one GPU also wait for each other, whereas each worker
gets its own share of the CPU. Before the trainings start, NeuralMI times a few
training steps of the first network on the CPU, with the number of threads that
each worker will get. If all the trainings would take less than
`GPU_WORTH_SECONDS` on the CPU, the call stays there. Otherwise NeuralMI times the
same steps on the GPU (CUDA, or else Apple's MPS). It uses the GPU only if the GPU
beats the CPU workers together by the factor `GPU_MARGIN`. Both constants are set
in `neural_mi/device.py`. The decision is logged at the info level. `runs` or the
trainings table records the `device` of each network. A worker process started by
`n_workers` trains on the CPU unless the call names a device. `device='cpu'`,
`'cuda'` or `'mps'` skips all of this. `Training(use_amp='auto')` uses mixed
precision on CUDA.

`Training(max_eval_samples=...)` limits memory by capping the number of samples
that any one evaluation uses. `Model(max_n_batches=...)` limits it by capping the
size of the block of the score matrix that is computed at once. The default
`Training(dataset_device='cpu')` keeps the data in main memory and moves each
batch to the device. `'auto'` keeps the data on the device and is faster when the
data fit in its memory.

A message that needs your attention is a Python warning. Such a message reports a
result that may be off, a setting that was not used as given or data that the
library had to change. `warnings.filterwarnings` silences a category or turns it
into an error. Each kind of warning is shown once per call, notebook cell or
`nmi.grouped_warnings()` block, with a count of its repeats. What the library did
as asked is logged at the info level and hidden by default.
`nmi.set_verbosity(level)` sets the logging level for the session.
`nmi.set_verbose(True)` shows the informational messages. A call keeps that level
unless it passes `verbose=True` (informational messages for this call) or
`verbose=False` (errors only). [MESSAGES.md](MESSAGES.md) explains each message
under its text. The library's own exceptions derive from `nmi.NeuralMIError`:
`nmi.DataShapeError` for a shape that a processor cannot read,
`nmi.InsufficientDataError` for too little data for the request, and
`nmi.TrainingError` for a training run that failed. The warnings about a quantity
combined from several networks share the class `nmi.CombinationWarning`, so that
`warnings.filterwarnings` can silence them as a group.

Within one call, each kind of warning or log message is shown the first time that
it comes up. Later messages whose first sentence differs only in its numbers are
counted, and so are exact repeats. The call ends with one line per kind that says
how often it was raised. In IPython and Jupyter, the count runs over every
NeuralMI call of a cell, so the line comes when the cell ends. A loop of calls in
one cell therefore shows each warning once. In a script,
`with nmi.grouped_warnings():` around the loop does the same. Messages raised
outside a NeuralMI call are left alone. A warning raised in a worker process names
the line of the call.
