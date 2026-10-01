# Parameters

This page lists every setting `nmi.run()` accepts with its default and what it
does. The settings are grouped into config objects. A plain `dict` with the same
keys works wherever a config object is accepted. A default of "X's" means the
setting falls back to the value given for X.

[USING.md](USING.md) shows how the settings combine for each task.
[THEORY.md](THEORY.md) explains the ones whose meaning is statistical.

## Contents

- [`run()`](#run)
- [`Processing`](#processing) and the [processor parameters](#processor-parameters)
- [`Model`](#model)
- [`Training`](#training)
- [`Split`](#split)
- [`Estimator`](#estimator)
- [`Output`](#output)
- Mode configs: [`Rigorous`](#rigorous), [`Precision`](#precision), [`Lag`](#lag),
  [`Transfer`](#transfer), [`Conditional`](#conditional), [`Interaction`](#interaction),
  [`Pairwise`](#pairwise), [`Dimensionality`](#dimensionality)

## `run()`

The arguments of `nmi.run()` itself, beside the config objects.

| Parameter | Default | Meaning |
|--|--|----|
| `x_data`, `y_data` | required | The two variables. `y_data` is optional for `mode='pairwise'` (pairs within X) and `mode='dimensionality'` (two halves of X). |
| `mode` | `'estimate'` | The analysis: `'estimate'`, `'sweep'`, `'rigorous'`, `'lag'`, `'precision'`, `'conditional'`, `'interaction'`, `'transfer'`, `'pairwise'` or `'dimensionality'`. |
| `sweep_grid` | `None` | A dict of setting names to the values to run. A list, tuple, range or array gives one configuration per value and a single value fixes the setting. Every combination is one configuration and `run_id` in the grid repeats each one. The keys are the settings of `Model`, `Training`, `Split` and `Estimator` and the processor parameters. `Split(mode=...)`, `Split(gap_fraction=...)`, `Estimator(name=...)` and `Estimator(params=...)` are named `split_mode`, `split_gap_fraction`, `estimator_name` and `estimator_params`. |
| `n_workers` | `1` | Worker processes. Tasks run in parallel across repeats, configurations, chunks, pairs or permutation trials. Results do not depend on the count. |
| `seed` | `None` | Seed for Python, NumPy and PyTorch. Each task re-seeds from it and a fixed per-task key. A seeded call gives the same result at any `n_workers`. |
| `verbose` | `None` | `True` logs informational messages for the call and `False` only warnings and errors. `None` keeps the level set by `nmi.set_verbosity()`. |
| `show_progress` | `True` | Show progress bars. |
| `device` | `None` | Compute device, `'cpu'`, `'cuda'` or `'mps'`. `None` picks the fastest available. |
| `permutation_test` | `False` | Test every row of the result against a null built by rerunning the call with X moved in time. Adds a `p_value` column. |
| `n_permutations` | `10` | Number of null trials. The smallest p-value reachable is `1 / (n_permutations + 1)`. |
| `permutation_shuffle` | `'circular'` | How X is moved: `'circular'` shifts it by one random offset with wrap-around, excluding offsets within 10% of the recording's length of zero; `'block'` reorders contiguous blocks one window long. |

The mode configs are passed under the mode's own name: `run(..., mode='lag',
lag=Lag(lag_range=range(-10, 11)))`. A config for another mode is ignored with a
warning.

## `Processing`

How each raw stream is read: its processor, the processor's parameters, and its
clock. With no `Processing`, the data are taken as already windowed:
`(n_samples, n_channels)` or `(n_samples, n_channels, window_size)`.

| Parameter | Default | Meaning |
|--|--|----|
| `x` | `None` | X's processor: `'continuous'`, `'spike'` or `'categorical'`. |
| `x_params` | `None` | X's processor parameters, listed below. |
| `y` | X's | Y's processor. |
| `y_params` | X's | Y's processor parameters. When Y falls back to X's, only the keys Y's processor takes are kept. |
| `w` | X's | The third stream's processor, for `mode='conditional'`, `'interaction'`, `'transfer'` and the named quantities that take one. |
| `w_params` | X's | The third stream's processor parameters. |
| `x_time`, `y_time`, `w_time` | `None` | A timestamp per sample. With a clock `window_size` and `step_size` are in the clock's units. Without one they are in samples (seconds with `sample_rate`). |

Every stream of a call is windowed on one grid on which a window means the same
interval in X, Y and W and is kept only where every stream is valid.

### Processor parameters

`window_size` must be set for at least one stream and is shared by every stream.

**`'continuous'`** reads a regularly sampled signal of shape `(n_timepoints,
n_channels)`.

| Parameter | Default | Meaning |
|--|--|----|
| `window_size` | required | Window length in samples (seconds with `sample_rate` or a clock). |
| `step_size` | one window | Distance between window starts. Below 1 it is a fraction of `window_size`. |
| `min_coverage_fraction` | `0.2` | Fraction of a window that must hold samples for the window to be kept. |
| `sample_rate` | `None` | Samples per second. Puts windows, steps and lags in seconds. |

**`'spike'`** reads a list with one array of spike times (in seconds) per
neuron.

| Parameter | Default | Meaning |
|--|--|----|
| `window_size` | required | Window length in seconds. |
| `step_size` | one window | Distance between window starts. Below 1 it is a fraction of `window_size`. |
| `bin_size` | `None` | Bin each window into counts of this width (seconds). Unset, each window holds the spike times themselves. |
| `normalize_bins` | `True` | With `bin_size`, divide counts by the bin width to give rates. |
| `max_spikes_per_window` | `None` | Cap on the spike-time slots per window; spikes beyond it are dropped. |
| `no_spike_value` | `0.0` | Value of an empty spike-time slot. |
| `n_seconds` | `None` | Recording length in seconds, when it runs past the last spike. |
| `drop_empty_windows` | `True` | Drop windows without spikes. `False` keeps silent windows and so changes the quantity estimated (see [USING.md](USING.md)). |
| `exclude_bursty_neurons` | `False` | Leave out neurons whose peak count exceeds `burst_threshold_multiplier` times the median. |
| `burst_threshold_multiplier` | `5.0` | The threshold for `exclude_bursty_neurons`. |
| `sample_rate` | `None` | `mode='lag'` reads it to put lags in seconds. |

**`'categorical'`** reads integer labels of shape `(n_timepoints, n_channels)`.
Other values are mapped to consecutive integers with a warning.

| Parameter | Default | Meaning |
|--|--|----|
| `window_size` | required | Window length in samples (seconds with `sample_rate` or a clock). |
| `step_size` | one window | Distance between window starts. Below 1 it is a fraction of `window_size`. |
| `encoding` | `'majority_vote'` | `'majority_vote'` (the most frequent label, one-hot), `'probability'` (the label frequencies) or `'full_trajectory'` (every sample one-hot). |
| `min_coverage_fraction` | `0.2` | Fraction of a window that must hold samples for the window to be kept. |
| `sample_rate` | `None` | Samples per second. Puts windows, steps and lags in seconds. |

## `Model`

The networks that embed X and Y and score their pairing.

| Parameter | Default | Meaning |
|--|--|----|
| `embedding_model` | `'mlp'` | The encoder: `'mlp'`, `'cnn'`, `'cnn2d'`, `'gru'`, `'lstm'`, `'lru'`, `'tcn'`, `'transformer'`, `'deepsets'`, `'pretrained_backbone'` or `'dual_branch'`. |
| `embedding_dim` | `64` | Size of each embedding. `mode='dimensionality'` sets it from [`Dimensionality`](#dimensionality) and ignores this value. |
| `hidden_dim` | `64` | Hidden width. A list such as `[256, 1024, 256]` sets each layer and overrides `n_layers` (MLP, CNN, CNN2D and TCN). |
| `n_layers` | `2` | Encoder depth. |
| `embedding_model_y` | X's | Y's encoder, when it differs from X's. |
| `embedding_dim_y` | X's | Y's embedding size. |
| `hidden_dim_y` | X's | Y's hidden width. |
| `n_layers_y` | X's | Y's depth. |
| `critic_type` | `'separable'` | How the embeddings are scored: `'separable'` (dot product), `'concat'` (one network on the joined inputs) or `'hybrid'` (separate embeddings scored by a small network on their concatenation). `mode='dimensionality'` uses `'hybrid'` when unset. |
| `hidden_dim_head` | `None` | Width of the hybrid critic's head; `None` takes `min(64, hidden_dim)`. |
| `n_layers_head` | `None` | Depth of the hybrid critic's head; `None` takes `max(1, n_layers - 1)`. |
| `kernel_size` | `3` | Kernel width of the CNN, CNN2D and TCN encoders. |
| `bidirectional` | `False` | Bidirectional GRU or LSTM. |
| `nhead` | `4` | Attention heads of the transformer encoder. |
| `branch_model` | `'gru'` | Each branch's encoder under `embedding_model='dual_branch'`. |
| `dropout` | `0.0` | Dropout after each hidden layer (MLP) or inside each block (LRU). |
| `norm_layer` | `'auto'` | Normalisation in the MLP encoder: `'layer'`, `'batch'` or `'none'`. `'auto'` is layer normalisation for the hybrid critic in `mode='dimensionality'` and none everywhere else. Layer normalisation divides out each sample's overall scale and loses the information that scale carries. |
| `use_spectral_norm` | `True` | Spectral normalisation of the MLP's hidden layers. |
| `bias` | `True` | Bias terms in the encoder's layers. |
| `shared_encoder` | `False` | One encoder for X and Y. `mode='dimensionality'` without `y_data` uses `True` when unset. |
| `max_n_batches` | `512` | Most samples an encoder embeds at once, to bound memory. A custom decision head or a variational concat critic also scores at most this many pairs at once. |
| `custom_critic` | `None` | A `torch.nn.Module` used as the whole critic; the architecture settings above are then ignored. |
| `custom_embedding_cls` | `None` | An encoder class to use in place of `embedding_model`. |
| `custom_embedding_cls_y` | X's | Y's encoder class. |
| `pytorch_predefined` | `None` | A torchvision model name, such as `'resnet18'`, for `embedding_model='pretrained_backbone'`. |
| `pretrained` | `False` | Load that backbone's ImageNet weights. |
| `use_variational` | `False` | A variational encoder for any `embedding_model`, trained with a KL term (see [THEORY.md](THEORY.md)). Under `critic_type='concat'` it makes each pair's score variational. |
| `beta` | `1024.0` | Weight of the MI term against the KL term in the variational loss. |
| `use_decoder` | `False` | Add a decoder that reconstructs each input from its embedding. |
| `decoder_lambda` | `0.001` | Weight of both reconstruction terms, measured against the MI term. |
| `decoder_lambda_x` | `decoder_lambda` | Weight of X's reconstruction term. |
| `decoder_lambda_y` | `decoder_lambda` | Weight of Y's reconstruction term. |
| `decoder_output_activation_x` | `'linear'` | X's decoder output: `'linear'` or `'sigmoid'` (mean squared error) or `'softmax'` (cross-entropy). |
| `decoder_output_activation_y` | `'linear'` | Y's decoder output. |

## `Training`

The optimisation loop and what is evaluated after it.

| Parameter | Default | Meaning |
|--|--|----|
| `n_epochs` | `50` | Training epochs. |
| `learning_rate` | `0.0005` | Optimiser learning rate. |
| `batch_size` | `128` | Training batch size. It sets how many negatives each training step sees and does not cap the reported estimate. |
| `patience` | `1000` | Epochs without improvement before training stops. The default exceeds `n_epochs` and keeps early stopping off until this is lowered. |
| `min_improvement` | `0.001` | Rise in the smoothed test MI that counts as an improvement for `patience`. |
| `median_window` | `5` | Width of the median filter on the test-MI curve before its peak is read. |
| `smoothing_sigma` | `1.0` | Width of the Gaussian filter applied after the median filter. |
| `peak_fraction` | `1.0` | Below 1, report the first epoch whose smoothed test MI reaches this fraction of the peak. The chosen epoch is `conservative_epoch` in `runs`. |
| `optimizer` | `'adam'` | `'adam'`, `'adamw'`, `'sgd'`, `'rmsprop'`, `'adagrad'`, or a `torch.optim.Optimizer` class. |
| `optimizer_params` | `{}` | Extra arguments for the optimiser, such as `{'weight_decay': 1e-4}`. |
| `scheduler` | `None` | `'cosine'`, `'cosine_warmup'`, `'step'`, `'plateau'`, or a `torch.optim.lr_scheduler` class. |
| `scheduler_params` | `{}` | Extra arguments for the scheduler. |
| `lr_head_multiplier` | `None` | Learning-rate multiplier for the hybrid critic's head. |
| `gradient_clip_val` | `None` | Clip the gradient norm to this value. |
| `use_amp` | `'auto'` | Mixed precision: `'auto'` uses it on CUDA only. |
| `max_eval_samples` | `5000` | Most samples any single evaluation uses, test-side or train-side. |
| `train_subset_size` | `None` | Size of the fixed training subset the reported estimate is evaluated on; `None` takes `min(n_train, max_eval_samples)`. Training uses every training sample either way. |
| `eval_train` | `False` | Also record the train-side MI every epoch: `True`, a fraction, a sample count, or `'full'`. |
| `shift_time` | `True` | Temporal data: re-tile the windows from a fresh random time offset every epoch. |
| `shift_windows` | `True` | The cheaper re-slicing form of `shift_time`, for regularly sampled continuous and categorical data. |
| `augmentation_params` | `{}` | Augmentations applied to both X and Y during training (listed in [USING.md](USING.md)). |
| `augmentation_params_x` | `augmentation_params` | X's augmentations; `{}` turns them off for X. |
| `augmentation_params_y` | `augmentation_params` | Y's augmentations. |
| `min_reliable_samples` | `None` | The chunk size below which a rigorous ladder warns; `None` derives it from `batch_size` and the train fraction. |
| `save_best_model_path` | `None` | Save the best epoch of every network the call trains here: a file name or a directory for generated names. A call that trains several networks adds the labels that identify each one to its name ([USING.md](USING.md#saving-the-trained-networks)). |
| `dataset_device` | `'cpu'` | Where the dataset tensors live. `'auto'` puts them on the compute device. |

## `Split`

How samples are divided into training and test sets.

| Parameter | Default | Meaning |
|--|--|----|
| `mode` | `'blocked'` | `'blocked'` holds out contiguous stretches, for time series; `'random'` holds out random samples, for independent ones. |
| `train_fraction` | `0.9` | Fraction of the samples used for training. |
| `n_test_blocks` | `5` | Number of contiguous test stretches under `'blocked'`. |
| `gap_fraction` | `0.5` | Gap left between training and test stretches as a fraction of a test block. |
| `train_indices`, `test_indices` | `None` | An explicit split given as both lists. It overrides the settings above. The indices address the rows the network trains on and are refused where those rows are not the ones passed: `mode='rigorous'`, `'lag'` and `'transfer'`, `rigorous=True`, a dimensionality `lag`, data windowed through `Processing`, and the named quantities that build their own rows. |

## `Estimator`

The bound the critic is trained on. A bare name, `estimator='smile'`, is accepted
too.

| Parameter | Default | Meaning |
|--|--|----|
| `name` | `'infonce'` | `'infonce'` (low variance, capped at the log of the evaluation sample count) or `'smile'` (lower bias, higher variance). |
| `params` | `{}` | Extra arguments for the bound. SMILE takes `clip`, default 5.0. |

## `Output`

Units, extra diagnostics, embeddings and display names.

| Parameter | Default | Meaning |
|--|--|----|
| `units` | `'bits'` | `'bits'` or `'nats'`, for every MI value the result holds. |
| `track_spectral_history` | `False` | Record the participation ratios and spectrum every epoch in `runs` (`spectral_metrics_history`). |
| `whitening` | `'std'` | Normalisation of the embeddings before the cross-covariance SVD behind the participation ratios and the rotated embeddings: `'std'`, `'zca'` or `None`. |
| `return_embeddings` | `False` | Keep each repeat's embeddings of every window, in `details[config_id]['embeddings']`. Available in the modes where a repeat is one network. |
| `track_embeddings` | `False` | Keep embeddings every epoch for `result.animate()`: `True` (512 samples), a count, a fraction or `'full'`. |
| `return_rotated_embeddings` | `False` | Also keep the embeddings rotated so that dimension 0 carries the most shared variance. |
| `rotated_embeddings_per_epoch` | `False` | With tracked embeddings, rotate each epoch on its own, not with the best epoch's rotation. |
| `return_rotation_matrices` | `False` | Keep the rotation matrices, to project new data into the same basis. |
| `max_index_reduction` | `0.05` | Largest fraction of windows time shifting may remove before a warning. |
| `channel_names_x`, `channel_names_y` | `None` | Channel names, for the pairwise heatmap. |

## Mode configs

### `Rigorous`

For `mode='rigorous'`. The same fit settings are fields of `Conditional`,
`Interaction` and `Transfer` for `rigorous=True`.

| Parameter | Default | Meaning |
|--|--|----|
| `gamma_range` | `range(1, 11)` | The subdivisions of the data: at $\gamma$ the data are cut into $\gamma$ chunks and each chunk is estimated on its own. |
| `curvature_t_threshold` | `2.0` | The t-statistic of the quadratic term below which the ladder counts as linear. |
| `min_gamma_points` | `5` | Fewest $\gamma$ values a reliable fit may use. |
| `confidence_level` | `0.68` | Confidence level of `mi_error`. |
| `residual_threshold` | `2.5` | Largest studentised residual before `fit_quality_warning` is set. |
| `leverage_threshold` | `0.2` | Largest relative shift of the intercept when the $\gamma = 1$ points are left out before `leverage_warning` is set. |
| `temporal_chunking` | `None` | Cut chunks as contiguous stretches (`True`) or random subsets (`False`); `None` decides from the data. |

### `Precision`

For `mode='precision'`.

| Parameter | Default | Meaning |
|--|--|----|
| `tau_grid` | required | The corruption levels $\tau$ to evaluate. |
| `corrupt_target` | `'x'` | Which side is corrupted: `'x'`, `'y'` or `'both'`. |
| `corruption_method` | `'rounding'` | `'rounding'` (move each value to the centre of its bin of width $\tau$) or `'noise'` (add uniform jitter on $[-\tau/2, \tau/2]$). Either way, entries that hold no measurement, such as unused spike-time slots, stay as they are. |
| `n_noise_samples` | `50` | Noise draws per $\tau$ under `'noise'`. |
| `threshold_ratio` | `0.9` | The precision is the smallest $\tau$ at which MI falls below this fraction of the baseline. A list gives one threshold per value. |

### `Lag`

For `mode='lag'`.

| Parameter | Default | Meaning |
|--|--|----|
| `lag_range` | required | The lags to test in samples (seconds with `sample_rate`). A positive lag compares X with Y's future. |
| `equalize_n` | `False` | Cut every lag to the sample count of the largest one so that all lags use the same data. |

### `Transfer`

For `mode='transfer'`.

| Parameter | Default | Meaning |
|--|--|----|
| `history_window` | required | Length of the X and Y histories, in rows. |
| `prediction_horizon` | `1` | How many rows ahead Y's future is. |
| `stride` | `1` | Rows between consecutive history windows. |
| `bidirectional` | `False` | Also estimate transfer entropy from Y to X and the directionality index. |
| `w_data` | `None` | A third process whose history is conditioned on, for conditional transfer entropy. |
| `rigorous` | `False` | Extrapolate each repeat to infinite data. |
| `gamma_range`, `curvature_t_threshold`, `min_gamma_points`, `confidence_level`, `residual_threshold`, `leverage_threshold` | as in `Rigorous` | The fit of `rigorous=True` with the meanings and defaults of [`Rigorous`](#rigorous). |

### `Conditional`

For `mode='conditional'`.

| Parameter | Default | Meaning |
|--|--|----|
| `w_data` | required | The conditioning variable $W$ in $I(X;Y \mid W)$. |
| `align` | `None` | `'dual_branch'` embeds W apart from X, for a W whose window length differs from X's. |
| `rigorous` | `False` | Extrapolate each repeat to infinite data. |
| `gamma_range`, `curvature_t_threshold`, `min_gamma_points`, `confidence_level`, `residual_threshold`, `leverage_threshold` | as in `Rigorous` | The fit of `rigorous=True` with the meanings and defaults of [`Rigorous`](#rigorous). |

### `Interaction`

For `mode='interaction'`.

| Parameter | Default | Meaning |
|--|--|----|
| `w_data` | required | The third population $W$. |
| `rigorous` | `False` | Extrapolate each repeat to infinite data. |
| `gamma_range`, `curvature_t_threshold`, `min_gamma_points`, `confidence_level`, `residual_threshold`, `leverage_threshold` | as in `Rigorous` | The fit of `rigorous=True` with the meanings and defaults of [`Rigorous`](#rigorous). |

### `Pairwise`

For `mode='pairwise'`.

| Parameter | Default | Meaning |
|--|--|----|
| `pairs` | every pair | The channel pairs `(i, j)` to estimate. |

### `Dimensionality`

For `mode='dimensionality'`.

| Parameter | Default | Meaning |
|--|--|----|
| `embedding_dims` | `None` | A list or a range of the values of `embedding_dim` to fit. `None` chooses them from the reference fit and stops once three values in a row reach the threshold. Given values are all fitted. |
| `n_restarts` | `4` | Networks trained at each `embedding_dim` of each split. The best one counts. |
| `saturation_ratio` | `0.95` | The fraction of the plateau the curve must reach. The reading is the smallest `embedding_dim` that reaches it. |
| `reference_dim` | `None` | The `embedding_dim` of the large reference fit that runs first. `None` fits 64 and refits at four times the participation ratio when that ratio reaches 24. |
| `split_method` | `'random'` | Without `y_data`, how X's channels are split in two: `'random'`, `'spatial'` (at the midpoint), `'index'` (by `channel_indices_x`), `'temporal'` (X against X at `lag`), or, for image data, `'horizontal'`, `'vertical'`, `'row_interleaved'`, `'col_interleaved'`, `'diagonal'`, `'antidiagonal'`. |
| `n_splits` | `None` | The number of random channel splits of X (5 when `None`). Only `split_method='random'` draws more than one. A call with `y_data` refuses it. |
| `lag` | `1` | The lag of `split_method='temporal'`, in samples. |
| `channel_indices_x` | `None` | X's channels under `split_method='index'`; Y is the rest. |
| `ceiling_mi_fraction` | `0.85` | Warn when the reference fits' held-out MI is this close to its ceiling. |
