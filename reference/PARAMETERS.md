# Parameters

This page lists every setting that `nmi.run()` accepts, with its default and
what it does. The settings are grouped into config objects. A plain `dict` with
the same keys works wherever a config object is accepted. A default of "X's"
means that the setting falls back to the value given for X.

[USING.md](USING.md) shows how the settings combine for each task.
[THEORY.md](THEORY.md) explains the settings whose meaning is statistical.

## Contents

- [`run()`](#run)
- [`Processing`](#processing) and the [processor parameters](#processor-parameters)
- [`Preprocessing`](#preprocessing)
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
| `sweep_grid` | `None` | A dict that maps setting names to the values to run. A list, tuple, range or array gives one configuration per value, while a single value fixes the setting. Every combination of values is one configuration. `run_id` in the grid repeats each configuration. The keys are the settings of `Model`, `Training`, `Split` and `Estimator` and the processor parameters. In a grid, `Split(mode=...)`, `Split(gap_fraction=...)`, `Estimator(name=...)` and `Estimator(params=...)` are named `split_mode`, `split_gap_fraction`, `estimator_name` and `estimator_params`. |
| `n_workers` | `1` | Worker processes. Tasks run in parallel across repeats, configurations, chunks, pairs or permutation trials. Each worker gets an equal share of the CPU threads. Results do not depend on the count. |
| `seed` | `None` | Seed for Python, NumPy and PyTorch. Each network seeds itself from it, from its repeat (`run_id`) and from its role in the estimate, but not from its place in a grid. Every configuration runs a given repeat from the same seeds. A seeded call gives the same result at any `n_workers`. Without a seed, one is drawn from NumPy's global state and recorded in `Results.params['random_seed']`. |
| `verbose` | `None` | `True` logs informational messages for the call and `False` only warnings and errors. `None` keeps the level set by `nmi.set_verbosity()`. |
| `show_progress` | `True` | Show progress bars. |
| `device` | `None` | Compute device, `'cpu'`, `'cuda'` or `'mps'`. `None` trains on the CPU unless a GPU, timed on the call's first network, is clearly faster ([USING.md](USING.md#speed-memory-and-logging)). |
| `permutation_test` | `False` | Test every row of the result against a null built by rerunning the call with X moved in time. Adds a `p_value` column. |
| `n_permutations` | `10` | Number of null trials. The smallest p-value reachable is `1 / (n_permutations + 1)`. |
| `permutation_shuffle` | `'circular'` | How X is moved. `'circular'` shifts it by one random offset with wrap-around and excludes offsets that lie within `permutation_min_shift` of the length of the recording from zero. `'block'` reorders contiguous blocks that are one window long. |
| `permutation_min_shift` | `0.1` | The share of the recording's length within which a circular shift counts as too close to zero, in either direction, to break the dependence. It must be at least 0 and below 0.5. |
| `min_permutations` | `100` | The call warns when a permutation test runs fewer trials than this, since the smallest p-value that the test can report is `1 / (n_permutations + 1)`. |

The mode configs are passed under the mode's own name: `run(..., mode='lag',
lag=Lag(lag_range=range(-10, 11)))`. A config for another mode is ignored with a
warning.

## `Processing`

`Processing` sets how each raw stream is processed: its processor, the
parameters of the processor and its clock. With no `Processing`, the data are
taken as already windowed, with shape `(n_samples, n_channels)` or
`(n_samples, n_channels, window_size)`.

| Parameter | Default | Meaning |
|--|--|----|
| `x` | `None` | X's processor: `'continuous'`, `'spike'` or `'categorical'`. |
| `x_params` | `None` | X's processor parameters, listed below. |
| `y` | X's | Y's processor. |
| `y_params` | X's | Y's processor parameters. When Y falls back to X's, only the keys Y's processor takes are kept. |
| `w` | X's | The third stream's processor, for `mode='conditional'`, `'interaction'`, `'transfer'` and the named quantities that take one. |
| `w_params` | X's | The third stream's processor parameters. When W falls back to X's, only the keys W's processor takes are kept. |
| `x_time`, `y_time`, `w_time` | `None` | A timestamp per sample. With a clock `window_size` and `step_size` are in the clock's units. Without one they are in samples (seconds with `sample_rate`). |

Every stream of a call is windowed on one grid. On this grid a window covers the
same interval in X, Y and W and is kept only where every stream is valid. Every
term of a quantity uses the rows of that grid ([USING.md](USING.md#one-grid-for-every-stream)).

### Processor parameters

`window_size` must be set for at least one stream. The window and the step of
the grid are those of the first stream that names a `window_size`. A stream that
asks for another window or for a step of another length is refused.
`nmi.offset_quantity` places a stream at its own offsets.

**`'continuous'`** reads a regularly sampled signal of shape `(n_timepoints,
n_channels)`.

| Parameter | Default | Meaning |
|--|--|----|
| `window_size` | required | Window length in samples (seconds with `sample_rate` or a clock). |
| `step_size` | one window | Distance between window starts. Below 1 it is a fraction of `window_size`. |
| `min_coverage_fraction` | `0.2` | Fraction of a window that must contain samples for the window to be kept. A window never runs past the first or the last sample of the stream, whatever this setting allows. |
| `sample_rate` | `None` | `mode='lag'` uses it to put lags in seconds. |
| `period_tolerance` | `0.05` | The call warns when the first interval of the time vector differs from the median interval by more than this fraction of the median interval. The median sets the period either way. |
| `padded_share` | `0.1` | The call warns when more than this share of the time points at which the stream is interpolated for its windows fall outside the recorded range. Those points are filled with zeros. |
| `gap_multiple` | `3.0` | An interval between samples longer than this many median periods counts as a gap. |
| `bridged_share` | `0.3` | The call warns when a window that it keeps for the estimate has more than this share of its samples interpolated across a gap. |

**`'spike'`** reads a list with one array of spike times (in seconds) per
neuron.

| Parameter | Default | Meaning |
|--|--|----|
| `window_size` | required | Window length in seconds. |
| `step_size` | one window | Distance between window starts. Below 1 it is a fraction of `window_size`. |
| `bin_size` | `None` | Bin each window into counts of this width (in seconds). When unset, each window contains the spike times themselves. |
| `normalize_bins` | `False` | With `bin_size`, divide the counts by the bin width to give rates. This is off by default, so a count enters as a count. As a rate, one spike becomes 100 in 10 ms bins and 1,000 in 1 ms bins. The networks handle such scales poorly unless the encoders scale their input, as `Model(input_scaling='auto')` does by default. |
| `max_spikes_per_window` | `None` | Maximum number of spike-time slots per window. Spikes beyond it are dropped. |
| `no_spike_value` | `0.0` | Value of an empty spike-time slot. |
| `n_seconds` | `None` | Recording length in seconds, when it runs past the last spike. |
| `drop_empty_windows` | `True` | Drop windows without spikes. This is the one default that changes the data, because in spike times silence looks the same as a stretch without recording. A row is kept only where every spike stream fired, in every term of a quantity. `False` keeps silent windows and so changes the quantity estimated (see [USING.md](USING.md#silent-spike-windows)). The setting has no effect on the rows that the quantities defined by time offsets build, because these quantities keep every window. |
| `exclude_bursty_neurons` | `False` | Leave out neurons whose peak count exceeds `burst_threshold_multiplier` times the median. |
| `burst_threshold_multiplier` | `5.0` | The threshold for `exclude_bursty_neurons`. |
| `sample_rate` | `None` | `mode='lag'` uses it to put lags in seconds. |

**`'categorical'`** reads integer labels of shape `(n_timepoints, n_channels)`.
Other values are mapped to consecutive integers with a warning.

| Parameter | Default | Meaning |
|--|--|----|
| `window_size` | required | Window length in samples (seconds with `sample_rate` or a clock). |
| `step_size` | one window | Distance between window starts. Below 1 it is a fraction of `window_size`. |
| `encoding` | `'majority_vote'` | `'majority_vote'` (the most frequent label, one-hot), `'probability'` (the label frequencies) or `'full_trajectory'` (every sample one-hot). |
| `min_coverage_fraction` | `0.2` | Fraction of a window that must contain samples for the window to be kept. A window never runs past the first or the last sample of the stream, whatever this setting allows. |
| `sample_rate` | `None` | `mode='lag'` uses it to put lags in seconds. |
| `max_encoding_width` | `500` | With `encoding='full_trajectory'`, the call warns when a window's input would be wider than this. The width is the number of categories times the samples per window. |

## `Preprocessing`

`Preprocessing` transforms each continuous stream before rows are built from
it. It does so only when asked and records what it did in
`result.params['preprocessing']`. Each transform is fitted on X, Y and W
separately, with the time transform first. Spike-time and categorical streams
take neither transform, while arrays passed in as rows take only the channel
transform. The transforms are in `nmi.preprocessing`.
[USING.md](USING.md#preprocessing-the-streams) shows them in use. The settings
after `time` belong to the data check. Every call runs this check on its streams
after any transform, whether or not it asks for one.

| Parameter | Default | Meaning |
|--|--|----|
| `channel` | `None` | A transform across the channels of each stream: `'zscore'` (centre each channel and divide it by its standard deviation), `'zca'` (sphere the channels), `'pca'` (rotate them onto their principal axes) or `'copula'` (map each channel through its ranks to a standard normal). Each leaves every quantity unchanged. `nmi.preprocessing.PCA(n_components=k)` keeps the first `k` components and warns. A transform of your own also goes here. |
| `time` | `None` | A transform along the time axis of each channel. `'ar'` filters each channel with its own autoregressive fit. AIC chooses the order up to `10 log10(T)`. `nmi.preprocessing.ARWhitening(order=p)` fixes the order instead. The filter leaves the rate quantities unchanged in the limit of long windows. It changes every other quantity, and the call then warns. |
| `scale_band` | `10.0` | The data check warns when the typical channel scale of a stream lies more than this factor from unit scale, in either direction. The same applies to the rows that a spike-time or categorical processor builds. |
| `predicted_share` | `0.9` | For a rate quantity on a series, the data check notes a channel whose own past predicts more than this share of its variance and suggests `time='ar'`. |
| `check_rows` | `100000` | The data check processes each stream in stretches of this many rows, so that it never copies a large stream. It fits the dependence of a channel on its own past on the first stretch. |
| `sample_rows` | `10000` | The data check measures the scale of the rows that a spike-time or categorical processor builds, and the quartiles and repeated values of each channel of a continuous stream, from this many rows at their start. |
| `spread_band` | `10.0` | The data check warns when the standard deviations of a stream's channels span more than this factor. |
| `offset_band` | `1e5` | The data check warns when a channel's mean lies more than this many of its standard deviations from zero, where single precision resolves its variation coarsely. |
| `tail_band` | `1.5` | The data check warns that a channel has heavy tails when its standard deviation exceeds this many interquartile ranges. |
| `tie_share` | `0.9` | When an encoder scales a stream, the data check warns about a channel with more than this share of its values equal to one value, as in padded or sparse rows. |

## `Model`

`Model` sets the networks that embed X and Y and score their pairings.

| Parameter | Default | Meaning |
|--|--|----|
| `embedding_model` | `'mlp'` | The encoder: `'mlp'`, `'cnn'`, `'cnn2d'`, `'gru'`, `'lstm'`, `'lru'`, `'tcn'`, `'transformer'`, `'deepsets'` or `'pretrained_backbone'`. In a difference quantity it encodes each part of X on a branch of its own ([USING.md](USING.md#the-encoder-of-a-difference)). |
| `embedding_dim` | `64` | Size of each embedding. `mode='dimensionality'` sets it from [`Dimensionality`](#dimensionality) and ignores this value. |
| `hidden_dim` | `64` | Hidden width. A list such as `[256, 1024, 256]` sets each layer and overrides `n_layers` (MLP, CNN, CNN2D and TCN). |
| `n_layers` | `2` | Encoder depth. For the MLP, 0 gives a linear encoder that maps the input straight to the embedding. The other encoders need at least one layer. |
| `embedding_model_y` | X's | Y's encoder, when it differs from X's. |
| `embedding_dim_y` | X's | Y's embedding size. |
| `hidden_dim_y` | X's | Y's hidden width. |
| `n_layers_y` | X's | Y's depth. |
| `critic_type` | `'separable'` | How the embeddings are scored: `'separable'` (dot product), `'concat'` (one network on the joined inputs) or `'hybrid'` (separate embeddings scored by a small network on their concatenation). `mode='dimensionality'` uses `'hybrid'` when unset. |
| `hidden_dim_head` | `None` | Width of the head of the hybrid critic. `None` takes `min(64, hidden_dim)`. |
| `n_layers_head` | `None` | Depth of the head of the hybrid critic. `None` takes `max(1, n_layers - 1)`. |
| `kernel_size` | `3` | Kernel width of the CNN, CNN2D and TCN encoders. |
| `cnn_pooling` | `'flatten'` | How the CNN encoder combines its convolutions. `'flatten'` keeps every position of the window, so it keeps where a pattern occurs, as a lag or a history window needs. `'mean'` averages over the window and ignores where a pattern occurs. |
| `bidirectional` | `False` | Bidirectional GRU or LSTM. |
| `nhead` | `4` | Attention heads of the transformer encoder. |
| `dropout` | `0.0` | Dropout after each hidden layer of an MLP, inside each block of the LRU, TCN and transformer encoders, and between the stacked layers of a GRU or LSTM. It also applies to the MLPs that the library builds around the encoders: the head of the hybrid critic, the network of the concat critic, the head of the branched encoder of a difference term and the head of the pretrained backbone. A setting that no network of the call uses warns. |
| `norm_layer` | `'auto'` | Normalisation in the encoders: `'layer'`, `'batch'` or `'none'`. The MLP and convolutional encoders (`'mlp'`, `'cnn'`, `'cnn2d'`, `'tcn'`) normalise each hidden layer before its activation, and `'deepsets'` the hidden layer of its rho. The recurrent and attention encoders (`'gru'`, `'lstm'`, `'lru'`, `'transformer'`) normalise the features that their output layer maps to the embedding. It also applies to the MLPs that the library builds around the encoders (the head of the hybrid critic, the network of the concat critic, the head of the branched encoder of a difference term and the head of the pretrained backbone). `'auto'` is layer normalisation for the hybrid critic in `mode='dimensionality'` and no normalisation everywhere else. Layer normalisation divides out the overall scale of each sample, so it loses the information that this scale carries. |
| `large_first_layer` | `500000` | A first embedding layer with more parameters than this is noted at the info level, since it may overfit a small dataset. |
| `input_scaling` | `'auto'` | Removes the units and the offset of the input of an encoder with statistics of the training rows, so that an estimate does not depend on the units of the data. `'auto'` scales each stream according to its kind. It centres and scales each channel of a continuous stream or of an array passed in as rows. For spike counts, it centres each neuron and divides the stream by one spread. It divides spike times by one root mean square without centring, so that the padding stays in place. It leaves categorical streams as they are. An encoder without biases is not centred. X, Y and W each have their own statistics. `'none'` scales nothing. The statistics are part of the network. They are saved with it and applied by `extract_embeddings`. Under `critic_type='concat'`, each side is scaled before the pair is formed. |
| `zero_start` | `True` | Every score of a critic that the library builds starts at exactly zero. The separable critic starts half of the readout rows of each side at zero: the first half for X and the second half for Y. The hybrid and concat critics start the last layer of the network that scores the pair at zero. A shared encoder keeps its start, because its one readout could start at zero only as a whole and would then never move. A custom critic starts as its recipe builds it. |
| `linear_path` | `True` | An MLP encoder adds a linear map from its flattened input to its embedding. The map has spectral normalisation and is multiplied by one trainable number that starts at zero. The path therefore starts closed and opens as training finds a use for it. It helps when the information is spread over many weak directions and the rows are few. Only MLP encoders with hidden layers have it. The MLP heads of the hybrid and concat critics do not. |
| `activation` | `'leaky_relu'` | The nonlinearity of every rectified hidden layer that the library builds. These are the hidden layers of the MLP, CNN, CNN2D, TCN and DeepSets encoders, the feed-forward blocks of the transformer, the MLPs that the library builds around the encoders, and the decoders. A ReLU gives zero output and zero gradient for a negative input, so a unit whose input is negative on every row stops learning. A leaky ReLU keeps a small slope for a negative input, so such a unit can still learn. The other options are `'relu'`, `'gelu'`, `'elu'`, `'silu'`, `'tanh'` and `'sigmoid'`. The GRU and LSTM cells and the LRU block keep their own nonlinearities. |
| `use_spectral_norm` | `True` | Spectral normalisation of the hidden layers of an MLP: the MLP encoder and the MLPs the library builds around the encoders (the hybrid critic's head, the concat critic's network, the head of a difference term's branched encoder, the pretrained backbone's head). |
| `bias` | `True` | Bias terms in the layers of every encoder and of the MLPs built around them. |
| `shared_encoder` | `False` | One encoder for X and Y. `mode='dimensionality'` without `y_data` uses `True` when unset. |
| `max_n_batches` | `512` | Most samples an encoder embeds at once, to bound memory. A custom decision head or a variational concat critic also scores at most this many pairs at once. |
| `custom_critic` | `None` | A class or function that builds the whole critic. Every task calls it under its own seed, so each repeat and each term of a difference trains its own critic. The call passes the parameters that it names among `input_dim_x`, `input_dim_y`, `n_channels_x`, `n_channels_y`, `window_size_x`, `window_size_y`, `embedding_dim`, `hidden_dim`, `n_layers` and the network settings above. In the joint term of a difference, X comes as the pair (rest of X, conditioning set). Each size of X is then a pair. An object is refused. Settings that the critic does not name are ignored with a warning. |
| `custom_embedding_cls` | `None` | An encoder class to use in place of `embedding_model`, as each branch in a difference quantity. |
| `custom_embedding_cls_y` | X's | Y's encoder class. |
| `pytorch_predefined` | `None` | A torchvision model name, such as `'resnet18'`, for `embedding_model='pretrained_backbone'`. `None` means `'resnet18'`. |
| `pretrained` | `False` | Load that backbone's ImageNet weights. |
| `use_variational` | `False` | A variational encoder for any `embedding_model`, trained with a KL term (see [THEORY.md](THEORY.md)). Under `critic_type='concat'` it makes each pair's score variational. |
| `beta` | `1024.0` | Weight of the MI term against the KL term in the variational loss. |
| `use_decoder` | `False` | Add a decoder that reconstructs each input from its embedding. Needs a critic that gives an embedding of each side (not `critic_type='concat'`). |
| `decoder_lambda` | `0.001` | Weight of both reconstruction terms, measured against the MI term. |
| `decoder_lambda_x` | `decoder_lambda` | Weight of X's reconstruction term. |
| `decoder_lambda_y` | `decoder_lambda` | Weight of Y's reconstruction term. |
| `decoder_output_activation_x` | `'linear'` | X's decoder output: `'linear'` or `'sigmoid'` (mean squared error) or `'softmax'` (cross-entropy). |
| `decoder_output_activation_y` | `'linear'` | Y's decoder output. |

## `Training`

`Training` sets the optimisation loop and what is evaluated after it.

| Parameter | Default | Meaning |
|--|--|----|
| `n_epochs` | `300` | The most epochs training runs. Training usually ends earlier, once the smoothed held-out curve has not improved for `patience` epochs. |
| `learning_rate` | `0.0005` | Optimiser learning rate. |
| `batch_size` | `128` | Training batch size. It sets how many negatives each training step sees and does not cap the reported estimate. |
| `patience` | `30` | Epochs without improvement of the smoothed held-out curve before training stops. With `shift_windows`, the curve keeps rising for longer, and training runs as long as the curve rises. Without shifting, the curve often reaches its top early and falls, so training ends soon after. A value of `n_epochs` or more turns stopping off. Early stopping only determines when training ends, because the network evaluated is always the one at `best_epoch`. |
| `min_improvement` | `0.001` | The rise in the smoothed held-out MI, as a fraction of its best value so far, that counts as an improvement for `patience`. |
| `median_window` | `5` | Width of the median filter that smooths the held-out curve before the epoch is chosen from it. |
| `smoothing_sigma` | `1.0` | Width of the Gaussian filter applied after the median filter. |
| `epoch_rule` | `'plateau_start'` | Which epoch of the smoothed held-out curve gives the reported values (`best_epoch`). `'plateau_start'` picks the earliest epoch at the top of the curve, within `start_noise` noise units of the level of the top. `'peak'` picks the maximum. On a clean peak the two agree. On a plateau, `'plateau_start'` picks its start, before the network fits its training rows further. `runs` gives the maximum as `peak_epoch` and a diagnosis of the curve as `curve_diagnosis`. |
| `peak_fraction` | `1.0` | Below 1, report the first epoch whose smoothed held-out MI reaches this fraction of its maximum, when that epoch is earlier than the one that `epoch_rule` picks. This is meant for a curve that saturates. The top of such a curve may be the ceiling of the estimator rather than the information, so the value is taken before the curve reaches its top. The chosen epoch is `conservative_epoch` in `runs`. |
| `level_noise` | `2.0` | The level of the top of the curve is the median of its smoothed values within this many noise units of the maximum. A noise unit is the epoch-to-epoch noise of the curve itself. |
| `start_noise` | `0.5` | `epoch_rule='plateau_start'` picks the first epoch within this many noise units of the level of the top. |
| `resolved_noise` | `8.0` | The curve is diagnosed `'not resolved'` unless its top lies this many noise units above zero. |
| `clear_se` | `3.0` | A held-out value counts as above zero when it exceeds this many of its standard errors. A curve whose top does not lie above zero in this sense is diagnosed `'not resolved'`. When terms are combined, a term whose held-out value does not lie above zero counts as having learned nothing. |
| `top_se` | `1.0` | The *band* is the set of epochs whose smoothed held-out value lies within this many standard errors of the maximum of the curve. The standard error is that of the maximum on the held-out rows ($\sqrt{2I/N}$ for $I$ nats on $N$ rows), so the held-out rows cannot tell the epochs of the band apart. `runs` gives the first and the last epoch of the band (`band_first_epoch`, `band_last_epoch`) and the lower and the higher of the estimates at these two epochs (`band_low`, `band_high`). This setting also determines which configurations of a sweep `result.best` lists after the best one. |
| `band_share` | `0.3` | The curve is diagnosed `'flat top'` when the estimates at the first and the last epoch of the band differ by more than this share of the estimate. |
| `fast_epochs` | `8` | The curve is diagnosed `'too fast'` when its top lies within the reach of the smoothing from its start and the curve falls by more than `fall_noise` noise units within this many epochs after the top. |
| `fall_noise` | `3.0` | The fall below the level of the top, in noise units, that the `fast_epochs` check requires. |
| `rising_epochs` | `20` | When training runs to `n_epochs`, the curve is diagnosed `'still rising'` if it rose by more than `rising_noise` noise units over its last `rising_epochs` epochs. |
| `rising_noise` | `3.0` | The rise, in noise units, that `rising_epochs` looks for. |
| `ceiling_fraction` | `0.85` | The call warns when a held-out or a training value exceeds this fraction of its InfoNCE ceiling (the logarithm of the number of rows per evaluation). Rigorous mode also warns about rungs above it. |
| `saturation_fraction` | `0.9` | The call warns that epoch selection may be unreliable when the smoothed held-out curve lies above this fraction of its ceiling for more than `saturated_epoch_fraction` of its epochs. |
| `gap_share` | `0.5` | The call warns when the held-out value of the network at the chosen epoch is below this share of a positive estimate and the difference exceeds `clear_se` of its standard errors. The check is skipped for a curve diagnosed `'not resolved'`, because the message for that curve already states both values. |
| `saturated_epoch_fraction` | `0.5` | The share of epochs above `saturation_fraction` of the ceiling at which the call warns. |
| `amplification_threshold` | `10.0` | The call warns when a quantity combined from several terms has an error-amplification factor of this or more. |
| `small_sample` | `200` | The call warns that the dataset is small when fewer rows than this reach the estimator. |
| `nan_streak` | `3` | Training stops with `TrainingError` after this many epochs in a row whose held-out evaluation is NaN. The call warns at each such epoch and skips it for early stopping. |
| `optimizer` | `'adam'` | `'adam'`, `'adamw'`, `'sgd'`, `'rmsprop'`, `'adagrad'`, or a `torch.optim.Optimizer` class. |
| `optimizer_params` | `{}` | Extra arguments for the optimiser, such as `{'weight_decay': 1e-4}`. |
| `scheduler` | `None` | `'cosine'`, `'cosine_warmup'`, `'step'`, `'plateau'`, or a `torch.optim.lr_scheduler` class. The named schedules are sized on `n_epochs` (the cap), so a run that `patience` stops ends before they do. `'plateau'` follows the held-out curve instead. |
| `scheduler_params` | `{}` | Extra arguments for the scheduler. |
| `lr_head_multiplier` | `None` | Learning-rate multiplier for the hybrid critic's head. |
| `gradient_clip_val` | `None` | Clip the gradient norm to this value. |
| `use_amp` | `'auto'` | Mixed precision: `'auto'` uses it on CUDA only. |
| `max_eval_samples` | `5000` | Most samples any single evaluation uses, test-side or train-side. A larger set is evaluated in blocks of at most this size whose values are averaged, so every row counts and the ceiling is that of one block. |
| `train_subset_size` | `None` | Size of the fixed training subset on which the reported estimate is evaluated: a count, or `'full'` for the whole training set. `None` takes `min(n_train, max_eval_samples)`. Like the held-out subset, it is one contiguous stretch per training segment for a blocked split and random rows for a random split. A subset larger than `max_eval_samples` is evaluated in blocks. Training uses every training sample either way. |
| `test_subset_size` | `None` | Size of the fixed held-out subset that is evaluated at every epoch: a count, or `'full'` for the whole held-out set. `None` takes `min(n_test, max_eval_samples)`. The subset is drawn once, right after the split, as one contiguous stretch per held-out block for a blocked split and as random rows for a random split. A subset larger than `max_eval_samples` is evaluated in blocks at every epoch. |
| `eval_train` | `False` | Also record the train-side MI every epoch: `True`, a fraction, a sample count, or `'full'`. A set larger than `max_eval_samples` is evaluated in blocks. |
| `shift_windows` | `True` | At every epoch the windows of the training rows start one offset later. The offset is drawn within one step and is the same for every stream. The rows stay the same rows, the grid leaves one step at its end for the shift, and evaluation uses the unshifted rows. It moves the rows that a processor cuts from raw streams in every mode except transfer entropy. It also moves the rows that a `stride` above one thins out in the quantities built by time offsets. Rows cut by sample number that start one sample apart never move. On a grid in time they move by a fraction of a sample. |
| `augmentation_params` | `{}` | Augmentations applied to both X and Y during training (listed in [USING.md](USING.md)). |
| `augmentation_params_x` | `augmentation_params` | The augmentations of X. `{}` turns them off for X. |
| `augmentation_params_y` | `augmentation_params` | Y's augmentations. |
| `min_reliable_samples` | `None` | The chunk size below which a rigorous ladder warns. `None` derives it from `batch_size` and the training fraction. |
| `save_best_model_path` | `None` | Save the best epoch of every network the call trains here: a file name or a directory for generated names. A call that trains several networks adds the labels that identify each one to its name ([USING.md](USING.md#saving-the-trained-networks)). |
| `dataset_device` | `'cpu'` | Where the dataset tensors live. `'auto'` puts them on the compute device. |

## `Split`

`Split` sets how the samples are divided into training and held-out sets.

| Parameter | Default | Meaning |
|--|--|----|
| `mode` | `'blocked'` | `'blocked'` holds out contiguous stretches, for time series. `'random'` holds out random samples, for independent samples. |
| `train_fraction` | `0.9` | Fraction of the samples used for training. |
| `n_test_blocks` | `5` | Number of contiguous test stretches under `'blocked'`. |
| `gap_fraction` | `0.5` | Gap left between training and test stretches as a fraction of a test block. |
| `train_indices`, `test_indices` | `None` | An explicit split given as two lists. It overrides the settings above. The indices address the rows that the network trains on, so they are refused where those rows are not the ones passed in: in `mode='rigorous'`, `'lag'` and `'transfer'`, with `rigorous=True`, with a dimensionality `lag`, for data windowed through `Processing` and for the named quantities that build their own rows. |

## `Estimator`

`Estimator` sets the bound on which the critic is trained, or selects KSG
instead of a critic. A bare name such as `estimator='smile'` is also accepted.

| Parameter | Default | Meaning |
|--|--|----|
| `name` | `'infonce'` | `'infonce'` (low variance, capped at the logarithm of the number of samples evaluated), `'smile'` (lower bias, higher variance) or `'ksg'` (nearest neighbours, no training, see [USING.md](USING.md#estimators)). |
| `params` | `{}` | Extra arguments for the estimator. SMILE takes `clip`, default 5.0. KSG takes `k` (4), `gaussianize` (True), `exclude` (`'auto'`) and `max_tied_share` (0.5). KSG warns when more than `max_tied_share` of a column's rows tie with another row. |

## `Output`

`Output` sets the units, extra diagnostics, embeddings and display names.

| Parameter | Default | Meaning |
|--|--|----|
| `units` | `'bits'` | `'bits'` or `'nats'`, for every MI value in the result. |
| `track_spectral_history` | `False` | Record the participation ratios and spectrum every epoch in `runs` (`spectral_metrics_history`). Needs a critic that gives an embedding of each side (not `critic_type='concat'`). |
| `whitening` | `'std'` | Normalisation of the embeddings before the cross-covariance SVD behind the participation ratios and the rotated embeddings: `'std'`, `'zca'` or `None`. |
| `return_embeddings` | `False` | Keep each repeat's embeddings of every window, in `details[config_id]['embeddings']`. Available in the modes where a repeat is one network, with a critic that gives an embedding of each side (not `critic_type='concat'`). |
| `track_embeddings` | `False` | Keep embeddings every epoch for `result.animate()`: `True` (512 samples), a count, a fraction or `'full'`. Needs a critic that gives an embedding of each side (not `critic_type='concat'`). |
| `return_rotated_embeddings` | `False` | Also keep the embeddings rotated so that dimension 0 has the most shared variance. |
| `rotated_embeddings_per_epoch` | `False` | With tracked embeddings, rotate each epoch on its own, not with the best epoch's rotation. |
| `return_rotation_matrices` | `False` | Keep the rotation matrices, to project new data into the same basis. |
| `channel_names_x`, `channel_names_y` | `None` | Channel names, for the pairwise heatmap. |

## Mode configs

### `Rigorous`

These settings are for `mode='rigorous'`. The same fit settings are also fields
of `Conditional`, `Interaction` and `Transfer`, for `rigorous=True`.

| Parameter | Default | Meaning |
|--|--|----|
| `gamma_range` | `range(1, 11)` | The subdivisions of the data: at $\gamma$ the data are cut into $\gamma$ chunks and each chunk is estimated on its own. |
| `curvature_t_threshold` | `2.0` | The t-statistic of the quadratic term below which the ladder counts as linear. |
| `min_gamma_points` | `5` | Fewest $\gamma$ values a reliable fit may use. |
| `confidence_level` | `0.68` | Confidence level of `mi_error`. |
| `residual_threshold` | `2.5` | Largest studentised residual before `fit_quality_warning` is set. |
| `leverage_threshold` | `0.2` | Largest relative shift of the intercept when the $\gamma = 1$ points are left out before `leverage_warning` is set. |
| `left_out_share` | `0.5` | The fit leaves out a value of $\gamma$, with all its rungs, when the rungs that produced nothing or did not resolve make up this share of its rungs or more. |
| `temporal_chunking` | `None` | Cut chunks as contiguous stretches (`True`) or random subsets (`False`). `None` cuts contiguous chunks from rows in time order (windows a processor cut, the rows of a named quantity, the histories of transfer entropy) and random subsets from other rows. |

### `Precision`

For `mode='precision'`.

| Parameter | Default | Meaning |
|--|--|----|
| `tau_grid` | required | The corruption levels $\tau$ to evaluate. |
| `corrupt_target` | `'x'` | Which side is corrupted: `'x'`, `'y'` or `'both'`. |
| `corruption_method` | `'rounding'` | `'rounding'` (move each value to the centre of its bin of width $\tau$) or `'noise'` (add uniform jitter on $[-\tau/2, \tau/2]$). Either way, entries that contain no measurement, such as unused spike-time slots, stay as they are. |
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
| `gamma_range`, `curvature_t_threshold`, `min_gamma_points`, `confidence_level`, `residual_threshold`, `leverage_threshold`, `left_out_share` | as in `Rigorous` | The fit of `rigorous=True` with the meanings and defaults of [`Rigorous`](#rigorous). |

### `Conditional`

For `mode='conditional'`.

| Parameter | Default | Meaning |
|--|--|----|
| `w_data` | required | The conditioning variable $W$ in $I(X;Y \mid W)$. |
| `rigorous` | `False` | Extrapolate each repeat to infinite data. |
| `gamma_range`, `curvature_t_threshold`, `min_gamma_points`, `confidence_level`, `residual_threshold`, `leverage_threshold`, `left_out_share` | as in `Rigorous` | The fit of `rigorous=True` with the meanings and defaults of [`Rigorous`](#rigorous). |

### `Interaction`

For `mode='interaction'`.

| Parameter | Default | Meaning |
|--|--|----|
| `w_data` | required | The third population $W$. |
| `rigorous` | `False` | Extrapolate each repeat to infinite data. |
| `gamma_range`, `curvature_t_threshold`, `min_gamma_points`, `confidence_level`, `residual_threshold`, `leverage_threshold`, `left_out_share` | as in `Rigorous` | The fit of `rigorous=True` with the meanings and defaults of [`Rigorous`](#rigorous). |

### `Pairwise`

For `mode='pairwise'`.

| Parameter | Default | Meaning |
|--|--|----|
| `pairs` | every pair | The channel pairs `(i, j)` to estimate. |

### `Dimensionality`

For `mode='dimensionality'`.

| Parameter | Default | Meaning |
|--|--|----|
| `embedding_dims` | `None` | A list or a range of the values of `embedding_dim` to fit. `None` chooses them from the reference fit (`grid_reach` and the settings after it), and each split stops once `confirm_steps` values in a row reach the threshold. Given values are all fitted. |
| `n_restarts` | `4` | Networks trained at each `embedding_dim` of each split. The best one counts. |
| `saturation_ratio` | `0.95` | The fraction of the plateau that the curve must reach. The reported dimension is the smallest `embedding_dim` that reaches it. |
| `reference_dim` | `None` | The `embedding_dim` of the large reference fit that runs first. `None` fits 64 and refits at `refit_reach` times the participation ratio when `grid_reach` times the ratio reaches `reference_fill` of 64. |
| `split_method` | `'random'` | Without `y_data`, how X's channels are split in two: `'random'`, `'spatial'` (at the midpoint), `'index'` (by `channel_indices_x`), `'temporal'` (X against X at `lag`), or, for image data, `'horizontal'`, `'vertical'`, `'row_interleaved'`, `'col_interleaved'`, `'diagonal'`, `'antidiagonal'`. |
| `n_splits` | `None` | The number of random channel splits of X (5 when `None`). Only `split_method='random'` draws more than one. No two of its splits divide the channels the same way, so a few channels can give fewer splits. A call with `y_data` refuses it. |
| `lag` | `1` | The lag of `split_method='temporal'`, in samples. |
| `channel_indices_x` | `None` | The channels of X under `split_method='index'`. Y gets the rest. |
| `ceiling_mi_fraction` | `0.85` | Warn when the reference fits' held-out MI is this close to its ceiling. |
| `grid_reach` | `2.0` | The default grid reaches this many times the participation ratio of the reference fit, because that ratio can fall below the true dimension. |
| `grid_small` | `10` | When the reach is at most this value, the default grid contains every `embedding_dim` from 1 to this value. |
| `grid_medium` | `20` | When the reach is above `grid_small` and at most this value, the default grid contains every `embedding_dim` from 1 to this value. |
| `grid_points` | `10` | When the reach is above `grid_medium`, the default grid contains this many values on a log scale from 1 to the reach. |
| `reference_fill` | `0.75` | The reference is refitted when `grid_reach` times its participation ratio reaches this fraction of its `embedding_dim`. |
| `refit_reach` | `4.0` | The refitted reference has this many times the participation ratio as its `embedding_dim`. Its product with `reference_fill` must exceed `grid_reach`, so that the refitted reference is larger. |
| `confirm_steps` | `3` | Each split's default grid stops once its curve has reached the threshold on this many values in a row. The call warns when fewer values confirm a reported dimension. |
| `restart_spread` | `0.1` | The call warns when restarts below the reported dimension differ by more than this share of the plateau. |
| `held_out_gap` | `0.1` | The call warns when the held-out MI of the reference networks falls short of their training MI by more than this share of it. |
