# Messages

This page lists every warning `neural_mi` emits, keyed by the text it prints.
Search for a distinctive phrase from the message to find its entry, where `...`
stands for the values that change from call to call. Each entry says what the
message means, why it fired, and what to do about it. The ceiling,
amplification and coverage warnings change how a result should be reported.

## Contents

- [Ceiling and capacity](#ceiling-and-capacity)
- [Amplification](#amplification)
- [Training](#training)
- [Permutation tests](#permutation-tests)
- [Windowing and coverage](#windowing-and-coverage)
- [Bias correction](#bias-correction)
- [Data shape and type](#data-shape-and-type)
- [Dimensionality](#dimensionality)
- [Settings that had no effect](#settings-that-had-no-effect)
- [Performance](#performance)
- [Other](#other)

---

## Ceiling and capacity

The estimators are lower bounds whose ceiling is the log of the number of
samples they are evaluated on. A value near its ceiling may have been limited by
the bound and not by the data.

**`InfoNCE estimate is near its ceiling`**
The reported MI is within 15% of $\log(\text{train\_eval\_size})$ or of the
held-out ceiling. The true value may be higher, and a curve that flattens here
has probably flattened against the bound. Raise `max_eval_samples` (and
`train_subset_size`, if set), or use `estimator='smile'`, which has no ceiling.

**`Peak-epoch selection may be unreliable`**
Much of the smoothed held-out MI curve sits near its ceiling, so the epoch with
the largest value is close to arbitrary, and so is the reported value computed at
it. Raise `max_eval_samples` so the curve has room to separate.

**`Rigorous fit uses gamma=..., which are saturated`**
The deep rungs of the ladder have the smallest chunks and the lowest ceilings.
When they are saturated, the fitted slope reflects the ceiling as much as the
bias, and the extrapolated intercept inherits it. Raise `max_eval_samples`, or
narrow `gamma_range` to leave out the deep end, keeping at least
`min_gamma_points` values.

**`Fit for ... marked unreliable: gamma=... used in the fit are ceiling-saturated`**
The same condition, reported after the fit, with `is_reliable` set to `False`.

**`Dimensionality: the underlying MI estimate (...) is near its evaluation ceiling`**
Stable directions found here are still trustworthy, since ceiling proximity
lowers the count without inventing directions. The count may be low. Raise
`max_eval_samples` for a fuller read.

---

## Amplification

A quantity formed as a difference of estimates inherits their errors while being
much smaller than either ([THEORY.md](THEORY.md#amplification)).

**`... has an error-amplification factor of ...x`**
The result is a small residual of two much larger estimates, so a 1% error on
each component becomes roughly the factor in percent on the result. Report the
components beside the point estimate, check the joint network for ceiling
saturation, and read the spread over several repeats before drawing a
conclusion.

**`... estimate is negative (...). This is theoretically impossible`**
The quantity cannot be negative, so the noise in the components exceeded the
difference between them. The true value is near zero. Add repeats
(`sweep_grid={'run_id': range(n)}`), train longer, or accept that the quantity is
not resolved at this sample size.

---

## Training

**`batch_size=... exceeds the ... training samples available`**
The batch has been capped to the training set. Batch size sets how many
negatives each training step sees and does not cap the reported value. Larger
batches need more training samples: a longer recording, a smaller `window_size`
or a smaller `step_size`.

**`train_subset_size=... exceeds the number of available training samples`**
The subset is clamped to the training set, and the reported value is evaluated
on at most `max_eval_samples` of those samples.

**`Very few samples detected (... samples)`**, **`Very few samples detected (... windows after processing)`**
Fewer than 200 samples reach the estimator, before or after windowing. Neural
estimators overfit badly at this scale. Add regularisation through
`Model(dropout=..., norm_layer=...)`, or window the data so more windows survive.

**`Small dataset detected (... windows). Regularisation may help`**
Between 200 and 500 windows, with a regularisation option not yet in use. The
message is advice only.

**`Large first embedding layer detected`**
The first layer holds more than 500,000 parameters because the flattened input is
wide, a common cause of overfitting when windows are long and samples few.
Reduce `window_size` or `hidden_dim`, or use an encoder that reads the window
axis itself (`cnn`, `gru`, `tcn`, `lru`).

**`Training completed all ... epoch(s) without early stopping, and the best (smoothed) test MI occurred at the final epoch`**
The MI may still have been rising when training stopped, so the reported value
may be under-trained. Raise `n_epochs`.

**`All test MI values in the training history are non-positive`**
The network learned nothing that generalises, so the reported value is 0 and the
training-side value, which would be pure overfitting, is kept as `raw_train_mi`.
Either there is no dependence to find or training failed. Check the learning
rate first, then the data. In a deliberate null this is the expected outcome.

**`NaN MI detected at epoch ... (consecutive NaN streak: .../3)`**
One evaluation produced NaN, and the epoch is skipped for early stopping. Three
in a row raise `TrainingError`. The usual causes are a learning rate that is too
high or a degenerate batch.

**`MI evaluation returned NaN`**, **`Score matrix contains NaN values during evaluation`**
Numerical instability inside the critic. Check `learning_rate` and `batch_size`,
and whether any channel has zero variance.

**`Score matrix contains Inf values. Clamping`**
The scores overflowed and were clamped so training can continue. Repeated
occurrences mean the critic is diverging, and `Model(norm_layer='layer')` may
help.

**`...=... weighs reconstruction at least as heavily as the mutual information it is there to regularize`**
The reconstruction weights are measured against the MI term, so a weight of 1
trades a unit of reconstruction error for a unit of shared information, and more
than that puts reconstruction in charge of the embedding. The default is 0.001.
With a variational encoder the weight a reconstruction carries is
`beta * lambda`.

---

## Permutation tests

**`With n_permutations=... the smallest p-value the permutation test can report is 1/...`**
With $n$ trials no p-value can fall below $1/(n+1)$, and a reliable p-value
usually needs 100 trials or more. The message also states what the test costs.

**`Each permutation reruns the whole call`**
Every null trial repeats the full call, every configuration of the grid included,
so the test costs `n_permutations` times the call. It is a warning when the grid
has several configurations.

**`Permutation trial failed`**, **`All ... permutation trials for mode='...' failed, so there is no null distribution`**
One trial raised, or every trial did. Without a null the test is inconclusive.
The log shows each failure.

**`permutation_test=True has no effect for mode='dimensionality'`**
The mode reports a count of stable directions, and there is no single value for
a null to sit under. A direction counts as stable only when it survives independent fits, and the
stability threshold sets that bar.

**`permutation_test=True has no effect for mode='pairwise' without y_data`**
Every pair is two channels of X, and moving X moves both sides of each pair
together, so no null is computed. Pass `y_data` for cross-pairwise MI, which
supports the test.

---

## Windowing and coverage

**`step_size=... was read as a fraction of window_size=...`**
A `step_size` below 1 is always a fraction of the window, so `step_size=0.25` is
75% overlap at any window size. In seconds that rule meets ordinary values:
`window_size=0.5, step_size=0.5` gives a 0.25 s step. The message names the step
applied and how to ask for the other reading, `step_size=None` for touching
windows and `step_size=a/window_size` for an absolute step `a`. It fires only
when `window_size` is below 1.

**`Window coverage validation kept ... of ... windows`**
Windows were dropped because a stream lacked valid data in them, and the message
names which. The estimate describes the retained windows, and a per-window value
is per retained window. Report the retention beside any estimate on real data.
For spike data, `{'drop_empty_windows': False}` in the processor parameters
keeps silent windows and estimates the unrestricted quantity
([USING.md](USING.md#silent-spike-windows)).

**`Blocked split may leak raw samples between train and test`**
The gap around each test block is shorter than a window, so windows on either
side of a boundary share raw samples. Raise `Split(gap_fraction=...)` to the
value the message gives.

**`Blocked split parameters produced an invalid configuration`**
The requested blocks do not fit the dataset, and the split fell back to random,
which leaks between neighbouring windows of a time series. Reduce
`n_test_blocks` or use more data.

**`mode='conditional': x_data/y_data have ... windows but w_data has ...`**, **`mode='interaction': x_data/y_data have ... windows but w_data has ...`**
The three variables reached the engine with slightly different window counts,
and all three were cut to the shorter length. `nmi.run()` aligns streams by
window time before this point, so the message means already-windowed arrays were
passed to the engine directly. The cut is correct only when the extra window is
at an edge. Pass arrays that agree in length.

**`mode='conditional': x_data window size (...) and w_data window size (...) differ`**, **`mode='interaction': x_data window size (...) and w_data window size (...) differ`**
The windows of X and W differ in length by a sample or two and were trimmed to
match. Check that `Processing(x_params=...)` and `Processing(w_params=...)`
agree on `window_size` and `sample_rate`.

**`ContinuousWindowDataset: .../... interpolated time points (...) are zero-padded due to data gaps`**
More than a tenth of the interpolated points fall outside the recorded range and
were filled with zeros. Check the time vector against the data.

**`ContinuousWindowDataset: .../... retained window(s) have over 30% of their samples bridged by interpolation`**
Windows that passed the coverage check are still mostly interpolated across a
large gap, and they are in the estimate. Raise `min_coverage_fraction` to drop
them.

**`SpikeWindowDataset: max_spikes_per_window cap applied`**
Spikes beyond the cap are dropped in each window. Shorten the window or raise the
cap if the dropped spikes matter.

**`Spike tensor allocation dominated by burst`**
One neuron's peak count sets the tensor width for every neuron, and the unused
width costs memory. `exclude_bursty_neurons=True` leaves such neurons out.

**`Excluding ... bursty neuron(s)`**
That exclusion is active, and the listed neurons are not in the estimate.

**`SubsetView: window count dropped from ... to ...`**
A time shift moved windows outside the valid recording range. A large drop means
the shifted windows cover a different subset from the unshifted ones.

**`Tried to time-shift a non-windowed dataset, skipping`**
The shift did nothing, since there is no window grid to move.

**`time_shift got offset_x=... != offset_y=...`**
Every stream shares one window grid, so only `offset_x` is applied.

**`Stream ... specifies window_size=..., but every stream shares a single WindowManager`**
One grid carries one window size, and the first stream to name one sets it. Give
every stream the same value.

**`Stream ... specifies step_size=... against the ... already set by an earlier stream`**
The same rule for the step.

**`shift_windows is off for this run: stream ... was given a time vector whose sampling is irregular`**
The reslicing route needs one sampling period for the whole recording, and a
clock with a gap has none. The windows are built the slower way instead. That route
checks every window against the timestamps and drops those that straddle the
gap. The estimate is unaffected, and the cheap per-epoch shift is lost. A
`sample_rate` in that stream's processor parameters states a period and keeps
the reslicing route.

**`shift_windows is off for this run: the streams' time vectors start ... apart`**
The reslicing route lines streams up by sample number, and sample number matches
real time only when their clocks start together. The slower route starts the grid at the
latest start among the streams and is used instead.

---

## Bias correction

**`gamma=...: smallest data subset has ... samples, below the ~... at which the held-out partition of a chunk can still fill one evaluation batch`**
The deep rungs of the ladder come from chunks too small to evaluate reliably, so
their values can be dominated by noise, and a straight line through noisy rungs
still looks straight. Use a smaller `gamma_range` or more data.
`Training(min_reliable_samples=...)` moves the point at which this fires.

**`Fit for ... is unreliable (final gamma points < ...)`**
Too few rungs remain to fit, and `is_reliable` is `False`. Widen `gamma_range`,
or lower `min_gamma_points` and accept a weaker fit.

**`Fit for ... is unreliable: no linear region was found`**
Enough rungs remain, but they never met the curvature criterion, so the fit uses
them without the assumption behind it holding. The intercept cannot be trusted.

**`Fit diagnostics triggered for ...: leverage_warning=...`**
Leaving out the $\gamma = 1$ rungs moves the intercept by more than
`leverage_threshold`, so the extrapolation leans on the full-data anchor.
`is_reliable` is `False`.

**`Rigorous analysis: expected ... tasks`**
The ladder has fewer trainings than requested. Find out why before trusting the
extrapolation, since the missing rungs are not a random subset.

**`Rigorous fit (rigorous=True): linear region too small after pruning`**, **`Rigorous fit (rigorous=True): no linear region was found`**, **`Rigorous fit (rigorous=True): fit diagnostics triggered`**
The same conditions for `rigorous=True` on `Conditional`, `Interaction`,
`Transfer` and the named quantities, where the combined values are fitted.

**`Rigorous fit (rigorous=True): the estimate for one chunk failed`**
One chunk's estimate raised, and the message carries the error. The fit goes on
without that chunk while at least `min_gamma_points` remain.

---

## Data shape and type

**`CategoricalWindowDataset: input data has dtype ..., not an integer type`**
The labels were mapped to consecutive integer codes in sorted order of the
distinct values. Pass integer codes to choose the mapping yourself.

**`... have different numbers of samples (...). Truncating ... to ... samples`**, **`X (...) and Y (...) differ in sample count; truncating`**
Both were cut to the shorter length. A large cut means the variables do not
cover the same interval. Fix the mismatch at its source.

**`x_data has ... channels, y_data has ...`**
Spike-type X and Y have different neuron counts, usually on purpose and
occasionally a mis-built list.

**`... spike times are not sorted`**
They are sorted automatically, and the message is harmless.

**`ContinuousWindowDataset: first inter-sample interval (...) differs from the median`**
The first gap in the time vector is unusual, and the median period is used. Check
the start of the recording.

**`processor_type mixes 'spike' (...) with '...' on stream ...`**
Spike times are in seconds, and a continuous or categorical stream without a
clock or a `sample_rate` is in samples, so the grid would line up windows that
cover different times. The message names the stream. Give it a `sample_rate` or a
time vector.

**`Unrecognised augmentation key(s) ...; they will be ignored`**
A key outside the vocabulary is skipped, so that augmentation is not applied
while training looks normal. The message lists the valid keys, and the cause is
usually a typo.

**`embedding_model='...' received 4-D input`**
The encoder flattens the spatial axes. Use `cnn2d` or `pretrained_backbone` for
image-shaped input.

**`Spatial augmentations ... require 4-D input`**
The image augmentations are skipped because the batch is not image-shaped.

**`PretrainedBackboneEmbedding: input has ... channel(s) but backbone '...' expects ...`**, **`PretrainedBackboneEmbedding: input spatial size (...) does not match`**
A trainable channel adapter or an upsampling layer was inserted so the backbone
can run. Both add parameters that are not in the pretrained model.

---

## Dimensionality

**`mode='dimensionality' with critic_type='separable'`**
A dot-product critic ties the geometry of the embeddings to the score, and that
tie can change which directions come out stable. The mode's default is the hybrid
critic.

**`Dimensionality: ... of ... split(s) did not converge`**
Training used the whole epoch budget without early stopping, so the count of
stable directions may be low. Raise `n_epochs` before reading it.

**`Dimensionality: fewer than 2 splits produced usable rotated embeddings`**
Stability needs at least two fits to compare, so no stable directions can be
reported.

**`split_method='...' on an odd channel count`**, **`split_method='index' with unequal channel counts`**, **`split_method='...' on non-square input`**, **`split_method='...' produced unequal halves`**
The two halves differ in size. `shared_encoder=True` needs equal halves, so the
shared encoder is turned off for the run.

**`... was set for mode='dimensionality', whose two sides are two halves of the same recording`**
A Y-side encoder setting (`embedding_model_y`, `custom_embedding_cls_y`,
`hidden_dim_y`, `n_layers_y` or `embedding_dim_y`) was given. An encoder that
differs between the halves makes the count of stable directions hard to read.
The run proceeds.

---

## Settings that had no effect

Each of these means a setting had no effect. None is an error, and each means the
run did something other than what was asked.

**`Mode config(s) ... were provided but mode='...'`**
Only the config of the active mode is used.

**`sweep_grid has no effect for mode='...'`**
`mode='estimate'` and `mode='precision'` run one configuration once. Use
`mode='sweep'` to repeat an estimate or sweep a setting, and run
`mode='precision'` once per setting to compare settings.

**`w_data was provided but mode='...' does not use it`**
Only `conditional`, `interaction` and `transfer` read a third variable.

**`hidden_dim is a list of length ..., so n_layers=... is ignored`**
The list sets the number of layers.

**`Custom train_indices and test_indices were provided`**
`Split`'s `mode`, `train_fraction`, `n_test_blocks` and `gap_fraction` are
ignored in favour of the given indices.

**`'lag' in sweep_grid is ignored when using mode='lag'`**
The mode sets the lag itself.

**`shift_time=True has no effect for mode='...'`**, **`shift_windows=True has no effect for mode='...'`**
The shift cannot reach this mode with these processors, and the message says
where it can ([USING.md](USING.md#window-shifting)).

**`return_rotated_embeddings=True has no effect for critic_type='concat'`**, **`return_rotated_embeddings=True was requested, but this trainer's model exposes no embedding_net_x`**
The concat critic has no separate encoders whose embeddings could be rotated.

**`return_rotated_embeddings=True requires track_embeddings to be enabled`**
Per-epoch rotation needs per-epoch embeddings. Set `track_embeddings`.

**`No dedicated decoder for embedding_model='...'; falling back to MLPDecoder`**
The reconstruction uses a generic decoder for this encoder.

**`Both indices and times provided, using times only`**
A subset of a temporal dataset was given both window indices and time ranges,
and the time ranges define it.

**`panels includes 'embeddings' but this repeat holds no 'embedding_history_x'`**
The animation drops its embedding panel. Set `Output(track_embeddings=...)` before
the run to record embeddings every epoch.

---

## Performance

**`n_workers=... has no effect here`**
The call trains one network, so there is nothing to run in parallel.

**`save_best_model_path: this call trains several networks and saves every one of them`**
Every network of the call is saved, each under the path with its identifying
labels added, and each path is recorded as `model_path`. Every file holds a whole
network, so a large grid or rigorous ladder can take a lot of disk space. To keep
one model, rerun the configuration you want on its own with this path
([USING.md](USING.md#saving-the-trained-networks)).

**`Running ... sequential tasks with dataset_device='...'`**
Freed tensors can linger in an accelerator's cache across many tasks. Use
`Training(dataset_device='cpu')` for long sweeps.

**`CategoricalWindowDataset: encoding='full_trajectory' will produce a ...-dimensional input`**
Categories times window length grows quickly. Use `'majority_vote'` or
`'probability'` unless the time course within the window matters.

**`track_embeddings=...: storing embeddings for all ... samples at every epoch`**
Storing every sample at every epoch takes a lot of memory. Pass a count to track
the first samples only.

---

## Other

**`Lag on pre-processed data with shape ...: axis 0 is windows, not timepoints`**
The data were already windowed, so a lag of $n$ shifts by $n$ windows, or
$n$ times `step_size` samples. Pass unwindowed data with a `Processing` to lag
in samples.

**`Lag units for '...' data are ambiguous without a sample_rate`**
The lag is read as a number of samples. A `sample_rate` puts it in seconds.

**`MI never dropped below ...% of baseline`**
`mode='precision'` never reached its threshold, so no precision can be reported.
Extend `tau_grid` upward.

**`mode='precision': MI goes negative from tau=... onward`**
Past its threshold the mode evaluates a critic trained on clean data against
corrupted inputs, and the bound is unbounded below. The depth of the fall
measures how far the bound has broken, so the message reports it as a
multiple of the baseline. The crossing of the threshold is still readable. Do not
quote or plot the tail.

**`bias=False was requested, but ... carries an input-independent additive term`**
A positional encoding or the biases of a pretrained backbone mean an all-zero
input does not embed to zero, even with the encoder's own bias terms removed.

**``bias=False applies to the embedding layers, but ....__init__ does not accept a `bias` argument``**
A custom encoder cannot receive the setting, so its layers keep their bias terms.
Accept `bias` in its `__init__` and pass it to the layers.

**`No tasks to run. Your sweep_grid might be empty`**
The grid produced no tasks.

**`FFMpeg not found; falling back to PillowWriter (GIF)`**
Install ffmpeg to write MP4.

**`No significant MI contour found at threshold ...`**
`analyze_mi_heatmap` found no region above its threshold. Lower
`absolute_mi_threshold`.
