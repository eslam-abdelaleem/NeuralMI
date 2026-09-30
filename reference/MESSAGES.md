# Messages

This page lists every warning `neural_mi` emits, keyed by the text it prints.
Search for a distinctive phrase from the message to find its entry. In the
entries `...` stands for the values that change from call to call. Each entry
says what the message means, why it fired, and what to do about it. The ceiling,
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
held-out ceiling. The true value may be higher. A curve that flattens here has
probably flattened against the bound. Raise `max_eval_samples` (and
`train_subset_size` if set) or use the less biased `estimator='smile'` at the cost of more variance.

**`Peak-epoch selection may be unreliable`**
When much of the smoothed held-out MI curve sits near its ceiling, the epoch
with the largest value and the value reported at it are close to arbitrary.
Raise `max_eval_samples` so the curve has room to separate.

**`Rigorous fit uses the saturated gamma=...`**
The deep rungs of the ladder have the smallest chunks and the lowest ceilings.
When they are saturated, the fitted slope and the extrapolated intercept reflect
the ceiling as much as the bias. Raise `max_eval_samples` or narrow
`gamma_range` to leave out the deep end while keeping at least
`min_gamma_points` values.

**`Fit for ... marked unreliable: gamma=... used in the fit are ceiling-saturated`**
The fit reports the same condition afterwards and sets `is_reliable` to `False`.

**`Dimensionality: the reference fits' MI (...) is near its evaluation ceiling`**
A plateau capped by the ceiling lets small embeddings reach the threshold early.
The reading can then fall below the dimension it is meant to bound. Raise
`max_eval_samples`.

---

## Amplification

A quantity formed as a difference of estimates inherits their errors while being
much smaller than either ([THEORY.md](THEORY.md#amplification)). The warnings in
this section share the `UserWarning` subclass `nmi.CombinationWarning` so that
`warnings.filterwarnings` can silence them as a group.

**`... has an error-amplification factor of ...x`**
Because the result is a small residual of much larger estimates (two for a
conditional quantity, three for interaction information), a 1% error on each
component becomes roughly the factor in percent on the result. Report the
components beside the point estimate, check the joint network for ceiling
saturation, and read the spread over several repeats before drawing a
conclusion.

**`... estimate is negative (...). The quantity cannot be negative`**
The quantity is reported as 0 with its measured value kept in `mi_raw` (or
`te_yx_raw`). The joint term came out below the term it contains although the
true values cannot be in that order. A high amplification factor means the two
are close and component noise flipped their order around a true value near zero.
Without more repeats (`sweep_grid={'run_id': range(n)}`) or more data the
quantity is not resolved at this sample size. At a low factor the gap is larger
than noise explains because one network fell short. That network is usually the
joint one and needs longer training or more capacity.

**`... repeats produced nothing (reported as 0) and are left out of mi_mean and mi_std`**, **`... rows of dataframe hold ... of their ... repeats that produced nothing`**
A repeat is reported as 0 when its network learned nothing that generalises or
its value came out negative. `mi_mean` and `mi_std` average the other repeats,
`n_zero` counts the zeros in each row, and a row whose repeats all produced
nothing reports 0. Among clearly positive repeats a zero is most likely a failed
run. When every repeat is near zero the quantity itself may be near zero. `runs`
keeps every repeat with its measured value.

**`The extrapolated ... is negative`**
The rigorous fit extrapolated below zero. The value is reported as 0 and the
extrapolated one is kept as `mi_raw`. The interval in the message usually
includes zero. The quantity is not resolved at this sample size.

**`The components of ... (...) are in an impossible order`**
A negative interaction information is a result because the quantity is signed
([THEORY.md](THEORY.md#interaction-information)). Its joint term $I(X,W;Y)$
contains both $I(X;Y)$ and $I(W;Y)$ and here came out below one of them. The
message reads the gap as the previous entry does.

---

## Training

**`batch_size=... exceeds the ... training samples available`**
The batch has been capped to the training set. Batch size sets how many
negatives each training step sees and does not cap the reported value. Larger
batches need more training samples: a longer recording, a smaller `window_size`
or a smaller `step_size`.

**`train_subset_size=... exceeds the number of available training samples`**
The subset is clamped to the training set. The reported value is evaluated on at
most `max_eval_samples` of those samples.

**`Only ... reach the estimator`**
Whether the fewer than 200 samples or windows that reach the estimator are
enough depends on the data. The samples an estimate needs grow roughly as $N
\sim d^2/I$ ([tutorial
02](https://eslam-abdelaleem.github.io/NeuralMI/tutorials/02_Your_First_Number.html),
section 2). Neither the number $d$ of latent dimensions carrying the shared
information nor the information $I$ itself is known before estimating.
`mode='rigorous'` tests whether the estimate still moves with the sample size
([THEORY.md](THEORY.md#finite-sampling-bias)). The message also lists the
regularisation settings not yet in use.

**`Large first embedding layer detected`**
The first layer holds more than 500,000 parameters because the flattened input
is wide. With long windows and few samples it commonly overfits. Reduce
`window_size` or `hidden_dim`. An encoder that reads the window axis itself
(`cnn`, `gru`, `tcn`, `lru`) also helps.

**`Training completed all ... epoch(s) without early stopping and the best (smoothed) test MI occurred at the final epoch`**
The reported value may be under-trained because the MI may still have been
rising when training stopped. Raise `n_epochs`.

**`All test MI values in the training history are non-positive`**
Because the network learned nothing that generalises, the reported value is 0.
The training-side value would be pure overfitting and is kept as `raw_train_mi`.
Either there is no dependence to find or training failed. Check the learning
rate before the data. In a deliberate null this is the expected outcome.

**`The train MI at the reported epoch is negative`**
The training-side value at the reported epoch came out negative after the test
MI had risen above zero. Because MI cannot be negative, the reported value is 0
and the measured one is kept as `raw_train_mi`. The usual causes are too few
epochs or too few samples.

**`NaN MI detected at epoch ... (consecutive NaN streak: .../3)`**
An epoch whose evaluation produced NaN is skipped for early stopping. Three in a
row raise `TrainingError`. The usual causes are a learning rate that is too high
or a degenerate batch.

**`MI evaluation returned NaN`**, **`Score matrix contains NaN values during evaluation`**
The critic became numerically unstable. Check `learning_rate`, `batch_size` and
whether any channel has zero variance.

**`Score matrix contains Inf values. Clamping`**
The scores overflowed and were clamped so training can continue. Repeated
occurrences mean the critic is diverging. `Model(norm_layer='layer')` may help.

**`...=... weighs reconstruction at least as heavily as the mutual information it is there to regularise`**
Because the reconstruction weights are measured against the MI term, a weight of
1 trades a unit of reconstruction error for a unit of shared information. More
than that puts reconstruction in charge of the embedding. The default is 0.001.
With a variational encoder the weight a reconstruction carries is `beta *
lambda`.

---

## Permutation tests

**`With n_permutations=... the smallest p-value the permutation test can report is 1/...`**
With $n$ trials no p-value can fall below $1/(n+1)$. A reliable p-value usually
needs 100 trials or more. The message also states what the test costs.

**`Each permutation reruns the whole call`**
Because every null trial repeats the full call with every configuration of the
grid, the test costs `n_permutations` times the call. It is a warning when the
grid has several configurations.

**`Permutation trial failed`**, **`All ... permutation trials for mode='...' failed and left no null distribution`**
One trial raised an exception or every trial did. Without a null the test is
inconclusive. The log shows each failure.

**`permutation_test=True has no effect for mode='dimensionality'`**
The curve and the dimension read from it are no single MI value for a null to
sit under. Run `mode='estimate'` with `permutation_test=True` to test whether
the two views share any information.

**`permutation_test=True has no effect for mode='pairwise' without y_data`**
Because every pair is two channels of X, moving X moves both sides of each pair
together and no null is computed. Cross-pairwise MI with `y_data` supports the
test.

---

## Windowing and coverage

**`step_size=... was read as a fraction of window_size=...`**
A `step_size` below 1 is always a fraction of the window. `step_size=0.25` is
75% overlap at any window size. In seconds that rule meets ordinary values:
`window_size=0.5, step_size=0.5` gives a 0.25 s step. The message names the step
applied and how to ask for the other reading with `step_size=None` for touching
windows or `step_size=a/window_size` for an absolute step `a`. It fires only
when `window_size` is below 1.

**`Window coverage validation kept ... of ... windows`**
Windows were dropped because the stream the message names lacked valid data in
them. The estimate describes the retained windows only and is in bits per
retained window. Report the retention beside any estimate on real data. For
spike data, `{'drop_empty_windows': False}` in the processor parameters keeps
silent windows and estimates the unrestricted quantity
([USING.md](USING.md#silent-spike-windows)).

**`Blocked split may leak raw samples between train and test`**
Because the gap around each test block is shorter than a window, windows on
either side of a boundary share raw samples. Raise `Split(gap_fraction=...)` to
the value the message gives.

**`Blocked split parameters produced an invalid configuration`**
Because the requested blocks do not fit the dataset, the split fell back to a
random split that leaks between neighbouring windows of a time series. Reduce
`n_test_blocks` or use more data.

**`mode='conditional': x_data and y_data have ... windows and w_data has ...`**, **`mode='interaction': x_data and y_data have ... windows and w_data has ...`**
The three variables reached the engine with slightly different window counts and
were all cut to the shorter length. Because `nmi.run()` aligns streams by window
time before this point, the message means already-windowed arrays were passed to
the engine directly. The cut is correct only when the extra window is at an
edge. Pass arrays that agree in length.

**`mode='conditional': x_data window size (...) and w_data window size (...) differ`**, **`mode='interaction': x_data window size (...) and w_data window size (...) differ`**
The windows of X and W differ in length by a sample or two and were trimmed to
match. Check that `Processing(x_params=...)` and `Processing(w_params=...)`
agree on `window_size` and `sample_rate`.

**`ContinuousWindowDataset: .../... interpolated time points (...) are zero-padded due to data gaps`**
More than a tenth of the interpolated points fall outside the recorded range and
were filled with zeros. Check the time vector against the data.

**`ContinuousWindowDataset: .../... retained window(s) have over 30% of their samples bridged by interpolation`**
Windows that passed the coverage check and entered the estimate are still mostly
interpolated across a large gap. Raise `min_coverage_fraction` to drop them.

**`SpikeWindowDataset: max_spikes_per_window cap applied`**
Spikes beyond the cap are dropped in each window. Shorten the window or raise the
cap if the dropped spikes matter.

**`Spike tensor allocation dominated by burst`**
The unused width costs memory because one neuron's peak count sets the tensor
width for every neuron. `exclude_bursty_neurons=True` leaves such neurons out.

**`Excluding ... bursty neuron(s)`**
The exclusion is active and leaves the listed neurons out of the estimate.

**`SubsetView: window count dropped from ... to ...`**
A time shift moved windows outside the valid recording range. A large drop means
the shifted windows cover a different subset from the unshifted ones.

**`Tried to time-shift a non-windowed dataset, skipping`**
The shift did nothing because there is no window grid to move.

**`time_shift got offset_x=... != offset_y=...`**
Only `offset_x` is applied because every stream shares one window grid.

**`Stream ... specifies window_size=.... Every stream shares a single WindowManager`**
The first stream to name a window size sets it for the one grid every stream
shares. Give every stream the same value.

**`Stream ... specifies step_size=... against the ... already set by an earlier stream`**
The same rule for the step.

**`shift_windows is off for this run because stream ... was given a time vector whose sampling is irregular`**
A clock with a gap has no single sampling period for the reslicing route to use.
The slower route builds the windows instead by checking every window against the
timestamps and dropping those that straddle the gap. The run loses the cheap
per-epoch shift without any effect on the estimate. A `sample_rate` in that
stream's processor parameters states a period and keeps the reslicing route.

**`shift_windows is off for this run because the streams' time vectors start ... apart`**
The reslicing route lines streams up by a sample number that matches real time
only when their clocks start together. The slower route used instead starts the
grid at the latest start among the streams.

---

## Bias correction

**`gamma=...: the smallest data subset has ... samples. Below about ... the held-out partition of a chunk cannot fill one evaluation batch`**
The values of the deep rungs can be dominated by noise because their chunks are
too small to evaluate reliably. A straight line through noisy rungs still looks
straight. Use a smaller `gamma_range` or more data.
`Training(min_reliable_samples=...)` moves the point at which this fires.

**`... of ... rungs of the gamma ladder produced nothing`**
Those rungs learned nothing that generalises or came out negative. The fit
leaves them out like failed chunks and counts them in the `zero_rungs` column of
`runs`. The message gives the count at each $\gamma$.

**`...half or more of the rungs produced nothing at gamma=...`**
The chunks at those values of $\gamma$ are likely too small to estimate the
quantity. A $\gamma$ with no rung left drops
out of the fit. With fewer than `min_gamma_points` values of $\gamma$ left the
fit is unreliable. Use a smaller `gamma_range` or more data.

**`...every rung of the gamma ladder produced nothing`**, **`...only gamma=... has rungs that produced a value`**
Since a line needs two values of $\gamma$, the fit reports 0 when no rung is
left and NaN when only one value of $\gamma$ is left. Either way `is_reliable`
is `False`. Use a smaller `gamma_range`, more data or more training.

**`Fit for ... is unreliable (final gamma points < ...)`**
`is_reliable` is `False` because too few rungs remain to fit. Widen
`gamma_range` or lower `min_gamma_points` and accept a weaker fit.

**`Fit for ... is unreliable. No linear region was found`**
Enough rungs remain but never met the curvature criterion. The fit uses them
without the linearity it assumes and gives an intercept that cannot be trusted.

**`Fit diagnostics triggered for ...: leverage_warning=...`**
Leaving out the $\gamma = 1$ rungs moves the intercept by more than
`leverage_threshold` because the extrapolation leans on the full-data anchor.
`is_reliable` is `False`.

**`Rigorous analysis expected ... tasks`**
The ladder has fewer trainings than requested. The missing rungs are not a
random subset. Find out why before trusting the extrapolation.

**`Rigorous fit (rigorous=True): the linear region is too small after pruning`**, **`Rigorous fit (rigorous=True): no linear region was found`**, **`Rigorous fit (rigorous=True): fit diagnostics triggered`**
These are the same conditions for `rigorous=True` on `Conditional`,
`Interaction`, `Transfer` and the named quantities that fit the combined values.

**`Rigorous fit (rigorous=True): the estimate for one chunk failed`**
One chunk's estimate raised the error the message carries. The fit goes on
without that chunk while at least `min_gamma_points` remain.

---

## Data shape and type

**`CategoricalWindowDataset: input data has dtype ..., not an integer type`**
The labels were mapped to consecutive integer codes in sorted order of the
distinct values. Pass integer codes to choose the mapping yourself.

**`... have different numbers of samples (...). Truncating ... to ... samples`**, **`X (...) and Y (...) differ in sample count`**
Both were cut to the shorter length. A large cut means the variables do not
cover the same interval. Fix the mismatch at its source.

**`x_data has ... channels, y_data has ...`**
Different neuron counts in spike-type X and Y are usually on purpose and
occasionally come from a mis-built list.

**`... spike times are not sorted`**
The message is harmless because the library sorts them automatically.

**`ContinuousWindowDataset: first inter-sample interval (...) differs from the median`**
The library uses the median period because the first gap in the time vector is
unusual. Check the start of the recording.

**`processor_type mixes 'spike' (...) with '...' on stream ...`**
Because spike times are in seconds and a continuous or categorical stream
without a clock or a `sample_rate` is in samples, the grid would line up windows
that cover different times. The message names the stream. Give it a
`sample_rate` or a time vector.

**`Unrecognised augmentation key(s) ... are ignored`**
Training looks normal and never applies the augmentation named by a key outside
the vocabulary. The message lists the valid keys. The cause is usually a typo.

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
A dot-product critic scores a pair by the inner product of its two embeddings.
Representing a nonlinear dependence that way takes more dimensions than the
shared structure has. The reading then overstates the dimension. The mode's
default is the hybrid critic.

**`Dimensionality: the grid runs over embedding_dim 1 to ... because the reference fit has a participation ratio of ...`**, **`Dimensionality: the grid runs over ... values of embedding_dim from 1 to ... on a log scale`**
The default grid reaches twice the participation ratio of the reference fit.
The ratio ran below the true dimension in the tests (14.6 for 16 latent
dimensions). The factor of two leaves room for that. When twice the ratio is at
most 10 the grid is 1 to 10 with no message. Up to 20 it is 1 to 20.
Past 20 it is 10 values on a log scale up to twice the ratio. The reading is
then one of those values. Pass `embedding_dims` as a list or a range for a
finer grid.

**`Dimensionality: the participation ratio of the reference fit (...) is close to its embedding_dim=...`**
A reference whose participation ratio reaches three eighths of its size may be
too small to hold every shared direction. It is refitted once at four times the
ratio. A `reference_dim` you set is kept as given.

**`Dimensionality: the grid stopped at embedding_dim=... once ... consecutive values reached ...`**
The default grid is fitted three values at a time. Three values in a row at the
threshold confirm the reading against the plateau the reference fit sets. The
values above them are skipped. Pass `embedding_dims` to fit every value.

**`Dimensionality: the curve of split(s) ... did not reach ... of its plateau by embedding_dim=...`**
The default grid ended below the threshold. It is extended once with values on
a log scale up to one below the reference. A curve still short of the threshold
after the extension reads the reference and raises the next message.

**`Dimensionality: the curve of split(s) ... reached ... of its plateau only at the reference embedding_dim=...`**
No fitted value below the reference reached the threshold. The reading equals
the reference and bounds nothing below it. Training may have stopped short
(raise `n_epochs`), the dimension may lie between the last grid value and the
reference (pass `embedding_dims` closer to it), or the reference may be too
small (pass a larger `reference_dim`).

**`Dimensionality: the views of split(s) ... share no information the estimator can find`**
Every fit of the split ended at 0. With a plateau of 0 the split has no reading
and is left out of the median. Check that the two views are aligned and that
training ran long enough to learn anything.

**`Dimensionality: the restarts at embedding_dim ... below the reading differ by more than ...`**
A restart can settle with fewer directions than its embedding allows and read
low. The curve uses the best restart at each value. When the restarts disagree
this much below the reading, the best one may also have settled short and the
reading may be too high. Raise `n_restarts`.

**`Dimensionality: at the reference fit the held-out MI (...) is more than ... below the plateau`**
The plateau is the training-side MI. A held-out value far below it means the
data hold few samples for the information each dimension carries. The training
side then keeps rising with the embedding size and pushes the reading up. In
the tests a gap of 28% came with a reading of 20 for 16 latent dimensions. The
reading still bounds the dimension from above. More data tightens it.

**`split_method='...' on an odd channel count`**, **`split_method='index' with unequal channel counts`**, **`split_method='...' on non-square input`**, **`split_method='...' produced unequal halves`**
The two halves differ in size. `shared_encoder=True` needs equal halves and is
turned off for the run.

**`... was set for mode='dimensionality'. The mode's two sides are two halves of the same recording`**
A Y-side encoder setting (`embedding_model_y`, `custom_embedding_cls_y`,
`hidden_dim_y`, `n_layers_y` or `embedding_dim_y`) was given. The run proceeds.
An encoder that differs between the halves makes the curve hard to read.

---

## Settings that had no effect

None of these is an error. Each means a setting had no effect and the run did
something other than what was asked.

**`Mode config(s) ... were provided for a call with mode='...'`**
Only the config of the active mode is used.

**`sweep_grid has no effect for mode='...'`**
`mode='estimate'` and `mode='precision'` run one configuration once. Use
`mode='sweep'` to repeat an estimate or sweep a setting. Run `mode='precision'`
once per setting to compare settings.

**`w_data was provided but mode='...' does not use it`**
Only `conditional`, `interaction` and `transfer` read a third variable.

**`hidden_dim is a list of length ... and sets the number of hidden layers`**
`n_layers` is ignored.

**`These settings have no effect in this call and are ignored: ...`**
Each listed setting applies only with an encoder, critic or switch that this
call does not use and that the message names. `kernel_size` applies to `'cnn'`,
`'cnn2d'` and `'tcn'`, `bidirectional` to `'gru'` and `'lstm'`, `nhead` to
`'transformer'`, `pytorch_predefined` and `pretrained` to
`'pretrained_backbone'`, and `branch_model` to `'dual_branch'`. The head
settings apply to `critic_type='hybrid'`, `beta` to `use_variational=True`, and
the decoder settings to `use_decoder=True`. With `custom_critic` the critic is
used as given. Every other `Model` setting is ignored. Of the `Output` settings,
`return_rotated_embeddings` needs `return_embeddings` or `track_embeddings`,
`rotated_embeddings_per_epoch` needs both `track_embeddings` and
`return_rotated_embeddings`, and `return_rotation_matrices` needs
`return_rotated_embeddings`. `mode='dimensionality'` sets `embedding_dim`
itself. A setting spelled out at its default does not count.

**`Custom train_indices and test_indices were provided`**
`Split`'s `mode`, `train_fraction`, `n_test_blocks` and `gap_fraction` are
ignored in favour of the given indices. The indices address the rows the network
trains on. The call refuses them where those rows are not the ones passed:
`mode='rigorous'`, `'lag'` and `'transfer'`, `rigorous=True`, a dimensionality
`lag`, data windowed through `Processing`, and the named quantities that build
their own rows.

**`'lag' in sweep_grid is ignored when using mode='lag'`**
The mode sets the lag itself.

**`shift_time=True has no effect for mode='...'`**, **`shift_windows=True has no effect for mode='...'`**
The shift cannot reach this mode with these processors. The message says where
it can ([USING.md](USING.md#window-shifting)).

**`return_rotated_embeddings=True has no effect for critic_type='concat'`**, **`return_rotated_embeddings=True was requested and this trainer's model exposes no embedding_net_x`**
The concat critic has no separate encoders whose embeddings could be rotated.

**`No dedicated decoder exists for embedding_model='...'`**
The reconstruction uses a generic decoder for this encoder.

**`Both indices and times provided, using times only`**
A subset of a temporal dataset given both window indices and time ranges is
defined by the time ranges.

**`panels includes 'embeddings' but this repeat holds no 'embedding_history_x'`**
The animation drops its embedding panel. Set `Output(track_embeddings=...)` before
the run to record embeddings every epoch.

---

## Performance

**`n_workers=... has no effect here`**
The call trains one network and has nothing to run in parallel.

**`save_best_model_path: this call trains several networks and saves every one of them`**
The call saves every network under the path with its identifying labels added
and records each path as `model_path`. Because every file holds a whole network,
a large grid or rigorous ladder can take a lot of disk space. To keep one model,
rerun the configuration you want on its own with this path
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

**`... was raised ... times in this ... and shown once`**
A rigorous ladder, a parallel grid or a loop of calls can raise the same warning
or log message many times. The first one was shown when it happened. Later ones
that differed only in their numbers were counted and not shown. The count covers
one call, one notebook cell or one `nmi.grouped_warnings()` block. The message
quotes the start of the first one.

**`Lag on pre-processed data with shape .... Axis 0 counts windows`**
Because the data were already windowed, a lag of $n$ shifts by $n$ windows and
so by $n$ times `step_size` samples. Pass unwindowed data with a `Processing` to
lag in samples.

**`Lag units for '...' data are ambiguous without a sample_rate`**
The lag is read as a number of samples. A `sample_rate` puts it in seconds.

**`MI never dropped below ...% of baseline`**
No precision can be reported because `mode='precision'` never reached its
threshold. Extend `tau_grid` upward.

**`mode='precision': MI goes negative from tau=... onward`**
Past its threshold the mode evaluates a critic trained on clean data against
corrupted inputs on which the bound is unbounded below. The depth of the fall
measures how far the bound has broken. The message reports it as a multiple of
the baseline. Those points are reported as 0 with their measured values kept in
`mi_raw`. The crossing of the threshold comes before them and is unaffected.

**`bias=False was requested and ... carries an input-independent additive term`**
A positional encoding or the biases of a pretrained backbone mean an all-zero
input does not embed to zero even with the encoder's own bias terms removed.

**``bias=False applies to the embedding layers. ....__init__ does not accept a `bias` argument``**
A custom encoder that cannot receive the setting keeps the bias terms in its
layers. Accept `bias` in its `__init__` and pass it to the layers.

**`No tasks to run. Your sweep_grid might be empty`**
The grid produced no tasks.

**`FFMpeg was not found and the animation is written with PillowWriter as a GIF`**
Install ffmpeg to write MP4.

**`No significant MI contour found at threshold ...`**
`analyze_mi_heatmap` found no region above its threshold. Lower
`absolute_mi_threshold`.
