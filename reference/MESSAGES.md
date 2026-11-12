# Messages

This page lists every warning that `neural_mi` emits under the text that it
prints. To find an entry, search for a distinctive phrase from the message. In
the entries, `...` stands for the values that change from call to call. Each
entry says what the message means, why it fired and what to do about it. The
ceiling, amplification and coverage warnings change how a result should be
reported.

A warning means that the call or its result needs your attention. It reports a
result that may be off, a setting that was not used as given or data that the
library had to change. Every warning is raised through Python's `warnings`
module, so a filter can silence a category or turn it into an error. Each kind of
warning is shown once per call, notebook cell or `nmi.grouped_warnings()` block.
What the library did as asked is logged at the info level instead and listed at
the end, under [Informational messages](#informational-messages).

## Contents

- [Ceiling and capacity](#ceiling-and-capacity)
- [Amplification](#amplification)
- [Training](#training)
- [Permutation tests](#permutation-tests)
- [Windowing and coverage](#windowing-and-coverage)
- [Bias correction](#bias-correction)
- [Preprocessing and the data check](#preprocessing-and-the-data-check)
- [Data shape and type](#data-shape-and-type)
- [Dimensionality](#dimensionality)
- [Settings that had no effect](#settings-that-had-no-effect)
- [Performance](#performance)
- [Other](#other)
- [Informational messages](#informational-messages)

---

## Ceiling and capacity

The estimators are lower bounds with a ceiling at the logarithm of the number of
samples on which they are evaluated. A value near its ceiling may have been
limited by the bound rather than by the data.

**`The held-out MI (...) is near its ceiling of log(...)`**
The held-out value of the network at the chosen epoch is above
`Training(ceiling_fraction=...)` of $\log n$, where $n$ is the number of held-out
rows that each evaluation scores. InfoNCE cannot give more than this ceiling, so
the true value may be higher. A curve that flattens here has probably flattened
against the bound. When the split has more held-out rows than one evaluation
scores, raising `max_eval_samples` (and `test_subset_size`, if set) puts more of
them into each evaluation. Otherwise only more held-out rows raise the ceiling,
either from more data or from a lower `Split(train_fraction=...)`.
`estimator='smile'` has no $\log n$ ceiling but has a higher variance. With `peak_fraction` below 1, the warning
judges the network at the chosen epoch (the conservative one), while `test_mi`
and `test_saturation` describe the network at `best_epoch`.

**`The estimate (...) is near its ceiling of log(...)`**
The estimate is above `Training(ceiling_fraction=...)` of $\log m$, where $m$ is
the number of training rows that each evaluation scores. The held-out value of
the same network, however, is not near its own ceiling. A network can come this
close to the ceiling on its training rows by memorising each row as well as by
learning the information, so the estimate may lie far above the true value. The
held-out value is a lower bound in expectation. `result.plot()` shows the
held-out curve. `Training(eval_train=True)` adds the training curve. The call then
does not also warn that the held-out value falls short of the estimate
([Training](#training)), because this message already states both values.

**`Peak-epoch selection may be unreliable`**
More than `Training(saturated_epoch_fraction=...)` of the epochs of the smoothed
held-out curve lie above `Training(saturation_fraction=...)` of its ceiling. With
so much of the curve near its ceiling, the epoch with the largest value and the
value reported there are close to arbitrary. The top of the curve may then be the
ceiling rather than the information. Raise `max_eval_samples` so that the curve
has room to separate, or set `peak_fraction` below 1 to take the value at an
earlier epoch, before the curve reaches its top.

**`Rigorous fit uses the saturated gamma=...`**
The deep rungs of the ladder have the smallest chunks and the lowest ceilings.
When they are saturated, the fitted slope and the extrapolated intercept reflect
the ceiling as much as the bias. Raise `max_eval_samples`, or narrow
`gamma_range` to leave out the deep end while keeping at least
`min_gamma_points` values.

**`Fit for ... marked unreliable: gamma=... used in the fit are ceiling-saturated`**
The fit reports the same condition afterwards and sets `is_reliable` to `False`.

**`Dimensionality: the reference fits' MI (...) is near its evaluation ceiling`**
The ceiling is the log of the held-out rows that one evaluation scores. When it
caps the plateau, small embeddings reach the threshold early.
The reported dimension can then fall below the dimension that it is meant to
bound. Raise `max_eval_samples`.

---

## Amplification

A quantity formed as a difference of estimates inherits their errors, even though
it is much smaller than either of them ([THEORY.md](THEORY.md#amplification)).
The warnings in this section share the `UserWarning` subclass
`nmi.CombinationWarning`, so that `warnings.filterwarnings` can silence them as a
group. With `estimator='ksg'` the terms are not separate estimates
([THEORY.md](THEORY.md#the-conditional-form)), so only a negative estimate is
reported.

**`... has an error-amplification factor of ...x, which reaches Training(amplification_threshold=...)`**
The result is a small residual of much larger estimates (two for a conditional
quantity, three for interaction information). If the errors of the components
were independent, a relative error on each component would become roughly the
factor times that error on the result. The components are trained on the same
rows, however, so their errors partly move together and some of this error
cancels. Report the components beside the point estimate, and look at the spread
over several repeats before drawing a conclusion.

**`... estimate is negative (...). The quantity cannot be negative`**
The quantity is reported as 0 with its measured value kept in `mi_raw` (or
`te_yx_raw`). The joint term came out below the term that it contains, although
the true values cannot be in that order. A high amplification factor means that
the two terms are close and that the noise of the components flipped their order
around a true value near zero. Without more repeats
(`sweep_grid={'run_id': range(n)}`) or more data, the quantity is not resolved
at this sample size. At a low factor, the gap is larger than noise can explain,
because one network fell short. That network is usually the joint one. It needs
longer training or more capacity. With `estimator='ksg'`, the result is one
conditional estimate. A negative value is then a true value near zero, measured
with the bias and the spread of the estimator.

**`... repeats produced nothing (reported as 0) and are left out of mi_mean and mi_std`**, **`... rows of dataframe hold ... of their ... repeats that produced nothing`**
A repeat of a single network is reported as 0 when the network learned nothing
that generalises or when its value came out negative. `mi_mean` and `mi_std` then
average the other repeats, while `n_zero` counts the zeros in each row. A row
whose repeats all produced nothing reports 0. A repeat of a difference quantity or
a rigorous fit is a measurement whatever its sign. Its row averages the measured
values of all repeats and leaves out only a repeat that has no value (the next
entry). Among clearly positive repeats, a zero is most likely a failed run. When
every repeat is near zero, the quantity itself may be near zero. `runs` keeps
every repeat with its measured value.

**`... of ... repeats are left out of mi_mean and mi_std because a term read negative on its training rows`**
A quantity combined from separately trained networks (conditional MI, transfer
entropy and the quantities built from it, interaction information, `mi_rate`)
takes the value of each term repeat by repeat. In the repeats counted, the
network of a term gave a negative value on its training rows while its held-out
value lay clearly above zero (the entry
[`The train MI at the reported epoch is negative (...) while the test MI (...) stands more than ... standard errors above zero`](#training)).
The two values of one network contradict each other, so the term has no value and
neither does its repeat. The result and every term of it are then NaN in `runs`,
`failed_term` names the term, `n_failed` counts the repeat in its row, and
`mi_mean` and `mi_std` average the other repeats. Treating the term as 0 instead
would report the other terms alone, as though the term were zero. The usual
causes are too few epochs or too few samples. Data far from unit scale can also
cause it when the encoders do not scale their input. The values of each network
are in `details[config_id]['trainings']`.

**`...: a term learned nothing distinguishable from zero and enters as 0`**
In the repeats counted, the network of a term learned nothing distinguishable
from zero. Either its held-out value never rose above zero, or its training value
came out negative while its held-out value lay within noise of zero. The term
enters the combination as 0. This is the right value for a term that is zero,
such as $I(X;Y)$ in a purely synergistic interaction or the target's own memory
in transfer entropy to a memoryless target. A term that is not zero but that its
network missed looks the same. The result is then off by the value of the term:
too high for a term that is subtracted and too low for one that is added. Nothing
else reveals this, because a difference with one term at 0 has an amplification
factor of 1. Where the term should not be zero, estimate it on its own with more
samples, or with more epochs when its training ran to `n_epochs`.

**`The extrapolated ... is negative`**
The rigorous fit extrapolated below zero. The value is reported as 0 with the
extrapolated value kept as `mi_raw`. The interval in the message usually includes
zero, so the quantity is not resolved at this sample size.

**`The components of ... (...) are in an impossible order`**
A negative interaction information is a valid result, because the quantity is
signed ([THEORY.md](THEORY.md#interaction-information)). Its joint term
$I(X,W;Y)$ contains both $I(X;Y)$ and $I(W;Y)$, but here it came out below one of
them. The message interprets this gap as the entry on a negative estimate does.

---

## Training

**`batch_size=... exceeds the ... training samples available`**
The batch has been capped at the size of the training set. The batch size sets
how many negatives each training step sees and does not cap the reported value.
Larger batches need more training samples, for example from a longer recording,
a smaller `window_size` or a smaller `step_size`.

**`train_subset_size=... exceeds the number of available training samples`**
The subset is clamped to the size of the training set. Every sample in it is
evaluated, in blocks of at most `max_eval_samples` when there are more samples
than that.

**`test_subset_size=... exceeds the number of held-out samples`**
The subset is clamped to the size of the held-out set. Every sample in it is
evaluated at every epoch, in blocks of at most `max_eval_samples` when there are
more samples than that.

**`Only ... reach the estimator, which is fewer than Training(small_sample=...)`**
Whether these few samples or windows are enough depends on the data. The number
of samples that an estimate needs grows roughly as $N \sim d^2/I$
([tutorial 02](https://eslam-abdelaleem.github.io/NeuralMI/tutorials/02_Your_First_Number.html),
section 2). Neither the number $d$ of latent dimensions that carry the shared
information nor the information $I$ itself is known before estimating.
`mode='rigorous'` tests whether the estimate still changes with the sample size
([THEORY.md](THEORY.md#finite-sampling-bias)). The message also lists the
regularisation that the call does not use yet and suggests smaller networks,
giving their current sizes.

**`The held-out MI was still rising when training reached n_epochs=...`**
Training used every epoch that it was allowed, and the smoothed held-out curve
rose by more than `Training(rising_noise=...)` times its own noise over the last
`Training(rising_epochs=...)` epochs. The reported value may then come from a
network that has not finished training. Raise `n_epochs`. `patience` still ends
training once the curve stops improving. A higher `learning_rate` instead gets
further in each epoch. Raised too far, it moves the top into the first few
epochs. The message about a top reached at epoch ... and gone soon after then
reports this. Here and in the three entries below, the message ends with the
number of held-out rows when there are fewer of them than one training batch (as
in the line of the data check on the rows, `Data check: ... reach the estimator, ...`).

**`The held-out MI curve does not rise clearly above its own noise`**
The top of the smoothed held-out curve lies less than
`Training(resolved_noise=...)` times the epoch-to-epoch noise of the curve above
zero, or less than `Training(clear_se=...)` standard errors of its held-out rows
($\sqrt{2I/N}$ for a top of $I$ nats on $N$ rows). The epoch-to-epoch noise is
measured from the curve itself, from its changes between epochs after any smooth
trend is removed. Every epoch scores the same held-out rows, so their sampling
error moves the whole curve together and never shows in that noise. A network
that learned nothing can therefore lie flat a little above zero. The data then
determine neither the epoch at which the estimate is taken nor the estimate. The
message ends with the estimate and the held-out value of the network at the
chosen epoch. Only the held-out value is measured on rows that the network did
not train on. For an estimate reported as 0 after a negative value, the message
also gives the measured value. The message suggests more epochs only when
training ran to `n_epochs`. When training stopped earlier because the curve had
stopped improving, more epochs would not change the curve, since the epochs after
its top did not raise it. A curve whose values are all non-positive gets the
message below instead.

**`The held-out MI reached its top at epoch ... and fell clearly soon after`**
The earliest epoch at the top of the smoothed curve lies within the reach of the
smoothing from the start, and the curve then falls by more than
`Training(fall_noise=...)` noise units within `Training(fast_epochs=...)` epochs.
The message gives this reach in epochs. `median_window` and `smoothing_sigma` set
it. Before the curve begins, the smoothing holds it at its first value, so the
start of the curve determined that epoch. A lower `learning_rate` spreads the
rise over more epochs.

**`The held-out MI stays within Training(top_se=...) standard errors of its top from epoch ... to epoch ...`**
The *band* is the set of epochs whose smoothed held-out value lies within
`Training(top_se=...)` standard errors of the maximum of the curve
($\sqrt{2I/N}$ for $I$ nats on $N$ held-out rows), so the held-out rows cannot
tell these epochs apart. The trainer evaluates the estimate at the first and the
last epoch of the band, on the same rows as the estimate. The message fires when
these two values differ by more than `Training(band_share=...)` of the estimate.
The estimate at the epoch that the rule chooses is then one of several values
that the held-out rows support equally well. Another configuration may determine
the estimate more precisely, or the data may not allow a finer estimate. A
reported value at or below zero is left out, because the entries on negative
values cover its warnings. `runs` keeps the epochs of the band as
`band_first_epoch` and `band_last_epoch` and its two values as `band_low` and
`band_high`. `result.plot()` shows the curve with the chosen epoch and the peak.

**`At the epoch read, the held-out MI of the network (...) is less than Training(gap_share=...) of the estimate`**
The estimate comes from the rows that the network trained on. The held-out value
of the same network comes from rows that it did not train on. Their ratio has no
units and says how much of the estimate the held-out rows confirm. The call warns
when the held-out value is below `Training(gap_share=...)` of a positive estimate
and the difference is more than `Training(clear_se=...)` of its standard errors.
The second condition keeps a small estimate on few rows from warning because of
noise alone. The estimate exceeds the information when the network fits its
training rows beyond what generalises to new rows. The rest of the estimate may
also be information that the network has not yet learned to generalise. The run
cannot show which of the two applies. The message therefore says that the
held-out rows do not confirm the estimate, rather than that the true value is
lower. More rows narrow the difference. The check is skipped in two cases whose
own messages already state both values: a curve that does not rise clearly above
its noise and an estimate near the ceiling of the training values.

**`All test MI values in the training history are non-positive`**
The network learned nothing that generalises, so the reported value is 0. The
training value would be pure overfitting and is kept as `raw_train_mi`. Either
there is no dependence to find or training failed. Check the learning rate before
you check the data. In a deliberate null, this is the expected outcome.

**`The train MI at the reported epoch is negative (...) while the test MI (...) stands more than ... standard errors above zero`**
The training value at the reported epoch came out negative. The held-out value of
the network, however, lay more than `Training(clear_se=...)` of its standard
errors above zero ($\sqrt{2I/N}$ for a held-out value of $I$ nats on $N$ rows).
MI cannot be negative, so the reported value is 0 and the measured value is kept
as `raw_train_mi`. The two values of the network contradict each other, so a
quantity combined from several networks has no value in that repeat (see the
entry in [Amplification](#amplification) on repeats left out). The usual causes
are too few epochs or too few samples. Data far from unit scale can also cause it
when the encoders do not scale their input. The message names too few epochs only
when training ran to `n_epochs`.

**`The train MI at the reported epoch is negative (...), and the test MI (...) is within ... standard errors of zero`**
Both values of the network lie within noise of zero. The training value is
negative, and the held-out value lies less than `Training(clear_se=...)` standard
errors above zero. The network learned nothing distinguishable from zero, as it
would for a dependence of zero. The reported value is 0 with the measured value
kept as `raw_train_mi`. In a quantity combined from several networks, the term
enters as 0.

**`NaN MI detected at epoch ... (consecutive NaN streak: ... of Training(nan_streak=...))`**
An epoch whose evaluation produced NaN is skipped for early stopping.
`Training(nan_streak=...)` such epochs in a row raise `TrainingError`. The usual
causes are a learning rate that is too high or a degenerate batch.

**`MI evaluation returned NaN`**, **`Score matrix contains NaN values during evaluation`**
The critic became numerically unstable. Check `learning_rate`, `batch_size` and
whether any channel has zero variance.

**`Score matrix contains Inf values. Clamping`**
The scores overflowed and were clamped so that training can continue. If this
keeps happening, the critic is diverging. `Model(norm_layer='layer')` may help.

**`...=... weighs reconstruction at least as heavily as the mutual information it is there to regularise`**
The reconstruction weights are measured against the MI term, so a weight of 1
trades a unit of reconstruction error for a unit of shared information. A larger
weight lets reconstruction dominate the embedding. The default is 0.001. With a
variational encoder, the effective weight of a reconstruction is `beta *
lambda`.

---

## Permutation tests

**`With n_permutations=... the smallest p-value the permutation test can report is 1/...`**
With $n$ trials, no p-value can fall below $1/(n+1)$. The call warns when $n$ is
below `min_permutations`. The message also states what the test costs.

**`Permutation trial failed`**, **`All ... permutation trials for mode='...' failed and left no null distribution`**
One trial or every trial raised an exception. Without a null distribution, the
test is inconclusive. The log shows each failure.

---

## Windowing and coverage

**`shift_windows=True has no effect here: the rows start one step apart`**
At `stride=1`, the rows of a quantity built by time offsets cover every step of
the series, so a shift would only relabel them. The rows shift when a stride
above one thins them out. They are then built at every step, and each epoch keeps
one row in every `stride` from a new phase.

**`At ...=0, ...() has nothing to condition on and computes ...`**
A conditional quantity built by time offsets was given an empty conditioning
history. It then has no difference to form and computes another quantity with
one network. `instantaneous_exchange(k=0)` and `directed_information_rate(k=0)`
compute `instantaneous_mi` ($I(X_0; Y_0)$). `mi_rate(h=0)` computes
$I(X_{-L..L}; Y_0)$ with $L$ = `half_width`. At `half_width=0` this is
`instantaneous_mi`. In a sweep that starts at 0, the first point is this
quantity. It is the baseline against which the conditioned values are compared.

**`step_size=... was read as a fraction of window_size=...`**
A `step_size` below 1 is always a fraction of the window, so `step_size=0.25` is
75% overlap at any window size. In seconds, the rule also applies to ordinary
values. `window_size=0.5, step_size=0.5` then gives a step of 0.25 s. The message
names the step applied and says how to get the other meaning: `step_size=None`
for touching windows or `step_size=a/window_size` for an absolute step `a`. It
fires only when `window_size` is below 1.

**`Window coverage validation kept ... of ... windows`**
Windows were dropped because the stream that the message names had no valid data
in them. The estimate describes only the retained windows and is in bits per
retained window. Report the retention beside any estimate on real data. For spike
data, `{'drop_empty_windows': False}` in the processor parameters keeps silent
windows and estimates the unrestricted quantity
([USING.md](USING.md#silent-spike-windows)).

**`Blocked split may leak raw samples between train and test`**
The gap around each test block is shorter than the stretch that one row covers,
so rows on either side of a boundary share raw samples. The rows can be the
windows of a processor, the rows that a named quantity cuts by offset or the
histories of transfer entropy. Each kind records how far apart its rows start and
how much one row covers. Raise `Split(gap_fraction=...)` to the value that the
message gives.

**`Random split of rows that share samples`**
`Split(mode='random')` drew the held-out rows one by one from rows that overlap
in time, so each held-out row lies among training rows that share most of its
samples. An encoder that responds to a pattern wherever it lies in a row treats
those rows as copies of the held-out one. Recurrent encoders and CNNs that average
over positions behave this way. Their held-out curve then cannot separate fitting
the training rows from learning the dependence. The curve keeps rising as the
network overfits, the library picks a late epoch, and the value there is too
high. An encoder that takes each position of a row on its own treats a row
shifted by one step as a different input and is less exposed. The default MLP is
such an encoder. Use the default `Split(mode='blocked')`. It holds out contiguous
blocks with a gap around each block.

**`Blocked split parameters produced an invalid configuration`**
The requested blocks do not fit the dataset, so the split fell back to a random
split. A random split leaks between neighbouring windows of a time series. Reduce
`n_test_blocks` or use more data.

**`ContinuousWindowDataset: .../... interpolated time points (...) are zero-padded due to data gaps`**
More than the share `padded_share` of the interpolated points fall outside the
recorded range, so they were filled with zeros. Check the time vector against
the data.

**`ContinuousWindowDataset: .../... retained window(s) have more than bridged_share=... of their samples bridged by interpolation`**
Windows that passed the coverage check and entered the estimate are still largely
interpolated across a gap. A gap is an interval between samples that is longer
than `gap_multiple` median sample periods. The processor parameter
`bridged_share` sets how much of a window may lie across gaps before the call
warns. Raise `min_coverage_fraction` to drop such windows.

**`SpikeWindowDataset: max_spikes_per_window cap applied`**
Spikes beyond the cap are dropped in each window. Shorten the window or raise the
cap if the dropped spikes matter.

**`Spike tensor allocation dominated by burst`**
The peak count of one neuron sets the width of the tensor for every neuron, so the
unused width costs memory. `exclude_bursty_neurons=True` leaves such neurons out.

---

## Bias correction

**`gamma=...: the smallest data subset has ... samples. Below about ... the held-out partition of a chunk cannot fill one evaluation batch`**
Noise can dominate the values of the deep rungs, because their chunks are too
small to evaluate reliably. The fit leaves out the rungs whose held-out curves do
not resolve (see the entry on rungs that did not resolve, below). The rungs that
do resolve can still be noisy. A straight line through noisy rungs still looks
straight. Use a smaller `gamma_range` or more data.
`Training(min_reliable_samples=...)` moves the point at which this message fires.

**`... of ... rungs of the gamma ladder produced nothing`**
Either the networks behind those rungs learned nothing that generalises (the rung
is 0), or a term of a difference failed on its chunk (the rung is NaN, as in the
[Amplification](#amplification) entry on repeats left out). Otherwise a rung of a
difference quantity keeps its signed measured value, even a negative one. The fit
leaves these rungs out like failed chunks and counts them in the `zero_rungs`
column of `runs`. The message gives the count at each $\gamma$.

**`... of ... rungs of the gamma ladder did not resolve`**
The held-out curves of the networks behind those rungs do not rise clearly above
their own noise (the entry on such curves above), so their values say nothing
about the dependence. If they entered the fit, they would pass its checks,
because noise hides curvature. The fit therefore leaves them out and counts them
in the `unresolved_rungs` column of `runs`. A rung of a difference quantity is
left out when its joint term does not resolve. The joint term contains every
variable that the other terms contain, while a smaller term can come out as
nothing because it is zero. The message gives the count at each $\gamma$.

**`...at gamma=... the rungs left out make up left_out_share=... or more of the rungs, so the fit leaves out those values of gamma as well`**
The rungs left out at these values of $\gamma$ produced nothing or did not
resolve. The chunks there are likely too small to estimate the quantity. The rungs
that remain come from the networks that happened to learn, so they would tilt
the line. The fit therefore leaves out these values of $\gamma$ with all their
rungs. With fewer than `min_gamma_points` values of $\gamma$ left, the fit is
unreliable. Use a smaller `gamma_range` or more data.

**`...every rung of the gamma ladder produced nothing`**, **`...no value of gamma is left`**, **`...only gamma=... has rungs left`**
A line needs two values of $\gamma$. The fit reports 0 when every rung produced
nothing, because then no network learned anything at any chunk size. It reports
NaN when no value of $\gamma$ is left for another reason, because nothing then
shows that the dependence is 0. It also reports NaN when only one value of
$\gamma$ is left. Either way, `is_reliable` is `False`. Use a smaller
`gamma_range`, more data or more training. With no rung left, the data may
contain no dependence at all. `permutation_test=True` on a single estimate tests
this.

**`Fit for ... is unreliable (final gamma points < ...)`**
`is_reliable` is `False` because too few rungs remain to fit. Widen
`gamma_range`, or lower `min_gamma_points` and accept a weaker fit.

**`Fit for ... is unreliable. No linear region was found`**
Enough rungs remain, but they never met the curvature criterion. The fit uses them
without the linearity that it assumes, so its intercept cannot be trusted.

**`Fit diagnostics triggered for ...: leverage_warning=...`**
Leaving out the rungs at $\gamma = 1$ moves the intercept by more than
`leverage_threshold`, because the extrapolation depends heavily on the point from
the full data. `is_reliable` is `False`.

**`Rigorous analysis expected ... tasks`**
The ladder has fewer trainings than requested. The missing rungs are not a random
subset, so find out why before you trust the extrapolation.

**`Rigorous fit (rigorous=True): the linear region is too small after pruning`**, **`Rigorous fit (rigorous=True): no linear region was found`**, **`Rigorous fit (rigorous=True): fit diagnostics triggered`**
These messages report the same conditions for `rigorous=True` on `Conditional`,
`Interaction`, `Transfer` and the named quantities. In these cases the fit uses
the combined values.

**`Rigorous fit (rigorous=True): the estimate for one chunk failed`**
The estimate of one chunk raised the error that the message gives. The fit goes
on without that chunk as long as at least `min_gamma_points` values remain.

---

## Preprocessing and the data check

`Preprocessing(channel=..., time=...)` transforms each continuous stream before
rows are built from it. Every call also checks its streams first
([USING.md](USING.md#preprocessing-the-streams)). The check changes nothing and
warns about anything that needs attention. Its report is logged at the info level
(`Data check for ...`, under [Informational messages](#informational-messages)).

**`... has non-finite values (NaN or infinite) in channel(s) ...`**
Every network that receives a NaN or an infinite value produces NaN, so the call
fails or reports nothing. A stream that a continuous or categorical processor
handles is refused before this point, so this message catches arrays passed in
already processed. Remove or fill the values before the call.

**`... has constant channel(s) ...`**
A constant channel carries no information. The transforms leave it unscaled. Drop
it before the call, especially before `channel='zca'`, because ZCA has no
direction along which to sphere it.

**`The channels of ... have a typical scale of ..., more than a factor of ... from unit scale`**, **`The channels of ... have a typical mean of ..., more than ... from zero`**
The encoders leave this stream as it is, as they do under
`Model(input_scaling='none')` and under a custom critic that does not take the
setting. The median standard deviation of its channels lies more than a factor of
`Preprocessing(scale_band=...)` from 1 in either direction, or their median mean
lies further than that from zero. The networks start from weights sized for
inputs near zero and of unit scale, so their estimate depends on units. The same
dependence can give a value close to its true value at unit scale and close to 0
far from it. The default `Model(input_scaling='auto')` centres and scales each
channel of a continuous stream inside the network.
`Preprocessing(channel='zscore')` does the same before the rows are built.
Neither changes any quantity. The check does not fire when a channel transform is
set.

**`The standard deviations of the channels of ... span a factor of ..., more than Preprocessing(spread_band=...)`**
When the encoders leave the channels as they are (`Model(input_scaling='none')`),
the loudest channels dominate what the networks receive. The estimate can then
miss information that the quiet channels carry. The default
`Model(input_scaling='auto')` brings each channel of a continuous stream to unit
scale and so gives every channel the same weight. This is right when the scales
differ only in units, as for variables recorded in different units. A channel can
also be quiet because it carries little, like a neuron that rarely fires or an
electrode that picks up only noise. Scaling then raises its noise to the level of
the signal, so the network needs more rows to learn to ignore it. The two cases
look the same to the check, so the message describes both. When the encoders
scale the stream, the message names `Model(input_scaling='none')` as the setting
that keeps the relative scales of the channels. Dividing the stream by one number
before the call (such as its overall standard deviation) then brings it near unit
scale.

**`... of ... sit more than Preprocessing(offset_band=...) of their standard deviations from zero`**
The networks process their rows in single precision (about seven significant
digits). For a channel whose mean is many standard deviations from zero, few of
these digits are left for its variation, although the variation is where the
information is. No scaling inside a network can recover them. Subtract the mean
of each channel before the call, while the data are still in double precision,
or set `Preprocessing(channel='zscore')` to centre each channel before the rows
are built. Data stored in single precision with such an offset have already lost
these digits.

**`In ... of ..., more than Preprocessing(tie_share=...) of the values are one repeated value`**
Some channels of a stream that the encoder scales channel by channel consist
mostly of one value. Padded or sparse rows look like this, for example spike rows
built by hand, sparse counts or the padding of a window with few events. Scaling
moves the repeated value away from 0, where an encoder may take it for an event.
Scaling also raises a channel that is quiet because it carries little to the
level of the others. Rows processed by a spike processor (`Processing(x='spike')`
and its kind) are instead divided by one number for the whole stream, so the
padding stays at 0 and quiet neurons stay quiet. `Model(input_scaling='none')`
leaves the rows as they are.

**`... of ... have heavy tails`**
In these channels, the standard deviation exceeds `Preprocessing(tail_band=...)`
interquartile ranges, as it does for lognormal values. The extreme values set the
standard deviation, so dividing by it squeezes most values of such a channel
close together. `Preprocessing(channel='copula')` maps each channel through its
ranks to a normal distribution before the rows are built. This leaves every
quantity unchanged. The entry above covers channels that are mostly one repeated
value.

**`The rows ...'s ... processor builds have a typical scale of ..., more than a factor of ... from unit scale`**
The rows of a spike stream come from its processor, after any preprocessing. Here
the encoders leave them as they are (`Model(input_scaling='none')`). The typical
standard deviation of the rows lies more than a factor of
`Preprocessing(scale_band=...)` from 1 in either direction. Spike times within a
window are small numbers in seconds. Binned counts divided by the bin width
(`normalize_bins`) are rates in the hundreds or thousands at fine bins. At such
scales a clear dependence can come out as 0. The default
`Model(input_scaling='auto')` divides the rows by one number for the whole
stream.

**`Preprocessing(time=...) filters each channel by its own past, which changes what ... measures`**
Filtering each process invertibly by its own past leaves the rate quantities
unchanged in the limit of long windows (`mi_rate`, `transfer_entropy`,
`conditional_transfer_entropy`, `directed_information_rate`,
`instantaneous_exchange`). It changes every other quantity, because the
information between the whitened streams is not the information between the
original ones. Use it with another quantity only when that is the quantity that
you want.

**`Preprocessing leaves ..., a ... stream, as it is`**
A spike-time or categorical stream has no continuous values to transform. Its
rows come from its processor. The settings of the processor determine their
values (`bin_size` and `normalize_bins` for spikes, `encoding` for categories).
For a spike stream, the default `Model(input_scaling='auto')` divides the rows
that an encoder receives by one number for the whole stream.

**`Preprocessing(time=...) leaves ... as it is in time`**
The stream came as rows that were already built (three or more dimensions), so
there is no single series to filter. It still takes the channel transform. Pass
the series instead and let a processor or a named quantity build the rows.

**`Preprocessing applies ..., a transform of your own`**
The library cannot check what a transform of your own does to the information in
a stream. An invertible transform keeps the information, but one that drops,
merges or clips values does not. The built-in transforms are invertible, apart
from `PCA` with fewer components than channels.

**`PCA keeps ... of the ... components of ..., those of largest variance`**
The kept components are the directions of largest variance, but the information
need not lie in them. A shared signal that lies under larger private ones, as in
field potentials, is cut first. `n_components=None` keeps every component. The
transform is then a rotation and leaves every quantity unchanged.

---

## Data shape and type

**`ContinuousWindowDataset: first inter-sample interval (...) differs from the median`**
The first interval of the time vector differs from the median interval by more
than the processor's `period_tolerance` of the median. The library uses the
median period. Check the start of the recording.

**`processor_type mixes 'spike' (...) with '...' on stream ...`**
Spike times are in seconds, while a continuous or categorical stream without a
clock or a `sample_rate` is in samples. The grid would therefore line up windows
that cover different times. The message names the stream. Give it a
`sample_rate` or a time vector.

**`Unrecognised augmentation key(s) ... are ignored`**
Training looks normal, but it never applies an augmentation whose key is not in
the vocabulary. The message lists the valid keys. The cause is usually a typo.

**`embedding_model='...' received 4-D input`**
The encoder flattens the spatial axes. Use `cnn2d` or `pretrained_backbone` for
image-shaped input.

**`Spatial augmentations ... require 4-D input`**
The image augmentations are skipped because the batch is not image-shaped.

**`KSG assumes continuous values, and ... of ...'s values... tie with another row's`**
KSG measures the information from the distances between neighbouring rows. Two
rows that tie lie at distance zero. KSG breaks ties with noise drawn separately
for each variable, so a tied group comes out of the ranks in a different order in
each variable. A variable and an exact copy of it then no longer look alike. A
conditional estimate given that copy comes out away from zero where the true
value is 0. The empty slots of spike windows, spike counts and categories tie, so
use the trained estimators on them. X, Y and Z are the variables of the estimate
as KSG sees it, with Z the conditioning set of a conditional estimate. The message
names the column whose values tie most. KSG warns when the share of tied rows in
that column exceeds `Estimator(params={'max_tied_share': ...})`.

---

## Dimensionality

**`mode='dimensionality' with critic_type='separable'`**
A dot-product critic scores a pair by the inner product of its two embeddings. A
nonlinear dependence then needs more dimensions than the shared structure has, so
the reported dimension is too high. The default of the mode is the hybrid critic.

**`Dimensionality: the curve of split(s) ... reached ... of its plateau only at the reference embedding_dim=...`**
No fitted value below the reference reached the threshold, so the reported
dimension equals the reference and bounds nothing below it. Training may have
stopped short (raise `n_epochs`), the dimension may lie between the last grid
value and the reference (pass `embedding_dims` closer to it), or the reference
may be too small (pass a larger `reference_dim`).

**`Dimensionality: the views of split(s) ... share no information the estimator can find`**
Every fit of the split ended at 0. With a plateau of 0, the split has no dimension
to report and is left out of the median. Check that the two views are aligned and
that training ran long enough to learn anything.

**`Dimensionality: the restarts at embedding_dim ... below the reported dimension differ by more than ...`**
A restart can settle with fewer directions than its embedding allows and give a
value that is too low. The curve uses the best restart at each value. When the
restarts disagree this much below the reported dimension, the best one may also
have settled short, so the reported dimension may be too high. Raise
`n_restarts`.

**`Dimensionality: for split(s) ..., the best restart at the reported dimension reaches ... of the plateau, but at one of the next ... values of embedding_dim the best restart falls below that share again`**
The curve is the running maximum of the best restarts, so a single restart that
comes out too high at a small embedding size sets the reported dimension by
itself. If the dimension is real, the values after it stay at the threshold. When
one of the next `Dimensionality(confirm_steps=...)` values falls below it, the
reported dimension may be too low. Raise `n_restarts` and check the curve with
`result.plot()`.

**`Dimensionality: the reported dimension of split(s) ... is among the last ... values of the grid`**
The grid ends before `Dimensionality(confirm_steps=...)` values in a row have
reached the threshold from the reported dimension on. The values beyond it were
not fitted, so nothing confirms that the curve stays at the threshold. Pass
`embedding_dims` that reach further beyond the reported dimension.

**`Dimensionality: at the reference fit the held-out MI (...) falls short of the training MI of the same networks (...) by more than ...`**
The check compares the median held-out value of the reference networks with
their median training value. A held-out value far below the training value means
that the data have few samples for the information that each dimension carries.
The training value then keeps rising with the embedding size and pushes the
reported dimension up. The reported dimension is still an upper bound. More data
make it tighter.

**`Dimensionality: ... channels divide into only ... different pairs of halves`**
With few channels, `split_method='random'` cannot draw `n_splits` different
divisions of the channels into two halves. The mode runs each possible division
once, because a repeated division would count one split twice in the median and
the spread. This is a warning when you set `n_splits` yourself. With the default
number of splits, the library logs the message at the info level instead.

**`split_method='...' produced unequal halves`**
The two halves differ in size. `shared_encoder=True` needs equal halves, so it is
turned off for the run. This is a warning when you set `shared_encoder` yourself.
When the library turns off its own default for the mode, it logs the message at
the info level instead.

**`... was set for mode='dimensionality'. The mode's two sides are two halves of the same recording`**
An encoder setting for Y (`embedding_model_y`, `custom_embedding_cls_y`,
`hidden_dim_y`, `n_layers_y` or `embedding_dim_y`) was given. The run proceeds,
but an encoder that differs between the halves makes the curve hard to
interpret.

---

## Settings that had no effect

None of these messages reports an error. Each means that a setting had no effect,
so the run did something other than what was asked.

**`Mode config(s) ... were provided for a call with mode='...'`**
Only the config of the active mode is used.

**`w_data was provided but mode='...' does not use it`**
Only `conditional`, `interaction` and `transfer` use a third variable.

**`hidden_dim is a list of length ... and sets the number of hidden layers`**
`n_layers` is ignored.

**`These settings have no effect in this call and are ignored: ...`**
Each listed setting applies only with an encoder, a critic or a switch that this
call does not use, and the message names it. The settings apply as follows:

- `kernel_size` applies to `'cnn'`, `'cnn2d'` and `'tcn'`.
- `bidirectional` applies to `'gru'` and `'lstm'`.
- `nhead` applies to `'transformer'`.
- `pytorch_predefined` and `pretrained` apply to `'pretrained_backbone'`.
- `use_spectral_norm`, `dropout`, `norm_layer` and `bias` apply to the MLP
  encoder and to the MLPs that the library builds around the encoders: the head
  of the hybrid critic, the network of the concat critic, the head of the
  branched encoder of a difference term and the head of the pretrained backbone.
- `dropout` also applies to the LRU, TCN and transformer encoders and to a GRU or
  LSTM of more than one layer.
- Under `critic_type='concat'` no encoder is built, so the encoder settings
  (including those for Y) and `shared_encoder` do not apply.
- The head settings apply to `critic_type='hybrid'`, `beta` applies to
  `use_variational=True` and the decoder settings apply to `use_decoder=True`.
- With `custom_critic`, the recipe builds the critic and takes only the
  architecture settings that it names. The other architecture settings are
  ignored. `use_variational` and `beta` still apply, because the loss of the
  trainer is the KL term that the critic returns minus `beta` times the MI term.
  `use_decoder=True` uses the embeddings that the recipe gives through
  `get_training_embeddings`.
- Of the `Output` settings, `return_rotated_embeddings` needs `return_embeddings`
  or `track_embeddings`, `rotated_embeddings_per_epoch` needs both
  `track_embeddings` and `return_rotated_embeddings`, and
  `return_rotation_matrices` needs `return_rotated_embeddings`.
- `mode='dimensionality'` sets `embedding_dim` itself.
- With `estimator='ksg'` nothing is trained, so every `Model`, `Training` and
  `Split` setting is listed.
- With `train_indices` and `test_indices`, the `Split` settings `mode`,
  `train_fraction`, `n_test_blocks` and `gap_fraction` have no effect.

A setting given explicitly at its default value does not count. A `sweep_grid`
over a setting that has no effect is refused before any work, because every
configuration would train the same networks.

**`shift_windows=True has no effect for mode='...' here`**
The rows of the call cannot move, because no processor windows the streams (the
rows were passed in already built). Shifting moves the rows that a processor cuts
from raw streams, in every mode except transfer entropy
([USING.md](USING.md#window-shifting)).

**`drop_empty_windows=True has no effect on stream ... here`**
A quantity defined by time offsets keeps every window in the rows that it builds,
because dropping one would put two distant windows next to each other and change
what an offset means. Silent stretches stay as silent rows. Gaps in the recording
stay visible to the coverage warnings.

**`panels includes 'embeddings' but this repeat holds no 'embedding_history_x'`**
The animation drops its embedding panel. Set `Output(track_embeddings=...)` before
the run to record the embeddings at every epoch.

---

## Performance

**`n_workers=... has no effect here`**
The call trains one network and has nothing to run in parallel.

**`Running ... sequential tasks with dataset_device='...'`**
Freed tensors can stay in the cache of an accelerator across many tasks. Use
`Training(dataset_device='cpu')` for long sweeps.

**`CategoricalWindowDataset: encoding='full_trajectory' will produce a ...-dimensional input`**
The input width of a window is the number of categories times the number of
samples per window, so it grows quickly with the window. The call warns when the
width exceeds the processor's `max_encoding_width`. Use `'majority_vote'` or
`'probability'` unless the time course within the window matters.

**`track_embeddings=...: storing embeddings for all ... samples at every epoch`**
Storing every sample at every epoch takes a lot of memory. Pass a count to track
only the first samples.

---

## Other

**`... was raised ... times in this ... and shown once`**
A rigorous ladder, a parallel grid or a loop of calls can raise the same warning
or log message many times. The first message was shown when it happened. Later
messages whose first sentence differed only in its numbers were counted rather
than shown, and so were exact repeats. The count covers one call, one notebook
cell or one `nmi.grouped_warnings()` block. The message quotes the start of the
first message. This line comes through the same channel as the message that it
counts. For a warning it is a warning of the same category, so a filter on that
category also covers the count. For a log message it is a log line at the same
level.

**`Lag on pre-processed data with shape .... Axis 0 counts windows`**
The data were already windowed, so a lag of $n$ shifts by $n$ windows and
therefore by $n$ times `step_size` samples. To lag in samples, pass unwindowed
data with a `Processing`.

**`Lag units for '...' data are ambiguous without a sample_rate`**
The lag is taken as a number of samples. With a `sample_rate`, it is in seconds.

**`MI never dropped below ...% of baseline`**
No precision can be reported because `mode='precision'` never reached its
threshold. Extend `tau_grid` upward.

**`mode='precision': MI goes negative from tau=... onward`**
Beyond its threshold, the mode evaluates a critic trained on clean data on
corrupted inputs. On these inputs the bound has no lower limit. The depth of the
fall measures how badly the bound has broken down. The message reports it as a
multiple of the baseline. These points are reported as 0 with their measured
values kept in `mi_raw`. The crossing of the threshold comes before them and is
not affected.

**`bias=False was requested and ... carries an input-independent additive term`**
A positional encoding or the biases of a pretrained backbone keep an all-zero
input from embedding to zero, even when the bias terms of the encoder are
removed.

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

---

## Informational messages

These messages report what the library did. They are logged at the info level and
shown with `verbose=True` or `nmi.set_verbose(True)`. They need no action.


**`Data check: ... reach the estimator, ...`**
This line is the part of the check that needs the rows. It is reported where the
rows are counted and gives three numbers: how many rows reach the estimator, how
many of them are held out (counted as the split counts them) and the typical scale
of the rows of each variable where the rows are built at that point. The held-out
rows are what `train_fraction` leaves, or the caller's `test_indices`. When there
are fewer of them than one training batch, the line says so. The held-out curve
determines the epoch at which the estimate is taken and shows whether the
estimate is resolved. Its rows serve as each other's negatives, so its ceiling is
the logarithm of their count. This ceiling lies below that of the training batch
whose objective the curve follows, and the noise of the curve grows as the rows
become fewer. A small held-out set costs the curve only where the curve then
cannot be interpreted. Every warning about the curve names the count (the entries
on the curve under [Training](#training)). More data or a lower
`Split(train_fraction=...)` gives the curve more rows. The rigorous ladder prints
the same line for its chunks and warns. With `estimator='ksg'`, nothing is held
out.

**`Data check for ...`**
Every call runs this check on its streams after any `Preprocessing` and before
rows are built. For each stream, the check gives the channel count and the range
of the channel scales. For a rate quantity on a series, it also gives the largest
share of the variance of a channel that the channel's own past predicts linearly.
This share comes from the autoregressive fit that `ARWhitening` uses, computed on
the first `Preprocessing(check_rows=...)` rows. A share above
`Preprocessing(predicted_share=...)` means that the conditioning term is large
and the rate is a small residual of it. The line then suggests
`Preprocessing(time='ar')`. This filter removes the term and leaves the rate
unchanged. Spike-time and categorical streams are only named, because their
processors build their rows. A stream whose encoder centres and scales it channel
by channel is marked as such. The line then ends by naming
`Model(input_scaling='none')` as the setting that turns this off.

**`Dimensionality: the grid runs over embedding_dim 1 to ... because the reference fit has a participation ratio of ...`**, **`Dimensionality: the grid runs over ... values of embedding_dim from 1 to ... on a log scale`**
The default grid reaches `Dimensionality(grid_reach=...)` times the participation
ratio of the reference fit, because the ratio can fall below the true dimension.
When that reach is at most `grid_small`, the grid is every value from 1 to
`grid_small` without a message. When it is at most `grid_medium`, the grid is
every value from 1 to `grid_medium`. Beyond that, the grid is `grid_points` values
on a log scale up to the reach. The reported dimension is then one of those
values. Pass `embedding_dims` as a list or a range for a finer grid.

**`Dimensionality: the participation ratio of the reference fit (...) is close to its embedding_dim=...`**
A reference may be too small to hold every shared direction when
`Dimensionality(grid_reach=...)` times its participation ratio reaches
`reference_fill` of its size. It is then refitted once at `refit_reach` times the
ratio. A `reference_dim` that you set is kept as given.

**`Dimensionality: the grid stopped at ... once ... consecutive values reached ...`**
The default grid is fitted `Dimensionality(confirm_steps=...)` values at a time.
That many values in a row at the threshold confirm the reported dimension against
the plateau that the reference fit sets. Each split stops there and skips the
values above, so its fits are the ones that a call with that split alone would
make. Pass `embedding_dims` to fit every value.

**`Dimensionality: the curve of split(s) ... did not reach ... of its plateau by embedding_dim=...`**
The default grid ended below the threshold. For these splits only, it is extended
once with values on a log scale up to one below the reference. A curve that is still short of the
threshold after the extension reports the reference as its dimension and raises
the warning that the curve reached its plateau only at the reference
([Dimensionality](#dimensionality)).

**`CategoricalWindowDataset: input data has dtype ..., not an integer type`**
The labels were mapped to consecutive integer codes in sorted order of the
distinct values. Pass integer codes to choose the mapping yourself.

**`Excluding ... bursty neuron(s)`**
The exclusion is active and leaves the listed neurons out of the estimate.

**`No dedicated decoder exists for embedding_model='...'`**
The reconstruction uses a generic decoder for this encoder.

**`Held-out MI evaluated on all ... rows in ... blocks of at most ... rows at every epoch, averaged`**
The held-out subset (`test_subset_size`) is larger than `max_eval_samples`, the
largest number of rows that one evaluation may use. At every epoch, each block is
evaluated on its own and the values are averaged, weighted by block size. The
precision is then that of all the rows, while the InfoNCE ceiling is that of one
block. Every epoch costs one evaluation per block. Raising `max_eval_samples`
evaluates in fewer, larger blocks, at a memory cost that grows with the square of
the block size.

**`Train-side MI evaluated on all ... rows in ... blocks of at most ... rows, averaged`**
The training subset behind the reported estimate (`train_subset_size`, or the
per-epoch set of `eval_train`) is larger than `max_eval_samples`, the largest
number of rows that one evaluation may use. Each block is evaluated on its own
and the values are averaged, weighted by block size. The precision is then that
of all the rows, while the InfoNCE ceiling is that of one block. Raising
`max_eval_samples` evaluates in fewer, larger blocks, at a memory cost that grows
with the square of the block size.

**`Large first embedding layer detected`**
The first layer has more than `Model(large_first_layer=...)` parameters because
the flattened input is wide. With long windows and few samples, such a layer
often overfits. Reduce `window_size` or `hidden_dim`. An encoder that handles the
window axis itself (`cnn`, `gru`, `tcn`, `lru`) also helps.

**`... spike times are not sorted`**
The library sorts the spike times automatically, so the message is harmless.

**`save_best_model_path: this call trains several networks and saves every one of them`**
The call saves every network under the path with its identifying labels added and
records each path as `model_path`. Every file contains a whole network, so a large
grid or a rigorous ladder can take a lot of disk space. To keep one model, rerun
the configuration that you want on its own with this path
([USING.md](USING.md#saving-the-trained-networks)).

**`PretrainedBackboneEmbedding: input has ... channel(s) but backbone '...' expects ...`**, **`PretrainedBackboneEmbedding: input spatial size (...) does not match`**
A trainable channel adapter or an upsampling layer was inserted so that the
backbone can run. Both add parameters that are not part of the pretrained model.

**`Each permutation reruns the whole call`**
Every null trial repeats the full call with every configuration of the grid, so
the test costs `n_permutations` times as much as the call.

**`SubsetView: window count dropped from ... to ...`**
The windows of the dataset that a `SubsetView` uses were built again (for example
with a new window size). Fewer of them now fall in the time regions of the view.
