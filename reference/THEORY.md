# Theory

This page explains what NeuralMI estimates and why its estimators behave as they
do. [USING.md](USING.md) shows the calls, and [PARAMETERS.md](PARAMETERS.md)
lists every setting.

## Contents

- [Estimating mutual information with a critic](#estimating-mutual-information-with-a-critic):
  [InfoNCE](#infonce), [SMILE](#smile)
- [The reported number](#the-reported-number):
  [finite-sampling bias](#finite-sampling-bias), [correcting it](#correcting-it),
  [the spread over repeats](#the-spread-over-repeats),
  [the permutation null](#the-permutation-null), [amplification](#amplification)
- [Quantities and units](#quantities-and-units):
  [what the number is per](#what-the-number-is-per), [rates](#getting-a-rate),
  [silent windows](#silent-windows),
  [temporal information quantities](#temporal-information-quantities),
  [interaction information](#interaction-information)
- [Cross-run-stable directions of shared structure](#cross-run-stable-directions-of-shared-structure)
- [Spike-timing precision](#spike-timing-precision)
- [Regularised objectives](#regularised-objectives)

---

## Estimating mutual information with a critic

The mutual information between $X$ and $Y$ is the Kullback-Leibler divergence
between their joint distribution and the product of their marginals:

$$
I(X; Y) = \int p(x, y) \log \frac{p(x, y)}{p(x)p(y)} \, dx \, dy
$$

Computing it directly requires the distributions themselves, and for the
high-dimensional, continuous data of neuroscience they are unknown. Binning and
kernel density estimates fail as the dimension grows.

Neural estimators avoid the densities by training a network, the critic $f(x,
y)$, to tell pairs that occurred together, $(x_i, y_i)$, from pairs drawn from
different samples of the same batch, $(x_i, y_j)$. Its scores enter a
variational lower bound on the MI, and training the critic tightens the bound
([Poole et al., 2019](https://proceedings.mlr.press/v97/poole19a.html)). The
bounds differ in how they trade bias, the average distance from the true value,
against variance, the spread over runs.

### InfoNCE

InfoNCE ([van den Oord et al., 2018](https://arxiv.org/abs/1807.03748)) is the
default estimator:

$$
I(X;Y) \ge \mathbb{E}\left[ f(x,y) - \log\left(\frac{1}{N}\sum_{j=1}^N e^{f(x,y_j)}\right) \right]
$$

For each true pair, the critic has to pick the real partner $y$ of $x$ out of
$N$ candidates, a classification problem. Its estimates vary little from run to
run.

The bound can never exceed $\log N$, where $N$ is the number of samples it is
evaluated on. Any distribution-free lower bound on MI from $N$ samples is
limited to order $\log N$
([McAllester and Stratos, 2020](https://proceedings.mlr.press/v108/mcallester20a.html)).
The reported values are evaluated on `train_eval_size` samples on the training
side and `eval_size` on the held-out side, each capped by `max_eval_samples`
and by how many samples the split produced. During training, $N$ is the batch
size, and `batch_size` shapes the gradients without capping the reported value.
A short recording, a large `window_size` or a small held-out fraction all shrink
$N$. Along a sweep over `window_size` the ceiling falls, since fewer windows fit
in the recording. The windowed MI grows with the window, so the estimate
reaches its ceiling from both directions. `runs` reports each side's ceiling in
`train_ceiling_mi` and `test_ceiling_mi`, and the fraction of it the estimate
reached in `train_saturation` and `test_saturation`.

### SMILE

SMILE ([Song and Ermon, 2020](https://arxiv.org/abs/1910.06222)) bounds the MI
through the Donsker-Varadhan representation used by MINE
([Belghazi et al., 2018](https://proceedings.mlr.press/v80/belghazi18a.html)),
with the scores of the unpaired samples clipped at $\tau$:

$$
I(X;Y) \ge \mathbb{E}\left[ f(x,y) \right] - \log \mathbb{E}\left[ e^{\text{clip}(f(x,y'), \tau)} \right]
$$

The normalising term is where MINE's variance comes from, and clipping keeps a
few large scores from dominating it. SMILE has no $\log N$ ceiling and is less
biased when the MI is large, at the price of a higher variance. A clip of
$\tau = 5$ is the default. A lower clip lowers the variance and raises the
bias.

:::{admonition} Choosing
:class: tip

InfoNCE is the general-purpose choice, and SMILE suits a true MI that may be
large compared with the log of the evaluation size. :::

:::{admonition} References
:class: note

- Aaron van den Oord, Yazhe Li and Oriol Vinyals. Representation learning with contrastive predictive coding. arXiv:1807.03748, 2018. [arxiv.org/abs/1807.03748](https://arxiv.org/abs/1807.03748)
- Ben Poole, Sherjil Ozair, Aaron van den Oord, Alex Alemi and George Tucker. On variational bounds of mutual information. *Proceedings of the 36th International Conference on Machine Learning*, PMLR 97:5171-5180, 2019. [proceedings.mlr.press/v97/poole19a](https://proceedings.mlr.press/v97/poole19a.html)
- David McAllester and Karl Stratos. Formal limitations on the measurement of mutual information. *Proceedings of the 23rd International Conference on Artificial Intelligence and Statistics*, PMLR 108:875-884, 2020. [proceedings.mlr.press/v108/mcallester20a](https://proceedings.mlr.press/v108/mcallester20a.html)
- Jiaming Song and Stefano Ermon. Understanding the limitations of variational mutual information estimators. *International Conference on Learning Representations*, 2020. [arxiv.org/abs/1910.06222](https://arxiv.org/abs/1910.06222)
- Mohamed Ishmael Belghazi, Aristide Baratin, Sai Rajeshwar, Sherjil Ozair, Yoshua Bengio, Aaron Courville and R Devon Hjelm. Mutual information neural estimation. *Proceedings of the 35th International Conference on Machine Learning*, PMLR 80, 2018. [proceedings.mlr.press/v80/belghazi18a](https://proceedings.mlr.press/v80/belghazi18a.html)
- Eslam Abdelaleem, K. Michael Martini and Ilya Nemenman. Accurate estimation of mutual information in high dimensional data. arXiv:2506.00330, 2025. [arxiv.org/abs/2506.00330](https://arxiv.org/abs/2506.00330)
:::

---

## The reported number

A training run splits the samples into a training set and a held-out set,
evaluates the MI on the held-out set at every epoch, smooths that curve, and
takes the epoch at its peak. The critic's weights at that epoch are then
evaluated on the training samples, and that value is `mi_estimate`. The
held-out value at the same epoch is `test_mi` in `runs`.

The held-out curve plays the part a bandwidth choice plays in kernel density
estimation. On the training side the critic can fit ever finer structure, and
the held-out side sets the scale of structure the data support. The MI is a
property of the dataset, and once the held-out curve has fixed the epoch, the
training side reads that property on the larger sample ([Abdelaleem et al.,
2025a](https://arxiv.org/abs/2506.00330)). The training side's larger evaluation
set also raises its $\log N$ ceiling and lowers its variance. `mode='rigorous'`
extrapolates the same training-side values, so every mode reports one kind of
number.

The two values at the selected epoch differ. The difference widens as the
data shrink, since a critic of fixed capacity fits a small training set more
easily. The reported value can sit slightly above the true one. Under a blocked
split the difference does not shrink when consecutive windows stop overlapping,
and it stays flat as the correlation time varies over more than an order of
magnitude. The smoothed curve, the selected epoch and the final training-side
evaluation are three separate reads, so exact agreement between the reported
numbers is not expected.

### Finite-sampling bias

Any estimate from $N$ samples carries a bias that depends on $N$. The classical
limited-sampling bias makes finite samples look more dependent than the source
and inflates the estimate. A variational lower bound whose critic learned from
little data undershoots. Which effect dominates depends on the regime, and in
both cases the bias shrinks with the sample size:

$$
I_{\text{estimated}}(N) \approx I_{\text{true}} + \frac{a}{N} + O\left(\frac{1}{N^2}\right)
$$

The estimate is therefore approximately linear in $1/N$
([Strong et al., 1998](https://doi.org/10.1103/PhysRevLett.80.197)), and the
linear trend can be fitted and removed.

### Correcting it

`mode='rigorous'` runs the estimate at several sample sizes and extrapolates
([Holmes and Nemenman, 2019](https://doi.org/10.1103/PhysRevE.100.022404);
[Abdelaleem et al., 2025a](https://arxiv.org/abs/2506.00330)).

1. At $\gamma = 1$ the whole dataset is one chunk. At $\gamma = 2$ it is cut into
   two halves that do not overlap, at $\gamma = 3$ into thirds, and so on, and
   each chunk is estimated on its own.
2. Substituting $N_{\text{chunk}} = N/\gamma$ into the bias formula gives
   $I_{\text{estimated}} \approx I_{\text{true}} + \frac{a}{N}\,\gamma$, so the
   estimate is linear in $\gamma$. A weighted linear regression of the estimates
   against $\gamma$ is fitted.
3. The fit is extrapolated to $\gamma = 0$, where each chunk would hold infinitely
   many samples and $1/N_{\text{chunk}} \to 0$. The intercept is the corrected
   estimate, and its confidence interval gives `mi_error`.

The relation is only approximately linear, since at large $\gamma$ the chunks
are small and finite-sample effects and under-fitted networks bend the curve. A
quadratic in $\gamma$ is fitted, and the largest $\gamma$ values are dropped
until the quadratic term is statistically indistinguishable from zero, $|a_2| /
\mathrm{SE}(a_2) <$ `curvature_t_threshold` (default 2.0, roughly the 5%
two-sided level). The remaining points give the final regression. The search
never trims below `min_gamma_points`, and when it reaches that floor with the
trend still curved, `linear_region_found` is `False`.

A fit is reliable when it used at least `min_gamma_points` values of $\gamma$,
found a linear region, does not move by more than `leverage_threshold` when the
$\gamma = 1$ points are left out, and, in `mode='rigorous'`, uses no $\gamma$
whose estimates sit at their ceiling. The leave-one-out check asks whether the
extrapolation depends on the full-data anchor, a question independent of the
scale of the estimates. $R^2$ and the studentised residuals are reported beside
the fit and decide nothing. Both depend on scale and behave badly under this
weighted fit. With many samples the estimates across $\gamma$ cluster tightly
and $R^2 = 1 - SS_\text{res}/SS_\text{tot}$ collapses with the small total
variance. The rungs at small $\gamma$ have little noise and dominate the mean
squared error, while those at large $\gamma$ are noisier by construction. The
residual ratio $e_i/s$ is therefore large even for a valid fit.

The extrapolation removes the part of the bias that changes with the sample
size. Bias already present at $\gamma = 1$ that is the same at every chunk size
sits in the intercept and passes through. A flat slope says the estimate is
stable over a range of $N$. That is information about the data, and it leaves
open whether the estimate is unbiased. Likewise `mi_error` says how well the
line is determined, and a reliable fit with a small `mi_error` means the part of
the bias that depends on $N$ has been removed and the fit is well determined. It
gives no bound on how far the intercept sits from the truth.

### The spread over repeats

`mi_std` is the standard deviation over repeats of the whole procedure on the
same data, each repeat retraining its networks from a new initialisation. It
measures how much the procedure's output moves. Repeats share their data, so the
spread says nothing about how the estimate would move on a new recording. It
is no interval on the population value. For the same reason the intervals of
rigorous repeats cannot be combined into one.

The spread is the most direct test of whether a value is resolved at all. A
quantity whose spread over repeats exceeds its own value is not distinguishable
from zero at that sample size, however confident its point estimate looks. For
a quantity built as a difference, the difference is formed repeat by repeat,
and its spread is the spread of those differences.

### The permutation null

A permutation test asks whether the observed value could arise with no
dependence between X and Y. Each null trial reruns the whole call with X moved
in time and everything else in place. A circular shift moves X by one random
offset with wrap-around, drawn at least 10% of the recording's length away from
zero in either direction, so no trial leaves X nearly aligned with Y. A block
shuffle cuts X into blocks one window long and reorders them. Either way X keeps
its own temporal structure, apart from the seams, and so do Y and W, and the
only thing removed is the alignment between X and the rest. A spike population
moves as a whole, so the structure among its neurons survives.

The p-value of an observed value against $n$ null trials is

$$
p = \frac{1 + \#\{\text{null} \geq \text{observed}\}}{1 + n},
$$

so the smallest value $n$ trials can report is $1/(n+1)$. A small p-value says
that dependence exists at that resolution and says nothing about its size.

The reported value of a network whose held-out MI never rose above zero is 0,
which piles null trials up at exactly zero. `null_distribution_raw` keeps every
trial's training-side value as measured, for inspecting the null's shape.

### Amplification

The conditional quantities are formed by combining separately trained estimates,
and the answer is often small compared with the numbers it came from:

$$
\begin{aligned}
I(X;Y \mid W) &= I(X,W;Y) - I(W;Y) \\
\mathrm{TE}_{X\to Y} &= I(X_{past}, Y_{past}; Y_0) - I(Y_{past}; Y_0) \\
II &= I(X,W;Y) - I(X;Y) - I(W;Y)
\end{aligned}
$$

Subtracting two similar numbers cancels most of the signal and none of the
error. `amplification_factor` is the condition number of the combination,

$$
\text{amp} = \frac{\sum_i |t_i|}{|\text{result}|},
$$

which for two terms is $(|t_1| + |t_2|) / |t_1 - t_2|$. A relative error
$\epsilon$ on each component becomes roughly $\text{amp}\,\epsilon$ on the
result.

| factor | reading |
|---|---|
| about 1 | little cancels, and component errors pass through at their own size |
| 2 to 10 | ordinary; report the components beside the result |
| 10 or more | a small residual of large, similar numbers, where a 1% component error becomes 10% or more; the library warns |
| infinite | the result is exactly zero |

The factor grows without bound as the result approaches zero, so it is largest
for the conclusion a conditional analysis is usually run to support, that W
explains X away:

| $I(X,W;Y)$ | $I(W;Y)$ | $I(X;Y \mid W)$ | amplification |
|---|---|---|---|
| 0.767 | 0.200 | 0.567 | 1.7 |
| 0.767 | 0.600 | 0.167 | 8.2 |
| 0.767 | 0.700 | 0.067 | 21.8 |
| 0.767 | 0.760 | 0.007 | 206.4 |

A negative estimate of a quantity that cannot be negative is usually explained
by a large factor. At 300 the components would need an accuracy better than
0.3% for the sign to be determined, and "the true value is near zero" is then
the better reading.

$\text{amp}\,\epsilon$ assumes the component errors are independent or aligned.
The components come from the same estimator on the same data with the same
architecture, so part of their bias is shared and cancels. The factor is
therefore a bound on the damage more than a prediction of it. Working the other way, the
joint term is the largest component, reaches the InfoNCE ceiling first, and
biases the difference toward zero, so `test_saturation` on the joint network
(in `details[config_id]['trainings']`) belongs beside any small value. The
factor says how fragile the arithmetic is. Whether the result is distinguishable
from zero is settled by the spread over repeats or by a permutation test.

---

## Quantities and units

### What the number is per

On data without a windowing processor, every reported value is per joint
observation, one row $(x_i, y_i)$. On windowed data it is per window,
$I(X_{1:w}; Y_{1:w})$ for windows of $w$ samples, and it does not convert to a
rate by division.

Monotonicity in the window holds by construction, since a longer window's
samples contain a shorter window's at the same start and MI cannot fall when
more variables are observed on either side. Going from $w$ to $w + 1$ gives
$I_{w+1} \ge I_w$ in two steps. A sweep over `window_size` whose raw values rise
and then fall is therefore reporting an estimation artefact. The usual causes
are the ceiling, which falls as fewer windows fit in the recording
([InfoNCE](#infonce)), and an embedding of fixed capacity that stops keeping up
with the growing input.

Extensivity, a stronger and separate claim, says the curve eventually rises
linearly as $I_w = \bar{I}w + b$, a form that holds only once $w$ exceeds the
dependence timescale of the two processes. Below that timescale the curve still
bends. The rate $\bar{I}$ is the slope of the linear part. The offset $b$ has
come out positive in every case examined, since the chain-rule increments
decrease toward $\bar{I}$ and their partial sums sit above $\bar{I}w$. It is the
subextensive part of the block MI, similar in shape to a single process's excess
entropy ([temporal information quantities](#temporal-information-quantities))
without being the same quantity.

A larger window generically reports a larger raw MI because it is built from
more data, whether or not the coupling changed. Windowed values at different
window sizes are therefore not comparable as "how much X and Y share", and a
windowed value quoted without its window size cannot be read.

A reported value is in one of these units:

| unit | where it comes from | comparable across |
|---|---|---|
| bits per observation | unwindowed data, one instant of X against one of Y | datasets of the same variables |
| bits per window | windowed data; grows with the window by construction | values at the same window size only |
| bits per step | a rate, from `mi_rate` or the slope of block MI | window sizes |
| a fraction of the target's entropy | a discrete target with $n$ levels, $H = \log_2 n$ when equally occupied | targets of different sizes |

"X and Y carry 3 bits" is a statement in bits per observation. Knowing Y removes
on average 3 bits of uncertainty about X, the equivalent of three yes-or-no
questions.

### Getting a rate

Since $I_w = \bar{I}w + b$ with $b$ nonzero, dividing one windowed value by its
duration leaves a residual $b/w$ that changes with the window. The rate comes
either from `mi_rate`, which estimates the per-step quantity directly
([temporal information quantities](#temporal-information-quantities)), or from
the slope of block MI against window size over a suitable range of $w$. That
range needs $w$ large enough for the curve to have straightened and small enough
for the estimates to stay below the ceiling. When no range satisfies both, the
slope is unavailable on that recording and `mi_rate` is the only route.

The lower end of the range is the dependence timescale, and the estimator can
find it. Sweeping `mi_rate`'s history length $h$ gives the chain-rule increments
$I(X_{all}; Y_t \mid Y_{t-h:t-1})$ at successive $h$, and the increments
converge to $\bar{I}$. The $h$ at which they settle is the timescale over which
Y's own past still carries information about its present, and above it the block
curve is linear. The autocorrelation time is a poorer substitute, since
autocorrelation is a second-order measure and a process can have none at a lag
where it has strong nonlinear dependence ([Fraser and Swinney,
1986](https://doi.org/10.1103/PhysRevA.33.1134)). It is only a lower bound on
the timescale.

A per-step quantity converts to bits per second by dividing by the bin width.
Transfer entropy, instantaneous exchange, the directed information rate, active
information storage and the MI rate are per step in this sense. Predictive and
cross-predictive information are not, since their $B$ group spans a window. A
rate quoted without its bin width cannot be read.

### Silent windows

A spike window without spikes is dropped by default. Writing
$A = \mathbb{1}\{\text{window kept}\}$ and using the fact that an empty spike
window is a fixed all-zero vector,

$$
I(X;Y) = I(A;Y) + p \, I(X;Y \mid A = 1),
$$

where $p$ is the fraction of windows kept. With silent windows dropped the
library estimates the last term, in bits per active window. Keeping them
(`drop_empty_windows=False`) estimates the left-hand side, in bits per window.
Scaling by $p$ does not convert between them, since $I(A;Y)$ is the information
carried by whether the population is active at all. Two neurons that fall
silent together and fire together, with unrelated patterns while active, have
almost no MI over active windows and substantial MI overall, so keeping silent
windows can move an estimate up as well as down.

:::{admonition} References
:class: note

- S. P. Strong, Roland Koberle, Rob R. de Ruyter van Steveninck and William Bialek. Entropy and information in neural spike trains. *Physical Review Letters* 80, 197, 1998. [doi.org/10.1103/PhysRevLett.80.197](https://doi.org/10.1103/PhysRevLett.80.197)
- Caroline M. Holmes and Ilya Nemenman. Estimation of mutual information for real-valued data with error bars and controlled bias. *Physical Review E* 100, 022404, 2019. [doi.org/10.1103/PhysRevE.100.022404](https://doi.org/10.1103/PhysRevE.100.022404)
- Eslam Abdelaleem, K. Michael Martini and Ilya Nemenman. Accurate estimation of mutual information in high dimensional data. arXiv:2506.00330, 2025. [arxiv.org/abs/2506.00330](https://arxiv.org/abs/2506.00330)
- Andrew M. Fraser and Harry L. Swinney. Independent coordinates for strange attractors from mutual information. *Physical Review A* 33, 1134, 1986. [doi.org/10.1103/PhysRevA.33.1134](https://doi.org/10.1103/PhysRevA.33.1134)
- James P. Crutchfield and David P. Feldman. Regularities unseen, randomness observed: levels of entropy convergence. *Chaos* 13, 25-54, 2003. [pubs.aip.org](https://pubs.aip.org/aip/cha/article/13/1/25/510735/Regularities-unseen-randomness-observed-Levels-of)
- Pierre-Olivier Amblard and Olivier J. J. Michel. The relation between Granger causality and directed information theory: a review. *Entropy* 15, 113-143, 2013. [doi.org/10.3390/e15010113](https://doi.org/10.3390/e15010113)
:::

### Temporal information quantities

A family of quantities describes how information is organised across time,
within one process or between two. Every one of them, interaction information
excepted, reduces to

$$
\boxed{\;I(A; B \mid C) \;=\; I([A, C]\,;\, B) \;-\; I(C; B)\;}
$$

for a particular choice of which time offsets go into $A$, $B$ and $C$. With
$X_{past} = X_{t-k:t-1}$, $X_{fut} = X_{t:t+k-1}$, $X_0 = X_t$ and
$X_{all}(L) = X_{t-L:t+L}$, a two-sided window of half-width $L$:

| quantity | $A$ | $B$ | $C$ |
|---|---|---|---|
| instantaneous MI | $X_0$ | $Y_0$ | none |
| active information storage | $X_{past}$ | $X_0$ | none |
| predictive information | $X_{past}$ | $X_{fut}$ | none |
| cross-predictive information | $X_{past}$ | $Y_{fut}$ | none |
| block MI | $X_{1:w}$ | $Y_{1:w}$ | none |
| transfer entropy | $X_{past}$ | $Y_0$ | $Y_{past}$ |
| conditional transfer entropy | $X_{past}$ | $Y_0$ | $Y_{past}, W_{past}$ |
| MI rate | $X_{all}(L)$ | $Y_0$ | $Y_{past}(h)$ |
| instantaneous exchange | $X_0$ | $Y_0$ | $X_{past}, Y_{past}$ |
| directed information rate | $X_{past}, X_0$ | $Y_0$ | $Y_{past}$ |

The first five have no conditioning set and are a single estimate of
$I(A;B)$, free of the [amplification](#amplification) that a difference carries.
The literature holds several definitions of these quantities, differing in
formalism and in name, and some are better justified than others. The question
being asked decides which set of offsets captures it.

**Instantaneous MI**, $I(X_0; Y_0)$, is the MI between two processes at matching
time indices.

**Active information storage**, $\mathrm{AIS}_X = I(X_{past}(k); X_0)$, measures
how much a process's recent history predicts its present value, the part of $X$
at each step that is stored and not new
([Lizier et al., 2012](https://doi.org/10.1016/j.ins.2012.04.016)).

**Predictive information**, $I_{pred}(k) = I(X_{past}(k); X_{fut}(k))$, measures
what a past says about its own future
([Bialek et al., 2001](https://direct.mit.edu/neco/article/13/11/2409/6472/Predictability-Complexity-and-Learning);
[Palmer et al., 2015](https://doi.org/10.1073/pnas.1506855112)). A longer future
can only reveal more about the past, so $I_{pred}(k) \ge \mathrm{AIS}_X$, by the
monotonicity argument of [what the number is per](#what-the-number-is-per). Its
limit as $k \to \infty$ is the excess entropy
([Crutchfield and Feldman, 2003](https://pubs.aip.org/aip/cha/article/13/1/25/510735/Regularities-unseen-randomness-observed-Levels-of)),
a property of the process. Sweeping $k$ reads it off as the plateau of the
curve, and a curve that keeps climbing belongs to a process whose excess entropy
is infinite. The growth law is often the finding. At large $k$ the predictive
information stays finite, grows logarithmically, or grows as a fractional power.
Logarithmic growth marks a process described by finitely many parameters, with
the coefficient counting them, and power-law growth marks a nonparametric one
([Bialek et al., 2001](https://direct.mit.edu/neco/article/13/11/2409/6472/Predictability-Complexity-and-Learning)).

**Cross-predictive information**, $I(X_{past}(k); Y_{fut}(k))$, asks how much a
window of X's past tells about an equally long window of Y's future, the
two-process version of predictive information.

**Block MI**, $I(X_{1:w}; Y_{1:w})$, is what a windowed estimate computes. Its
growth with $w$ is discussed under [what the number is per](#what-the-number-is-per).

**Transfer entropy**, $\mathrm{TE}_{X\to Y} = I(X_{past}; Y_0 \mid Y_{past})$,
conditions on Y's past so that Y's own storage is not counted as something X
transferred ([Schreiber, 2000](https://doi.org/10.1103/PhysRevLett.85.461)). A
history too short leaves part of that storage uncontrolled and inflates the
transfer, and a history too long inflates the critic's input and can cost
accuracy. A sweep over `history_window` finds the plateau, and a curve still
moving at the longest history has not converged. For Gaussian variables transfer
entropy equals Granger causality
([Barnett et al., 2009](https://doi.org/10.1103/PhysRevLett.103.238701)).

Transfer entropy measures the predictive information a source adds under
conditioning, and that differs from a causal effect or an information flow in
the sense its name suggests. It mixes information the source provides on its own
with information that arises only from the source and the target's past together
([James et al., 2016](https://doi.org/10.1103/PhysRevLett.116.238701)).

**Conditional transfer entropy**,
$\mathrm{TE}_{X\to Y \mid W} = I(X_{past}; Y_0 \mid Y_{past}, W_{past})$, adds a
third process's history to the conditioning set. It asks how much of X's
apparent influence on Y survives once W's history is controlled for. When W
removes most of it, the dependence likely passes through W.

**MI rate**, $I(X_{all}(L); Y_0 \mid Y_{past}(h))$, is the two-sided per-step
information rate between X and Y
([Gelfand and Yaglom, 1959](https://doi.org/10.1090/trans2/012)). It asks how
much X in a symmetric window around the present tells about Y's present, once
enough of Y's past is controlled for. The quantity is the joint limit
$L \to \infty$ and $h \to \infty$, and the two windows bias the estimate in
opposite directions. Too small an $h$ leaves Y's own dependence uncontrolled and
reads high, and too narrow an $L$ leaves signal out and reads low. A curve that
has flattened along one window can therefore sit at a converged-looking wrong
value set by the other. Sweep one, fix it past its knee, then sweep the other.

$X_{all}$ is two-sided because the MI rate is the chain-rule decomposition of
the symmetric block MI, whose term at time $t$ conditions on all of X, including
the part after $t$. Truncating X to its past gives the directed information
rate below, and the difference between the two is the reverse transfer entropy.

**Instantaneous exchange**, $I(X_0; Y_0 \mid X_{past}(k), Y_{past}(k))$, asks how
much X and Y share at the same instant beyond what their separate pasts explain.
It is the term that completes transfer entropy in the decomposition of directed
information
([Amblard and Michel, 2009](https://arxiv.org/abs/0911.2873);
[Amblard and Michel, 2014](https://arxiv.org/abs/1203.5572)), and for Gaussian
processes it corresponds to Geweke's instantaneous linear feedback
([Geweke, 1982](https://doi.org/10.1080/01621459.1982.10477803)). Zero-lag
coupling is often a nuisance, as with volume conduction in EEG. When a shared
driver acts on both processes within one time step, it is the dominant coupling,
and discarding it discards the dependence. Which case holds is a property of the
system. As a difference of two estimates, it inherits their
[amplification](#amplification) when it is small.

**Directed information rate**, $I(X_{past}(k), X_0; Y_0 \mid Y_{past}(k))$, is
everything X's past and present together tell about Y's present beyond Y's own
past ([Massey, 1990](https://www.isiweb.ee.ethz.ch/archive/massey_pub/pdf/BI532.pdf)).
One application of the chain rule to the $A$ group splits it exactly:

$$
I(X_{past}, X_0; Y_0 \mid Y_{past}) = I(X_{past}; Y_0 \mid Y_{past}) + I(X_0; Y_0 \mid Y_{past}, X_{past}).
$$

Adding the two estimated parts would carry transfer entropy's small,
high-variance residual into a better-behaved quantity, so
`directed_information_rate` estimates it from its own $A$, $B$ and $C$, and the
decomposition checks the result.

The MI rate, instantaneous exchange and the directed information rate have $A$
and $C$ groups of different lengths. Concatenating them would mean zero-padding
the shorter one and leaving the network to learn to ignore the padding, so they
embed the two groups in separate branches of one network
(`Model(embedding_model='dual_branch')`) and fuse the results.

#### Identities the catalogue satisfies

Storage and transfer split what both histories tell about the next step of X
([Lizier et al., 2012](https://doi.org/10.1016/j.ins.2012.04.016)):

$$
I(X_{past}, Y_{past}; X_0) = \mathrm{AIS}_X + \mathrm{TE}_{Y \to X}
$$

Directed information splits into a lagged and an instantaneous part:

$$
\text{DI rate} = \mathrm{TE}_{X \to Y} + \text{instantaneous exchange}
$$

Both follow from the chain rule with matched conditioning and hold at any finite
history, so a deviation points to the estimation.

Massey's conservation law holds per step
([Massey and Massey, 2005](https://doi.org/10.1109/ISIT.2005.1523313)):

$$
\text{MI rate} = \text{DI rate}(X \to Y) + \mathrm{TE}_{Y \to X}
$$

It is a limit, reached once $L$ and $h$ are both past the dependence timescale,
and at small windows the gap is a property of the truncation. The gap between the
symmetric and the directed quantity is the reverse transfer entropy, so the two
coincide only without feedback. A shared latent creates feedback in Massey's
sense even with no causal path between the processes, and the MI rate then needs
the two-sided window on X that the directed rate does without.

:::{admonition} References
:class: note

- Thomas Schreiber. Measuring information transfer. *Physical Review Letters* 85, 461, 2000. [doi.org/10.1103/PhysRevLett.85.461](https://doi.org/10.1103/PhysRevLett.85.461)
- Joseph T. Lizier, Mikhail Prokopenko and Albert Y. Zomaya. Local measures of information storage in complex distributed computation. *Information Sciences* 208, 39-54, 2012. [doi.org/10.1016/j.ins.2012.04.016](https://doi.org/10.1016/j.ins.2012.04.016)
- William Bialek, Ilya Nemenman and Naftali Tishby. Predictability, complexity, and learning. *Neural Computation* 13, 2409-2463, 2001. [direct.mit.edu](https://direct.mit.edu/neco/article/13/11/2409/6472/Predictability-Complexity-and-Learning)
- Stephanie E. Palmer, Olivier Marre, Michael J. Berry II and William Bialek. Predictive information in a sensory population. *Proceedings of the National Academy of Sciences* 112, 6908-6913, 2015. [doi.org/10.1073/pnas.1506855112](https://doi.org/10.1073/pnas.1506855112)
- James P. Crutchfield and David P. Feldman. Regularities unseen, randomness observed: levels of entropy convergence. *Chaos* 13, 25-54, 2003. [pubs.aip.org](https://pubs.aip.org/aip/cha/article/13/1/25/510735/Regularities-unseen-randomness-observed-Levels-of)
- Lionel Barnett, Adam B. Barrett and Anil K. Seth. Granger causality and transfer entropy are equivalent for Gaussian variables. *Physical Review Letters* 103, 238701, 2009. [doi.org/10.1103/PhysRevLett.103.238701](https://doi.org/10.1103/PhysRevLett.103.238701)
- Ryan G. James, Nix Barnett and James P. Crutchfield. Information flows? A critique of transfer entropies. *Physical Review Letters* 116, 238701, 2016. [doi.org/10.1103/PhysRevLett.116.238701](https://doi.org/10.1103/PhysRevLett.116.238701)
- I. M. Gelfand and A. M. Yaglom. Calculation of the amount of information about a random function contained in another such function. *American Mathematical Society Translations*, Series 2, 12, 199-246, 1959. [doi.org/10.1090/trans2/012](https://doi.org/10.1090/trans2/012)
- Pierre-Olivier Amblard and Olivier J. J. Michel. Relating Granger causality to directed information theory for networks of stochastic processes. arXiv:0911.2873, 2009. [arxiv.org/abs/0911.2873](https://arxiv.org/abs/0911.2873)
- Pierre-Olivier Amblard and Olivier J. J. Michel. Causal conditioning and instantaneous coupling in causality graphs. *Information Sciences* 264, 279-290, 2014. [arxiv.org/abs/1203.5572](https://arxiv.org/abs/1203.5572)
- John Geweke. Measurement of linear dependence and feedback between multiple time series. *Journal of the American Statistical Association* 77, 304-313, 1982. [doi.org/10.1080/01621459.1982.10477803](https://doi.org/10.1080/01621459.1982.10477803)
- James L. Massey. Causality, feedback and directed information. *Proceedings of the 1990 International Symposium on Information Theory and its Applications*, 303-305, 1990. [isiweb.ee.ethz.ch](https://www.isiweb.ee.ethz.ch/archive/massey_pub/pdf/BI532.pdf)
- James L. Massey and Peter C. Massey. Conservation of mutual and directed information. *Proceedings of the 2005 IEEE International Symposium on Information Theory*, 157-158, 2005. [doi.org/10.1109/ISIT.2005.1523313](https://doi.org/10.1109/ISIT.2005.1523313)
:::

### Interaction information

Interaction information is the one quantity here that is not a single
$I(A;B \mid C)$. It asks how the information X and Y share changes once a third
variable W is also observed:

$$
II = I(X, W; Y) - I(X; Y) - I(W; Y)
$$

and it is estimated from three separate MI estimates. Since
$I(X,W;Y) - I(W;Y) = I(X;Y \mid W)$, it can also be written as the gap between a
conditional and an unconditional MI, $II = I(X;Y \mid W) - I(X;Y)$.

Interaction information is signed, and both signs have a standard reading. When
$II < 0$, X and W carry overlapping information about Y, and observing both
tells less than the sum of observing each. A shared upstream driver is the
common cause. X and W are then each partial proxies for it, and knowing one
reduces what the other can add. When $II > 0$, X and W together reveal something
about Y that neither reveals alone, as in an XOR-like dependence of Y on a
combination of X and W that neither shows in isolation.

Interaction information is a net measure, with synergy and redundancy entering
at opposite signs, so a system with 0.5 bits of each returns $II = 0$, the same
as a system with no interaction. A value near zero cannot tell "nothing is
happening" from "two things are happening and cancelling". Separating them takes
a partial information decomposition, which asks for four terms (redundancy, two
unique terms and synergy) while classical information theory supplies three
equations relating them ([Williams and Beer,
2010](https://arxiv.org/abs/1004.2515)). The system is underdetermined, any
decomposition needs an extra axiom, and different reasonable axioms give
different answers on the same data ([Bertschinger et al.,
2014](https://doi.org/10.3390/e16042161)). The library implements none of them
for that reason.

Papers differ in the sign convention of this quantity. McGill's interaction
information ([McGill, 1954](https://doi.org/10.1007/BF02289159)) and Bell's
co-information ([Bell,
2003](https://www.kecl.ntt.co.jp/icl/signal/ica2003/cdrom/data/0187.pdf)) differ
by a sign for odd numbers of variables, and papers order the subtraction
differently. Under the convention above, positive means net synergy and negative
means net redundancy. A published value is compared by its formula, beside its
name.

Interaction information combines three estimates and is usually small compared
with them, so its [amplification](#amplification) is often large. The three
components belong beside it in any report.

:::{admonition} References
:class: note

- William J. McGill. Multivariate information transmission. *Psychometrika* 19, 97-116, 1954. [doi.org/10.1007/BF02289159](https://doi.org/10.1007/BF02289159)
- Anthony J. Bell. The co-information lattice. *Proceedings of the 4th International Symposium on Independent Component Analysis and Blind Signal Separation (ICA 2003)*, 921-926, 2003. [PDF](https://www.kecl.ntt.co.jp/icl/signal/ica2003/cdrom/data/0187.pdf)
- Paul L. Williams and Randall D. Beer. Nonnegative decomposition of multivariate information. arXiv:1004.2515, 2010. [arxiv.org/abs/1004.2515](https://arxiv.org/abs/1004.2515)
- Nils Bertschinger, Johannes Rauh, Eckehard Olbrich, Jürgen Jost and Nihat Ay. Quantifying unique information. *Entropy* 16, 2161-2183, 2014. [doi.org/10.3390/e16042161](https://doi.org/10.3390/e16042161)
:::

---

## Cross-run-stable directions of shared structure

A nonlinear encoder with more embedding capacity than the number of latent
factors two views share can build combinations of them, products and
higher-order mixtures, that look after training like independent factors in the
spectrum. Every measure computed from one trained embedding's spectrum, such as
the participation ratio, the eigengap or the singular-value profile, is blind to
the difference. No exact count of "the" dimensionality can be read from a single
trained spectrum ([Gulati et al., 2026](https://proceedings.mlr.press/v326/gulati26a.html)).
`mode='dimensionality'` reports what can be read: a regime estimate taken before
training, the directions that reproduce across independent fits, and the
participation ratios as a secondary description.

### A regime read before training

The regime diagnostic centres each view's channels, computes the eigenvalues of
the channel correlation matrix, and looks at the ratios of consecutive
eigenvalues. An isolated large ratio marks a separable-like regime, with each
channel driven mostly by one factor. A flat curve of ratios marks an
entangled-like regime, with channels reflecting several factors jointly, as in
mixed selectivity. It runs once, without training, and is reported as `regime_x`
and `regime_y`. The threshold, a peak ratio of 3.0, is a heuristic calibrated on
two validated cases, about 14 to 16 for a clean separable case and about 1.7 for an entangled one.

### Directions that reproduce

The mode uses the hybrid critic unless another is set. It embeds X and Y
separately and scores their concatenation with a small network, avoiding the
rigid geometry of a dot product. The embedding is kept small, 8 dimensions unless
set, since spare capacity is what lets an encoder build combinations.

With `Output(whitening='std')`, the default, each embedding dimension is divided
by its standard deviation,

$$
\tilde{Z}_{X,i} = \frac{Z_{X,i}}{\mathrm{std}(Z_{X,i})}, \qquad \tilde{Z}_{Y,i} = \frac{Z_{Y,i}}{\mathrm{std}(Z_{Y,i})},
$$

and the cross-covariance of the whitened held-out embeddings,

$$
C_{XY} = \frac{1}{N-1} (\tilde{Z}_X - \bar{\tilde{Z}}_X)^T (\tilde{Z}_Y - \bar{\tilde{Z}}_Y),
$$

is decomposed by SVD into a rotation, ordering the directions by shared
variance, and singular values $\sigma_i$. One spectrum cannot separate true
factors from constructed ones, so the fit is repeated `n_splits` times and each
rank's direction is compared across fits, on held-out data only. A rank is
stable when its direction correlates across every pair of fits at least at
`stability_threshold` (0.7), since a constructed combination belongs to one fit
and is not expected to reproduce. It must also clear a noise floor, a mean
strength of at least `min_strength_fraction` (0.05) of the strongest rank,
because a pure-noise direction can correlate across fits by chance. Adjacent
stable ranks whose strengths differ by less than `degeneracy_ratio_threshold`
(1.3) are reported as a group whose members exist and cannot be ordered.

`stable_directions` lists the ranks that are individually stable,
`stable_but_degenerate_groups` the groups, and `n_stable_total` their count, a
lower bound on the number of shared directions.

Without `y_data` the mode splits one dataset into two halves that share no
channel, at random by default or by the chosen `split_method`, and each fit uses
a new split as well as a new initialisation. With `y_data` it compares X and Y
directly, and the fits differ in their initialisation.

### Participation ratios

Each fit also reports two participation ratios of its own spectrum, in `runs`:

$$
\mathrm{PR}_{\text{singular}} = \frac{\left(\sum_i \sigma_i\right)^2}{\sum_i \sigma_i^2}, \qquad \mathrm{PR}_{\text{eig}} = \frac{\left(\sum_i \sigma_i^2\right)^2}{\sum_i \sigma_i^4}.
$$

Both measure how spread out one spectrum is. `pr_eig` weights by
$\lambda_i = \sigma_i^2$ and responds more to the rank of the representation
than `pr_singular`, which weights by $\sigma_i$. Neither can tell true factors
from constructed ones.

### Convergence and the ceiling

A fit that has not converged has an incomplete spectrum, and its directions can
mislead. Each fit counts as converged when its best epoch comes before its last,
and `converged` is `True` only when every fit converged. When the MI estimate
sits within `ceiling_mi_fraction` (0.85) of its ceiling the mode warns, since a
spectrum built on a saturated estimate needs extra scrutiny. A converged fit near
its ceiling reports fewer stable directions than exist.

:::{admonition} References
:class: note

- Paarth Gulati, Eslam Abdelaleem, Audrey Sederberg and Ilya Nemenman. Mutual information and task-relevant latent dimensionality. *Proceedings of GRaM: the Second Edition of the Workshop on Geometry-grounded Representation Learning and Generative Modeling*, PMLR 326:262-293, 2026. [proceedings.mlr.press/v326/gulati26a](https://proceedings.mlr.press/v326/gulati26a.html)
- Ege Altan, Sara A. Solla, Lee E. Miller and Eric J. Perreault. Estimating the dimensionality of the manifold underlying multi-electrode neural recordings. *PLoS Computational Biology* 17, e1008591, 2021. [doi.org/10.1371/journal.pcbi.1008591](https://doi.org/10.1371/journal.pcbi.1008591)
:::

---

## Spike-timing precision

Many neural codes rely on spike timing at the millisecond scale
([Tang et al., 2014](https://doi.org/10.1371/journal.pbio.1002018)), and the
precision at which a representation carries information shows in how the
information degrades as the timing is perturbed
([Ortega et al., 2023](https://doi.org/10.1371/journal.pcbi.1011170)).
`mode='precision'` trains once and evaluates many times, so no network is
retrained per level of corruption.

A critic is first trained on the uncorrupted data, and its baseline MI is
recorded. The network is then frozen, and the data are corrupted over a grid of
levels $\tau$. Rounding, the default, moves every value to the centre of its bin
of width $\tau$,

$$
\tilde{X} = \tau \left( \left\lfloor \frac{X}{\tau} \right\rfloor + \frac{1}{2} \right),
$$

and needs one evaluation per level. Noise adds a draw from $U(-\tau/2, \tau/2)$
and is averaged over `n_noise_samples` draws. Both keep the timing resolution at
$\tau$.

The corruption touches only entries that hold a measurement. A spike window
stores its spike times in a fixed number of slots, and the unused slots carry an
empty value, zero by default. Jittering those slots would hand the frozen critic
spikes that never happened: on a timing-coded spike pair, 69% of the entries of
each window were unused slots, and jittering them took the MI from 1.17 to
$-6.3$ bits at the smallest $\tau$, 5 ms. The unused slots, empty bins and
zero-padded gaps therefore stay as they are. Bin centres sit $\tau/2$ away from
every multiple of $\tau$, so rounding never moves a spike onto an empty value of
zero either. Rounding to the nearest multiple would do so. On the same data it
deletes 5.7% of the spikes at $\tau = 0.2$ s and 25% at $\tau = 0.4$ s, and the
lost spikes mix into the loss of timing being measured. Under both methods the
number of spikes stays fixed and only their timing degrades.

The precision is the smallest $\tau$ at which the corrupted MI falls below a
fraction $\rho$ of the baseline:

$$
\tau^* = \min \{\tau : I(\tilde{X}^{\tau}; Y) < \rho \, I(X; Y)\}
$$

With the default $\rho = 0.9$, $\tau^*$ is the finest corruption that has already
cost more than 10% of the information. Several ratios at once, such as
`threshold_ratio=[0.9, 0.75, 0.5]`, trace the whole profile, from the onset of
the loss to its collapse.

The frozen critic was trained on clean inputs, and a lower bound can fall
arbitrarily far below zero on inputs unlike its training data. Past the
threshold the curve measures how far the bound has broken, and the crossing of
the threshold is the reading.

:::{admonition} References
:class: note

- Claire Tang, Diala Chehayeb, Kyle Srivastava, Ilya Nemenman and Samuel J. Sober. Millisecond-scale motor encoding in a cortical vocal area. *PLoS Biology* 12, e1002018, 2014. [doi.org/10.1371/journal.pbio.1002018](https://doi.org/10.1371/journal.pbio.1002018)
- Joy Ortega, Tobias Niebur, Leo Wood, Rachel Conn and Simon Sponberg. An information theoretic method to resolve millisecond-scale spike timing precision in a comprehensive motor program. *PLoS Computational Biology* 19, e1011170, 2023. [doi.org/10.1371/journal.pcbi.1011170](https://doi.org/10.1371/journal.pcbi.1011170)
:::

---

## Regularised objectives

The standard objective trains the critic to maximise the MI alone,
$\mathcal{L} = -\hat{I}(Z_X; Z_Y)$. Two regularisers can be added, a
variational encoder and a reconstruction decoder, and together they give the
deep variational symmetric information bottleneck
([Abdelaleem et al., 2025b](http://jmlr.org/papers/v26/24-0204.html)).

### Variational encoders

A variational encoder learns a distribution over embeddings, $q(z \mid x)$, a
Gaussian with mean and variance $(\mu_x, \sigma_x)$, in place of a single
embedding per input. A KL term pulls each distribution toward a standard normal
prior:

$$
\mathcal{L} = D_{\text{KL}}(q(z_x \mid x) \,\|\, p(z_x)) + D_{\text{KL}}(q(z_y \mid y) \,\|\, p(z_y)) - \beta \, \hat{I}(Z_X; Z_Y)
$$

`Model(use_variational=True)` places a wrapper on top of any encoder. The
encoder maps the input to a deterministic embedding, and the wrapper adds linear
heads for $\mu$ and $\log \sigma^2$ and samples with the reparameterisation
trick. The KL term is averaged per sample, so $\beta$ has the same meaning at
any batch size. The default $\beta = 1024$ lets the MI term dominate while the
KL term still penalises degenerate distributions. A smaller $\beta$ strengthens
the pull toward the prior, and $\beta \ll 1$ can collapse the embeddings onto it
and lower the estimated MI.

The concat critic has no embedding of X or of Y, only one network scoring each
pair, so the variational layer sits on that network's output. Each score becomes
a draw from a Gaussian whose mean and variance the network produces, and the KL
term pulls every score toward the prior. It regularises the critic and has no
information-bottleneck reading, which needs the separable or hybrid critic. On
correlated Gaussians with 2.00 bits, the concat critic gave 1.87 bits without the
layer and 1.94 with it at the default $\beta$, and 1.51 at $\beta = 1$.

### Reconstruction decoders

`Model(use_decoder=True)` adds a decoder $d_X$ that maps $Z_X$ back to the input,
and likewise for Y, trained together with the critic:

$$
\mathcal{L} = -\Big[\hat{I}(Z_X; Z_Y) - \lambda_X \, \mathcal{L}_\text{rec}(X,\hat{X}) - \lambda_Y \, \mathcal{L}_\text{rec}(Y,\hat{Y})\Big]
$$

with $\hat{X} = d_X(Z_X)$ and $\hat{Y} = d_Y(Z_Y)$. $\lambda_X$ and $\lambda_Y$
(`decoder_lambda_x`, `decoder_lambda_y`) set how much reconstruction error
counts against one nat of shared information, and the default is small (0.001).
The reconstruction loss follows the decoder's output:

| output activation | data | loss |
|---|---|---|
| `'linear'` (default) | continuous | mean squared error |
| `'sigmoid'` | binary, such as spike presence | mean squared error |
| `'softmax'` | categorical, one-hot over channels | $-\sum_c y_c \log p_c$, the cross-entropy |

### Both together

The loss with both regularisers switched on is

$$
\mathcal{L} = \overline{D}_\text{KL}(Z_X) + \overline{D}_\text{KL}(Z_Y) - \beta\Big[\hat{I}(Z_X; Z_Y) - \lambda_X \, \mathcal{L}_\text{rec}(X,\hat{X}) - \lambda_Y \, \mathcal{L}_\text{rec}(Y,\hat{Y})\Big].
$$

$\beta$ scales everything the objective is asked to preserve, the MI term and both
reconstructions, so the effective weight on a reconstruction is $\beta\lambda$. A
third constant on the MI term would be redundant, since scaling all three terms
by $c$ and $\beta$ by $1/c$ leaves the objective unchanged. The KL terms pull the
embeddings toward the prior, and the decoders make each embedding keep enough to
rebuild its own input. The objective therefore favours embeddings that are informative
about the other variable, regular in distribution, and faithful to their own
input. This is the deep variational symmetric information bottleneck of
[Abdelaleem et al., 2025b](http://jmlr.org/papers/v26/24-0204.html), an
instance of the multivariate information bottleneck
([Friedman et al., 2001](https://arxiv.org/abs/1301.2270)).

:::{admonition} References
:class: note

- Eslam Abdelaleem, Ilya Nemenman and K. Michael Martini. Deep variational multivariate information bottleneck: a framework for variational losses. *Journal of Machine Learning Research* 26, 2025. [jmlr.org/papers/v26/24-0204](http://jmlr.org/papers/v26/24-0204.html)
- Nir Friedman, Ori Mosenzon, Noam Slonim and Naftali Tishby. Multivariate information bottleneck. *Proceedings of the 17th Conference on Uncertainty in Artificial Intelligence*, 152-161, 2001. [arxiv.org/abs/1301.2270](https://arxiv.org/abs/1301.2270)
:::
