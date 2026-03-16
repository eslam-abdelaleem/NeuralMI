# %% [markdown]
# # Tutorial 9: The Complete Pipeline — From Recording to Publication-Ready Result
#
# Every previous tutorial has isolated one concept: data formatting, splits,
# sweeps, rigorous estimation, temporal analysis, population geometry, model
# choices. This tutorial puts them all together as a single, end-to-end
# pipeline applied to a realistic neuroscience scenario.
#
# **The scientific question:** Does a hippocampal population encode the
# animal's spatial position? If so, (a) how strong is that encoding, (b) when
# does the encoding peak relative to behavior, and (c) is there more information
# beyond the animal's current running direction?
#
# We use synthetic data that mimics real hippocampal recordings — a population
# of place-cell-like neurons whose firing is tuned to position along a linear
# track. This means we have ground truth and can verify each step of the pipeline.
#
# **Pipeline steps:**
#
# 1. **Generate data** — simulate a hippocampal-like recording
# 2. **Sanity check** — quick estimate and null permutation test
# 3. **Sweep window size** — find the analysis window that captures peak information
# 4. **Sweep architecture** — find the smallest sufficient model
# 5. **Diagnose training** — verify the model is generalising, not overfitting
# 6. **Rigorous estimate** — bias-corrected result with confidence interval
# 7. **Temporal: lag analysis** — when is encoding strongest relative to behavior?
# 8. **Conditional MI** — does the population encode position above and beyond direction?
# 9. **Report** — summarise findings in one print block

# %% [markdown]
# ## Step 0: Imports and Data Generation
#
# We generate a population of N=80 neurons firing as a function of position
# on a 1-metre linear track. Each neuron has a Gaussian place field; its
# firing rate is a function of the animal's current position. The behavioral
# variables — position (continuous) and direction (binary: left or right) —
# are the targets. Position is the primary variable of interest; direction
# will be used for the conditional MI.

# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import neural_mi as nmi

sns.set_context("talk")
np.random.seed(42)

# --- Simulation parameters ---
N_NEURONS    = 80       # number of simulated place cells
N_TIMEPOINTS = 8000     # recording length (time steps)
SAMPLE_RATE  = 100.0    # Hz
TRACK_LENGTH = 1.0      # metres
NOISE_SIGMA  = 0.3      # noise added to firing rates

# --- Simulate animal trajectory ---
# Random walk on [0, TRACK_LENGTH]; smooth with short window
raw_pos = np.cumsum(np.random.randn(N_TIMEPOINTS) * 0.01)
# Fold back at boundaries (reflecting)
pos = np.zeros(N_TIMEPOINTS)
pos[0] = TRACK_LENGTH / 2
v = 0.01
for t in range(1, N_TIMEPOINTS):
    v = v + np.random.randn() * 0.003
    v = np.clip(v, -0.03, 0.03)
    pos[t] = pos[t-1] + v
    if pos[t] < 0:
        pos[t] = -pos[t]; v = abs(v)
    if pos[t] > TRACK_LENGTH:
        pos[t] = 2 * TRACK_LENGTH - pos[t]; v = -abs(v)

# Running direction: 1 = rightward (velocity > 0), 0 = leftward
direction = (np.convolve(np.gradient(pos), np.ones(5)/5, mode='same') > 0).astype(int)

# --- Place cell firing rates ---
# Each neuron has a preferred position drawn uniformly along the track
preferred_pos = np.linspace(0, TRACK_LENGTH, N_NEURONS)
field_width   = 0.15   # sigma of Gaussian place field
# Firing rate: Gaussian bump around preferred position + noise
pos_col = pos[:, np.newaxis]                       # (T, 1)
pref_row = preferred_pos[np.newaxis, :]            # (1, N)
rates = np.exp(-0.5 * ((pos_col - pref_row) / field_width)**2)
spikes = rates + NOISE_SIGMA * np.random.randn(N_TIMEPOINTS, N_NEURONS)
spikes = np.clip(spikes, 0, None)   # firing rates are non-negative

print(f"Neural population:   {spikes.shape}  (n_timepoints × n_neurons)")
print(f"Position:            {pos.shape}      (n_timepoints,)")
print(f"Direction:           {direction.shape} (n_timepoints,)  values: {np.unique(direction)}")

# Reshape into (n_timepoints, n_channels) — expected format for continuous processor
pos_2d = pos[:, np.newaxis]               # (T, 1)
direction_2d = direction[:, np.newaxis].astype(int)   # (T, 1)

# %% [markdown]
# ## Step 1: Sanity Check — Quick Estimate + Permutation Test
#
# Before running the full pipeline, always run a quick estimate to confirm
# that MI is non-zero and above the null. If the estimate is at or below the
# null, something is wrong with the data format, split mode, or processor
# before you invest computation in sweeps and rigorous estimation.

# %%
PROC_PARAMS = {'window_size': 50, 'sample_rate': SAMPLE_RATE}   # 500 ms windows

# Quick estimate
result_quick = nmi.run(
    x_data=spikes, y_data=pos_2d,
    mode='estimate',
    processor_type_x='continuous', processor_params_x=PROC_PARAMS,
    processor_type_y='continuous', processor_params_y=PROC_PARAMS,
    split_mode='blocked',
    base_params={'n_epochs': 60, 'patience': 15, 'hidden_dim': 64, 'embedding_dim': 32},
    permutation_test=True, n_permutations=1,
    random_seed=42, show_progress=False,
)

null_mi = result_quick.details.get('null_distribution', [None])[0]
print("--- Sanity Check ---")
print(f"MI estimate : {result_quick.mi_estimate:.3f} bits")
if null_mi is not None:
    print(f"Null MI     : {null_mi:.3f} bits")
    margin = result_quick.mi_estimate - null_mi
    print(f"Margin      : {margin:.3f} bits  "
          f"({'PASS — signal is well above null' if margin > 0.1 else 'WARNING — small margin'})")

# %% [markdown]
# The estimate should be clearly above the null. If not, check the processor
# parameters and split mode before proceeding.

# %% [markdown]
# ## Step 2: Sweep Window Size
#
# The window size determines how much temporal context is available to the model.
# Too small → the model cannot see the relevant pattern; too large → you create
# too few windows and the model overfits.
#
# We sweep over a range of window sizes (in samples at 100 Hz) and look for the
# plateau — the region where MI stops increasing as the window grows. The optimal
# window size is the smallest one in the plateau region.

# %%
window_sizes = [10, 20, 30, 50, 75, 100]  # in samples = 100–1000 ms at 100 Hz

result_window_sweep = nmi.run(
    x_data=spikes, y_data=pos_2d,
    mode='sweep',
    processor_type_x='continuous',
    processor_type_y='continuous',
    split_mode='blocked',
    base_params={'n_epochs': 60, 'patience': 15, 'hidden_dim': 64, 'embedding_dim': 32},
    sweep_grid={'window_size': window_sizes},
    random_seed=42, show_progress=False,
)

df_window = result_window_sweep.dataframe
print("--- Window Size Sweep ---")
print(df_window[['window_size', 'mi_mean']].to_string(index=False))

# Find plateau: first window where MI is within 5% of the max
max_mi = df_window['mi_mean'].max()
plateau_ws = df_window.loc[df_window['mi_mean'] >= 0.95 * max_mi, 'window_size'].min()
print(f"\nOptimal window size (95% plateau): {plateau_ws} samples "
      f"({plateau_ws / SAMPLE_RATE * 1000:.0f} ms)")

BEST_WINDOW = int(plateau_ws)
PROC_PARAMS_BEST = {'window_size': BEST_WINDOW, 'sample_rate': SAMPLE_RATE}

# %% [markdown]
# ## Step 3: Sweep Model Architecture
#
# With the window size fixed, we now find the minimal sufficient architecture.
# Overly large models waste compute and are prone to overfitting; overly small
# ones fail to capture the relevant information. We sweep `embedding_dim` and
# look for the plateau — the smallest embedding dimension that captures the
# full signal.

# %%
result_arch_sweep = nmi.run(
    x_data=spikes, y_data=pos_2d,
    mode='sweep',
    processor_type_x='continuous', processor_params_x=PROC_PARAMS_BEST,
    processor_type_y='continuous', processor_params_y=PROC_PARAMS_BEST,
    split_mode='blocked',
    base_params={'n_epochs': 80, 'patience': 20, 'hidden_dim': 64},
    sweep_grid={'embedding_dim': [8, 16, 32, 64, 128]},
    random_seed=42, show_progress=False,
)

df_arch = result_arch_sweep.dataframe
print("--- Architecture Sweep (embedding_dim) ---")
print(df_arch[['embedding_dim', 'mi_mean']].to_string(index=False))

max_mi_arch = df_arch['mi_mean'].max()
best_emb = df_arch.loc[df_arch['mi_mean'] >= 0.95 * max_mi_arch, 'embedding_dim'].min()
print(f"\nOptimal embedding_dim: {best_emb}")

BEST_EMB = int(best_emb)

# %% [markdown]
# ## Step 4: Diagnose Training — Check for Overfitting
#
# With the best architecture identified, confirm that the model is generalising
# and not memorising the training set. We use `eval_train=True` to track both
# train and test MI across epochs.

# %%
result_diag = nmi.run(
    x_data=spikes, y_data=pos_2d,
    mode='estimate',
    processor_type_x='continuous', processor_params_x=PROC_PARAMS_BEST,
    processor_type_y='continuous', processor_params_y=PROC_PARAMS_BEST,
    split_mode='blocked',
    base_params={
        'n_epochs': 120, 'patience': 30,
        'hidden_dim': 64, 'embedding_dim': BEST_EMB,
    },
    scheduler='cosine',
    eval_train=True,
    random_seed=42, show_progress=False,
)

train_hist = result_diag.details.get('train_mi_history', [])
test_hist  = result_diag.details.get('test_mi_history', [])
peak_train = max(train_hist) if train_hist else float('nan')
peak_test  = max(test_hist)  if test_hist  else float('nan')
gap        = peak_train - peak_test

print("--- Training Diagnostics ---")
print(f"Peak train MI : {peak_train:.3f} bits")
print(f"Peak test MI  : {peak_test:.3f} bits")
print(f"Gap (train-test) : {gap:.3f} bits  "
      f"({'overfitting — reduce model size or add regularisation' if gap > 0.5 else 'OK'})")
print(f"Best epoch    : {result_diag.details.get('best_epoch', '?')}")

# %% [markdown]
# If the gap is large (> 0.5 bits), reduce `embedding_dim` or add
# `dropout=0.2` and `norm_layer='layer'` before proceeding to rigorous estimation.

# %% [markdown]
# ## Step 5: Rigorous Estimate — Bias Correction
#
# The quick estimate from Step 1 is biased upward at finite sample sizes.
# `mode='rigorous'` corrects this by training on subsets of increasing size
# and extrapolating to infinite data.

# %%
result_rigorous = nmi.run(
    x_data=spikes, y_data=pos_2d,
    mode='rigorous',
    processor_type_x='continuous', processor_params_x=PROC_PARAMS_BEST,
    processor_type_y='continuous', processor_params_y=PROC_PARAMS_BEST,
    split_mode='blocked',
    base_params={
        'n_epochs': 100, 'patience': 25,
        'hidden_dim': 64, 'embedding_dim': BEST_EMB,
    },
    scheduler='cosine',
    random_seed=42, show_progress=False,
)

mi_corrected = result_rigorous.mi_estimate
mi_error     = result_rigorous.details.get('mi_error', float('nan'))
is_reliable  = result_rigorous.details.get('is_reliable', None)

print("--- Rigorous Estimate ---")
print(f"MI (corrected) : {mi_corrected:.3f} ± {mi_error:.3f} bits")
print(f"Reliable       : {is_reliable}")
if not is_reliable:
    print("  ⚠  Extrapolation may be unreliable. Consider more data or a simpler model.")

# %% [markdown]
# ## Step 6: Temporal Analysis — Lag
#
# Does the neural population encode the animal's current position, or is there
# a delay between the position and the population response? We sweep over lags
# from −500 ms to +500 ms (negative = neural activity leads behavior; positive
# = neural activity lags behavior).

# %%
# Lag in samples; with window_size=BEST_WINDOW at 100 Hz
# We sweep ±5 windows (each window is BEST_WINDOW samples / 100 Hz seconds long)
lag_range_s = np.arange(-5, 6)  # in windows

result_lag = nmi.run(
    x_data=spikes, y_data=pos_2d,
    mode='lag',
    processor_type_x='continuous',
    processor_params_x={**PROC_PARAMS_BEST, 'sample_rate': SAMPLE_RATE},
    processor_type_y='continuous',
    processor_params_y={**PROC_PARAMS_BEST, 'sample_rate': SAMPLE_RATE},
    split_mode='blocked',
    base_params={
        'n_epochs': 80, 'patience': 20,
        'hidden_dim': 64, 'embedding_dim': BEST_EMB,
    },
    lag_range=lag_range_s,
    random_seed=42, show_progress=False,
)

df_lag = result_lag.dataframe
peak_lag_row = df_lag.loc[df_lag['mi_mean'].idxmax()]
print("--- Lag Analysis ---")
print(df_lag[['lag', 'mi_mean']].to_string(index=False))
print(f"\nPeak MI at lag = {peak_lag_row['lag']} windows "
      f"({peak_lag_row['lag'] * BEST_WINDOW / SAMPLE_RATE * 1000:.0f} ms) "
      f": {peak_lag_row['mi_mean']:.3f} bits")
print("(lag=0 = simultaneous; negative = neural leads behavioral variable)")

# %% [markdown]
# ## Step 7: Conditional MI — Encoding Above and Beyond Direction
#
# The hippocampal population encodes both position (continuous) and running
# direction (binary). Does the population carry position-specific information
# above and beyond direction?
#
# We compute:
#   I(spikes ; position | direction) = I(spikes, direction ; position) - I(direction ; position)
#
# This is done via `mode='conditional'` with `z_data=direction`.

# %%
result_cond = nmi.run(
    x_data=spikes, y_data=pos_2d,
    z_data=direction_2d,
    mode='conditional',
    processor_type_x='continuous', processor_params_x=PROC_PARAMS_BEST,
    processor_type_y='continuous', processor_params_y=PROC_PARAMS_BEST,
    z_processor_type='categorical',
    z_processor_params={'window_size': BEST_WINDOW, 'sample_rate': SAMPLE_RATE},
    split_mode='blocked',
    base_params={
        'n_epochs': 100, 'patience': 25,
        'hidden_dim': 64, 'embedding_dim': BEST_EMB,
    },
    scheduler='cosine',
    random_seed=42, show_progress=False,
)

cmi         = result_cond.details.get('cmi_estimate', result_cond.mi_estimate)
mi_xz_y     = result_cond.details.get('mi_xz_y', float('nan'))
mi_z_y      = result_cond.details.get('mi_z_y', float('nan'))

print("--- Conditional MI ---")
print(f"I(spikes, direction ; position) = {mi_xz_y:.3f} bits")
print(f"I(direction ; position)         = {mi_z_y:.3f} bits")
print(f"I(spikes ; position | direction) = {cmi:.3f} bits")
if cmi > 0.05:
    print("→ The population carries position-specific information beyond direction.")
else:
    print("→ Position encoding may be fully explained by direction in this simulation.")

# %% [markdown]
# ## Step 8: Summary Report
#
# All the pieces are now in place. Here is a single print block summarising
# the full analysis — the kind of output you would generate before writing
# up the results for a manuscript.

# %%
print("=" * 60)
print("  NeuralMI Full Pipeline — Summary Report")
print("=" * 60)
print(f"  Recording:       {N_NEURONS} neurons × {N_TIMEPOINTS} time steps @ {SAMPLE_RATE} Hz")
print(f"  Window size:     {BEST_WINDOW} samples ({BEST_WINDOW / SAMPLE_RATE * 1000:.0f} ms)")
print(f"  Embedding dim:   {BEST_EMB}")
print()
print(f"  MI (quick):      {result_quick.mi_estimate:.3f} bits")
print(f"  Null MI:         {null_mi:.3f} bits  (permutation test)")
print(f"  Signal margin:   {result_quick.mi_estimate - null_mi:.3f} bits")
print()
print(f"  MI (rigorous):   {mi_corrected:.3f} ± {mi_error:.3f} bits  "
      f"(reliable={is_reliable})")
print()
peak_l = peak_lag_row['lag']
print(f"  Peak lag:        {peak_l:+d} windows "
      f"({peak_l * BEST_WINDOW / SAMPLE_RATE * 1000:+.0f} ms)")
print()
print(f"  Conditional MI:  {cmi:.3f} bits  (position | direction)")
print("=" * 60)

# %% [markdown]
# ## Visualisation: Summary Figure
#
# A single 2×2 figure covering the four key results: (1) null test,
# (2) window-size sweep, (3) lag curve, (4) rigorous bias-correction fit.

# %%
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: Null test
ax = axes[0, 0]
bars = ax.bar(['MI estimate', 'Null MI'],
              [result_quick.mi_estimate, null_mi if null_mi else 0],
              color=['steelblue', 'tomato'], alpha=0.85, width=0.4)
ax.set_ylabel('MI (bits)')
ax.set_title('Permutation Test')
ax.set_ylim(bottom=0)

# Panel 2: Window size sweep
ax = axes[0, 1]
ws_vals = df_window['window_size'].values
mi_vals = df_window['mi_mean'].values
ax.plot(ws_vals / SAMPLE_RATE * 1000, mi_vals, 'o-', color='steelblue', linewidth=2, markersize=7)
ax.axvline(BEST_WINDOW / SAMPLE_RATE * 1000, color='tomato', linestyle='--', linewidth=1.5,
           label=f'Chosen: {BEST_WINDOW / SAMPLE_RATE * 1000:.0f} ms')
ax.set_xlabel('Window size (ms)')
ax.set_ylabel('MI (bits)')
ax.set_title('Window-Size Sweep')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel 3: Lag curve
ax = axes[1, 0]
lags_ms = df_lag['lag'].values * BEST_WINDOW / SAMPLE_RATE * 1000
ax.plot(lags_ms, df_lag['mi_mean'].values, 'o-', color='darkorange', linewidth=2, markersize=7)
ax.axvline(peak_lag_row['lag'] * BEST_WINDOW / SAMPLE_RATE * 1000,
           color='tomato', linestyle='--', linewidth=1.5, label=f"Peak: {peak_l:+d} windows")
ax.set_xlabel('Lag (ms); negative = neural leads')
ax.set_ylabel('MI (bits)')
ax.set_title('Lag Analysis')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel 4: Conditional MI decomposition
ax = axes[1, 1]
labels_cond = ['I(pop, dir; pos)', 'I(dir; pos)', 'I(pop; pos | dir)']
vals_cond   = [mi_xz_y, mi_z_y, cmi]
colors_cond = ['steelblue', 'gray', 'darkorange']
ax.bar(labels_cond, vals_cond, color=colors_cond, alpha=0.85)
ax.set_ylabel('MI (bits)')
ax.set_title('Conditional MI Decomposition')
ax.set_ylim(bottom=0)

plt.suptitle('Full Pipeline Summary', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Key Takeaways from This Pipeline
#
# 1. **Always start with a sanity check.** A quick estimate + permutation test
#    costs one training run and immediately tells you whether the data alignment,
#    processor, and split mode are working correctly.
#
# 2. **Window size is the most important hyperparameter for windowed data.**
#    Sweep it first. Looking for the plateau prevents both under-windowing
#    (missing information) and over-windowing (creating too few samples).
#
# 3. **Architecture sweeps should follow, not precede, the window sweep.**
#    MI estimates are not comparable across window sizes, so the architecture
#    must be tuned for the chosen window.
#
# 4. **`eval_train=True` costs almost nothing and saves a lot of debugging.**
#    A large train-test gap early in the pipeline tells you to simplify the
#    model before investing in the rigorous analysis.
#
# 5. **`scheduler='cosine'` is a safe default upgrade** with no hyperparameters
#    to tune. Add it once you have identified the best architecture.
#
# 6. **The rigorous estimate is the number to report.** The quick estimate
#    from Step 1 is biased upward; `mode='rigorous'` removes that bias and
#    provides a confidence interval.
#
# 7. **Lag analysis is a free scientific question.** If your data has any
#    temporal structure, always check whether the encoding is instantaneous
#    or delayed — this can reveal the direction of information flow.
#
# 8. **Conditional MI separates what a variable encodes from what it shares**
#    with a known variable. Use it whenever you have multiple behavioral
#    variables and want to attribute encoding beyond one of them.
