# Roadmap

## Design work

The design items aim to declare every fact about a setting or a mode in one
place. Everything else is then derived from that declaration or checked against
it. The first three items depend on each other and are best done in
order. The charectrization of the problem is correct, but solutions can vary, so whatever is below is a good suggestion at best.

### 1. One declaration per setting

**Now.** A setting such as `train_fraction` is written in up to seven places:
the config field, the conversion code that renames and routes it, a parameter of
`_run_flat()` whose default is the one that applies, the line that copies it
into the settings dictionary, its type, bound and a second default in
`defaults.py`, its allowed values in `ALLOWED_VALUES` in `validation.py`, and
its row in PARAMETERS.md. Tests pin the two defaults to each other, PARAMETERS.md
to the config classes and every setting to code that reads it. The rename table `_GRID_NAMES` in `run.py` repeats the renames in the config
classes.

**Design.** The config classes become the table. Each field declares its engine
name, type, default, bounds or allowed values, the encoders or processors it
applies to where that is restricted, and a one-line description:

```python
train_fraction: Optional[float] = setting(default=0.9, bounds=(0, 1),
                                          doc="Share of the data used for training.")
mode: Optional[str] = setting(key='split_mode', default='blocked',
                              choices=('blocked', 'random'), doc="...")
```

The field still holds `None` until the user sets it. The library can then tell a
setting the user chose from a default. The schema tables in `defaults.py`,
`ALLOWED_VALUES`, `_GRID_NAMES` and the renames in the conversion code are
generated from the declarations or deleted. PARAMETERS.md is generated from them
or checked against them.

**Public interface.** Unchanged.

**Verification.** A test that every config field is declared, a one-off
equivalence test that every generated default equals today's effective default,
the drift tests and the full suite.

### 2. One resolution step and a small entry point

**Now.** `run()` converts the config objects into keyword arguments sorted into
three groups. `_run_flat()` takes them as 75 parameters, copies them into a
settings dictionary and then validates and fills defaults. The split lets the
processor-grid loop re-enter the engine once per value of a processor setting
and keeps the public signature small. Neither needs the keyword layer that
holds the second copy of the defaults. Modes also rewrite settings partway
through a run. `mode='dimensionality'` sets `critic_type`, `norm_layer`,
`n_epochs`, `patience` and `shared_encoder` inside its own module.

**Design.** `run()` resolves the configs once into a read-only settings object
holding the user's values, the declared defaults and the mode's own overrides.
The object is validated once and records which settings the user set. The
`sweep_grid` keys and `result.params` stay the same because the object keeps
today's flat names (`split_mode`, `n_epochs`). `_execute(data, settings)`
replaces `_run_flat()`. The processor-grid loop re-enters it with an updated
copy. The engine's reads of the settings stay as they are. The three-group
routing, `_ENGINE_PARAMS`, every `_inject` line and the test file that replaces
`_run_flat()` to inspect the routing all go. Tests of the resolution step take
their place.

**Public interface.** Unchanged.

**Verification.** The full suite, the equivalence harness from the results
refactor (identical numbers on its reference calls), and the five tutorials
re-executed to identical outputs. Because the tutorials are seeded, any
difference in them is a change.

### 3. One mode registry

**Now.** What a mode accepts is spread over eight tuples in `run.py` (`_MODES`,
`_MODE_CONFIG_CLASSES`, `_PERMUTABLE_MODES`, `_WINDOWS_IN_TASK` and the four
`_SHIFT_*` tuples), the dispatch in `produce()` in `analysis/modes.py`, the
defaults some modes set inside their own code, and the capability table in
USING.md. Adding a mode takes the six steps listed in INTERNALS.md, across six
modules and two reference documents.

**Design.** Each mode is declared once: its config class, its producer, its axis
keys, what it accepts (repeats, a configuration grid, `rigorous`, a permutation
null, returned embeddings, custom split indices), whether it windows inside its
tasks, where window shifting is safe, and its default overrides. The tuples, the
dispatch and the scattered overrides are generated from the registry or deleted.
A test checks the USING.md capability table against it. Adding a mode becomes a
config class, a producer and one registry entry.

**Public interface.** Unchanged.

**Verification.** The contract test (every mode against repeats, grids and
rigorous), the full suite and a test that the registry and USING.md agree.

### 4. Split the largest modules

`run.py` (about 1,800 lines) shrinks with items 2 and 3 and should be split
after them. `training/trainer.py` (about 1,350 lines) mixes the training loop,
evaluation, the split builders and the window shifting. `analysis/rigorous.py`
(about 1,300) mixes the fit with the orchestration of the ladder.
`data/temporal.py` (about 1,250) holds all three processors.
`models/embeddings.py` (about 1,250) holds every architecture family. Each can
be split along those lines without changing what it exports.

**Public interface.** Unchanged. `neural_mi` re-exports the public names.

**Verification.** The full suite and an import check of every public name.

### 5. The engine working on config objects end to end

Items 1 to 3 give one source of truth while keeping the flat names the engine
reads. Moving the engine onto the config objects themselves would touch about
320 reads of the settings across 20 modules and 60 places in the tests. It would
also change the public `sweep_grid` keys and the layout of `result.params` and
so need a major version. It is for later, after items 1 to 3.

**Public interface.** `sweep_grid` keys and `result.params` change.

## Estimation

### Train the terms of a difference together (considered, not planned)

**Idea.** One model per difference, the conditioning set's pathway shared by
the terms, one training on the sum of their losses, and both terms read at one
epoch, so that errors in the shared pathway enter both terms alike.

**Pilot.** Transfer entropy on linear-Gaussian data, 64 source neurons over 10
bins, $2\times10^5$ samples, one recording, each term reported on its training
rows as the library reports it. With each term read at the peak of its own
held-out curve, sharing the target's-past encoder between the terms gave the
values of training them apart, from time-shifted sources and from aligned ones.
Reading both terms at one epoch raised the shifted-source values clearly above
zero. On a shifted source the joint term peaks within a few epochs and the
conditioning term much later, so no single epoch suits both. Sharing every
weight changed nothing further.

**Why not.** The branched encoder (`BranchedEmbedding`) already gives the
conditioning set the same architecture in every term, and that is what made
the errors cancel. Sharing weights would add a summed loss with a weight
between the terms, tie the terms to one training loop, and narrow what the
conditioning set's encoder can learn for the joint term, which matters where
the conditioning set acts only together with the rest. A loss on the difference
itself is not a bound and reads high by construction.

**Revisit if** the two terms' encoders of a large conditioning set are seen to
learn it differently.

### Separate settings for the terms of a quantity

**Idea.** `model=` and `training=` keep applying to every term that is trained
separately. An optional `terms=` mapping gives any of them its own settings,
keyed by the names the library already reports in
`details[config_id]['trainings']`:

| mode | terms |
|---|---|
| transfer | `i_xypast_yfuture`, `i_ypast_yfuture` (and `i_yxpast_xfuture`, `i_xpast_xfuture` with `bidirectional`) |
| conditional | `mi_xw_y`, `mi_w_y` |
| interaction | `mi_xw_y`, `mi_x_y`, `mi_w_y` |

```python
nmi.run(x, y, mode='interaction', interaction=Interaction(w_data=w),
        model=Model(hidden_dim=128, dropout=0.1),
        terms={'mi_x_y': {'model': Model(hidden_dim=64, dropout=0.2)},
               'mi_w_y': {'training': Training(learning_rate=1e-5)}})
```

A term's unset fields inherit from `model=` and `training=`, as `hidden_dim_y`
inherits `hidden_dim`. A quantity added later with more terms gets the option
by naming its terms.

**Evidence.** In the storage-transfer paper (transfer entropy on spikes, four
settings, 3 to 5 recordings per point), choosing each term's setting by its own
held-out value mattered mainly at long histories. With one source and 50 bins,
the error against the exact value fell to 0.44 mbit, from 0.80 with one fixed
setting. There the term of the target's own past did best with more
regularisation and the full term with more capacity. Elsewhere the choice made
little difference, including where only the ratio of the terms' input sizes grew
(99 sources against the target's own 5 lags).

**Scope.**
- Model and Training fields only. Split and estimator stay shared, because every
  term must evaluate the same rows. Processing stays per stream
  (`Processing(x=..., w=...)` already lets each stream be its own modality).
  Processing per term would change the conditioning variable between the terms
  and break the chain rule.
- Carried in the settings dictionary, so rigorous chunks, permutation nulls,
  both directions of transfer and the named quantities (which forward
  `**run_kwargs`) inherit it. The engine change is a lookup by term name in
  `_joint_marginal_difference`, which already takes `marginal_base_params` and
  knows both terms' names, and in interaction's `mi_w_y` term.
- Refused for a name the mode does not have (the message lists the mode's
  terms), in modes with one term, and when one field of one term is set twice:
  in `terms=` and in the grid, or in `terms=` while an unprefixed grid key varies
  it for every term.
- `result.params` records the settings of every term.

**Sweeps per term.** A grid key prefixed with a term's name varies that term
only, and unprefixed keys vary every term, as now:

```python
sweep_grid={'mi_x_y.hidden_dim': [64, 128], 'mi_w_y.learning_rate': [1e-5, 5e-6]}
```

Every combination of the keys is one configuration. A first version keeps the
existing loop of `produce_difference`: each configuration trains every term,
and the prefixed values reach their term the way `terms=` does. Rigorous mode,
the nulls and the results need nothing new, and the runs carry the prefixed keys
as columns. Its cost is retraining: k values of one term's key retrain the other
terms k times. Prefixed keys are refused outside Model and Training fields.



**Cost.** Small in the engine. Most of the work is the new setting, so it is
cheaper after design item 1.

## Estimators and quantities from other toolboxes

From a comparison with JIDT, IDTxl, HOI, Frites, MINT, PyInform, dit and
infomeasure (October 2026).

**Preprocessing, embedding and objective as separate choices.** Rationale:
several estimators of other toolboxes are combinations of the three. Options:
- preprocessing: z-score, whitening, ZCA, PCA, copula (ranks to normal);
- embedding: linear (an MLP with `n_layers=0`, already possible) or any
  existing architecture;
- objective: InfoNCE, SMILE, or a Gaussian objective (log-determinant of the
  canonical correlations: CCA with linear embeddings, deep CCA otherwise; not a
  bound, read on held-out data).

Gaussian and Gaussian-copula MI then come for free (z-score or copula, identity
embedding, Gaussian objective in closed form).

**Pointwise and chunk-wise values.** Rationale: a trained critic scores single
samples, so information can be resolved in time, by trial or by condition (local
values as in JIDT; local storage and transfer as in Fiorenza et al. 2025).
Options: a readout after training, like `mode='precision'`, or an option of any
training. Chunk values scored against the whole recording, so they average back
to the global estimate.

**Composite quantities.** Rationale: Shannon combinations of MI terms the
library already estimates, in the style of `interaction_information`. Options:
total correlation, S-information, O-information. Partial information
decomposition only as the minimum-MI version, named as such, since every
version needs an extra definition of redundancy.

**Discrete estimators for spike words.** Rationale: baselines for binned spikes
beside KSG. Options: plug-in, Miller–Madow, Panzeri–Treves, NSB (start from the
storage-transfer paper's version), quadratic extrapolation through the rigorous
ladder.

## Smaller items

**Check the README, ANATOMY and the tutorials in CI.**
`docs/figures/readme_figures.ipynb` runs the README quickstart and the code in
`reference/ANATOMY.md` only when someone runs it. A CI job that executes its
checking sections would catch that drift at the pull request. A slower job could
execute the five tutorials. The docs workflow deploys only from `main`. A
job that builds the docs on every push to `dev` and attaches the site as an
artefact would preview them without deploying.


**Archive releases on Zenodo.** JOSS asks for an archived release with a DOI at
acceptance. Connecting the repository to Zenodo gives every release one. When
the paper is accepted, `CITATION.cff` should cite it.

**Give the internal docstrings a language pass.** The printed messages and the
docstrings rendered in the API reference have had one. The other docstrings are
developer notes and have not.