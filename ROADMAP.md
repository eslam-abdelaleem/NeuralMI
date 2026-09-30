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
