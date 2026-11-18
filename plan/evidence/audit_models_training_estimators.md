1. MAP

- models/embeddings.py (1454 lines): every encoder, plus input scaling and the variational wrapper. String names (in utils._EMBEDDING_CLASSES): mlp=MLP, cnn=CNN1D, cnn2d=CNN2D, gru=GRU, lstm=LSTM, tcn=TCN, transformer=Transformer, pretrained_backbone=PretrainedBackboneEmbedding, lru=LRUEmbedding, deepsets=DeepSets. Classes with no string name: BranchedEmbedding (the X encoder of a difference term), VariationalWrapper, ChannelScaling and InputScaling, and the helpers LRULayer, LRUBlock and LockedDropout. The shared parts are BaseEmbedding (input_style, zero_preserving, settings, readout) and the norm helpers _make_norm and _normed. Called by utils.build_critic and decoders.
- models/critics.py: SeparableCritic, HybridCritic, ConcatCritic and BaseCritic, with CRITICS = {separable, hybrid, concat}. Also embeds_each_side, side_embedding_sizes, fit_input_scaling and has_input_scaling. Used by utils, trainer, task and run.
- models/decoders.py: one decoder per encoder for `use_decoder` (the reconstruction loss), plus PairDecoder and build_decoder (an if/elif chain on the name). Its only caller is analysis/task.py:323, through a lazy import. It is not exported from models/__init__ because nothing outside task uses it.
- training/trainer.py: the Trainer class (`__init__`, plus train(), which is 995 lines), split helpers, blocked evaluation, the decoder loss, AMP, seconds_per_step, and checkpoint copies. Used by analysis/task.py. dimensionality.py, precision.py, run.py and rigorous.py reach into its private or split helpers.
- training/epochs.py: chooses the epoch (choose_epoch, top_band, band_floor), diagnoses the curve, and holds the warning texts. Used by trainer, results, sweep and run.
- estimators: bounds.py has infonce and smile, which are registered, plus tuba, nwj and js_fgan, which are not. ksg.py has ksg_mi and ksg_cmi. The registries are ESTIMATORS {infonce, smile}, NONPARAMETRIC {ksg} and NONPARAMETRIC_CONDITIONAL {ksg}.
- embeddings_io.py: model-file naming (resolve_model_path, model_file, with_model_labels, announce_saving_several, saved_paths), save_network, and the public extract_embeddings. Used by run, quantities, task and 6 analysis modules.
- utils.py (lines 468–1060): the critic factory, the custom-critic recipe handling, input-scaling wiring and build_optimizer_and_scheduler. The rest of utils covers device, windows/offsets, lag shifting, spectral SVD and split-leak warnings.

2. FINDINGS (most valuable first)

1. **The encoder name list lives in at least 6 places, so adding an encoder takes many edits.** utils.py:30 `_EMBEDDING_CLASSES`, validation.py:45 (a literal list), the decoders.py:563–594 if/elif chain, task.py:242 `_4D_NATIVE`, dimensionality.py:241, plus comments in config.py:86 and defaults.py:129. INTERNALS.md:371 tells contributors to edit utils and validation by hand. By contrast, validation already derives estimators (from ESTIMATORS) and activations (from ACTIVATIONS) from registries. critic_type (validation.py:44 against critics.CRITICS) and optimizer/scheduler (validation.py:50–51 against utils._OPTIMIZERS/_SCHEDULER_NAMES) are duplicated the same way.
   - Change: put one ENCODERS registry in models/; give each class `decoder_cls` and `accepts_images` attributes; have validation derive its lists from the registries. Effort M, low risk.

2. **The critic factory and optimizer builder sit in utils.py, which makes utils a hub.** Lines 19–25 import every model. data/temporal.py:6 and data/static.py:6 import utils only for get_device, so the data layer drags in all of the models. run.py:202 imports the private `_EMBEDDING_CLASSES`, `_names` and `_accepts_kwarg`.
   - Change: move build_critic, _library_critic, _custom_critic, the recipe/scaling helpers and MODEL_SETTINGS/ARCHITECTURE_KEYS into models/factory.py. Move build_optimizer_and_scheduler to training/optim.py. Effort M, low risk (import paths only).
   - MODEL_SETTINGS (utils.py:515) can be computed as the union of the classes' `settings`.
   - Every `settings` dict maps each key to itself (I checked all 11), so it should be a frozenset.

3. **Trainer.train is one 995-line method (trainer.py:242–1237) with about 45 keyword arguments.** task.py:384–449 forwards them one by one, re-resolving defaults the Trainer already has. It has clear seams:
   - (a) The split helpers: _create_random_split, _create_blocked_split, _select_eval_indices, held_out_rows, rows_to_fill_a_batch, too_few_held_out. dimensionality.py:110–118 already calls them as unbound methods with a `SimpleNamespace(_rng=...)` standing in for self, and `_rng_of` (trainer.py:57) exists only for that.
   - (b) Parsing of eval_train and track_embeddings (two near-identical blocks, lines 616–684, each containing a redundant inner `if`).
   - (c) The keep/drop logic for checkpoint copies (731–861).
   - (d) The post-hoc diagnosis and warnings (943–1115), which belongs beside epochs.py.
   - (e) Embedding tracking and rotation (803–825, 1171–1201), which is analysis work.
   - (f) The decoder reconstruction report.
   - Change: move (a) to training/split.py as plain functions that take an rng. Pass a settings dataclass (the Training config) instead of 45 kwargs. Move (d) into epochs.py. Effort L, medium risk.

4. **The estimator registry is spread over several dicts, plus an identity check.** ESTIMATOR_DEFAULTS (estimators/__init__.py:30) is never read anywhere and repeats smile's clip=5.0 signature default. The trainer special-cases InfoNCE with `self.estimator_fn is infonce_lower_bound` (trainer.py:944, 1109). KSG takes about 9 branches across 6 modules: run.py:309/428/827/1588, task.py:291, rigorous.py:665, sweep.py:317, device.py:54, validation.py:49. Each one is a reasonable "trains nothing" gate.
   - Change: a single ESTIMATORS table of spec objects with fields `fn`, `trains_network`, `conditional` and `has_log_n_ceiling`; delete ESTIMATOR_DEFAULTS. Effort S–M, low risk.
   - bounds.py keeps tuba and nwj as unregistered "reference implementations" (dead for a v1.0). Remove them or register them.

5. **Saving happens two ways, and one of them is dead.** Trainer.train(save_best_model_path) writes a bare state_dict (trainer.py:935). The only caller passes None (task.py:433) and saves the bundled format through save_network instead. embeddings_io.py:223–239 keeps an "old format, raw state dict" loading path (and _REQUIRED_BUILD_KEYS) that only this dead route could produce; it is tested only by a simulated file (tests/test_embedding_extraction.py:177).
   - Change: remove the Trainer parameter and the old-format loader. Split embeddings_io: save_network and extract_embeddings go to models/io.py; the file-naming helpers (labels, paths) go to the run/analysis plumbing. Effort S, low risk, since nothing is published yet.

6. **Decoders do not follow the encoder settings and have probable bugs.**
   - task.py:335 builds both decoders from X's `embedding_model`, ignoring embedding_model_y and custom classes. Those silently fall back to MLPDecoder, as does deepsets.
   - hidden_dim given as a list (which MLP, CNN and TCN accept) would reach `nn.Linear(embedding_dim, [..])` in the decoders. This is uncertain: I could not run it because torch is not installed here.
   - TransformerDecoder silently rounds hidden_dim (decoders.py:407).
   - The build_decoder docstring mentions a "**kwargs" argument that no longer exists.
   - Change: register a decoder per encoder class (finding 1) and pass the per-side spec. Effort M.

7. **Critics: inconsistent and dead pieces.**
   - get_training_embeddings's docstring says get_embeddings wraps no_grad; it does not. The two methods are the same code.
   - Its ConcatCritic fallback `_flat(x), _flat(y)` (critics.py:290–292) cannot be reached, because task.py blocks decoders for concat, and get_embeddings raises for concat anyway.
   - `isinstance(kl, float)` (critics.py:231, 486) is never true.
   - The class-level comment block is pasted 3 times (critics.py:303–306, 352–355, 428–431).
   - ConcatCritic calls its scoring head `embedding_net`.
   - `max_n_batches` is really a row/pair budget. It is used in two different senses: rows per encoder chunk, and `max_n_batches // batch_size` rows of pairs. A separate constant, _PAIR_BLOCK_ELEMENTS, controls the fast path.
   - Encoder chunking during training saves no memory under autograd, and it splits BatchNorm statistics per chunk once a batch exceeds 512.
   - Change: rename to eval_chunk_rows, make one method, drop the dead branches. Effort S.

8. **Encoder copy-paste and inconsistent constructor arguments.**
   - GRU and LSTM (embeddings.py:404–502) are near-identical and could be one recurrent class with a cell type.
   - Xavier initialisation is written 5 times (_BaseMLP, CNN1D ×2, CNN2D ×2).
   - A hidden_dim list works for MLP, CNN, CNN2D and TCN, raises an error for GRU, LSTM, LRU and Transformer, and DeepSets silently takes hidden_dim[0] (line 871).
   - Every built-in except MLP and CNN1D swallows `**kwargs`, which only hides typos, since build_critic passes exactly the declared settings.
   - PretrainedBackbone's "older torchvision" fallback (lines 1069–1077) is dead with torchvision>=0.19 pinned in pyproject, and get_model raises ValueError, not AttributeError. Its log message uses `pytorch_predefined`, which may be None.
   - VariationalWrapper's docstring refers to the removed `VarMLP` and to `base_params`; so do decoders.py:9 and trainer.py:464.
   - Effort S–M.

9. **Small leftovers in the factory.**
   - utils.py:839–850: `bias_x = bias_y` are always equal, and `for Model_, unused in ...` is a leftover of per-side bias.
   - `_names` and `_accepts_kwarg` (utils.py:468, 644) are near-duplicates.
   - build_optimizer_and_scheduler's docstring says precision.py uses it; it does not. 'step' passes step_size and also `**scheduler_params`, which would collide if the user sets step_size. 'cosine_warmup' ignores scheduler_params.
   - critics.py imports from embeddings lazily for no reason (there is no cycle).
   - Effort S.

10. **Trainer.__init__ has no docstring and other small issues.** Its docstring sits after `self._rng = np.random` (trainer.py:161), so `__doc__` is None, which I confirmed with ast. Decoders are moved to the device twice (232 and 495). The train() results dict carries internal plumbing (`_frozen_eval_x`/`_frozen_eval_y`, used by task and precision). Precision reaches into `trainer._safe_eval_mi` and `last_train_indices`. Docstrings repeat default values (beta 1024 and others) that can drift from the schema.

Cross-area, for the other agents: utils.py also mixes spectral analysis (cross_covariance_svd), lag shifting and offsets. Precision gets its trainer passed through a `_trainer` key in the results dict.

3. KEEP

- `default_of` for every signature default.
- `BaseEmbedding.settings`, `reads`, `input_style`, `zero_preserving` and `readout` as class-declared capabilities (keep the idea; just make settings a set).
- The ARCHITECTURE_KEYS whitelist that ties build_critic to save_network.
- `_make_norm` with RMSNorm keeping zeros intact when bias=False.
- `_score_pairs`'s affine-split fast path.
- epochs.py as a pure, separately testable module.
- The recipe-only custom_critic design.
- ksg.py: compact, cited, and with a Theiler window.
- InputScaling storing its statistics as buffers, so they are saved with the network.