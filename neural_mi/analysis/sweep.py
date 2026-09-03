# neural_mi/analysis/sweep.py
"""Provides the ParameterSweep class for running hyperparameter sweeps.

This module defines the core logic for executing multiple training runs in
parallel across a grid of hyperparameters.
"""
import warnings
import torch
import itertools
import uuid
import os
import torch.multiprocessing as mp
import numpy as np
from tqdm.auto import tqdm
from typing import List, Dict, Any, Optional, Sequence

from neural_mi.analysis.task import run_training_task
from neural_mi.logger import logger, worker_init_args
from neural_mi.utils import mi_report_units
from neural_mi.utils import _configure_multiprocessing, _ensure_cpu
from neural_mi.defaults import PROCESSOR_PARAMS_SCHEMA

def _product_dict(**kwargs: Dict[str, List]) -> List[Dict[str, Any]]:
    """Helper to create a list of dictionaries from a grid."""
    keys = kwargs.keys()
    vals = kwargs.values()
    return [dict(zip(keys, instance)) for instance in itertools.product(*vals)]

class ParameterSweep:
    """Manages the execution of a hyperparameter sweep.

    This class prepares and distributes training tasks across multiple processes
    to efficiently explore a grid of hyperparameters.
    """
    def __init__(self, x_data, y_data, base_params, **kwargs):
        """
        Parameters
        ----------
        x_data : torch.Tensor
            Data for variable X.
        y_data : torch.Tensor
            Data for variable Y.
        base_params : Dict[str, Any]
            A dictionary of fixed parameters for the MI estimator's trainer.
        **kwargs : Dict[str, Any]
            Additional keyword arguments to be added to `base_params`.
        """
        self.x_data, self.y_data = x_data, y_data
        self.base_params = base_params.copy()

        # If data is already a tensor (processed), we can infer dimensions
        if isinstance(x_data, torch.Tensor) and x_data.ndim == 3:
            self.base_params.update({
                'input_dim_x': x_data.shape[1] * x_data.shape[2],
                'input_dim_y': y_data.shape[1] * y_data.shape[2] if y_data is not None else 0,
                'n_channels_x': x_data.shape[1],
                'n_channels_y': y_data.shape[1] if y_data is not None else 0,
                **kwargs
            })
        elif isinstance(x_data, tuple) and isinstance(x_data[0], torch.Tensor) and x_data[0].ndim == 3:
            # Compound "X-role" data (DualBranchEmbedding's two-tensor input,
            # mode='conditional'(align='dual_branch')) -- dims become a
            # matching 2-tuple instead of a single int, exactly what
            # DualBranchEmbedding's constructor expects as `input_dim`. Only
            # when already windowed (3-D): a raw (2-D) tuple -- shift_windows
            # reachability for dual_branch, X and C not yet windowed -- has
            # no `.shape[2]` to read yet either, and falls through to the
            # `else` branch below, dims inferred later by task.py once
            # actually windowed, same as a raw single tensor already does.
            a_data, c_data = x_data
            self.base_params.update({
                'input_dim_x': (a_data.shape[1] * a_data.shape[2], c_data.shape[1] * c_data.shape[2]),
                'input_dim_y': y_data.shape[1] * y_data.shape[2] if y_data is not None else 0,
                'n_channels_x': (a_data.shape[1], c_data.shape[1]),
                'n_channels_y': y_data.shape[1] if y_data is not None else 0,
                **kwargs
            })
        else:
             self.base_params.update(kwargs)

    def _run_parallel(self, tasks: List[tuple], n_workers: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Executes a list of prepared tasks in parallel.
        """
        if not tasks:
            logger.warning("No tasks to run. Your sweep_grid might be empty.")
            return []

        # Default to sequential if n_workers is not specified or is 1
        effective_workers = n_workers if n_workers is not None else 1

        show_progress = self.base_params.get('show_progress', True)

        if effective_workers <= 1:
            logger.info("Starting parameter sweep sequentially (n_workers=1)...")
            # Pre-flight memory warning: on-device dataset storage with many tasks
            # can exhaust accelerator/unified memory (see dataset_device param).
            _dd = self.base_params.get('dataset_device', 'cpu')
            _dd_str = str(_dd).lower()
            if _dd_str not in ('cpu', 'none') and len(tasks) > 20:
                _ds_bytes = 0
                for _arr in (self.x_data, self.y_data):
                    if _arr is None:
                        continue
                    for _part in (_arr if isinstance(_arr, tuple) else (_arr,)):
                        if isinstance(_part, torch.Tensor):
                            _ds_bytes += _part.element_size() * _part.nelement()
                        elif hasattr(_part, 'nbytes'):
                            _ds_bytes += _part.nbytes
                if _ds_bytes > 0:
                    warnings.warn(
                        f"Running {len(tasks)} sequential tasks with "
                        f"dataset_device='{_dd}' (dataset ≈ {_ds_bytes / 1e9:.2f} GB). "
                        f"On accelerators, freed tensors may linger in the allocator "
                        f"cache between tasks and exhaust system memory. If you "
                        f"experience slowdown or a system freeze, add "
                        f"dataset_device='cpu' to base_params.",
                        UserWarning,
                        stacklevel=3,
                    )
            all_results = [run_training_task(task) for task in tqdm(tasks, desc="Sequential Sweep Progress", disable=not show_progress or len(tasks) == 1)]
        else:
            logger.info(f"Starting parameter sweep with {effective_workers} workers...")
            _configure_multiprocessing()
            # Use 'spawn' start method for cross-platform safety.
            # On macOS and Windows, 'fork' is either unavailable or unsafe with
            # PyTorch's CUDA context. On Linux, 'spawn' is slightly slower than
            # 'fork' but avoids deadlocks in multi-threaded environments.
            _log_init, _log_args = worker_init_args()
            with mp.get_context("spawn").Pool(processes=effective_workers,
                                          initializer=_log_init, initargs=_log_args) as pool:
                all_results = list(tqdm(
                    pool.imap(run_training_task, tasks), total=len(tasks),
                    desc="Parameter Sweep Progress", unit="task", disable=not show_progress
                ))
        return all_results
    
    def _prepare_tasks(
        self,
        sweep_grid: Dict[str, List],
        is_proc_sweep: Optional[bool] = None,
        max_samples_per_task: Optional[int] = None,
        **kwargs,
    ) -> List[tuple]:
        """Prepares the tasks for the parameter sweep.

        Parameters
        ----------
        is_proc_sweep : bool or None, optional
            When ``True``, raw (un-processed) data is forwarded to each task
            so that each worker runs the processor independently — required
            when processor parameters are part of the sweep grid.  When
            ``False``, the pre-processed tensors stored in ``self.x_data`` are
            forwarded directly (faster; avoids repeated processing).
            If ``None`` (default), the value is inferred automatically: data
            that is already a 3-D ``torch.Tensor`` (shape ``(N, C, W)``) is
            treated as pre-processed; everything else is treated as raw.
        """
        # Auto-detect when not provided
        if is_proc_sweep is None:
            is_proc_sweep = not (isinstance(self.x_data, torch.Tensor) and self.x_data.ndim == 3)
        tasks = []
        run_id_base = str(uuid.uuid4())
        sweep_grid = sweep_grid or {}

        if self.base_params.get('critic_type') == 'concat' and 'embedding_dim' in sweep_grid:
            raise ValueError(
                "'embedding_dim' cannot be swept when critic_type='concat'. "
                "ConcatCritic has no separate embedding networks, so embedding_dim "
                "has no effect. Remove 'embedding_dim' from sweep_grid, or switch "
                "to critic_type='separable' or 'hybrid'."
            )

        param_combinations = _product_dict(**sweep_grid) if sweep_grid else [{}]

        # When data has already been pre-processed (processor ran upstream in run()),
        # the sequential-model check below is not applicable — the tensor is already
        # shaped correctly for GRU/LSTM regardless of what processor_type_x says.
        _already_preprocessed = bool(
            self.base_params.get('processor_params_x', {}) and
            self.base_params.get('processor_params_x', {}).get('preprocessed', False)
        )
        for i_combo, params in enumerate(param_combinations):
            _emb = params.get('embedding_model', self.base_params.get('embedding_model', 'mlp'))
            _proc = params.get('processor_type_x', self.base_params.get('processor_type_x', None))
            if not _already_preprocessed and _proc is None and str(_emb).lower() in ('gru', 'lstm'):
                raise ValueError(
                    f"sweep_grid contains embedding_model='{_emb}' but processor_type_x=None "
                    f"produces a StaticDataset with no time dimension. Remove 'gru'/'lstm' "
                    f"from the sweep or set a windowed processor_type_x."
                )

            current_params = {**self.base_params, **params}

            # --- SMART MODEL SAVING LOGIC ---
            base_save_path = current_params.get('save_best_model_path')
            if base_save_path and params:
                root, ext = os.path.splitext(base_save_path)
                # Create a clean suffix from the parameters being swept
                suffix = "_" + "_".join([f"{str(k)}_{str(v)}" for k, v in params.items()])
                # Remove spaces or problematic characters if any exist in the values
                suffix = suffix.replace(" ", "")
                current_params['save_best_model_path'] = f"{root}{suffix}{ext}"
            # --------------------------------
            
            # Initialize from base_params, then update from kwargs (if any), then sweep params.
            # Only inject keys that belong to the processor schema — prevents model
            # architecture params (embedding_dim, n_layers, etc.) from bleeding into
            # processor_params_x/y when both processor and model params are swept together.
            proc_type_x = self.base_params.get('processor_type_x', None)
            proc_type_y = self.base_params.get('processor_type_y', proc_type_x)
            # When processor_type is None (no processor set), fall back to the
            # union of ALL schema keys so that any legitimate processor param in
            # the sweep grid (e.g. window_size) can still reach processor_params_x/y.
            # This prevents model-arch params (embedding_dim, n_layers, …) from
            # bleeding in while remaining agnostic about which processor is used.
            _all_proc_keys = set().union(*PROCESSOR_PARAMS_SCHEMA.values())
            valid_proc_keys_x = set(PROCESSOR_PARAMS_SCHEMA.get(proc_type_x, _all_proc_keys if proc_type_x is None else []))
            valid_proc_keys_y = set(PROCESSOR_PARAMS_SCHEMA.get(proc_type_y, _all_proc_keys if proc_type_y is None else []))
            proc_params_from_sweep_x = {k: v for k, v in params.items() if k in valid_proc_keys_x}
            proc_params_from_sweep_y = {k: v for k, v in params.items() if k in valid_proc_keys_y}

            task_processor_params_x = (self.base_params.get('processor_params_x') or {}).copy()
            if 'processor_params_x' in kwargs:
                task_processor_params_x.update(kwargs['processor_params_x'])
            task_processor_params_x.update(proc_params_from_sweep_x)

            task_processor_params_y = (self.base_params.get('processor_params_y') or {}).copy()
            if 'processor_params_y' in kwargs:
                task_processor_params_y.update(kwargs['processor_params_y'])
            task_processor_params_y.update(proc_params_from_sweep_y)

            current_params.update({
                'processor_params_x': task_processor_params_x,
                'processor_params_y': task_processor_params_y,
            })
            
            if is_proc_sweep:
                # Raw data path: processor runs inside the worker, so tensors
                # must still be on CPU before crossing the process boundary.
                task_data_x = _ensure_cpu(self.x_data)
                task_data_y = _ensure_cpu(self.y_data)
            else:
                x_to_send, y_to_send = self.x_data, self.y_data
                if max_samples_per_task and isinstance(self.x_data, tuple):
                    raise NotImplementedError(
                        "max_samples_per_task is not supported with compound "
                        "(tuple) X-role data (mode='conditional'(align='dual_branch')). "
                        "Not needed for the quantities that use this path."
                    )
                if max_samples_per_task and self.x_data is not None and self.x_data.shape[0] > max_samples_per_task:
                    indices = np.random.choice(self.x_data.shape[0], max_samples_per_task, replace=False)
                    x_to_send = self.x_data[indices]
                    y_to_send = self.y_data[indices] if self.y_data is not None else None
                task_data_x = _ensure_cpu(x_to_send)
                task_data_y = _ensure_cpu(y_to_send)

            task_run_id = f"{run_id_base}_c{i_combo}"
            # A purely deterministic per-task key for run_training_task's seeding
            # (see task.py) -- unlike task_run_id above, it does not include the
            # random run_id_base prefix, so a fixed random_seed reproduces the
            # same task_seed (and therefore the same result) on every call.
            current_params['_seed_key'] = f"c{i_combo}"
            tasks.append((task_data_x, task_data_y, current_params.copy(), task_run_id))
        
        logger.debug(f"Created {len(tasks)} tasks for the sweep.")
        return tasks

    def run(self, sweep_grid: Dict[str, List], is_proc_sweep: Optional[bool] = None, n_workers: Optional[int] = None,
            max_samples_per_task: Optional[int] = None, **kwargs) -> List[Dict[str, Any]]:
        """Executes the hyperparameter sweep in parallel."""
        tasks = self._prepare_tasks(sweep_grid, is_proc_sweep, max_samples_per_task, **kwargs)
        results = self._run_parallel(tasks, n_workers)
        logger.info("Parameter sweep finished.")
        return results


#: Amplification factors at or above this are warned about.  A difference
#: carrying 10x its components' relative error is fragile enough that the point
#: estimate should not be read on its own.
AMPLIFICATION_WARN_THRESHOLD = 10.0


def amplification_factor(components: Sequence[float], result: float) -> float:
    """Error-amplification factor for a quantity built by combining MI terms.

    Every quantity with a conditioning variable is computed as a combination of
    separately-trained MI estimates rather than being estimated directly, so
    ``I(X;Y|W) = I(X,W;Y) - I(W;Y)`` and
    ``II = I(X,W;Y) - I(X;Y) - I(W;Y)``.  Subtracting two similar numbers
    cancels most of the signal and none of the error, so the *relative* error on
    the answer is larger than the relative error on either component.  This
    function returns the condition number of that combination,

    .. math:: \\kappa = \\frac{\\sum_i |t_i|}{|\\text{result}|}

    which for the two-term case is the ``(t1 + t2) / (t1 - t2)`` given in
    ``THEORY.md``.  A component-wise relative error of ``eps`` becomes roughly
    ``kappa * eps`` on the result.

    Interpreting it:

    * ``kappa ~ 1`` means almost nothing cancels; the result is essentially one
      of the components and errors pass through undamaged.
    * ``kappa >= 10`` means a small residual is being extracted from large,
      similar numbers.  A 1% component error becomes 10% or worse, and the
      result can change sign.

    The factor grows without bound as the result approaches zero, so it is
    largest for exactly the conclusion people most want to draw ("W explains
    away X").  A near-zero conditional quantity is the hardest value in the
    taxonomy to defend.

    Two caveats.  The components share data, architecture and estimator, so part
    of their bias is common-mode and cancels; ``kappa`` is therefore an upper
    bound on the damage rather than a prediction.  Working the other way, the
    joint term is always the largest of the components and so saturates the
    InfoNCE ceiling first, which biases the result toward zero.

    Parameters
    ----------
    components : sequence of float
        The component MI estimates being combined.
    result : float
        The combined quantity.

    Returns
    -------
    float
        The amplification factor, or ``inf`` when ``result`` is exactly zero.
    """
    if result == 0:
        return float('inf')
    return float(sum(abs(c) for c in components) / abs(result))


def _joint_marginal_difference(
    joint_x, joint_y, marginal_x, marginal_y,
    base_params: Dict[str, Any], sweep_grid: Optional[Dict[str, Any]], n_workers: int,
    *,
    quantity_name: str,
    joint_label: str, marginal_label: str,
    joint_key: str, marginal_key: str,
    is_proc_sweep: bool = False,
    marginal_base_params: Optional[Dict[str, Any]] = None,
) -> tuple:
    """Estimate a chain-rule difference I(joint) - I(marginal) via two
    independent ParameterSweep runs.

    Shared by conditional MI (I(X;Y|W) = I(XW;Y) - I(W;Y)) and transfer
    entropy in both directions (TE(X→Y) = I(xy_past;y_future) -
    I(y_past;y_future), and the same with X/Y swapped for TE(Y→X)) -- all
    three are the identical joint/marginal/difference/negative-value-warning
    pattern, differing only in which arrays go in and what the quantity is
    called in log/error messages.

    Parameters
    ----------
    joint_x, joint_y : torch.Tensor
        Data for the joint-sweep ParameterSweep(x_data=joint_x, y_data=joint_y).
    marginal_x, marginal_y : torch.Tensor
        Data for the marginal-sweep ParameterSweep(x_data=marginal_x, y_data=marginal_y).
    quantity_name : str
        Human-readable name of the estimated quantity for log/error/warning
        text, e.g. ``"Conditional MI"`` or ``"TE(X→Y)"``.
    joint_label, marginal_label : str
        The two MI terms' names for log text, e.g. ``"XZ;Y"`` / ``"Z;Y"``.
    joint_key, marginal_key : str
        The caller's result-dict key names for the two component MI values,
        named in the negative-value warning so a user knows where to find them.
    is_proc_sweep : bool, optional
        Pass ``True`` when ``joint_x``/``marginal_x`` are raw, unwindowed
        data (shift_windows/shift_time reachability -- the caller has
        already concatenated the conditioning variable onto X at the raw
        level, before windowing, so both sweeps window and shift their own
        copy independently). Default ``False`` matches every other caller's
        already-windowed data.
    marginal_base_params : Dict[str, Any], optional
        Use this instead of ``base_params`` for the marginal sweep only.
        Needed when the joint and marginal legs are raw, differently-shaped
        categorical concatenations (e.g. joint=XZ with two channel blocks,
        marginal=Z alone with one) that each need their own
        ``processor_params_x['_categorical_block_specs']`` -- a single
        shared ``base_params`` can't carry both.  ``None`` (default) reuses
        ``base_params`` for both sweeps, unchanged from before this
        parameter existed.

    Returns
    -------
    tuple[float, float, float, list, list]
        ``(difference, mi_joint, mi_marginal, results_joint, results_marginal)``.
    """
    logger.info(f"{quantity_name}: estimating I({joint_label})...")
    sweep_joint = ParameterSweep(x_data=joint_x, y_data=joint_y, base_params=base_params.copy())
    results_joint = sweep_joint.run(sweep_grid=sweep_grid or {}, n_workers=n_workers, is_proc_sweep=is_proc_sweep)

    logger.info(f"{quantity_name}: estimating I({marginal_label})...")
    sweep_marginal = ParameterSweep(x_data=marginal_x, y_data=marginal_y,
                                    base_params=(marginal_base_params or base_params).copy())
    results_marginal = sweep_marginal.run(sweep_grid=sweep_grid or {}, n_workers=n_workers, is_proc_sweep=is_proc_sweep)

    joint_vals = [r['train_mi'] for r in results_joint if 'train_mi' in r]
    marginal_vals = [r['train_mi'] for r in results_marginal if 'train_mi' in r]
    if not joint_vals:
        raise RuntimeError(f"{quantity_name}: all I({joint_label}) runs failed — no valid train_mi values.")
    if not marginal_vals:
        raise RuntimeError(f"{quantity_name}: all I({marginal_label}) runs failed — no valid train_mi values.")
    mi_joint = float(np.mean(joint_vals))
    mi_marginal = float(np.mean(marginal_vals))
    difference = mi_joint - mi_marginal

    amp = amplification_factor([mi_joint, mi_marginal], difference)
    # Report in the units the caller asked for. These values are nats internally
    # and the frame is converted downstream, so a message quoting the raw value
    # would not match the number the caller ends up reading.
    _scale, _units = mi_report_units(base_params)
    logger.info(
        f"{quantity_name}: I({joint_label})={mi_joint * _scale:.4f}, "
        f"I({marginal_label})={mi_marginal * _scale:.4f}, "
        f"difference={difference * _scale:.4f} {_units}, "
        f"amplification factor={amp:.1f}x."
    )

    # A negative difference is always a high-amplification case, so the two
    # conditions are reported as one warning rather than two: the amplification
    # factor is the mechanism behind the impossible sign, not a separate issue.
    if difference < 0:
        warnings.warn(
            f"{quantity_name} estimate is negative ({difference * _scale:.4f} {_units}). This is "
            f"theoretically impossible and arises from noise in the two independent "
            f"MI estimates whose difference defines it "
            f"(I({joint_label})={mi_joint * _scale:.4f}, I({marginal_label})={mi_marginal * _scale:.4f}, "
            f"error-amplification factor {amp:.0f}x). At that amplification the "
            f"components would need sub-{100.0 / amp:.2g}% accuracy for the sign of the "
            f"result to be determined at all, so the most likely reading is that the true "
            f"value is near zero rather than that it is negative. Common causes: too few "
            f"training runs (increase sweep_grid run_id range), high estimator "
            f"variance (try more epochs or a larger batch_size), or very small true "
            f"value close to zero. The raw component estimates are available in the "
            f"returned dict ('{joint_key}', '{marginal_key}') for manual inspection.",
            UserWarning, stacklevel=3,
        )
    elif amp >= AMPLIFICATION_WARN_THRESHOLD:
        warnings.warn(
            f"{quantity_name} has an error-amplification factor of {amp:.1f}x. It is a "
            f"small residual ({difference * _scale:.4f} {_units}) of two much larger estimates "
            f"(I({joint_label})={mi_joint * _scale:.4f}, I({marginal_label})={mi_marginal * _scale:.4f}), so a "
            f"relative error of eps on each component becomes roughly {amp:.0f}*eps on the "
            f"result: a 1% component error is about {amp:.0f}% here. Do not read the point "
            f"estimate on its own. Report the components ('{joint_key}', '{marginal_key}') "
            f"alongside it, check the joint term for ceiling saturation (it is the larger "
            f"of the two and saturates first, which biases the result toward zero), and "
            f"prefer more data or more training before concluding that the true value is "
            f"small.",
            UserWarning, stacklevel=3,
        )
    return difference, mi_joint, mi_marginal, results_joint, results_marginal


def _extract_embeddings(task_results: list) -> Optional[Dict[str, Any]]:
    """Pull ``embeddings_x``/``embeddings_y`` out of a ``ParameterSweep``
    task-result list and strip them from every entry.

    Shared by conditional MI, interaction information, and transfer entropy's
    joint leg -- all three are a chain-rule difference of two independently-
    trained models (see ``_joint_marginal_difference``), so ``return_embeddings``
    threaded into ``base_params`` produces an ``embeddings_x``/``embeddings_y``
    pair buried inside each task dict in the *joint* leg's result list
    (``task.py``'s extraction already runs; it was just never surfaced to the
    caller's top-level result). Uses the last entry with the key present --
    same "no natural aggregation, pick one representative result" convention
    as ``transfer.py``'s ``_extract_diagnostics`` and ``run.py``'s
    ``mode='sweep'`` embeddings handling -- but also strips the key from
    every entry (not just the ones before the chosen one), since an
    embedding array is large enough that leaving copies scattered across
    ``raw_*`` would meaningfully bloat the result, unlike a few scalar
    diagnostics.
    """
    embeddings = None
    for r in reversed(task_results):
        if 'embeddings_x' in r:
            embeddings = {'embeddings_x': r.get('embeddings_x'), 'embeddings_y': r.get('embeddings_y')}
            break
    for r in task_results:
        r.pop('embeddings_x', None)
        r.pop('embeddings_y', None)
    return embeddings