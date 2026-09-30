# neural_mi/analysis/sweep.py
"""Provides the ParameterSweep class for running hyperparameter sweeps.

This module defines the core logic for executing multiple training runs in
parallel across a grid of hyperparameters.
"""
import warnings
import torch
import itertools
import uuid
import torch.multiprocessing as mp
import numpy as np
from tqdm.auto import tqdm
from typing import List, Dict, Any, Optional, Sequence

from neural_mi.analysis.task import run_training_task
from neural_mi.exceptions import CombinationWarning
from neural_mi.logger import CapturedTask, logger, released, user_stacklevel, worker_init_args
from neural_mi.embeddings_io import with_model_labels
from neural_mi.utils import mi_report_units
from neural_mi.utils import _configure_multiprocessing, _ensure_cpu
from neural_mi.defaults import PROCESSOR_PARAMS_SCHEMA

def _product_dict(**kwargs: Dict[str, List]) -> List[Dict[str, Any]]:
    """Helper to create a list of dictionaries from a grid."""
    keys = kwargs.keys()
    vals = kwargs.values()
    return [dict(zip(keys, instance)) for instance in itertools.product(*vals)]


def merge_grid_values(base_params: Dict[str, Any], values: Dict[str, Any],
                      extra_processor_params: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
    """One task's parameters: `base_params` with one grid point applied.

    Every grid value lands at the top level. A value whose key is a processor
    parameter (``window_size``, ``step_size``, ``bin_size``, ...) also lands in
    ``processor_params_x``/``processor_params_y``, since that is where the
    processors read it. Only keys the side's processor accepts are copied, so a
    model setting such as ``embedding_dim`` never reaches a processor. A side
    with no processor accepts any processor key.
    """
    params = {**base_params, **values}
    proc_type_x = base_params.get('processor_type_x', None)
    proc_type_y = base_params.get('processor_type_y', proc_type_x)
    all_keys = set().union(*PROCESSOR_PARAMS_SCHEMA.values())
    extra = extra_processor_params or {}
    for side, proc_type in (('x', proc_type_x), ('y', proc_type_y)):
        accepted = set(PROCESSOR_PARAMS_SCHEMA.get(proc_type, all_keys if proc_type is None else []))
        merged = dict(base_params.get(f'processor_params_{side}') or {})
        merged.update(extra.get(f'processor_params_{side}', {}))
        merged.update({k: v for k, v in values.items() if k in accepted})
        params[f'processor_params_{side}'] = merged
    return params

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
                        f"experience slowdown or a system freeze, set "
                        f"Training(dataset_device='cpu').",
                        UserWarning,
                        stacklevel=user_stacklevel(),
                    )
            all_results = [run_training_task(task) for task in tqdm(tasks, desc="Sweep", disable=not show_progress or len(tasks) == 1)]
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
                    released(pool.imap(CapturedTask(run_training_task), tasks)), total=len(tasks),
                    desc="Sweep", unit="task", disable=not show_progress
                ))
        return all_results
    
    def _prepare_tasks(
        self,
        sweep_grid: Dict[str, List],
        is_proc_sweep: Optional[bool] = None,
        **kwargs,
    ) -> List[tuple]:
        """Prepares the tasks for the parameter sweep.

        Parameters
        ----------
        is_proc_sweep : bool or None, optional
            When ``True``, raw (un-processed) data is forwarded to each task
            so that each worker runs the processor independently, required
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
                "ConcatCritic has no separate embedding networks and ignores "
                "embedding_dim. Remove 'embedding_dim' from sweep_grid or switch "
                "to critic_type='separable' or 'hybrid'."
            )

        param_combinations = _product_dict(**sweep_grid) if sweep_grid else [{}]

        # When data has already been pre-processed (processor ran upstream in run()),
        # the sequential-model check below is not applicable, the tensor is already
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
                    f"sweep_grid contains embedding_model='{_emb}'. That encoder needs a time "
                    f"axis. X has no processor and so no time axis. Remove 'gru'/'lstm' from the "
                    f"sweep or set Processing(x=...) to a windowed processor."
                )

            current_params = merge_grid_values(
                self.base_params, params,
                {k: kwargs[k] for k in ('processor_params_x', 'processor_params_y') if k in kwargs})

            # Each network saved from a grid is named by its grid values.
            current_params = with_model_labels(current_params, **params)

            if is_proc_sweep:
                # Raw data path: processor runs inside the worker, so tensors
                # must still be on CPU before crossing the process boundary.
                task_data_x = _ensure_cpu(self.x_data)
                task_data_y = _ensure_cpu(self.y_data)
            else:
                task_data_x = _ensure_cpu(self.x_data)
                task_data_y = _ensure_cpu(self.y_data)

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
            **kwargs) -> List[Dict[str, Any]]:
        """Executes the hyperparameter sweep in parallel."""
        tasks = self._prepare_tasks(sweep_grid, is_proc_sweep, **kwargs)
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
    separately-trained MI estimates instead of being estimated directly, so
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
    bound on the damage instead of a prediction.  Working the other way, the
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
        The amplification factor, ``inf`` when ``result`` is exactly zero, or
        ``nan`` when every component is zero, where nothing is combined.
    """
    total = sum(abs(c) for c in components)
    if total == 0:
        return float('nan')
    if result == 0:
        return float('inf')
    return float(total / abs(result))


def warn_combination(quantity_name: str, result: float, joint: tuple, marginals: Sequence[tuple],
                     base_params: Dict[str, Any], *, signed: bool = False,
                     raw_key: str = 'mi_raw') -> None:
    """Warn when a quantity combined from MI estimates cannot be read as it stands.

    ``joint`` and each of ``marginals`` are ``(label, value, key)``: the term's
    name for the message, its estimate in nats, and the key it is reported
    under. Each marginal is contained in the joint term, so the joint cannot
    carry less information than any of them. A joint estimate that came out
    below a marginal is reported first, and otherwise a high amplification
    factor. ``signed`` marks a quantity whose negative values are meaningful,
    such as interaction information. Any other quantity is reported as 0 when
    it comes out negative, with the measured value kept under ``raw_key``.
    """
    _scale, _units = mi_report_units(base_params)
    j_label, j_value, _ = joint
    terms = [joint, *marginals]
    keys = ', '.join(f"'{key}'" for _, _, key in terms)
    short = [m for m in marginals if j_value < m[1]]
    if short:
        m_label, m_value, _ = max(short, key=lambda m: m[1])
        pair_amp = amplification_factor([j_value, m_value], j_value - m_value)
        order = (f"The joint term I({j_label})={j_value * _scale:.4f} came out below "
                 f"the I({m_label})={m_value * _scale:.4f} it contains.")
        if pair_amp >= AMPLIFICATION_WARN_THRESHOLD:
            reading = (f"The two are close enough (error-amplification factor {pair_amp:.0f}x) "
                       f"for an error of about {100.0 / pair_amp:.2g}% in either one to flip the "
                       f"order. The true difference between them is most likely near zero. More "
                       f"repeats (a longer sweep_grid run_id range) or more data narrow it.")
        else:
            reading = (f"At this error-amplification factor ({pair_amp:.1f}x) flipping the order "
                       f"takes an error of more than {100.0 / pair_amp:.2g}% in a component. One "
                       f"of the two networks most likely fell short. The joint network is the "
                       f"usual one because its input is larger. Train longer or with more "
                       f"capacity before reading the result.")
        if signed:
            warnings.warn(
                f"The components of {quantity_name} ({result * _scale:.4f} {_units}) are in an "
                f"impossible order. {order} {reading} The component estimates are in "
                f"result.runs ({keys}).",
                CombinationWarning, stacklevel=user_stacklevel(),
            )
        else:
            warnings.warn(
                f"{quantity_name} estimate is negative ({result * _scale:.4f} {_units}). The "
                f"quantity cannot be negative. It is reported as 0 {_units} and the measured value "
                f"is kept as {raw_key} in result.runs. {order} {reading} The component estimates "
                f"are in result.runs ({keys}).",
                CombinationWarning, stacklevel=user_stacklevel(),
            )
        return
    amp = amplification_factor([value for _, value, _ in terms], result)
    if amp >= AMPLIFICATION_WARN_THRESHOLD:
        count = {2: 'two', 3: 'three'}.get(len(terms), str(len(terms)))
        listing = ', '.join(f"I({label})={value * _scale:.4f}" for label, value, _ in terms)
        warnings.warn(
            f"{quantity_name} has an error-amplification factor of {amp:.1f}x. It is a "
            f"small residual ({result * _scale:.4f} {_units}) of {count} much larger estimates "
            f"({listing}). A relative error of eps on each component becomes roughly "
            f"{amp:.0f}*eps on the result (about {amp:.0f}% for a 1% component error). "
            f"Report the components ({keys}) beside the point estimate. The joint term is "
            f"the largest. It reaches its ceiling first and then biases the result downward. "
            f"Check it for ceiling saturation. Look for more data or more training "
            f"before concluding that the true value is small.",
            CombinationWarning, stacklevel=user_stacklevel(),
        )


def combined_spread(value_lists, signs) -> Optional[float]:
    """Run-to-run spread of a signed combination of component estimates.

    A quantity built by subtraction reports the difference of the components'
    means. Taking the same combination run by run instead leaves that number
    exactly where it was, because for equal-length lists

    .. math:: \\overline{a} - \\overline{b} = \\overline{(a - b)}

    and it makes the spread of the combination available, which the mean of
    each component separately cannot give. That spread is what says whether a
    difference is resolved at all: a conditional quantity whose spread exceeds
    its own value has neither a size nor a sign worth reading.

    The pairing across components is arbitrary and that is correct here. Run
    *r* of the joint and run *r* of the marginal are independent draws, so the
    variance of their difference is the sum of their variances, the
    quantity being asked for.

    Returns ``None`` when there are fewer than two runs, or when a component
    lost runs to failures and the lists no longer line up. Both cases mean
    there is no spread to report instead of a spread of zero.
    """
    n = len(value_lists[0])
    if n < 2 or any(len(values) != n for values in value_lists):
        return None
    combined = [sum(sign * values[i] for sign, values in zip(signs, value_lists))
                for i in range(n)]
    return float(np.std(combined, ddof=1))


def _joint_marginal_difference(
    joint_x, joint_y, marginal_x, marginal_y,
    base_params: Dict[str, Any], sweep_grid: Optional[Dict[str, Any]], n_workers: int,
    *,
    quantity_name: str,
    joint_label: str, marginal_label: str,
    joint_key: str, marginal_key: str,
    is_proc_sweep: bool = False,
    marginal_base_params: Optional[Dict[str, Any]] = None,
    reported: bool = True,
    raw_key: str = 'mi_raw',
) -> tuple:
    """Estimate a chain-rule difference I(joint) - I(marginal) via two
    independent ParameterSweep runs.

    Shared by conditional MI (I(X;Y|W) = I(XW;Y) - I(W;Y)) and transfer
    entropy in both directions (TE(X→Y) = I(xy_past;y_future) -
    I(y_past;y_future), and the same with X/Y swapped for TE(Y→X)). All
    three follow the identical joint/marginal/difference/negative-value-warning
    pattern and differ only in which arrays go in and what the quantity is
    called in log and error messages.

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
        data (shift_windows/shift_time reachability. The caller has
        already concatenated the conditioning variable onto X at the raw
        level, before windowing, so both sweeps window and shift their own
        copy independently). Default ``False`` matches every other caller's
        already-windowed data.
    marginal_base_params : Dict[str, Any], optional
        Use this instead of ``base_params`` for the marginal sweep only.
        Needed when the joint and marginal legs are raw, differently-shaped
        categorical concatenations (e.g. joint=XZ with two channel blocks,
        marginal=Z alone with one) that each need their own
        ``processor_params_x['_categorical_block_specs']``, and a single
        shared ``base_params`` cannot carry both. ``None`` (default) reuses
        ``base_params`` for both sweeps.
    reported : bool, optional
        ``True`` when the difference is the quantity the caller reads: it is
        checked with :func:`warn_combination`, and returned as 0 when it comes
        out negative. Pass ``False`` when it is one term of a larger quantity,
        whose caller checks that quantity itself.
    raw_key : str, optional
        Where the caller keeps the measured value, named in the warning.

    Returns
    -------
    tuple[float, float, float, list, list, tuple[list, list]]
        ``(difference, mi_joint, mi_marginal, results_joint, results_marginal,
        per_run)``, where ``per_run`` is the two components' ``train_mi``
        values one row per run. :func:`combined_spread` turns those into the
        spread of the difference.
    """
    logger.info(f"{quantity_name}: estimating I({joint_label})...")
    sweep_joint = ParameterSweep(x_data=joint_x, y_data=joint_y,
                                 base_params=with_model_labels(base_params, component=joint_key).copy())
    results_joint = sweep_joint.run(sweep_grid=sweep_grid or {}, n_workers=n_workers, is_proc_sweep=is_proc_sweep)

    logger.info(f"{quantity_name}: estimating I({marginal_label})...")
    sweep_marginal = ParameterSweep(x_data=marginal_x, y_data=marginal_y,
                                    base_params=with_model_labels(marginal_base_params or base_params,
                                                                  component=marginal_key).copy())
    results_marginal = sweep_marginal.run(sweep_grid=sweep_grid or {}, n_workers=n_workers, is_proc_sweep=is_proc_sweep)

    joint_vals = [r['train_mi'] for r in results_joint if 'train_mi' in r]
    marginal_vals = [r['train_mi'] for r in results_marginal if 'train_mi' in r]
    per_run = (joint_vals, marginal_vals)
    if not joint_vals:
        raise RuntimeError(f"{quantity_name}: all I({joint_label}) runs failed, no valid train_mi values.")
    if not marginal_vals:
        raise RuntimeError(f"{quantity_name}: all I({marginal_label}) runs failed, no valid train_mi values.")
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

    if reported:
        warn_combination(quantity_name, difference, (joint_label, mi_joint, joint_key),
                         [(marginal_label, mi_marginal, marginal_key)], base_params,
                         raw_key=raw_key)
        difference = max(difference, 0.0)
    return difference, mi_joint, mi_marginal, results_joint, results_marginal, per_run
