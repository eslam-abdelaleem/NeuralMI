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
from typing import List, Dict, Any, Optional

from neural_mi.analysis.task import run_training_task
from neural_mi.logger import logger
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
                    if isinstance(_arr, torch.Tensor):
                        _ds_bytes += _arr.element_size() * _arr.nelement()
                    elif hasattr(_arr, 'nbytes'):
                        _ds_bytes += _arr.nbytes
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
            with mp.get_context("spawn").Pool(processes=effective_workers) as pool:
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
                if max_samples_per_task and self.x_data is not None and self.x_data.shape[0] > max_samples_per_task:
                    indices = np.random.choice(self.x_data.shape[0], max_samples_per_task, replace=False)
                    x_to_send = self.x_data[indices]
                    y_to_send = self.y_data[indices] if self.y_data is not None else None
                task_data_x = _ensure_cpu(x_to_send)
                task_data_y = _ensure_cpu(y_to_send)

            task_run_id = f"{run_id_base}_c{i_combo}"
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


def _joint_marginal_difference(
    joint_x, joint_y, marginal_x, marginal_y,
    base_params: Dict[str, Any], sweep_grid: Optional[Dict[str, Any]], n_workers: int,
    *,
    quantity_name: str,
    joint_label: str, marginal_label: str,
    joint_key: str, marginal_key: str,
) -> tuple:
    """Estimate a chain-rule difference I(joint) - I(marginal) via two
    independent ParameterSweep runs.

    Shared by conditional MI (I(X;Y|Z) = I(XZ;Y) - I(Z;Y)) and transfer
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

    Returns
    -------
    tuple[float, float, float, list, list]
        ``(difference, mi_joint, mi_marginal, results_joint, results_marginal)``.
    """
    logger.info(f"{quantity_name}: estimating I({joint_label})...")
    sweep_joint = ParameterSweep(x_data=joint_x, y_data=joint_y, base_params=base_params.copy())
    results_joint = sweep_joint.run(sweep_grid=sweep_grid or {}, n_workers=n_workers, is_proc_sweep=False)

    logger.info(f"{quantity_name}: estimating I({marginal_label})...")
    sweep_marginal = ParameterSweep(x_data=marginal_x, y_data=marginal_y, base_params=base_params.copy())
    results_marginal = sweep_marginal.run(sweep_grid=sweep_grid or {}, n_workers=n_workers, is_proc_sweep=False)

    joint_vals = [r['train_mi'] for r in results_joint if 'train_mi' in r]
    marginal_vals = [r['train_mi'] for r in results_marginal if 'train_mi' in r]
    if not joint_vals:
        raise RuntimeError(f"{quantity_name}: all I({joint_label}) runs failed — no valid train_mi values.")
    if not marginal_vals:
        raise RuntimeError(f"{quantity_name}: all I({marginal_label}) runs failed — no valid train_mi values.")
    mi_joint = float(np.mean(joint_vals))
    mi_marginal = float(np.mean(marginal_vals))
    difference = mi_joint - mi_marginal

    logger.info(
        f"{quantity_name}: I({joint_label})={mi_joint:.4f}, I({marginal_label})={mi_marginal:.4f}, "
        f"difference={difference:.4f} nats (converted to requested output_units by the caller)."
    )

    if difference < 0:
        warnings.warn(
            f"{quantity_name} estimate is negative ({difference:.4f} nats). This is "
            f"theoretically impossible and arises from noise in the two independent "
            f"MI estimates whose difference defines it. Common causes: too few "
            f"training runs (increase sweep_grid run_id range), high estimator "
            f"variance (try more epochs or a larger batch_size), or very small true "
            f"value close to zero. The raw component estimates are available in the "
            f"returned dict ('{joint_key}', '{marginal_key}') for manual inspection.",
            UserWarning, stacklevel=3,
        )
    return difference, mi_joint, mi_marginal, results_joint, results_marginal