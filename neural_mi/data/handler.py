from typing import Optional, Union, Dict, Any, Literal
import torch
import numpy as np
from .processors import ContinuousProcessor, SpikeProcessor

ProcessorType = Literal['continuous', 'spike']

class DataHandler:
    """Handles the processing of input data for MI estimation.

    This class acts as a centralized pre-processing interface. It determines
    whether to apply a data processor (e.g., for windowing continuous data)
    or to simply validate and convert already-processed data. It ensures
    that the data passed to the analysis modules is always a 3D torch.Tensor.

    Parameters
    ----------
    x_data : np.ndarray or torch.Tensor
        The raw or pre-processed data for variable X.
    y_data : np.ndarray or torch.Tensor
        The raw or pre-processed data for variable Y.
    processor_type : {'continuous', 'spike'}, optional
        The type of processing to apply. If None, data is assumed to be
        pre-processed and 3D.
    processor_params : dict, optional
        A dictionary of parameters for the chosen processor, e.g.,
        `{'window_size': 10}`.

    Notes
    -----
    When using the 'continuous' processor, this class uses a heuristic to
    orient the data correctly. It assumes that for a 2D input array, the
    dimension with more elements is the time dimension. If your data has
    more channels than time points, this heuristic will be incorrect. In
    such cases, please ensure your data is pre-transposed to the expected
    `(n_channels, n_timepoints)` format before passing it to the function.

    """
    def __init__(
        self,
        x_data: Union[np.ndarray, torch.Tensor],
        y_data: Union[np.ndarray, torch.Tensor],
        processor_type: Optional[ProcessorType] = None,
        processor_params: Optional[Dict[str, Any]] = None
    ):
        self.x_data = x_data
        self.y_data = y_data
        self.processor_type = processor_type
        self.processor_params = processor_params if processor_params is not None else {}

    def process(self) -> (torch.Tensor, torch.Tensor):
        """Processes the input data based on the handler's configuration.

        If a `processor_type` was specified, this method applies the
        corresponding processor to the raw data. Otherwise, it validates
        that the input data is already in the required 3D format.

        Returns
        -------
        x_processed : torch.Tensor
            The processed 3D data for variable X.
        y_processed : torch.Tensor
            The processed 3D data for variable Y.

        Raises
        ------
        ValueError
            If `processor_type` is None and the input data is not 3D.
            If an unknown `processor_type` is specified.
        """
        # If no processor is specified, validate that data is already 3D
        if self.processor_type is None:
            if self.x_data.ndim != 3 or self.y_data.ndim != 3:
                raise ValueError(
                    "If 'processor_type' is not specified, input data must be "
                    "pre-processed and have 3 dimensions (samples, channels, features)."
                )
            x_processed = torch.as_tensor(self.x_data, dtype=torch.float32)
            y_processed = torch.as_tensor(self.y_data, dtype=torch.float32)
            return x_processed, y_processed

        # If a processor is specified, apply it
        if self.processor_type == 'continuous':
            processor = ContinuousProcessor(**self.processor_params)
            # Ensure data is numpy for the processor
            x_np = self.x_data.numpy() if isinstance(self.x_data, torch.Tensor) else self.x_data
            y_np = self.y_data.numpy() if isinstance(self.y_data, torch.Tensor) else self.y_data

            # Heuristic: The processor expects (channels, time). If the data has more rows
            # than columns, assume it's (time, channels) and transpose it.
            if x_np.shape[0] > x_np.shape[1]:
                x_np = x_np.T
            if y_np.shape[0] > y_np.shape[1]:
                y_np = y_np.T

            x_processed = processor.process(x_np)
            y_processed = processor.process(y_np)
        elif self.processor_type == 'spike':
            processor = SpikeProcessor(**self.processor_params)
            # Spike processor expects a list of arrays
            x_list = list(self.x_data)
            y_list = list(self.y_data)

            # Find the global time range across both datasets to ensure
            # the output tensors have the same number of windows.
            t_start_x = min([ch[0] for ch in x_list if len(ch) > 0], default=0)
            t_start_y = min([ch[0] for ch in y_list if len(ch) > 0], default=0)
            global_t_start = min(t_start_x, t_start_y)

            t_end_x = max([ch[-1] for ch in x_list if len(ch) > 0], default=global_t_start)
            t_end_y = max([ch[-1] for ch in y_list if len(ch) > 0], default=global_t_start)
            global_t_end = max(t_end_x, t_end_y)

            x_processed = processor.process(x_list, t_start=global_t_start, t_end=global_t_end)
            y_processed = processor.process(y_list, t_start=global_t_start, t_end=global_t_end)
        else:
            raise ValueError(f"Unknown processor_type: '{self.processor_type}'")

        x_processed = torch.as_tensor(x_processed, dtype=torch.float32)
        y_processed = torch.as_tensor(y_processed, dtype=torch.float32)

        return x_processed, y_processed