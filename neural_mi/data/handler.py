# neural_mi/data/handler.py

from typing import Optional, Union, Dict, Any, Literal, Tuple
import torch
import numpy as np

from .processors import ContinuousProcessor, SpikeProcessor
from neural_mi.logger import logger
from neural_mi.exceptions import DataShapeError

ProcessorType = Literal['continuous', 'spike']

class DataHandler:
    def __init__(self, x_data: Union[np.ndarray, torch.Tensor, list],
                 y_data: Union[np.ndarray, torch.Tensor, list],
                 processor_type: Optional[ProcessorType] = None,
                 processor_params: Optional[Dict[str, Any]] = None):
        self.x_data = x_data
        self.y_data = y_data
        self.processor_type = processor_type
        self.processor_params = processor_params if processor_params is not None else {}

    def process(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.processor_type is None:
            # ... (unchanged)
            if not isinstance(self.x_data, torch.Tensor) or not isinstance(self.y_data, torch.Tensor) or self.x_data.ndim != 3 or self.y_data.ndim != 3:
                raise DataShapeError(
                    "If 'processor_type' is not specified, input data must be pre-processed 3D torch.Tensors."
                )
            return self.x_data.float(), self.y_data.float()

        if self.processor_type == 'continuous':
            proc_params = self.processor_params.copy()
            data_format = proc_params.pop('data_format', 'channels_first')
            
            processor = ContinuousProcessor(**proc_params)
            
            x_np = self.x_data.numpy() if isinstance(self.x_data, torch.Tensor) else np.array(self.x_data)
            y_np = self.y_data.numpy() if isinstance(self.y_data, torch.Tensor) else np.array(self.y_data)
            
            # *** FIX: Use the explicit data_format parameter for transposition ***
            if data_format == 'channels_last':
                logger.debug("Transposing data from (time, channels) to (channels, time).")
                x_np, y_np = x_np.T, y_np.T

            x_processed = processor.process(x_np)
            y_processed = processor.process(y_np)

        elif self.processor_type == 'spike':
            # ... (unchanged)
            processor = SpikeProcessor(**self.processor_params)
            x_list = list(self.x_data)
            y_list = list(self.y_data)
            t_start = min([ch[0] for ch in x_list + y_list if len(ch) > 0], default=0)
            t_end = max([ch[-1] for ch in x_list + y_list if len(ch) > 0], default=t_start)
            x_processed = processor.process(x_list, t_start=t_start, t_end=t_end)
            y_processed = processor.process(y_list, t_start=t_start, t_end=t_end)
        else:
            raise ValueError(f"Unknown processor_type: '{self.processor_type}'")

        x_tensor = torch.as_tensor(x_processed, dtype=torch.float32)
        y_tensor = torch.as_tensor(y_processed, dtype=torch.float32)
        
        if x_tensor.shape[0] != y_tensor.shape[0]:
             raise DataShapeError(f"Processed tensors have mismatched number of samples: X has {x_tensor.shape[0]}, Y has {y_tensor.shape[0]}.")
             
        return x_tensor, y_tensor