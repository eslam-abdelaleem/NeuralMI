import torch
import torch.optim as optim
from typing import Dict, Any, Optional, Union, Tuple, List
import pandas as pd
import numpy as np

from neural_mi.estimators import ESTIMATORS
from neural_mi.models.embeddings import MLP, VarMLP, BaseEmbedding
from neural_mi.models.critics import SeparableCritic, ConcatCritic
from neural_mi.training.trainer import Trainer


def build_critic(
    critic_type: str,
    embedding_params: Dict[str, Any],
    use_variational: bool = False,
    custom_embedding_model: Optional[type] = None
) -> Union[SeparableCritic, ConcatCritic]:
    """
    Dynamically builds a critic model based on the specified type.

    This is a shared utility function to ensure critics are built consistently.
    """
    input_dim_x = embedding_params['input_dim_x']
    input_dim_y = embedding_params['input_dim_y']
    hidden_dim = embedding_params['hidden_dim']
    n_layers = embedding_params['n_layers']

    if custom_embedding_model:
        if not issubclass(custom_embedding_model, BaseEmbedding):
            raise TypeError("custom_embedding_model must be a subclass of models.BaseEmbedding")
        EmbeddingModel = custom_embedding_model
    else:
        EmbeddingModel = VarMLP if use_variational else MLP

    if critic_type == 'separable':
        embed_dim = embedding_params['embedding_dim']
        embedding_net_x = EmbeddingModel(input_dim_x, hidden_dim, embed_dim, n_layers)
        embedding_net_y = EmbeddingModel(input_dim_y, hidden_dim, embed_dim, n_layers)
        return SeparableCritic(embedding_net_x, embedding_net_y)
    elif critic_type == 'concat':
        concat_input_dim = input_dim_x + input_dim_y
        embedding_net = EmbeddingModel(concat_input_dim, hidden_dim, 1, n_layers)
        return ConcatCritic(embedding_net)
    else:
        raise ValueError(f"Unknown critic_type: {critic_type}")


def run_training_task(args: Tuple[torch.Tensor, torch.Tensor, Dict[str, Any], str]) -> Dict[str, Any]:
    """
    A top-level function that can be pickled for multiprocessing.

    This function wraps the training process for a single set of parameters.
    """
    import os
    import tempfile
    import platform

    if platform.system() == "Darwin" or os.getenv("FORCE_CUSTOM_TMPDIR"):
        custom_temp = os.path.expanduser('~/.neural_mi_tmp')
        try:
            os.makedirs(custom_temp, exist_ok=True)
            os.environ['TMPDIR'] = custom_temp
            os.environ['TEMP'] = custom_temp
            os.environ['TMP'] = custom_temp
            tempfile.tempdir = custom_temp
        except (OSError, PermissionError):
            pass

    import torch.optim as optim
    from neural_mi.estimators import ESTIMATORS
    from neural_mi.training.trainer import Trainer

    x_data, y_data, params, run_id = args
    use_variational = params.get('use_variational', False)

    custom_model = params.get('custom_embedding_model')
    critic = build_critic(
        params['critic_type'],
        params,
        use_variational=use_variational,
        custom_embedding_model=custom_model
    )

    optimizer = optim.Adam(critic.parameters(), lr=params['learning_rate'])
    device = params.get('device')

    estimator_fn = ESTIMATORS[params['estimator_name']]

    trainer = Trainer(
        model=critic, estimator_fn=estimator_fn, optimizer=optimizer,
        device=device, use_variational=use_variational, beta=params.get('beta', 1.0)
    )

    results = trainer.train(
        x_data=x_data, y_data=y_data, n_epochs=params['n_epochs'],
        batch_size=params['batch_size'], patience=params['patience'],
        smoothing_sigma=params.get('smoothing_sigma', 2.0),
        median_window=params.get('median_window', 5),
        min_improvement=params.get('min_improvement', 0.001),
        run_id=run_id
    )
    return {**params, **results}


def _validate_device(device: Union[str, torch.device]) -> torch.device:
    """
    Validates the device parameter and returns a torch.device object.
    """
    if isinstance(device, torch.device):
        return device
    if not isinstance(device, str):
        raise TypeError(f"Device must be a string or torch.device, got {type(device)}")

    if device == 'auto':
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        return torch.device(device)
    except RuntimeError as e:
        raise ValueError(f"Invalid device string: '{device}'. PyTorch error: {e}")


def find_saturation_point(
    summary_df: pd.DataFrame,
    param_col: str,
    mean_col: str,
    std_col: str,
    strictness: List[float] = [1.0]
) -> Dict[float, Any]:
    """
    Naively finds the saturation point of an MI curve.
    """
    if not isinstance(strictness, list):
        strictness = [strictness]

    df = summary_df.sort_values(param_col).reset_index()
    mi_diff = df[mean_col].diff().to_numpy()

    estimated_dims = {}
    for s in strictness:
        saturation_indices = np.where(mi_diff[1:] < (s * df[std_col].iloc[1:]))[0]

        if len(saturation_indices) > 0:
            saturation_dim = df[param_col].iloc[saturation_indices[0] + 1]
        else:
            saturation_dim = df[param_col].max()
        estimated_dims[s] = saturation_dim

    return estimated_dims