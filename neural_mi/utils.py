import torch
import torch.optim as optim

from neural_mi.estimators import ESTIMATORS
from neural_mi.models.embeddings import MLP, VarMLP, BaseEmbedding
from neural_mi.models.critics import SeparableCritic, ConcatCritic
from neural_mi.training.trainer import Trainer


def build_critic(critic_type, embedding_params, use_variational=False, custom_embedding_model=None):
    """
    Dynamically builds a critic model based on the specified type.

    This is a shared utility function to ensure critics are built consistently.
    """
    input_dim_x = embedding_params['input_dim_x']
    input_dim_y = embedding_params['input_dim_y']
    hidden_dim = embedding_params['hidden_dim']
    n_layers = embedding_params['n_layers']

    # Use custom model if provided, otherwise default to MLP/VarMLP
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


def run_training_task(args):
    """
    A top-level function that can be pickled for multiprocessing.

    This function wraps the training process for a single set of parameters.
    """
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
    device = get_device()

    # Look up the estimator function from its name to avoid pickling issues
    estimator_fn = ESTIMATORS[params['estimator_name']]

    trainer = Trainer(
        model=critic, estimator_fn=estimator_fn, optimizer=optimizer,
        device=device, use_variational=use_variational, beta=params.get('beta', 1.0)
    )

    results = trainer.train(
        x_data=x_data, y_data=y_data, n_epochs=params['n_epochs'],
        batch_size=params['batch_size'], patience=params['patience'], run_id=run_id
    )
    return {**params, **results}


def get_device():
    """
    Selects the appropriate device (CUDA or CPU) for tensor computations.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def find_saturation_point(summary_df, param_col, mean_col, std_col, strictness=[1.0]):
    """
    Naively finds the saturation point of an MI curve.

    This function identifies the "elbow" of a curve, which is taken as the
    point where the increase in MI begins to level off.

    Parameters
    ----------
    summary_df : pd.DataFrame
        A DataFrame with columns for the parameter, mean MI, and std MI.
    param_col : str
        The name of the column containing the swept parameter.
    mean_col : str
        The name of the column containing the mean MI.
    std_col : str
        The name of the column containing the std of the MI.
    strictness : list, optional
        A list of strictness values to test. A higher value means the
        saturation point is detected earlier.

    Returns
    -------
    dict
        A dictionary mapping each strictness value to the estimated
        saturation dimension.
    """
    import pandas as pd
    import numpy as np
    import torch

    if not isinstance(strictness, list):
        strictness = [strictness]

    df = summary_df.sort_values(param_col).reset_index()
    mi_diff = df[mean_col].diff().to_numpy()

    estimated_dims = {}
    for s in strictness:
        # Condition: Increase in MI is less than strictness * std of the current point
        saturation_indices = np.where(mi_diff[1:] < (s * df[std_col].iloc[1:]))[0]

        if len(saturation_indices) > 0:
            # The first index where this is true corresponds to the saturation dimension
            saturation_dim = df[param_col].iloc[saturation_indices[0] + 1]
        else:
            # If no saturation is found, return the max embedding dim
            saturation_dim = df[param_col].max()
        estimated_dims[s] = saturation_dim

    return estimated_dims