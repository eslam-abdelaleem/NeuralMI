# tests/test_estimators.py
import pytest
import torch
import neural_mi as nmi
from neural_mi.estimators import infonce_lower_bound, nwj_lower_bound, tuba_lower_bound, smile_lower_bound
from neural_mi.training.trainer import Trainer
from neural_mi.utils import build_critic
import torch.optim as optim

# A minimal set of parameters for running a quick training session
TRAINER_PARAMS_MINIMAL = {
    'n_epochs': 10,
    'batch_size': 64,
    'learning_rate': 1e-3,
    'patience': 3
}

class TestEstimators:
    @pytest.fixture
    def scores(self):
        return torch.randn(64, 64)

    def test_infonce_bound(self, scores):
        mi = infonce_lower_bound(scores)
        assert isinstance(mi, torch.Tensor) and mi.ndim == 0

    def test_nwj_and_tuba_relationship(self, scores):
        nwj = nwj_lower_bound(scores)
        tuba = tuba_lower_bound(scores - 1.0)
        assert torch.isclose(nwj, tuba)

    @pytest.mark.parametrize("estimator_name", ['infonce', 'nwj', 'tuba', 'smile'])
    def test_estimator_accuracy_on_known_data(self, estimator_name):
        """
        Tests if the estimators can recover a known ground-truth MI.
        """
        
        ground_truth_mi = 2.0
        n_samples = 2000
        dim = 5
        x_raw, y_raw = nmi.datasets.generate_correlated_gaussians(
            n_samples=n_samples, dim=dim, mi=ground_truth_mi, use_torch=True
        )
        x_data = x_raw.reshape(n_samples, 1, dim)
        y_data = y_raw.reshape(n_samples, 1, dim)

        embedding_params = {
            'input_dim_x': dim, 'input_dim_y': dim, 'embedding_dim': 16,
            'hidden_dim': 64, 'n_layers': 2
        }
        critic = build_critic('separable', embedding_params)
        optimizer = optim.Adam(critic.parameters(), lr=TRAINER_PARAMS_MINIMAL['learning_rate'])
        estimator_fn = nmi.estimators.ESTIMATORS[estimator_name]

        trainer = Trainer(
            model=critic, estimator_fn=estimator_fn, optimizer=optimizer,
            device=torch.device('cpu')
        )
        results = trainer.train(
            x_data, y_data, n_epochs=TRAINER_PARAMS_MINIMAL['n_epochs'],
            batch_size=TRAINER_PARAMS_MINIMAL['batch_size'],
            patience=TRAINER_PARAMS_MINIMAL['patience'], verbose=False, output_units='bits'
        )
        estimated_mi = results['test_mi'] / torch.log(torch.tensor(2.0))
        assert abs(estimated_mi - ground_truth_mi) < 1.0, \
            f"Estimator '{estimator_name}' failed accuracy test. " \
            f"Expected: {ground_truth_mi}, Got: {estimated_mi:.3f}"

    def test_smile_estimator_with_clip_param_full_pipeline(self):
        """
        Tests that the 'smile' estimator's 'clip' parameter is correctly
        used within the full nmi.run pipeline.
        """
        x_data, y_data = nmi.datasets.generate_correlated_gaussians(n_samples=1000, dim=5, mi=3.0)
        x_data, y_data = x_data.T, y_data.T

        # Define the full set of base_params required by build_critic
        test_base_params = {
            'n_epochs': 5,
            'batch_size': 64,
            'embedding_dim': 8,
            'hidden_dim': 32, 
            'n_layers': 1,
            'learning_rate': 1e-3,
            'patience': 3
        }

        # Run without clipping
        results_unclipped = nmi.run(
            x_data=x_data, y_data=y_data, mode='estimate', estimator='smile',
            processor_type='continuous', processor_params={'window_size': 1},
            base_params=test_base_params,
            verbose=False, random_seed=42, n_workers=1
        )

        # Run with a strong clipping value
        results_clipped = nmi.run(
            x_data=x_data, y_data=y_data, mode='estimate', estimator='smile',
            estimator_params={'clip': 0.1}, # Apply strong clipping
            processor_type='continuous', processor_params={'window_size': 1},
            base_params=test_base_params,
            verbose=False, random_seed=42, n_workers=1
        )

        assert isinstance(results_unclipped.mi_estimate, float)
        assert isinstance(results_clipped.mi_estimate, float)
        # Assert that the clipping had a significant effect on the final result
        import numpy as np
        assert not np.isclose(results_unclipped.mi_estimate, results_clipped.mi_estimate)