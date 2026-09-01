# tests/test_estimators.py
import pytest
import numpy as np
import torch
import neural_mi as nmi
from neural_mi import Model, Training, Estimator, Processing
from neural_mi.estimators import infonce_lower_bound, smile_lower_bound
from neural_mi.training.trainer import Trainer
from neural_mi.utils import build_critic
from neural_mi.data.handler import create_dataset
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

    def test_smile_bound(self, scores):
        mi = smile_lower_bound(scores)
        assert isinstance(mi, torch.Tensor) and mi.ndim == 0

    @pytest.mark.parametrize("estimator_name", ['infonce', 'smile'])
    def test_estimator_accuracy_on_known_data(self, estimator_name):
        """Each estimator recovers a known ground-truth MI through the real API.

        This goes through `nmi.run()` rather than assembling a Trainer by hand.
        The hand-rolled version tested a configuration the library never
        produces, and SMILE diverges to NaN there on every data draw tried
        (10 of 10), while InfoNCE tolerates it. That test passed only because
        the one legacy-global-RNG draw it happened to use was on the good side
        of the knife edge; giving the generator a real `seed` removed the luck
        and exposed it. Through `nmi.run()` SMILE is stable across 8 draws.

        Tolerance measured, not guessed: at this budget InfoNCE lands in
        1.97-2.04 and SMILE in 1.81-1.88 against a truth of 2.0, so the worst
        observed error is ~0.19. 0.6 is ~3x that: comfortable against seed
        noise, and failing an estimator that learned nothing. The previous
        value of 2.0 admitted anything in (0, 4), including 0.
        """
        ground_truth_mi = 2.0
        x_raw, y_raw = nmi.generators.generate_correlated_gaussians(
            n_samples=2000, dim=5, mi=ground_truth_mi, seed=0
        )
        result = nmi.run(
            np.asarray(x_raw, dtype='float32'), np.asarray(y_raw, dtype='float32'),
            mode='estimate',
            model=nmi.Model(hidden_dim=64, embedding_dim=16),
            training=nmi.Training(**TRAINER_PARAMS_MINIMAL),
            split=nmi.Split(mode='random'),
            estimator=nmi.Estimator(name=estimator_name),
            n_workers=1, show_progress=False, seed=0,
        )
        estimated_mi = result.mi_estimate
        assert abs(estimated_mi - ground_truth_mi) < 0.6, (
            f"{estimator_name} estimated {estimated_mi:.3f} against a "
            f"ground truth of {ground_truth_mi}")

    def test_smile_estimator_with_clip_param_full_pipeline(self):
        """
        Tests that the 'smile' estimator's 'clip' parameter is correctly
        used within the full nmi.run pipeline.
        """
        x_data, y_data = nmi.generators.generate_correlated_gaussians(n_samples=1000, dim=5, mi=3.0)

        model = Model(embedding_dim=8, hidden_dim=32, n_layers=1)
        training = Training(n_epochs=5, batch_size=64, learning_rate=1e-3, patience=3)
        proc = Processing(x='continuous', x_params={'window_size': 1},
                          y='continuous', y_params={'window_size': 1})

        # Run without clipping
        results_unclipped = nmi.run(
            x_data, y_data, mode='estimate', estimator='smile',
            processing=proc, model=model, training=training,
            verbose=False, seed=42, n_workers=1
        )

        # Run with a strong clipping value
        results_clipped = nmi.run(
            x_data, y_data, mode='estimate',
            estimator=Estimator(name='smile', params={'clip': 0.1}),  # strong clipping
            processing=proc, model=model, training=training,
            verbose=False, seed=42, n_workers=1
        )

        assert isinstance(results_unclipped.mi_estimate, float)
        assert isinstance(results_clipped.mi_estimate, float)
        # Assert that the clipping had a significant effect on the final result
        import numpy as np
        assert not np.isclose(results_unclipped.mi_estimate, results_clipped.mi_estimate)
