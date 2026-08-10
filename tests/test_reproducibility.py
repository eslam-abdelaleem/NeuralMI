# tests/test_reproducibility.py
"""Tests that random_seed actually reproduces results across separate run()
calls with n_workers=1, as documented in run()'s own docstring.

Found via a broad correctness audit: ParameterSweep._prepare_tasks (sweep.py)
and AnalysisWorkflow._prepare_tasks (rigorous.py) built each task's run_id
with a fresh `str(uuid.uuid4())` prefix on every call. task.py then derived
the per-task training seed by hashing that run_id -- so the effective seed
differed on every single run() invocation regardless of an explicit
random_seed, even with n_workers=1. Fixed by seeding from a separate,
purely-deterministic '_seed_key' (built from the task's combination/gamma/
subset indices) instead of the display-only run_id.
"""
import numpy as np
import neural_mi as nmi
from neural_mi import Model, Training, Split


_MODEL = Model(embedding_dim=8, hidden_dim=16)
_TRAINING = Training(n_epochs=3, batch_size=64)
_SPLIT = Split(mode='random')


def _make_data(seed=0, n=300):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 1)).astype(np.float32)
    y = (x + 0.1 * rng.normal(size=(n, 1))).astype(np.float32)
    return x, y


class TestEstimateModeReproducibility:

    def test_same_seed_reproduces_across_separate_calls(self):
        x, y = _make_data()

        def run_once():
            return nmi.run(x, y, mode='estimate', model=_MODEL, training=_TRAINING,
                           split=_SPLIT, seed=42, n_workers=1, show_progress=False).mi_estimate

        assert run_once() == run_once() == run_once()

    def test_different_seeds_need_not_reproduce(self):
        """Sanity check that the fixture actually exercises real randomness --
        otherwise a bug that ignores random_seed entirely could pass the test
        above vacuously."""
        x, y = _make_data()
        r1 = nmi.run(x, y, mode='estimate', model=_MODEL, training=_TRAINING,
                    split=_SPLIT, seed=1, n_workers=1, show_progress=False).mi_estimate
        r2 = nmi.run(x, y, mode='estimate', model=_MODEL, training=_TRAINING,
                    split=_SPLIT, seed=2, n_workers=1, show_progress=False).mi_estimate
        assert r1 != r2


class TestSweepModeReproducibility:

    def test_sweep_reproduces_across_separate_calls(self):
        x, y = _make_data()

        def run_once():
            result = nmi.run(x, y, mode='sweep', sweep_grid={'run_id': list(range(3))},
                             model=_MODEL, training=_TRAINING, split=_SPLIT,
                             seed=7, n_workers=1, show_progress=False)
            return result.details['raw_results']['train_mi'].tolist()

        assert run_once() == run_once()

    def test_sweep_tasks_get_distinct_seeds(self):
        """Different run_id values within one sweep must not collapse onto
        the same task_seed (that would defeat the point of averaging over
        repeats to estimate variance)."""
        x, y = _make_data()
        result = nmi.run(x, y, mode='sweep', sweep_grid={'run_id': list(range(3))},
                         model=_MODEL, training=_TRAINING, split=_SPLIT,
                         seed=7, n_workers=1, show_progress=False)
        train_mis = result.details['raw_results']['train_mi'].tolist()
        assert len(set(train_mis)) > 1

    def test_seed_key_does_not_leak_into_dataframe(self):
        x, y = _make_data()
        result = nmi.run(x, y, mode='sweep', sweep_grid={'run_id': list(range(2))},
                         model=_MODEL, training=_TRAINING, split=_SPLIT,
                         seed=7, n_workers=1, show_progress=False)
        assert '_seed_key' not in result.details['raw_results'].columns
        assert '_seed_key' not in result.dataframe.columns


class TestRigorousModeReproducibility:

    def test_rigorous_reproduces_across_separate_calls(self):
        x, y = _make_data(n=400)

        def run_once():
            result = nmi.run(x, y, mode='rigorous', model=_MODEL, training=_TRAINING,
                             split=_SPLIT, rigorous={'gamma_range': range(1, 4), 'min_gamma_points': 2},
                             seed=11, n_workers=1, show_progress=False)
            return result.dataframe['train_mi'].tolist()

        assert run_once() == run_once()

    def test_seed_key_does_not_leak_into_rigorous_dataframe(self):
        x, y = _make_data(n=400)
        result = nmi.run(x, y, mode='rigorous', model=_MODEL, training=_TRAINING,
                         split=_SPLIT, rigorous={'gamma_range': range(1, 4), 'min_gamma_points': 2},
                         seed=11, n_workers=1, show_progress=False)
        assert '_seed_key' not in result.dataframe.columns
