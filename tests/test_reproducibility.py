# tests/test_reproducibility.py
"""Tests that random_seed actually reproduces results across separate run()
calls, at any n_workers, as documented in run()'s own docstring.

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


class TestReproducibilityUnderParallelism:
    """`run()` used to warn "Reproducibility with random_seed is not guaranteed
    with n_workers > 1" on every parallel call. It was false, and it survived
    because every test in this file pinned n_workers=1 -- the suite only ever
    checked the case the warning declared safe.

    The guarantee is real and comes from `run_training_task` re-seeding
    random/numpy/torch inside each worker from `random_seed` plus a
    deterministic per-task key, which makes worker count and scheduling order
    irrelevant. These pin it for the shared task path and for the two modes
    that dispatch differently, so the claim cannot silently regress.

    The cost of the false warning was concrete: it pushes callers onto
    n_workers=1 to protect a property they already have, which is a straight
    multiple on wall clock for the repeat-heavy `sweep_grid={'run_id': ...}`
    runs that the amplification warning tells them to do.
    """

    def test_estimate_matches_between_serial_and_parallel(self):
        x, y = _make_data()
        kw = dict(mode='estimate', model=_MODEL, training=_TRAINING,
                  split=_SPLIT, seed=42, show_progress=False,
                  sweep_grid={'run_id': [0, 1]})
        serial = nmi.run(x, y, n_workers=1, **kw).mi_estimate
        parallel = nmi.run(x, y, n_workers=2, **kw).mi_estimate
        assert serial == parallel

    def test_no_reproducibility_warning_is_emitted(self, caplog):
        x, y = _make_data()
        with caplog.at_level('WARNING', logger='neural_mi'):
            nmi.run(x, y, mode='estimate', model=_MODEL, training=_TRAINING,
                    split=_SPLIT, seed=42, n_workers=2, show_progress=False)
        assert not any('eproducibility' in r.message for r in caplog.records)

    def test_dimensionality_matches_between_serial_and_parallel(self):
        from neural_mi import Dimensionality
        rng = np.random.default_rng(0)
        x = rng.normal(size=(400, 4)).astype(np.float32)
        y = (x @ rng.normal(size=(4, 4)) + 0.2 * rng.normal(size=(400, 4))).astype(np.float32)
        kw = dict(mode='dimensionality', model=Model(embedding_dim=4, hidden_dim=16),
                  training=_TRAINING, split=_SPLIT, seed=5, show_progress=False,
                  dimensionality=Dimensionality(n_splits=2))
        # mode='dimensionality' has no mi_estimate; its result is the spectrum
        # and the stability verdict read off it, so compare those.
        r1, r2 = nmi.run(x, y, n_workers=1, **kw), nmi.run(x, y, n_workers=2, **kw)
        for col in ('pr_eig_mean', 'pr_singular_mean', 'mi_mean'):
            assert np.array_equal(r1.dataframe[col].to_numpy(),
                                  r2.dataframe[col].to_numpy()), col
        assert r1.details['n_stable_total'] == r2.details['n_stable_total']
        assert r1.details['stable_directions'] == r2.details['stable_directions']

    def test_pairwise_matches_between_serial_and_parallel(self):
        rng = np.random.default_rng(0)
        d = rng.normal(size=(300, 4)).astype(np.float32)
        kw = dict(mode='pairwise', model=Model(embedding_dim=4, hidden_dim=16),
                  training=_TRAINING, split=_SPLIT, seed=3, show_progress=False)
        m1 = np.asarray(nmi.run(d, n_workers=1, **kw).details['mi_matrix'])
        m2 = np.asarray(nmi.run(d, n_workers=2, **kw).details['mi_matrix'])
        assert np.array_equal(np.nan_to_num(m1), np.nan_to_num(m2))
