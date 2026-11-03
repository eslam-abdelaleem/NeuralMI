# tests/test_branched_embedding.py
"""The X encoder of every term of a difference quantity (BranchedEmbedding), and the three quantities whose
conditioning set has a window length of its own (mi_rate, instantaneous_exchange, directed_information_rate).

Ground truth is the exact Gaussian log-det formula in its conditional form.
"""
import dataclasses
import numpy as np
import pytest
import torch

import neural_mi as nmi
from neural_mi import Model, Training, Conditional
from neural_mi.analysis import task as task_module
from neural_mi.models.embeddings import BranchedEmbedding, GRU, MLP


def _components(result):
    """The component means of a one-configuration difference quantity, by name."""
    row = result.dataframe.iloc[0]
    return {c[:-len('_mean')]: row[c] for c in row.index if c.endswith('_mean')}


# --------------------------------------------------------------------------
# Self-contained ground-truth oracle (exact conditional MI, Gaussian AR(1))
# --------------------------------------------------------------------------
class _SharedLatentOracle:
    """Z_t = phi*Z_{t-1} + eta_t (eta ~ N(0,1)); X_t = a*Z_t + eps_x; Y_t = b*Z_t + eps_y."""

    def __init__(self, phi=0.85, a=1.0, b=1.0, sx=0.5, sy=0.5):
        self.phi, self.a, self.b, self.sx, self.sy = phi, a, b, sx, sy

    def _cz(self, h):
        return (self.phi ** abs(h)) / (1 - self.phi ** 2)

    def _cov_entry(self, var_i, off_i, var_j, off_j):
        h = off_j - off_i
        cz = self._cz(h)
        if var_i == 'x' and var_j == 'x':
            return self.a ** 2 * cz + (self.sx ** 2 if h == 0 else 0.0)
        if var_i == 'y' and var_j == 'y':
            return self.b ** 2 * cz + (self.sy ** 2 if h == 0 else 0.0)
        return self.a * self.b * cz

    def _cov_matrix(self, spec):
        n = len(spec)
        M = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                M[i, j] = self._cov_entry(spec[i][0], spec[i][1], spec[j][0], spec[j][1])
        return M

    def _logdet(self, spec):
        if not spec:
            return 0.0  # det of the empty (0x0) covariance matrix is 1
        _, ld = np.linalg.slogdet(self._cov_matrix(spec))
        return ld

    def cmi_bits(self, spec_a, spec_b, spec_c):
        """Exact I(A;B|C) in bits. spec_c=[] reduces to unconditioned I(A;B)."""
        ld_ac = self._logdet(spec_a + spec_c)
        ld_bc = self._logdet(spec_b + spec_c)
        ld_abc = self._logdet(spec_a + spec_b + spec_c)
        ld_c = self._logdet(spec_c)
        return float((ld_ac + ld_bc - ld_abc - ld_c) / (2 * np.log(2)))

    def mi_rate_exact(self, h, W):
        a = [('x', s) for s in range(-W, W + 1)]
        b = [('y', 0)]
        c = [('y', s) for s in range(-h, 0)]
        return self.cmi_bits(a, b, c)

    def inst_exchange_exact(self, k):
        a = [('x', 0)]
        b = [('y', 0)]
        c = [('x', s) for s in range(-k, 0)] + [('y', s) for s in range(-k, 0)]
        return self.cmi_bits(a, b, c)

    def dir_info_rate_exact(self, k):
        a = [('x', s) for s in range(-k, 1)]
        b = [('y', 0)]
        c = [('y', s) for s in range(-k, 0)]
        return self.cmi_bits(a, b, c)

    def sample(self, T, seed=0, burn=200):
        rng = np.random.default_rng(seed)
        z = 0.0
        Z = np.zeros(T + burn)
        for t in range(T + burn):
            z = self.phi * z + rng.normal()
            Z[t] = z
        Z = Z[burn:]
        x = (self.a * Z + self.sx * rng.normal(size=T)).astype(np.float32)
        y = (self.b * Z + self.sy * rng.normal(size=T)).astype(np.float32)
        return x.reshape(-1, 1), y.reshape(-1, 1)  # (T, 1) -- one channel


# The plumbing tests use a small GRU and a short budget for speed. The accuracy tests use the defaults.
_DB_MODEL = Model(embedding_model='gru', embedding_dim=8, hidden_dim=32, n_layers=1)
_TRAINING = Training(n_epochs=40, learning_rate=1e-3, batch_size=128, patience=10)
_FAST = dict(model=Model(embedding_dim=8, hidden_dim=16, n_layers=1), training=Training(n_epochs=2, patience=1),
             show_progress=False, verbose=False, seed=0)


class TestBranchedEmbeddingUnit:
    def test_a_pair_of_different_lengths(self):
        emb = BranchedEmbedding((3, 2), 16, 8, 1, branch_cls=GRU)
        assert emb((torch.randn(5, 3, 7), torch.randn(5, 2, 4))).shape == (5, 8)

    def test_one_part(self):
        emb = BranchedEmbedding(4, 16, 8, 1, branch_cls=GRU)
        assert len(emb.branches) == 1 and emb(torch.randn(5, 4, 6)).shape == (5, 8)

    def test_gradient_reaches_every_branch_and_the_head(self):
        emb = BranchedEmbedding((3, 2), 16, 8, 1, branch_cls=GRU)
        emb((torch.randn(5, 3, 7), torch.randn(5, 2, 4))).sum().backward()
        assert all(p.grad is not None for p in emb.parameters())


@pytest.fixture
def encoders(monkeypatch):
    """The X encoder of every network a call builds, in order: (class, branch classes, channels of each part)."""
    seen = []
    build = task_module.build_critic

    def spy(critic_type, params, custom_cls=None):
        critic = build(critic_type, params, custom_cls)
        net = critic.embedding_net_x
        seen.append((type(net), [type(b) for b in getattr(net, 'branches', [])], params.get('n_channels_x')))
        return critic
    monkeypatch.setattr(task_module, 'build_critic', spy)
    return seen


def _data(n=600, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, 3))
    return x, 0.5 * np.roll(x[:, :1], 1, axis=0) + rng.standard_normal((n, 1)), rng.standard_normal((n, 2))


class TestEveryTermOfADifference:
    """The joint term reads the pair (rest of X, conditioning set) on two branches; every other term reads its
    one part on one branch of the same class, under the same head."""

    def _terms(self, seen):
        return {len(branches): (cls, set(branches), channels) for cls, branches, channels in seen}

    def test_transfer(self, encoders):
        x, y, _ = _data()
        nmi.transfer_entropy(x, y, history_window=3, **_FAST)
        terms = self._terms(encoders)
        assert terms[2] == (BranchedEmbedding, {MLP}, (3, 1))      # the source's past, the target's
        assert terms[1] == (BranchedEmbedding, {MLP}, 1)

    def test_conditional(self, encoders):
        x, y, w = _data()
        nmi.run(x, y, mode='conditional', conditional=Conditional(w_data=w), **_FAST)
        assert self._terms(encoders)[2][2] == (3, 2)               # (X, W)

    def test_interaction(self, encoders):
        x, y, w = _data()
        nmi.run(x, y, mode='interaction', interaction=nmi.Interaction(w_data=w), **_FAST)
        assert self._terms(encoders)[2][2] == (2, 3)               # (W, X): the conditioning set is X
        assert {len(b) for _, b, _ in encoders} == {1, 2}

    def test_a_custom_encoder_is_the_branch(self, encoders):
        x, y, _ = _data()
        nmi.transfer_entropy(x, y, history_window=3, **{**_FAST, 'model': Model(custom_embedding_cls=GRU)})
        assert all(set(branches) == {GRU} for _, branches, _ in encoders)

    def test_a_single_mi_is_not_branched(self, encoders):
        x, y, _ = _data()
        nmi.run(x, y, mode='estimate', **_FAST)
        assert all(cls is MLP for cls, _, _ in encoders)

    def test_shared_encoder_is_refused(self):
        x, y, _ = _data()
        with pytest.raises(ValueError, match="shared_encoder"):
            nmi.transfer_entropy(x, y, history_window=3, **{**_FAST, 'model': Model(shared_encoder=True)})


class TestPairOfDifferentLengths:
    def _mismatched_data(self, N=400):
        a = np.random.randn(N, 2, 6).astype(np.float32)
        c = np.random.randn(N, 3, 4).astype(np.float32)
        y = np.random.randn(N, 2, 1).astype(np.float32)
        return a, c, y

    def test_a_pair_of_different_lengths_runs(self):
        a, c, y = self._mismatched_data()
        r = nmi.run(a, y, mode='conditional', conditional=Conditional(w_data=c),
                    model=_DB_MODEL, training=_TRAINING, show_progress=False, seed=0)
        assert np.isfinite(r.mi_estimate)
        assert 'mi_xw_y' in _components(r) and 'mi_w_y' in _components(r)

    @pytest.mark.slow
    def test_n_workers_2(self):
        a, c, y = self._mismatched_data()
        r = nmi.run(a, y, mode='conditional', conditional=Conditional(w_data=c),
                    model=_DB_MODEL, training=_TRAINING, n_workers=2, show_progress=False,
                    sweep_grid={'run_id': [0, 1]}, seed=0)
        assert np.isfinite(r.mi_estimate)

    def test_rigorous_end_to_end(self):
        np.random.seed(0)
        a, c, y = self._mismatched_data()
        r = nmi.run(a, y, mode='conditional',
                    conditional=Conditional(w_data=c, rigorous=True, gamma_range=range(1, 4)),
                    model=_DB_MODEL, training=_TRAINING, show_progress=False)
        fit = r.runs.iloc[0]
        # A ladder left with fewer than two values of gamma that produced a value
        # cannot be extrapolated and reports an unreliable NaN.
        if len(fit['gammas_used']) >= 2:
            assert np.isfinite(r.mi_estimate)
        else:
            assert not fit['is_reliable']

    def test_a_permutation_null_moves_x_of_the_pair(self):
        a, c, y = self._mismatched_data(N=300)
        r = nmi.run(a, y, mode='conditional', conditional=Conditional(w_data=c),
                    model=_DB_MODEL, training=_TRAINING, show_progress=False, seed=0,
                    permutation_test=True, n_permutations=2)
        assert len(r.details[0]['null_distribution']) == 2

    @pytest.mark.parametrize('extra', [{'use_decoder': True}, {'use_variational': True}],
                             ids=['decoder', 'variational'])
    def test_every_stage_takes_the_pair(self, extra):
        a, c, y = self._mismatched_data(N=200)
        r = nmi.run(a, y, mode='conditional', conditional=Conditional(w_data=c),
                    model=Model(embedding_model='gru', embedding_dim=8, hidden_dim=16, n_layers=1, **extra),
                    training=Training(n_epochs=2, patience=1,
                                      augmentation_params={'gaussian_noise': {'std': 0.1},
                                                           'intensity_scale': {}}),
                    show_progress=False, seed=0)
        assert np.isfinite(r.mi_estimate)

    def test_a_named_quantity_runs_its_permutation_null(self):
        """mi_rate reads the pair (X's window, Y's past), and its null moves X within it."""
        x, y = torch.randn(400, 1), torch.randn(400, 1)
        r = nmi.mi_rate(x, y, h=3, half_width=2, model=Model(embedding_dim=4, hidden_dim=8, n_layers=1),
                        training=Training(n_epochs=1, patience=1), show_progress=False, seed=0,
                        permutation_test=True, n_permutations=2)
        assert len(r.details[0]['null_distribution']) == 2

    def test_the_corruptions_act_on_each_part(self):
        from neural_mi.data.static import StaticDataset
        torch.manual_seed(0)
        pair = (torch.randn(20, 2, 6), torch.randn(20, 3, 4))
        dataset = StaticDataset(pair)
        dataset.apply_precision(0.5)
        for part, original in zip(dataset.data, pair):
            assert part.shape == original.shape
            assert torch.allclose(part, 0.5 * (torch.floor(original / 0.5) + 0.5))
        dataset.apply_noise(0.0)                          # every corruption starts from the uncorrupted pair
        assert all(torch.equal(part, original) for part, original in zip(dataset.data, pair))

    def test_the_concat_critic_takes_the_pair(self):
        a, c, y = self._mismatched_data(N=200)
        r = nmi.run(a, y, mode='conditional', conditional=Conditional(w_data=c),
                    model=Model(critic_type='concat', hidden_dim=16, n_layers=1, input_scaling='standard'),
                    training=Training(n_epochs=2, patience=1), show_progress=False, seed=0)
        assert np.isfinite(r.mi_estimate)


class TestQuantitiesShapesAndSweep:
    def test_zero_lag_forms(self):
        x, y = torch.randn(300, 1), torch.randn(300, 1)
        model = Model(embedding_dim=8, hidden_dim=16, n_layers=1)
        assert isinstance(nmi.mi_rate(x, y, h=0, half_width=5, model=model, training=_TRAINING,
                                      show_progress=False), nmi.Results)
        assert isinstance(nmi.instantaneous_exchange(x, y, k=0, model=model, training=_TRAINING,
                                                     show_progress=False), nmi.Results)

    def test_mi_rate_scalar_returns_results(self):
        x, y = torch.randn(400, 1), torch.randn(400, 1)
        r = nmi.mi_rate(x, y, h=3, half_width=5, model=_DB_MODEL, training=_TRAINING, show_progress=False)
        assert np.isfinite(r.mi_estimate)

    def test_instantaneous_exchange_scalar_returns_results(self):
        x, y = torch.randn(400, 1), torch.randn(400, 1)
        r = nmi.instantaneous_exchange(x, y, k=3, model=_DB_MODEL, training=_TRAINING, show_progress=False)
        assert np.isfinite(r.mi_estimate)

    def test_directed_information_rate_scalar_returns_results(self):
        x, y = torch.randn(400, 1), torch.randn(400, 1)
        r = nmi.directed_information_rate(x, y, k=3, model=_DB_MODEL, training=_TRAINING, show_progress=False)
        assert np.isfinite(r.mi_estimate)

    def test_mi_rate_sweep_returns_results(self):
        x, y = torch.randn(500, 1), torch.randn(500, 1)
        r = nmi.mi_rate(x, y, h=[0, 2, 4], half_width=5, model=_DB_MODEL, training=_TRAINING,
                        n_workers=2, show_progress=False)
        assert list(r.dataframe['h']) == [0, 2, 4]
        assert r.dataframe['mi_mean'].notna().all()

    def test_instantaneous_exchange_sweep_returns_results(self):
        x, y = torch.randn(500, 1), torch.randn(500, 1)
        r = nmi.instantaneous_exchange(x, y, k=[0, 2, 4], model=_DB_MODEL, training=_TRAINING,
                                       n_workers=2, show_progress=False)
        assert list(r.dataframe['k']) == [0, 2, 4]

    @pytest.mark.slow
    def test_directed_information_rate_sweep_returns_results(self):
        x, y = torch.randn(500, 1), torch.randn(500, 1)
        r = nmi.directed_information_rate(x, y, k=[1, 2, 3], model=_DB_MODEL, training=_TRAINING,
                                          n_workers=2, show_progress=False)
        assert list(r.dataframe['k']) == [1, 2, 3]


# --------------------------------------------------------------------------
# Accuracy against exact Gaussian ground truth, with the default model and
# training.
# --------------------------------------------------------------------------
class TestAccuracyAgainstOracle:
    _oracle = _SharedLatentOracle(phi=0.85, a=1.0, b=1.0, sx=0.5, sy=0.5)

    @pytest.mark.slow
    def test_mi_rate_accuracy(self):
        h, W = 3, 5
        x, y = self._oracle.sample(6000, seed=10)
        exact = self._oracle.mi_rate_exact(h, W)
        r = nmi.mi_rate(torch.from_numpy(x), torch.from_numpy(y), h=h, half_width=W,
                        show_progress=False, seed=0)
        assert abs(r.mi_estimate - exact) < 0.5

    @pytest.mark.slow
    def test_instantaneous_exchange_accuracy(self):
        k = 3
        x, y = self._oracle.sample(6000, seed=11)
        exact = self._oracle.inst_exchange_exact(k)
        r = nmi.instantaneous_exchange(torch.from_numpy(x), torch.from_numpy(y), k=k,
                                       show_progress=False, seed=0)
        assert abs(r.mi_estimate - exact) < 0.5

    @pytest.mark.slow
    def test_directed_information_rate_accuracy(self):
        k = 3
        x, y = self._oracle.sample(6000, seed=12)
        exact = self._oracle.dir_info_rate_exact(k)
        r = nmi.directed_information_rate(torch.from_numpy(x), torch.from_numpy(y), k=k,
                                          show_progress=False, seed=0)
        assert abs(r.mi_estimate - exact) < 0.5

    def test_dir_info_rate_identity_cross_check(self):
        """dir_info_rate == TE(X->Y) + instantaneous_exchange, exact on the
        oracle (residual ~1e-14 in the original harness). A test-suite
        cross-check only, not the production path (see quantities.py's
        directed_information_rate docstring for why)."""
        k = 3
        te_exact = self._oracle.cmi_bits(
            [('x', s) for s in range(-k, 0)], [('y', 0)], [('y', s) for s in range(-k, 0)],
        )
        ie_exact = self._oracle.inst_exchange_exact(k)
        dir_exact = self._oracle.dir_info_rate_exact(k)
        assert abs((te_exact + ie_exact) - dir_exact) < 1e-8


class TestPairUnderShiftWindows:
    """shift_windows for a pair X: one plan cuts X, C and Y on one grid, and
    one offset per epoch moves every stream, so the two parts of the pair
    move together and both terms read the same rows."""

    @pytest.mark.parametrize('w_params, key', [({'window_size': 8, 'step_size': 8}, 'window_size'),
                                               ({'window_size': 20, 'step_size': 10}, 'step_size')])
    def test_a_c_window_or_step_unlike_x_is_refused(self, w_params, key):
        """A C with a window or step of its own would put C on a grid of its
        own, so the two terms would read different rows."""
        np.random.seed(0)
        T = 3000
        x = np.random.randn(T, 2).astype('float32')
        c = np.random.randn(T, 1).astype('float32')
        y = np.random.randn(T, 2).astype('float32')
        processing = nmi.Processing(x='continuous', x_params={'window_size': 20, 'step_size': 20},
                                    y='continuous', y_params={'window_size': 20, 'step_size': 20})
        with pytest.raises(ValueError, match=f"Stream 'w' asks for {key}=.* The streams of one quantity "
                                             f"share one grid"):
            nmi.run(
                x, y, mode='conditional',
                conditional=Conditional(w_data=c),
                processing=dataclasses.replace(processing, w='continuous', w_params=w_params),
                model=_DB_MODEL,
                training=Training(n_epochs=2, patience=1, shift_windows=True),
                n_workers=1, show_progress=False, seed=0,
            )

    def test_c_equal_to_x_gives_near_zero_cmi_under_shift(self):
        """C as an exact copy of X carries no information beyond X, so
        I(X,C;Y) should closely match I(C;Y) and CMI should be near zero with
        shift_windows on. If the two parts of the pair moved apart, C would
        look like new information relative to X and inflate I(X,C;Y)."""
        np.random.seed(0)
        torch.manual_seed(0)
        T, window_size = 4000, 20
        x = np.random.randn(T, 2).astype('float32')
        y = np.random.randn(T, 2).astype('float32')
        c = x.copy()  # an exact copy of X: no information beyond X
        processing = nmi.Processing(x='continuous', x_params={'window_size': window_size, 'step_size': window_size},
                                    y='continuous', y_params={'window_size': window_size, 'step_size': window_size})
        results = nmi.run(
            x, y, mode='conditional',
            conditional=Conditional(w_data=c),
            processing=dataclasses.replace(processing, w='continuous', w_params={'window_size': window_size, 'step_size': window_size}),
            model=_DB_MODEL,
            training=Training(n_epochs=15, patience=5, batch_size=32, shift_windows=True),
            n_workers=1, show_progress=False, seed=0,
        )
        details = _components(results)
        assert np.isfinite(details['mi_xw_y']) and np.isfinite(details['mi_w_y'])
        assert abs(details['mi_xw_y'] - details['mi_w_y']) < 0.3, (
            f"C=X exactly should make I(X,C;Y)={details['mi_xw_y']:.3f} closely match "
            f"I(C;Y)={details['mi_w_y']:.3f}; a large gap suggests the two parts "
            f"of the pair moved apart under shift_windows."
        )
        assert abs(results.mi_estimate) < 0.3, (
            f"CMI should be near zero when C=X exactly, got {results.mi_estimate:.3f}"
        )

    def test_rigorous_with_shift_windows_cuts_the_plan(self):
        """rigorous=True with shift_windows=True: every chunk is the plan of a
        stretch of the rows, which both terms read."""
        np.random.seed(0)
        T = 2000
        x = np.random.randn(T, 2).astype('float32')
        c = np.random.randn(T, 1).astype('float32')
        y = np.random.randn(T, 2).astype('float32')
        processing = nmi.Processing(x='continuous', x_params={'window_size': 20, 'step_size': 20},
                                    y='continuous', y_params={'window_size': 20, 'step_size': 20})
        result = nmi.run(
            x, y, mode='conditional',
            conditional=Conditional(w_data=c, rigorous=True,
                                    gamma_range=range(1, 4), min_gamma_points=2),
            processing=dataclasses.replace(processing, w='continuous', w_params={'window_size': 20, 'step_size': 20}),
            model=_DB_MODEL,
            training=Training(n_epochs=1, patience=1, shift_windows=True, min_reliable_samples=1),
            n_workers=1, show_progress=False, seed=0,
        )
        assert result.details[0]['trainings'].shape[0] == 6
