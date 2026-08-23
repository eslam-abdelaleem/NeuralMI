# tests/test_dual_branch_embedding.py
"""Tests for Stage 4: DualBranchEmbedding and the three quantities that need
it (mi_rate, instantaneous_exchange, directed_information_rate), where A and
C genuinely differ in window length beyond mode='conditional''s small trim
tolerance.

Ground truth reuses the exact Gaussian log-det conditional-MI formula (the
same construction validated throughout this session's oracle work), extended
here to a conditional form so it also covers the C != [] case these three
quantities need, not just the unconditioned I(A;B) Stage 1 covers.
"""
import numpy as np
import pandas as pd
import pytest
import torch

import neural_mi as nmi
from neural_mi import Model, Training, Conditional
from neural_mi.config import Transfer
from neural_mi.models.embeddings import DualBranchEmbedding, GRU


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


_DB_MODEL = Model(embedding_model='gru', custom_embedding_cls=DualBranchEmbedding,
                   embedding_dim=8, hidden_dim=32, n_layers=1)
_TRAINING = Training(n_epochs=40, learning_rate=1e-3, batch_size=128, patience=10)


# --------------------------------------------------------------------------
# Unit tests: DualBranchEmbedding directly, no training loop
# --------------------------------------------------------------------------
class TestDualBranchEmbeddingUnit:
    def test_dual_mode_output_shape(self):
        emb = DualBranchEmbedding(input_dim=(3, 2), hidden_dim=16, embed_dim=8, n_layers=1)
        a_batch = torch.randn(5, 3, 7)   # window length 7
        c_batch = torch.randn(5, 2, 4)   # window length 4, DIFFERENT from a_batch's
        out = emb((a_batch, c_batch))
        assert out.shape == (5, 8)

    def test_single_mode_output_shape(self):
        emb = DualBranchEmbedding(input_dim=4, hidden_dim=16, embed_dim=8, n_layers=1)
        batch = torch.randn(5, 4, 6)
        out = emb(batch)
        assert out.shape == (5, 8)

    def test_dual_mode_gradient_flows_to_both_branches(self):
        emb = DualBranchEmbedding(input_dim=(3, 2), hidden_dim=16, embed_dim=8, n_layers=1)
        a_batch = torch.randn(5, 3, 7)
        c_batch = torch.randn(5, 2, 4)
        out = emb((a_batch, c_batch))
        out.sum().backward()
        for p in emb.branch_a.parameters():
            assert p.grad is not None and torch.any(p.grad != 0)
        for p in emb.branch_c.parameters():
            assert p.grad is not None and torch.any(p.grad != 0)
        for p in emb.fusion.parameters():
            assert p.grad is not None

    def test_dual_construction_rejects_plain_forward(self):
        emb = DualBranchEmbedding(input_dim=(3, 2), hidden_dim=16, embed_dim=8, n_layers=1)
        with pytest.raises(ValueError):
            emb(torch.randn(5, 3, 7))

    def test_single_construction_rejects_tuple_forward(self):
        emb = DualBranchEmbedding(input_dim=4, hidden_dim=16, embed_dim=8, n_layers=1)
        with pytest.raises(ValueError):
            emb((torch.randn(5, 4, 6), torch.randn(5, 4, 3)))

    def test_custom_branch_cls_subclass_pattern(self):
        """DualBranchEmbedding's documented subclass pattern for a non-default
        branch_cls: only RNN-family classes fit (raw channel-count input_dim,
        3D (batch, dim, len) forward), same as the default GRU branch_cls."""
        from neural_mi.models.embeddings import LSTM

        class _DualBranchLSTMMock(DualBranchEmbedding):
            def __init__(self, input_dim, hidden_dim, embed_dim, n_layers, **kwargs):
                super().__init__(input_dim, hidden_dim, embed_dim, n_layers, branch_cls=LSTM, **kwargs)

        emb = _DualBranchLSTMMock(input_dim=(3, 2), hidden_dim=16, embed_dim=8, n_layers=1)
        assert isinstance(emb.branch_a, LSTM)
        out = emb((torch.randn(5, 3, 7), torch.randn(5, 2, 4)))
        assert out.shape == (5, 8)


# --------------------------------------------------------------------------
# Integration through mode='conditional' with a genuine A/C length mismatch
# --------------------------------------------------------------------------
class TestDualBranchIntegration:
    def _mismatched_data(self, N=400):
        a = np.random.randn(N, 2, 6).astype(np.float32)
        c = np.random.randn(N, 3, 4).astype(np.float32)
        y = np.random.randn(N, 2, 1).astype(np.float32)
        return a, c, y

    def test_align_dual_branch_runs_and_returns_finite(self):
        a, c, y = self._mismatched_data()
        r = nmi.run(a, y, mode='conditional', conditional=Conditional(z_data=c, align='dual_branch'),
                    model=_DB_MODEL, training=_TRAINING, show_progress=False, seed=0)
        assert isinstance(r, nmi.Results)
        assert np.isfinite(r.mi_estimate)
        assert 'mi_xz_y' in r.details and 'mi_z_y' in r.details

    def test_align_none_still_raises_on_mismatch_beyond_tolerance(self):
        """Regression: without align='dual_branch', a mismatch this large must
        still hard-error exactly as before Stage 4 (additive, not a silent
        behavior change to the default path)."""
        a, c, y = self._mismatched_data()
        with pytest.raises(ValueError):
            nmi.run(a, y, mode='conditional', conditional=Conditional(z_data=c),
                    model=Model(embedding_dim=8, hidden_dim=16, n_layers=1),
                    training=_TRAINING, show_progress=False)

    def test_dual_branch_n_workers_2(self):
        a, c, y = self._mismatched_data()
        r = nmi.run(a, y, mode='conditional', conditional=Conditional(z_data=c, align='dual_branch'),
                    model=_DB_MODEL, training=_TRAINING, n_workers=2, show_progress=False,
                    sweep_grid={'run_id': [0, 1]}, seed=0)
        assert np.isfinite(r.mi_estimate)

    def test_dual_branch_rigorous_end_to_end(self):
        a, c, y = self._mismatched_data()
        r = nmi.run(a, y, mode='conditional',
                    conditional=Conditional(z_data=c, align='dual_branch', rigorous=True,
                                            gamma_range=range(1, 4)),
                    model=_DB_MODEL, training=_TRAINING, show_progress=False)
        assert np.isfinite(r.mi_estimate)

    def test_permutation_test_with_dual_branch_raises_clear_error(self):
        a, c, y = self._mismatched_data()
        with pytest.raises(NotImplementedError):
            nmi.run(a, y, mode='conditional', conditional=Conditional(z_data=c, align='dual_branch'),
                    model=_DB_MODEL, training=_TRAINING, show_progress=False, permutation_test=True)


# --------------------------------------------------------------------------
# Plain (non-dual-branch) paths must be unaffected
# --------------------------------------------------------------------------
class TestPlainPathsUnaffected:
    def test_plain_conditional_matched_lengths_unaffected(self):
        N = 300
        a = np.random.randn(N, 2, 5).astype(np.float32)
        c = np.random.randn(N, 2, 5).astype(np.float32)
        y = np.random.randn(N, 2, 1).astype(np.float32)
        r = nmi.run(a, y, mode='conditional', conditional=Conditional(z_data=c),
                    model=Model(embedding_dim=8, hidden_dim=16, n_layers=1),
                    training=_TRAINING, show_progress=False, seed=0)
        assert np.isfinite(r.mi_estimate)

    def test_plain_transfer_without_w_data_unaffected(self):
        N = 300
        x = np.random.randn(N, 1).astype(np.float32)
        y = np.random.randn(N, 1).astype(np.float32)
        r = nmi.run(x, y, mode='transfer', transfer=Transfer(history_window=3),
                    model=Model(embedding_dim=8, hidden_dim=16, n_layers=1),
                    training=_TRAINING, show_progress=False, seed=0)
        assert np.isfinite(r.mi_estimate)


# --------------------------------------------------------------------------
# neural_mi.quantities convenience-function wiring
# --------------------------------------------------------------------------
class TestQuantitiesRequireDualBranchModel:
    def test_mi_rate_requires_dual_branch_model(self):
        x, y = torch.randn(300, 1), torch.randn(300, 1)
        with pytest.raises(ValueError):
            nmi.mi_rate(x, y, h=3, W=5, model=Model(embedding_dim=8, hidden_dim=16, n_layers=1),
                        training=_TRAINING, show_progress=False)

    def test_instantaneous_exchange_requires_dual_branch_model(self):
        x, y = torch.randn(300, 1), torch.randn(300, 1)
        with pytest.raises(ValueError):
            nmi.instantaneous_exchange(x, y, k=3, model=Model(embedding_dim=8, hidden_dim=16, n_layers=1),
                                       training=_TRAINING, show_progress=False)

    def test_directed_information_rate_requires_dual_branch_model(self):
        x, y = torch.randn(300, 1), torch.randn(300, 1)
        with pytest.raises(ValueError):
            nmi.directed_information_rate(x, y, k=3, model=Model(embedding_dim=8, hidden_dim=16, n_layers=1),
                                          training=_TRAINING, show_progress=False)

    def test_mi_rate_h_zero_needs_no_dual_branch_model(self):
        x, y = torch.randn(300, 1), torch.randn(300, 1)
        r = nmi.mi_rate(x, y, h=0, W=5, model=Model(embedding_dim=8, hidden_dim=16, n_layers=1),
                        training=_TRAINING, show_progress=False)
        assert isinstance(r, nmi.Results)

    def test_instantaneous_exchange_k_zero_needs_no_dual_branch_model(self):
        x, y = torch.randn(300, 1), torch.randn(300, 1)
        r = nmi.instantaneous_exchange(x, y, k=0, model=Model(embedding_dim=8, hidden_dim=16, n_layers=1),
                                       training=_TRAINING, show_progress=False)
        assert isinstance(r, nmi.Results)


class TestQuantitiesShapesAndSweep:
    def test_mi_rate_scalar_returns_results(self):
        x, y = torch.randn(400, 1), torch.randn(400, 1)
        r = nmi.mi_rate(x, y, h=3, W=5, model=_DB_MODEL, training=_TRAINING, show_progress=False)
        assert isinstance(r, nmi.Results)
        assert np.isfinite(r.mi_estimate)

    def test_instantaneous_exchange_scalar_returns_results(self):
        x, y = torch.randn(400, 1), torch.randn(400, 1)
        r = nmi.instantaneous_exchange(x, y, k=3, model=_DB_MODEL, training=_TRAINING, show_progress=False)
        assert isinstance(r, nmi.Results)
        assert np.isfinite(r.mi_estimate)

    def test_directed_information_rate_scalar_returns_results(self):
        x, y = torch.randn(400, 1), torch.randn(400, 1)
        r = nmi.directed_information_rate(x, y, k=3, model=_DB_MODEL, training=_TRAINING, show_progress=False)
        assert isinstance(r, nmi.Results)
        assert np.isfinite(r.mi_estimate)

    def test_mi_rate_sweep_returns_dataframe(self):
        x, y = torch.randn(500, 1), torch.randn(500, 1)
        df = nmi.mi_rate(x, y, h=[0, 2, 4], W=5, model=_DB_MODEL, training=_TRAINING,
                          n_workers=2, show_progress=False)
        assert isinstance(df, pd.DataFrame)
        assert list(df['h']) == [0, 2, 4]
        assert df['mi_estimate'].notna().all()

    def test_instantaneous_exchange_sweep_returns_dataframe(self):
        x, y = torch.randn(500, 1), torch.randn(500, 1)
        df = nmi.instantaneous_exchange(x, y, k=[0, 2, 4], model=_DB_MODEL, training=_TRAINING,
                                        n_workers=2, show_progress=False)
        assert isinstance(df, pd.DataFrame)
        assert list(df['k']) == [0, 2, 4]

    def test_directed_information_rate_sweep_returns_dataframe(self):
        x, y = torch.randn(500, 1), torch.randn(500, 1)
        df = nmi.directed_information_rate(x, y, k=[1, 2, 3], model=_DB_MODEL, training=_TRAINING,
                                           n_workers=2, show_progress=False)
        assert isinstance(df, pd.DataFrame)
        assert list(df['k']) == [1, 2, 3]


# --------------------------------------------------------------------------
# Accuracy against exact Gaussian ground truth (slower, generous tolerance --
# short training budget, so this checks "in the right ballpark", not
# precision; the harness scripts already recorded high-fidelity numbers with
# a much larger training budget outside this test suite).
# --------------------------------------------------------------------------
class TestAccuracyAgainstOracle:
    _oracle = _SharedLatentOracle(phi=0.85, a=1.0, b=1.0, sx=0.5, sy=0.5)

    def test_mi_rate_accuracy(self):
        h, W = 3, 5
        x, y = self._oracle.sample(6000, seed=10)
        exact = self._oracle.mi_rate_exact(h, W)
        r = nmi.mi_rate(torch.from_numpy(x), torch.from_numpy(y), h=h, W=W,
                        model=_DB_MODEL, training=_TRAINING, show_progress=False, seed=0)
        assert abs(r.mi_estimate - exact) < 0.5

    def test_instantaneous_exchange_accuracy(self):
        k = 3
        x, y = self._oracle.sample(6000, seed=11)
        exact = self._oracle.inst_exchange_exact(k)
        r = nmi.instantaneous_exchange(torch.from_numpy(x), torch.from_numpy(y), k=k,
                                       model=_DB_MODEL, training=_TRAINING, show_progress=False, seed=0)
        assert abs(r.mi_estimate - exact) < 0.5

    def test_directed_information_rate_accuracy(self):
        k = 3
        x, y = self._oracle.sample(6000, seed=12)
        exact = self._oracle.dir_info_rate_exact(k)
        r = nmi.directed_information_rate(torch.from_numpy(x), torch.from_numpy(y), k=k,
                                          model=_DB_MODEL, training=_TRAINING, show_progress=False, seed=0)
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
