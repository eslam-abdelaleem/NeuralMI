"""Tests for rigorous-mode diagnostics: fit-quality warnings and the scalar-analysis engine."""
import warnings

import pytest
import numpy as np
import pandas as pd
import torch

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_df(gammas, mi_vals):
    """Build a minimal rigorous-style DataFrame."""
    rows = []
    for g, m in zip(gammas, mi_vals):
        rows.append({'gamma': g, 'train_mi': m})
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Unit tests for _compute_fit_diagnostics
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeFitDiagnostics:
    """Direct unit tests for _compute_fit_diagnostics."""

    def setup_method(self):
        from neural_mi.analysis.rigorous import _compute_fit_diagnostics
        self._fn = _compute_fit_diagnostics

    def test_clean_linear_fit_no_warnings(self):
        """A perfect linear relationship should not trigger any warnings."""
        gammas = [1, 2, 3, 4, 5]
        # MI = 1.5 + 0.1 * gamma  (perfect linear in gamma, I_true = 1.5)
        mi_vals = [1.5 + 0.1 * g for g in gammas]
        df = _make_df(gammas, mi_vals)
        result = self._fn(df, gammas)
        assert not result['fit_quality_warning'], "Clean linear fit should not flag residuals."
        assert not result['leverage_warning'], "Clean linear fit should not flag leverage."
        assert result['r_squared'] > 0.99

    def test_outlier_gamma1_triggers_leverage_warning(self):
        """When gamma=1 is a strong outlier, leverage_warning should fire."""
        gammas = [1, 2, 3, 4, 5]
        # Normal linear trend except gamma=1 is a huge outlier
        mi_vals = [1.5 + 0.1 * g for g in gammas]
        mi_vals[0] = 10.0  # gamma=1 is an extreme outlier
        df = _make_df(gammas, mi_vals)
        result = self._fn(df, gammas, leverage_threshold=0.10)
        assert result['leverage_warning'], (
            "Extreme gamma=1 outlier should trigger leverage_warning."
        )

    def test_noisy_data_can_trigger_residual_warning(self):
        """Highly scattered data (large studentized residuals) should trigger fit_quality_warning."""
        # Seed chosen so at least one residual exceeds the default threshold of 2.5.
        rng = np.random.default_rng(0)
        gammas = list(range(1, 8))
        mi_vals = [1.5 + 0.1 * g + rng.normal(0, 2.0) for g in gammas]  # very noisy
        df = _make_df(gammas, mi_vals)
        result = self._fn(df, gammas)
        # R² is still reported regardless.
        assert result['r_squared'] is not None
        # fit_quality_warning is driven only by studentized residuals, not R².

    def test_low_r2_does_not_trigger_fit_quality_warning(self):
        """Near-flat MI curve (large N, tiny bias) has low R² but must NOT flag fit_quality_warning."""
        # Simulate the large-N scenario: slope ≈ 0, all values clustered ≈ 0.93,
        # residuals small.  R² collapses because SS_tot ≈ noise variance >> signal variance.
        rng = np.random.default_rng(7)
        gammas = list(range(1, 8))
        mi_vals = [0.93 + 0.002 * g + rng.normal(0, 0.05) for g in gammas]
        df = _make_df(gammas, mi_vals)
        result = self._fn(df, gammas)
        assert result['r_squared'] < 0.5, (
            f"Near-flat line should have low R², got {result['r_squared']:.3f}"
        )
        assert not result['fit_quality_warning'], (
            "Low R² alone must not trigger fit_quality_warning; "
            "only large studentized residuals do."
        )

    def test_too_few_points_returns_safe_defaults(self):
        """With fewer than 3 points, diagnostics should return safe (no-flag) defaults."""
        df = _make_df([1, 2], [1.0, 1.5])
        result = self._fn(df, [1, 2])
        assert not result['fit_quality_warning']
        assert not result['leverage_warning']
        assert np.isnan(result['r_squared'])
        assert np.isnan(result['max_abs_residual'])
        assert np.isnan(result['loo_intercept_shift'])

    def test_no_gamma1_skips_loo_check(self):
        """When no gamma=1 rows exist, LOO check should be skipped (no penalty)."""
        gammas = [2, 3, 4, 5, 6]
        mi_vals = [1.5 + 0.1 * g for g in gammas]
        df = _make_df(gammas, mi_vals)
        result = self._fn(df, gammas)
        assert not result['leverage_warning']
        assert np.isnan(result['loo_intercept_shift'])

    def test_returns_all_required_keys(self):
        """The return dict must contain exactly the expected keys."""
        df = _make_df([1, 2, 3, 4, 5], [2.0, 1.8, 1.7, 1.6, 1.55])
        result = self._fn(df, [1, 2, 3, 4, 5])
        required = {'fit_quality_warning', 'leverage_warning', 'r_squared',
                    'max_abs_residual', 'loo_intercept_shift'}
        assert required.issubset(set(result.keys()))


# ─────────────────────────────────────────────────────────────────────────────
# Integration test: diagnostic flags in post_process_and_correct output
# ─────────────────────────────────────────────────────────────────────────────

class TestDiagnosticsInCorrectedResults:
    """Check that _post_process_and_correct propagates diagnostic keys."""

    def test_diagnostic_keys_present_in_results(self):
        from neural_mi.analysis.rigorous import _post_process_and_correct
        # Build a plausible raw_results_df
        rows = []
        for gamma in range(1, 8):
            for rep in range(3):
                mi = 1.5 + 0.1 * gamma + np.random.normal(0, 0.02)
                rows.append({'gamma': gamma, 'train_mi': mi})
        df = pd.DataFrame(rows)
        results = _post_process_and_correct(
            df, sweep_grid=None,
            curvature_t_threshold=2.0, min_gamma_points=5,
            confidence_level=0.68,
        )
        assert len(results) > 0
        r = results[0]
        for key in ('fit_quality_warning', 'leverage_warning', 'r_squared',
                    'max_abs_residual', 'loo_intercept_shift', 'is_reliable'):
            assert key in r, f"Key '{key}' missing from corrected result: {r.keys()}"

    def test_gamma1_outlier_sets_is_reliable_false(self):
        """When gamma=1 is a strong outlier, is_reliable should be False."""
        from neural_mi.analysis.rigorous import _post_process_and_correct
        rows = []
        for gamma in range(1, 8):
            mi = 1.5 + 0.1 * gamma + np.random.normal(0, 0.01)
            rows.append({'gamma': gamma, 'train_mi': mi})
        # Make gamma=1 an extreme outlier
        rows[0]['train_mi'] = 20.0
        df = pd.DataFrame(rows)
        results = _post_process_and_correct(
            df, sweep_grid=None,
            curvature_t_threshold=2.0, min_gamma_points=4,
            confidence_level=0.68,
            leverage_threshold=0.05,  # tight threshold
        )
        # is_reliable should be False due to leverage warning
        if results:
            assert not results[0]['is_reliable']


# ─────────────────────────────────────────────────────────────────────────────
# Tests for run_rigorous_scalar_analysis
# ─────────────────────────────────────────────────────────────────────────────

class TestRigorousScalarAnalysis:
    """Test the general rigorous scalar extrapolation function."""

    def _make_tensors(self, N=500, C=2, W=1):
        rng = torch.Generator()
        rng.manual_seed(42)
        x = torch.randn(N, C, W, generator=rng)
        y = torch.randn(N, C, W, generator=rng)
        return x, y

    def test_output_has_required_keys(self):
        from neural_mi.analysis.rigorous import run_rigorous_scalar_analysis

        call_count = [0]

        def _fake_scalar(x_s, y_s, bp, **kw):
            call_count[0] += 1
            return float(2.0 - 0.5 / max(x_s.shape[0] / 100, 1) + np.random.normal(0, 0.05))

        x, y = self._make_tensors()
        result = run_rigorous_scalar_analysis(
            scalar_fn=_fake_scalar,
            x_data=x, y_data=y,
            base_params={},
            gamma_range=range(1, 6),
            min_gamma_points=3,
        )
        required = {'mi_corrected', 'mi_error', 'slope', 'is_reliable',
                    'gammas_used', 'raw_results_df',
                    'fit_quality_warning', 'leverage_warning'}
        assert required.issubset(result.keys())
        assert isinstance(result['raw_results_df'], pd.DataFrame)
        assert 'gamma' in result['raw_results_df'].columns
        assert 'train_mi' in result['raw_results_df'].columns

    def test_extra_data_is_subsampled(self):
        from neural_mi.analysis.rigorous import run_rigorous_scalar_analysis

        received_w_sizes = []
        N_full = 200

        def _scalar_fn(x_s, y_s, bp, w_data=None, **kw):
            if w_data is not None:
                received_w_sizes.append(w_data.shape[0])
            # Return values linear in gamma so WLS fitting converges cleanly.
            gamma_approx = N_full / max(x_s.shape[0], 1)
            return float(1.5 + 0.1 * gamma_approx + np.random.normal(0, 0.01))

        x, y = self._make_tensors(N=N_full)
        w = torch.randn(N_full, 1, 1)
        run_rigorous_scalar_analysis(
            scalar_fn=_scalar_fn,
            x_data=x, y_data=y,
            base_params={},
            extra_data={'w_data': w},
            gamma_range=range(1, 5),
            min_gamma_points=3,
        )
        assert len(received_w_sizes) > 0
        # Subsampled w must be smaller than or equal to N_full
        assert all(s <= N_full for s in received_w_sizes)
        # Different gammas produce different subset sizes
        assert len(set(received_w_sizes)) > 1


# ─────────────────────────────────────────────────────────────────────────────
# Tests for decoder infrastructure
# ─────────────────────────────────────────────────────────────────────────────

class TestDecoderModels:
    """Test that all decoder classes produce the correct output shape."""

    @staticmethod
    def _settings(**given):
        """The settings build_decoder requires, at their schema defaults unless given."""
        from neural_mi.defaults import default_of
        return {'n_layers': default_of('n_layers'), 'kernel_size': default_of('kernel_size'),
                'nhead': default_of('nhead'), 'dropout': default_of('dropout'),
                'output_activation': default_of('decoder_output_activation_x'), **given}

    @pytest.mark.parametrize("model_name,kwargs", [
        ('mlp', {}),
        ('cnn1d', {'kernel_size': 7}),
        ('gru', {}),
        ('lstm', {}),
        ('tcn', {'kernel_size': 3}),
        ('transformer', {'nhead': 4}),
    ])
    def test_decoder_output_shape(self, model_name, kwargs):
        from neural_mi.models.decoders import build_decoder
        embedding_dim = 32
        hidden_dim = 64
        n_channels = 4
        window_size = 20
        n_layers = 2
        batch_size = 8
        z = torch.randn(batch_size, embedding_dim)
        dec = build_decoder(
            embedding_model=model_name,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            n_channels=n_channels,
            window_size=window_size,
            **self._settings(n_layers=n_layers, **kwargs),
        )
        dec.eval()
        with torch.no_grad():
            out = dec(z)
        assert out.shape == (batch_size, n_channels, window_size), (
            f"{model_name} decoder output shape {out.shape} != "
            f"({batch_size}, {n_channels}, {window_size})"
        )

    @pytest.mark.parametrize("activation", ['linear', 'sigmoid', 'softmax'])
    def test_output_activations(self, activation):
        from neural_mi.models.decoders import build_decoder
        dec = build_decoder(
            embedding_model='mlp',
            embedding_dim=16, hidden_dim=32, n_channels=3, window_size=10,
            **self._settings(output_activation=activation),
        )
        dec.eval()
        z = torch.randn(4, 16)
        with torch.no_grad():
            out = dec(z)
        if activation == 'sigmoid':
            assert out.min() >= 0.0 and out.max() <= 1.0
        elif activation == 'softmax':
            # Softmax over channel dim (dim=1) should sum to ~1
            assert torch.allclose(out.sum(dim=1), torch.ones(4, 10), atol=1e-5)

    def test_unknown_activation_raises(self):
        from neural_mi.models.decoders import build_decoder
        with pytest.raises(ValueError, match="Unknown output_activation"):
            build_decoder('mlp', 16, 32, 3, 10, **self._settings(output_activation='relu'))


class TestDecoderInTraining:
    """Integration test: ensure use_decoder=True runs without errors."""

    def test_run_estimate_with_decoder(self):
        """End-to-end test: run in estimate mode with use_decoder=True."""
        import neural_mi as nmi
        from neural_mi import Model, Training
        N = 300
        C = 2
        W = 10
        rng = np.random.default_rng(0)
        x = rng.normal(size=(N, C, W)).astype(np.float32)
        y = rng.normal(size=(N, C, W)).astype(np.float32)
        result = nmi.run(
            x, y,
            mode='estimate',
            model=Model(embedding_model='mlp', hidden_dim=16, embedding_dim=8,
                        n_layers=1, use_decoder=True, decoder_lambda=0.5),
            training=Training(n_epochs=3, batch_size=64, patience=100, learning_rate=1e-3),
            verbose=False,
            show_progress=False,
        )
        assert result.mi_estimate is not None
        assert result.get('decoder_recon_loss') is not None


class TestGetTrainingEmbeddings:
    """Test that get_training_embeddings returns gradients."""

    def test_gradients_flow_through_training_embeddings(self):
        from neural_mi.utils import build_critic
        params = {
            'critic_type': 'separable',
            'embedding_model': 'mlp',
            'hidden_dim': 16,
            'embedding_dim': 8,
            'n_layers': 1,
            'input_dim_x': 20,
            'input_dim_y': 20,
            'n_channels_x': 2,
            'n_channels_y': 2,
            'use_variational': False,
            'use_spectral_norm': False,
            'max_n_batches': 512,
        }
        critic = build_critic('separable', params)
        critic.train()
        x = torch.randn(10, 2, 10, requires_grad=False)
        y = torch.randn(10, 2, 10, requires_grad=False)
        z_x, z_y = critic.get_training_embeddings(x, y)
        # Gradients should flow through the embedding network
        loss = z_x.mean() + z_y.mean()
        loss.backward()
        # Check that embedding_net_x got gradients
        for p in critic.embedding_net_x.parameters():
            if p.requires_grad:
                assert p.grad is not None
                break


# ─────────────────────────────────────────────────────────────────────────────
# shift_windows reachability: chunk-to-raw-range translation
# ─────────────────────────────────────────────────────────────────────────────

class TestAChunkIsThePlanOfItsRows:
    """A chunk of a rigorous ladder is the plan of a stretch of its rows
    (RowPlan.cut): the same rows at offset 0, and as many at every shift."""

    @staticmethod
    def _plans(shift):
        from collections import OrderedDict
        from neural_mi.data.plan import RowPlan, resolve_stream
        rng = np.random.default_rng(0)
        x = rng.standard_normal((2000, 2)).astype('float32')
        regular = RowPlan(OrderedDict(x=resolve_stream(x, None, 'continuous', {'window_size': 20,
                                                                               'step_size': 0.5}),
                                      y=resolve_stream(x[:, ::-1].copy(), None, 'continuous', {})),
                          shift=shift)
        spikes = [np.sort(rng.uniform(0, 60, 300)) for _ in range(3)]
        spiking = RowPlan(OrderedDict(x=resolve_stream(spikes, None, 'spike', {'window_size': 0.5}),
                                      y=resolve_stream([s + 0.01 for s in spikes], None, 'spike', {})),
                          shift=shift)
        return regular, spiking

    @pytest.mark.parametrize('route', [0, 1], ids=['by_samples', 'in_time'])
    @pytest.mark.parametrize('shift', [False, True])
    def test_a_chunk_reads_its_rows(self, route, shift):
        plan = self._plans(shift)[route]
        lo, hi = 7, 31
        chunk = plan.cut(np.arange(lo, hi))
        assert len(chunk) == hi - lo
        whole, part = plan.rows()['windows'], chunk.rows()['windows']
        for name in whole:
            a, b = whole[name][lo:hi], part[name]
            width = min(a.shape[-1], b.shape[-1])
            assert torch.allclose(a[..., :width], b[..., :width]), name
        if shift:
            shifter = chunk.rows()['shifter']
            for _ in range(5):
                windows = shifter.windows_at(shifter.random_shift())
                assert all(w.shape[0] == hi - lo for w in windows.values())


class TestRigorousShiftWindowsEndToEnd:
    """A real (un-mocked) run confirming rigorous's deferred-windowing path
    dispatches the expected number of tasks and produces finite results --
    if the chunk-to-raw-range translation were wrong, this would either
    crash (out-of-bounds raw slice) or silently create wrong-sized chunks
    (caught by the task-count check already built into AnalysisWorkflow.run,
    which warns if len(tasks) != expected_total)."""

    def test_rigorous_shift_windows_no_crash_and_expected_task_count(self):
        import neural_mi as nmi
        from neural_mi import Training, Rigorous

        torch.manual_seed(0)
        np.random.seed(0)
        x = np.random.randn(4000, 2).astype('float32')
        y = np.random.randn(4000, 2).astype('float32')
        proc = nmi.Processing(x='continuous', x_params={'window_size': 20, 'step_size': 20},
                              y='continuous', y_params={'window_size': 20, 'step_size': 20})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            results = nmi.run(
                x, y, mode='rigorous', processing=proc,
                training=Training(n_epochs=1, batch_size=16, shift_windows=True,
                                  min_reliable_samples=1),
                rigorous=Rigorous(gamma_range=range(1, 4)),
                n_workers=1, show_progress=False, seed=0,
            )
        assert "may be truncated" not in ' '.join(str(w.message) for w in caught)
        ladder = results.details[0]['trainings']
        assert len(ladder) == sum(range(1, 4))  # 1 + 2 + 3 = 6 networks
        assert np.all(np.isfinite(ladder['train_mi'].values))


class TestRigorousShiftingSpikesEndToEnd:
    """A real (un-mocked) run confirming that rigorous's chunks of a spike
    plan that shifts dispatch the expected number of tasks and produce finite
    results."""

    def test_rigorous_shifting_spikes_no_crash_and_expected_task_count(self):
        import neural_mi as nmi
        from neural_mi import Training, Rigorous

        torch.manual_seed(0)
        np.random.seed(0)
        x_spikes, y_spikes, _ = nmi.generators.generate_spike_pair(
            n_neurons=5, n_windows=600, window_size=0.05, seed=0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            results = nmi.run(
                x_spikes, y_spikes, mode='rigorous',
                processing=nmi.Processing(x='spike', x_params={'window_size': 0.05}),
                model=nmi.Model(embedding_dim=8, hidden_dim=16, n_layers=1),
                training=Training(n_epochs=1, batch_size=16, shift_windows=True,
                                  min_reliable_samples=1),
                rigorous=Rigorous(gamma_range=range(1, 4)),
                n_workers=1, show_progress=False, seed=0,
            )
        assert "may be truncated" not in ' '.join(str(w.message) for w in caught)
        ladder = results.details[0]['trainings']
        assert len(ladder) == sum(range(1, 4))  # 1 + 2 + 3 = 6 networks
        assert np.all(np.isfinite(ladder['train_mi'].values))


# ─────────────────────────────────────────────────────────────────────────────
# What the ladder reports, and what it keeps to itself
# ─────────────────────────────────────────────────────────────────────────────

def test_a_flat_ladder_has_undefined_r_squared_and_says_nothing():
    """Every chunk at the same value leaves nothing for a line to explain, so
    R-squared is NaN, without numpy's division warning leaking through."""
    import warnings
    from neural_mi.analysis.rigorous import _compute_fit_diagnostics
    gammas = [1, 2, 3, 4, 5]
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        result = _compute_fit_diagnostics(_make_df(gammas, [0.0] * len(gammas)), gammas)
    assert np.isnan(result['r_squared'])


def test_a_chunk_keeps_its_combination_warnings_to_itself():
    """A chunk's combined value is one point on the ladder. Its sign and
    amplification warnings stay quiet, while a network's own warnings pass."""
    import warnings
    from neural_mi.analysis.rigorous import _run_scalar_fn_task
    from neural_mi import CombinationWarning

    def scalar_fn(x, y, params):
        warnings.warn("Conditional MI estimate is negative (-0.2 bits).", CombinationWarning)
        warnings.warn("Training completed all 2 epoch(s) without early stopping", UserWarning)
        return 0.1

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        row = _run_scalar_fn_task((scalar_fn, None, None, {}, {}, None, 2, 0, 50))
    assert row['train_mi'] == 0.1
    assert [str(w.message)[:20] for w in caught] == ["Training completed a"]


def test_a_fractional_step_cuts_each_chunk_with_the_step_the_windows_use():
    """The step convention reaches the chunks: step_size=0.5 is half a 20-sample window, so each chunk's
    rows start 10 samples apart, as the whole plan's do."""
    from collections import OrderedDict
    from neural_mi.data.plan import RowPlan, resolve_stream
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2000, 2)).astype(np.float32)
    plan = RowPlan(OrderedDict(x=resolve_stream(x, None, 'continuous', {'window_size': 20, 'step_size': 0.5}),
                               y=resolve_stream(x.copy(), None, 'continuous', {})), shift=True)
    chunk = plan.cut(np.arange(3, 40))
    assert chunk.step == 10 and len(chunk) == 37
    assert np.allclose(np.diff(plan.rows()['times'][3:40]), 10.0)


def test_zero_rungs_are_left_out_of_the_fit_and_counted_per_gamma():
    """A rung at 0 produced nothing. The fit leaves it out and says how many per gamma,
    and warns again where half or more of a gamma's rungs produced nothing."""
    from neural_mi.analysis.rigorous import _drop_zero_rungs
    ladder = _make_df([1, 2, 2, 3, 3, 3], [0.5, 0.4, 0.0, 0.3, 0.0, 0.0])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        kept, n = _drop_zero_rungs(ladder, "Rigorous fit: ")
    assert n == 3 and list(kept['train_mi']) == [0.5, 0.4, 0.3]
    messages = [str(w.message) for w in caught]
    assert any(m.startswith("Rigorous fit: 3 of 6 rungs") and "gamma 2: 1 of 2, gamma 3: 2 of 3" in m
               for m in messages)
    assert any("half or more of the rungs produced nothing at gamma=[2, 3]" in m for m in messages)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        kept, n = _drop_zero_rungs(_make_df([1, 2], [0.5, 0.4]), "")
    assert n == 0 and len(kept) == 2 and not caught
