# tests/results_factory.py
"""Results objects in the shape run() returns, for tests of Results and its plots.

Every builder goes through :func:`neural_mi.analysis.assemble.build_results`,
the function every mode uses, so a test built here exercises the real contract
instead of a hand-written imitation of it.
"""
import numpy as np
import pandas as pd

from neural_mi.analysis.assemble import build_results


def make_results(mode, rows, *, config_keys=(), axis_keys=(), details=None, units='bits',
                 mean_cols=(), first_cols=(), std_cols=(), row_scalars=None, **params):
    return build_results(mode, {'output_units': units, **params}, rows,
                         config_keys=config_keys, axis_keys=axis_keys, mean_cols=mean_cols,
                         first_cols=first_cols, std_cols=std_cols, details=details or {},
                         row_scalars=row_scalars)


def estimate(mi=0.45, history=(0.1, 0.3, 0.5, 0.4, 0.45), best_epoch=2, units='bits',
             embeddings=None, **extra):
    """One configuration, one repeat, one network."""
    row = {'config_id': 0, 'run_id': 0, 'mi': mi, 'train_mi': mi,
           'test_mi': max(history) if history else mi, **extra}
    if history is not None:
        row['test_mi_history'] = list(history)
    if best_epoch is not None:
        row['best_epoch'] = best_epoch
    details = {0: {'embeddings': {0: embeddings}}} if embeddings else None
    return make_results('estimate', [row], mean_cols=('test_mi',), units=units, details=details)


def sweep(values=(4, 8, 16, 32), means=(0.5, 0.9, 1.2, 1.25), key='embedding_dim',
          n_runs=2, units='bits', spread=0.05):
    """One configuration per value of `key`, `n_runs` repeats each."""
    rows = []
    for cid, (value, mean) in enumerate(zip(values, means)):
        for rid in range(n_runs):
            offset = spread * (rid - (n_runs - 1) / 2)
            rows.append({'config_id': cid, key: value, 'run_id': rid, 'mi': mean + offset,
                         'train_mi': mean + offset, 'test_mi': mean,
                         'test_mi_history': [mean / 2, mean]})
    return make_results('sweep', rows, config_keys=(key,), mean_cols=('test_mi',), units=units,
                        sweep_grid={key: list(values), 'run_id': list(range(n_runs))})


def rigorous(fits=None, gammas=range(1, 6), units='bits', seed=0):
    """One configuration, one repeat per fit, each with its own gamma ladder."""
    fits = fits or [{'mi': 0.55, 'mi_error': 0.05, 'slope': -0.5, 'is_reliable': True,
                     'gammas_used': list(gammas)}]
    rng = np.random.default_rng(seed)
    rows, ladder = [], []
    for rid, fit in enumerate(fits):
        fit = {'gammas_used': list(gammas), 'slope': 0.0, **fit}
        rows.append({'config_id': 0, 'run_id': rid, **fit})
        for gamma in gammas:
            for chunk in range(gamma):
                ladder.append({'run_id': rid, 'gamma': gamma, 'chunk': chunk,
                               'train_mi': fit['mi'] + fit.get('slope', 0) * gamma
                               + 0.01 * rng.standard_normal()})
    scalars = {0: {'mi_error': fits[0].get('mi_error') if len(fits) == 1 else float('nan'),
                   'n_reliable': sum(bool(f.get('is_reliable')) for f in fits)}}
    return make_results('rigorous', rows, details={0: {'trainings': pd.DataFrame(ladder)}},
                        row_scalars=scalars, units=units)


def difference(mode, components, mi, *, n_runs=1, units='bits', scalars=None, **params):
    """A conditional, interaction or transfer result: component values per repeat."""
    rows = [{'config_id': 0, 'run_id': rid, 'mi': mi, **components} for rid in range(n_runs)]
    return make_results(mode, rows, mean_cols=tuple(components), units=units,
                        row_scalars={0: scalars} if scalars else None, **params)


def precision(taus=(0.0, 0.5, 1.0), mis=(1.2, 1.1, 0.9), units='bits', **info):
    rows = [{'config_id': 0, 'tau': tau, 'mi': mi} for tau, mi in zip(taus, mis)]
    details = {0: {'baseline_mi': mis[0], **info}}
    return make_results('precision', rows, axis_keys=('tau',), details=details, units=units)


def pairwise(matrix, units='bits', names=None):
    n = matrix.shape[0]
    rows = [{'config_id': 0, 'ch_x': i, 'ch_y': j, 'run_id': 0, 'mi': float(matrix[i, j])}
            for i in range(n) for j in range(i + 1, n)]
    entry = {'mi_matrix': matrix, 'n_channels': n}
    if names is not None:
        entry['variable_names_x'] = list(names)
        entry['variable_names_y'] = list(names)
    return make_results('pairwise', rows, axis_keys=('ch_x', 'ch_y'), details={0: entry},
                        units=units)
