"""Which rungs of a rigorous ladder the fit should use, behind C21.

Runs mode='rigorous' on the data of tutorial 02 (10 latent dimensions, 500
channels, 4 bits) at several sample sizes, and on independent data, then refits
each ladder offline under four rules:

- before C21: leave out the rungs that produced nothing (0 or NaN) only.
- drop unresolved: also leave out every rung whose held-out curve did not
  resolve (curve_diagnosis 'not resolved').
- drop unresolved, then half-lost gammas: also drop a gamma that lost half or
  more of its rungs, so that no gamma enters the fit through its higher readings.
- gate only: fit as before C21, and call the fit unreliable when a gamma it
  uses lost half or more of its rungs to either cause.

The library's rule after C21 is the third. The first line of each case gives
what the library itself reported.

Usage: python scratch/investigation/c21_rung_rules.py <repo> [out.txt]
"""
import sys
import time


def per_gamma(ladder):
    zero = (ladder['train_mi'] == 0) | ladder['train_mi'].isna()
    unres = ~zero & (ladder['curve_diagnosis'] == 'not resolved')
    g = ladder.assign(_zero=zero, _unres=unres).groupby('gamma')
    return g.agg(n=('train_mi', 'size'), zero=('_zero', 'sum'), unresolved=('_unres', 'sum'),
                 mean=('train_mi', 'mean'))


def refit(ladder, rule):
    import warnings
    from neural_mi.analysis.rigorous import _post_process_and_correct
    zero = (ladder['train_mi'] == 0) | ladder['train_mi'].isna()
    unres = ~zero & (ladder['curve_diagnosis'] == 'not resolved')
    df = ladder
    if rule in ('drop unresolved', 'drop unresolved, then half-lost gammas'):
        df = ladder[~unres]
    if rule == 'drop unresolved, then half-lost gammas':
        lost = (zero | unres).groupby(ladder['gamma']).mean()
        df = df[~df['gamma'].isin(lost[lost >= 0.5].index)]
    # Without the diagnoses the library's fit leaves out only the rungs that
    # produced nothing, so each rule decides alone which rungs it keeps.
    df = df.drop(columns=['curve_diagnosis'])
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            fits = _post_process_and_correct(df.copy(), sweep_grid={}, curvature_t_threshold=2.0,
                                             min_gamma_points=5, confidence_level=0.68)
    except Exception:
        return None
    if not fits:
        return None
    fit = fits[0]
    if rule == 'gate only':
        lost = (zero | unres).groupby(ladder['gamma']).mean()
        if any(lost.get(g, 0) >= 0.5 for g in fit['gammas_used']):
            fit = {**fit, 'is_reliable': False}
    return fit


def main(repo, out_path=None):
    sys.path.insert(0, repo)
    import numpy as np
    import neural_mi as nmi
    from neural_mi.generators import generate_nonlinear_from_latent

    lines = []

    def say(text=''):
        print(text, flush=True)
        lines.append(text)

    cases = [(f'4 bits, N={n}, seed {s}', n, 4.0, s)
             for n, s in ((400, 0), (600, 0), (600, 1), (800, 0), (800, 1), (1200, 0), (2000, 0))]
    cases.append(('independent, N=2000', 2000, 0.0, 0))
    rules = ('before C21', 'drop unresolved', 'drop unresolved, then half-lost gammas', 'gate only')
    for name, n, mi, seed in cases:
        t0 = time.time()
        x, y = generate_nonlinear_from_latent(n_samples=n, latent_dim=10, observed_dim=500,
                                              mi=max(mi, 4.0), seed=seed)
        if mi == 0.0:
            # Pair X with the Y of another draw: same marginals, no dependence.
            _, y = generate_nonlinear_from_latent(n_samples=n, latent_dim=10, observed_dim=500,
                                                  mi=4.0, seed=seed + 1)
        r = nmi.run(x, y, mode='rigorous', split=nmi.Split(mode='random'),
                    estimator=nmi.Estimator('infonce'), n_workers=8, seed=seed, show_progress=False)
        ladder = r.details[0]['trainings']
        value = 'NaN' if r.mi_estimate is None else f'{r.mi_estimate:.3f}'
        say(f'{name}: exact {mi} bits, library {value} '
            f'is_reliable {r.get("is_reliable")}  ({time.time() - t0:.0f}s)')
        pg = per_gamma(ladder)
        say('  gamma  n  zero  unresolved  mean of all rungs')
        for g, row in pg.iterrows():
            say(f'  {int(g):>5} {int(row["n"]):>2} {int(row["zero"]):>5} {int(row["unresolved"]):>11}'
                f'  {row["mean"]:>8.3f}')
        for rule in rules:
            fit = refit(ladder, rule)
            if fit is None:
                say(f'  {rule:<40} no fit')
                continue
            say(f'  {rule:<40} {fit["mi_corrected"]:>7.3f} +/- {fit["mi_error"]:.3f}  '
                f'gammas {[int(g) for g in fit["gammas_used"]]}  '
                f'r2 {fit.get("r_squared", float("nan")):.3f}  reliable {fit["is_reliable"]}')
        say()
    if out_path:
        with open(out_path, 'w') as fh:
            fh.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
