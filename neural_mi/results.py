# neural_mi/results.py
"""The `Results` object every analysis mode returns.

Every mode fills the same fields: ``runs`` holds one row per repeat,
``dataframe`` one row per configuration (and per value of the mode's own axis),
``mi_estimate`` the headline when there is exactly one such row, ``params`` the
full configuration the call ran with, and ``details`` the structured
diagnostics of each configuration.
"""
import os
import math
import datetime
import pickle
import json
from dataclasses import dataclass, field
from typing import Optional, Any, Dict, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from neural_mi.logger import logger

# How summary() names the value of a quantity that is not a plain MI.
_QUANTITY_LABELS = {'conditional': 'I(X;Y|W)', 'interaction': 'II', 'transfer': 'TE(X→Y)'}
# The axes whose values are separate networks. Precision's tau is one network
# evaluated many times.
_NETWORK_AXES = {'lag': ('lag',), 'pairwise': ('ch_x', 'ch_y'),
                 'dimensionality': ('split_id', 'embedding_dim')}

# Per-component columns of the difference quantities, in the order they are
# drawn, with their display labels.
_COMPONENTS = {
    'conditional': [('mi_xw_y', 'I(X,W;Y)'), ('mi_w_y', 'I(W;Y)')],
    'interaction': [('mi_xw_y', 'I(X,W;Y)'), ('mi_x_y', 'I(X;Y)'), ('mi_w_y', 'I(W;Y)')],
}


def _scalar(value: Any) -> Any:
    """A numpy scalar as its Python value; anything else unchanged."""
    if isinstance(value, np.generic):
        return value.item()
    return value


def _finite(value) -> bool:
    return value is not None and not (isinstance(value, float) and math.isnan(value))


def _fit_lines(fit: Dict[str, Any], units: str) -> List[str]:
    """The summary lines of one extrapolation fit: its intervals and reliability."""
    lines = []
    if _finite(fit.get('mi_error')):
        lines.append(f"  CI half-width : {fit['mi_error']:.4f} {units}  [confidence interval on the fitted mean]")
    if _finite(fit.get('mi_error_pred')):
        lines.append(f"  PI half-width : {fit['mi_error_pred']:.4f} {units}  [prediction interval, more conservative]")
    r_squared = fit.get('r_squared')
    if fit.get('is_reliable') is False:
        lines.append("  ⚠  is_reliable = False: the extrapolation is unreliable.")
        reasons = _unreliable_reasons(fit)
        if reasons:
            lines.append(f"     Reason(s): {'; '.join(reasons)}")
    elif fit.get('is_reliable') is True:
        lines.append("  ✓  is_reliable = True" + (f", R² = {r_squared:.3f}" if _finite(r_squared) else ""))
    return lines


def _unreliable_reasons(fit: Dict[str, Any]) -> List[str]:
    """The checks that set ``is_reliable=False`` on this fit.

    These are the four that decide it. ``fit_quality_warning`` and R² are
    reported beside the fit and decide nothing, so they are not reasons.
    """
    reasons = []
    if fit.get('enough_gamma_points') is False:
        reasons.append("too few gamma points (enough_gamma_points=False)")
    if fit.get('linear_region_found') is False:
        reasons.append("no linear region found (linear_region_found=False)")
    if fit.get('leverage_warning'):
        shift = fit.get('loo_intercept_shift')
        reasons.append("gamma=1 leverage (leverage_warning=True, LOO shift"
                       + (f"={shift:.3f}" if _finite(shift) else "") + ")")
    saturated = fit.get('saturated_gammas')
    if saturated is not None and len(saturated):
        reasons.append(f"ceiling-saturated gammas in the fit ({list(saturated)})")
    return reasons


def _rigorous_unreliable_reason(fit: Dict[str, Any]) -> str:
    """Compact ' (reason)' suffix for an is_reliable=False annotation."""
    reasons = [r.split(' (')[0] for r in _unreliable_reasons(fit)]
    return f" ({'; '.join(reasons)})" if reasons else ""


@dataclass
class Results:
    """The outcome of one call to :func:`neural_mi.run` or a named quantity.

    Attributes
    ----------
    mode : str
        The analysis mode that produced the result.
    params : dict
        The full configuration the call ran with, defaults included. Swept
        keys hold their grid; ``config_keys`` and ``axis_keys`` name the
        columns that index ``dataframe``.
    mi_estimate : float or None
        ``dataframe['mi_mean']`` when ``dataframe`` has exactly one row,
        otherwise ``None``. In the units set by ``Output(units=...)``.
    dataframe : pandas.DataFrame
        One row per configuration and axis value: ``config_id``, the grid keys,
        the axis keys, ``mi_mean``, ``mi_std``, ``n_runs`` and the mode's
        per-configuration columns. ``mi_std`` is the spread of repeats of the
        procedure on the same data and is NaN when there was one repeat. It is
        never an interval on the population value.
    runs : pandas.DataFrame
        One row per repeat: ``config_id``, the grid and axis keys, the repeat
        index (``run_id``), ``mi`` (the repeat's value of the
        quantity) and its diagnostics.
    details : dict
        ``{config_id: {...}}``: structured diagnostics per configuration.

    Read a single value with :meth:`get`. It finds the value in whichever of
    these holds it, as long as the answer is unambiguous.
    """
    mode: str
    params: Dict[str, Any] = field(default_factory=dict)
    mi_estimate: Optional[float] = None
    dataframe: Optional[pd.DataFrame] = None
    runs: Optional[pd.DataFrame] = None
    details: Dict[int, Dict[str, Any]] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Reading values
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        rep = f"Results(mode='{self.mode}'"
        if self.mi_estimate is not None:
            rep += f", mi_estimate={self.mi_estimate:.4f}"
        if self.dataframe is not None:
            rep += f", dataframe_rows={len(self.dataframe)}"
        if self.runs is not None:
            rep += f", runs={len(self.runs)}"
        return rep + ")"

    def _is_rigorous(self) -> bool:
        return self.mode == 'rigorous' or (
            self.mode in ('conditional', 'interaction', 'transfer')
            and bool((self.params or {}).get('rigorous')))

    @property
    def config_ids(self) -> List[int]:
        """The configurations this result holds, in grid order."""
        if self.dataframe is not None and 'config_id' in self.dataframe.columns:
            return [int(c) for c in pd.unique(self.dataframe['config_id'])]
        return sorted(int(k) for k in self.details)

    def get(self, key: str, default: Any = None) -> Any:
        """Read one value without knowing which table holds it.

        Looks in ``dataframe``, then ``details``, then the embeddings kept per
        repeat, then ``runs``, and returns the value when exactly one row,
        configuration or repeat holds it. Where several do, it raises and names
        the table to read, since picking one of them would be a silent choice.

        Parameters
        ----------
        key : str
            The column or diagnostic to read.
        default : Any, optional
            Returned when no table holds `key`.
        """
        df = self.dataframe
        if df is not None and key in df.columns:
            if len(df) == 1:
                return _scalar(df[key].iloc[0])
            raise ValueError(
                f"'{key}' has {len(df)} values, one per row of result.dataframe. "
                f"Read result.dataframe['{key}'] instead."
            )

        holders = [cid for cid, entry in self.details.items() if key in entry]
        if holders:
            if len(holders) == 1:
                return self.details[holders[0]][key]
            raise ValueError(
                f"'{key}' is recorded for {len(holders)} configurations. "
                f"Read result.details[config_id]['{key}'] instead."
            )

        found = [(cid, rid, emb[key])
                 for cid, entry in self.details.items()
                 for rid, emb in (entry.get('embeddings') or {}).items() if key in emb]
        if found:
            if len(found) == 1:
                return found[0][2]
            raise ValueError(
                f"'{key}' is recorded for {len(found)} repeats. Read "
                f"result.details[config_id]['embeddings'][run_id]['{key}'] instead."
            )

        runs = self.runs
        if runs is not None and key in runs.columns:
            if len(runs) == 1:
                return _scalar(runs[key].iloc[0])
            raise ValueError(
                f"'{key}' has {len(runs)} values, one per repeat. Read "
                f"result.runs['{key}'] instead."
            )
        return default

    def _one_config(self, config_id: Optional[int], what: str) -> int:
        ids = self.config_ids or [0]
        if config_id is not None:
            if config_id not in ids:
                raise ValueError(f"config_id={config_id} is not in this result. It holds {ids}.")
            return config_id
        if len(ids) == 1:
            return ids[0]
        raise ValueError(
            f"{what} needs one configuration and this result holds {len(ids)}. "
            f"Pass config_id=... (one of {ids})."
        )

    def _repeat_view(self, config_id: Optional[int] = None, run_id: Any = None,
                     what: str = "This view", **axis: Any) -> Dict[str, Any]:
        """Everything recorded about one network, flattened into one dict.

        Merges that network's row of ``runs``, its configuration's ``details``
        and its embeddings. ``run_id`` and, in a mode whose axis runs over
        separate networks, the axis values pick the network. A call that
        matches more than one network is refused with the keys to pass.
        """
        cid = self._one_config(config_id, what)
        network_axes = _NETWORK_AXES.get(self.mode, ())
        unknown = sorted(set(axis) - set(network_axes))
        if unknown:
            raise ValueError(
                f"{', '.join(unknown)} does not pick a network of mode='{self.mode}'. Its "
                f"networks are picked by {', '.join(network_axes + ('run_id',))}."
            )
        view: Dict[str, Any] = {}
        entry = self.details.get(cid, {})
        view.update({k: v for k, v in entry.items() if k not in ('embeddings', 'trainings')})
        runs = self.runs
        key = run_id
        if runs is not None and not runs.empty:
            rows = runs[runs['config_id'] == cid] if 'config_id' in runs.columns else runs
            picked = {k: v for k, v in dict(axis, run_id=run_id).items() if v is not None}
            for k, v in picked.items():
                if k in rows.columns:
                    rows = rows[rows[k] == v]
            if rows.empty:
                raise ValueError(
                    f"No network of configuration {cid} has "
                    f"{', '.join(f'{k}={v!r}' for k, v in picked.items())}."
                )
            names = [k for k in network_axes + ('run_id',) if k in rows.columns]
            if names and len(rows[names].drop_duplicates()) > 1:
                open_keys = [k for k in names if rows[k].nunique() > 1]
                values = '; '.join(f"{k} in {sorted(_scalar(v) for v in rows[k].unique())}"
                                   for k in open_keys)
                raise ValueError(
                    f"{what} shows one network and {len(rows[names].drop_duplicates())} of "
                    f"configuration {cid} match. Pass "
                    f"{', '.join(f'{k}=...' for k in open_keys)} ({values})."
                )
            row = rows.iloc[0]
            view.update({k: _scalar(row[k]) for k in rows.columns})
            rid = _scalar(row['run_id']) if 'run_id' in rows.columns else run_id
            # A mode with an axis keeps embeddings per axis value and repeat.
            axis_keys = [k for k in (self.params.get('axis_keys') or []) if k in rows.columns]
            key = (*(_scalar(row[k]) for k in axis_keys), rid) if axis_keys else rid
        embeddings = entry.get('embeddings') or {}
        chosen = embeddings.get(key)
        if chosen is None and len(embeddings) == 1:
            chosen = next(iter(embeddings.values()))
        if chosen:
            view.update(chosen)
        return view

    # ------------------------------------------------------------------
    # Printing
    # ------------------------------------------------------------------

    def summary(self) -> None:
        """Print a readable summary of the result to stdout."""
        sep = "─" * 56
        units = self.params.get('output_units', 'bits')
        df = self.dataframe
        print(sep)
        print(f"  NeuralMI Results  |  mode = '{self.mode}'")
        print(sep)
        n_cfg = len(self.config_ids)
        n_rows = 0 if df is None else len(df)
        if n_rows == 1:
            row = df.iloc[0]
            n_runs = int(row.get('n_runs', 1))
            std = row.get('mi_std')
            spread = (f" ± {std:.4f} (spread over {n_runs} repeats)"
                      if std is not None and not pd.isna(std) else "")
            label = _QUANTITY_LABELS.get(self.mode, 'MI estimate')
            print(f"  {label:<12}: {self.mi_estimate:.4f} {units}{spread}"
                  if self.mi_estimate is not None else f"  {label:<12}: (none)")
            for col, label in _COMPONENTS.get(self.mode, []):
                if f'{col}_mean' in df.columns:
                    print(f"  {label:<12}: {row[f'{col}_mean']:.4f} {units}")
            if 'amplification_factor' in df.columns and not pd.isna(row['amplification_factor']):
                print(f"  Amplification factor : {row['amplification_factor']:.1f}x")
            if self.mode == 'transfer' and 'te_yx_mean' in df.columns:
                print(f"  TE(Y→X)     : {row['te_yx_mean']:.4f} {units}")
                if 'directionality_index_mean' in df.columns:
                    print(f"  Directionality index : {row['directionality_index_mean']:.4f}"
                          f"  (+1 = X→Y, -1 = Y→X, 0 = symmetric)")
            if self._is_rigorous() and self.runs is not None and len(self.runs) == 1:
                fit = {k: _scalar(v) for k, v in self.runs.iloc[0].items()}
                for line in _fit_lines(fit, units):
                    print(line)
            elif 'n_reliable' in df.columns:
                print(f"  Reliable fits : {int(row['n_reliable'])} of {n_runs}")
        elif self.mode == 'dimensionality':
            for cid in self.config_ids:
                entry = self.details.get(cid, {})
                spread = entry.get('dimension_at_most_std')
                spread = f" ± {spread:.1f} over splits" if spread is not None else ""
                print(f"  At most {entry.get('dimension_at_most')} embedding dimensions carry "
                      f"{entry.get('saturation_ratio', 0.95):.0%} of the MI{spread}"
                      + (f"  (configuration {cid})" if n_cfg > 1 else ""))
                print(f"  Participation ratio at embedding_dim={entry.get('reference_dim')}: "
                      f"{entry.get('pr_singular', float('nan')):.1f}")
            print(f"  The curve over embedding dimensions is in result.dataframe")
        else:
            print(f"  {n_rows} rows over {n_cfg} configuration(s); see result.dataframe")
        if self.mode == 'precision':
            for key, label in (('baseline_mi', 'Baseline MI'), ('precision_tau', 'Precision τ'),
                               ('threshold_value', 'Threshold MI')):
                try:
                    value = self.get(key)
                except ValueError:
                    value = None
                if value is not None:
                    print(f"  {label:<12}: {value:.4g}")
        if self.mode == 'pairwise' and n_cfg == 1:
            matrix = self.get('mi_matrix')
            if matrix is not None:
                finite = matrix[np.isfinite(matrix)]
                if len(finite):
                    print(f"  MI matrix   : {matrix.shape[0]} × {matrix.shape[1]}, range "
                          f"{finite.min():.4f} to {finite.max():.4f} {units}")
        if df is not None:
            print(f"  dataframe   : {df.shape[0]} rows × {df.shape[1]} cols")
        if self.runs is not None:
            print(f"  runs        : {len(self.runs)} repeats")
        print(sep)

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def _grouped_keys(self) -> List[str]:
        keys = list(self.params.get('config_keys') or [])
        if self.mode == 'lag':
            keys = ['lag'] + keys
        return [k for k in keys if self.dataframe is not None and k in self.dataframe.columns]

    def plot(self, ax: Optional[plt.Axes] = None, config_id: Optional[int] = None, **kwargs) -> plt.Axes:
        """Draw the result.

        A result over several configurations is drawn as MI against the swept
        keys: a line for one key, a heatmap for two, bars for more (override
        with ``kind='line'|'heatmap'|'bar'``). A single configuration gets its
        mode's own figure: the training curve for estimate, the extrapolation
        for rigorous, the components for the difference quantities, the MI
        against tau for precision, the matrix for pairwise and the curve over
        embedding dimensions for dimensionality. ``config_id`` picks one configuration out of
        several for those per-configuration figures.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw into; a new figure is created when omitted.
        config_id : int, optional
            The configuration to draw with its mode's own figure.
        **kwargs
            ``show``, ``units``, ``title``, ``figsize``, ``kind``, and anything
            the underlying plot function accepts.
        """
        from neural_mi.visualize.plot import (
            plot_sweep_curve, plot_sweep_heatmap, plot_sweep_bar,
            plot_bias_correction_fit, plot_dimensionality_curve,
        )

        show = kwargs.pop('show', True)
        units = kwargs.pop('units', self.params.get('output_units', 'bits'))
        n_cfg = len(self.config_ids)
        grouped = self._grouped_keys()
        per_config_modes = ('pairwise', 'dimensionality', 'precision')
        draw_grouped = (config_id is None and grouped
                        and (self.mode in ('sweep', 'lag') or n_cfg > 1)
                        and self.mode not in per_config_modes)

        own_figure = self.mode in ('dimensionality', 'pairwise') and not draw_grouped
        if ax is None and not own_figure:
            _, ax = plt.subplots(1, 1, figsize=kwargs.pop('figsize', (10, 6)))

        if draw_grouped:
            kind = kwargs.pop('kind', None)
            if kind is None:
                kind = 'line' if len(grouped) <= 1 else ('heatmap' if len(grouped) == 2 else 'bar')
            if kind == 'line':
                plot_sweep_curve(self.dataframe, param_col=grouped[0], units=units,
                                 ax=ax, show=show, **kwargs)
            elif kind == 'heatmap':
                if len(grouped) != 2:
                    raise ValueError(
                        f"kind='heatmap' needs exactly 2 swept keys, found {len(grouped)}: "
                        f"{grouped}. Use kind='bar' for 3 or more."
                    )
                plot_sweep_heatmap(self.dataframe, param_x=grouped[0], param_y=grouped[1],
                                   units=units, ax=ax, show=show, **kwargs)
            elif kind == 'bar':
                plot_sweep_bar(self.dataframe, param_cols=grouped, units=units,
                               ax=ax, show=show, **kwargs)
            else:
                raise ValueError(f"Unknown kind='{kind}'. Expected 'line', 'heatmap' or 'bar'.")
            return ax

        cid = self._one_config(config_id, f"Results.plot() for mode='{self.mode}'")
        row = self.dataframe[self.dataframe['config_id'] == cid] if self.dataframe is not None else None
        runs = self.runs[self.runs['config_id'] == cid] if self.runs is not None else None
        entry = self.details.get(cid, {})
        colours = [p['color'] for p in plt.rcParams['axes.prop_cycle']]

        if self.mode in ('estimate', 'sweep'):
            if runs is None or runs.empty or 'test_mi_history' not in runs.columns:
                raise ValueError(
                    "This result holds no training history to draw. Each repeat records "
                    "'test_mi_history' in result.runs during training."
                )
            index_col = 'run_id' if 'run_id' in runs.columns else None
            for i, (_, rep) in enumerate(runs.iterrows()):
                history = list(rep['test_mi_history'])
                colour = colours[i % len(colours)]
                label = 'Test MI' if len(runs) == 1 else f"run {rep[index_col]}"
                ax.plot(range(len(history)), history, color=colour, linewidth=1.5, label=label)
                if len(runs) == 1:
                    train_history = rep.get('train_mi_history')
                    if isinstance(train_history, (list, np.ndarray)) and len(train_history):
                        ax.plot(range(len(train_history)), list(train_history), color='darkorange',
                                linewidth=1.5, linestyle='--', alpha=0.8, label='Train MI')
                best = rep.get('best_epoch')
                if best is not None and not pd.isna(best) and 0 <= int(best) < len(history):
                    best = int(best)
                    ax.axvline(best, color=colour if len(runs) > 1 else 'tomato',
                               linestyle='--', linewidth=1.2,
                               label=f'Best epoch ({best})' if len(runs) == 1 else None)
                conservative = rep.get('conservative_epoch')
                if (len(runs) == 1 and conservative is not None and not pd.isna(conservative)
                        and 0 <= int(conservative) < len(history)):
                    conservative = int(conservative)
                    ax.axvline(conservative, color='mediumseagreen', linestyle=':', linewidth=1.5,
                               label=f'Conservative epoch ({conservative}), used for estimate')
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel(f'MI ({units})', fontsize=12)
            ax.set_title(kwargs.pop('title', 'Training curve'), fontsize=13)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        elif self.mode == 'lag':
            plot_sweep_curve(self.dataframe[self.dataframe['config_id'] == cid], param_col='lag',
                             units=units, ax=ax, show=show, **kwargs)
            return ax

        elif self._is_rigorous():
            trainings = entry.get('trainings')
            if trainings is None or runs is None or runs.empty:
                raise ValueError("This rigorous result holds no extrapolation to draw.")
            lines = []
            for i, (_, rep) in enumerate(runs.iterrows()):
                rid = rep.get('run_id', 0)
                ladder = trainings[trainings['run_id'] == rid] if 'run_id' in trainings.columns else trainings
                if 'component' in ladder.columns:
                    ladder = ladder[ladder['component'] == 'combined']
                fit = {k: _scalar(rep[k]) for k in rep.index}
                fit['mi_corrected'] = fit['mi_raw'] if _finite(fit.get('mi_raw')) else fit.get('mi')
                label = None if len(runs) == 1 else f"run {rid}"
                plot_bias_correction_fit(ladder, fit, units=units, ax=ax, show=False,
                                         label=label,
                                         color=None if len(runs) == 1 else colours[i % len(colours)],
                                         **kwargs)
                reliable = fit.get('is_reliable')
                prefix = '' if len(runs) == 1 else f"run {rid}: "
                if reliable is False:
                    lines.append(f"⚠ {prefix}extrapolation unreliable{_rigorous_unreliable_reason(fit)}")
                elif reliable is True:
                    lines.append(f"✓ {prefix}extrapolation reliable")
            if lines:
                ax.text(0.02, 0.98, '\n'.join(lines), transform=ax.transAxes, va='top',
                        ha='left', fontsize=9,
                        bbox=dict(facecolor='white', edgecolor='gray', alpha=0.85,
                                  boxstyle='round,pad=0.3'))
            if len(runs) > 1:
                ax.legend(fontsize=9)

        elif self.mode in ('conditional', 'interaction', 'transfer'):
            r = row.iloc[0]
            if self.mode == 'transfer':
                labels = ['TE(X→Y)']
                values = [r['mi_mean']]
                if 'te_yx_mean' in row.columns:
                    labels.append('TE(Y→X)')
                    values.append(r['te_yx_mean'])
                title = 'Transfer entropy'
                if 'directionality_index_mean' in row.columns:
                    di = r['directionality_index_mean']
                    verdict = ('X → Y dominates' if di > 0.1 else
                               'Y → X dominates' if di < -0.1 else '≈ symmetric')
                    title += f'\nDirectionality index = {di:.3f}  ({verdict})'
            else:
                labels = [label for _, label in _COMPONENTS[self.mode]]
                values = [r[f'{col}_mean'] for col, _ in _COMPONENTS[self.mode]]
                labels.append('I(X;Y|W)' if self.mode == 'conditional' else 'II')
                values.append(r['mi_mean'])
                title = ('Conditional MI components' if self.mode == 'conditional'
                         else 'Interaction information components')
            bars = ax.bar(labels, values, color=colours[:len(values)], width=0.5, edgecolor='white')
            ax.axhline(0, color='black', linewidth=0.8)
            ax.set_ylabel(f'MI ({units})', fontsize=12)
            ax.set_title(kwargs.pop('title', title), fontsize=13)
            ax.grid(True, axis='y', alpha=0.3)
            span = max(abs(v) for v in values) or 1.0
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + span * 0.02 * (1 if val >= 0 else -1),
                        f'{val:.3f}', ha='center', va='bottom' if val >= 0 else 'top', fontsize=9)

        elif self.mode == 'precision':
            df = row.sort_values('tau')
            ax.plot(df['tau'], df['mi_mean'], 'o-', color='steelblue', linewidth=2,
                    markersize=5, label='MI vs corruption')
            if 'mi_std' in df.columns and df['mi_std'].notna().any():
                ax.fill_between(df['tau'], df['mi_mean'] - df['mi_std'].fillna(0),
                                df['mi_mean'] + df['mi_std'].fillna(0),
                                alpha=0.2, color='steelblue')
            threshold = entry.get('threshold_value')
            tau_star = entry.get('precision_tau')
            baseline = entry.get('baseline_mi')
            if threshold is not None:
                ax.axhline(threshold, color='tomato', linestyle='--', linewidth=1.5,
                           label=f'Threshold ({threshold:.3f} {units})')
            if tau_star is not None:
                ax.axvline(tau_star, color='darkorange', linestyle='--', linewidth=1.5,
                           label=f'Precision τ = {tau_star:.4g}')
            if baseline is not None:
                ax.annotate(f'Baseline MI = {baseline:.3f} {units}',
                            xy=(df['tau'].iloc[0], baseline), xytext=(0.05, 0.92),
                            textcoords='axes fraction', fontsize=9, color='gray',
                            arrowprops=dict(arrowstyle='->', color='gray', lw=1))
            ax.set_xlabel('Corruption level (τ)', fontsize=12)
            ax.set_ylabel(f'MI ({units})', fontsize=12)
            ax.set_title(kwargs.pop('title', 'MI against timing corruption'), fontsize=13)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        elif self.mode == 'pairwise':
            import seaborn as sns
            matrix = entry.get('mi_matrix')
            if matrix is None:
                raise ValueError("This pairwise result holds no 'mi_matrix' to draw.")
            title = kwargs.pop('title', 'Pairwise MI matrix')
            fmt = kwargs.pop('fmt', '.3f')
            cmap = kwargs.pop('cmap', 'viridis')
            figsize = kwargs.pop('figsize', None)
            n_rows, n_cols = matrix.shape
            if ax is None:
                if figsize is None:
                    figsize = (max(4, n_cols * 0.65 + 1.2), max(3, n_rows * 0.65 + 1.0))
                _, ax = plt.subplots(1, 1, figsize=figsize)
            symmetric = n_rows == n_cols and np.allclose(matrix, matrix.T, equal_nan=True)
            mask = np.zeros_like(matrix, dtype=bool)
            if symmetric:
                np.fill_diagonal(mask, True)
            names_x = entry.get('variable_names_x') or [str(i) for i in range(n_cols)]
            names_y = entry.get('variable_names_y') or [str(i) for i in range(n_rows)]
            sns.heatmap(matrix, mask=mask, annot=True, fmt=fmt, cmap=cmap,
                        xticklabels=names_x, yticklabels=names_y,
                        cbar_kws={'label': f'MI ({units})'}, ax=ax, **kwargs)
            ax.set_title(title, fontsize=13)
            ax.set_xlabel('Channel Y', fontsize=11)
            ax.set_ylabel('Channel X', fontsize=11)

        elif self.mode == 'dimensionality':
            ax = plot_dimensionality_curve(runs, entry, ax=ax, show=show, units=units, **kwargs)
            return ax

        else:
            raise NotImplementedError(f"Plotting is not implemented for mode='{self.mode}'.")

        if show:
            plt.tight_layout()
            plt.show()
        return ax

    @staticmethod
    def compare(results_list: List['Results'], labels: Optional[List[str]] = None,
                ax: Optional[plt.Axes] = None, **kwargs) -> plt.Axes:
        """Overlay several results of the same mode on one axis.

        Estimate results overlay their training curves; results over one swept
        key (sweep, lag, or any mode over a one-key grid) overlay their curves;
        rigorous results overlay their extrapolations. Every result must hold
        one configuration for the curve and extrapolation overlays.

        Parameters
        ----------
        results_list : list of Results
            Two or more results of the same mode.
        labels : list of str, optional
            Legend labels, ``'Result 0'``, ``'Result 1'``, ... by default.
        ax : matplotlib.axes.Axes, optional
            Axes to draw into.
        **kwargs
            ``show``, ``figsize``, ``units`` and plot keyword arguments.
        """
        if not results_list:
            raise ValueError("results_list is empty.")
        if len(results_list) < 2:
            raise ValueError("results_list must contain at least two Results objects to compare.")
        modes = [r.mode for r in results_list]
        if len(set(modes)) > 1:
            raise ValueError(f"All Results objects must share the same mode. Found: {modes}.")
        mode = modes[0]
        if labels is None:
            labels = [f"Result {i}" for i in range(len(results_list))]
        if len(labels) != len(results_list):
            raise ValueError(
                f"labels length ({len(labels)}) must match results_list length ({len(results_list)})."
            )

        from neural_mi.visualize.plot import plot_sweep_curve, plot_bias_correction_fit

        show = kwargs.pop('show', True)
        figsize = kwargs.pop('figsize', (10, 6))
        units = kwargs.pop('units', results_list[0].params.get('output_units', 'bits'))
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=figsize)
        colours = [p['color'] for p in plt.rcParams['axes.prop_cycle']]

        if mode in ('estimate', 'rigorous'):
            for i, (res, label) in enumerate(zip(results_list, labels)):
                n_repeats = 0 if res.runs is None else len(res.runs)
                if n_repeats > 1:
                    raise ValueError(
                        f"Result '{label}' (index {i}) holds {n_repeats} repeats. compare() "
                        f"overlays one {'training curve' if mode == 'estimate' else 'extrapolation'} "
                        f"per result. Call result.plot() to draw every repeat of one result."
                    )

        if mode == 'estimate':
            for i, (res, label) in enumerate(zip(results_list, labels)):
                view = res._repeat_view()
                history = view.get('test_mi_history')
                if history is None:
                    raise ValueError(f"Result '{label}' (index {i}) holds no 'test_mi_history'.")
                history = list(history)
                ax.plot(range(len(history)), history, color=colours[i % len(colours)],
                        linewidth=1.5, label=label)
                best = view.get('best_epoch')
                if best is not None and not pd.isna(best) and 0 <= int(best) < len(history):
                    ax.axvline(int(best), color=colours[i % len(colours)], linestyle='--',
                               linewidth=1, alpha=0.6)
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel(f'Test MI ({units})', fontsize=12)
            ax.set_title('Training curves', fontsize=13)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        elif mode == 'rigorous':
            lines = []
            for i, (res, label) in enumerate(zip(results_list, labels)):
                cid = res._one_config(None, f"Result '{label}'")
                runs = res.runs[res.runs['config_id'] == cid]
                trainings = res.details.get(cid, {}).get('trainings')
                if trainings is None or runs.empty:
                    raise ValueError(f"Rigorous result '{label}' (index {i}) holds no extrapolation.")
                rep = runs.iloc[0]
                fit = {k: _scalar(rep[k]) for k in rep.index}
                fit['mi_corrected'] = fit['mi_raw'] if _finite(fit.get('mi_raw')) else fit.get('mi')
                rid = fit.get('run_id', 0)
                ladder = trainings[trainings['run_id'] == rid] if 'run_id' in trainings.columns else trainings
                plot_bias_correction_fit(ladder, fit, units=units, ax=ax, label=label,
                                         color=colours[i % len(colours)], show=False, **kwargs)
                if fit.get('is_reliable') is False:
                    lines.append(f"⚠ {label}: unreliable{_rigorous_unreliable_reason(fit)}")
                elif fit.get('is_reliable') is True:
                    lines.append(f"✓ {label}: reliable")
            ax.legend(fontsize=9)
            if lines:
                ax.text(0.02, 0.98, '\n'.join(lines), transform=ax.transAxes, va='top', ha='left',
                        fontsize=8, bbox=dict(facecolor='white', edgecolor='gray', alpha=0.85,
                                              boxstyle='round,pad=0.3'))

        else:
            for i, (res, label) in enumerate(zip(results_list, labels)):
                keys = res._grouped_keys()
                if len(keys) != 1:
                    raise ValueError(
                        f"Result '{label}' (index {i}) is indexed by {keys or 'no swept key'}. "
                        f"compare() overlays results over exactly one swept key. Call "
                        f"result.plot() on each result."
                    )
                plot_sweep_curve(res.dataframe, param_col=keys[0], units=units, ax=ax,
                                 label=label, color=colours[i % len(colours)], **kwargs)
            ax.legend(fontsize=9)

        if show:
            plt.tight_layout()
            plt.show()
        return ax

    def animate(self, config_id: Optional[int] = None, run_id: Any = None, **kwargs):
        """Animate one network's training history as a GIF or MP4.

        A thin wrapper around :func:`neural_mi.visualize.animate_training`.
        ``config_id``, ``run_id`` and the axis values of the mode (``lag``;
        ``ch_x`` and ``ch_y``; ``split_id`` and ``embedding_dim``) pick the
        network when the result holds more than one. Every other keyword
        argument is forwarded unchanged.
        """
        from neural_mi.visualize.animate import animate_training
        return animate_training(self, config_id=config_id, run_id=run_id, **kwargs)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @staticmethod
    def _free_path(path: Optional[str], mode: str, ext: str) -> str:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"neuralmi_{mode}_{timestamp}{ext}"
        if path is None:
            filepath = os.path.join(os.getcwd(), base_name)
        elif os.path.isdir(path):
            filepath = os.path.join(path, base_name)
        else:
            filepath = path
        if os.path.exists(filepath):
            root, suffix = os.path.splitext(filepath)
            counter = 1
            while os.path.exists(f"{root}_{counter}{suffix}"):
                counter += 1
            filepath = f"{root}_{counter}{suffix}"
        return filepath

    def save(self, path: Optional[str] = None) -> str:
        """Pickle this result and return the absolute path written.

        With no `path`, or a directory, the file is named
        ``neuralmi_{mode}_{YYYYMMDD_HHMMSS}.pkl``. A numeric suffix is appended
        to avoid overwriting an existing file.
        """
        filepath = self._free_path(path, self.mode, '.pkl')
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        logger.info(f"Results saved to {filepath}")
        return os.path.abspath(filepath)

    @classmethod
    def load(cls, path: str) -> 'Results':
        """Load a result written by :meth:`save`."""
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected a Results object in '{path}', got {type(obj).__name__}.")
        return obj

    def to_dict(self) -> dict:
        """A JSON-ready dict: arrays as nested lists, tables as lists of records."""
        def cvt(obj):
            if obj is None or isinstance(obj, (bool, int, float, str)):
                return obj
            if isinstance(obj, np.generic):
                return obj.item()
            if isinstance(obj, dict):
                return {str(k): cvt(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [cvt(v) for v in obj]
            if isinstance(obj, pd.DataFrame):
                return [{str(k): cvt(v) for k, v in rec.items()} for rec in obj.to_dict(orient='records')]
            if hasattr(obj, 'tolist'):
                return obj.tolist()
            return f"<{type(obj).__name__}>"

        return {
            'mode': self.mode,
            'mi_estimate': self.mi_estimate,
            'params': cvt(self.params or {}),
            'dataframe': cvt(self.dataframe) if self.dataframe is not None else None,
            'runs': cvt(self.runs) if self.runs is not None else None,
            'details': cvt(self.details or {}),
        }

    def to_json(self, path: Optional[str] = None) -> str:
        """Write :meth:`to_dict` as JSON and return the absolute path written.

        Arrays are written in full as nested lists. For an exact round trip use
        :meth:`save` and :meth:`load`.
        """
        payload = self.to_dict()
        filepath = self._free_path(path, self.mode, '.json')
        with open(filepath, 'w') as f:
            json.dump(payload, f, indent=2)
        logger.info(f"Results exported to {filepath}")
        return os.path.abspath(filepath)
