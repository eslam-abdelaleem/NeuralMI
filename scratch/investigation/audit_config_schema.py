"""Settings the schema knows that no config can set, and config fields the schema does not know."""
import dataclasses
import neural_mi.config as cfg
from neural_mi.defaults import BASE_PARAMS_SCHEMA, MODE_KWARGS_SCHEMA

mode_cfg = {'rigorous': cfg.Rigorous, 'precision': cfg.Precision, 'lag': cfg.Lag, 'transfer': cfg.Transfer,
            'dimensionality': cfg.Dimensionality, 'conditional': cfg.Conditional,
            'interaction': cfg.Interaction, 'pairwise': cfg.Pairwise}
print("Mode settings in MODE_KWARGS_SCHEMA with no field on the mode's config:")
for mode, spec in MODE_KWARGS_SCHEMA.items():
    cls = mode_cfg.get(mode)
    if cls is None:
        continue
    fields = {f.name for f in dataclasses.fields(cls)}
    missing = sorted(k for k in spec if k not in fields and k != 'n_workers')
    if missing:
        print(f"  {mode:14s}: {missing}")

print("\nBASE_PARAMS_SCHEMA user settings reachable from no config (Model/Training/Split/Estimator/Output):")
reach = set()
for cls, rename in ((cfg.Model, {}), (cfg.Training, {}), (cfg.Split, {'mode': 'split_mode', 'gap_fraction': 'split_gap_fraction'}),
                    (cfg.Estimator, {'name': 'estimator_name', 'params': 'estimator_params'}),
                    (cfg.Output, {'units': 'output_units'})):
    for f in dataclasses.fields(cls):
        reach.add(rename.get(f.name, f.name))
internal = {'input_dim_x', 'input_dim_y', 'n_channels_x', 'n_channels_y', 'processor_type_x', 'processor_type_y',
            'processor_params_x', 'processor_params_y', 'train_indices', 'test_indices', 'gamma', 'lag',
            'x_time', 'y_time', 'random_seed', 'verbose', 'show_progress', 'device'}
print("  ", sorted(k for k in BASE_PARAMS_SCHEMA if k not in reach and k not in internal))
