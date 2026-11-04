"""Defaults come from the schema alone, and a saved network carries every setting its build reads.

Every default below ``run()`` is the schema's (``defaults.default_of``), and
``build_critic`` reads only ``utils.ARCHITECTURE_KEYS``, the list
``save_network`` saves, so ``extract_embeddings`` rebuilds a network as it was
trained.
"""
import ast
import inspect
import pathlib

import numpy as np
import pytest
import torch

import neural_mi
from neural_mi.defaults import BASE_PARAMS_SCHEMA, default_of
from neural_mi.embeddings_io import extract_embeddings, save_network
from neural_mi.models import critics, decoders, embeddings
from neural_mi.training.trainer import Trainer
from neural_mi.utils import ARCHITECTURE_KEYS, build_critic

# Signature parameters that share a setting's name but hold something else. A critic left without an
# embedding_dim reads it from its encoders.
_OTHER_MEANING = {('Trainer.train', 'scheduler'), ('Trainer.__init__', 'estimator_params'),
                  ('SeparableCritic', 'embedding_dim'), ('HybridCritic', 'embedding_dim'),
                  ('ConcatCritic', 'embedding_dim')}

PACKAGE = pathlib.Path(neural_mi.__file__).resolve().parent
_SETTINGS = {name for name, schema in BASE_PARAMS_SCHEMA.items() if schema.get('default') is not None}


def _signatures():
    out = [('Trainer.train', Trainer.train), ('Trainer.__init__', Trainer.__init__)]
    for module in (embeddings, decoders, critics):
        out += [(name, cls.__init__) for name, cls in inspect.getmembers(module, inspect.isclass)
                if cls.__module__ == module.__name__]
    return out


def _public_interface(where, name):
    """The signatures that keep literal defaults: run() and the named quantities, whose signatures are the
    interface until ROADMAP design item 1 declares each setting once, _run_flat, which that item replaces,
    and a generator's own network width, which only shares a setting's name."""
    return ((where == 'run.py' and name in ('run', '_run_flat'))
            or (where == 'quantities.py' and not name.startswith('_'))
            or (where == 'generators/oracle.py' and name == 'generate_nonlinear_from_latent'))


def _literal(node):
    """A literal other than None, which means unset."""
    if isinstance(node, ast.Constant):
        return node.value is not None
    if isinstance(node, ast.UnaryOp) and isinstance(node.operand, ast.Constant):
        return True
    return isinstance(node, (ast.Tuple, ast.List)) and all(_literal(e) for e in node.elts)


def _literal_copies():
    """Every place in the package that writes a setting's default as a literal: a ``.get(name, literal)``
    fallback, or a parameter default outside the public interface."""
    found = []
    for path in sorted(PACKAGE.rglob('*.py')):
        where = path.relative_to(PACKAGE).as_posix()
        for node in ast.walk(ast.parse(path.read_text())):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'get'
                    and len(node.args) == 2 and isinstance(node.args[0], ast.Constant)
                    and node.args[0].value in _SETTINGS and _literal(node.args[1])):
                found.append(f"{where}:{node.lineno} .get({node.args[0].value!r}, {ast.unparse(node.args[1])})")
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not _public_interface(where, node.name):
                a = node.args
                positional = a.posonlyargs + a.args
                pairs = list(zip(positional[len(positional) - len(a.defaults):], a.defaults))
                pairs += [(arg, d) for arg, d in zip(a.kwonlyargs, a.kw_defaults) if d is not None]
                found += [f"{where}:{node.lineno} {node.name}({arg.arg}={ast.unparse(d)})"
                          for arg, d in pairs if arg.arg in _SETTINGS and _literal(d)]
    return found


def test_no_default_is_written_a_second_time():
    """A setting's default is written once, in the schema. Below the public signatures a function reads it
    with setting() or default_of(). A ``.get(name, literal)`` fallback or a literal parameter default would
    be a second copy, free to drift from the schema's."""
    copies = _literal_copies()
    assert not copies, "defaults written outside the schema:\n" + "\n".join(copies)


@pytest.mark.parametrize('label, fn', _signatures(), ids=[label for label, _ in _signatures()])
def test_every_default_of_a_setting_is_the_schemas(label, fn):
    for name, parameter in inspect.signature(fn).parameters.items():
        schema = BASE_PARAMS_SCHEMA.get(name, {})
        if parameter.default is inspect.Parameter.empty or 'default' not in schema:
            continue
        if (label, name) in _OTHER_MEANING or (name == 'norm_layer' and parameter.default is None):
            continue           # norm_layer arrives resolved: 'auto' is None outside mode='dimensionality'
        expected = default_of('decoder_lambda') if name.startswith('decoder_lambda_') else default_of(name)
        assert parameter.default == expected, f"{label}({name}={parameter.default!r}) is not the schema's {expected!r}"


def _params(**overrides):
    params = {k: v['default'] for k, v in BASE_PARAMS_SCHEMA.items() if 'default' in v}
    params.update(input_dim_x=6, input_dim_y=4, n_channels_x=2, n_channels_y=2, hidden_dim=8,
                  embedding_dim=4, n_layers=1)
    params.update(overrides)
    return params


class TestASavedNetworkRebuildsAsTrained:

    @pytest.mark.parametrize('overrides, x', [
        ({'bias': False, 'critic_type': 'hybrid', 'hidden_dim_head': 5}, torch.randn(7, 2, 3)),
        ({'embedding_model': 'cnn', 'cnn_pooling': 'mean'}, torch.randn(7, 2, 3)),
        ({'_difference_term': True, 'n_channels_x': (1, 1), 'input_dim_x': (3, 5)},
         (torch.randn(7, 1, 3), torch.randn(7, 1, 5))),
        ({'embedding_model_y': 'gru', 'critic_type': 'hybrid'}, torch.randn(7, 2, 3)),
    ], ids=['bias_and_head', 'cnn_pooling', 'difference_term', 'y_side'])
    def test_extract_embeddings_rebuilds_it(self, tmp_path, overrides, x):
        params = _params(save_best_model_path=str(tmp_path / 'net.pt'), **overrides)
        critic = build_critic(params['critic_type'], params).eval()
        path = save_network(critic, params, ARCHITECTURE_KEYS)
        y = torch.randn(7, 2, 2)
        zx, zy = extract_embeddings(path, x, y)
        with torch.no_grad():
            ex, ey = critic.get_embeddings(x, y)
        assert np.allclose(zx, ex.numpy(), atol=1e-6) and np.allclose(zy, ey.numpy(), atol=1e-6)

    def test_nothing_outside_the_list_reaches_the_build(self):
        torch.manual_seed(0)
        full = build_critic('separable', _params(learning_rate=0.5, some_other_key=1))
        torch.manual_seed(0)
        listed = build_critic('separable', {k: v for k, v in _params().items() if k in ARCHITECTURE_KEYS})
        assert all(torch.equal(a, b) for a, b in zip(full.state_dict().values(), listed.state_dict().values()))

    def test_a_class_that_cannot_be_saved_is_named_on_rebuild(self, tmp_path):
        class Local(torch.nn.Module):
            def __init__(self, input_dim, hidden_dim, embedding_dim, n_layers):
                super().__init__()
                self.net = torch.nn.Linear(input_dim, embedding_dim)

            def forward(self, x):
                return self.net(x.reshape(x.shape[0], -1))

        params = _params(save_best_model_path=str(tmp_path / 'net.pt'), custom_embedding_cls=Local)
        path = save_network(build_critic('separable', params), params, ARCHITECTURE_KEYS)
        x, y = torch.randn(5, 2, 3), torch.randn(5, 2, 2)
        with pytest.raises(ValueError, match='custom_embedding_cls'):
            extract_embeddings(path, x, y)
        zx, _ = extract_embeddings(path, x, y, base_params={'custom_embedding_cls': Local})
        assert zx.shape == (5, 4)
