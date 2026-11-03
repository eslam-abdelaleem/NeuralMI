"""A custom critic is a recipe the library builds in each task.

``Model(custom_critic=...)`` takes a class or function, which every task calls
under its own seed with the sizes and settings it names, as
``custom_embedding_cls`` works. Every repeat and every term of a difference
therefore trains a critic of its own. An object is refused.
"""
import warnings

import numpy as np
import pytest
import torch
import torch.nn as nn

import neural_mi as nmi

_TRAINING = nmi.Training(n_epochs=2, batch_size=32)


class Recipe(nmi.models.BaseCritic):
    """A dot-product critic sized by what the library passes, recording every build. In the joint term of a
    difference X is the pair (rest of X, conditioning set), and its size is a pair too."""
    built = []

    def __init__(self, input_dim_x, input_dim_y, embedding_dim):
        super().__init__()
        Recipe.built.append((input_dim_x, input_dim_y, embedding_dim))
        self.fx = nn.Linear(sum(input_dim_x) if isinstance(input_dim_x, tuple) else input_dim_x, embedding_dim)
        self.fy = nn.Linear(input_dim_y, embedding_dim)

    def forward(self, x, y):
        parts = x if isinstance(x, tuple) else (x,)
        x = torch.cat([part.reshape(part.shape[0], -1) for part in parts], dim=1)
        zx, zy = self.fx(x), self.fy(y.reshape(y.shape[0], -1))
        return zx @ zy.T, torch.tensor(0.0)


def _data(n=300, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, 2)).astype(np.float32)
    return x, (x + rng.standard_normal((n, 2))).astype(np.float32)


def _run(*args, **kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = nmi.run(*args, training=_TRAINING, show_progress=False, n_workers=1, **kwargs)
    return result, [str(w.message) for w in caught]


@pytest.mark.parametrize('value', [Recipe(2, 2, 4), object()], ids=['module', 'not_callable'])
def test_an_object_is_refused_with_the_new_form(value):
    x, y = _data()
    with pytest.raises(TypeError, match=r"custom_critic=MyCritic"):
        nmi.run(x, y, model=nmi.Model(custom_critic=value), training=_TRAINING, show_progress=False)


def test_every_repeat_builds_its_own_critic_with_the_sizes_it_names():
    Recipe.built.clear()
    x, y = _data()
    _run(x, y, mode='sweep', model=nmi.Model(custom_critic=Recipe, embedding_dim=6),
         sweep_grid={'run_id': range(3)}, seed=0, device='cpu')     # choosing a device would build one more
    assert Recipe.built == [(2, 2, 6)] * 3


def test_every_term_of_a_difference_builds_its_own():
    Recipe.built.clear()
    x, y = _data()
    w = x[:, :1].copy()
    result, _ = _run(x, y, mode='conditional', conditional=nmi.Conditional(w_data=w),
                     model=nmi.Model(custom_critic=Recipe, embedding_dim=4), seed=0, device='cpu')
    assert Recipe.built == [(1, 2, 4), ((2, 1), 2, 4)]        # the conditioning term, then the pair (X, W)
    assert result.runs is not None and len(result.runs) == 1


def test_a_recipe_that_cannot_take_the_pair_is_told_why():
    class OneX(nmi.models.BaseCritic):
        def __init__(self, input_dim_x, input_dim_y, embedding_dim):
            super().__init__()
            self.fx, self.fy = nn.Linear(input_dim_x, embedding_dim), nn.Linear(input_dim_y, embedding_dim)

        def forward(self, x, y):
            return self.fx(x.reshape(x.shape[0], -1)) @ self.fy(y.reshape(y.shape[0], -1)).T, torch.tensor(0.0)

    x, y = _data()
    with pytest.raises(TypeError) as caught:
        _run(x, y, mode='conditional', conditional=nmi.Conditional(w_data=x[:, :1].copy()),
             model=nmi.Model(custom_critic=OneX), seed=0, device='cpu')
    assert any('the pair (rest of X, conditioning set)' in note for note in caught.value.__notes__)


def test_the_same_seed_gives_the_same_result():
    x, y = _data()
    first, _ = _run(x, y, model=nmi.Model(custom_critic=Recipe), seed=3)
    second, _ = _run(x, y, model=nmi.Model(custom_critic=Recipe), seed=3)
    assert first.mi_estimate == second.mi_estimate


def test_a_setting_the_recipe_does_not_take_says_so():
    x, y = _data()
    _, messages = _run(x, y, model=nmi.Model(custom_critic=Recipe, embedding_dim=8, hidden_dim=16), seed=0)
    said = [m for m in messages if 'no effect' in m]
    assert said and 'hidden_dim' in said[0] and 'embedding_dim' not in said[0].split('ignored:')[1]


def test_a_recipe_that_needs_more_than_the_library_passes_is_refused():
    def needs_more(input_dim_x, something_else):
        return Recipe(input_dim_x, 2, 4)

    x, y = _data()
    with pytest.raises(Exception, match='something_else'):
        nmi.run(x, y, model=nmi.Model(custom_critic=needs_more), training=_TRAINING, show_progress=False,
                n_workers=1)
